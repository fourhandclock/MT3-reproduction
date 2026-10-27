import copy

import numpy as np
import pytest

from acoustic10.decoder import SourceTimeDecoder
from streaming.decoder import OnlineDecoder


def config():
    return {"feature": {"hop_length": 160, "sample_rate": 16000},
            "model": {"max_fret": 24, "pitch_min": 40, "pitch_max": 88},
            "decoder": {"minimum_note_seconds": .04, "minimum_retrigger_seconds": .06},
            "streaming": {"confirm_frames": 4, "release_confirm_frames": 2}}


def parameters(backdate=0):
    return {"onset_threshold": .4, "activity_threshold": .6,
            "onset_backdate_seconds": backdate}


def probabilities(frets, onsets=None):
    count = len(frets)
    state = np.zeros((count, 6, 26), dtype=np.float32)
    state[:, :, -1] = .95
    for frame, fret in enumerate(frets):
        if fret is not None:
            state[frame, 0, -1] = .05
            state[frame, 0, fret] = .95
    onset = np.zeros((count, 6), dtype=np.float32)
    if onsets is not None:
        onset[:, 0] = onsets
    return {"state": state, "onset": onset, "offset": np.zeros_like(onset),
            "pitch": np.zeros((count, 49), dtype=np.float32)}


def sliced(values, start, end):
    return {key: value[start:end] for key, value in values.items()}


def source_notes(notes):
    return [{key: value for key, value in note.items() if "emitted_at" not in key}
            for note in notes]


def verify_pairs(stream):
    for note in stream.notes:
        pair = [event for event in stream.events if event["note_id"] == note["note_id"]]
        assert [event["type"] for event in pair] == ["note_on", "note_off"]
        assert pair[0]["start"] == pair[0]["time"] == pair[0]["source_time"] == note["start"]
        assert pair[1]["end"] == pair[1]["time"] == pair[1]["source_time"] == note["end"]
        assert note["end"] > note["start"] >= 0
        assert all(event["emitted_at_audio_seconds"] >= event["source_time"] for event in pair)


@pytest.mark.parametrize("backdate", [0., .03, .06])
def test_zero_delay_is_exact_parent_identity_for_events_notes_and_state(backdate):
    frets = [5] * 20 + [None] * 2 + [8] * 8 + [4] * 5
    onset = np.zeros(len(frets)); onset[[1, 8, 18, 24, 34]] = 1
    values = probabilities(frets, onset)
    baseline = OnlineDecoder(config(), parameters(backdate))
    translated = SourceTimeDecoder(config(), parameters(backdate), 0.)
    for left, right in [(0, 4), (4, 9), (9, 22), (22, 35)]:
        original_events = baseline.push(sliced(values, left, right), right / 100)
        delayed_events = translated.push(sliced(values, left, right), right / 100)
        assert original_events == delayed_events
    assert baseline.finish(.35) == translated.finish(.35)
    assert baseline.events == translated.events
    assert baseline.notes == translated.notes
    assert baseline._strings == translated._strings
    assert translated.clamp_counts["total"] == 0


def test_global_delay_translates_all_boundaries_once_and_keeps_internal_backdate():
    frets = [None] * 19 + [5] * 18 + [None] * 2
    onset = np.zeros(len(frets)); onset[30] = 1
    values = probabilities(frets, onset)
    baseline = OnlineDecoder(config(), parameters(.06))
    translated = SourceTimeDecoder(config(), parameters(.06), .06)
    baseline.push(values); translated.push(values)
    baseline.finish(.39); translated.finish(.39)
    assert len(translated.notes) == len(baseline.notes) == 2
    # The parent backdates only the initial candidate: .20 - .06 = .14.
    # Publication shifts that .14 to .08; retrigger .31 becomes .25.
    assert [note["start"] for note in translated.notes] == pytest.approx([.08, .25])
    assert [note["end"] for note in translated.notes] == pytest.approx([.25, .32])
    for original, shifted in zip(baseline.events, translated.events):
        assert shifted["source_time"] == pytest.approx(original["source_time"] - .06)
        assert shifted["emitted_at_audio_seconds"] == original["emitted_at_audio_seconds"]
        assert shifted["flush"] == original["flush"]
    assert baseline._strings == translated._strings
    assert translated.onset_backdate_seconds == .06
    assert translated.clamp_counts["total"] == 0
    verify_pairs(translated)


def test_early_clamping_keeps_all_attacks_positive_and_nonoverlapping():
    stream = SourceTimeDecoder(config(), parameters(.06), .2)
    stream.push(probabilities([4] * 4 + [8] * 4 + [None] * 2))
    stream.finish(.1)
    assert len(stream.notes) == 2
    first, second = stream.notes
    assert first["start"] == 0.
    assert first["end"] == second["start"] == 1e-9
    assert second["end"] == 2e-9
    assert stream.clamp_counts == {"onset_zero": 2, "onset_previous_release": 1,
                                   "offset_positive_interval": 2, "total": 5}
    # Small public intervals do not change the parent's raw state or guards.
    assert stream._notes[1]["end"] == pytest.approx(.09)
    assert stream._notes[1]["start"] > stream._notes[0]["start"]
    verify_pairs(stream)


def test_eof_close_translates_source_only_and_remains_censored():
    stream = SourceTimeDecoder(config(), parameters(), .06)
    stream.push(probabilities([None] * 10 + [3] * 8), .2)
    on = copy.deepcopy(stream.events)
    off = stream.finish(.185, .21)
    assert off[0]["source_time"] == pytest.approx(.125)
    assert off[0]["emitted_at_audio_seconds"] == .21
    assert off[0]["flush"] is True
    assert stream.notes[0]["censored_at_end"] is True
    assert stream.events[:len(on)] == on
    assert stream.finish(.185) == []
    verify_pairs(stream)


def test_chunks_and_frame_arrival_have_same_public_intervals():
    frets = [None] * 8 + [3] * 10 + [None] * 2 + [6] * 7 + [None] * 2
    values = probabilities(frets)
    framewise = SourceTimeDecoder(config(), parameters(.03), .06)
    full = SourceTimeDecoder(config(), parameters(.03), .06)
    chunked = SourceTimeDecoder(config(), parameters(.03), .06)
    framewise.push(values)
    full.push(values, .29)
    for left, right in [(0, 3), (3, 9), (9, 17), (17, 20), (20, 29)]:
        chunked.push(sliced(values, left, right), right / 100)
    for stream in [framewise, full, chunked]:
        stream.finish(.29)
        verify_pairs(stream)
    assert source_notes(framewise.notes) == source_notes(full.notes) == source_notes(chunked.notes)
    assert framewise.clamp_counts == full.clamp_counts == chunked.clamp_counts
    assert framewise.events[0]["emitted_at_audio_seconds"] <= chunked.events[0]["emitted_at_audio_seconds"]
    assert full.events[0]["emitted_at_audio_seconds"] == .29


def test_event_prefix_and_returned_note_mapping_cannot_be_changed_later():
    stream = SourceTimeDecoder(config(), parameters(), .06)
    stream.push(probabilities([None] * 10 + [5] * 4), .14)
    published = copy.deepcopy(stream.events)
    returned = stream.events; returned[0]["start"] = 999
    assert stream.events == published
    stream.push(probabilities([5] * 5 + [None] * 2), .21)
    stream.finish(.21)
    assert stream.events[:len(published)] == published
    notes = stream.notes; notes[0]["start"] = 999
    counts = stream.clamp_counts; counts["total"] = 999
    assert stream.notes[0]["start"] != 999
    assert stream.clamp_counts["total"] != 999
    verify_pairs(stream)


def test_same_fret_retrigger_guards_stay_on_raw_internal_clock():
    onset = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0]
    values = probabilities([5] * len(onset), onset)
    baseline = OnlineDecoder(config(), parameters())
    stream = SourceTimeDecoder(config(), parameters(), .06)
    baseline.push(values); stream.push(values)
    baseline.finish(.14); stream.finish(.14)
    assert len(stream.notes) == len(baseline.notes) == 2
    assert [note["start"] for note in stream.notes] == pytest.approx([0, .03])
    assert [note["end"] for note in stream.notes] == pytest.approx([.03, .08])
    assert baseline._strings == stream._strings
    verify_pairs(stream)


def test_mapping_does_not_bypass_parent_atomic_validation():
    stream = SourceTimeDecoder(config(), parameters(), .06)
    values = probabilities([5] * 4)
    with pytest.raises(ValueError, match="availability"):
        stream.push(values, .03)
    assert stream.events == [] and stream.framecount == 0
    assert stream._mapped_notes == {} and stream.clamp_counts["total"] == 0
    bad = dict(values, onset=np.ones((3, 6)))
    with pytest.raises(ValueError, match="same frame"):
        stream.push(bad)
    assert stream.events == [] and stream._mapped_notes == {}


@pytest.mark.parametrize("delay", [-.01, float("nan"), float("inf")])
def test_invalid_source_delay_is_rejected(delay):
    with pytest.raises(ValueError, match="nonnegative and finite"):
        SourceTimeDecoder(config(), parameters(), delay)
