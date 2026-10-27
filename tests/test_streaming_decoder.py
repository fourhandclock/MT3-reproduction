import copy

import numpy as np
import pytest

from streaming.decoder import OnlineDecoder


def config():
    return {'feature': {'hop_length': 160, 'sample_rate': 16000},
            'model': {'max_fret': 24, 'pitch_min': 40, 'pitch_max': 88},
            'decoder': {'minimum_note_seconds': 0.04, 'minimum_retrigger_seconds': 0.06},
            'streaming': {'confirm_frames': 4, 'release_confirm_frames': 2}}


def decoder(backdate=0):
    return OnlineDecoder(config(), {'onset_threshold': 0.4, 'activity_threshold': 0.6,
                                    'onset_backdate_seconds': backdate})


def probabilities(frets, onsets=None):
    """Make explicit activity evidence on string zero; other strings are quiet."""
    count = len(frets)
    state = np.zeros((count, 6, 26), dtype=np.float32)
    state[:, :, -1] = 0.95
    for frame, fret in enumerate(frets):
        if fret is not None:
            state[frame, 0, -1] = 0.05
            state[frame, 0, fret] = 0.95
    onset = np.zeros((count, 6), dtype=np.float32)
    if onsets is not None:
        onset[:, 0] = onsets
    return {'state': state, 'onset': onset, 'offset': np.zeros_like(onset),
            'pitch': np.zeros((count, 49), dtype=np.float32)}


def sliced(values, start, end):
    return {key: value[start:end] for key, value in values.items()}


def source_notes(notes):
    return [{key: value for key, value in note.items() if 'emitted_at' not in key} for note in notes]


def test_confirmation_and_release_are_causal_and_backdate_only_source_time():
    stream = decoder(backdate=0.03)
    assert stream.push(probabilities([5, 5, 5])) == []
    on = stream.push(probabilities([5]), available_at_audio_seconds=0.08)
    assert len(on) == 1 and on[0]['type'] == 'note_on'
    assert on[0]['start'] == 0
    assert on[0]['emitted_at_audio_seconds'] == pytest.approx(0.08)
    assert stream.notes == []
    assert stream.push(probabilities([None]), available_at_audio_seconds=0.08) == []
    off = stream.push(probabilities([None]), available_at_audio_seconds=0.08)
    assert off[0]['end'] == pytest.approx(0.05)
    assert off[0]['emitted_at_audio_seconds'] == pytest.approx(0.08)
    assert not off[0]['flush']
    assert stream.notes[0]['censored_at_end'] is False
    assert stream.framecount == stream.frame_count == 6


def test_unstable_candidates_and_single_inactivity_frame_do_not_split_notes():
    stream = decoder()
    events = stream.push(probabilities([4, 4, 8, 4, 4, 4, 4, None, 4, 4, None, None]))
    assert [event['type'] for event in events] == ['note_on', 'note_off']
    assert stream.notes[0]['start'] == pytest.approx(0.04)
    assert stream.notes[0]['end'] == pytest.approx(0.11)


def test_fret_change_is_confirmed_and_backdating_never_crosses_published_attack():
    stream = decoder(backdate=0.06)
    stream.push(probabilities([4] * 4))
    first = copy.deepcopy(stream.events)
    assert stream.push(probabilities([8] * 3)) == []
    changed = stream.push(probabilities([8]))
    assert [event['type'] for event in changed] == ['note_off', 'note_on']
    assert changed[0]['end'] == changed[1]['start']
    assert changed[1]['start'] > first[0]['start']
    assert stream.events[:len(first)] == first
    stream.finish(0.08)
    assert all(note['end'] > note['start'] for note in stream.notes)
    assert stream.notes[0]['end'] <= stream.notes[1]['start']


def test_retrigger_uses_upward_crossing_and_respects_minimum_spacing():
    stream = decoder()
    # First crossing happens before confirmation. Early second crossing is
    # ignored; the high plateau is not repeatedly treated as a new attack.
    onset = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0]
    stream.push(probabilities([5] * len(onset), onset))
    stream.finish(0.14)
    assert [note['start'] for note in stream.notes] == pytest.approx([0.01, 0.09])
    assert [note['end'] for note in stream.notes] == pytest.approx([0.09, 0.14])
    assert [event['type'] for event in stream.events] == ['note_on', 'note_off', 'note_on', 'note_off']
    assert stream.events[1]['source_time'] == pytest.approx(0.09)
    assert stream.events[1]['emitted_at_audio_seconds'] == pytest.approx(0.1)
    assert stream.events[2]['emitted_at_audio_seconds'] == pytest.approx(0.1)
    assert stream.notes[0]['censored_at_end'] is False
    assert stream.notes[1]['censored_at_end'] is True


def test_final_frame_retrigger_remains_unconfirmed_and_eof_preserves_the_existing_note():
    stream = decoder()
    stream.push(probabilities([5] * 19))
    published = copy.deepcopy(stream.events)
    # This was previously published immediately at .20, making finish(.20)
    # fail after already issuing a zero-length attack to its caller.
    assert stream.push(probabilities([5], [1])) == []
    assert stream.events == published
    flushed = stream.finish(0.2)
    assert len(flushed) == 1 and flushed[0]['type'] == 'note_off'
    assert flushed[0]['flush'] is True
    assert len(stream.notes) == 1
    assert stream.notes[0]['start'] == pytest.approx(0.01)
    assert stream.notes[0]['end'] == pytest.approx(0.2)
    assert stream.events[:len(published)] == published


def test_retrigger_confirmation_across_push_boundary_preserves_trigger_source_time():
    stream = decoder()
    stream.push(probabilities([5] * 18))
    assert stream.push(probabilities([5], [1])) == []
    events = stream.push(probabilities([5], [0]), available_at_audio_seconds=0.25)
    assert [event['type'] for event in events] == ['note_off', 'note_on']
    assert [event['source_time'] for event in events] == pytest.approx([0.19, 0.19])
    assert [event['emitted_at_audio_seconds'] for event in events] == pytest.approx([0.25, 0.25])
    stream.finish(0.2, available_at_audio_seconds=0.25)
    assert [note['start'] for note in stream.notes] == pytest.approx([0.01, 0.19])
    assert all(note['end'] > note['start'] for note in stream.notes)


@pytest.mark.parametrize('interruption', [None, 7])
def test_retrigger_is_cancelled_without_a_subsequent_same_fret_frame(interruption):
    stream = decoder()
    stream.push(probabilities([5] * 9))
    assert stream.push(probabilities([5], [1])) == []
    assert stream.push(probabilities([interruption, 5], [0, 0])) == []
    stream.finish(0.12)
    assert len(stream.notes) == 1
    assert stream.notes[0]['end'] == pytest.approx(0.12)


def test_future_extension_preserves_the_published_event_prefix():
    prefix = probabilities([2] * 4 + [None] * 2 + [7] * 4)
    short = decoder()
    long = decoder()
    short.push(prefix, available_at_audio_seconds=0.1)
    long.push(prefix, available_at_audio_seconds=0.1)
    published = copy.deepcopy(short.events)
    long.push(probabilities([7, 7, None, None, 9, 9, 9, 9]), available_at_audio_seconds=0.18)
    long.finish(0.18)
    assert long.events[:len(published)] == published
    # Returned dictionaries cannot corrupt later publications or stored state.
    returned = short.events
    returned[0]['start'] = 999
    assert short.events == published


def test_chunk_and_frame_arrival_produce_identical_notes_with_different_delivery_latency():
    values = probabilities([3] * 8 + [None] * 2 + [6] * 6 + [None] * 2)
    framewise = decoder(backdate=0.03)
    full = decoder(backdate=0.03)
    chunked = decoder(backdate=0.03)
    framewise.push(values)
    full.push(values, available_at_audio_seconds=0.18)
    for left, right in ((0, 3), (3, 8), (8, 11), (11, 18)):
        chunked.push(sliced(values, left, right), available_at_audio_seconds=right / 100)
    for stream in (framewise, full, chunked):
        stream.finish(0.18)
    assert source_notes(framewise.notes) == source_notes(full.notes) == source_notes(chunked.notes)
    assert framewise.events[0]['emitted_at_audio_seconds'] == pytest.approx(0.04)
    assert chunked.events[0]['emitted_at_audio_seconds'] == pytest.approx(0.08)
    assert full.events[0]['emitted_at_audio_seconds'] == pytest.approx(0.18)
    assert all(event['emitted_at_audio_seconds'] >= event['source_time'] for event in chunked.events)
    # Off/on pairs are exact and IDs remain stable across arrival patterns.
    for stream in (framewise, full, chunked):
        for note in stream.notes:
            pair = [event for event in stream.events if event['note_id'] == note['note_id']]
            assert [event['type'] for event in pair] == ['note_on', 'note_off']
            assert pair[0]['start'] == note['start'] and pair[1]['end'] == note['end']


def test_finish_marks_terminal_release_and_keeps_every_published_note():
    stream = decoder()
    stream.push(probabilities([5] * 4 + [7] * 3))
    flushed = stream.finish(0.07, available_at_audio_seconds=0.1)
    assert len(stream.notes) == 1  # The final candidate was never confirmed.
    assert len(flushed) == 1 and flushed[0]['flush']
    assert flushed[0]['end'] == pytest.approx(0.07)
    assert flushed[0]['emitted_at_audio_seconds'] == pytest.approx(0.1)
    assert stream.notes[0]['censored_at_end']
    assert stream.finish(0.07) == []
    with pytest.raises(RuntimeError):
        stream.push(probabilities([5]))


def test_probability_and_arrival_validation_is_atomic():
    stream = decoder()
    values = probabilities([5] * 4)
    with pytest.raises(ValueError, match='availability'):
        stream.push(values, available_at_audio_seconds=0.03)
    assert stream.framecount == 0 and stream.events == []
    malformed = dict(values, pitch=np.zeros((3, 49)))
    with pytest.raises(ValueError, match='same frame count'):
        stream.push(malformed)
    assert stream.framecount == 0
    stream.push(values, available_at_audio_seconds=0.08)
    with pytest.raises(ValueError, match='prior'):
        stream.push(probabilities([None]), available_at_audio_seconds=0.06)
    with pytest.raises(ValueError, match='prior'):
        stream.push(probabilities([None]))
    with pytest.raises(ValueError, match='Duration'):
        stream.finish(0.03)
    assert stream.framecount == 4 and len(stream.events) == 1


def test_simultaneous_strings_get_independent_exact_event_pairs():
    stream = decoder()
    values = probabilities([5] * 4 + [None] * 2)
    values['state'][:4, 1, -1] = 0.05
    values['state'][:4, 1, 3] = 0.95
    stream.push(values)
    assert {note['pitch'] for note in stream.notes} == {45, 48}
    assert len({note['note_id'] for note in stream.notes}) == 2
    assert all(note['start'] == pytest.approx(0.01) and note['end'] == pytest.approx(0.05)
               for note in stream.notes)
