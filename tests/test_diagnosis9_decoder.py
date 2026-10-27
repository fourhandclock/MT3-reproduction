import copy

import numpy as np
import pytest

from diagnosis9.decoder import SmoothedDecoder
from streaming.decoder import OnlineDecoder


def configuration():
    return {'feature': {'hop_length': 160, 'sample_rate': 16000},
            'model': {'max_fret': 24, 'pitch_min': 40, 'pitch_max': 88},
            'decoder': {'minimum_note_seconds': .04, 'minimum_retrigger_seconds': .06},
            'streaming': {'confirm_frames': 4, 'release_confirm_frames': 2}}


def parameters(alpha=1, confirm=4):
    return {'state_alpha': alpha, 'confirm_frames': confirm,
            'onset_threshold': .55, 'activity_threshold': .35,
            'onset_backdate_seconds': .06}


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
    return {'state': state, 'onset': onset, 'offset': np.zeros_like(onset),
            'pitch': np.zeros((count, 49), dtype=np.float32)}


def sliced(values, start, end):
    return {key: value[start:end] for key, value in values.items()}


def without_delivery(values):
    return [{key: value for key, value in item.items() if 'emitted_at' not in key} for item in values]


@pytest.mark.parametrize('chunked', [False, True])
def test_alpha_one_is_exact_identity_for_notes_events_and_source_times(chunked):
    values = probabilities([5] * 18 + [5] * 2 + [8] * 5 + [None] * 3,
                           [0] * 18 + [1, 0] + [0] * 8)
    original = OnlineDecoder(configuration(), parameters())
    smoothed = SmoothedDecoder(configuration(), parameters())
    chunks = ((0, 4), (4, 19), (19, 23), (23, 28)) if chunked else ((0, 28),)
    for first, last in chunks:
        block = sliced(values, first, last)
        assert smoothed.push(block, last / 100) == original.push(block, last / 100)
    assert smoothed.finish(.28) == original.finish(.28)
    assert smoothed.notes == original.notes
    assert smoothed.events == original.events
    assert smoothed.framecount == smoothed.frame_count == original.framecount
    assert np.array_equal(smoothed.smoothed_state, values['state'][-1])


def test_alpha_one_preserves_float64_input_values_and_retrigger_at_eof():
    values = {key: value.astype(np.float64) for key, value in probabilities([5] * 20, [0] * 19 + [1]).items()}
    values['state'][0, 0, 5] = .950000000001
    snapshot = {key: value.copy() for key, value in values.items()}
    original = OnlineDecoder(configuration(), parameters())
    smoothed = SmoothedDecoder(configuration(), parameters())
    assert smoothed.push(values) == original.push(values)
    assert smoothed.finish(.2) == original.finish(.2)
    assert smoothed.notes == original.notes
    assert len(smoothed.notes) == 1
    assert smoothed.notes[0]['censored_at_end'] is True
    for key in values:
        assert np.array_equal(values[key], snapshot[key])


def test_ema_uses_only_prefix_evidence_and_events_remain_append_only():
    prefix = probabilities([3] * 8 + [None] * 5 + [7] * 8)
    future = probabilities([19] * 15 + [None] * 10)
    joined = {key: np.concatenate([prefix[key], future[key]]) for key in prefix}
    short = SmoothedDecoder(configuration(), parameters(.25))
    extended = SmoothedDecoder(configuration(), parameters(.25))
    short.push(prefix)
    snapshot = copy.deepcopy(short.events)
    extended.push(joined)
    prior_events = [event for event in extended.events if event['emitted_at_audio_seconds'] <= .21 + 1e-9]
    assert prior_events == snapshot
    short.push(future)
    assert short.events == extended.events
    assert np.array_equal(short.smoothed_state, extended.smoothed_state)
    assert short.events[:len(snapshot)] == snapshot


def test_smoothing_is_packet_invariant_and_fresh_instance_resets_history():
    values = probabilities([3] * 8 + [8, 3] * 4 + [6] * 8 + [None] * 8)
    framewise = SmoothedDecoder(configuration(), parameters(.5, confirm=2))
    chunked = SmoothedDecoder(configuration(), parameters(.5, confirm=2))
    fresh = SmoothedDecoder(configuration(), parameters(.5, confirm=2))
    framewise.push(values)
    for first, last in ((0, 3), (3, 8), (8, 17), (17, 23), (23, 32)):
        chunked.push(sliced(values, first, last), last / 100)
    for decoder in (framewise, chunked):
        decoder.finish(.32)
    assert without_delivery(framewise.notes) == without_delivery(chunked.notes)
    assert without_delivery(framewise.events) == without_delivery(chunked.events)
    assert np.array_equal(framewise.smoothed_state, chunked.smoothed_state)
    assert fresh.framecount == 0 and fresh.smoothed_state is None
    fresh.push(values)
    fresh.finish(.32)
    assert fresh.events == framewise.events and fresh.notes == framewise.notes


def test_past_vote_smoothing_reduces_fret_pulses_without_amplifying_evidence():
    values = probabilities([5] * 12 + [8, 5, 5, 5] * 4 + [5] * 12 + [None] * 8)
    direct = SmoothedDecoder(configuration(), parameters(1, confirm=1))
    smoothed = SmoothedDecoder(configuration(), parameters(.25, confirm=1))
    direct.push(values)
    smoothed.push(values)
    direct.finish(len(values['state']) / 100)
    smoothed.finish(len(values['state']) / 100)
    assert len(smoothed.notes) < len(direct.notes)
    assert {note['fret'] for note in smoothed.notes} == {5}
    weak = probabilities([5] * 20)
    weak['state'][:, 0, 5] = .2
    weak['state'][:, 0, -1] = .8
    uncertain = SmoothedDecoder(configuration(), parameters(.25, confirm=1))
    assert uncertain.push(weak) == []
    assert uncertain.finish(.2) == []
    assert uncertain.smoothed_state[0, 5] == pytest.approx(.2)
    assert uncertain.smoothed_state[0, -1] == pytest.approx(.8)


def test_constructor_and_returned_state_do_not_share_caller_mutable_data():
    config, params = configuration(), parameters(.25, confirm=2)
    frozen_config, frozen_params = copy.deepcopy(config), copy.deepcopy(params)
    decoder = SmoothedDecoder(config, params)
    assert config == frozen_config and params == frozen_params
    config['streaming']['confirm_frames'] = 100
    params['state_alpha'] = 1
    values = probabilities([4] * 8)
    returned = decoder.push(values)
    expected_state = decoder.smoothed_state.copy()
    stored_events = decoder.events
    returned[0]['start'] = 999
    copied_state = decoder.smoothed_state
    copied_state[:] = 0
    values['state'][:] = 0
    assert np.array_equal(decoder.smoothed_state, expected_state)
    assert decoder.events == stored_events
    assert decoder.confirm_frames == 2 and decoder.state_alpha == .25
    decoder.finish(.08)
    notes = decoder.notes
    notes[0]['pitch'] = -1
    assert decoder.notes[0]['pitch'] == 44


@pytest.mark.parametrize('alpha', [0, -.25, 1.01, np.nan, np.inf, -np.inf])
def test_invalid_smoothing_alpha_rejected(alpha):
    with pytest.raises(ValueError, match='alpha'):
        SmoothedDecoder(configuration(), parameters(alpha))


def test_invalid_entire_block_rejected_before_ema_or_history_change():
    decoder = SmoothedDecoder(configuration(), parameters(.25))
    values = probabilities([4] * 4)
    with pytest.raises(ValueError, match='availability'):
        decoder.push(values, .03)
    assert decoder.smoothed_state is None and decoder.framecount == 0
    assert decoder.events == []
    malformed = probabilities([4] * 4)
    malformed['state'][-1, 0, 4] = np.nan
    with pytest.raises(ValueError, match='finite'):
        decoder.push(malformed)
    assert decoder.smoothed_state is None and decoder.framecount == 0
    with pytest.raises(ValueError, match='same frame count'):
        decoder.push(dict(values, pitch=np.zeros((3, 49))))
    assert decoder.smoothed_state is None
    decoder.push(values, .08)
    before_state = decoder.smoothed_state.copy()
    before_events = decoder.events
    for available in (.06, None):
        with pytest.raises(ValueError, match='prior'):
            decoder.push(probabilities([None]), available)
        assert np.array_equal(decoder.smoothed_state, before_state)
        assert decoder.events == before_events and decoder.framecount == 4


def test_empty_blocks_and_eof_obey_original_lifecycle_contract():
    decoder = SmoothedDecoder(configuration(), parameters(.5))
    assert decoder.push(probabilities([])) == []
    assert decoder.smoothed_state is None
    decoder.push(probabilities([4] * 19 + [4], [0] * 19 + [1]))
    assert len(decoder.finish(.2)) == 1
    assert len(decoder.notes) == 1  # Final-frame retrigger stays unconfirmed.
    assert decoder.finish(.2) == []
    before_state = decoder.smoothed_state
    with pytest.raises(RuntimeError):
        decoder.push(probabilities([4]))
    assert np.array_equal(decoder.smoothed_state, before_state)
