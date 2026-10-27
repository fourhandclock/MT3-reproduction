import copy
import json
from pathlib import Path

import numpy as np
import pytest

from decoder14.core import (BASELINE, candidates, choose, delayed_oracle, packet_latency,
                            replay, score_case, validate_config, without_compute)
from decoder14.workflow import identity_check, projection_proof


CONFIG = {'feature': {'sample_rate': 16000, 'hop_length': 160},
          'model': {'max_fret': 24, 'pitch_min': 40, 'pitch_max': 88},
          'decoder': {'minimum_note_seconds': .04, 'minimum_retrigger_seconds': .06},
          'streaming': {'confirm_frames': 4, 'release_confirm_frames': 2}}
SCORE = {'score_seconds': .8, 'frame_rate': 100, 'onset_tolerance_seconds': .05,
         'offset_tolerance_seconds': .05, 'offset_ratio': .2}


def note(start=.2, end=.5, fret=3):
    return {'start': start, 'end': end, 'string': 0, 'fret': fret, 'pitch': 40 + fret}


def test_fixed_grid_is_unique_and_identity_first():
    cfg = json.loads(Path('configs/decoder14.json').read_text())
    grid = candidates(cfg)
    assert grid[0] == BASELINE and len(grid) == 27
    assert len({tuple(row.values()) for row in grid}) == 27


@pytest.mark.parametrize('key,value', [('new_optimizer_updates', 1), ('source_delay_frames', 6),
    ('packet_ms', 40), ('neural_inference_calls', 1), ('test_inference_allowed', True),
    ('production_promotion', True), ('models', ['group8_250']), ('source_root', 'outputs/streaming')])
def test_protocol_changes_are_rejected(key, value):
    cfg = json.loads(Path('configs/decoder14.json').read_text())
    cfg[key] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_selection_uses_notes_and_stable_identity_tie():
    rows = [{'candidate': i, 'onset_f1': .2, 'onset_offset_f1': .1} for i in range(3)]
    assert choose(rows) == 0
    rows[2]['onset_offset_f1'] = .15
    assert choose(rows) == 2
    rows[1]['onset_f1'] = .21
    assert choose(rows) == 1
    rows[1]['onset_f1'] = float('nan')
    with pytest.raises(ValueError):
        choose(rows)


@pytest.mark.parametrize('backdate', [0., .03, .06])
def test_130ms_oracle_translation_preserves_safe_source_intervals_and_delays_availability(backdate):
    ref = [note(), note(.5, .7)]
    values, _ = delayed_oracle(ref, 100, CONFIG, 1., 0)
    delayed, _ = delayed_oracle(ref, 100, CONFIG, 1., 13)
    pars = {**BASELINE, 'onset_backdate_seconds': backdate}
    original, _, _ = replay(values, CONFIG, pars, 1., 0.)
    corrected, events, _ = replay(delayed, CONFIG, pars, 1., .13)
    assert len(corrected) == len(original) == 2  # includes same-fret retrigger
    for left, right in zip(original, corrected):
        assert (left['start'], left['end']) == pytest.approx((right['start'], right['end']))
        assert right['on_emitted_at_audio_seconds'] - left['on_emitted_at_audio_seconds'] == pytest.approx(.13, abs=.011)
    _, raw, _ = replay(delayed, CONFIG, pars, 1., 0.)
    assert projection_proof(raw, events, .13)['source_translation_verified']
    assert all('delivered_at_seconds' not in event for event in events)
    assert all('on_delivered_at_seconds' not in row for row in corrected)


def test_compensation_applies_once_to_releases_and_never_fakes_the_eof_tail():
    values, _ = delayed_oracle([note(.2, 2.)], 100, CONFIG, 1., 13)
    notes, events, _ = replay(values, CONFIG, {**BASELINE, 'onset_backdate_seconds': 0.}, 1.)
    assert notes[0]['start'] == pytest.approx(.2)
    assert notes[0]['end'] == pytest.approx(.87)
    assert notes[0]['off_emitted_at_audio_seconds'] == 1.
    assert notes[0]['censored_at_end'] and events[-1]['flush']


def test_packet_size_changes_availability_without_moving_source_intervals():
    values, _ = delayed_oracle([note(.21, .51)], 100, CONFIG, 1., 13)
    a, _, _ = replay(values, CONFIG, BASELINE, 1., packet_ms=20)
    b, _, _ = replay(values, CONFIG, BASELINE, 1., packet_ms=80)
    assert [(n['start'], n['end']) for n in a] == [(n['start'], n['end']) for n in b]
    assert b[0]['on_emitted_at_audio_seconds'] >= a[0]['on_emitted_at_audio_seconds']


def test_packet_arrival_uses_integer_sample_arithmetic_like_original_pcm_stream():
    values, _ = delayed_oracle([note(1.21, 1.38)], 200, CONFIG, 2., 0)
    _, events, _ = replay(values, CONFIG, BASELINE, 2., 0.)
    assert events[-1]['emitted_at_audio_seconds'] == 1.4


@pytest.mark.parametrize('packet', [0, 15, -20, float('nan')])
def test_invalid_packet_rejected(packet):
    values, _ = delayed_oracle([], 10, CONFIG, .1, 13)
    with pytest.raises(ValueError):
        replay(values, CONFIG, BASELINE, .1, packet_ms=packet)


def test_empty_replay_and_no_match_latency_are_explicit():
    values, _ = delayed_oracle([], 0, CONFIG, 0., 13)
    notes, events, counts = replay(values, CONFIG, BASELINE, 0.)
    assert notes == events == [] and counts['total'] == 0
    assert packet_latency([], [], SCORE, 0.)['p50_ms'] is None


def test_replay_does_not_mutate_probabilities_and_remains_prefix_causal():
    values, _ = delayed_oracle([note()], 100, CONFIG, 1., 13)
    original = {key: value.copy() for key, value in values.items()}
    _, events, _ = replay(values, CONFIG, BASELINE, 1.)
    values['state'][70:] = np.roll(values['state'][70:], 1, axis=-1)
    _, changed, _ = replay(values, CONFIG, BASELINE, 1.)
    assert [e for e in events if e['emitted_at_audio_seconds'] <= .7] == [
        e for e in changed if e['emitted_at_audio_seconds'] <= .7]
    for key in ['onset', 'offset', 'pitch']:
        np.testing.assert_array_equal(values[key], original[key])


def test_cached_identity_ignores_only_measured_completion_fields():
    values, _ = delayed_oracle([note()], 100, CONFIG, 1., 13)
    notes, events, _ = replay(values, CONFIG, BASELINE, 1.)
    case = score_case([note()], notes, SCORE, 1.)
    previous = copy.deepcopy({**case, 'notes': notes, 'events': events})
    previous['notes'][0]['on_delivered_at_seconds'] = 99.
    previous['events'][0]['delivered_at_seconds'] = 99.
    assert identity_check({'previous': previous}, notes, events, case) is None
    previous['events'][0]['emitted_at_audio_seconds'] += .01
    with pytest.raises(ValueError, match='Identity'):
        identity_check({'previous': previous}, notes, events, case)


def test_projection_proof_rejects_double_compensation_or_moved_availability():
    values, _ = delayed_oracle([note()], 100, CONFIG, 1., 13)
    _, raw, _ = replay(values, CONFIG, BASELINE, 1., 0.)
    _, projected, _ = replay(values, CONFIG, BASELINE, 1., .13)
    changed = copy.deepcopy(projected)
    changed[0]['source_time'] -= .13
    with pytest.raises(ValueError, match='exactly once'):
        projection_proof(raw, changed, .13)
    changed = copy.deepcopy(projected)
    changed[0]['emitted_at_audio_seconds'] -= .13
    with pytest.raises(ValueError, match='availability'):
        projection_proof(raw, changed, .13)


@pytest.fixture
def rendered_review(tmp_path):
    from decoder14.core import MODELS
    from decoder14.report import write_report
    from streaming.evaluate import save_json
    clip = '01_Jazz2-187-F#_comp'
    metrics = {'onset': {'f1': .1}, 'onset_offset': {'f1': .05}, 'frame': {'f1': .2},
               'reference': [], 'predicted': [], 'onset_matches': [], 'score_duration': 10.}
    modes, selected = {}, {}
    exact_bytes = b'{\r\n  "notes": []\r\n}\r\n'
    for model in MODELS:
        selected[model] = {'candidate': 22, 'parameters': BASELINE}
        for variant in ['strict', 'selected']:
            mode = model + '_' + variant
            row = {'id': clip, 'model': model, 'mode': mode, 'variant': variant, 'parameters': BASELINE,
                   'notes': [], 'events': [], 'pitch_metrics': metrics, 'tab_metrics': {'onset': {'f1': .08}},
                   'head_metrics': {'exact_state_f1': .07, 'pitch': {'overall': {'f1': .1}, 'low_register': {'f1': .09}}},
                   'packet_latency': {'samples_ms': []}, 'relativeaudio': 'validation/clips/' + clip + '.wav'}
            stem = tmp_path / 'validation/results' / mode / clip
            save_json(stem.with_name(stem.name + '-metrics.json'), row)
            stem.with_suffix('.json').write_bytes(exact_bytes)
            modes[mode] = {'onset_f1': .1, 'onset_offset_f1': .05, 'frame_f1': .2,
                           'tab_onset_f1': .08, 'matched_onsets': 1, 'predicted_notes': 10}
    save_json(tmp_path / 'selection.json', {'models': selected})
    save_json(tmp_path / 'validation/summary.json', {'modes': modes, 'oracles': {}})
    for name in ['protocol.json', 'grid.json', 'oracle.json']:
        save_json(tmp_path / name, {})
    write_report(tmp_path)
    page = (tmp_path / 'review.html').read_text(encoding='utf-8')
    payload = json.JSONDecoder().raw_decode(page.split('const DATA=', 1)[1])[0]
    return payload, exact_bytes


def test_musical_sharp_filenames_are_url_encoded_in_audio_and_all_exports(rendered_review):
    payload, _ = rendered_review
    for row in payload:
        assert '%23' in row['audio'] and '#' not in row['audio']
        assert all('%23' in link and '#' not in link and '\\' not in link for link in row['links'].values())


def test_download_payload_preserves_exact_windows_json_bytes(rendered_review):
    payload, original_bytes = rendered_review
    assert all(row['json'].encode('utf-8') == original_bytes for row in payload)
