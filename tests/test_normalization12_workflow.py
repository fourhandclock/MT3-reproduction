import copy
import json

import numpy as np
import pytest

from baseline.download import sha256
from normalization12 import workflow
from normalization12.data import validate_config


def configuration():
    return json.load(open('configs/normalization12.json'))


def test_only_normalization_and_fixed_new_budget_may_change():
    value = configuration()
    validate_config(value)
    for change in [lambda c: c['training'].update(learning_rate=.002),
                   lambda c: c.update(planned_new_optimizer_updates=500),
                   lambda c: c.update(common_target_delay_frames=6),
                   lambda c: c.update(test_inference_allowed=True)]:
        other = copy.deepcopy(value)
        change(other)
        with pytest.raises(ValueError):
            validate_config(other)


def historical_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, 'ROOT', tmp_path)
    root = tmp_path / 'outputs/spectral11'
    probability_path = root / 'validation/probabilities/short128/clip.npz'
    probability_path.parent.mkdir(parents=True)
    probability = {'state': np.ones((4, 6, 26), np.float32) / 26}
    np.savez_compressed(probability_path, **probability)
    note = {'pitch': 40, 'string': 0, 'fret': 0, 'start': .1, 'end': .2,
            'on_emitted_at_audio_seconds': .14, 'off_emitted_at_audio_seconds': .22,
            'on_delivered_at_seconds': .15, 'off_delivered_at_seconds': .23}
    heads = {'exact_state': {'f1': .5}, 'active_fret_accuracy': .5, 'pitch': {'overall': {'f1': .5}}}
    pitch = {'onset': {'f1': .5}, 'predicted': [dict(note)]}
    tabs = {'onset': {'f1': .4}, 'predicted': [dict(note)]}
    row = {'probability_path': 'validation/probabilities/short128/clip.npz',
           'probability_sha256': sha256(probability_path), 'notes': [note],
           'head_metrics': heads, 'pitch_metrics': pitch, 'tab_metrics': tabs}
    file = root / 'validation/results/short128/clip-metrics.json'
    file.parent.mkdir(parents=True)
    file.write_text(json.dumps(row))
    return probability, [copy.deepcopy(note)], heads, copy.deepcopy(pitch), copy.deepcopy(tabs)


def test_historical_control_replay_allows_remeasured_delivery_only(tmp_path, monkeypatch):
    probability, notes, heads, pitch, tabs = historical_fixture(tmp_path, monkeypatch)
    for note in [notes[0], pitch['predicted'][0], tabs['predicted'][0]]:
        note['on_delivered_at_seconds'] = .19
        note['off_delivered_at_seconds'] = .29
    result = workflow.check_control_replay('clip', probability, notes, heads, pitch, tabs)
    assert result['probabilities_bit_identical']
    assert result['source_notes_identical'] and result['note_and_tab_scores_identical']


def test_historical_control_replay_rejects_changed_source_or_acoustic_score(tmp_path, monkeypatch):
    probability, notes, heads, pitch, tabs = historical_fixture(tmp_path, monkeypatch)
    notes[0]['start'] = .11
    with pytest.raises(ValueError, match='source notes'):
        workflow.check_control_replay('clip', probability, notes, heads, pitch, tabs)
    notes[0]['start'] = .1
    pitch['onset']['f1'] = .6
    with pytest.raises(ValueError, match='note scores'):
        workflow.check_control_replay('clip', probability, notes, heads, pitch, tabs)
