import copy
import json

import numpy as np
import pytest

from baseline.download import sha256
from optimization13 import workflow
from optimization13.data import build_schedule, validate_config


def configuration():
    return json.load(open('configs/optimization13.json'))


def test_continuation_recipe_rejects_new_rate_extra_epochs_or_test_scope():
    value = configuration()
    validate_config(value)
    for change in [lambda c: c['training'].update(learning_rate=.001),
                   lambda c: c['training'].update(epochs=6),
                   lambda c: c.update(data_epoch_start=0),
                   lambda c: c.update(planned_new_optimizer_updates=1000),
                   lambda c: c.update(test_inference_allowed=True)]:
        changed = copy.deepcopy(value)
        change(changed)
        with pytest.raises(ValueError):
            validate_config(changed)


def test_next_epoch_schedule_has_new_but_shared_deterministic_exposure():
    value = configuration()
    records = [{'entry': {'id': 'train'}, 'features': np.zeros((1200, 216), np.float32)}]
    schedule = build_schedule(value, records)
    assert schedule == build_schedule(value, records)
    assert len(schedule) == 1000
    assert {row['epoch'] for row in schedule} == set(range(6, 11))
    assert all(row['frames'] == 200 and 0 <= row['start_frame'] <= 1000 for row in schedule)


def parent_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, 'ROOT', tmp_path)
    root = tmp_path / 'outputs/normalization12'
    probability_path = root / 'validation/probabilities/group1/clip.npz'
    probability_path.parent.mkdir(parents=True)
    probability = {'state': np.ones((4, 6, 26), np.float32) / 26}
    np.savez_compressed(probability_path, **probability)
    note = {'pitch': 40, 'start': .1, 'end': .2, 'on_emitted_at_audio_seconds': .14,
            'off_emitted_at_audio_seconds': .22, 'on_delivered_at_seconds': .15, 'off_delivered_at_seconds': .23}
    heads = {'exact_state': {'f1': .5}, 'active_fret_accuracy': .5, 'pitch': {'overall': {'f1': .5}}}
    metrics = {'onset': {'f1': .5}, 'predicted': [dict(note)]}
    path = root / 'validation/results/group1/clip-metrics.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'probability_path': 'validation/probabilities/group1/clip.npz',
                               'probability_sha256': sha256(probability_path), 'notes': [note],
                               'head_metrics': heads, 'pitch_metrics': metrics, 'tab_metrics': metrics}))
    return probability, [copy.deepcopy(note)], heads, copy.deepcopy(metrics), copy.deepcopy(metrics)


def test_parent_reference_preserves_source_predictions_and_remeasures_clock(tmp_path, monkeypatch):
    p, notes, heads, metrics, tabs = parent_fixture(tmp_path, monkeypatch)
    for note in [notes[0], metrics['predicted'][0], tabs['predicted'][0]]:
        note['on_delivered_at_seconds'] = .19
    checked = workflow.check_parent_replay('group1', 'clip', p, notes, heads, metrics, tabs)
    assert checked['probabilities_bit_identical'] and checked['source_notes_identical']


def test_parent_reference_rejects_shifted_source(tmp_path, monkeypatch):
    p, notes, heads, metrics, tabs = parent_fixture(tmp_path, monkeypatch)
    notes[0]['start'] = .11
    with pytest.raises(ValueError, match='source notes'):
        workflow.check_parent_replay('group1', 'clip', p, notes, heads, metrics, tabs)
