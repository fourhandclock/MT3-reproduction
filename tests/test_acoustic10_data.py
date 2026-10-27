import copy
import json

import numpy as np
import pytest
import torch

from acoustic10.data import batch_from_schedule, build_schedule, prepare_targets, validate_config
from acoustic10.training import head_metrics, tensor_digest
from acoustic10.audit import weight_digest
from acoustic10.workflow import aligned_head_metrics


def config():
    return json.load(open('configs/acoustic10.json'))


def records():
    target = {'state': np.full((500, 6), 25, np.int16),
              'onset': np.zeros((500, 6), np.float16), 'offset': np.zeros((500, 6), np.float16),
              'pitch': np.zeros((500, 49), np.uint8)}
    target['state'][20:90, 0] = 5
    return [{'entry': {'id': 'train-only'}, 'features': np.ones((500, 216), np.float32), 'targets': target}]


def test_cap_cannot_expand_or_enable_test_inference():
    value = config()
    validate_config(value)
    for mutate in [lambda c: c['training'].update(max_steps=201),
                   lambda c: c.update(total_optimizer_update_cap=1000),
                   lambda c: c.update(test_inference_allowed=True)]:
        changed = copy.deepcopy(value)
        mutate(changed)
        with pytest.raises(ValueError):
            validate_config(changed)


def test_shared_crop_gain_schedule_is_deterministic_and_exact_budget():
    value = config()
    first = build_schedule(value, records())
    assert first == build_schedule(value, records())
    assert len(first) == 800
    assert {row['epoch'] for row in first} == set(range(1, 6))
    assert all(row['frames'] == 200 and 0 <= row['start_frame'] <= 300 for row in first)
    assert len({row['gain_log'] for row in first}) > 100


def test_delay_applies_before_crop_with_same_raw_frame_support_and_inputs():
    value = config()
    model = json.load(open('configs/streaming.json'))
    source = records()
    baseline = prepare_targets(source, 0, value, model)
    delayed = prepare_targets(source, 6, value, model)
    np.testing.assert_array_equal(baseline[0]['mask'], delayed[0]['mask'])
    descriptor = [{'record_index': 0, 'start_frame': 22, 'frames': 60, 'gain_log': .2}]
    norm = {'mean': [0] * 216, 'std': [1] * 216}
    x, immediate = batch_from_schedule(descriptor, source, baseline, norm, 8)
    y, shifted = batch_from_schedule(descriptor, source, delayed, norm, 8)
    assert torch.equal(x, y)
    assert immediate['state'][0, 0, 0] == 5
    assert shifted['state'][0, 0, 0] == 25  # Source frame16, not a crop-local shifted frame0.
    assert shifted['state'][0, 4, 0] == 5
    assert torch.equal(immediate['mask'], shifted['mask'])
    assert np.all(source[0]['features'] == 1)


def test_head_score_uses_common_source_span_instead_of_raw_availability_cutoff():
    value = config()
    targets = records()[0]['targets']
    probability = np.zeros((506, 6, 26), np.float32)
    probability[:, :, 25] = 1
    for i in range(500):
        probability[i + 6, 0, :] = 0
        probability[i + 6, 0, targets['state'][i, 0]] = 1
    metric = aligned_head_metrics({'state': probability}, targets, value, 6, 500)
    assert metric['exact_state_f1'] == metric['active_fret_accuracy'] == 1
    assert metric['source_frames_scored'] == 500
    assert metric['prediction_start_frame'] == 6


def test_inactive_only_correctness_cannot_pass_active_fret_metric():
    target = np.full((200, 6), 25, np.int16)
    target[50:100, 0] = 5
    probability = np.zeros((200, 6, 26), np.float32)
    probability[:, :, 25] = 1
    metric = head_metrics(probability, target, np.ones(200), 25)
    assert metric['active_fret_accuracy'] == metric['exact_state_f1'] == 0
    assert metric['active_frames'] == 50


def test_model_digest_matches_independent_canonical_audit():
    state = {'first': torch.arange(9, dtype=torch.float32).reshape(3, 3), 'second': torch.ones(4)}
    assert tensor_digest(state) == weight_digest(state)
