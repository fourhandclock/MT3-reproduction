import copy
import json

import numpy as np
import pytest
import torch

from spectral11.training import actual_optimizer_steps, verified_sanity
from spectral11.workflow import aligned_heads, validate_fixed_settings


def configuration():
    return json.load(open('configs/spectral11.json'))


def model_configuration():
    return json.load(open('configs/streaming.json'))


def test_diagnostic_or_decoder_tuning_is_rejected():
    value = configuration()
    validate_fixed_settings(value)
    for key, replacement in [('decoder_parameters', {'onset_threshold': .4}),
                             ('pitch_diagnostics', {'threshold': .25, 'low_register_midi': [40, 51]}),
                             ('production_promotion', True)]:
        changed = copy.deepcopy(value)
        changed[key] = replacement
        with pytest.raises(ValueError):
            validate_fixed_settings(changed)


def test_both_head_streams_align_to_original_source_and_low_pitch_mask():
    value = configuration()
    target = {'state': np.full((1000, 6), 25, np.int16), 'pitch': np.zeros((1000, 49), np.uint8)}
    target['state'][2:12, 0] = 0  # Entirely inside the excluded source prefix.
    target['pitch'][2:12, 0] = 1
    target['state'][900:999, 0] = 0
    target['pitch'][900:999, 0] = 1
    state = np.zeros((1200, 6, 26), np.float32)
    state[:, :, 25] = 1
    pitch = np.zeros((1200, 49), np.float32)
    for i in range(26, 1000):
        state[i + 13, 0, :] = 0
        state[i + 13, 0, target['state'][i, 0]] = 1
        pitch[i + 13] = target['pitch'][i]
    heads = aligned_heads({'state': state, 'pitch': pitch}, target, value, model_configuration(), 1000)
    assert heads['exact_state_f1'] == heads['pitch']['overall']['f1'] == heads['pitch']['low_register']['f1'] == 1
    assert heads['pitch']['low_register']['reference_count'] == 99
    assert heads['prediction_start_frame'] == 13 and heads['source_frames_scored'] == 1000


def test_low_pitch_diagnostic_does_not_include_high_notes():
    value = configuration()
    target = {'state': np.full((1000, 6), 25, np.int16), 'pitch': np.zeros((1000, 49), np.uint8)}
    target['pitch'][100:200, 30] = 1
    state = np.zeros((1200, 6, 26), np.float32)
    state[:, :, 25] = 1
    pitch = np.zeros((1200, 49), np.float32)
    pitch[113:213, 30] = 1
    heads = aligned_heads({'state': state, 'pitch': pitch}, target, value, model_configuration(), 1000)
    assert heads['pitch']['overall']['f1'] == 1
    assert heads['pitch']['low_register']['reference_count'] == 0


def test_optimizer_counters_measure_actual_adam_steps():
    parameter = torch.nn.Parameter(torch.tensor([1., 2.]))
    optimizer = torch.optim.AdamW([parameter])
    assert actual_optimizer_steps(optimizer)['minimum'] == 0
    for _ in range(3):
        optimizer.zero_grad()
        parameter.square().sum().backward()
        optimizer.step()
    assert actual_optimizer_steps(optimizer) == {'minimum': 3, 'maximum': 3, 'parameter_states': 1}


def test_sanity_reference_reuses_passed_previous_evidence_without_updates():
    reference = verified_sanity(configuration())
    assert reference['passed'] and reference['prior_optimizer_updates'] == 100
    assert reference['new_optimizer_updates'] == 0
