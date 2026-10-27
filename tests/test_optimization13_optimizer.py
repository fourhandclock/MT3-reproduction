"""Continuation must preserve AdamW history, not merely reload model weights."""

import copy
import json
from pathlib import Path

import pytest
import torch

from normalization12.model import CausalCNNGRU
from optimization13.optimizer import recursive_digest, restore_parent


@pytest.fixture(scope='module')
def small_parent():
    config = {
        'feature': {'n_bins': 8},
        'model': {'cnn_norm_groups': 1, 'channels': [8, 8, 8],
                  'projection_size': 8, 'gru_hidden_size': 8, 'gru_layers': 2,
                  'bidirectional': False, 'dropout': .1, 'output_size': 8,
                  'max_fret': 2, 'pitch_min': 40, 'pitch_max': 42},
    }
    torch.manual_seed(900)
    model = CausalCNNGRU(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.0001)
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.full_like(parameter, (index + 1) / 100.)
    # One synthetic optimizer step initializes real AdamW moment structures;
    # no audio or acoustic labels are involved in these contract tests.
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    for state in optimizer.state.values():
        state['step'].fill_(250)
    optimizer.param_groups[0]['initial_lr'] = .001
    return {'config': config, 'model': copy.deepcopy(model.state_dict()),
            'optimizer': copy.deepcopy(optimizer.state_dict()), 'step': 250,
            'normalization': {'mean': [0.] * 8, 'std': [1.] * 8},
            'positive_weights': {'onset': 20., 'offset': 20., 'pitch': 8.}}


def test_complete_restore_preserves_weights_moments_options_and_parent(small_parent):
    before = recursive_digest(small_parent)
    model, optimizer, proof = restore_parent(small_parent, torch.device('cpu'))
    assert recursive_digest(model.state_dict()) == recursive_digest(small_parent['model'])
    assert recursive_digest(optimizer.state_dict()) == recursive_digest(small_parent['optimizer'])
    assert proof['parameter_states'] == 34
    assert proof['actual_optimizer_step_counters'] == {'minimum': 250, 'maximum': 250, 'parameter_states': 34}
    assert proof['new_optimizer_updates'] == 0
    assert proof['optimizer_options']['initial_lr'] == .001
    assert proof['optimizer_options']['lr'] == .0001
    assert proof['parent_optimizer_sha256'] == proof['restored_optimizer_sha256']
    assert proof['historical_rng_restored'] is False
    assert proof['historical_amp_scaler_restored'] is False
    json.dumps(proof, allow_nan=False)
    assert recursive_digest(small_parent) == before


def test_next_step_is_251_and_does_not_mutate_saved_parent(small_parent):
    before = recursive_digest(small_parent)
    model, optimizer, _ = restore_parent(small_parent, 'cpu')
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert all(state['step'].item() == 251 for state in optimizer.state.values())
    assert recursive_digest(optimizer.state_dict()) != recursive_digest(small_parent['optimizer'])
    assert recursive_digest(model.state_dict()) != recursive_digest(small_parent['model'])
    assert recursive_digest(small_parent) == before


@pytest.mark.parametrize('option,value', [
    ('lr', .001), ('lr', float('nan')), ('weight_decay', 0.), ('eps', 1e-7),
    ('betas', (.8, .999)), ('betas', [.9, .999]), ('amsgrad', True),
    ('capturable', True), ('foreach', False), ('initial_lr', .0001),
    ('maximize', 0), ('fused', False),
])
def test_noncanonical_optimizer_recipe_is_rejected(small_parent, option, value):
    parent = copy.deepcopy(small_parent)
    parent['optimizer']['param_groups'][0][option] = value
    with pytest.raises(ValueError):
        restore_parent(parent, 'cpu')


@pytest.mark.parametrize('alteration', [
    'missing_state', 'missing_moment', 'unknown_moment', 'wrong_shape',
    'wrong_dtype', 'nonfinite_moment', 'negative_square', 'counter249',
    'counter_fraction', 'counter_vector', 'counter_dtype', 'wrong_order',
    'duplicate_id', 'multiple_groups', 'extra_group_option',
    'wrong_weight_shape', 'wrong_weight_dtype', 'nonfinite_weight',
    'reordered_weights', 'missing_metadata', 'parent_step',
])
def test_corrupt_or_ambiguous_parent_is_rejected(small_parent, alteration):
    parent = copy.deepcopy(small_parent)
    state = parent['optimizer']['state'][0]
    names = list(parent['model'])
    group = parent['optimizer']['param_groups'][0]
    if alteration == 'missing_state':
        del parent['optimizer']['state'][0]
    elif alteration == 'missing_moment':
        del state['exp_avg']
    elif alteration == 'unknown_moment':
        state['unknown'] = torch.tensor(1.)
    elif alteration == 'wrong_shape':
        state['exp_avg'] = torch.zeros(1)
    elif alteration == 'wrong_dtype':
        state['exp_avg'] = state['exp_avg'].double()
    elif alteration == 'nonfinite_moment':
        state['exp_avg'].flatten()[0] = float('inf')
    elif alteration == 'negative_square':
        state['exp_avg_sq'].flatten()[0] = -1.
    elif alteration == 'counter249':
        state['step'].fill_(249)
    elif alteration == 'counter_fraction':
        state['step'].fill_(250.5)
    elif alteration == 'counter_vector':
        state['step'] = torch.tensor([250.])
    elif alteration == 'counter_dtype':
        state['step'] = state['step'].double()
    elif alteration == 'wrong_order':
        group['params'][0], group['params'][1] = group['params'][1], group['params'][0]
    elif alteration == 'duplicate_id':
        group['params'][0] = group['params'][1]
    elif alteration == 'multiple_groups':
        parent['optimizer']['param_groups'].append(copy.deepcopy(group))
    elif alteration == 'extra_group_option':
        group['unknown'] = 1
    elif alteration == 'wrong_weight_shape':
        parent['model'][names[0]] = torch.zeros(1)
    elif alteration == 'wrong_weight_dtype':
        parent['model'][names[0]] = parent['model'][names[0]].double()
    elif alteration == 'nonfinite_weight':
        parent['model'][names[0]].flatten()[0] = float('nan')
    elif alteration == 'reordered_weights':
        parent['model'] = dict(reversed(list(parent['model'].items())))
    elif alteration == 'missing_metadata':
        del parent['normalization']
    elif alteration == 'parent_step':
        parent['step'] = 249
    with pytest.raises(ValueError):
        restore_parent(parent, 'cpu')


@pytest.mark.parametrize('field,value', [
    ('mean', [0.] * 7), ('std', [0.] * 8), ('std', [float('nan')] * 8),
])
def test_invalid_normalization_is_rejected(small_parent, field, value):
    parent = copy.deepcopy(small_parent)
    parent['normalization'][field] = value
    with pytest.raises(ValueError, match='normalization'):
        restore_parent(parent, 'cpu')


def test_wrong_requested_learning_rate_is_rejected(small_parent):
    with pytest.raises(ValueError, match='Held learning rate'):
        restore_parent(small_parent, 'cpu', expected_lr=.001)


def test_digest_is_order_independent_but_retains_types_shapes_and_values():
    source = {'state': {0: {'step': torch.tensor(250.), 'moment': torch.tensor([.1, .2])}},
              'options': {'flag': False, 'rate': .0001, 'betas': (.9, .999)}}
    assert recursive_digest(source) == recursive_digest(dict(reversed(list(source.items()))))
    for path, new in [('flag', 0), ('rate', .001), ('betas', [.9, .999])]:
        changed = copy.deepcopy(source)
        changed['options'][path] = new
        assert recursive_digest(changed) != recursive_digest(source)
    for tensor in [torch.tensor([.1, .3]), torch.tensor([[.1, .2]]), torch.tensor([.1, .2], dtype=torch.float64)]:
        changed = copy.deepcopy(source)
        changed['state'][0]['moment'] = tensor
        assert recursive_digest(changed) != recursive_digest(source)


def test_scalar_digest_and_dictionary_key_types_are_distinct():
    assert recursive_digest(torch.tensor(250.)) != recursive_digest(torch.tensor([250.]))
    assert recursive_digest({0: None}) != recursive_digest({'0': None})
    assert recursive_digest((1, 2)) != recursive_digest([1, 2])


def test_group_metadata_does_not_change_full_constructor_draws():
    config = json.loads((Path(__file__).parents[1] / 'configs/streaming.json').read_text())
    first = copy.deepcopy(config)
    second = copy.deepcopy(config)
    first['model']['cnn_norm_groups'] = 8
    second['model']['cnn_norm_groups'] = 1
    torch.manual_seed(43)
    group8 = CausalCNNGRU(first)
    rng8 = torch.get_rng_state().clone()
    torch.manual_seed(43)
    group1 = CausalCNNGRU(second)
    assert recursive_digest(group8.state_dict()) == recursive_digest(group1.state_dict())
    assert torch.equal(rng8, torch.get_rng_state())
    assert sum(value.numel() for value in group8.parameters()) == 931145
    assert len(list(group8.parameters())) == 34
    assert all(layer.normalization.num_groups == 8 for layer in group8.normalizations)
    assert all(layer.normalization.num_groups == 1 for layer in group1.normalizations)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is optional for checkpoint restoration')
def test_cuda_migration_preserves_device_agnostic_optimizer_digest(small_parent):
    _, optimizer, proof = restore_parent(small_parent, torch.device('cuda'))
    assert recursive_digest(optimizer.state_dict()) == recursive_digest(small_parent['optimizer'])
    assert all(state['exp_avg'].device.type == 'cuda' for state in optimizer.state.values())
    assert proof['optimizer_state_identical'] is True
