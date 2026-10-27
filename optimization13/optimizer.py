"""Restore an AdamW continuation without discarding its accumulated moments.

Tensor devices are deliberately excluded from the digest; shape, dtype, value,
container type, scalar type, and every optimizer option remain part of it. This
permits CPU-to-CUDA restoration while rejecting numerical or recipe changes.
"""

import copy
import hashlib
import math
import struct

import torch

from normalization12.model import CausalCNNGRU


def recursive_digest(value):
    """Hash nested checkpoint data deterministically, independent of device."""
    digest = hashlib.sha256()

    def emit(part):
        digest.update(len(part).to_bytes(8, 'big'))
        digest.update(part)

    def visit(item):
        if isinstance(item, torch.Tensor):
            if item.layout != torch.strided:
                raise ValueError('Only dense checkpoint tensors can be hashed.')
            emit(b'tensor')
            emit(str(item.dtype).encode())
            visit(list(item.shape))
            data = item.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            emit(data.numpy().tobytes())
        elif isinstance(item, dict):
            emit(b'dict')
            emit(str(len(item)).encode())
            # AdamW has integer state keys and string option keys. Their types
            # remain distinct, and insertion order is immaterial to the hash.
            if any(type(key) not in (str, int) for key in item):
                raise ValueError('Checkpoint dictionary keys must be strings or integers.')
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            emit(b'list' if isinstance(item, list) else b'tuple')
            emit(str(len(item)).encode())
            for child in item:
                visit(child)
        elif item is None:
            emit(b'none')
        elif type(item) is bool:
            emit(b'bool')
            emit(b'1' if item else b'0')
        elif type(item) is int:
            emit(b'int')
            emit(str(item).encode())
        elif type(item) is float:
            emit(b'float')
            emit(struct.pack('!d', item))
        elif type(item) is str:
            emit(b'str')
            emit(item.encode())
        else:
            raise ValueError('Unsupported checkpoint value type: ' + type(item).__name__)

    visit(value)
    return digest.hexdigest()


def _number(value, expected, description):
    if type(value) not in (int, float) or not math.isfinite(value) or value != expected:
        raise ValueError(description + ' differs from the frozen continuation recipe.')


def _validate_metadata(parent):
    required = {'model', 'optimizer', 'step', 'config', 'normalization', 'positive_weights'}
    if not required.issubset(parent):
        raise ValueError('Parent checkpoint is missing continuation metadata.')
    if type(parent['step']) is not int or parent['step'] != 250:
        raise ValueError('Continuation must start from parent step 250.')
    normalization = parent['normalization']
    bins = parent['config']['feature']['n_bins']
    if not isinstance(normalization, dict) or set(normalization) != {'mean', 'std'}:
        raise ValueError('Parent normalization must contain mean and std.')
    for name in ['mean', 'std']:
        values = normalization[name]
        if (not isinstance(values, list) or len(values) != bins
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)
                or (name == 'std' and any(value <= 0 for value in values))):
            raise ValueError('Parent normalization has an invalid ' + name + ' array.')
    weights = parent['positive_weights']
    if (not isinstance(weights, dict) or set(weights) != {'onset', 'offset', 'pitch'}
            or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                   for value in weights.values())):
        raise ValueError('Parent positive weights are invalid.')


def _validate_group(saved, parameter_count, expected_lr):
    if not isinstance(saved, dict) or set(saved) != {'state', 'param_groups'}:
        raise ValueError('Parent optimizer state dictionary is invalid.')
    groups = saved['param_groups']
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError('Continuation requires the single saved AdamW parameter group.')
    group = groups[0]
    expected_options = {
        'lr': expected_lr, 'betas': (.9, .999), 'eps': 1e-8,
        'weight_decay': .0001, 'amsgrad': False, 'maximize': False,
        'foreach': None, 'capturable': False, 'differentiable': False,
        'fused': None, 'decoupled_weight_decay': True, 'initial_lr': .001,
    }
    if set(group) != set(expected_options) | {'params'}:
        raise ValueError('Parent AdamW options differ from the saved canonical options.')
    for name, expected in expected_options.items():
        if name in ['lr', 'eps', 'weight_decay', 'initial_lr']:
            _number(group[name], expected, 'Parent AdamW ' + name)
        elif type(group[name]) is not type(expected) or group[name] != expected:
            raise ValueError('Parent AdamW ' + name + ' differs from the frozen recipe.')
    if (not isinstance(group['params'], list)
            or any(type(value) is not int for value in group['params'])
            or group['params'] != list(range(parameter_count))):
        raise ValueError('Saved optimizer parameter ordering differs from model ordering.')
    if (not isinstance(saved['state'], dict)
            or any(type(key) is not int for key in saved['state'])
            or set(saved['state']) != set(group['params'])):
        raise ValueError('Parent AdamW is missing parameter moment states.')


def restore_parent(parent, device, expected_lr=1e-4):
    """Return the parent model, its exact AdamW state, and a JSON-ready proof.

    The caller sets its new common random seed before invoking this function.
    Construction consumes the ordinary model initialization draws, then every
    weight is overwritten with the parent. Historical RNG and AMP scaler state
    were not saved, so this is a seeded continuation rather than an exact replay
    of an uninterrupted historical training process. No update occurs here.
    """
    _number(expected_lr, .0001, 'Held learning rate')
    _validate_metadata(parent)
    original_optimizer_digest = recursive_digest(parent['optimizer'])
    original_model_digest = recursive_digest(parent['model'])
    metadata_digest = recursive_digest({key: parent[key]
                                       for key in ['config', 'normalization', 'positive_weights']})
    model = CausalCNNGRU(copy.deepcopy(parent['config'])).to(device)
    parameters = list(model.named_parameters())
    if list(parent['model']) != [name for name, _ in parameters]:
        raise ValueError('Parent model keys/order differ from the expected parameter order.')
    _validate_group(parent['optimizer'], len(parameters), expected_lr)
    for index, (name, parameter) in enumerate(parameters):
        weight = parent['model'][name]
        if (not isinstance(weight, torch.Tensor) or weight.shape != parameter.shape
                or weight.dtype != parameter.dtype or not bool(torch.isfinite(weight).all())):
            raise ValueError('Parent weight shape/dtype/value is invalid for ' + name)
        state = parent['optimizer']['state'][index]
        if not isinstance(state, dict) or set(state) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError('Parent moment fields are missing or unknown for ' + name)
        step = state['step']
        if (not isinstance(step, torch.Tensor) or step.shape != torch.Size([])
                or step.dtype != torch.float32 or float(step.item()) != 250.):
            raise ValueError('Every parent parameter step counter must equal 250.')
        for moment in ['exp_avg', 'exp_avg_sq']:
            value = state[moment]
            if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                    or value.dtype != parameter.dtype or not bool(torch.isfinite(value).all())):
                raise ValueError('Parent moment shape/dtype/value is invalid for ' + name)
            if moment == 'exp_avg_sq' and bool((value < 0).any()):
                raise ValueError('Squared parent moments must be nonnegative.')
    model.load_state_dict(parent['model'], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=expected_lr, weight_decay=.0001)
    # Optimizer loading may keep CPU tensors by reference. A deep copy ensures
    # the continuation cannot later mutate the in-memory parent checkpoint.
    optimizer.load_state_dict(copy.deepcopy(parent['optimizer']))
    restored_optimizer_digest = recursive_digest(optimizer.state_dict())
    restored_model_digest = recursive_digest(model.state_dict())
    if (restored_optimizer_digest != original_optimizer_digest
            or restored_model_digest != original_model_digest):
        raise ValueError('Restoration changed parent numerical state or optimizer options.')
    if (recursive_digest(parent['optimizer']) != original_optimizer_digest
            or recursive_digest(parent['model']) != original_model_digest):
        raise ValueError('Restoration mutated the parent checkpoint.')
    counters = [int(state['step'].item()) for state in optimizer.state.values()]
    proof = {
        'parent_step': 250, 'new_optimizer_updates': 0,
        'held_learning_rate': expected_lr,
        'model_parameters': sum(parameter.numel() for _, parameter in parameters),
        'parameter_names': [name for name, _ in parameters],
        'parameter_states': len(counters),
        'actual_optimizer_step_counters': {'minimum': min(counters), 'maximum': max(counters),
                                           'parameter_states': len(counters)},
        'parent_model_sha256': original_model_digest, 'restored_model_sha256': restored_model_digest,
        'parent_optimizer_sha256': original_optimizer_digest,
        'restored_optimizer_sha256': restored_optimizer_digest,
        'parent_metadata_sha256': metadata_digest,
        'model_weights_identical': True, 'optimizer_state_identical': True,
        'parent_checkpoint_unmutated': True,
        'optimizer_options': {key: list(value) if isinstance(value, tuple) else value
                              for key, value in parent['optimizer']['param_groups'][0].items()
                              if key != 'params'},
        'historical_rng_restored': False, 'historical_amp_scaler_restored': False,
        'seed_scope': 'Caller restarts a common seed; historical RNG/scaler states were not saved.',
        'device': str(device),
    }
    return model, optimizer, proof
