import copy
import math
import numpy as np
import pytest
import torch
from acoustic10.training import optimize as legacy_optimize
from baseline.download import ROOT
from decoder14.workflow import read
from fret19.loss import conditional_fret_loss
from multires17.model import CausalCNNGRU
from optimization13.data import build_schedule
from optimization13.optimizer import recursive_digest
from balance22.data import validate_config
from balance22.evaluation import choose, fret_metrics, MODES
from balance22.loss import class_weights, weighted_fret_loss
from balance22.restore import restore_parent
from balance22.training import optimize
from balance22.workflow import run


def labels():
    return {'state': torch.full((1, 3, 6), 25, dtype=torch.long), 'mask': torch.ones(1, 3)}


def test_inverse_sqrt_weights_are_per_string_capped_and_unseen_zero():
    counts = np.zeros((6, 25), np.int64)
    counts[0, :4] = [1600, 400, 100, 1]
    counts[1, :2] = [4, 1]
    weights = class_weights(counts)
    assert weights.dtype == np.float32
    assert weights[0, :4].tolist() == [1., 2., 4., 4.]
    assert weights[1, :2].tolist() == [1., 2.]
    assert np.all(weights[counts == 0] == 0)


def test_training_available_unexposed_class_gets_fixed_cap_without_validation_counts():
    unique = np.zeros((6, 25), np.int64)
    available = unique.copy()
    unique[4, 6], available[4, 6] = 100, 120
    available[4, 17] = 50
    weights = class_weights(unique, available)
    assert weights[4, 6] == 1 and weights[4, 17] == 4
    assert weights[0, 24] == 0
    with pytest.raises(ValueError): class_weights(unique, np.zeros((6, 25), int))


@pytest.mark.parametrize('kind', ['shape', 'float', 'negative'])
def test_bad_training_counts_fail(kind):
    counts = np.zeros((6, 25), int)
    if kind == 'shape': counts = counts[:, :24]
    if kind == 'float': counts = counts.astype(float)
    if kind == 'negative': counts[0, 0] = -1
    with pytest.raises(ValueError): class_weights(counts)


def test_weighted_ce_uses_sum_weights_and_has_no_inactive_or_masked_gradient():
    torch.manual_seed(13)
    logits = torch.randn(1, 3, 6, 26, requires_grad=True)
    target = labels()
    target['state'][0, 0, :3] = torch.tensor([2, 4, -100])
    target['state'][0, 1, 0] = 3
    target['mask'][0, 1] = 0
    weights = torch.ones(6, 25)
    weights[1, 4] = 4
    ce = torch.nn.functional.cross_entropy(torch.stack([logits[0, 0, 0, :25], logits[0, 0, 1, :25]]),
                                           torch.tensor([2, 4]), reduction='none')
    loss = weighted_fret_loss(logits, target, weights)
    torch.testing.assert_close(loss, (ce[0] + 4 * ce[1]) / 5)
    loss.backward()
    assert torch.count_nonzero(logits.grad[..., 25]) == 0
    assert torch.count_nonzero(logits.grad[0, 1:]) == 0
    assert torch.count_nonzero(logits.grad[0, 0, 2:]) == 0


def test_unit_class_weights_match_uniform_active_ce():
    torch.manual_seed(8)
    logits = torch.randn(1, 3, 6, 26)
    target = labels()
    target['state'][0, 0, :3] = torch.tensor([1, 2, 3])
    torch.testing.assert_close(weighted_fret_loss(logits, target, torch.ones(6, 25)), conditional_fret_loss(logits, target))
    changed = logits.clone()
    changed[..., 25] = 10000
    torch.testing.assert_close(weighted_fret_loss(changed, target, torch.ones(6, 25)), conditional_fret_loss(logits, target))


@pytest.mark.parametrize('kind', ['inactive', 'conflict', 'masked'])
def test_empty_active_support_is_differentiable_zero(kind):
    logits = torch.randn(1, 3, 6, 26, requires_grad=True)
    target = labels()
    if kind == 'conflict': target['state'][:] = -100
    if kind == 'masked': target['state'][:] = 3; target['mask'][:] = 0
    loss = weighted_fret_loss(logits, target, torch.ones(6, 25))
    assert loss.item() == 0 and loss.requires_grad
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


@pytest.mark.parametrize('kind', ['shape', 'negative', 'nan', 'cap', 'learnable', 'unseen'])
def test_invalid_or_unseen_active_weights_fail(kind):
    logits = torch.zeros(1, 3, 6, 26)
    target = labels()
    target['state'][0, 0, 0] = 2
    weights = torch.ones(6, 25)
    if kind == 'shape': weights = weights[:, :24]
    if kind == 'negative': weights[0, 2] = -1
    if kind == 'nan': weights[0, 2] = float('nan')
    if kind == 'cap': weights[0, 2] = 4.1
    if kind == 'learnable': weights.requires_grad_(True)
    if kind == 'unseen': weights[0, 2] = 0
    with pytest.raises(ValueError): weighted_fret_loss(logits, target, weights)


@pytest.fixture(scope='module')
def parent():
    torch.manual_seed(710)
    config = {'feature': {'n_bins': 8}, 'feature_planes': [2048, 2048],
        'model': {'channels': [8, 8, 8], 'projection_size': 8, 'gru_hidden_size': 8, 'gru_layers': 2,
                  'bidirectional': False, 'dropout': .1, 'output_size': 8, 'max_fret': 24, 'pitch_min': 40, 'pitch_max': 88},
        'training': {'inactive_class_weight': .35}}
    model = CausalCNNGRU(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.0001)
    for i, p in enumerate(model.parameters()): p.grad = torch.full_like(p, (i + 1) / 100.)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    # Synthetic fixture counters, not additional acoustic updates.
    for value in optimizer.state.values(): value['step'].fill_(500)
    optimizer.param_groups[0]['initial_lr'] = .001
    return {'model': copy.deepcopy(model.state_dict()), 'optimizer': copy.deepcopy(optimizer.state_dict()),
        'step': 500, 'config': config, 'normalization': {'mean': [[0.] * 8] * 2, 'std': [[1.] * 8] * 2},
        'positive_weights': {'onset': 1., 'offset': 1., 'pitch': 1.}, 'scaler': {},
        'rng_cpu': torch.get_rng_state().clone(), 'rng_cuda': [], 'source_scheduler': {'last_epoch': 250}}


def test_step500_full_restoration_keeps_parent_and_paired_rng(parent):
    before = recursive_digest(parent)
    outputs = []
    for _ in range(3):
        model, optimizer, scaler, proof = restore_parent(parent, 'cpu')
        assert proof['parent_step'] == 500 and proof['parent_state_sha256'] == proof['restored_state_sha256']
        assert recursive_digest(optimizer.state_dict()) == recursive_digest(parent['optimizer'])
        outputs.append(model(torch.zeros(1, 10, 2, 8)))
    for key in outputs[0]:
        assert torch.equal(outputs[0][key], outputs[1][key]) and torch.equal(outputs[1][key], outputs[2][key])
    assert recursive_digest(parent) == before


@pytest.mark.parametrize('kind', ['step', 'counter', 'moment', 'plane', 'normalization', 'rng', 'scaler'])
def test_bad_parent_restoration_fails(parent, kind):
    value = copy.deepcopy(parent)
    if kind == 'step': value['step'] = 250
    if kind == 'counter': value['optimizer']['state'][0]['step'].fill_(499)
    if kind == 'moment': del value['optimizer']['state'][0]['exp_avg']
    if kind == 'plane': value['config']['feature_planes'] = [2048, 4096]
    if kind == 'normalization': value['normalization']['std'][0][0] = 0
    if kind == 'rng': value['rng_cpu'] = torch.zeros(3)
    if kind == 'scaler': value['scaler'] = {'scale': 1024.}
    with pytest.raises((ValueError, RuntimeError)): restore_parent(value, 'cpu')


def toy_batch():
    target = {'state': torch.full((1, 10, 6), 25, dtype=torch.long), 'mask': torch.ones(1, 10),
              'onset': torch.zeros(1, 10, 6), 'offset': torch.zeros(1, 10, 6), 'pitch': torch.zeros(1, 10, 49)}
    target['state'][:, :, 0] = 2
    return torch.zeros(1, 10, 2, 8), target


def test_original_arm_is_bit_exact_to_one_legacy_toy_step(parent):
    x, target = toy_batch()
    spec = {'mixed_precision': False, 'gradient_clip': 1.}
    a, oa, sa, _ = restore_parent(parent, 'cpu')
    old, old_grad = legacy_optimize(a, x, target, oa, sa, parent['config'], parent['positive_weights'], spec, torch.device('cpu'))
    b, ob, sb, _ = restore_parent(parent, 'cpu')
    new, new_grad, counts = optimize(b, x, target, ob, sb, parent['config'], parent['positive_weights'], spec, torch.device('cpu'), 0.)
    assert old == new and old_grad == new_grad and int(counts.sum()) == 10
    assert recursive_digest(a.state_dict()) == recursive_digest(b.state_dict())
    assert recursive_digest(oa.state_dict()) == recursive_digest(ob.state_dict())


@pytest.mark.parametrize('weighted', [False, True])
def test_auxiliary_toy_step_has_expected_loss_and_exact_counter(parent, weighted):
    x, target = toy_batch()
    model, optimizer, scaler, _ = restore_parent(parent, 'cpu')
    weights = torch.ones(6, 25) if weighted else None
    values, gradient, counts = optimize(model, x, target, optimizer, scaler, parent['config'], parent['positive_weights'],
        {'mixed_precision': False, 'gradient_clip': 1.}, torch.device('cpu'), .5, weights)
    expected = values['state'] + values['onset'] + .5 * values['offset'] + .5 * values['pitch'] + .5 * values['active_fret_ce']
    assert values['total'] == pytest.approx(expected, abs=1e-6) and int(counts.sum()) == 10 and np.isfinite(gradient)
    assert all(state['step'].item() == 501 for state in optimizer.state.values())


@pytest.mark.parametrize('key,value', [('data_epoch_start', 5), ('total_optimizer_update_cap', 500),
    ('test_inference_allowed', True), ('production_promotion', True), ('decoder_parameters', {}), ('arms', [])])
def test_fixed_protocol_rejects_mutations(key, value):
    config = read(ROOT / 'configs/balance22.json')
    validate_config(config)
    config[key] = value
    with pytest.raises(ValueError): validate_config(config)


def test_schedule_uses_exactly_epochs11_to13_and_selection_ties_are_fixed():
    config = read(ROOT / 'configs/balance22.json')
    records = [{'entry': {'id': str(i)}, 'features': np.zeros((2400, 2))} for i in range(3)]
    schedule = build_schedule(config, records)
    assert len(schedule) == 600 and {row['epoch'] for row in schedule} == {11, 12, 13}
    assert schedule == build_schedule(config, records)
    scores = {mode: {'exact_state_f1': .1, 'onset_f1': .2} for mode in MODES[1:]}
    assert choose(scores) == 'original650'
    scores['uniform_ce650']['onset_f1'] = .21
    assert choose(scores) == 'uniform_ce650'
    scores['balanced_ce650']['exact_state_f1'] = .11
    assert choose(scores) == 'balanced_ce650'
    scores['parent500'] = scores['original650']
    with pytest.raises(ValueError): choose(scores)


def test_equal_position_fret_accuracy_uses_pooled_supported_labels():
    matrix = np.zeros((6, 25, 25), int)
    matrix[0, 0, 0] = 90
    matrix[0, 1, 0] = 10
    value = fret_metrics(matrix)
    assert value['accuracy'] == .9 and value['equal_supported_position_accuracy'] == .5
    assert value['supported_positions'] == 2


def test_started_run_cannot_repeat_acoustic_training(tmp_path):
    (tmp_path / 'protocol.json').write_text('{}')
    with pytest.raises(ValueError, match='already started'):
        run(read(ROOT / 'configs/balance22.json'), tmp_path, torch.device('cpu'))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_real_control500_amp_restoration_and_weighted_gradient_without_optimizer_update():
    parent = torch.load(ROOT / 'outputs/fret19/models/control/final.pt', map_location='cpu', weights_only=True)
    model, optimizer, scaler, proof = restore_parent(parent, 'cuda')
    assert proof['parent_step'] == 500 and proof['model_parameters'] == 931361
    assert scaler.state_dict()['_growth_tracker'] == 500
    x = torch.linspace(-1, 1, 40 * 2 * 216, device='cuda').reshape(1, 40, 2, 216)
    with torch.autocast('cuda', dtype=torch.float16): logits = model(x)['state']
    target = {'state': torch.full((1, 40, 6), 3, device='cuda', dtype=torch.long), 'mask': torch.ones(1, 40, device='cuda')}
    loss = weighted_fret_loss(logits, target, torch.ones(6, 25, device='cuda'))
    assert loss.dtype == torch.float32
    scaler.scale(loss).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(state['step'].item() == 500 for state in optimizer.state.values())
