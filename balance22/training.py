"""Three full-state continuations; auxiliary fret weighting is the declared factor."""

import copy
import json
import time
import numpy as np
import torch
from acoustic10.data import prepare_targets
from baseline.download import ROOT, sha256
from cnn_gru.model import transcription_loss
from cnn_gru.training import save_checkpoint
from fret19.loss import active_support, conditional_fret_loss
from multires17.data import batch_from_schedule
from optimization13.optimizer import recursive_digest
from spectral11.training import actual_optimizer_steps, verified_sanity
from streaming.data import fingerprint
from streaming.evaluate import save_json
from .loss import weighted_fret_loss
from .restore import restore_parent


def producer_hashes():
    names = ['__init__.py', '__main__.py', 'data.py', 'loss.py', 'restore.py', 'training.py', 'workflow.py']
    paths = [ROOT / 'balance22' / name for name in names] + [ROOT / 'configs/balance22.json']
    # Frozen predecessors already bind all borrowed numerical producers.
    prior = json.loads((ROOT / 'outputs/prior21/verification.json').read_bytes())
    names = set(prior['code_sha256']) | {n for n in prior['input_sha256'] if n.endswith('.py')}
    paths += [ROOT / name for name in names]
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in sorted(set(paths))}


def optimize(model, x, target, optimizer, scaler, model_config, positive_weights, spec, device, coefficient, weights=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda' and spec['mixed_precision']):
        outputs = model(x)
        losses = transcription_loss(outputs, target, model_config, positive_weights)
    valid = active_support(outputs['state'], target)
    classes = torch.arange(6, device=device).expand_as(target['state']) * 25 + target['state']
    counts = torch.bincount(classes[valid], minlength=150).reshape(6, 25).cpu().numpy()
    if coefficient:
        extra = conditional_fret_loss(outputs['state'], target) if weights is None else weighted_fret_loss(outputs['state'], target, weights)
        losses = {**losses, 'active_fret_ce': extra, 'total': losses['total'] + coefficient * extra}
    if not torch.isfinite(losses['total']):
        raise FloatingPointError('Nonfinite loss; no replacement acoustic updates')
    scaler.scale(losses['total']).backward()
    scaler.unscale_(optimizer)
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), spec['gradient_clip'])
    if not torch.isfinite(gradient):
        raise FloatingPointError('Nonfinite gradient; no replacement acoustic updates')
    scaler.step(optimizer)
    scaler.update()
    return {key: float(value.detach()) for key, value in losses.items()}, float(gradient), counts


def train_arm(arm, config, base, audit, records, provenance, parent, schedule, weights, output, device):
    directory = output / 'models' / arm['name']
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError('This Stage22 arm already started; do not repeat its training budget')
    model, optimizer, scaler, proof = restore_parent(parent, device)
    snapshot = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
        'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
        'step': 500, 'config': parent['config'], 'normalization': parent['normalization'],
        'positive_weights': parent['positive_weights'], 'source_scheduler': copy.deepcopy(parent['source_scheduler']),
        'restoration_proof': proof}
    save_checkpoint(directory / 'resume-state.pt', snapshot)
    spec = config['training']
    model_config = copy.deepcopy(parent['config'])
    model_config['training'].update(spec)
    model_config['stage_budget'] = {'total_max_new_optimizer_updates': 450, 'new_arm_updates': 150,
        'shared_parent_updates': 500, 'final_lifetime_updates': 650, 'new_sanity_updates': 0, 'purpose': 'Stage22 matched class weighting'}
    target_records = prepare_targets(records, 13, config, base)
    tensor_weights = torch.as_tensor(weights, device=device) if arm['weighting'] == 'clipped_inverse_sqrt' else None
    history, loss_rows = [], []
    class_counts = np.zeros((6, 25), np.int64)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for new_step in range(1, 151):
        x, target = batch_from_schedule(schedule[(new_step - 1) * 4:new_step * 4], records, target_records,
            audit['normalization'], 26, {'name': 'short_short'})
        losses, gradient, counts = optimize(model, x.to(device), {k: v.to(device) for k, v in target.items()},
            optimizer, scaler, model_config, audit['positive_weights'], spec, device, arm['active_fret_ce_coefficient'], tensor_weights)
        counters = actual_optimizer_steps(optimizer)
        step = 500 + new_step
        if counters != {'minimum': step, 'maximum': step, 'parameter_states': 34}:
            raise ValueError('Actual optimizer counters differ; stop without replacement updates')
        loss_rows.append(losses)
        class_counts += counts
        if new_step % 50 == 0:
            history.append({'data_epoch': 10 + new_step // 50, 'new_optimizer_steps': new_step, 'optimizer_steps': step,
                'actual_optimizer_step_counters': counters, 'learning_rate': optimizer.param_groups[0]['lr'],
                'training_losses': {k: float(np.mean([r[k] for r in loss_rows])) for k in losses},
                'valid_active_string_frames': int(class_counts.sum()), 'active_class_presentations': class_counts.tolist(),
                'class_weighted_target_mass': float((class_counts * weights).sum()) if tensor_weights is not None else float(class_counts.sum()),
                'gradient_norm': gradient})
            loss_rows = []
            class_counts = np.zeros((6, 25), np.int64)
            print('Stage22', arm['name'], 'epoch', history[-1]['data_epoch'], 'new update', new_step,
                  'state loss', round(history[-1]['training_losses']['state'], 4), flush=True)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    receipt = {'name': arm['name'], 'new_optimizer_steps': 150, 'parent_optimizer_steps': 500, 'optimizer_steps': 650,
        'selected_step': 650, 'history': history, 'actual_optimizer_step_counters': counters,
        'active_fret_ce_coefficient': arm['active_fret_ce_coefficient'], 'weighting': arm['weighting'],
        'class_weights_sha256': sha256(output / 'class-weights.json'), 'restoration_proof': proof,
        'resume_state_sha256': sha256(directory / 'resume-state.pt'), 'source_sha256': producer_hashes(),
        'crop_schedule_sha256': fingerprint(schedule), 'gain_schedule_sha256': fingerprint([r['gain_log'] for r in schedule]),
        'config_sha256': fingerprint(config), 'config': config, 'model_config': model_config,
        'normalization': audit['normalization'], 'positive_weights': audit['positive_weights'],
        'model_parameters': sum(p.numel() for p in model.parameters()), 'scheduler_used': False,
        'early_stopping_used': False, 'validation_used_during_training': False, 'test_inference_calls': 0,
        'device': str(device), 'training_seconds': time.perf_counter() - started,
        'peak_gpu_mib': torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None, **provenance}
    checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
        'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
        'step': 650, 'config': model_config, 'normalization': audit['normalization'], 'positive_weights': audit['positive_weights'],
        'source_scheduler': copy.deepcopy(parent['source_scheduler']), 'scheduler_used': False, 'receipt': receipt}
    save_checkpoint(directory / 'final.pt', checkpoint)
    save_json(directory / 'training-receipt.json', receipt)
    return receipt
