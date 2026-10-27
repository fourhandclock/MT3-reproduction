"""Fixed-budget acoustic optimization, separated from validation selection."""

import copy
import hashlib
import json
import time

import numpy as np
import torch

from baseline.download import ROOT, sha256
from cnn_gru.model import transcription_loss
from cnn_gru.training import save_checkpoint
from diagnosis9.head_audit import exact_state_metric
from streaming.data import fingerprint
from streaming.evaluate import save_json
from streaming.model import CausalCNNGRU
from .data import batch_from_schedule, prepare_targets


def tensor_digest(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        value = value.detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode()
        payload = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        for part in (header, payload):
            digest.update(len(part).to_bytes(8, 'big'))
            digest.update(part)
    return digest.hexdigest()


def producer_hashes():
    paths = [ROOT / 'configs/acoustic10.json'] + [ROOT / 'acoustic10' / name for name in
            ['__init__.py', 'data.py', 'targets.py', 'decoder.py', 'training.py', 'workflow.py']]
    paths += list((ROOT / 'streaming').glob('*.py'))
    paths += [ROOT / 'cnn_gru/model.py', ROOT / 'baseline/evaluation.py', ROOT / 'diagnosis9/head_audit.py']
    return {str(path.relative_to(ROOT)).replace('\\', '/'): sha256(path) for path in sorted(paths)}


def head_metrics(probability, target, mask, inactive):
    valid = (np.asarray(mask) > 0)[:, None] & (target != -100)
    observed = probability.argmax(axis=-1)
    filtered = np.where(valid, target, -100)
    metric = exact_state_metric(filtered, observed, inactive)
    active = valid & (target != inactive)
    active_count = int(active.sum())
    accuracy = float((active & (target == observed)).sum() / active_count) if active_count else 0.
    safe = np.maximum(target, 0)
    likelihood = np.take_along_axis(probability, safe[..., None], axis=-1)[..., 0]
    weight = np.where(target == inactive, .35, 1.) * valid
    loss = float((-np.log(np.maximum(likelihood, 1e-12)) * weight).sum() / max(weight.sum(), 1))
    return {'exact_state': metric, 'exact_state_f1': metric['f1'], 'active_fret_accuracy': accuracy,
            'active_frames': active_count, 'state_loss': loss,
            'distinct_frets': int(len(np.unique(target[active]))) if active_count else 0}


def evaluate_batch(model, x, target, model_config, weights):
    model.eval()
    with torch.inference_mode():
        logits = model(x)
        losses = transcription_loss(logits, target, model_config, weights)
        probability = torch.softmax(logits['state'].float(), -1).cpu().numpy()
    metrics = []
    for i in range(len(x)):
        metrics.append(head_metrics(probability[i], target['state'][i].cpu().numpy(),
                                    target['mask'][i].cpu().numpy(), model_config['model']['max_fret'] + 1))
    state = target['state'].cpu().numpy()
    mask = target['mask'].cpu().numpy()
    pooled = head_metrics(probability.reshape(-1, 6, probability.shape[-1]), state.reshape(-1, 6),
                          mask.reshape(-1), model_config['model']['max_fret'] + 1)
    return {**pooled, 'losses': {key: float(value) for key, value in losses.items()}, 'crops': metrics}


def optimize(model, x, targets, optimizer, scaler, model_config, weights, spec, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=device.type == 'cuda' and spec['mixed_precision']):
        losses = transcription_loss(model(x), targets, model_config, weights)
    if not torch.isfinite(losses['total']):
        raise FloatingPointError('Nonfinite acoustic loss; no further update is allowed.')
    scaler.scale(losses['total']).backward()
    scaler.unscale_(optimizer)
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), spec['gradient_clip'])
    if not torch.isfinite(gradient):
        raise FloatingPointError('Nonfinite acoustic gradient; no further update is allowed.')
    scaler.step(optimizer)
    scaler.update()
    return {key: float(value.detach()) for key, value in losses.items()}, float(gradient)


def model_configuration(base, config, sanity=False):
    value = copy.deepcopy(base)
    value['training'].update(config['training'])
    value['stage_budget'] = {'total_max_optimizer_updates': 500, 'sanity_updates': 100,
                             'arm_updates': 200, 'purpose': 'Stage 10 separate matched acoustic trial.'}
    if sanity:
        value['model']['dropout'] = config['sanity']['dropout']
    return value


def run_sanity(config, base, audit, datasets, provenance, output, device):
    path = output / 'sanity.json'
    if path.exists() or (output / 'sanity-checkpoint.pt').exists():
        raise ValueError('Sanity training already exists; do not repeat its update budget.')
    torch.manual_seed(config['seed'] + 1)
    spec = config['training']
    model_config = model_configuration(base, config, sanity=True)
    records = datasets['train']
    fps = base['feature']['sample_rate'] / base['feature']['hop_length']
    descriptors = [{'epoch': 0, 'sample_index': 0, 'record_index': index,
                    'id': records[index]['entry']['id'], 'start_frame': round(seconds * fps),
                    'frames': round(config['sanity']['crop_seconds'] * fps), 'gain_log': 0.}
                   for index in config['sanity']['train_entry_indices']
                   for seconds in config['sanity']['crop_starts_seconds']]
    x, target = batch_from_schedule(descriptors, records, prepare_targets(records, 0, config, base),
                                    audit['normalization'], spec['edge_mask_frames'])
    x = x.to(device)
    target = {key: value.to(device) for key, value in target.items()}
    model = CausalCNNGRU(model_config).to(device)
    initial_digest = tensor_digest(model.state_dict())
    initial = evaluate_batch(model, x, target, model_config, audit['positive_weights'])
    if (initial['active_frames'] < config['sanity']['minimum_active_frames']
            or initial['distinct_frets'] < config['sanity']['minimum_distinct_frets']):
        raise ValueError('Prespecified sanity batch has insufficient labelled active/fret coverage.')
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
    # Float32 sanity optimizes the same loss without stochastic dropout/augmentation/AMP.
    sanity_spec = {**spec, 'mixed_precision': False}
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    history = []
    started = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    steps = config['sanity']['optimizer_updates']
    for step in range(1, steps + 1):
        losses, gradient = optimize(model, x, target, optimizer, scaler, model_config,
                                    audit['positive_weights'], sanity_spec, device)
        if step % (steps // config['sanity']['blocks']) == 0:
            metrics = evaluate_batch(model, x, target, model_config, audit['positive_weights'])
            history.append({'optimizer_steps': step, 'metrics': metrics, 'gradient_norm': gradient})
            save_checkpoint(output / 'sanity-checkpoint.pt', {'model': model.state_dict(), 'step': step,
                            'config': model_config, 'source_sha256': producer_hashes(),
                            'initial_weights_sha256': initial_digest, 'descriptors': descriptors})
            print('Sanity update', step, 'active fret accuracy', round(metrics['active_fret_accuracy'], 3),
                  'exact-state F1', round(metrics['exact_state_f1'], 3), flush=True)
    final = history[-1]['metrics']
    ratio = final['state_loss'] / initial['state_loss']
    rule = config['sanity']
    passed = (ratio <= rule['maximum_state_loss_ratio']
              and final['active_fret_accuracy'] >= rule['minimum_active_fret_accuracy']
              and final['exact_state_f1'] >= rule['minimum_exact_state_f1'])
    result = {'passed': passed, 'optimizer_steps': steps, 'blocks_completed': len(history),
              'initial': initial, 'final': final, 'state_loss_ratio': ratio,
              'coverage': {'active_frames': initial['active_frames'], 'distinct_frets': initial['distinct_frets']},
              'criteria': rule, 'history': history, 'descriptors': descriptors,
              'initial_weights_sha256': initial_digest, 'config': config,
              'config_sha256': fingerprint(config), 'source_sha256': producer_hashes(), **provenance,
              'device': str(device), 'training_seconds': time.perf_counter() - started,
              'peak_gpu_mib': torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None,
              'scope': 'One fixed four-crop TRAINING batch repeated 100 updates in five reporting blocks. '
                       'No validation/test access or generalization evidence. Float32, dropout zero and gain zero.'}
    save_json(path, result)
    return result


def train_arm(arm, config, base, audit, datasets, provenance, schedule, initial_path, output, device):
    directory = output / 'models' / arm['name']
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError('Arm output already exists; its 200 updates must not be repeated.')
    torch.manual_seed(config['seed'])
    spec = config['training']
    model_config = model_configuration(base, config)
    model = CausalCNNGRU(model_config).to(device)
    state = torch.load(initial_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'])
    initial_digest = tensor_digest(model.state_dict())
    targets = prepare_targets(datasets['train'], arm['target_delay_frames'], config, base)
    support_digest = fingerprint([{'id': record['entry']['id'], 'frames': len(record['features']),
                                  'ignored_global_leading_frames': config['common_leading_mask_frames'],
                                  'ignored_crop_edge_frames': spec['edge_mask_frames']}
                                 for record in datasets['train']])
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=spec['max_steps'],
                                                         eta_min=spec['learning_rate'] * .1)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024, enabled=device.type == 'cuda' and spec['mixed_precision'])
    history, losses_in_epoch = [], []
    started = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, spec['max_steps'] + 1):
        descriptors = schedule[(step - 1) * spec['batch_size']:step * spec['batch_size']]
        x, target = batch_from_schedule(descriptors, datasets['train'], targets, audit['normalization'], spec['edge_mask_frames'])
        x = x.to(device)
        target = {key: value.to(device) for key, value in target.items()}
        losses, gradient = optimize(model, x, target, optimizer, scaler, model_config,
                                    audit['positive_weights'], spec, device)
        scheduler.step()
        losses_in_epoch.append(losses)
        if step % 40 == 0:
            history.append({'epoch': step // 40, 'optimizer_steps': step,
                            'training_losses': {key: float(np.mean([row[key] for row in losses_in_epoch])) for key in losses},
                            'gradient_norm': gradient})
            losses_in_epoch = []
            receipt = {'name': arm['name'], 'target_delay_frames': arm['target_delay_frames'],
                       'optimizer_steps': step, 'selected_step': spec['max_steps'], 'history': history,
                       'initial_weights_sha256': initial_digest, 'initial_checkpoint_sha256': sha256(initial_path),
                       'crop_schedule_sha256': fingerprint(schedule),
                       'gain_schedule_sha256': fingerprint([row['gain_log'] for row in schedule]),
                       'common_support_sha256': support_digest, 'config_sha256': fingerprint(config),
                       'common_leading_mask_frames': config['common_leading_mask_frames'],
                       'delay_frames': arm['target_delay_frames'], 'warmstarted': False,
                       'source_sha256': producer_hashes(), **provenance, 'config': config,
                       'model_config': model_config, 'normalization': audit['normalization'],
                       'positive_weights': audit['positive_weights'], 'early_stopping_used': False,
                       'validation_used_during_training': False, 'warm_started_from_sanity': False,
                       'device': str(device), 'training_seconds': time.perf_counter() - started,
                       'peak_gpu_mib': torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None}
            save_checkpoint(directory / 'last.pt', {'model': model.state_dict(), 'step': step,
                            'receipt': receipt, 'config': model_config, 'normalization': audit['normalization'],
                            'positive_weights': audit['positive_weights']})
            save_json(directory / 'training-receipt.json', receipt)
            print('Acoustic arm', arm['name'], 'epoch', step // 40, 'update', step,
                  'state loss', round(history[-1]['training_losses']['state'], 3), flush=True)
    save_checkpoint(directory / 'final.pt', {'model': model.state_dict(), 'step': spec['max_steps'],
                    'receipt': receipt, 'config': model_config, 'normalization': audit['normalization'],
                    'positive_weights': audit['positive_weights']})
    return receipt
