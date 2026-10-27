"""One fresh 250-update normalization treatment, matched to the frozen control."""

import copy
import time

import numpy as np
import torch

from acoustic10.data import batch_from_schedule, prepare_targets
from acoustic10.training import optimize, tensor_digest
from baseline.download import ROOT, sha256
from cnn_gru.training import save_checkpoint
from spectral11.training import actual_optimizer_steps, verified_sanity
from streaming.data import fingerprint
from streaming.evaluate import save_json
from .model import CausalCNNGRU


def producer_hashes():
    paths = [ROOT / 'configs/normalization12.json'] + [ROOT / 'normalization12' / name for name in
             ['__init__.py', 'data.py', 'training.py', 'workflow.py', 'model.py']]
    paths += [ROOT / 'spectral11' / name for name in ['data.py', 'training.py', 'workflow.py']]
    paths += list((ROOT / 'streaming').glob('*.py'))
    paths += [ROOT / 'acoustic10' / name for name in ['data.py', 'training.py', 'targets.py', 'decoder.py', 'workflow.py']]
    paths += [ROOT / 'cnn_gru' / name for name in ['data.py', 'model.py', 'training.py', 'inference.py']]
    paths += [ROOT / 'baseline/evaluation.py', ROOT / 'diagnosis9/head_audit.py']
    return {str(path.relative_to(ROOT)).replace('\\', '/'): sha256(path) for path in sorted(set(paths))}


def model_configuration(base, config, groups):
    value = copy.deepcopy(base)
    value['model']['cnn_norm_groups'] = groups
    value['training'].update(config['training'])
    value['stage_budget'] = {'total_max_new_optimizer_updates': 500, 'actual_new_arm_updates': 250,
                             'historical_control_updates': 250, 'new_sanity_updates': 0,
                             'purpose': 'Stage12 frame-local normalization.'}
    return value


def train_treatment(config, base, audit, records, provenance, schedule, output, device):
    directory = output / 'models/group1'
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError('Treatment training already exists; do not repeat its update budget.')
    torch.manual_seed(config['seed'])
    spec = config['training']
    model_config = model_configuration(base, config, 1)
    model = CausalCNNGRU(model_config).to(device)
    initial_path = ROOT / config['control_initial_weights']
    initial = torch.load(initial_path, map_location='cpu', weights_only=True)
    model.load_state_dict(initial['model'])
    initial_digest = tensor_digest(model.state_dict())
    control_receipt = provenance['control_reference']['historical_receipt']
    if initial_digest != control_receipt['initial_weights_sha256']:
        raise ValueError('Treatment initialization differs from the historical control.')
    targets = prepare_targets(records, 13, config, base)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=250, eta_min=spec['learning_rate'] * .1)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024, enabled=device.type == 'cuda' and spec['mixed_precision'])
    history, losses_in_epoch = [], []
    started = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, 251):
        descriptors = schedule[(step - 1) * 4:step * 4]
        x, target = batch_from_schedule(descriptors, records, targets, audit['normalization'], spec['edge_mask_frames'])
        x = x.to(device)
        target = {key: value.to(device) for key, value in target.items()}
        losses, gradient = optimize(model, x, target, optimizer, scaler, model_config,
                                    audit['positive_weights'], spec, device)
        counters = actual_optimizer_steps(optimizer)
        if counters['minimum'] != step or counters['maximum'] != step:
            raise ValueError('An optimizer update was skipped; stop without replacement updates.')
        scheduler.step()
        losses_in_epoch.append(losses)
        if step % 50 == 0:
            history.append({'epoch': step // 50, 'optimizer_steps': step, 'actual_optimizer_step_counters': counters,
                            'training_losses': {key: float(np.mean([row[key] for row in losses_in_epoch])) for key in losses},
                            'gradient_norm': gradient})
            losses_in_epoch = []
            receipt = {'name': 'group1', 'cnn_norm_groups': 1, 'optimizer_steps': step,
                       'new_optimizer_steps': step, 'control_reused': False, 'selected_step': 250, 'history': history,
                       'actual_optimizer_step_counters': counters,
                       'optimizer_parameter_steps': {'minimum': counters['minimum'], 'maximum': counters['maximum']},
                       'warmstarted': False, 'target_delay_frames': 13, 'common_leading_mask_frames': 26,
                       'delay_frames': 13,
                       'crop_edge_mask_frames': 26, 'initial_weights_sha256': initial_digest,
                       'initial_checkpoint_sha256': sha256(initial_path),
                       'crop_schedule_sha256': fingerprint(schedule),
                       'gain_schedule_sha256': fingerprint([row['gain_log'] for row in schedule]),
                       'common_support_sha256': control_receipt['common_support_sha256'],
                       'config_sha256': fingerprint(config), 'source_sha256': producer_hashes(),
                       'config': config, 'model_config': model_config, 'normalization': audit['normalization'],
                       'positive_weights': audit['positive_weights'], 'early_stopping_used': False,
                       'validation_used_during_training': False, 'device': str(device),
                       'model_parameters': sum(p.numel() for p in model.parameters()),
                       'training_seconds': time.perf_counter() - started,
                       'peak_gpu_mib': torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None,
                       **provenance}
            checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
                          'receipt': receipt, 'config': model_config, 'normalization': audit['normalization'],
                          'positive_weights': audit['positive_weights']}
            save_checkpoint(directory / 'last.pt', checkpoint)
            save_json(directory / 'training-receipt.json', receipt)
            print('Normalization group1 epoch', step // 50, 'actual update', step,
                  'state loss', round(history[-1]['training_losses']['state'], 3), flush=True)
    save_checkpoint(directory / 'final.pt', checkpoint)
    return receipt
