"""Matched extra optimizer exposure with restored AdamW state and fixed end LR."""

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
from .optimizer import recursive_digest, restore_parent


def producer_hashes():
    paths = [ROOT / 'configs/optimization13.json'] + [ROOT / 'optimization13' / name for name in
             ['__init__.py', 'data.py', 'optimizer.py', 'training.py', 'workflow.py']]
    paths += [ROOT / 'normalization12' / name for name in ['model.py', 'data.py', 'training.py', 'workflow.py']]
    paths += [ROOT / 'spectral11' / name for name in ['data.py', 'training.py', 'workflow.py']]
    paths += list((ROOT / 'streaming').glob('*.py'))
    paths += [ROOT / 'acoustic10' / name for name in ['data.py', 'training.py', 'targets.py', 'decoder.py', 'workflow.py']]
    paths += [ROOT / 'cnn_gru' / name for name in ['data.py', 'model.py', 'training.py', 'inference.py']]
    paths += [ROOT / 'baseline/evaluation.py', ROOT / 'diagnosis9/head_audit.py']
    return {str(path.relative_to(ROOT)).replace('\\', '/'): sha256(path) for path in sorted(set(paths))}


def train_continuation(arm, config, base, audit, records, provenance, parent, schedule, output, device):
    name = arm['name']
    directory = output / 'models' / name
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'resume-state.pt').exists() or (directory / 'last.pt').exists():
        raise ValueError('The continuation already started; do not repeat its new-update budget.')
    torch.manual_seed(config['torch_continuation_seed'])
    spec = config['training']
    model, optimizer, proof = restore_parent(parent, device, spec['learning_rate'])
    resume_path = directory / 'resume-state.pt'
    save_checkpoint(resume_path, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': 250,
                    'config': parent['config'], 'normalization': parent['normalization'],
                    'positive_weights': parent['positive_weights'], 'restoration_proof': proof})
    support = fingerprint([{'id': row['entry']['id'], 'frames': len(row['features']),
                            'ignored_global_leading_frames': 26, 'ignored_crop_edge_frames': 26,
                            'target_delay_frames': 13} for row in records])
    if support != parent['receipt']['common_support_sha256']:
        raise ValueError('Continuation target support differs from the parent.')
    model_config = copy.deepcopy(parent['config'])
    model_config['training'].update(spec)
    model_config['stage_budget'] = {'max_new_optimizer_updates': 500, 'new_per_arm': 250,
                                   'parent_updates': 250, 'final_lifetime_updates': 500,
                                   'new_sanity_updates': 0, 'purpose': 'Stage13 continuation-dose trial.'}
    targets = prepare_targets(records, 13, config, base)
    scaler = torch.amp.GradScaler('cuda', init_scale=config['amp_scaler_reset']['initial_scale'],
                                 enabled=device.type == 'cuda' and spec['mixed_precision'])
    history, losses_in_epoch = [], []
    started = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for new_step in range(1, 251):
        x, target = batch_from_schedule(schedule[(new_step - 1) * 4:new_step * 4], records, targets,
                                        audit['normalization'], spec['edge_mask_frames'])
        x = x.to(device)
        target = {key: value.to(device) for key, value in target.items()}
        losses, gradient = optimize(model, x, target, optimizer, scaler, model_config,
                                    audit['positive_weights'], spec, device)
        lifetime = 250 + new_step
        counters = actual_optimizer_steps(optimizer)
        if counters['minimum'] != lifetime or counters['maximum'] != lifetime:
            raise ValueError('Actual optimizer counters differ from lifetime steps; no replacement updates allowed.')
        if any(group['lr'] != .0001 for group in optimizer.param_groups):
            raise ValueError('The fixed stored-end learning rate changed.')
        losses_in_epoch.append(losses)
        if new_step % 50 == 0:
            history.append({'additional_epoch': new_step // 50, 'data_epoch': 5 + new_step // 50,
                            'optimizer_steps': lifetime, 'new_optimizer_steps': new_step,
                            'actual_optimizer_step_counters': counters,
                            'training_losses': {key: float(np.mean([row[key] for row in losses_in_epoch])) for key in losses},
                            'gradient_norm': gradient, 'learning_rate': optimizer.param_groups[0]['lr']})
            losses_in_epoch = []
            reference = provenance['parents'][name]
            receipt = {'name': name, 'cnn_norm_groups': arm['cnn_norm_groups'],
                       'parent_optimizer_steps': 250, 'optimizer_steps': lifetime, 'new_optimizer_steps': new_step,
                       'selected_step': 500, 'additional_epochs_completed': len(history), 'history': history,
                       'actual_optimizer_step_counters': counters,
                       'optimizer_parameter_steps': {'minimum': counters['minimum'], 'maximum': counters['maximum']},
                       'parent_checkpoint_path': reference['checkpoint_path'],
                       'parent_checkpoint_sha256': reference['checkpoint_sha256'],
                       'parent_reference': reference, 'resume_state_path': str(resume_path.relative_to(output)).replace('\\', '/'),
                       'resume_state_sha256': sha256(resume_path), 'restoration_proof': proof,
                       'restored_from_parent': True, 'historical_rng_restored': False,
                       'historical_scaler_restored': False, 'torch_continuation_seed': config['torch_continuation_seed'],
                       'scaler_initial_scale': 1024, 'learning_rate_rule': 'constant stored-end learning rate',
                       'learning_rate': .0001, 'scheduler_used': False,
                       'target_delay_frames': 13, 'delay_frames': 13, 'common_leading_mask_frames': 26,
                       'crop_edge_mask_frames': 26, 'crop_schedule_sha256': fingerprint(schedule),
                       'gain_schedule_sha256': fingerprint([row['gain_log'] for row in schedule]),
                       'common_support_sha256': support,
                       'original_fresh_initial_weights_sha256': parent['receipt']['initial_weights_sha256'],
                       'restored_model_sha256': proof['restored_model_sha256'],
                       'restored_optimizer_sha256': proof['restored_optimizer_sha256'],
                       'config_sha256': fingerprint(config), 'source_sha256': producer_hashes(),
                       'config': config, 'model_config': model_config, 'normalization': audit['normalization'],
                       'positive_weights': audit['positive_weights'], 'early_stopping_used': False,
                       'validation_used_during_training': False, 'device': str(device),
                       'model_parameters': sum(p.numel() for p in model.parameters()),
                       'training_seconds': time.perf_counter() - started,
                       'peak_gpu_mib': torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None,
                       **provenance}
            checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': lifetime,
                          'receipt': receipt, 'config': model_config, 'normalization': audit['normalization'],
                          'positive_weights': audit['positive_weights'], 'scaler': scaler.state_dict(),
                          'rng_cpu': torch.get_rng_state(),
                          'rng_cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}
            save_checkpoint(directory / 'last.pt', checkpoint)
            save_json(directory / 'training-receipt.json', receipt)
            print('Continuation', name, 'new update', new_step, 'lifetime', lifetime,
                  'state loss', round(history[-1]['training_losses']['state'], 3), flush=True)
    save_checkpoint(directory / 'final.pt', checkpoint)
    if (recursive_digest(parent['model']) != proof['parent_model_sha256']
            or recursive_digest(parent['optimizer']) != proof['parent_optimizer_sha256']):
        raise ValueError('Continuation mutated the in-memory historical parent.')
    return receipt
