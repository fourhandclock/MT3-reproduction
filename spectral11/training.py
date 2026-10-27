"""Fresh matched 250-update models; only window features/statistics vary."""

import copy
import json
import time

import numpy as np
import torch

from acoustic10.data import batch_from_schedule, prepare_targets
from acoustic10.training import optimize, tensor_digest
from baseline.download import ROOT, sha256
from cnn_gru.training import save_checkpoint
from streaming.data import fingerprint
from streaming.evaluate import save_json
from streaming.model import CausalCNNGRU


def producer_hashes():
    paths = [ROOT / 'configs/spectral11.json'] + [ROOT / 'spectral11' / name for name in
             ['__init__.py', 'data.py', 'training.py', 'workflow.py']]
    paths += list((ROOT / 'streaming').glob('*.py'))
    paths += [ROOT / 'acoustic10' / name for name in ['data.py', 'training.py', 'targets.py', 'decoder.py', 'workflow.py']]
    paths += [ROOT / 'cnn_gru' / name for name in ['data.py', 'model.py', 'training.py', 'inference.py']]
    paths += [ROOT / 'baseline/evaluation.py', ROOT / 'diagnosis9/head_audit.py']
    return {str(path.relative_to(ROOT)).replace('\\', '/'): sha256(path) for path in sorted(set(paths))}


def model_configuration(base, config):
    value = copy.deepcopy(base)
    value['training'].update(config['training'])
    value['stage_budget'] = {'total_max_optimizer_updates': 500, 'arm_updates': 250,
                             'new_sanity_updates': 0, 'purpose': 'Stage11 window comparison.'}
    return value


def verified_sanity(config):
    path = ROOT / config['sanity_reference']
    sanity = json.loads(path.read_text(encoding='utf-8'))
    verification = json.loads((ROOT / 'outputs/acoustic10/verification.json').read_text(encoding='utf-8'))
    # The earlier stage binds its source/config and sanity artifact in the evaluation run.
    run = json.loads((ROOT / 'outputs/acoustic10/validation/run.json').read_text(encoding='utf-8'))
    if run['sanity'] != sanity or not sanity['passed'] or sanity['optimizer_steps'] != 100:
        raise ValueError('The recorded training-batch sanity evidence changed or failed.')
    initial, final, coverage = sanity['initial'], sanity['final'], sanity['coverage']
    if (final['state_loss'] / initial['state_loss'] > .5 or final['active_fret_accuracy'] < .8
            or final['exact_state_f1'] < .75 or coverage['active_frames'] < 100 or coverage['distinct_frets'] < 4):
        raise ValueError('Prior sanity measurements fail the prespecified gate.')
    for name, digest in sanity['source_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError('A source underlying the earlier sanity evidence changed.')
    return {'path': str(path.relative_to(ROOT)).replace('\\', '/'), 'sha256': sha256(path),
            'checkpoint_sha256': sha256(ROOT / 'outputs/acoustic10/sanity-checkpoint.pt'),
            'passed': True, 'prior_optimizer_updates': 100, 'new_optimizer_updates': 0,
            'prior_verification_sha256': sha256(ROOT / 'outputs/acoustic10/verification.json'),
            'prior_stage_total_updates': verification['total_new_optimizer_updates'],
            'final_exact_state_f1': final['exact_state_f1'], 'final_active_fret_accuracy': final['active_fret_accuracy'],
            'scope': 'Verified previous four-crop training-only check; no new gradients or warmstart.'}


def actual_optimizer_steps(optimizer):
    values = [int(state['step'].item()) for state in optimizer.state.values() if 'step' in state]
    if not values:
        return {'minimum': 0, 'maximum': 0, 'parameter_states': 0}
    return {'minimum': min(values), 'maximum': max(values), 'parameter_states': len(values)}


def train_arm(arm, config, base, audit, records, provenance, schedule, initial_path, output, device):
    name = arm['name']
    directory = output / 'models' / name
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError('The matched arm already exists; its update budget cannot be repeated.')
    torch.manual_seed(config['seed'])
    spec = config['training']
    model_config = model_configuration(base, config)
    model = CausalCNNGRU(model_config).to(device)
    state = torch.load(initial_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'])
    initial_digest = tensor_digest(model.state_dict())
    targets = prepare_targets(records, config['common_target_delay_frames'], config, base)
    support_digest = fingerprint([{'id': record['entry']['id'], 'frames': len(record['features']),
                                   'ignored_global_leading_frames': config['common_leading_mask_frames'],
                                   'ignored_crop_edge_frames': spec['edge_mask_frames'],
                                   'target_delay_frames': config['common_target_delay_frames']}
                                  for record in records])
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=250, eta_min=spec['learning_rate'] * .1)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024, enabled=device.type == 'cuda' and spec['mixed_precision'])
    history, losses_in_epoch = [], []
    started = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    normalization = audit['normalization'][name]
    for step in range(1, 251):
        descriptors = schedule[(step - 1) * 4:step * 4]
        x, target = batch_from_schedule(descriptors, records, targets, normalization, spec['edge_mask_frames'])
        x = x.to(device)
        target = {key: value.to(device) for key, value in target.items()}
        losses, gradient = optimize(model, x, target, optimizer, scaler, model_config,
                                    audit['positive_weights'], spec, device)
        counters = actual_optimizer_steps(optimizer)
        if counters['minimum'] != step or counters['maximum'] != step:
            raise ValueError('AMP skipped an optimizer update; stop without adding replacement updates.')
        scheduler.step()
        losses_in_epoch.append(losses)
        if step % 50 == 0:
            history.append({'epoch': step // 50, 'optimizer_steps': step, 'actual_optimizer_step_counters': counters,
                            'training_losses': {key: float(np.mean([row[key] for row in losses_in_epoch])) for key in losses},
                            'gradient_norm': gradient})
            losses_in_epoch = []
            receipt = {'name': name, 'optimizer_steps': step, 'selected_step': 250, 'history': history,
                       'actual_optimizer_step_counters': counters, 'warmstarted': False,
                       'optimizer_parameter_steps': {'minimum': counters['minimum'], 'maximum': counters['maximum']},
                       'delay_frames': config['common_target_delay_frames'],
                       'target_delay_frames': config['common_target_delay_frames'],
                       'common_leading_mask_frames': config['common_leading_mask_frames'],
                       'crop_edge_mask_frames': spec['edge_mask_frames'],
                       'initial_weights_sha256': initial_digest, 'initial_checkpoint_sha256': sha256(initial_path),
                       'crop_schedule_sha256': fingerprint(schedule),
                       'gain_schedule_sha256': fingerprint([row['gain_log'] for row in schedule]),
                       'common_support_sha256': support_digest, 'config_sha256': fingerprint(config),
                       'source_sha256': producer_hashes(), 'config': config, 'model_config': model_config,
                       'normalization': normalization, 'positive_weights': audit['positive_weights'],
                       'early_stopping_used': False, 'validation_used_during_training': False,
                       'device': str(device), 'model_parameters': sum(p.numel() for p in model.parameters()),
                       'training_seconds': time.perf_counter() - started,
                       'peak_gpu_mib': torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None,
                       **provenance}
            checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
                          'receipt': receipt, 'config': model_config, 'normalization': normalization,
                          'positive_weights': audit['positive_weights']}
            save_checkpoint(directory / 'last.pt', checkpoint)
            save_json(directory / 'training-receipt.json', receipt)
            print('Spectral arm', name, 'epoch', step // 50, 'update', step,
                  'state loss', round(history[-1]['training_losses']['state'], 3), flush=True)
    save_checkpoint(directory / 'final.pt', checkpoint)
    return receipt
