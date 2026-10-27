"""Reuse exact short-window inputs and verify the historical training recipe."""

import json
from pathlib import Path

import torch

from acoustic10.data import build_schedule, load_inputs as frozen_inputs
from acoustic10.training import tensor_digest
from baseline.download import ROOT, sha256
from streaming.data import fingerprint
from streaming.evaluate import save_json


def validate_config(config):
    old = json.loads((ROOT / config['control_stage_config']).read_text(encoding='utf-8'))
    if config['arms'] != [{'name': 'group8', 'cnn_norm_groups': 8}, {'name': 'group1', 'cnn_norm_groups': 1}]:
        raise ValueError('Only the prespecified eight/one-group comparison is allowed.')
    if (config['total_optimizer_update_cap'] != 500 or config['planned_new_optimizer_updates'] != 250
            or config['training'] != old['training'] or config['seed'] != old['seed']
            or config['common_target_delay_frames'] != 13 or config['common_leading_mask_frames'] != 26
            or config['decoder_parameters'] != old['decoder_parameters']
            or config['pitch_diagnostics'] != old['pitch_diagnostics']
            or config['validation_packet_ms'] != old['validation_packet_ms']
            or config['test_inference_allowed'] is not False or config['production_promotion'] is not False):
        raise ValueError('The historical recipe, timing, diagnostics and new-update budget must remain fixed.')
    return old


def load_inputs(config, output):
    stage11_config = validate_config(config)
    verification_path = ROOT / 'outputs/spectral11/verification.json'
    verification = json.loads(verification_path.read_text(encoding='utf-8'))
    control_path = ROOT / config['control_checkpoint']
    if sha256(control_path) != verification['checkpoint_sha256']['short128']:
        raise ValueError('The preserved historical control checkpoint changed.')
    checkpoint = torch.load(control_path, map_location='cpu', weights_only=True)
    receipt_path = ROOT / config['control_receipt']
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if checkpoint['receipt'] != receipt or receipt['optimizer_steps'] != 250:
        raise ValueError('Historical control receipt/checkpoint disagree.')
    for name, digest in receipt['source_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError('Historical control producer changed: ' + name)
    if receipt['config'] != stage11_config:
        raise ValueError('Historical stage configuration changed.')
    initial_path = ROOT / config['control_initial_weights']
    initial = torch.load(initial_path, map_location='cpu', weights_only=True)
    if (sha256(initial_path) != receipt['initial_checkpoint_sha256']
            or tensor_digest(initial['model']) != receipt['initial_weights_sha256']):
        raise ValueError('Historical fresh initialization changed.')
    old_config = json.loads((ROOT / 'configs/acoustic10.json').read_text(encoding='utf-8'))
    base, audit, datasets, provenance = frozen_inputs(old_config)
    if base['feature']['window_length'] != 2048 or base['feature']['n_fft'] != 2048:
        raise ValueError('The shared short analysis window changed.')
    if audit['normalization'] != receipt['normalization'] or audit['positive_weights'] != receipt['positive_weights']:
        raise ValueError('Historical normalization or loss weighting changed.')
    for key in ['manifest_sha256', 'cache_audit_sha256']:
        if provenance[key] != receipt[key]:
            raise ValueError('Historical input identity changed.')
    schedule_path = ROOT / config['control_crop_schedule']
    saved_schedule = json.loads(schedule_path.read_text(encoding='utf-8'))
    schedule = saved_schedule['descriptors']
    if (schedule != build_schedule(config, datasets['train'])
            or fingerprint(schedule) != saved_schedule['sha256']
            or fingerprint(schedule) != receipt['crop_schedule_sha256']
            or fingerprint([row['gain_log'] for row in schedule]) != receipt['gain_schedule_sha256']):
        raise ValueError('The fixed historical crop/gain schedule changed.')
    support = fingerprint([{'id': row['entry']['id'], 'frames': len(row['features']),
                            'ignored_global_leading_frames': 26, 'ignored_crop_edge_frames': 26,
                            'target_delay_frames': 13} for row in datasets['train']])
    if support != receipt['common_support_sha256']:
        raise ValueError('Historical target support changed.')
    reference = {'checkpoint_path': config['control_checkpoint'], 'checkpoint_sha256': sha256(control_path),
                 'receipt_path': config['control_receipt'], 'receipt_sha256': sha256(receipt_path),
                 'initial_checkpoint_path': config['control_initial_weights'], 'initial_checkpoint_sha256': sha256(initial_path),
                 'crop_schedule_path': config['control_crop_schedule'], 'crop_schedule_file_sha256': sha256(schedule_path),
                 'crop_schedule_sha256': sha256(schedule_path),
                 'verification_path': str(verification_path.relative_to(ROOT)).replace('\\', '/'),
                 'verification_sha256': sha256(verification_path),
                 'historical_optimizer_updates': 250, 'new_control_optimizer_updates': 0,
                 'historical_receipt': receipt, 'scope': 'Exact frozen Stage11 short128 control; no new control training.'}
    input_audit = {**provenance, 'normalization': audit['normalization'], 'positive_weights': audit['positive_weights'],
                   'control_reference': reference, 'config_sha256': fingerprint(config),
                   'new_optimizer_updates_for_preparation': 0, 'features_rewritten': False, 'targets_rewritten': False}
    output = Path(output).resolve()
    save_json(output / 'input-audit.json', input_audit)
    provenance.update(input_audit_sha256=sha256(output / 'input-audit.json'), control_reference=reference)
    return base, audit, datasets, provenance, checkpoint, schedule
