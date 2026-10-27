"""Frozen inputs, immutable parents and the next five deterministic crop epochs."""

import json
from pathlib import Path

import numpy as np
import torch

from acoustic10.data import load_inputs as frozen_inputs
from baseline.download import ROOT, sha256
from streaming.data import fingerprint
from streaming.evaluate import save_json


def validate_config(config):
    old = json.loads((ROOT / 'configs/normalization12.json').read_text(encoding='utf-8'))
    spec = {**old['training'], 'learning_rate': .0001}
    if (config['training'] != spec or config['seed'] != 42 or config['torch_continuation_seed'] != 43
            or config['data_epoch_start'] != 5 or config['total_optimizer_update_cap'] != 500
            or config['planned_new_optimizer_updates'] != 500 or config['common_target_delay_frames'] != 13
            or config['common_leading_mask_frames'] != 26 or config['amp_scaler_reset'] != {'initial_scale': 1024}
            or config['decoder_parameters'] != old['decoder_parameters']
            or config['pitch_diagnostics'] != old['pitch_diagnostics'] or config['validation_packet_ms'] != 20
            or config['test_inference_allowed'] is not False or config['production_promotion'] is not False):
        raise ValueError('The fixed continuation budget, restored-end learning rate and matching recipe changed.')
    expected = [{'name': 'group8', 'cnn_norm_groups': 8, 'parent_checkpoint': 'outputs/spectral11/models/short128/final.pt'},
                {'name': 'group1', 'cnn_norm_groups': 1, 'parent_checkpoint': 'outputs/normalization12/models/group1/final.pt'}]
    if config['arms'] != expected:
        raise ValueError('Only the two verified historical parent models can be continued.')


def build_schedule(config, records):
    spec = config['training']
    schedule = []
    for epoch in range(config['data_epoch_start'], config['data_epoch_start'] + spec['epochs']):
        generator = torch.Generator().manual_seed(config['seed'] + epoch)
        torch.empty((), dtype=torch.int64).random_(generator=generator)
        order = torch.randperm(spec['samples_per_epoch'], generator=generator).tolist()
        for index in order:
            rng = np.random.default_rng(config['seed'] + epoch * 100000 + index)
            record_index = int(rng.integers(len(records)))
            length = round(spec['crop_seconds'] * 100)
            start = int(rng.integers(len(records[record_index]['features']) - length + 1))
            schedule.append({'epoch': epoch + 1, 'sample_index': index, 'record_index': record_index,
                             'id': records[record_index]['entry']['id'], 'start_frame': start,
                             'frames': length, 'gain_log': float(rng.normal(0, spec['gain_augmentation_log_std']))})
    return schedule


def load_inputs(config, output):
    validate_config(config)
    old_config = json.loads((ROOT / 'configs/acoustic10.json').read_text(encoding='utf-8'))
    base, audit, datasets, provenance = frozen_inputs(old_config)
    verification_path = ROOT / 'outputs/normalization12/verification.json'
    verification = json.loads(verification_path.read_text(encoding='utf-8'))
    parents, references = {}, {}
    for arm in config['arms']:
        name = arm['name']
        path = ROOT / arm['parent_checkpoint']
        if sha256(path) != verification['checkpoint_sha256'][name]:
            raise ValueError('Parent checkpoint differs from the completed Stage12 verification.')
        parent = torch.load(path, map_location='cpu', weights_only=True)
        receipt = parent['receipt']
        if parent['step'] != 250 or receipt['optimizer_steps'] != 250:
            raise ValueError('Parent must be the fixed actual step250 checkpoint.')
        for filename, digest in receipt['source_sha256'].items():
            if sha256(ROOT / filename) != digest:
                raise ValueError('A producer underlying the parent changed: ' + filename)
        groups = parent['config']['model'].get('cnn_norm_groups', 8)
        if groups != arm['cnn_norm_groups'] or parent['config']['feature'] != base['feature']:
            raise ValueError('Parent architecture/grouping/frontend differs from the matched recipe.')
        if parent['normalization'] != audit['normalization'] or parent['positive_weights'] != audit['positive_weights']:
            raise ValueError('Parent normalization or loss balancing changed.')
        expected_training = {**config['training'], 'learning_rate': .001}
        if any(parent['config']['training'][key] != value for key, value in expected_training.items()):
            raise ValueError('The parent training recipe is incompatible with this continuation.')
        for key in ['manifest_sha256', 'cache_audit_sha256']:
            if receipt[key] != provenance[key]:
                raise ValueError('Parent training source identity changed.')
        parents[name] = parent
        references[name] = {'checkpoint_path': arm['parent_checkpoint'], 'checkpoint_sha256': sha256(path),
                            'historical_optimizer_steps': 250, 'cnn_norm_groups': groups,
                            'historical_receipt': receipt, 'verification_path': str(verification_path.relative_to(ROOT)).replace('\\', '/'),
                            'verification_sha256': sha256(verification_path),
                            'scope': 'Immutable trained parent; restored model and AdamW state, no file mutation.'}
    if parents['group8']['receipt']['initial_weights_sha256'] != parents['group1']['receipt']['initial_weights_sha256']:
        raise ValueError('The two historical parents do not originate from identical initial tensors.')
    input_audit = {**provenance, 'normalization': audit['normalization'], 'positive_weights': audit['positive_weights'],
                   'parents': references, 'config_sha256': fingerprint(config), 'new_optimizer_updates_for_preparation': 0,
                   'features_rewritten': False, 'targets_rewritten': False}
    output = Path(output).resolve()
    save_json(output / 'input-audit.json', input_audit)
    provenance.update(input_audit_sha256=sha256(output / 'input-audit.json'), parents=references)
    return base, audit, datasets, provenance, parents, build_schedule(config, datasets['train'])
