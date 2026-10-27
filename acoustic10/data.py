"""Verify and reuse frozen train/validation caches, without preparing test audio."""

import json
from pathlib import Path

import numpy as np
import torch

from baseline.download import ROOT, sha256
from streaming.data import cache_path, fingerprint
from .targets import delay_targets


def validate_config(config):
    spec = config['training']
    planned = config['sanity']['optimizer_updates'] + len(config['arms']) * spec['max_steps']
    if (config['total_optimizer_update_cap'] != 500 or planned > 500
            or spec['epochs'] != 5 or spec['samples_per_epoch'] != 160
            or spec['batch_size'] != 4 or spec['num_workers'] != 0
            or spec['max_steps'] != 200 or config['sanity']['optimizer_updates'] != 100
            or config['test_inference_allowed'] is not False):
        raise ValueError('Stage 10 requires the fixed 100+200+200 budget and no test inference.')
    if config['arms'] != [{'name': 'baseline', 'target_delay_frames': 0},
                          {'name': 'delay60', 'target_delay_frames': 6}]:
        raise ValueError('Only the two prespecified target-timing arms are allowed.')
    if config['common_leading_mask_frames'] != 6:
        raise ValueError('Both arms require the same six-frame leading support mask.')
    if config['decoder_parameters'] != {'onset_threshold': .55, 'activity_threshold': .35,
                                         'onset_backdate_seconds': .06}:
        raise ValueError('The prespecified decoder parameters must remain fixed.')
    if spec['samples_per_epoch'] * spec['epochs'] // spec['batch_size'] != spec['max_steps']:
        raise ValueError('Epoch sample schedule must give exactly 200 updates.')


def load_inputs(config, include_validation=True):
    validate_config(config)
    model_config = json.loads((ROOT / config['model_config']).read_text(encoding='utf-8'))
    manifest_path = ROOT / 'data/cnn_gru/manifest.json'
    audit_path = ROOT / 'data/streaming/cache_audit.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if audit['config_sha256'] != fingerprint(model_config) or audit['manifest_sha256'] != sha256(manifest_path):
        raise ValueError('Frozen cache configuration or source split changed.')
    parent_path = ROOT / 'outputs/streaming/model/best-inference.pt'
    parent = torch.load(parent_path, map_location='cpu', weights_only=True)
    for name, digest in parent['source_sha256'].items():
        if sha256(ROOT / 'streaming' / name) != digest:
            raise ValueError('Frozen streaming source changed: ' + name)
    if parent['legacy_loss_sha256'] != sha256(ROOT / 'cnn_gru/model.py'):
        raise ValueError('Frozen shared loss changed.')
    if parent['config'] != model_config or parent['normalization'] != audit['normalization']:
        raise ValueError('Frozen model/cache normalization differs.')
    datasets = {}
    inputs = {}
    for split in (['train', 'validation'] if include_validation else ['train']):
        entries = manifest[split]
        records = {record['id']: record for record in audit['splits'][split]}
        if len(entries) != (48 if split == 'train' else 12) or set(records) != {e['id'] for e in entries}:
            raise ValueError('Incomplete training/validation split.')
        datasets[split], inputs[split] = [], []
        seconds = model_config['train_audio_seconds'] if split == 'train' else model_config['evaluation_audio_seconds']
        for entry in entries:
            for kind in ['audio', 'annotation']:
                if sha256(ROOT / entry[kind]) != entry[kind + '_sha256']:
                    raise ValueError('Original ' + split + ' source changed.')
            prefix = cache_path(entry, model_config, seconds)
            record = records[entry['id']]
            for kind, extension in [('features', '.npy'), ('targets', '.npz')]:
                if sha256(prefix.with_suffix(extension)) != record[kind + '_sha256']:
                    raise ValueError('Frozen ' + split + ' cache changed.')
            features = np.load(prefix.with_suffix('.npy'), mmap_mode='r')
            with np.load(prefix.with_suffix('.npz'), allow_pickle=False) as arrays:
                targets = {key: arrays[key].copy() for key in arrays.files}
            datasets[split].append({'entry': entry, 'features': features, 'targets': targets, 'info': record})
            inputs[split].append({**record, 'audio_sha256': entry['audio_sha256'],
                                  'annotation_sha256': entry['annotation_sha256'],
                                  'prefix': str(prefix.relative_to(ROOT)).replace('\\', '/')})
    if {e['id'] for e in manifest['train']} & {e['id'] for e in manifest['validation']}:
        raise ValueError('Training/validation IDs overlap.')
    provenance = {'manifest_sha256': sha256(manifest_path), 'cache_audit_sha256': sha256(audit_path),
                  'frozen_parent_checkpoint_sha256': sha256(parent_path), 'inputs': inputs,
                  'normalization_source': '48 training recordings only; frozen stage-8 statistics',
                  'validation_arrays_loaded': bool(include_validation),
                  'test_audio_loaded': False, 'test_inference_calls': 0}
    return model_config, audit, datasets, provenance


def build_schedule(config, records):
    """Record the exact repeated-sampling crop order and gain draws for both arms."""
    training = config['training']
    schedule = []
    length = round(training['crop_seconds'] * 100)
    for epoch in range(training['epochs']):
        generator = torch.Generator().manual_seed(config['seed'] + epoch)
        # Same random-sampler order as a DataLoader: generator also reserves a base seed.
        torch.empty((), dtype=torch.int64).random_(generator=generator)
        order = torch.randperm(training['samples_per_epoch'], generator=generator).tolist()
        for index in order:
            rng = np.random.default_rng(config['seed'] + epoch * 100000 + index)
            record_index = int(rng.integers(len(records)))
            start = int(rng.integers(len(records[record_index]['features']) - length + 1))
            gain = float(rng.normal(0, training['gain_augmentation_log_std']))
            schedule.append({'epoch': epoch + 1, 'sample_index': index, 'record_index': record_index,
                             'id': records[record_index]['entry']['id'], 'start_frame': start,
                             'frames': length, 'gain_log': gain})
    return schedule


def prepare_targets(records, delay, config, model_config):
    result = []
    for record in records:
        targets = delay_targets(record['targets'], delay, model_config)
        mask = np.ones(len(targets['state']), np.float32)
        mask[:config['common_leading_mask_frames']] = 0
        targets['mask'] = mask
        result.append(targets)
    return result


def batch_from_schedule(descriptors, records, targets, normalization, edge):
    xs, pieces = [], []
    mean = np.asarray(normalization['mean'], np.float32)
    std = np.asarray(normalization['std'], np.float32)
    for descriptor in descriptors:
        index, start, length = descriptor['record_index'], descriptor['start_frame'], descriptor['frames']
        x = records[index]['features'][start:start + length].astype(np.float32)
        x += descriptor['gain_log']
        xs.append(np.clip((x - mean) / std, -5, 5))
        item = {key: value[start:start + length].copy() for key, value in targets[index].items()}
        if edge:
            item['mask'][:edge] = item['mask'][-edge:] = 0
        pieces.append(item)
    tensors = {key: torch.from_numpy(np.stack([item[key] for item in pieces])).long() if key == 'state'
               else torch.from_numpy(np.stack([item[key] for item in pieces]).astype(np.float32))
               for key in pieces[0]}
    return torch.from_numpy(np.stack(xs)), tensors
