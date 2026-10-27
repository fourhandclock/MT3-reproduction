"""Isolated past-only spectral caches, with frozen source labels and splits.

Only the 48 training and 12 validation recordings are read.  The short arm
reuses the exact stage-8 feature files; the long arm changes the Hann window
and FFT together, retaining the 16-kHz/10-ms grid and all 216 log frequencies.
"""

import copy
import json
from pathlib import Path

import numpy as np

from acoustic10.data import load_inputs as load_frozen_inputs
from baseline.download import ROOT, sha256
from cnn_gru.data import load_audio
from streaming.data import fingerprint
from streaming.frontend import CausalFrontend


ARMS = [
    {'name': 'short128', 'window_length': 2048, 'n_fft': 2048},
    {'name': 'long256', 'window_length': 4096, 'n_fft': 4096},
]


def validate_config(config):
    """Reject changes that would silently add another experimental variable."""
    spec = config['training']
    if config['arms'] != ARMS:
        raise ValueError('Only the prespecified 2048/2048 and 4096/4096 arms are allowed.')
    if (config['total_optimizer_update_cap'] != 500
            or len(config['arms']) * spec['max_steps'] != 500
            or spec['epochs'] != 5 or spec['samples_per_epoch'] != 200
            or spec['batch_size'] != 4 or spec['max_steps'] != 250
            or spec['num_workers'] != 0
            or spec['samples_per_epoch'] * spec['epochs'] // spec['batch_size'] != 250):
        raise ValueError('Stage 11 requires the fixed 250+250 update schedule.')
    if (config['common_target_delay_frames'] != 13
            or config['common_leading_mask_frames'] != 26
            or spec['edge_mask_frames'] != 26
            or spec['crop_seconds'] != 2.0
            or spec['gain_augmentation_log_std'] != .15):
        raise ValueError('Both arms require the same 130-ms delay and 26-frame masks.')
    if config['test_inference_allowed'] is not False:
        raise ValueError('Stage 11 uses training and validation only.')


def arm_model_configs(config, original):
    """Keep the model, labels, grid and log frequencies unchanged between arms."""
    result = {}
    for arm in config['arms']:
        item = copy.deepcopy(original)
        item['feature'].update(window_length=arm['window_length'], n_fft=arm['n_fft'])
        result[arm['name']] = item
    return result


def normalization_from_training(records, n_bins=216):
    """Reproduce the old float32-square/float64-reduction rule exactly."""
    count = 0
    sums = np.zeros(n_bins, np.float64)
    squares = np.zeros(n_bins, np.float64)
    for record in records:
        x = np.asarray(record['features']).astype(np.float32)
        if x.ndim != 2 or x.shape[1] != n_bins or not np.isfinite(x).all():
            raise ValueError('Training features must have finite frames by frequency bins.')
        sums += x.sum(axis=0, dtype=np.float64)
        squares += np.square(x).sum(axis=0, dtype=np.float64)
        count += len(x)
    if count == 0:
        raise ValueError('Normalization requires training frames.')
    mean = sums / count
    std = np.sqrt(np.maximum(squares / count - mean ** 2, .09))
    return {'mean': mean.tolist(), 'std': std.tolist()}


def chunked_features(audio, model_config, packet_samples=320):
    """Deterministic unpaced PCM replay, without end padding or future samples."""
    if isinstance(packet_samples, bool) or not isinstance(packet_samples, int) or packet_samples < 1:
        raise ValueError('Packet length must be a positive integer.')
    frontend = CausalFrontend(model_config)
    blocks = [frontend.push(audio[start:start + packet_samples])
              for start in range(0, len(audio), packet_samples)]
    if not blocks:
        return np.empty((0, model_config['feature']['n_bins']), np.float32)
    return np.concatenate(blocks)


def _long_cache(record, source_info, model_config, seconds):
    identity = {
        'feature': model_config['feature'], 'seconds': seconds,
        'audio_sha256': record['entry']['audio_sha256'],
        'annotation_sha256': record['entry']['annotation_sha256'],
        'frozen_targets_sha256': source_info['targets_sha256'],
        'frontend_sha256': sha256(ROOT / 'streaming/frontend.py'),
        'load_audio_sha256': sha256(ROOT / 'cnn_gru/data.py'),
        'spectral_data_sha256': sha256(Path(__file__)),
        'grid': 'right edge (frame+1)*160 samples; no end padding',
    }
    prefix = ROOT / 'data/spectral11/cache' / (record['entry']['id'] + '-' + fingerprint(identity)[:16])
    feature_path, receipt_path = prefix.with_suffix('.npy'), prefix.with_suffix('.json')
    if feature_path.exists() != receipt_path.exists():
        raise ValueError('Incomplete long-window cache; preserve and inspect the partial files.')
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt.get('identity') != identity or sha256(feature_path) != receipt.get('features_sha256'):
            raise ValueError('Long-window cache identity or feature bytes changed.')
    else:
        audio, duration = load_audio(ROOT / record['entry']['audio'], seconds, model_config)
        if len(audio) // model_config['feature']['hop_length'] != len(record['features']):
            raise ValueError('Long and short windows must emit the same number of right-edge frames.')
        if duration != source_info['duration']:
            raise ValueError('Audio loading changed the frozen excerpt duration.')
        features = CausalFrontend(model_config).push(audio)
        replayed = chunked_features(audio, model_config)
        if not np.array_equal(features, replayed):
            raise ValueError('Long features depend on packet boundaries.')
        prefix.parent.mkdir(parents=True, exist_ok=True)
        np.save(feature_path, features)
        receipt = {
            'identity': identity, 'id': record['entry']['id'], 'duration': duration,
            'frames': len(features), 'features_sha256': sha256(feature_path),
            'full_vs_20ms_packets_byte_equal': True, 'lookahead_samples': 0,
        }
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    features = np.load(feature_path, mmap_mode='r')
    if (features.shape != record['features'].shape or features.dtype != np.float32
            or not np.isfinite(features).all()):
        raise ValueError('Long-window features have an invalid shape, dtype or value.')
    info = {
        **record['info'], 'features_sha256': receipt['features_sha256'],
        'features_path': feature_path.relative_to(ROOT).as_posix(),
        'receipt_path': receipt_path.relative_to(ROOT).as_posix(),
        'receipt_sha256': sha256(receipt_path),
        'targets_path': source_info['prefix'] + '.npz',
        'source_features_sha256': source_info['features_sha256'],
        'audio_sha256': record['entry']['audio_sha256'],
        'annotation_sha256': record['entry']['annotation_sha256'],
    }
    return {'entry': record['entry'], 'features': features,
            'targets': record['targets'], 'info': info}


def prepare(config, output):
    """Verify frozen inputs, build the long cache and return both matched arms.

    Returns ``(model_configs, audit, datasets, provenance)``.  ``datasets`` is
    indexed by arm then train/validation.  Original targets are shared in
    memory and reused on disk; this function neither delays nor writes them.
    ``audit['normalization']`` is indexed by arm. Positive loss weights remain
    the frozen stage-8 values, so only features and their training statistics
    change. Preparing the inputs performs zero optimizer updates.
    """
    validate_config(config)
    legacy_config_path = ROOT / 'configs/acoustic10.json'
    legacy_config = json.loads(legacy_config_path.read_text(encoding='utf-8'))
    original, frozen_audit, frozen, provenance = load_frozen_inputs(legacy_config)
    models = arm_model_configs(config, original)
    datasets = {arm['name']: {'train': [], 'validation': []} for arm in config['arms']}
    receipts = {arm['name']: {'train': [], 'validation': []} for arm in config['arms']}
    for split in ['train', 'validation']:
        source_infos = provenance['inputs'][split]
        if len(source_infos) != len(frozen[split]):
            raise ValueError('Frozen input provenance is incomplete.')
        seconds = original['train_audio_seconds'] if split == 'train' else original['evaluation_audio_seconds']
        for record, source_info in zip(frozen[split], source_infos):
            if record['entry']['id'] != source_info['id']:
                raise ValueError('Frozen cache and provenance order disagree.')
            # Reference the same .npy: no rewriting or approximate regeneration.
            short = {**record, 'info': {
                **record['info'], 'features_path': source_info['prefix'] + '.npy',
                'targets_path': source_info['prefix'] + '.npz',
                'audio_sha256': record['entry']['audio_sha256'],
                'annotation_sha256': record['entry']['annotation_sha256'],
                'byte_identical_frozen_features': True,
            }}
            long = _long_cache(record, source_info, models['long256'], seconds)
            for name, item in [('short128', short), ('long256', long)]:
                datasets[name][split].append(item)
                receipts[name][split].append(item['info'])
            print('Spectral cache', split, record['entry']['id'], len(record['features']), 'frames', flush=True)
    normalization = {name: normalization_from_training(parts['train']) for name, parts in datasets.items()}
    if normalization['short128'] != frozen_audit['normalization']:
        raise ValueError('Recomputed short-arm train-only normalization differs from the frozen cache.')
    audit = {
        'config_sha256': fingerprint(config), 'model_configs': models,
        'normalization': normalization, 'positive_weights': frozen_audit['positive_weights'],
        'normalization_source': '48 training recordings only; recomputed separately for each arm',
        'positive_weights_source': 'Frozen stage-8 common loss weights',
        'arms': receipts, 'test_audio_loaded': False, 'new_optimizer_updates': 0,
        'frontend_sha256': sha256(ROOT / 'streaming/frontend.py'),
        'data_wrapper_sha256': sha256(Path(__file__)),
    }
    provenance = {
        **provenance, 'legacy_verifier_config_sha256': sha256(legacy_config_path),
        'normalization_source': audit['normalization_source'], 'spectral_receipts': receipts,
        'cache_audit_sha256_stage8': provenance['cache_audit_sha256'],
        'source_split_scope': '48 train and 12 validation; test audio and caches unread',
        'original_targets_rewritten': False, 'test_inference_calls': 0,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'cache-audit.json'
    path.write_text(json.dumps({**audit, 'provenance': provenance}, indent=2), encoding='utf-8')
    provenance['spectral_cache_receipts_path'] = path.relative_to(ROOT).as_posix()
    provenance['spectral_cache_receipts_sha256'] = sha256(path)
    return models, audit, datasets, provenance
