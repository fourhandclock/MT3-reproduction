import copy
import json

import numpy as np
import pytest

from baseline.download import sha256
from spectral11 import data
from streaming.frontend import CausalFrontend


def config():
    return {
        'arms': copy.deepcopy(data.ARMS), 'total_optimizer_update_cap': 500,
        'common_target_delay_frames': 13, 'common_leading_mask_frames': 26,
        'test_inference_allowed': False,
        'training': {'epochs': 5, 'samples_per_epoch': 200, 'batch_size': 4,
                     'max_steps': 250, 'num_workers': 0, 'edge_mask_frames': 26,
                     'crop_seconds': 2.0, 'gain_augmentation_log_std': .15},
    }


def model(window=2048):
    return {
        'feature': {'sample_rate': 16000, 'hop_length': 160, 'window_length': window,
                    'n_fft': window, 'n_bins': 216, 'log_floor': 1e-5,
                    'fmin_midi': 40, 'bins_per_octave': 36},
        'train_audio_seconds': .2, 'evaluation_audio_seconds': .2,
        'model': {'max_fret': 24, 'pitch_min': 40, 'pitch_max': 88},
    }


@pytest.mark.parametrize('window', [2048, 4096])
@pytest.mark.parametrize('packet', [1, 159, 320, 1277])
def test_full_and_irregular_packets_are_byte_equal(window, packet):
    samples = np.random.default_rng(7).normal(0, .1, 3271).astype(np.float32)
    expected = CausalFrontend(model(window)).push(samples)
    replay = data.chunked_features(samples, model(window), packet)
    np.testing.assert_array_equal(expected, replay)
    assert expected.shape == (20, 216)  # No padded final 71 samples.


@pytest.mark.parametrize('window', [2048, 4096])
def test_future_samples_cannot_change_an_emitted_prefix(window):
    rng = np.random.default_rng(3)
    first = rng.normal(0, .1, 9600).astype(np.float32)
    changed = first.copy()
    changed[3200:] = rng.normal(0, 2, len(first) - 3200)
    left = CausalFrontend(model(window)).push(first)
    right = CausalFrontend(model(window)).push(changed)
    prefix = CausalFrontend(model(window)).push(first[:3200])
    np.testing.assert_array_equal(left[:20], right[:20])
    np.testing.assert_array_equal(prefix, left[:20])


def test_window_ablation_retains_frequency_grid_and_target_model():
    original = model()
    before = copy.deepcopy(original)
    arms = data.arm_model_configs(config(), original)
    assert original == before
    assert arms['short128'] == original
    expected = copy.deepcopy(original)
    expected['feature'].update(window_length=4096, n_fft=4096)
    assert arms['long256'] == expected
    assert 16000 / 2048 == 7.8125
    assert 16000 / 4096 == 3.90625
    # These are FFT grid spacings, not a promise of note resolution.


@pytest.mark.parametrize('change', [
    lambda c: c['training'].update(max_steps=251),
    lambda c: c['training'].update(epochs=6),
    lambda c: c.update(test_inference_allowed=True),
    lambda c: c.update(common_target_delay_frames=6),
    lambda c: c['training'].update(edge_mask_frames=8),
    lambda c: c['arms'][1].update(n_fft=8192),
])
def test_experiment_cannot_silently_expand_or_add_variables(change):
    source = config()
    data.validate_config(source)
    change(source)
    with pytest.raises(ValueError):
        data.validate_config(source)


def test_train_statistic_dtype_and_std_floor_match_frozen_rule():
    values = np.array([[1.2, -3.0], [2.4, -3.0], [4.8, -3.0]], np.float32)
    result = data.normalization_from_training([{'features': values}], 2)
    mean = values.sum(axis=0, dtype=np.float64) / 3
    variance = np.square(values).sum(axis=0, dtype=np.float64) / 3 - mean ** 2
    assert result == {'mean': mean.tolist(), 'std': np.sqrt(np.maximum(variance, .09)).tolist()}
    assert result['std'][1] == .3
    with pytest.raises(ValueError):
        data.normalization_from_training([], 2)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    root = tmp_path / 'workspace'
    for folder in ['configs', 'streaming', 'cnn_gru', 'data/old']:
        (root / folder).mkdir(parents=True, exist_ok=True)
    (root / 'configs/acoustic10.json').write_text('{}')
    (root / 'streaming/frontend.py').write_text('frozen frontend')
    (root / 'cnn_gru/data.py').write_text('frozen audio loader')
    rng = np.random.default_rng(19)
    audio = rng.normal(0, .1, 3200).astype(np.float32)
    features = CausalFrontend(model()).push(audio)
    datasets, infos = {}, {}
    for split, count in [('train', 48), ('validation', 12)]:
        datasets[split], infos[split] = [], []
        for index in range(count):
            name = f'{split}-{index}'
            prefix = root / 'data/old' / name
            np.save(prefix.with_suffix('.npy'), features)
            targets = {'state': np.full((20, 6), 25, np.int16),
                       'onset': np.zeros((20, 6), np.float16),
                       'offset': np.zeros((20, 6), np.float16),
                       'pitch': np.zeros((20, 49), np.uint8)}
            np.savez_compressed(prefix.with_suffix('.npz'), **targets)
            info = {'id': name, 'frames': 20, 'duration': .2,
                    'features_sha256': sha256(prefix.with_suffix('.npy')),
                    'targets_sha256': sha256(prefix.with_suffix('.npz'))}
            entry = {'id': name, 'audio': name + '.wav', 'annotation': name + '.jams',
                     'audio_sha256': 'audio-identity', 'annotation_sha256': 'annotation-identity'}
            datasets[split].append({'entry': entry, 'features': np.load(prefix.with_suffix('.npy'), mmap_mode='r'),
                                    'targets': targets, 'info': info})
            infos[split].append({**info, 'prefix': prefix.relative_to(root).as_posix()})
    frozen = {'normalization': data.normalization_from_training(datasets['train']),
              'positive_weights': {'onset': 9., 'offset': 8., 'pitch': 4.}}
    provenance = {'inputs': infos, 'manifest_sha256': 'manifest-identity',
                  'cache_audit_sha256': 'audit-identity',
                  'frozen_parent_checkpoint_sha256': 'parent-identity',
                  'test_audio_loaded': False, 'test_inference_calls': 0}
    monkeypatch.setattr(data, 'ROOT', root)
    monkeypatch.setattr(data, 'load_frozen_inputs', lambda _: (model(), frozen, datasets, provenance))
    monkeypatch.setattr(data, 'load_audio', lambda *args: (audio.copy(), .2))
    return root, datasets, frozen, audio


def test_prepare_preserves_original_targets_short_bytes_and_train_only_statistics(inputs):
    root, old, frozen, audio = inputs
    before = {record['entry']['id']: sha256(root / ('data/old/' + record['info']['id'] + '.npz'))
              for split in old.values() for record in split}
    models, audit, datasets, provenance = data.prepare(config(), root / 'outputs/spectral11')
    assert audit['normalization']['short128'] == frozen['normalization']
    assert audit['positive_weights'] == frozen['positive_weights']
    expected_long = CausalFrontend(models['long256']).push(audio)
    assert audit['normalization']['long256'] == data.normalization_from_training([{'features': expected_long}] * 48)
    for name in ['short128', 'long256']:
        for split in ['train', 'validation']:
            assert len(datasets[name][split]) == (48 if split == 'train' else 12)
            for old_record, record in zip(old[split], datasets[name][split]):
                assert record['targets'] is old_record['targets']
                assert sha256(root / record['info']['targets_path']) == before[record['entry']['id']]
                assert sha256(root / record['info']['features_path']) == record['info']['features_sha256']
                if name == 'short128':
                    assert record['features'] is old_record['features']
                    assert record['info']['features_sha256'] == old_record['info']['features_sha256']
    assert provenance['test_audio_loaded'] is False
    assert provenance['original_targets_rewritten'] is False
    receipt = root / provenance['spectral_cache_receipts_path']
    assert receipt.name == 'cache-audit.json'
    assert sha256(receipt) == provenance['spectral_cache_receipts_sha256']
    assert json.loads(receipt.read_text())['provenance']['manifest_sha256'] == 'manifest-identity'


def test_cached_long_features_reject_tampering(inputs):
    root, *_ = inputs
    _, _, datasets, _ = data.prepare(config(), root / 'outputs/spectral11')
    feature_path = root / datasets['long256']['train'][0]['info']['features_path']
    value = np.load(feature_path)
    value[0, 0] += .1
    datasets['long256']['train'][0]['features']._mmap.close()
    np.save(feature_path, value)
    with pytest.raises(ValueError, match='identity or feature bytes changed'):
        data.prepare(config(), root / 'outputs/spectral11')


def test_frozen_input_failure_prevents_new_caches(inputs, monkeypatch):
    root, *_ = inputs
    def reject(_):
        raise ValueError('Frozen parent/source changed')
    monkeypatch.setattr(data, 'load_frozen_inputs', reject)
    with pytest.raises(ValueError, match='Frozen parent/source changed'):
        data.prepare(config(), root / 'outputs/spectral11')
    assert not (root / 'data/spectral11').exists()
