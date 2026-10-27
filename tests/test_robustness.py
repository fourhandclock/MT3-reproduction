import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from robustness.effects import apply_effect,rms
from robustness.data import AugmentedCropDataset,base_config
from cnn_gru.data import CropDataset


@pytest.fixture
def config():
    with open('configs/robustness.json') as source: return json.load(source)


def test_effects_preserve_length_finiteness_silence_and_input(config):
    rng=np.random.default_rng(5);audio=rng.normal(0,0.05,4096).astype(np.float32);original=audio.copy()
    for parameters in config['robustness']['test_conditions'].values():
        output=apply_effect(audio,16000,parameters)
        assert output.shape==audio.shape and np.isfinite(output).all()
        assert np.max(np.abs(output))<=0.99001
        np.testing.assert_array_equal(apply_effect(np.zeros_like(audio),16000,parameters),0)
    np.testing.assert_array_equal(audio,original)
    np.testing.assert_array_equal(apply_effect(audio,16000,{}),audio)


def test_drive_does_not_shift_fundamental_pitch_or_add_fixed_delay():
    sample_rate=16000
    audio=(0.15*np.sin(2*np.pi*440*np.arange(sample_rate)/sample_rate)).astype(np.float32)
    output=apply_effect(audio,sample_rate,{'drive_db':20,'cabinet_hz':4000})
    assert np.fft.rfftfreq(len(output),1/sample_rate)[np.argmax(np.abs(np.fft.rfft(output)))]==440
    assert abs(rms(output)-rms(audio))<1e-5
    impulse=np.zeros(4096,dtype=np.float32);impulse[2048]=0.5
    processed=apply_effect(impulse,sample_rate,{'drive_db':16,'cabinet_hz':5000})
    assert np.argmax(np.abs(processed))==2048


def test_effect_recipe_retains_baseline_model_and_training_budget(config):
    with open('configs/cnn_gru.json') as source: assert base_config(config)==json.load(source)
    assert sum(config['robustness']['training_mix'].values())==1
    assert 'combined_unseen' not in config['robustness']['validation_conditions']


def test_clean_view_keeps_exact_crop_gain_and_labels(config):
    manifest_path=Path('data/cnn_gru/manifest.json')
    if not manifest_path.exists(): pytest.skip('Prepared GuitarSet cache required for integration check.')
    manifest=json.loads(manifest_path.read_text());audit=json.loads(Path('data/cnn_gru/cache_audit.json').read_text())
    config=copy.deepcopy(config);config['robustness']['training_mix']={'clean':1,'eq':0,'overdrive':0}
    baseline=CropDataset(manifest['train'],base_config(config),audit['normalization'])
    augmented=AugmentedCropDataset(manifest['train'],config,audit['normalization'])
    baseline.epoch=augmented.epoch=3
    left,left_targets=baseline[23];right,right_targets=augmented[23]
    assert torch.equal(left,right)
    assert all(torch.equal(left_targets[key],right_targets[key]) for key in left_targets)
