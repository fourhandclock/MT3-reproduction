import hashlib
import json
from pathlib import Path

import numpy as np

from baseline.download import ROOT,sha256
from cnn_gru.data import cache_path,features,fingerprint,load_audio,validate_splits
from .effects import apply_effect


def base_config(config):
    return {key:value for key,value in config.items() if key!='robustness'}


def inputs(config):
    reference=json.loads((ROOT/'configs/cnn_gru.json').read_text())
    if base_config(config)!=reference: raise ValueError('The augmentation trial must retain the original GRU recipe.')
    manifest=json.loads((ROOT/'data/cnn_gru/manifest.json').read_text())
    audit=json.loads((ROOT/'data/cnn_gru/cache_audit.json').read_text())
    validate_splits(manifest)
    if manifest['config']!=reference or audit['config_sha256']!=fingerprint(reference): raise ValueError('Base dataset configuration changed.')
    return manifest,audit


def effect_cache(entry,config,split,view):
    tag=fingerprint({'feature':config['feature'],'effects':config['robustness'],
                     'effect_source':sha256(ROOT/'robustness/effects.py'),
                     'audio':entry['audio_sha256'],'split':split,'view':view})[:16]
    return ROOT/'data/robustness/cache'/(entry['id']+'-'+view+'-'+tag)


def training_effect(entry,view,config):
    key=int(hashlib.sha256((entry['id']+view).encode()).hexdigest()[:8],16)
    rng=np.random.default_rng(config['seed']+key)
    if view=='eq':
        low=float(rng.uniform(-6,6));high=float(rng.uniform(-6,6))
        if abs(low-high)<3: high=low-4 if low>0 else low+4
        return {'low_db':low,'high_db':high,'split_hz':float(rng.uniform(700,1400))}
    return {'drive_db':float(rng.uniform(6,16)),'cabinet_hz':float(rng.uniform(4000,7000))}


def prepare_cache(config):
    import soundfile as sf
    manifest,audit=inputs(config)
    cache_audit={'config_sha256':fingerprint(config),'base_manifest_sha256':sha256(ROOT/'data/cnn_gru/manifest.json'),
                 'normalization':'Unchanged clean-training statistics; no validation/test/external statistics.',
                 'effect_rendering':'Complete polyphonic waveform; 4x oversampled tanh; zero-phase filters; RMS matched.',
                 'entries':[]}
    for split in ['train','validation','test']:
        seconds=config['train_audio_seconds'] if split=='train' else config['evaluation_audio_seconds']
        views=['eq','overdrive'] if split=='train' else list(config['robustness'][split+'_conditions'])
        for entry in manifest[split]:
            if sha256(ROOT/entry['audio'])!=entry['audio_sha256']: raise ValueError('Audio checksum mismatch.')
            audio,duration=load_audio(ROOT/entry['audio'],seconds,config)
            base=cache_path(entry,config,seconds)
            frame_count=len(np.load(base.with_suffix('.npy'),mmap_mode='r'))
            for view in views:
                if view=='clean': continue
                parameters=training_effect(entry,view,config) if split=='train' else config['robustness'][split+'_conditions'][view]
                prefix=effect_cache(entry,config,split,view);prefix.parent.mkdir(parents=True,exist_ok=True)
                if not prefix.with_suffix('.json').exists():
                    processed=apply_effect(audio,config['feature']['sample_rate'],parameters)
                    x=features(processed,config)
                    if len(x)!=frame_count: raise ValueError('Effect changed frame/label alignment.')
                    np.save(prefix.with_suffix('.npy'),x)
                    waveform=None
                    if split!='train':
                        waveform=prefix.with_suffix('.wav')
                        sf.write(waveform,processed,config['feature']['sample_rate'],subtype='FLOAT')
                    info={'id':entry['id'],'split':split,'view':view,'duration':duration,'frames':frame_count,
                          'parameters':parameters,'features_sha256':sha256(prefix.with_suffix('.npy')),
                          'waveform':waveform.relative_to(ROOT).as_posix() if waveform else None}
                    prefix.with_suffix('.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
                info=json.loads(prefix.with_suffix('.json').read_text())
                if sha256(prefix.with_suffix('.npy'))!=info['features_sha256']: raise ValueError('Effect cache changed.')
                cache_audit['entries'].append(info)
            print('Rendered effect views',split,entry['id'],flush=True)
    destination=ROOT/'data/robustness';destination.mkdir(exist_ok=True)
    (destination/'cache_audit.json').write_text(json.dumps(cache_audit,indent=2),encoding='utf-8')


class AugmentedCropDataset:
    def __init__(self,entries,config,normalization):
        self.entries,self.config,self.epoch=entries,config,0
        self.mean=np.array(normalization['mean'],dtype=np.float32);self.std=np.array(normalization['std'],dtype=np.float32)

    def __len__(self): return self.config['training']['samples_per_epoch']

    def __getitem__(self,index):
        import torch
        config=self.config;training=config['training']
        rng=np.random.default_rng(config['seed']+self.epoch*100000+index)
        entry=self.entries[int(rng.integers(len(self.entries)))]
        base=cache_path(entry,config,config['train_audio_seconds'])
        original=np.load(base.with_suffix('.npy'),mmap_mode='r')
        length=round(training['crop_seconds']*config['feature']['sample_rate']/config['feature']['hop_length'])
        start=int(rng.integers(len(original)-length+1))
        gain=rng.normal(0,training['gain_augmentation_log_std'])
        variant_rng=np.random.default_rng(config['seed']+900000000+self.epoch*100000+index)
        probabilities=config['robustness']['training_mix']
        view=variant_rng.choice(['clean','eq','overdrive'],p=[probabilities[name] for name in ['clean','eq','overdrive']])
        source=original if view=='clean' else np.load(effect_cache(entry,config,'train',view).with_suffix('.npy'),mmap_mode='r')
        x=source[start:start+length].astype(np.float32)+gain
        x=np.clip((x-self.mean)/self.std,-5,5)
        with np.load(base.with_suffix('.npz')) as labels:
            targets={key:torch.from_numpy(labels[key][start:start+length].copy()).long() if key=='state'
                     else torch.from_numpy(labels[key][start:start+length].astype(np.float32)) for key in labels.files}
        mask=np.ones(length,dtype=np.float32);edge=training['edge_mask_frames']
        if edge: mask[:edge]=mask[-edge:]=0
        targets['mask']=torch.from_numpy(mask)
        return torch.from_numpy(x),targets
