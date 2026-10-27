import hashlib
import json
from pathlib import Path

import numpy as np

from baseline.download import ROOT,sha256
from baseline.evaluation import reference_notes
from cnn_gru.data import load_audio,validate_splits
from .targets import make_targets


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def cache_path(entry,config,seconds):
    digest=fingerprint({'feature':config['feature'],'seconds':seconds,'audio_sha256':entry['audio_sha256'],
                        'annotation_sha256':entry['annotation_sha256'],'frontend_sha256':sha256(ROOT/'streaming/frontend.py'),
                        'target_builder_sha256':sha256(ROOT/'streaming/targets.py'),
                        'data_wrapper_sha256':sha256(ROOT/'streaming/data.py'),
                        'target_model':{k:config['model'][k] for k in ['max_fret','pitch_min','pitch_max']},
                        'target_time_grid':'rightedge n+1 hops; stable explicit indices and observed-boundary censoring'})[:16]
    return ROOT/'data/streaming/cache'/(entry['id']+'-'+digest)


def prepare(config):
    from .frontend import features
    manifest_path=ROOT/'data/cnn_gru/manifest.json';manifest=json.loads(manifest_path.read_text());validate_splits(manifest)
    audit={'config_sha256':fingerprint(config),'manifest_sha256':sha256(manifest_path),'splits':{},
           'frontend_sha256':sha256(ROOT/'streaming/frontend.py'),
           'target_time_grid':'Frame k covers only audio through (k+1)*hop/sample_rate.'}
    for split in ['train','validation','test']:
        seconds=config['train_audio_seconds'] if split=='train' else config['evaluation_audio_seconds']
        audit['splits'][split]=[]
        for entry in manifest[split]:
            for key in ['audio','annotation']:
                if sha256(ROOT/entry[key])!=entry[key+'_sha256']:raise ValueError('Streaming source changed.')
            prefix=cache_path(entry,config,seconds);prefix.parent.mkdir(parents=True,exist_ok=True)
            if not prefix.with_suffix('.json').exists():
                audio,duration=load_audio(ROOT/entry['audio'],seconds,config)
                x=features(audio,config)
                targets,conflicts=make_targets(reference_notes(ROOT/entry['annotation']),len(x),config,duration)
                np.save(prefix.with_suffix('.npy'),x);np.savez_compressed(prefix.with_suffix('.npz'),**targets)
                info={'id':entry['id'],'duration':duration,'frames':len(x),'conflict_string_frames':conflicts,
                      'features_sha256':sha256(prefix.with_suffix('.npy')),'targets_sha256':sha256(prefix.with_suffix('.npz'))}
                prefix.with_suffix('.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
            info=json.loads(prefix.with_suffix('.json').read_text())
            if sha256(prefix.with_suffix('.npy'))!=info['features_sha256'] or sha256(prefix.with_suffix('.npz'))!=info['targets_sha256']:
                raise ValueError('Causal feature cache changed.')
            audit['splits'][split].append(info)
            print('Causal cache',split,entry['id'],info['frames'],'frames',flush=True)
    total=0;sums=np.zeros(config['feature']['n_bins'],np.float64);squares=sums.copy()
    onset_sum=offset_sum=pitch_sum=0.;boundary_count=pitch_count=0
    for entry in manifest['train']:
        prefix=cache_path(entry,config,config['train_audio_seconds'])
        x=np.load(prefix.with_suffix('.npy'),mmap_mode='r').astype(np.float32)
        sums+=x.sum(axis=0,dtype=np.float64);squares+=np.square(x).sum(axis=0,dtype=np.float64);total+=len(x)
        with np.load(prefix.with_suffix('.npz')) as targets:
            onset_sum+=targets['onset'].sum(dtype=np.float64);offset_sum+=targets['offset'].sum(dtype=np.float64)
            pitch_sum+=targets['pitch'].sum(dtype=np.float64);boundary_count+=targets['onset'].size;pitch_count+=targets['pitch'].size
    mean=sums/total;std=np.sqrt(np.maximum(squares/total-mean**2,.09))
    audit['normalization']={'mean':mean.tolist(),'std':std.tolist()};audit['normalization_source']='train only'
    audit['positive_weights']={'onset':float(np.clip((boundary_count-onset_sum)/max(onset_sum,1),2,20)),
                               'offset':float(np.clip((boundary_count-offset_sum)/max(offset_sum,1),2,20)),
                               'pitch':float(np.clip((pitch_count-pitch_sum)/max(pitch_sum,1),1,8))}
    path=ROOT/'data/streaming/cache_audit.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(audit,indent=2),encoding='utf-8');return manifest,audit


class CropDataset:
    def __init__(self,entries,config,normalization):
        self.entries,self.config,self.epoch=entries,config,0
        self.mean=np.asarray(normalization['mean'],np.float32);self.std=np.asarray(normalization['std'],np.float32)

    def __len__(self):return self.config['training']['samples_per_epoch']

    def __getitem__(self,index):
        import torch
        config=self.config;train=config['training'];rng=np.random.default_rng(config['seed']+self.epoch*100000+index)
        entry=self.entries[int(rng.integers(len(self.entries)))];prefix=cache_path(entry,config,config['train_audio_seconds'])
        source=np.load(prefix.with_suffix('.npy'),mmap_mode='r')
        length=round(train['crop_seconds']*config['feature']['sample_rate']/config['feature']['hop_length'])
        start=int(rng.integers(len(source)-length+1));x=source[start:start+length].astype(np.float32)
        x+=rng.normal(0,train['gain_augmentation_log_std']);x=np.clip((x-self.mean)/self.std,-5,5)
        with np.load(prefix.with_suffix('.npz')) as all_targets:
            targets={k:torch.from_numpy(all_targets[k][start:start+length].copy()).long() if k=='state'
                     else torch.from_numpy(all_targets[k][start:start+length].astype(np.float32)) for k in all_targets.files}
        mask=np.ones(length,np.float32);edge=train['edge_mask_frames']
        if edge:mask[:edge]=mask[-edge:]=0
        targets['mask']=torch.from_numpy(mask);return torch.from_numpy(x),targets
