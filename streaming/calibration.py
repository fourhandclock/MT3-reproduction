import json
from pathlib import Path

import numpy as np
import torch

from baseline.download import ROOT,sha256
from baseline.evaluation import evaluate,reference_notes
from cnn_gru.training import save_checkpoint
from .data import cache_path
from .inference import decode_probabilities,predict_features
from .model import CausalCNNGRU


def calibrate(output,device):
    """Freeze weights; select note timing/thresholds using validation recordings only."""
    output=Path(output).resolve();parent_path=output/'model/best.pt'
    parent=torch.load(parent_path,map_location=device,weights_only=True);config=parent['config']
    for name,digest in parent['source_sha256'].items():
        if name!='inference.py' and sha256(ROOT/'streaming'/name)!=digest:raise ValueError('Model/training source changed.')
    model=CausalCNNGRU(config).to(device);model.load_state_dict(parent['model']);model.eval()
    manifest=json.loads((ROOT/'data/cnn_gru/manifest.json').read_text())
    if sha256(ROOT/'data/cnn_gru/manifest.json')!=parent['manifest_sha256']:raise ValueError('Calibration split changed.')
    dataset=[]
    for entry in manifest['validation']:
        prefix=cache_path(entry,config,config['evaluation_audio_seconds'])
        x=np.load(prefix.with_suffix('.npy'),mmap_mode='r')
        probability=predict_features(model,x,parent['normalization'],config,device)
        dataset.append((probability,reference_notes(ROOT/entry['annotation']),json.loads(prefix.with_suffix('.json').read_text())['duration']))
    grid={'onset_threshold':[.25,.4,.55],'activity_threshold':[.25,.35,.45,.6],
          'onset_backdate_seconds':[0.,.06,.12,.18]}
    candidates=[];best=None
    for onset in grid['onset_threshold']:
        for activity in grid['activity_threshold']:
            for backdate in grid['onset_backdate_seconds']:
                parameters={'onset_threshold':onset,'activity_threshold':activity,'onset_backdate_seconds':backdate}
                rows=[]
                for probability,reference,duration in dataset:
                    notes,_=decode_probabilities(probability,config,parameters,duration)
                    rows.append(evaluate(reference,notes,parent['scoring_config'],duration))
                candidate={'parameters':parameters,'onset_f1':float(np.mean([r['onset']['f1'] for r in rows])),
                           'onset_offset_f1':float(np.mean([r['onset_offset']['f1'] for r in rows])),
                           'frame_f1':float(np.mean([r['frame']['f1'] for r in rows]))}
                candidates.append(candidate)
                if best is None or (candidate['onset_f1'],candidate['onset_offset_f1'])>(best['onset_f1'],best['onset_offset_f1']):best=candidate
            print('Validation calibration',onset,activity,'best onset F1',round(best['onset_f1'],3),flush=True)
    selection={'parameters':best['parameters'],'validation':best,'grid':grid,'candidates':candidates,
               'parent_checkpoint_sha256':sha256(parent_path),'model_optimizer_updates':parent['step'],
               'additional_optimizer_updates':0,'used_test_recordings':False,
               'source_time_delivery_separate':True,'expanded_after_original_validation':True,
               'reason':'Validate detector thresholds and source-time estimates for the new causal frontend; expanding source backdate never changes emission time.'}
    (output/'model/decoder-calibration.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    derived={key:value for key,value in parent.items() if key not in ['optimizer','scheduler','scaler','rng_cpu','rng_cuda']}
    derived['decoder_parameters']=best['parameters'];derived['inference_calibration']=selection
    derived['training_source_sha256']=dict(parent['source_sha256'])
    derived['source_sha256']={name:sha256(ROOT/'streaming'/name) for name in parent['source_sha256']}
    derived['inference_source_updates']='Event-copy isolation, empty input and all-packet backlog bookkeeping; feature/model/decoder/loss weights unchanged.'
    save_checkpoint(output/'model/best-inference.pt',derived)
    print('Frozen causal model calibrated:',best,flush=True);return selection
