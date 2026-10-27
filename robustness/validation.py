import json

import numpy as np

from baseline.download import ROOT
from baseline.evaluation import evaluate,reference_notes
from cnn_gru.data import cache_path
from cnn_gru.inference import decode,predict_probabilities,tablature_metrics
from .data import effect_cache


def validate(model,manifest,audit,config,evaluation_config,device):
    predictions=[]
    for entry in manifest['validation']:
        base=cache_path(entry,config,config['evaluation_audio_seconds'])
        duration=json.loads(base.with_suffix('.json').read_text())['duration']
        reference=reference_notes(ROOT/entry['annotation'])
        for view in config['robustness']['validation_conditions']:
            prefix=base if view=='clean' else effect_cache(entry,config,'validation',view)
            x=np.load(prefix.with_suffix('.npy'),mmap_mode='r')
            probability=predict_probabilities(model,x,audit['normalization'],config,device)
            predictions.append((view,probability,reference,duration,base))
    best=None
    fps=config['feature']['sample_rate']/config['feature']['hop_length']
    for onset in config['decoder']['onset_thresholds']:
        for activity in config['decoder']['activity_thresholds']:
            parameters={'onset_threshold':onset,'activity_threshold':activity}
            rows=[];condition_scores={name:[] for name in config['robustness']['validation_conditions']}
            for view,probability,reference,duration,base in predictions:
                notes=decode(probability,config,parameters)
                result=evaluate(reference,notes,evaluation_config,duration)
                with np.load(base.with_suffix('.npz')) as target:
                    tab=tablature_metrics(reference,notes,evaluation_config,evaluation_config['score_seconds'],duration,probability,target,fps)
                rows.append((result,tab));condition_scores[view].append(result['onset']['f1'])
            candidate={'onset_f1':float(np.mean([row[0]['onset']['f1'] for row in rows])),
                       'onset_offset_f1':float(np.mean([row[0]['onset_offset']['f1'] for row in rows])),
                       'tab_onset_f1':float(np.mean([row[1]['onset']['f1'] for row in rows])),
                       'tab_frame_f1':float(np.mean([row[1]['frame']['f1'] for row in rows])),
                       'condition_onset_f1':{name:float(np.mean(values)) for name,values in condition_scores.items()},
                       'parameters':parameters}
            if best is None or (candidate['onset_f1'],candidate['onset_offset_f1'])>(best['onset_f1'],best['onset_offset_f1']): best=candidate
    return best
