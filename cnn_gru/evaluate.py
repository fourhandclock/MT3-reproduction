import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from baseline.__main__ import environment
from baseline.download import ROOT,sha256
from baseline.evaluation import evaluate,reference_notes
from .data import cache_path,features,load_audio,validate_splits
from .inference import decode,predict_probabilities,read_string_midi,save_midi,tablature_metrics
from .model import GuitarCNNGRU


def evaluate_checkpoint(config,output,device):
    output=Path(output).resolve()
    checkpoint_path=output/'best-inference.pt' if (output/'best-inference.pt').exists() else output/'best.pt'
    checkpoint=torch.load(checkpoint_path,map_location=device,weights_only=True)
    if checkpoint['config']!=config: raise ValueError('Evaluation config differs from selected checkpoint.')
    manifest_path=ROOT/'data/cnn_gru/manifest.json'
    if checkpoint['manifest_sha256']!=sha256(manifest_path): raise ValueError('Data manifest changed since training.')
    for name,digest in checkpoint['source_sha256'].items():
        if sha256(ROOT/'cnn_gru'/name)!=digest: raise ValueError(f'Evaluated source differs from training: {name}')
    manifest=json.loads(manifest_path.read_text());validate_splits(manifest)
    evaluation_config=json.loads((ROOT/'configs/stage1.json').read_text())
    audit=json.loads((ROOT/'data/cnn_gru/cache_audit.json').read_text())
    model=GuitarCNNGRU(config).to(device);model.load_state_dict(checkpoint['model']);model.eval()
    parameters=checkpoint['decoder_parameters']
    destination=output/'evaluation';destination.mkdir(exist_ok=True)
    (destination/'clips').mkdir(exist_ok=True)
    (destination/'cnn-gru').mkdir(exist_ok=True)
    first=manifest['test'][0]
    warm_audio,_=load_audio(ROOT/first['audio'],2,config)
    warm_features=features(warm_audio,config)
    predict_probabilities(model,warm_features,checkpoint['normalization'],config,device)
    if device.type=='cuda': torch.cuda.synchronize()
    rows=[]
    fps=config['feature']['sample_rate']/config['feature']['hop_length']
    for index,entry in enumerate(manifest['test']):
        if sha256(ROOT/entry['audio'])!=entry['audio_sha256']: raise IOError('Audio checksum mismatch.')
        if sha256(ROOT/entry['annotation'])!=entry['annotation_sha256']: raise IOError('Annotation checksum mismatch.')
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        started=time.perf_counter()
        audio,duration=load_audio(ROOT/entry['audio'],config['evaluation_audio_seconds'],config)
        x=features(audio,config)
        probabilities=predict_probabilities(model,x,checkpoint['normalization'],config,device)
        predicted=decode(probabilities,config,parameters)
        midi_path=destination/'cnn-gru'/(entry['id']+'.mid')
        save_midi(predicted,midi_path)
        if device.type=='cuda': torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        predicted=read_string_midi(midi_path)
        reference=reference_notes(ROOT/entry['annotation'])
        result=evaluate(reference,predicted,evaluation_config,duration)
        prefix=cache_path(entry,config,config['evaluation_audio_seconds'])
        with np.load(prefix.with_suffix('.npz')) as target:
            tab=tablature_metrics(reference,predicted,evaluation_config,evaluation_config['score_seconds'],duration,
                                  probabilities,target,fps)
        result.update({'id':entry['id'],'model':'cnn-gru','tablature':tab,'processing_seconds':elapsed,
                       'input_duration_seconds':duration,'real_time_factor':elapsed/duration,'drum_notes':0,
                       'gpu_peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None})
        (destination/'cnn-gru'/(entry['id']+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
        np.savez_compressed(destination/'cnn-gru'/(entry['id']+'.npz'),**probabilities)
        with (destination/'cnn-gru'/(entry['id']+'-tab.csv')).open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=['start','end','string','fret','pitch','program'])
            writer.writeheader();writer.writerows(predicted)
        shutil.copyfile(ROOT/'outputs/stage1/clips'/(entry['id']+'.wav'),destination/'clips'/(entry['id']+'.wav'))
        rows.append(result)
        print(f'CNN-GRU {index+1}/12 {entry["id"]}: onset F1={result["onset"]["f1"]:.3f}, '
              f'tab onset F1={tab["onset"]["f1"]:.3f}, {elapsed:.2f}s',flush=True)
    run={'selected_ids':[entry['id'] for entry in manifest['test']], 'config':evaluation_config,
         'model_config':config,'checkpoint_sha256':sha256(checkpoint_path),'checkpoint_file':checkpoint_path.name,
         'selected_epoch':checkpoint['history'][-1]['epoch'],
         'trained_optimizer_steps':checkpoint['step'],'decoder_parameters':parameters,'device':str(device),
         'environment':environment(),'normalization_source':'train only','selection_source':'progression-2 validation only',
         'timing_scope':'Warm file read, resampling, fresh CQT, normalization, network, decoding and MIDI write; no cached test features used for timing.',
         'source_sha256':checkpoint['source_sha256'],
         'training_source_sha256':checkpoint.get('training_source_sha256',checkpoint['source_sha256']),
         'postprocessing_note':checkpoint.get('postprocessing_note'),
         'evaluation_source_sha256':sha256(ROOT/'cnn_gru/evaluate.py')}
    (destination/'run.json').write_text(json.dumps(run,indent=2),encoding='utf-8')
    from .report import write_report
    write_report(output,rows,run)
