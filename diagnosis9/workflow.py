import itertools
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from baseline.download import ROOT,sha256
from baseline.evaluation import evaluate,reference_notes
from cnn_gru.data import load_audio
from cnn_gru.inference import tablature_metrics
from streaming.data import cache_path
from streaming.evaluate import aggregate,load_checkpoint,save_json
from streaming.export import export
from streaming.latency import measure
from .decoder import SmoothedDecoder
from .replay import replay


def production_hashes():
    return {str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in
            [ROOT/'diagnosis9/decoder.py',ROOT/'diagnosis9/replay.py',ROOT/'diagnosis9/workflow.py',ROOT/'configs/diagnosis9.json']}


def load_inputs(config,output):
    output=Path(output).resolve();audit=json.loads((output/'validation-head-diagnostics.json').read_text())
    checkpoint_path=ROOT/config['checkpoint_root']/'model/best-inference.pt'
    if sha256(checkpoint_path)!=audit['checkpoint_sha256']:raise ValueError('Frozen causal checkpoint changed.')
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    for name,digest in checkpoint['source_sha256'].items():
        if sha256(ROOT/'streaming'/name)!=digest:raise ValueError('Frozen causal source changed.')
    manifest_path=ROOT/'data/cnn_gru/manifest.json';manifest=json.loads(manifest_path.read_text())
    if sha256(manifest_path)!=audit['manifest_sha256'] or sha256(manifest_path)!=checkpoint['manifest_sha256']:
        raise ValueError('Validation source split changed.')
    if config['baseline_parameters']!=checkpoint['decoder_parameters']:raise ValueError('Strict baseline parameters differ.')
    row_keys=[r['id'] for r in audit['clips']]
    if len(row_keys)!=len(set(row_keys)) or set(row_keys)!={e['id'] for e in manifest['validation']}:
        raise ValueError('Incomplete or duplicate validation head cache.')
    rows={r['id']:r for r in audit['clips']};dataset=[]
    for entry in manifest['validation']:
        for field in ['audio','annotation']:
            if sha256(ROOT/entry[field])!=entry[field+'_sha256']:raise ValueError('Original validation source changed.')
        row=rows[entry['id']];probability_path=ROOT/row['probability_path']
        if sha256(probability_path)!=row['probability_sha256']:raise ValueError('Validation probabilities changed.')
        prefix=cache_path(entry,checkpoint['config'],config['input_seconds'])
        if sha256(prefix.with_suffix('.npy'))!=row['features_sha256'] or sha256(prefix.with_suffix('.npz'))!=row['targets_sha256']:
            raise ValueError('Validation feature/target cache changed.')
        with np.load(probability_path) as saved:probability={k:saved[k].copy() for k in saved.files}
        dataset.append((entry,probability,reference_notes(ROOT/entry['annotation']),row['duration']))
    return checkpoint_path,checkpoint,manifest,dataset


def decode(probability,model_config,parameters,duration,packet_ms):
    decoder=SmoothedDecoder(model_config,parameters);fps=model_config['feature']['sample_rate']/model_config['feature']['hop_length']
    size=max(1,round(packet_ms*fps/1000));length=len(probability['state'])
    for start in range(0,length,size):
        end=min(start+size,length);decoder.push({k:v[start:end] for k,v in probability.items()},end/fps)
    decoder.finish(duration,duration);return decoder.notes,decoder.events


def select(config,output):
    output=Path(output).resolve();path,checkpoint,manifest,dataset=load_inputs(config,output)
    score=checkpoint['scoring_config'];candidates=[];best=None
    grid=config['decoder_grid']
    for alpha,confirm,backdate in itertools.product(grid['state_alpha'],grid['confirm_frames'],grid['onset_backdate_seconds']):
        parameters={**config['baseline_parameters'],'state_alpha':alpha,'confirm_frames':confirm,'onset_backdate_seconds':backdate}
        metrics=[];rows=[]
        for entry,probability,reference,duration in dataset:
            notes,events=decode(probability,checkpoint['config'],parameters,duration,config['packet_ms'])
            pitch=evaluate(reference,notes,score,duration)
            tab=tablature_metrics(reference,notes,score,config['score_seconds'],duration)
            metrics.append(pitch);rows.append({'id':entry['id'],'onset_f1':pitch['onset']['f1'],
                                               'onset_offset_f1':pitch['onset_offset']['f1'],'tab_onset_f1':tab['onset']['f1'],
                                               'predicted_notes':pitch['onset']['predicted_count'],'missed_notes':pitch['errors']['missed_notes'],
                                               'extra_notes':pitch['errors']['extra_notes']})
        candidate={'parameters':parameters,'onset_f1':float(np.mean([m['onset']['f1'] for m in metrics])),
                   'onset_offset_f1':float(np.mean([m['onset_offset']['f1'] for m in metrics])),
                   'frame_f1':float(np.mean([m['frame']['f1'] for m in metrics])),'clips':rows}
        candidates.append(candidate)
        if best is None or (candidate['onset_f1'],candidate['onset_offset_f1'])>(best['onset_f1'],best['onset_offset_f1']):best=candidate
        print('Validation smoothing',alpha,confirm,backdate,'onsetF1',round(candidate['onset_f1'],3),flush=True)
    identity=next(c for c in candidates if c['parameters']['state_alpha']==1 and c['parameters']['confirm_frames']==4 and c['parameters']['onset_backdate_seconds']==.06)
    selection={'config':config,'selected_parameters':best['parameters'],'selected_validation':best,'strict_validation':identity,
               'candidates':candidates,'checkpoint_sha256':sha256(path),'manifest_sha256':sha256(ROOT/'data/cnn_gru/manifest.json'),
               'head_diagnostics_sha256':sha256(output/'validation-head-diagnostics.json'),
               'selection_rule':config['selection_rule'],'selected_from_test':False,'optimizer_updates':0,
               'production_source_sha256':production_hashes()}
    save_json(output/'selection.json',selection)
    print('Selected causal decoder ablation:',best['parameters'],'validation onset',best['onset_f1'],flush=True)
    return selection


def evaluate_selected(config,output,device):
    output=Path(output).resolve();selection=json.loads((output/'selection.json').read_text())
    if selection['config']!=config or selection['production_source_sha256']!=production_hashes():
        raise ValueError('Decoder/config changed after validation selection.')
    path,checkpoint,manifest,_=load_inputs(config,output)
    if selection['checkpoint_sha256']!=sha256(path):raise ValueError('Selected checkpoint changed.')
    _,_,model=load_checkpoint(ROOT/config['checkpoint_root'],device)
    parameters={'strict':config['baseline_parameters'],'repair':selection['selected_parameters']}
    warm,_=load_audio(ROOT/manifest['validation'][0]['audio'],1,checkpoint['config'])
    for mode in ['strict','repair']:replay(model,warm,checkpoint['normalization'],checkpoint['config'],parameters[mode],device,config['packet_ms'],mode)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    records=[];directory=output/'evaluation';(directory/'clips').mkdir(parents=True,exist_ok=True)
    for index,entry in enumerate(manifest['test']):
        for field in ['audio','annotation']:
            if sha256(ROOT/entry[field])!=entry[field+'_sha256']:raise ValueError('Reused diagnostic test input changed.')
        audio,duration=load_audio(ROOT/entry['audio'],config['input_seconds'],checkpoint['config'])
        relative=Path('evaluation/clips')/(entry['id']+'.wav');sf.write(output/relative,audio,checkpoint['config']['feature']['sample_rate'],subtype='FLOAT')
        reference=reference_notes(ROOT/entry['annotation'])
        # Alternate which decoder is first without changing either fixed recipe.
        for mode in (['strict','repair'] if index%2==0 else ['repair','strict']):
            notes,events,timing=replay(model,audio,checkpoint['normalization'],checkpoint['config'],parameters[mode],device,config['packet_ms'],mode)
            pitch=evaluate(reference,notes,checkpoint['scoring_config'],duration)
            tab=tablature_metrics(reference,notes,checkpoint['scoring_config'],config['score_seconds'],duration)
            latency=measure(reference,notes,checkpoint['scoring_config'],duration)
            record={'id':entry['id'],'mode':mode,'notes':notes,'reference':reference,'events':events,'pitch_metrics':pitch,
                    'tab_metrics':tab,'latency':latency,'timing':timing,'relativeaudio':relative.as_posix()}
            stem=directory/'results'/mode/entry['id'];export(notes,stem,{'mode':mode,'checkpoint_sha256':sha256(path),'decoder_parameters':parameters[mode]})
            save_json(stem.with_name(stem.name+'-metrics.json'),record);save_json(stem.with_name(stem.name+'-events.json'),{'mode':mode,'events':events})
            records.append(record);print('Diagnostic test',entry['id'],mode,'onsetF1',round(pitch['onset']['f1'],3),flush=True)
    summary=aggregate(records);save_json(directory/'summary.json',summary)
    head=json.loads((output/'validation-head-diagnostics.json').read_text())
    oracle=json.loads((output/'oracle-diagnostics.json').read_text())
    hybrid=json.loads((output/'hybrid-head-diagnostics.json').read_text())
    run={'config':config,'selection':{'parameters':selection['selected_parameters'],
             'validation_onset_f1':selection['selected_validation']['onset_f1'],'strict_validation_onset_f1':selection['strict_validation']['onset_f1']},
         'checkpoint_sha256':sha256(path),'optimizer_updates':0,'new_training_epochs':0,'weights_changed':False,
         'same_features_normalization_model':True,'device':str(device),'peak_gpu_mib':torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
         'head_diagnostics_summary':head,'oracle_summary':{'summary':oracle['summary'],'protocol':oracle['protocol']},
         'hybrid_summary':hybrid.get('summary'),
         'reused_test_set':True,'test_used_for_selection':False,'scoring_config':checkpoint['scoring_config'],
         'selection_sha256':sha256(output/'selection.json'),'source_sha256':{p.name:sha256(p) for p in sorted((ROOT/'diagnosis9').glob('*.py'))},
         'diagnostic_artifacts_sha256':{p.name:sha256(p) for p in [output/'validation-head-diagnostics.json',output/'oracle-diagnostics.json',output/'hybrid-head-diagnostics.json']},
         'scope':'Reused progression-3 controlled diagnostic. Validation-only decoder selection; no new generalization or microphone claim.'}
    save_json(directory/'run.json',run)
    from .report import write_report
    write_report(output,records,summary,run);return summary
