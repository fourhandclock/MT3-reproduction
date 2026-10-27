import json
import shutil
import time
from pathlib import Path

import numpy as np

from baseline.download import ROOT,sha256
from baseline.evaluation import evaluate,reference_notes
from cnn_gru.data import cache_path,validate_splits
from cnn_gru.inference import predict_probabilities,read_string_midi,tablature_metrics
from .decoder import lowest_fret,solve
from .export import export
from .metrics import playability,preserved


def note_events(notes):
    return [{key:note[key] for key in ['pitch','start','end']} for note in notes]


def select(config,output,device):
    import torch
    from cnn_gru.model import GuitarCNNGRU
    manifest=json.loads((ROOT/'data/cnn_gru/manifest.json').read_text());validate_splits(manifest)
    model_config=json.loads((ROOT/'configs/cnn_gru.json').read_text())
    score_config=json.loads((ROOT/'configs/stage1.json').read_text())
    checkpoint_path=ROOT/'outputs/cnn_gru/best-inference.pt'
    checkpoint=torch.load(checkpoint_path,map_location=device,weights_only=True)
    if checkpoint['config']!=model_config or checkpoint['manifest_sha256']!=sha256(ROOT/'data/cnn_gru/manifest.json'):
        raise ValueError('Acoustic-model configuration or dataset changed.')
    checkpoint_digest=sha256(checkpoint_path)
    for name,digest in checkpoint['source_sha256'].items():
        if sha256(ROOT/'cnn_gru'/name)!=digest: raise ValueError('Frozen acoustic model source changed.')
    model=GuitarCNNGRU(model_config).to(device);model.load_state_dict(checkpoint['model']);model.eval()
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    cache=output/'validation_probabilities';cache.mkdir(exist_ok=True)
    dataset=[]
    for entry in manifest['validation']:
        prefix=cache_path(entry,model_config,model_config['evaluation_audio_seconds'])
        probability_path=cache/(entry['id']+'-'+checkpoint_digest[:16]+'.npz')
        if not probability_path.exists():
            x=np.load(prefix.with_suffix('.npy'),mmap_mode='r')
            probability=predict_probabilities(model,x,checkpoint['normalization'],model_config,device)
            np.savez_compressed(probability_path,**probability)
        probability={key:value for key,value in np.load(probability_path).items()}
        reference=[note for note in reference_notes(ROOT/entry['annotation']) if note['start']<config['input_seconds']]
        dataset.append((entry['id'],reference,note_events(reference),probability))
    grid=[]
    for weight in config['acoustic_weights']:
        rows=[]
        for clip_id,reference,notes,probability in dataset:
            started=time.perf_counter();decoded,diagnostic=solve(notes,config,probability,weight)
            if not preserved(notes,decoded): raise ValueError('Fingering changed a pitch or interval.')
            tab=tablature_metrics(reference,decoded,score_config,config['score_seconds'],config['input_seconds'])
            constraints=playability(decoded,config)
            rows.append({'id':clip_id,'string_onset_f1':tab['onset']['f1'],'assigned_fraction':constraints['assignment_fraction'],
                         'hand_travel':constraints['hand_position_travel_frets'],'seconds':time.perf_counter()-started,
                         'unassigned':diagnostic['unassigned']})
        result={'acoustic_weight':weight,'mean_string_onset_f1':float(np.mean([row['string_onset_f1'] for row in rows])),
                'mean_assignment_fraction':float(np.mean([row['assigned_fraction'] for row in rows])),
                'mean_hand_travel':float(np.mean([row['hand_travel'] for row in rows])),'excerpts':rows}
        grid.append(result)
        print('Validation fingering weight',weight,'string F1',round(result['mean_string_onset_f1'],3),
              'coverage',round(result['mean_assignment_fraction'],3),flush=True)
    best=max(grid,key=lambda row:(row['mean_string_onset_f1'],row['mean_assignment_fraction'],-row['mean_hand_travel']))
    selection={'config':config,'acoustic_weight':best['acoustic_weight'],'validation_grid':grid,
               'selection_rule':'Reference-note string-aware onset F1 on progression-2 validation, then coverage and hand-travel tie breaks.',
               'checkpoint_sha256':sha256(checkpoint_path),'manifest_sha256':sha256(ROOT/'data/cnn_gru/manifest.json'),
               'selected_from_test':False,'optimizer_updates':0,
               'source_sha256':{name:sha256(ROOT/'fingering'/name) for name in ['decoder.py','metrics.py','workflow.py']}}
    (output/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    print('Selected fingering weight:',best['acoustic_weight'],flush=True)


def aggregate(records):
    result={}
    for scenario in ['oracle','predicted']:
        result[scenario]={}
        subset=[record for record in records if record['scenario']==scenario]
        for mode in sorted({record['mode'] for record in subset}):
            rows=[record for record in subset if record['mode']==mode]
            result[scenario][mode]={'count':len(rows),
                'mean_pitch_onset_f1':float(np.mean([row['pitch_metrics']['onset']['f1'] for row in rows])),
                'mean_string_onset_f1':float(np.mean([row['tab_metrics']['onset']['f1'] for row in rows])),
                'mean_string_onset_offset_f1':float(np.mean([row['tab_metrics']['onset_offset']['f1'] for row in rows])),
                'assignment_fraction':sum(row['playability']['assigned_count'] for row in rows)/max(1,sum(row['playability']['note_count'] for row in rows)),
                'unassigned_notes':sum(row['playability']['unassigned_count'] for row in rows),
                'same_string_overlap_events':sum(row['playability']['same_string_overlap_events'] for row in rows),
                'span_violation_events':sum(row['playability']['span_violation_events'] for row in rows),
                'finger_proxy_violation_events':sum(row['playability']['finger_proxy_violation_events'] for row in rows),
                'hand_position_travel_frets':sum(row['playability']['hand_position_travel_frets'] for row in rows),
                'solver_seconds':sum(row['solver_seconds'] for row in rows)}
    return result


def evaluate_fingering(config,output):
    output=Path(output).resolve();selection=json.loads((output/'selection.json').read_text())
    if selection['config']!=config: raise ValueError('Selection/configuration mismatch.')
    for name,digest in selection['source_sha256'].items():
        if sha256(ROOT/'fingering'/name)!=digest: raise ValueError('Decoder source changed after selection.')
    if sha256(ROOT/'outputs/cnn_gru/best-inference.pt')!=selection['checkpoint_sha256']: raise ValueError('Acoustic checkpoint changed.')
    manifest=json.loads((ROOT/'data/cnn_gru/manifest.json').read_text());validate_splits(manifest)
    if sha256(ROOT/'data/cnn_gru/manifest.json')!=selection['manifest_sha256']: raise ValueError('Manifest changed.')
    score_config=json.loads((ROOT/'configs/stage1.json').read_text())
    folder=output/'evaluation';folder.mkdir(exist_ok=True);(folder/'clips').mkdir(exist_ok=True)
    all_records=[]
    weight=selection['acoustic_weight']
    for entry in manifest['test']:
        reference=[note for note in reference_notes(ROOT/entry['annotation']) if note['start']<config['input_seconds']]
        raw=read_string_midi(ROOT/'outputs/cnn_gru/evaluation/cnn-gru'/(entry['id']+'.mid'))
        with np.load(ROOT/'outputs/cnn_gru/evaluation/cnn-gru'/(entry['id']+'.npz')) as saved:
            probability={key:saved[key].copy() for key in saved.files}
        for scenario,inputs,original_mode in [('oracle',note_events(reference),'reference_positions'),
                                              ('predicted',note_events(raw),'raw_cnn')]:
            variants=[(original_mode,reference if scenario=='oracle' else raw)]
            variants.append(('lowest_fret',lowest_fret(inputs,config)))
            for mode,acoustic_weight in [('constrained_pitch_only',0.0),('constrained_acoustic',weight)]:
                started=time.perf_counter();decoded,diagnostic=solve(inputs,config,probability,acoustic_weight)
                variants.append((mode,(decoded,time.perf_counter()-started,diagnostic)))
            for mode,value in variants:
                if isinstance(value,tuple): notes,seconds,diagnostic=value
                else: notes,seconds,diagnostic=value,0.0,{}
                if not preserved(inputs,notes): raise ValueError('Pitch or timing changed.')
                constraints=playability(notes,config)
                if mode.startswith('constrained') and any(constraints[key] for key in ['same_string_overlap_events','span_violation_events','finger_proxy_violation_events']):
                    raise ValueError('Search emitted an assigned note violating its constraints.')
                pitch=evaluate(reference,notes,score_config,config['input_seconds'])
                tab=tablature_metrics(reference,notes,score_config,config['score_seconds'],config['input_seconds'])
                record={'id':entry['id'],'scenario':scenario,'mode':mode,'notes':notes,'reference':reference,
                        'playability':constraints,'pitch_metrics':pitch,'tab_metrics':tab,'solver_seconds':seconds,
                        'solver_diagnostic':diagnostic,'pitch_timing_preserved':True}
                stem=folder/scenario/mode/entry['id']
                export(notes,stem,{'mode':mode,'scenario':scenario,'fretboard':config,
                                  'acoustic_weight':weight if mode=='constrained_acoustic' else 0,
                                  'pitch_timing_preserved':True,'unassigned':constraints['unassigned_count']})
                stem.with_name(stem.name+'-metrics.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
                all_records.append(record)
        shutil.copyfile(ROOT/'outputs/stage1/clips'/(entry['id']+'.wav'),folder/'clips'/(entry['id']+'.wav'))
        print('Fingering comparison completed:',entry['id'],flush=True)
    summary=aggregate(all_records)
    (folder/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    run={'selection':selection,'config':config,'input_recordings':len(manifest['test']),
         'optimizer_updates':0,'pitch_timing_preserved_in_all_modes':True,
         'source_sha256':{name:sha256(ROOT/'fingering'/name) for name in ['decoder.py','metrics.py','workflow.py','export.py']}}
    (folder/'run.json').write_text(json.dumps(run,indent=2),encoding='utf-8')
    from .report import write_report
    write_report(output,all_records,summary,run)
