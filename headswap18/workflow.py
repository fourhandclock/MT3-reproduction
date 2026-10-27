"""Four fixed head combinations with no training, neural inference or clock invention."""

import csv
import json
import shutil
from pathlib import Path
import numpy as np
from baseline.download import ROOT,sha256
from decoder14.core import replay,score_case,without_compute,summarize
from decoder14.workflow import read
from diagnosis15.analysis import taxonomy,add_note_evidence,CATEGORIES
from streaming.export import export
from .data import MODELS,COMBINATIONS,validate_config,load_inputs,cross_heads
from .analysis import state_diagnosis,summarize_frames,contrasts,note_changes


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def sources():
    paths=[ROOT/'headswap18'/name for name in ['__init__.py','__main__.py','data.py','analysis.py','workflow.py']]
    paths+=[ROOT/name for name in ['configs/headswap18.json','baseline/download.py','baseline/evaluation.py',
        'decoder14/core.py','decoder14/workflow.py','diagnosis15/analysis.py','diagnosis9/head_audit.py',
        'diagnosis9/oracles.py','streaming/targets.py','streaming/decoder.py','streaming/export.py',
        'acoustic10/decoder.py','acoustic10/targets.py','cnn_gru/inference.py','fingering/export.py']]
    return {p.relative_to(ROOT).as_posix():sha256(p) for p in paths}


def make_case(item,mode,config,base,score,arrays):
    state_source,onset_source=COMBINATIONS[mode]
    probability=cross_heads(item['probabilities'],state_source,onset_source)
    notes,events,clamps=replay(probability,base,config['decoder_parameters'],12.,.13,20)
    metrics=score_case(item['reference'],notes,score,12.)
    original=state_source==onset_source
    if original:
        prior=item['prior'][state_source]
        if (notes!=without_compute(prior['notes']) or events!=without_compute(prior['events'])
                or metrics['pitch_metrics']!=without_compute(prior['pitch_metrics'])
                or metrics['tab_metrics']!=without_compute(prior['tab_metrics'])):
            raise ValueError('Original source decisions/scores do not reproduce Stage17 exactly')
    classified=add_note_evidence(taxonomy(item['reference'],notes,score,config),arrays[state_source])
    return {'id':item['id'],'mode':mode,'state_source':state_source,'onset_source':onset_source,
        'head_provenance':{key:state_source if key!='onset' else onset_source for key in probability},
        'source_probability_sha256':{m:item['prior'][m]['probability_sha256'] for m in set([state_source,onset_source])},
        'original_combination':original,'original_source_reproduced':True if original else None,
        'notes':notes,'events':events,'source_clamps':clamps,'reference':item['reference'],**metrics,
        'taxonomy':classified,'state_diagnosis_path':f'validation/frames/{state_source}/{item["id"]}.json',
        'relativeaudio':f'validation/clips/{item["id"]}.wav',
        'clock_scope':'Estimated source and known packet availability only; no measured processing completion or speed.',
        'privileged_label_substitution':False}


def write_evidence(path,result):
    headers=['reference_index','start','end','pitch','string','fret','category','prediction_index','onset_error_ms',
             'valid_source_frames','same_string_fret_fraction','any_string_pitch_fraction','auxiliary_pitch_fraction',
             'longest_correct_state_run_frames']
    with Path(path).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=headers);writer.writeheader()
        for row in result['reference_rows']:
            values={**row,**row['note'],**row['evidence']};writer.writerow({k:values.get(k) for k in headers})


def summarize_cases(cases):
    result={}
    for mode in COMBINATIONS:
        rows=[r for r in cases if r['mode']==mode]
        result[mode]={**summarize(rows),
            'frame_f1':float(np.mean([r['pitch_metrics']['frame']['f1'] for r in rows])),
            'string_onset_f1':float(np.mean([r['tab_metrics']['onset']['f1'] for r in rows])),
            'official_extra_notes':sum(r['pitch_metrics']['errors']['extra_notes'] for r in rows),
            'reference_categories':{c:sum(r['taxonomy']['reference_counts'][c] for r in rows) for c in CATEGORIES},
            'unpaired_after_associations':sum(r['taxonomy']['extra_predictions'] for r in rows),
            'count':len(rows),'state_source':COMBINATIONS[mode][0],'onset_source':COMBINATIONS[mode][1]}
    return result


def aggregate(cases,frames):
    modes=summarize_cases(cases)
    lookup={(r['id'],r['mode']):r for r in cases}
    changes=[]
    for clip in [r['id'] for r in cases if r['mode']=='control']:
        changes.extend({'id':clip,**r} for r in note_changes(lookup[(clip,'control')],lookup[(clip,'fused')]))
    counts={k:sum(r['change']==k for r in changes) for k in ['matched_both','fused_only','control_only','missed_both']}
    return {'modes':modes,'frames':summarize_frames(frames),'contrasts':contrasts(modes),
            'original_note_changes':counts,'original_note_change_rows':changes,
            'new_optimizer_updates':0,'neural_inference_calls':0,'test_inference_calls':0,
            'production_promoted':False,'model_selection_performed':False}


def run(config,output):
    validate_config(config);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol.json').exists():
        raise ValueError('Stage18 already exists here; reuse report or audit')
    base,score,inputs,bindings=load_inputs(config)
    protocol={'config':config,'source_sha256':sources(),'input_sha256':bindings,
        'decoder_config':base,'scoring_config':score,'combinations':{k:list(v) for k,v in COMBINATIONS.items()},
        'new_optimizer_updates':0,'neural_inference_calls':0,'test_inference_calls':0,'production_promoted':False,
        'diagonal_first':True,'model_selection_performed':False,
        'policy_rule':'Argmax is the Stage17 selection policy; decoder_activity uses the fixed best-active/(best-active+inactive+1e-8) >=0.25 gate. Report both; do not replace the prior selection.',
        'frame_rule':'Original source targets0:1000 versus recorded predictions13:1013; exclude first26 source frames and per-string conflicts. Pitch uses the unchanged Stage17 support.',
        'note_rule':'Score onsets in original0:10s, with original input through12s. Frame leading masks do not trim note metrics. Nearby associations are descriptive, exclusive and one-to-one.',
        'clock_rule':config['clock_rule']}
    save(output/'protocol.json',protocol)
    frames=[];arrays={}
    for item in inputs:
        arrays[item['id']]={}
        for model in MODELS:
            diagnosis,aligned=state_diagnosis(item['probabilities'][model],item['targets'],config)
            prior=item['prior'][model]
            if (diagnosis['policies']['argmax']['exact_state']!=prior['head_metrics']['exact_state']
                    or diagnosis['auxiliary_pitch']!=prior['head_metrics']['pitch']['overall']
                    or diagnosis['state_derived_pitch']!=prior['state_derived_pitch']):
                raise ValueError('Stage17 source-aligned raw-head metrics do not reproduce')
            row={'id':item['id'],'model':model,'diagnosis':diagnosis}
            save(output/'validation/frames'/model/(item['id']+'.json'),row);frames.append(row)
            arrays[item['id']][model]=aligned
        audio=output/'validation/clips'/(item['id']+'.wav');audio.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(item['audio'],audio)
    cases=[]
    # Complete and verify all24 diagonal source cases before any crossed replay.
    for mode in COMBINATIONS:
        for item in inputs:
            row=make_case(item,mode,config,base,score,arrays[item['id']])
            stem=output/'validation/results'/mode/item['id']
            export(row['notes'],stem,{'stage':18,'mode':mode,'state_source':row['state_source'],
                'onset_source':row['onset_source'],'new_optimizer_updates':0,'neural_inference_calls':0,
                'decoder_parameters':config['decoder_parameters'],'source_delay_seconds':.13,
                'clock_scope':row['clock_scope'],'privileged_label_substitution':False})
            save(stem.with_name(stem.name+'-metrics.json'),row)
            save(stem.with_name(stem.name+'-events.json'),{'events':row['events'],'clock_scope':row['clock_scope']})
            write_evidence(stem.with_name(stem.name+'-evidence.csv'),row['taxonomy']);cases.append(row)
        print('Stage18',mode,'replayed',len(inputs),'clips',flush=True)
    summary=aggregate(cases,frames);save(output/'summary.json',summary)
    for model in MODELS:
        for policy in ['argmax','decoder_activity']:
            path=output/f'confusion-{model}-{policy}.csv'
            with path.open('w',newline='',encoding='utf-8') as stream:
                writer=csv.writer(stream);writer.writerow(['reference/predicted']+list(range(25))+['inactive'])
                for index,row in enumerate(summary['frames'][model]['policies'][policy]['confusion']):
                    writer.writerow([index if index<25 else 'inactive']+row)
    save(output/'run.json',{'config':config,'source_sha256':sources(),'input_sha256':bindings,
        'protocol_sha256':sha256(output/'protocol.json'),'new_optimizer_updates':0,'neural_inference_calls':0,
        'test_inference_calls':0,'production_promoted':False,'model_selection_performed':False,
        'original_reproductions':24,'crossed_cases':24,'cases':48,'frame_cases':24,
        'case_order':list(COMBINATIONS),'scope':'Frozen cached probability replay, no new processing or microphone latency measurements'})
    return summary


def load_cases(output):
    return [read(p) for p in sorted((Path(output)/'validation/results').glob('*/*-metrics.json'))]
