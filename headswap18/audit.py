"""Recompute every diagnostic and check exact source exports, without a model."""

import csv
from collections import Counter
from pathlib import Path
import numpy as np
from baseline.download import ROOT,sha256
from decoder14.audit import midi_intervals
from decoder14.workflow import read
from diagnosis15.analysis import CATEGORIES
from .data import MODELS,COMBINATIONS,validate_config,load_inputs
from .analysis import state_diagnosis
from .workflow import sources,make_case,aggregate,load_cases


def require(value,message):
    if not value:raise ValueError(message)


def verify_frame_partition(diagnosis):
    for policy,row in diagnosis['policies'].items():
        counts=row['counts'];confusion=np.asarray(row['confusion'])
        require(confusion.shape==(26,26) and np.all(confusion>=0),'Invalid fret confusion')
        require(confusion.sum()==counts['valid_string_frames']==counts['reference_active_frames']+counts['reference_inactive_frames'],
                'Frame support does not conserve valid/conflict masks')
        require(sum(counts[k] for k in ['correct_fret','predicted_inactive','wrong_octave_same_string','other_wrong_fret'])
                ==counts['reference_active_frames'],'Active-frame categories do not conserve support')
        require(int(confusion[:25,:25].trace())==row['exact_state']['true_positives']==counts['correct_fret'],
                'Exact state true positives differ from confusion diagonal')
        require(int(confusion[:25,:].sum())==row['exact_state']['reference_count'] and
                int(confusion[:,:25].sum())==row['exact_state']['predicted_count'],'State metric support differs from confusion')
        require(int(confusion[:25,:25].sum())==row['activity']['true_positives'] and
                int(confusion[25,:25].sum())==counts['active_on_reference_inactive'],'Occupancy metric differs from confusion')
        conditional=row['fret_given_reference_and_prediction_active']
        require(conditional['count']==int(confusion[:25,:25].sum()) and conditional['correct']==counts['correct_fret'],
                'Conditional fret denominator changed')


def verify_note_partition(row):
    result=row['taxonomy'];refs,predictions=result['reference_rows'],result['prediction_rows']
    pairs=[(r['reference_index'],r['prediction_index']) for r in refs if r['prediction_index'] is not None]
    require([r['reference_index'] for r in refs]==list(range(len(refs))) and
            [r['prediction_index'] for r in predictions]==list(range(len(predictions))),'Reference/prediction identities changed')
    require(len({j for i,j in pairs})==len(pairs),'Prediction associated to more than one reference')
    require(sum(result['reference_counts'].values())==len(refs)==result['reference_count'] and
            set(result['reference_counts'])==set(CATEGORIES),'Reference taxonomy does not conserve support')
    require(len(pairs)+result['extra_predictions']==len(predictions)==result['prediction_count'],
            'Prediction taxonomy does not conserve support')
    for i,j in pairs:
        require(predictions[j]['reference_index']==i and predictions[j]['category']==refs[i]['category'],'Association is not bidirectional')
    require(result['matched_onsets']==row['pitch_metrics']['onset']['true_positives'],'Diagnostic association changed official matches')


def audit(output):
    output=Path(output);protocol=read(output/'protocol.json');config=protocol['config'];validate_config(config)
    require(protocol['source_sha256']==sources(),'Frozen Stage18 source changed')
    base,score,inputs,bindings=load_inputs(config)
    require(protocol['input_sha256']==bindings and protocol['decoder_config']==base and protocol['scoring_config']==score,
            'Input/decoder/scoring binding changed')
    require(protocol['new_optimizer_updates']==protocol['neural_inference_calls']==protocol['test_inference_calls']==0
            and not protocol['production_promoted'] and not protocol['model_selection_performed'],'Zero-budget scope changed')
    run=read(output/'run.json')
    require(run['source_sha256']==sources() and run['input_sha256']==bindings and
            run['protocol_sha256']==sha256(output/'protocol.json'),'Run source/input binding changed')
    require(run['new_optimizer_updates']==run['neural_inference_calls']==run['test_inference_calls']==0
            and not run['production_promoted'] and not run['model_selection_performed'],'Run budget/selection scope changed')
    require(run['case_order']==list(COMBINATIONS) and run['original_reproductions']==24 and run['crossed_cases']==24,'Diagonal-first scope changed')
    cases=load_cases(output);lookup={(r['id'],r['mode']):r for r in cases}
    require(len(lookup)==len(cases)==48,'Missing/duplicated cached replay')
    frames=[];ordered=[];notes_count=events_count=references_count=0
    for mode in COMBINATIONS:
        for item in inputs:
            arrays={}
            for model in MODELS:
                diagnosis,arrays[model]=state_diagnosis(item['probabilities'][model],item['targets'],config)
                frame={'id':item['id'],'model':model,'diagnosis':diagnosis}
                require(read(output/'validation/frames'/model/(item['id']+'.json'))==frame,'Saved frame diagnosis differs')
                verify_frame_partition(diagnosis)
                if mode=='control':frames.append(frame)
            expected=make_case(item,mode,config,base,score,arrays);row=lookup[(item['id'],mode)]
            require(expected==row,'Source decisions, head provenance, evidence or scoring differs')
            verify_note_partition(row)
            require(len(row['events'])==2*len(row['notes']),'Event ledger lost a boundary')
            require(all('delivered_at_seconds' not in e and e['emitted_at_audio_seconds']+1e-9>=e['source_time'] for e in row['events']),
                    'Measured completion invented or packet precedes source estimate')
            require(all('on_delivered_at_seconds' not in n and 'off_delivered_at_seconds' not in n for n in row['notes']),
                    'A parent completion clock was assigned to a synthetic hybrid')
            stem=output/'validation/results'/mode/item['id']
            exported=read(stem.with_suffix('.json'))
            require(exported['notes']==row['notes'] and read(stem.with_name(stem.name+'-events.json'))['events']==row['events'],
                    'Exact JSON/event export differs')
            with stem.with_suffix('.csv').open(newline='',encoding='utf-8') as stream:csv_rows=list(csv.DictReader(stream))
            require([(float(n['start']),float(n['end']),int(n['pitch']),int(n['string']),int(n['fret'])) for n in csv_rows]==
                    [(n['start'],n['end'],n['pitch'],n['string'],n['fret']) for n in row['notes']],'Note CSV differs')
            ticks=Counter((round(n['start']*960),max(round(n['start']*960)+1,round(n['end']*960)),n['pitch']) for n in row['notes'])
            require(midi_intervals(stem.with_suffix('.mid'))==ticks,'MIDI lost a published attack')
            with stem.with_name(stem.name+'-evidence.csv').open(newline='',encoding='utf-8') as stream:evidence=list(csv.DictReader(stream))
            require(len(evidence)==len(row['taxonomy']['reference_rows']),'Evidence CSV omitted a reference')
            for raw,saved in zip(evidence,row['taxonomy']['reference_rows']):
                flattened={**saved,**saved['note'],**saved['evidence']}
                require(all(text==('' if flattened[key] is None else str(flattened[key])) for key,text in raw.items()),'Evidence CSV differs')
            require(sha256(output/row['relativeaudio'])==sha256(item['audio']),'Copied audio differs')
            notes_count+=len(row['notes']);events_count+=len(row['events']);references_count+=row['taxonomy']['reference_count'];ordered.append(row)
    summary=aggregate(ordered,frames)
    require(summary==read(output/'summary.json'),'Summary aggregation differs')
    contrast=summary['contrasts'];total=contrast['total_fused_minus_control']
    require(abs(contrast['onset_swap_at_control_state']+contrast['state_swap_at_fused_onset']-total)<1e-12 and
            abs(contrast['state_swap_at_control_onset']+contrast['onset_swap_at_fused_state']-total)<1e-12,'Contrast paths do not conserve total difference')
    changes=summary['original_note_changes'];counts=summary['modes']
    require(sum(changes.values())==counts['control']['reference_notes']==counts['fused']['reference_notes'] and
            changes['matched_both']+changes['control_only']==counts['control']['matched_onsets'] and
            changes['matched_both']+changes['fused_only']==counts['fused']['matched_onsets'],'Reference note-change identities differ')
    for model in MODELS:
        for policy in ['argmax','decoder_activity']:
            with (output/f'confusion-{model}-{policy}.csv').open(newline='',encoding='utf-8') as stream:values=list(csv.reader(stream))
            require(len(values)==27 and np.array_equal(np.asarray([r[1:] for r in values[1:]],int),
                    summary['frames'][model]['policies'][policy]['confusion']),'Aggregate confusion CSV differs')
    return {'passed':True,'new_optimizer_updates':0,'neural_inference_calls':0,'test_inference_calls':0,
        'production_promoted':False,'model_selection_performed':False,'validation_cases':48,'original_reproductions':24,
        'crossed_cases':24,'frame_cases':24,'exact_notes':notes_count,'exact_events':events_count,
        'reference_evidence_rows':references_count,'frame_and_note_partitions_verified':True,
        'no_measured_completion_clock':True,'prior_stage17_bindings_verified':True,
        'artifact_sha256':{name:sha256(output/name) for name in ['protocol.json','run.json','summary.json']}}
