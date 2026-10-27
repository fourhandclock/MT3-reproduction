"""Reproduce step250-to500 restoration, active support, scores and exact exports."""

import csv
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from acoustic10.data import prepare_targets
from baseline.download import ROOT,sha256
from baseline.evaluation import evaluate,reference_notes
from cnn_gru.inference import tablature_metrics
from decoder14.audit import midi_intervals
from decoder14.core import replay,without_compute
from decoder14.workflow import read
from headswap18.analysis import state_diagnosis
from headswap18.audit import verify_frame_partition
from optimization13.audit import optimizer_counters
from optimization13.optimizer import recursive_digest,_validate_group
from spectral11.workflow import aligned_heads
from streaming.data import fingerprint
from streaming.latency import measure
from .data import load_inputs,validate_config
from .training import producer_hashes
from .workflow import MODES,frame_config,check_parent,summarize,choose


def require(value,message):
    if not value:raise ValueError(message)


def audit(output):
    output=Path(output);protocol=read(output/'protocol.json');config=protocol['config'];validate_config(config)
    require(protocol['source_sha256']==producer_hashes(),'Frozen Stage19 sources changed')
    base,cache,datasets,provenance,parent,expected=load_inputs(config,output)
    require(protocol['total_new_optimizer_update_cap']==500 and protocol['historical_unique_parent_updates']==250
            and protocol['new_sanity_updates']==0 and not protocol['test_inference_allowed'],'Budget/test scope changed')
    schedule=read(output/'crop-schedule.json')
    require(schedule['descriptors']==expected and schedule['sha256']==fingerprint(expected)
            and len(expected)==1000 and {r['epoch'] for r in expected}=={6,7,8,9,10},'Shared new exposure changed')
    targets=prepare_targets(datasets['train'],13,config,base);support=[]
    for row in expected:
        start,length=row['start_frame'],row['frames'];values=targets[row['record_index']]
        state=values['state'][start:start+length];mask=values['mask'][start:start+length].copy()
        mask[:26]=mask[-26:]=0;support.append(int(((mask>0)[:,None]&(state>=0)&(state<25)).sum()))
    support_epochs=[sum(support[i:i+200]) for i in range(0,1000,200)]
    training=read(output/'training-summary.json')
    require(training['new_optimizer_updates']==500 and training['historical_unique_parent_updates']==250
            and training['final_lifetime_updates_per_arm']==500 and training['new_sanity_updates']==0
            and training['test_inference_calls']==0 and not training['validation_used_during_training'],'Training accounting changed')
    require(training['source_sha256']==producer_hashes() and training['protocol_sha256']==sha256(output/'protocol.json'),'Training source binding changed')
    parent_states={k:recursive_digest(parent[k]) for k in ['model','optimizer','scaler','rng_cpu','rng_cuda']}
    paths={'parent250':ROOT/config['parent_checkpoint']};final_rng=[]
    for arm in config['arms']:
        name=arm['name'];directory=output/'models'/name
        snapshot=torch.load(directory/'resume-state.pt',map_location='cpu',weights_only=True)
        checkpoint=torch.load(directory/'final.pt',map_location='cpu',weights_only=True);receipt=read(directory/'training-receipt.json')
        require(receipt==checkpoint['receipt']==training['arms'][name],'Receipt and final checkpoint disagree')
        require(snapshot['step']==250 and all(recursive_digest(snapshot[k])==v for k,v in parent_states.items()),'Actual model/AdamW/scaler/RNG restoration differs')
        require(snapshot['config']==parent['config'] and snapshot['normalization']==parent['normalization']
                and snapshot['positive_weights']==parent['positive_weights'] and snapshot['source_scheduler']==parent['scheduler'],
                'Restored metadata changed')
        proof=receipt['restoration_proof']
        require(proof==snapshot['restoration_proof'] and proof['parent_step']==250 and proof['new_optimizer_updates']==0
                and proof['parent_state_sha256']==proof['restored_state_sha256']==parent_states
                and proof['historical_rng_restored'] and proof['historical_scaler_restored'],'Restoration proof changed')
        require(receipt['new_optimizer_steps']==250 and receipt['parent_optimizer_steps']==250
                and receipt['optimizer_steps']==checkpoint['step']==receipt['selected_step']==500
                and receipt['conditional_fret_coefficient']==arm['conditional_fret_coefficient'],'New/lifetime counters or coefficient differ')
        require(optimizer_counters(checkpoint)=={'minimum':500,'maximum':500,'parameter_states':34},'Actual final AdamW counters differ')
        _validate_group(checkpoint['optimizer'],34,.0001)
        require(checkpoint['scaler']=={**parent['scaler'],'_growth_tracker':parent['scaler']['_growth_tracker']+250},'AMP scale/history changed unexpectedly')
        require(not checkpoint['scheduler_used'] and checkpoint['source_scheduler']==parent['scheduler']
                and not receipt['scheduler_used'] and not receipt['early_stopping_used']
                and not receipt['validation_used_during_training'] and receipt['test_inference_calls']==0,'Training scheduling/selection scope changed')
        require(receipt['source_sha256']==producer_hashes() and receipt['config_sha256']==fingerprint(config)
                and receipt['input_audit_sha256']==sha256(output/'input-audit.json')
                and receipt['resume_state_sha256']==sha256(directory/'resume-state.pt'),'Source/input/restoration binding changed')
        require(receipt['crop_schedule_sha256']==fingerprint(expected) and receipt['gain_schedule_sha256']==fingerprint([r['gain_log'] for r in expected]),'Crop/gain exposure changed')
        require(receipt['model_parameters']==931361 and checkpoint['config']==receipt['model_config']
                and checkpoint['config']['feature_planes']==[2048,2048]
                and checkpoint['normalization']==parent['normalization'] and checkpoint['positive_weights']==parent['positive_weights'],
                'Model capacity, frontend or loss balancing changed')
        history=receipt['history']
        require([r['data_epoch'] for r in history]==[6,7,8,9,10] and
                [r['new_optimizer_steps'] for r in history]==[50,100,150,200,250] and
                [r['optimizer_steps'] for r in history]==[300,350,400,450,500],'New epoch ledger changed')
        require([r['auxiliary_valid_active_string_frames'] for r in history]==support_epochs,'Auxiliary supervised support differs from independently reconstructed crops')
        for row in history:
            step=row['optimizer_steps'];loss=row['training_losses']
            require(row['actual_optimizer_step_counters']=={'minimum':step,'maximum':step,'parameter_states':34}
                    and row['learning_rate']==.0001,'Block counters/rate differ')
            total=loss['state']+loss['onset']+.5*loss['offset']+.5*loss['pitch']
            if arm['conditional_fret_coefficient']:total+=.5*loss['conditional_fret']
            require(('conditional_fret' in loss)==bool(arm['conditional_fret_coefficient']) and abs(total-loss['total'])<1e-5,'Loss decomposition differs')
        final_rng.append({k:recursive_digest(checkpoint[k]) for k in ['rng_cpu','rng_cuda']});paths[name+'500']=directory/'final.pt'
    require(final_rng[0]==final_rng[1],'Paired final random streams differ')
    run=read(output/'validation/run.json');score=run['scoring_config']
    require(run['checkpoint_sha256']=={m:sha256(p) for m,p in paths.items()} and run['source_sha256']==producer_hashes()
            and run['test_inference_calls']==0 and not run['production_promoted'],'Validation models/source scope changed')
    records=[read(p) for p in sorted((output/'validation/results').glob('*/*-metrics.json'))];lookup={(r['id'],r['mode']):r for r in records}
    require(len(records)==len(lookup)==36,'Missing/duplicated validation replay')
    ordered=[];note_count=event_count=bit_identical=0
    for index,item in enumerate(datasets['validation']):
        entry=item['entry'];reference=reference_notes(ROOT/entry['annotation'])
        for mode in MODES[index%3:]+MODES[:index%3]:
            row=lookup[(entry['id'],mode)];path=output/row['probability_path']
            require(sha256(path)==row['probability_sha256'],'Probability recording changed')
            with np.load(path,allow_pickle=False) as archive:probability={k:archive[k] for k in archive.files}
            notes,events,_=replay(probability,base,config['decoder_parameters'],12.)
            require(without_compute(row['notes'])==notes and without_compute(row['events'])==events,'Exact source decoder replay differs')
            require(all(e['delivered_at_seconds']+1e-9>=e['emitted_at_audio_seconds'] for e in row['events']),'Completion precedes availability')
            heads=aligned_heads(probability,item['targets'],config,base,1000);diagnosis,_=state_diagnosis(probability,item['targets'],frame_config(config))
            require(row['head_metrics']==heads and row['frame_diagnosis']==diagnosis and row['state_derived_pitch']==diagnosis['state_derived_pitch'],'Raw-head/frame diagnosis differs')
            verify_frame_partition(diagnosis)
            require(row['pitch_metrics']==evaluate(reference,row['notes'],score,12.) and
                    row['tab_metrics']==tablature_metrics(reference,row['notes'],score,10,12.) and
                    row['latency']==measure(reference,row['notes'],score,12.),'Note/string/completion scoring differs')
            if mode=='parent250':
                proof=check_parent(item,probability,row['notes'],row['events'],heads,diagnosis,row['pitch_metrics'],row['tab_metrics'])
                require(proof==row['parent_reproduction'],'Parent reproduction evidence differs');bit_identical+=int(proof['probabilities_bit_identical'])
            stem=output/'validation/results'/mode/entry['id']
            require(read(stem.with_suffix('.json'))['notes']==row['notes'] and read(stem.with_name(stem.name+'-events.json'))['events']==row['events'],'JSON/event export differs')
            with stem.with_suffix('.csv').open(newline='',encoding='utf-8') as stream:values=list(csv.DictReader(stream))
            require([(float(n['start']),float(n['end']),int(n['pitch']),int(n['string']),int(n['fret'])) for n in values]==
                    [(n['start'],n['end'],n['pitch'],n['string'],n['fret']) for n in row['notes']],'CSV differs')
            ticks=Counter((round(n['start']*960),max(round(n['start']*960)+1,round(n['end']*960)),n['pitch']) for n in row['notes'])
            require(midi_intervals(stem.with_suffix('.mid'))==ticks,'MIDI lost an attack')
            require(len(row['events'])==2*len(row['notes']),'Event ledger lost a boundary')
            require(row['timing']['lookahead_samples']==0 and row['timing']['feature_windows_samples']==[2048,2048],'Feature timing changed')
            ordered.append(row);note_count+=len(row['notes']);event_count+=len(row['events'])
    summary=summarize(ordered);require(summary==read(output/'validation/summary.json'),'Summary aggregation differs')
    selection=read(output/'selection.json');winner=choose({m:summary['modes'][m] for m in MODES[1:]})
    require(selection==run['selection'] and selection['selected_arm']==winner and selection['eligible_modes']==MODES[1:]
            and not selection['selected_from_test'] and not selection['production_promoted'],'Fixed-final selection changed')
    return {'passed':True,'new_optimizer_updates':500,'historical_unique_parent_updates':250,'final_lifetime_updates_per_arm':500,
        'new_sanity_updates':0,'validation_cases':36,'parent_reproductions':12,'parent_probability_sets_bit_identical':bit_identical,
        'exact_notes':note_count,'exact_events':event_count,'paired_final_rng_identical':True,'test_inference_calls':0,
        'auxiliary_active_support_per_epoch':support_epochs,'source_timing_and_exports_verified':True,'selected_arm':winner,
        'production_promoted':False,'artifact_sha256':{name:sha256(output/name) for name in ['protocol.json','input-audit.json',
        'crop-schedule.json','training-summary.json','selection.json','validation/run.json','validation/summary.json']}}
