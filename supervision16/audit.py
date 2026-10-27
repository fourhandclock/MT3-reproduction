"""Recompute validation decisions and verify the500-to750 continuation lineage."""

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from acoustic10.data import load_inputs as frozen_inputs
from baseline.download import ROOT,sha256
from baseline.evaluation import evaluate,reference_notes
from cnn_gru.inference import tablature_metrics
from decoder14.audit import midi_intervals
from decoder14.core import replay,without_compute
from decoder14.workflow import read
from diagnosis15.analysis import activity_states,pitch_union
from diagnosis9.head_audit import binary_metric
from optimization13.audit import optimizer_counters
from optimization13.data import build_schedule
from optimization13.optimizer import _validate_group,recursive_digest
from spectral11.workflow import aligned_heads
from streaming.data import fingerprint
from streaming.evaluate import aggregate
from streaming.latency import measure
from .data import validate_config
from .training import producer_hashes
from .workflow import MODES,check_parent


def require(value,message):
    if not value:raise ValueError(message)


def audit(output):
    output=Path(output)
    protocol=read(output/'protocol.json');config=protocol['config'];validate_config(config)
    require(protocol['source_sha256']==producer_hashes(),'Frozen production code changed.')
    require(protocol['total_new_optimizer_update_cap']==500 and protocol['historical_unique_parent_updates']==500
            and protocol['new_sanity_updates']==0 and protocol['test_inference_allowed'] is False,'Budget/test scope changed.')
    base,cache,datasets,provenance=frozen_inputs(read(ROOT/'configs/acoustic10.json'))
    inputs=read(output/'input-audit.json')
    for key in ['manifest_sha256','cache_audit_sha256','inputs','normalization_source','normalization','positive_weights']:
        expected=cache[key] if key in ['normalization','positive_weights'] else provenance[key]
        require(inputs[key]==expected,'Frozen original input audit changed: '+key)
    require(inputs['features_rewritten'] is False and inputs['targets_rewritten'] is False,'Original caches were rewritten.')
    for path,digest in inputs['proposal_bindings'].items():
        require(sha256(ROOT/path)==digest,'Stage15 proposal identity changed.')
    schedule=read(output/'crop-schedule.json')
    expected=build_schedule(config,datasets['train'])
    require(schedule['descriptors']==expected and schedule['sha256']==fingerprint(expected),'Matched epoch11-15 crop/gain schedule changed.')
    require(len(expected)==1000 and {r['epoch'] for r in expected}=={11,12,13,14,15},'New crop epoch accounting changed.')
    parent_path=ROOT/config['parent_checkpoint']
    require(sha256(parent_path)==config['parent_checkpoint_sha256'],'The shared original parent changed.')
    parent=torch.load(parent_path,map_location='cpu',weights_only=True)
    states={key:recursive_digest(parent[key]) for key in ['model','optimizer','scaler','rng_cpu','rng_cuda']}
    training=read(output/'training-summary.json')
    require(training['new_optimizer_updates']==500 and training['historical_unique_parent_updates']==500
            and training['final_lifetime_updates_per_arm']==750 and training['new_sanity_updates']==0
            and training['test_inference_calls']==0 and training['validation_used_during_training'] is False,'Training summary accounting changed.')
    require(training['source_sha256']==producer_hashes() and training['protocol_sha256']==sha256(output/'protocol.json'),'Training summary provenance changed.')
    final_rng={}
    final_paths={'parent500':parent_path}
    for arm in config['arms']:
        name=arm['name'];directory=output/'models'/name
        snapshot=torch.load(directory/'resume-state.pt',map_location='cpu',weights_only=True)
        checkpoint=torch.load(directory/'final.pt',map_location='cpu',weights_only=True)
        receipt=read(directory/'training-receipt.json')
        require(receipt==checkpoint['receipt']==training['arms'][name],'Receipt/checkpoint/training summary disagree.')
        require(snapshot['step']==500 and all(recursive_digest(snapshot[key])==digest for key,digest in states.items()),'Pre-update model/AdamW/AMP/RNG restoration differs from the actual parent.')
        require(snapshot['config']==parent['config'] and snapshot['normalization']==parent['normalization']
                and snapshot['positive_weights']==parent['positive_weights'],'Restored metadata changed.')
        proof=receipt['restoration_proof']
        require(proof==snapshot['restoration_proof'] and proof['parent_state_sha256']==proof['restored_state_sha256']==states
                and proof['parent_step']==500 and proof['new_optimizer_updates']==0
                and proof['historical_rng_restored'] and proof['historical_scaler_restored'],'Restoration proof changed.')
        require(receipt['new_optimizer_steps']==250 and receipt['optimizer_steps']==checkpoint['step']==receipt['selected_step']==750
                and receipt['parent_optimizer_steps']==500 and receipt['state_pitch_coefficient']==arm['state_pitch_coefficient'],
                'Actual new/lifetime counters or treatment changed.')
        require(optimizer_counters(checkpoint)=={'minimum':750,'maximum':750,'parameter_states':34},'Final AdamW counters changed.')
        _validate_group(checkpoint['optimizer'],34,.0001)
        require(receipt['resume_state_sha256']==sha256(directory/'resume-state.pt')
                and receipt['source_sha256']==producer_hashes() and receipt['config_sha256']==fingerprint(config)
                and receipt['input_audit_sha256']==sha256(output/'input-audit.json'),'Receipt source/input binding changed.')
        require(receipt['crop_schedule_sha256']==schedule['sha256'] and receipt['common_support_sha256']==parent['receipt']['common_support_sha256']
                and receipt['gain_schedule_sha256']==fingerprint([r['gain_log'] for r in expected]),'Shared crop/target/gain exposure changed.')
        require(receipt['early_stopping_used'] is False and receipt['validation_used_during_training'] is False
                and receipt['scheduler_used'] is False and receipt['test_inference_calls']==0,'Validation or scheduling entered training.')
        history=receipt['history']
        require(len(history)==5 and [r['optimizer_steps'] for r in history]==[550,600,650,700,750]
                and [r['new_optimizer_steps'] for r in history]==[50,100,150,200,250]
                and [r['data_epoch'] for r in history]==[11,12,13,14,15],'Epoch/block ledger changed.')
        for row in history:
            require(row['actual_optimizer_step_counters']=={'minimum':row['optimizer_steps'],'maximum':row['optimizer_steps'],'parameter_states':34}
                    and row['learning_rate']==.0001,'Block AdamW counters/rate changed.')
            losses=row['training_losses']
            total=losses['state']+losses['onset']+.5*losses['offset']+.5*losses['pitch']
            total+=arm['state_pitch_coefficient']*losses.get('state_pitch',0.)
            require(abs(total-losses['total'])<1e-5 and ('state_pitch' in losses)==bool(arm['state_pitch_coefficient']),
                    'Loss decomposition no longer matches the declared intervention.')
        require(checkpoint['scaler']=={**parent['scaler'],'_growth_tracker':parent['scaler']['_growth_tracker']+250},'AMP scaler history changed unexpectedly.')
        require(checkpoint['normalization']==parent['normalization'] and checkpoint['positive_weights']==parent['positive_weights'],
                'Final feature normalization or inherited loss balancing changed.')
        final_rng[name]={key:recursive_digest(checkpoint[key]) for key in ['rng_cpu','rng_cuda']}
        final_paths[name+'750']=directory/'final.pt'
    require(final_rng['control']==final_rng['state_pitch'],'Paired random streams did not consume identically.')
    run=read(output/'validation/run.json');score=run['scoring_config']
    require(run['checkpoint_sha256']=={mode:sha256(path) for mode,path in final_paths.items()}
            and run['source_sha256']==producer_hashes() and run['test_inference_calls']==0,'Validation checkpoint/source scope changed.')
    records=[read(path) for path in sorted((output/'validation/results').glob('*/*-metrics.json'))]
    lookup={(r['id'],r['mode']):r for r in records}
    require(len(lookup)==len(records)==36,'Missing/duplicated validation replay.')
    ordered=[];note_count=event_count=bit_identical=0
    for index,item in enumerate(datasets['validation']):
        entry=item['entry'];reference=reference_notes(ROOT/entry['annotation'])
        for mode in MODES[index%3:]+MODES[:index%3]:
            row=lookup[(entry['id'],mode)]
            path=output/row['probability_path'];require(sha256(path)==row['probability_sha256'],'Recorded packet probabilities changed.')
            with np.load(path,allow_pickle=False) as archive:probability={key:archive[key] for key in archive.files}
            notes,events,_=replay(probability,base,config['decoder_parameters'],12.)
            require(without_compute(row['notes'])==notes and without_compute(row['events'])==events,'Exact cached-source replay changed.')
            require(all(e['delivered_at_seconds']+1e-9>=e['emitted_at_audio_seconds'] for e in row['events']),'Virtual completion precedes availability.')
            heads=aligned_heads(probability,item['targets'],config,base,1000)
            require(row['head_metrics']==heads,'Source-aligned raw-head diagnostics changed.')
            observed,_=activity_states(probability['state'][13:1013],.25);valid=np.arange(1000)>=26
            require(row['state_derived_pitch']==binary_metric(item['targets']['pitch'][:1000][valid].astype(bool),pitch_union(observed)[valid]),'State-derived pitch metric changed.')
            require(row['pitch_metrics']==evaluate(reference,row['notes'],score,12.)
                    and row['tab_metrics']==tablature_metrics(reference,row['notes'],score,10,12.)
                    and row['latency']==measure(reference,row['notes'],score,12.),'Saved note/string/delivery scoring changed.')
            if mode=='parent500':
                proof=check_parent(item,probability,row['notes'],row['events'],heads,row['pitch_metrics'],row['tab_metrics'])
                require(proof==row['parent_reproduction'],'Parent reproduction evidence changed.')
                bit_identical+=int(proof['probabilities_bit_identical'])
            stem=output/'validation/results'/mode/entry['id']
            require(read(stem.with_suffix('.json'))['notes']==row['notes']
                    and read(stem.with_name(stem.name+'-events.json'))['events']==row['events'],'Exact JSON/event export changed.')
            with stem.with_suffix('.csv').open(newline='',encoding='utf-8') as stream:values=list(csv.DictReader(stream))
            require([(float(n['start']),float(n['end']),int(n['pitch']),int(n['string']),int(n['fret'])) for n in values]
                    ==[(n['start'],n['end'],n['pitch'],n['string'],n['fret']) for n in row['notes']],'CSV source intervals changed.')
            ticks=Counter((round(n['start']*960),max(round(n['start']*960)+1,round(n['end']*960)),n['pitch']) for n in row['notes'])
            require(midi_intervals(stem.with_suffix('.mid'))==ticks,'MIDI lost a quantized attack.')
            require(len(row['events'])==2*len(row['notes']),'Event ledger lost a boundary.')
            note_count+=len(row['notes']);event_count+=len(row['events']);ordered.append(row)
    summary=aggregate(ordered)
    for mode in MODES:
        rows=[r for r in ordered if r['mode']==mode]
        for key in ['exact_state_f1','active_fret_accuracy']:summary['modes'][mode][key]=float(np.mean([r['head_metrics'][key] for r in rows]))
        for field,key in [('auxiliary_pitch_f1','overall'),('low_register_pitch_f1','low_register')]:
            summary['modes'][mode][field]=float(np.mean([r['head_metrics']['pitch'][key]['f1'] for r in rows]))
        summary['modes'][mode]['state_derived_pitch_f1']=float(np.mean([r['state_derived_pitch']['f1'] for r in rows]))
    require(summary==read(output/'validation/summary.json'),'Summary aggregation changed.')
    selection=read(output/'selection.json')
    control=summary['modes']['control750'];treatment=summary['modes']['state_pitch750']
    winner='state_pitch750' if (treatment['exact_state_f1'],treatment['onset_f1'])>(control['exact_state_f1'],control['onset_f1']) else 'control750'
    require(selection['selected_arm']==winner and selection['eligible_modes']==['control750','state_pitch750']
            and selection['production_promoted'] is False and selection['selected_from_test'] is False,'Fixed-final selection changed.')
    return {'passed':True,'new_optimizer_updates':500,'historical_unique_parent_updates':500,'final_lifetime_updates_per_arm':750,
            'new_sanity_updates':0,'validation_cases':36,'parent_reproductions':12,'parent_probability_sets_bit_identical':bit_identical,
            'exact_notes':note_count,'exact_events':event_count,'paired_final_rng_identical':True,'test_inference_calls':0,
            'source_timing_and_exports_verified':True,'selected_arm':winner,'production_promoted':False,
            'artifact_sha256':{name:sha256(output/name) for name in ['protocol.json','input-audit.json','crop-schedule.json',
                'training-summary.json','selection.json','validation/run.json','validation/summary.json']}}
