"""Reproduce metrics/exports and check actual fresh paired training state."""

import csv
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from acoustic10.training import tensor_digest
from baseline.download import ROOT,sha256
from baseline.evaluation import evaluate,reference_notes
from cnn_gru.inference import tablature_metrics
from decoder14.audit import midi_intervals
from decoder14.core import replay,without_compute
from decoder14.workflow import read
from diagnosis15.analysis import activity_states,pitch_union
from diagnosis9.head_audit import binary_metric
from optimization13.audit import optimizer_counters
from optimization13.optimizer import recursive_digest,_validate_group
from spectral11.workflow import aligned_heads
from streaming.data import fingerprint
from streaming.latency import measure
from .data import load_inputs,validate_config,model_configuration
from .model import CausalCNNGRU
from .training import producer_hashes
from .workflow import MODES,summarize,choose


def require(value,message):
    if not value:raise ValueError(message)


def audit(output):
    output=Path(output);protocol=read(output/'protocol.json');config=protocol['config'];validate_config(config)
    require(protocol['source_sha256']==producer_hashes(),'Frozen producer changed')
    require(protocol['total_new_optimizer_update_cap']==500 and protocol['historical_warmstart_updates']==0
            and protocol['new_sanity_updates']==0 and protocol['test_inference_allowed'] is False,'Budget/test scope changed')
    base,cache,datasets,provenance,expected=load_inputs(config,output)
    require(protocol['input_audit_sha256']==sha256(output/'input-audit.json'),'Input binding changed')
    parity=read(output/'frontend-parity.json')
    require(protocol['frontend_parity_sha256']==sha256(output/'frontend-parity.json') and parity['passed']
            and len(parity['records'])==24 and all(r['full_packet_cache_byte_equal'] and r['lookahead_samples']==0
            and r['frames']==1200 and r['first_frame_available_sample']==160 for r in parity['records']), 'Frontend parity changed')
    schedule=read(output/'crop-schedule.json')
    require(schedule['descriptors']==expected and schedule['sha256']==fingerprint(expected)
            and len(expected)==1000 and {r['epoch'] for r in expected}=={1,2,3,4,5},'Fresh crop schedule changed')
    training=read(output/'training-summary.json')
    require(training['new_optimizer_updates']==500 and training['final_lifetime_updates_per_arm']==250
            and training['historical_warmstart_updates']==0 and training['new_sanity_updates']==0
            and training['test_inference_calls']==0 and training['validation_used_during_training'] is False,'Training ledger changed')
    require(training['source_sha256']==producer_hashes() and training['protocol_sha256']==sha256(output/'protocol.json'),'Training source binding changed')
    initial_path=output/'initial-weights.pt';initial=torch.load(initial_path,map_location='cpu',weights_only=True)
    torch.manual_seed(config['seed']);fresh=CausalCNNGRU(model_configuration(base,config,config['arms'][0]))
    require(tensor_digest(initial['model'])==tensor_digest(fresh.state_dict()) and initial['step']==0,'Seed42 fresh initialization changed')
    initial_digests=[];final_rng=[];final_paths={}
    for arm in config['arms']:
        name=arm['name'];directory=output/'models'/name
        before=torch.load(directory/'before-training.pt',map_location='cpu',weights_only=True)
        checkpoint=torch.load(directory/'final.pt',map_location='cpu',weights_only=True)
        receipt=read(directory/'training-receipt.json')
        require(receipt==checkpoint['receipt']==training['arms'][name],'Training receipt differs')
        require(before['step']==0 and before['optimizer']['state']=={} and before['scheduler']['last_epoch']==0
                and before['optimizer']['param_groups'][0]['lr']==.001,'Arm did not start with fresh optimizer/scheduler')
        for key in ['model','rng_cpu','rng_cuda']:
            require(recursive_digest(before[key])==recursive_digest(initial[key]),'Shared initialization/randomness differs')
        require(before['scaler'].get('scale',1024)==1024 and before['scaler'].get('_growth_tracker',0)==0,'Initial scaler differs')
        digest=recursive_digest(before);initial_digests.append(digest)
        require(receipt['initial_training_state_sha256']==digest and receipt['initial_weights_sha256']==tensor_digest(initial['model'])
                and receipt['initial_checkpoint_sha256']==sha256(initial_path),'Initial state receipt changed')
        require(receipt['model_parameters']==initial['model_parameters']==sum(p.numel() for p in fresh.parameters())==931361
                and receipt['model_config']==model_configuration(base,config,arm),'Capacity or model configuration changed')
        require(receipt['new_optimizer_steps']==receipt['optimizer_steps']==checkpoint['step']==receipt['lifetime_updates']==250
                and not receipt['warmstarted'],'Actual new/lifetime counters changed')
        require(optimizer_counters(checkpoint)=={'minimum':250,'maximum':250,'parameter_states':34},'Final AdamW counters changed')
        _validate_group(checkpoint['optimizer'],34,.0001)
        require(checkpoint['scheduler']['last_epoch']==250 and abs(checkpoint['scheduler']['_last_lr'][0]-.0001)<1e-12,'Final cosine schedule changed')
        require(checkpoint['scaler'].get('_growth_tracker',250)==250 and checkpoint['scaler'].get('scale',1024)==1024,'AMP growth history changed')
        require(receipt['source_sha256']==producer_hashes() and receipt['input_audit_sha256']==sha256(output/'input-audit.json')
                and receipt['crop_schedule_sha256']==fingerprint(expected)
                and receipt['gain_schedule_sha256']==fingerprint([r['gain_log'] for r in expected]),'Source/input/crop bindings changed')
        require(not receipt['early_stopping_used'] and not receipt['validation_used_during_training'],'Training selection changed')
        require(checkpoint['normalization']==cache['normalization'][name] and checkpoint['positive_weights']==cache['positive_weights'],'Feature/loss balancing changed')
        history=receipt['history']
        require([r['epoch'] for r in history]==[1,2,3,4,5] and [r['optimizer_steps'] for r in history]==[50,100,150,200,250],'Epoch ledger changed')
        for row in history:
            losses=row['training_losses'];step=row['optimizer_steps']
            require(row['actual_optimizer_step_counters']=={'minimum':step,'maximum':step,'parameter_states':34},'Block counters changed')
            require(set(losses)=={'total','state','onset','offset','pitch'} and
                    abs(losses['total']-(losses['state']+losses['onset']+.5*losses['offset']+.5*losses['pitch']))<1e-5,'Legacy loss decomposition changed')
        final_rng.append({k:recursive_digest(checkpoint[k]) for k in ['rng_cpu','rng_cuda']})
        final_paths[name]=directory/'final.pt'
    require(len(set(initial_digests))==1 and final_rng[0]==final_rng[1],'Paired random streams differ')
    run=read(output/'validation/run.json');score=run['scoring_config']
    require(run['checkpoint_sha256']=={m:sha256(p) for m,p in final_paths.items()} and run['source_sha256']==producer_hashes()
            and run['test_inference_calls']==0 and not run['production_promoted'],'Validation model/scope changed')
    records=[read(p) for p in sorted((output/'validation/results').glob('*/*-metrics.json'))]
    lookup={(r['id'],r['mode']):r for r in records}
    require(len(records)==len(lookup)==24,'Missing or duplicated validation replay')
    ordered=[];note_count=event_count=0
    for index,item in enumerate(datasets['validation']):
        entry=item['entry'];reference=reference_notes(ROOT/entry['annotation'])
        for mode in MODES[index%2:]+MODES[:index%2]:
            row=lookup[(entry['id'],mode)];path=output/row['probability_path']
            require(sha256(path)==row['probability_sha256'],'Recorded probabilities changed')
            with np.load(path,allow_pickle=False) as archive:probability={k:archive[k] for k in archive.files}
            require(all(len(v)==1200 and np.isfinite(v).all() for v in probability.values()),'Probability frame grid differs')
            notes,events,_=replay(probability,base,config['decoder_parameters'],12.)
            require(without_compute(row['notes'])==notes and without_compute(row['events'])==events,'Decoder source replay differs')
            require(all(e['delivered_at_seconds']+1e-9>=e['emitted_at_audio_seconds'] for e in row['events']),'Delivery precedes packet')
            require(row['head_metrics']==aligned_heads(probability,item['targets'],config,base,1000),'Raw-head metrics differ')
            states,_=activity_states(probability['state'][13:1013],.25);valid=np.arange(1000)>=26
            require(row['state_derived_pitch']==binary_metric(item['targets']['pitch'][:1000][valid].astype(bool),pitch_union(states)[valid]),'State pitch metrics differ')
            require(row['pitch_metrics']==evaluate(reference,row['notes'],score,12.) and
                    row['tab_metrics']==tablature_metrics(reference,row['notes'],score,10,12.) and
                    row['latency']==measure(reference,row['notes'],score,12.),'Note/string/latency metrics differ')
            stem=output/'validation/results'/mode/entry['id']
            require(read(stem.with_suffix('.json'))['notes']==row['notes'] and
                    read(stem.with_name(stem.name+'-events.json'))['events']==row['events'],'Exact JSON/events differ')
            with stem.with_suffix('.csv').open(newline='',encoding='utf-8') as stream:values=list(csv.DictReader(stream))
            require([(float(n['start']),float(n['end']),int(n['pitch']),int(n['string']),int(n['fret'])) for n in values]==
                    [(n['start'],n['end'],n['pitch'],n['string'],n['fret']) for n in row['notes']],'CSV differs')
            ticks=Counter((round(n['start']*960),max(round(n['start']*960)+1,round(n['end']*960)),n['pitch']) for n in row['notes'])
            require(midi_intervals(stem.with_suffix('.mid'))==ticks,'MIDI lost an attack')
            require(len(row['events'])==2*len(row['notes']),'Event ledger lost a boundary')
            require(row['timing']['lookahead_samples']==0 and row['timing']['packet_ms']==20 and
                    row['timing']['feature_windows_samples']==next(a['windows'] for a in config['arms'] if a['name']==mode),'Frontend timing contract changed')
            ordered.append(row);note_count+=len(row['notes']);event_count+=len(row['events'])
    summary=summarize(ordered)
    require(summary==read(output/'validation/summary.json'),'Summary aggregation differs')
    selection=read(output/'selection.json');winner=choose(summary['modes'])
    require(selection==run['selection'] and selection['selected_arm']==winner and selection['eligible_modes']==MODES
            and not selection['selected_from_test'] and not selection['production_promoted'],'Final-only selection changed')
    return {'passed':True,'new_optimizer_updates':500,'historical_warmstart_updates':0,'final_lifetime_updates_per_arm':250,
        'new_sanity_updates':0,'validation_cases':24,'frontend_parity_cases':24,'exact_notes':note_count,'exact_events':event_count,
        'shared_initial_training_state_identical':True,'paired_final_rng_identical':True,'test_inference_calls':0,
        'selected_arm':winner,'production_promoted':False,'source_timing_and_exports_verified':True,
        'artifact_sha256':{name:sha256(output/name) for name in ['protocol.json','input-audit.json','frontend-parity.json',
            'crop-schedule.json','initial-weights.pt','training-summary.json','selection.json','validation/run.json','validation/summary.json']}}
