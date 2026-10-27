"""Original-loss versus direct state-pitch supervision, with an immutable parent."""

import gc
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from acoustic10.workflow import pcm_replay
from baseline.download import ROOT,sha256
from baseline.evaluation import evaluate,reference_notes
from cnn_gru.data import load_audio
from cnn_gru.inference import tablature_metrics
from decoder14.core import without_compute
from decoder14.workflow import read
from diagnosis15.analysis import activity_states,pitch_union
from diagnosis9.head_audit import binary_metric
from normalization12.model import CausalCNNGRU
from spectral11.workflow import aligned_heads
from streaming.evaluate import aggregate,save_json
from streaming.export import export
from streaming.latency import measure
from .data import load_inputs,validate_config
from .training import producer_hashes,train_arm,verified_sanity


MODES=['parent500','control750','state_pitch750']


def choose(summaries):
    if set(summaries)!={'control750','state_pitch750'}:
        raise ValueError('Only the two fixed final750 models are eligible.')
    values={}
    for mode,row in summaries.items():
        values[mode]=tuple(row[key] for key in ['exact_state_f1','onset_f1'])
        if any(not np.isfinite(v) or not 0<=v<=1 for v in values[mode]):
            raise ValueError('Selection metrics must be finite and in[0,1].')
    return 'state_pitch750' if values['state_pitch750']>values['control750'] else 'control750'


def run(config,output,device):
    validate_config(config)
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol.json').exists():
        raise ValueError('Stage16 already started here; training is not repeated. Reuse evaluate/report/audit.')
    base,audit,datasets,provenance,parent,schedule=load_inputs(config,output)
    sanity=verified_sanity(config)
    protocol={'config':config,'source_sha256':producer_hashes(),'parent':provenance['parent'],
              'total_new_optimizer_update_cap':500,'planned_new_optimizer_updates':{'control':250,'state_pitch':250},
              'historical_unique_parent_updates':500,'new_sanity_updates':0,'sanity_reference':sanity,
              'test_inference_allowed':False,'production_promotion':False,'selection_rule':config['selection_rule']}
    save_json(output/'protocol.json',protocol)
    from streaming.data import fingerprint
    save_json(output/'crop-schedule.json',{'descriptors':schedule,'sha256':fingerprint(schedule)})
    receipts={}
    for arm in config['arms']:
        receipts[arm['name']]=train_arm(arm,config,base,audit,datasets['train'],provenance,parent,schedule,output,device)
        gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
    if sum(r['new_optimizer_steps'] for r in receipts.values())!=500:
        raise ValueError('Actual new-update accounting differs from the fixed500 budget.')
    save_json(output/'training-summary.json',{'config':config,'arms':receipts,'new_optimizer_updates':500,
        'historical_unique_parent_updates':500,'final_lifetime_updates_per_arm':750,'new_sanity_updates':0,
        'additional_epoch_blocks_per_arm':5,'historical_rng_restored':True,'historical_scaler_restored':True,
        'validation_used_during_training':False,'test_inference_calls':0,'sanity_reference':sanity,
        'source_sha256':producer_hashes(),'protocol_sha256':sha256(output/'protocol.json')})
    del datasets,parent
    gc.collect()
    return evaluate_validation(config,output,device)


def check_parent(item,probability,notes,events,heads,metrics,tabs):
    root=ROOT/'outputs/decoder14'
    archived=read(root/'validation/results/group8_500_selected'/(item['entry']['id']+'-metrics.json'))
    prior=read(ROOT/'outputs/optimization13/validation/results/group8_500'/(item['entry']['id']+'-metrics.json'))
    path=ROOT/'outputs/optimization13'/prior['probability_path']
    if sha256(path)!=prior['probability_sha256']:
        raise ValueError('Parent probability recording changed.')
    exact=True
    maximum=0.
    with np.load(path,allow_pickle=False) as arrays:
        if set(arrays.files)!=set(probability):raise ValueError('Parent probability interface changed.')
        for key in arrays.files:
            if arrays[key].shape!=probability[key].shape:raise ValueError('Parent probability frame grid changed.')
            exact=exact and np.array_equal(arrays[key],probability[key])
            maximum=max(maximum,float(np.max(np.abs(arrays[key]-probability[key]))))
    if maximum>1e-6:raise ValueError('Parent replay differs beyond the prescribed numerical tolerance.')
    if (without_compute(notes)!=archived['notes'] or without_compute(events)!=archived['events']
            or without_compute(metrics)!=archived['pitch_metrics'] or without_compute(tabs)!=archived['tab_metrics']
            or heads!=archived['head_metrics']):
        raise ValueError('Parent source decisions, calibrated scores or raw heads changed.')
    return {'probabilities_bit_identical':bool(exact),'max_absolute_probability_difference':maximum,
            'source_decisions_identical':True,'source_scores_identical':True,'head_metrics_identical':True,
            'historical_probability_sha256':prior['probability_sha256']}


def evaluate_validation(config,output,device):
    output=Path(output).resolve()
    protocol=read(output/'protocol.json')
    if protocol['config']!=config or protocol['source_sha256']!=producer_hashes():
        raise ValueError('Frozen Stage16 code/config changed.')
    base,audit,datasets,provenance,parent,_=load_inputs(config,output)
    paths={'parent500':ROOT/config['parent_checkpoint'],
           'control750':output/'models/control/final.pt','state_pitch750':output/'models/state_pitch/final.pt'}
    checkpoints={'parent500':parent}
    networks={}
    for mode,path in paths.items():
        checkpoint=parent if mode=='parent500' else torch.load(path,map_location='cpu',weights_only=True)
        if mode!='parent500' and (checkpoint['step']!=750 or checkpoint['receipt']['source_sha256']!=producer_hashes()
                or checkpoint['receipt']['input_audit_sha256']!=sha256(output/'input-audit.json')):
            raise ValueError('Final model/source/input binding changed.')
        checkpoints[mode]=checkpoint
        model=CausalCNNGRU(checkpoint['config']).to(device)
        model.load_state_dict(checkpoint['model']);model.eval();networks[mode]=model
    score={key:read(ROOT/'configs/stage1.json')[key] for key in
           ['score_seconds','frame_rate','onset_tolerance_seconds','offset_tolerance_seconds','offset_ratio']}
    warm,_=load_audio(ROOT/datasets['validation'][0]['entry']['audio'],1,base)
    for mode in MODES:
        pcm_replay(networks[mode],warm,checkpoints[mode],config['decoder_parameters'],device,20,.13)
    records=[]
    for index,item in enumerate(datasets['validation']):
        entry=item['entry']
        audio,duration=load_audio(ROOT/entry['audio'],12,base)
        relative=Path('validation/clips')/(entry['id']+'.wav')
        (output/relative).parent.mkdir(parents=True,exist_ok=True)
        sf.write(output/relative,audio,16000,subtype='FLOAT')
        reference=reference_notes(ROOT/entry['annotation'])
        order=MODES[index%3:]+MODES[:index%3]
        for mode in order:
            notes,events,timing,probability=pcm_replay(networks[mode],audio,checkpoints[mode],config['decoder_parameters'],device,20,.13,capture=True)
            probability_path=output/'validation/probabilities'/mode/(entry['id']+'.npz')
            probability_path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(probability_path,**probability)
            heads=aligned_heads(probability,item['targets'],config,base,1000)
            observed,_=activity_states(probability['state'][13:1013],.25)
            valid=np.arange(1000)>=26
            state_pitch=binary_metric(item['targets']['pitch'][:1000][valid].astype(bool),pitch_union(observed)[valid])
            metrics=evaluate(reference,notes,score,duration)
            tabs=tablature_metrics(reference,notes,score,10,duration)
            reproduction=check_parent(item,probability,notes,events,heads,metrics,tabs) if mode=='parent500' else None
            row={'id':entry['id'],'mode':mode,'notes':notes,'events':events,'reference':reference,
                 'head_metrics':heads,'state_derived_pitch':state_pitch,'pitch_metrics':metrics,'tab_metrics':tabs,
                 'timing':timing,'latency':measure(reference,notes,score,duration),'relativeaudio':relative.as_posix(),
                 'probability_path':probability_path.relative_to(output).as_posix(),'probability_sha256':sha256(probability_path),
                 'input_provenance':provenance['inputs']['validation'][index],
                 'parent_reproduction':reproduction}
            stem=output/'validation/results'/mode/entry['id']
            export(notes,stem,{'stage':16,'mode':mode,'checkpoint_sha256':sha256(paths[mode]),
                              'decoder_parameters':config['decoder_parameters'],'source_delay_seconds':.13,
                              'clock_scope':'Fresh paired PCM replay; virtual completion includes measured processing.'})
            save_json(stem.with_name(stem.name+'-metrics.json'),row)
            save_json(stem.with_name(stem.name+'-events.json'),{'events':events})
            records.append(row)
            print(mode,entry['id'],'state F1',round(heads['exact_state_f1'],4),'note onset F1',round(metrics['onset']['f1'],4),flush=True)
    summary=aggregate(records)
    for mode in MODES:
        rows=[r for r in records if r['mode']==mode]
        for field,key in [('exact_state_f1','exact_state_f1'),('active_fret_accuracy','active_fret_accuracy')]:
            summary['modes'][mode][field]=float(np.mean([r['head_metrics'][key] for r in rows]))
        summary['modes'][mode]['auxiliary_pitch_f1']=float(np.mean([r['head_metrics']['pitch']['overall']['f1'] for r in rows]))
        summary['modes'][mode]['low_register_pitch_f1']=float(np.mean([r['head_metrics']['pitch']['low_register']['f1'] for r in rows]))
        summary['modes'][mode]['state_derived_pitch_f1']=float(np.mean([r['state_derived_pitch']['f1'] for r in rows]))
    winner=choose({mode:summary['modes'][mode] for mode in ['control750','state_pitch750']})
    selection={'selected_arm':winner,'eligible_modes':['control750','state_pitch750'],'rule':config['selection_rule'],
               'scores':{mode:{key:summary['modes'][mode][key] for key in ['exact_state_f1','onset_f1']} for mode in ['control750','state_pitch750']},
               'selected_from_test':False,'production_promoted':False}
    save_json(output/'selection.json',selection)
    save_json(output/'validation/summary.json',summary)
    run={'config':config,'scoring_config':score,'selection':selection,'new_optimizer_updates':500,
         'historical_unique_parent_updates':500,'final_lifetime_updates_per_arm':750,'test_inference_calls':0,
         'production_promoted':False,'checkpoint_paths':{mode:path.relative_to(ROOT).as_posix() for mode,path in paths.items()},
         'checkpoint_sha256':{mode:sha256(path) for mode,path in paths.items()},'source_sha256':producer_hashes(),
         'input_provenance':provenance,'training_summary':read(output/'training-summary.json'),
         'scope':'One shared parent, one new loss term, matched new exposure. Reused validation development comparison; no independent-test or product promotion.'}
    save_json(output/'validation/run.json',run)
    return summary
