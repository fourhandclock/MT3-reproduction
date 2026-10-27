"""Matched same-parent continuations with original and decoder-policy diagnostics."""

import gc
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from baseline.download import ROOT,sha256
from baseline.evaluation import evaluate,reference_notes
from cnn_gru.data import load_audio
from cnn_gru.inference import tablature_metrics
from decoder14.core import without_compute
from decoder14.workflow import read
from diagnosis9.head_audit import pooled
from headswap18.analysis import state_diagnosis
from multires17.model import CausalCNNGRU
from multires17.workflow import pcm_replay
from spectral11.workflow import aligned_heads
from streaming.data import fingerprint
from streaming.evaluate import aggregate,save_json
from streaming.export import export
from streaming.latency import measure
from .data import load_inputs,validate_config
from .training import train_arm,producer_hashes,verified_sanity

MODES=['parent250','control500','conditional_fret500']


def choose(summaries):
    if set(summaries)!={'control500','conditional_fret500'}:raise ValueError('Only the two fixed step500 finals are eligible')
    scores={m:tuple(r[k] for k in ['exact_state_f1','onset_f1']) for m,r in summaries.items()}
    if any(not np.isfinite(v) or not 0<=v<=1 for values in scores.values() for v in values):raise ValueError('Invalid selection metric')
    return 'conditional_fret500' if scores['conditional_fret500']>scores['control500'] else 'control500'


def frame_config(config):
    return {'source_frames':1000,'source_delay_frames':config['common_target_delay_frames'],
        'leading_mask_frames':config['common_leading_mask_frames'],'auxiliary_pitch_threshold':.5,
        'decoder_parameters':config['decoder_parameters']}


def run(config,output,device):
    validate_config(config);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol.json').exists():raise ValueError('Stage19 already started; its training budget cannot be repeated')
    base,audit,datasets,provenance,parent,schedule=load_inputs(config,output)
    sanity=verified_sanity(config)
    save_json(output/'protocol.json',{'config':config,'source_sha256':producer_hashes(),'parent':provenance['parent'],
        'total_new_optimizer_update_cap':500,'planned_new_optimizer_updates':{'control':250,'conditional_fret':250},
        'historical_unique_parent_updates':250,'new_sanity_updates':0,'sanity_reference':sanity,
        'test_inference_allowed':False,'production_promotion':False,'selection_rule':config['selection_rule']})
    save_json(output/'crop-schedule.json',{'descriptors':schedule,'sha256':fingerprint(schedule)})
    receipts={}
    for arm in config['arms']:
        receipts[arm['name']]=train_arm(arm,config,base,audit,datasets['train'],provenance,parent,schedule,output,device)
        gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
    if sum(r['new_optimizer_steps'] for r in receipts.values())!=500:raise ValueError('Actual new-update accounting differs')
    save_json(output/'training-summary.json',{'config':config,'arms':receipts,'new_optimizer_updates':500,
        'historical_unique_parent_updates':250,'final_lifetime_updates_per_arm':500,'new_sanity_updates':0,
        'additional_sampled_epoch_blocks_per_arm':5,'validation_used_during_training':False,'test_inference_calls':0,
        'source_sha256':producer_hashes(),'protocol_sha256':sha256(output/'protocol.json'),'sanity_reference':sanity})
    del datasets,parent;gc.collect()
    return evaluate_validation(config,output,device)


def check_parent(item,probability,notes,events,heads,diagnosis,metrics,tabs):
    root=ROOT/'outputs/multires17'
    prior=read(root/'validation/results/short_short'/(item['entry']['id']+'-metrics.json'))
    path=root/prior['probability_path']
    if sha256(path)!=prior['probability_sha256']:raise ValueError('Parent probability file changed')
    exact=True;maximum=0.
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files)!=set(probability):raise ValueError('Parent head interface changed')
        for key in archive.files:
            if archive[key].shape!=probability[key].shape:raise ValueError('Parent frame grid changed')
            exact=exact and np.array_equal(archive[key],probability[key])
            maximum=max(maximum,float(np.max(np.abs(archive[key]-probability[key]))))
    previous=read(ROOT/'outputs/headswap18/validation/frames/short_short'/(item['entry']['id']+'.json'))['diagnosis']
    if (maximum>1e-6 or without_compute(notes)!=without_compute(prior['notes'])
            or without_compute(events)!=without_compute(prior['events']) or heads!=prior['head_metrics']
            or diagnosis!=previous or without_compute(metrics)!=without_compute(prior['pitch_metrics'])
            or without_compute(tabs)!=without_compute(prior['tab_metrics'])):
        raise ValueError('Parent source decisions, raw heads or scores do not reproduce')
    return {'probabilities_bit_identical':bool(exact),'max_absolute_probability_difference':maximum,
        'source_decisions_identical':True,'source_scores_identical':True,'frame_diagnosis_identical':True,
        'historical_probability_sha256':prior['probability_sha256']}


def evaluate_validation(config,output,device):
    output=Path(output).resolve();protocol=read(output/'protocol.json')
    if protocol['config']!=config or protocol['source_sha256']!=producer_hashes():raise ValueError('Frozen Stage19 code/config changed')
    base,audit,datasets,provenance,parent,_=load_inputs(config,output)
    paths={'parent250':ROOT/config['parent_checkpoint'],'control500':output/'models/control/final.pt',
           'conditional_fret500':output/'models/conditional_fret/final.pt'}
    checkpoints={};networks={}
    for mode,path in paths.items():
        checkpoint=parent if mode=='parent250' else torch.load(path,map_location='cpu',weights_only=True)
        if mode!='parent250' and (checkpoint['step']!=500 or checkpoint['receipt']['source_sha256']!=producer_hashes()
                or checkpoint['receipt']['input_audit_sha256']!=sha256(output/'input-audit.json')):raise ValueError('Final model/source/input binding changed')
        model=CausalCNNGRU(checkpoint['config']).to(device);model.load_state_dict(checkpoint['model']);model.eval()
        checkpoints[mode]=checkpoint;networks[mode]=model
    score={k:read(ROOT/'configs/stage1.json')[k] for k in ['score_seconds','frame_rate','onset_tolerance_seconds','offset_tolerance_seconds','offset_ratio']}
    warm,_=load_audio(ROOT/datasets['validation'][0]['entry']['audio'],1,base)
    for mode in MODES:pcm_replay(networks[mode],warm,checkpoints[mode],config['decoder_parameters'],device)
    records=[]
    for index,item in enumerate(datasets['validation']):
        entry=item['entry'];audio,duration=load_audio(ROOT/entry['audio'],12,base)
        relative=Path('validation/clips')/(entry['id']+'.wav');(output/relative).parent.mkdir(parents=True,exist_ok=True)
        sf.write(output/relative,audio,16000,subtype='FLOAT');reference=reference_notes(ROOT/entry['annotation'])
        for mode in MODES[index%3:]+MODES[:index%3]:
            notes,events,timing,probability=pcm_replay(networks[mode],audio,checkpoints[mode],config['decoder_parameters'],device,True)
            probability_path=output/'validation/probabilities'/mode/(entry['id']+'.npz');probability_path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(probability_path,**probability)
            heads=aligned_heads(probability,item['targets'],config,base,1000)
            diagnosis,_=state_diagnosis(probability,item['targets'],frame_config(config))
            metrics=evaluate(reference,notes,score,duration);tabs=tablature_metrics(reference,notes,score,10,duration)
            proof=check_parent(item,probability,notes,events,heads,diagnosis,metrics,tabs) if mode=='parent250' else None
            row={'id':entry['id'],'mode':mode,'notes':notes,'events':events,'reference':reference,
                'head_metrics':heads,'frame_diagnosis':diagnosis,'state_derived_pitch':diagnosis['state_derived_pitch'],
                'pitch_metrics':metrics,'tab_metrics':tabs,'timing':timing,'latency':measure(reference,notes,score,duration),
                'relativeaudio':relative.as_posix(),'probability_path':probability_path.relative_to(output).as_posix(),
                'probability_sha256':sha256(probability_path),'input_provenance':provenance['inputs']['validation'][index],
                'parent_reproduction':proof}
            stem=output/'validation/results'/mode/entry['id']
            export(notes,stem,{'stage':19,'mode':mode,'checkpoint_sha256':sha256(paths[mode]),
                'decoder_parameters':config['decoder_parameters'],'source_delay_seconds':.13,
                'clock_scope':'Fresh paired PCM replay; virtual completion includes measured processing'})
            save_json(stem.with_name(stem.name+'-metrics.json'),row);save_json(stem.with_name(stem.name+'-events.json'),{'events':events});records.append(row)
            print(mode,entry['id'],'state F1',round(heads['exact_state_f1'],4),'onset F1',round(metrics['onset']['f1'],4),flush=True)
    summary=summarize(records);winner=choose({m:summary['modes'][m] for m in MODES[1:]})
    selection={'selected_arm':winner,'eligible_modes':MODES[1:],'rule':config['selection_rule'],
        'scores':{m:{k:summary['modes'][m][k] for k in ['exact_state_f1','onset_f1']} for m in MODES[1:]},
        'selected_from_test':False,'production_promoted':False}
    save_json(output/'selection.json',selection);save_json(output/'validation/summary.json',summary)
    save_json(output/'validation/run.json',{'config':config,'scoring_config':score,'selection':selection,
        'new_optimizer_updates':500,'historical_unique_parent_updates':250,'final_lifetime_updates_per_arm':500,
        'test_inference_calls':0,'production_promoted':False,'checkpoint_paths':{m:p.relative_to(ROOT).as_posix() for m,p in paths.items()},
        'checkpoint_sha256':{m:sha256(p) for m,p in paths.items()},'source_sha256':producer_hashes(),
        'input_provenance':provenance,'training_summary':read(output/'training-summary.json'),
        'scope':'Same restored parent and matched new exposure; only conditional-fret auxiliary supervision differs. Reused development clips, no independent test or default promotion'})
    return summary


def summarize(records):
    result=aggregate(records)
    for mode in MODES:
        rows=[r for r in records if r['mode']==mode];value=result['modes'][mode]
        for key in ['exact_state_f1','active_fret_accuracy']:value[key]=float(np.mean([r['head_metrics'][key] for r in rows]))
        for field,key in [('auxiliary_pitch_f1','overall'),('low_register_pitch_f1','low_register')]:
            value[field]=float(np.mean([r['head_metrics']['pitch'][key]['f1'] for r in rows]))
        value['state_derived_pitch_f1']=float(np.mean([r['state_derived_pitch']['f1'] for r in rows]))
        value['frame_policies']={}
        for policy in ['argmax','decoder_activity']:
            metrics=[r['frame_diagnosis']['policies'][policy] for r in rows]
            value['frame_policies'][policy]={'exact_state_mean_clip_f1':float(np.mean([r['exact_state']['f1'] for r in metrics])),
                'exact_state_pooled':pooled([r['exact_state'] for r in metrics]),'activity_pooled':pooled([r['activity'] for r in metrics]),
                'counts':{k:sum(r['counts'][k] for r in metrics) for k in metrics[0]['counts']},
                'conditional_fret_correct':sum(r['fret_given_reference_and_prediction_active']['correct'] for r in metrics),
                'conditional_fret_count':sum(r['fret_given_reference_and_prediction_active']['count'] for r in metrics)}
        decoder=value['frame_policies']['decoder_activity']
        value['conditional_fret_accuracy']=decoder['conditional_fret_correct']/decoder['conditional_fret_count'] if decoder['conditional_fret_count'] else None
        best=[r['frame_diagnosis']['best_active_fret_on_reference_active'] for r in rows]
        count=sum(r['count'] for r in best);value['best_active_fret_accuracy']=sum(r['correct'] for r in best)/count if count else None
    return result
