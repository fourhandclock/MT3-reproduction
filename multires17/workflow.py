"""Fresh matched feature-fusion training and paired PCM development evaluation."""

import gc
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from acoustic10.decoder import SourceTimeDecoder
from acoustic10.workflow import RecordedDecoder
from baseline.download import ROOT,sha256
from baseline.evaluation import evaluate,reference_notes
from cnn_gru.data import load_audio
from cnn_gru.inference import tablature_metrics
from decoder14.workflow import read
from diagnosis15.analysis import activity_states,pitch_union
from diagnosis9.head_audit import binary_metric
from spectral11.workflow import aligned_heads
from streaming.data import fingerprint
from streaming.evaluate import aggregate,save_json
from streaming.export import export
from streaming.inference import PCMStream
from streaming.latency import measure
from .data import load_inputs,validate_config,model_configuration
from .frontend import MultiResolutionFrontend
from .model import CausalCNNGRU
from .training import initialize,producer_hashes,train_arm,verified_sanity

MODES=['short_short','short_long']


def choose(summaries):
    if set(summaries)!=set(MODES):
        raise ValueError('Both fixed final250 models are required')
    scores={mode:tuple(summaries[mode][k] for k in ['exact_state_f1','onset_f1']) for mode in MODES}
    if any(not np.isfinite(v) or not 0<=v<=1 for score in scores.values() for v in score):
        raise ValueError('Selection metrics must be finite in [0,1]')
    return 'short_long' if scores['short_long']>scores['short_short'] else 'short_short'


def verify_frontends(config,base,records):
    receipts=[]
    for item in records:
        audio,_=load_audio(ROOT/item['entry']['audio'],12,base)
        for arm in config['arms']:
            model_config=model_configuration(base,config,arm)
            full=MultiResolutionFrontend(model_config).push(audio)
            frontend=MultiResolutionFrontend(model_config)
            chunked=np.concatenate([frontend.push(audio[start:start+320]) for start in range(0,len(audio),320)])
            second=item['features'] if arm['name']=='short_short' else item['long_features']
            expected=np.stack([item['features'],second],axis=1)
            if not np.array_equal(full,expected) or not np.array_equal(chunked,expected):
                raise ValueError('Two-plane full/packet/frozen-cache equality failed')
            receipts.append({'id':item['entry']['id'],'mode':arm['name'],'frames':len(full),
                'full_packet_cache_byte_equal':True,'lookahead_samples':0,
                'final_cached_samples':frontend.cached_samples,'first_frame_available_sample':160})
    return receipts


def pcm_replay(model,audio,checkpoint,parameters,device,capture=False):
    stream=PCMStream(model,checkpoint['normalization'],checkpoint['config'],parameters,device)
    stream.frontend=MultiResolutionFrontend(checkpoint['config'])
    stream.decoder=SourceTimeDecoder(checkpoint['config'],parameters,.13)
    if capture:stream.decoder=RecordedDecoder(stream.decoder)
    for start in range(0,len(audio),320):stream.push(audio[start:start+320])
    stream.finish();times=np.asarray(stream.packet_times);duration=len(audio)/16000
    timing={'processing_seconds':float(times.sum()),'input_seconds':duration,
        'real_time_factor':float(times.sum())/duration if duration else None,
        'chunk_processing_p50_ms':float(np.quantile(times,.5)*1000) if len(times) else None,
        'chunk_processing_p95_ms':float(np.quantile(times,.95)*1000) if len(times) else None,
        'packet_ms':20,'lookahead_samples':0,'max_virtual_backlog_ms':max(stream.packet_backlogs or [0])*1000,
        'source_delay_compensation_seconds':.13,'source_clamps':stream.decoder.clamp_counts,
        'peak_frontend_cached_samples':stream.peak_frontend_cache_samples,
        'feature_windows_samples':checkpoint['config']['feature_planes'],
        'unique_frontend_fft_count':len(stream.frontend.frontends),
        'timing_scope':'Warm mono16k PCM push, all feature planes, normalization, neural model and decoder; file read/resampling and hardware capture excluded.',
        'delivery_clock':'max(previous finish, packet arrival)+measured processing; source shift does not advance delivery.',
        'exact_probability_recording_included':capture}
    return stream.notes,stream.events,timing,stream.decoder.probabilities() if capture else None


def run(config,output,device):
    validate_config(config);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol.json').exists():
        raise ValueError('Stage17 already started; its update budget cannot be repeated')
    base,audit,datasets,provenance,schedule=load_inputs(config,output)
    parity=verify_frontends(config,base,datasets['validation'])
    save_json(output/'frontend-parity.json',{'passed':True,'records':parity,'new_optimizer_updates':0})
    sanity=verified_sanity(config)
    save_json(output/'protocol.json',{'config':config,'source_sha256':producer_hashes(),
        'planned_new_optimizer_updates':{mode:250 for mode in MODES},'total_new_optimizer_update_cap':500,
        'historical_warmstart_updates':0,'new_sanity_updates':0,'sanity_reference':sanity,
        'test_inference_allowed':False,'production_promotion':False,'selection_rule':config['selection_rule'],
        'input_audit_sha256':sha256(output/'input-audit.json'),'frontend_parity_sha256':sha256(output/'frontend-parity.json')})
    save_json(output/'crop-schedule.json',{'descriptors':schedule,'sha256':fingerprint(schedule)})
    initialize(config,base,output/'initial-weights.pt')
    receipts={}
    for arm in config['arms']:
        receipts[arm['name']]=train_arm(arm,config,base,audit,datasets['train'],provenance,schedule,output,device)
        gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
    if sum(r['new_optimizer_steps'] for r in receipts.values())!=500:
        raise ValueError('New optimizer update budget differs')
    save_json(output/'training-summary.json',{'config':config,'arms':receipts,'new_optimizer_updates':500,
        'historical_warmstart_updates':0,'final_lifetime_updates_per_arm':250,'new_sanity_updates':0,
        'sampled_crop_epochs_per_arm':5,'validation_used_during_training':False,'test_inference_calls':0,
        'source_sha256':producer_hashes(),'protocol_sha256':sha256(output/'protocol.json')})
    del datasets;gc.collect()
    return evaluate_validation(config,output,device)


def evaluate_validation(config,output,device):
    output=Path(output).resolve();protocol=read(output/'protocol.json')
    if protocol['config']!=config or protocol['source_sha256']!=producer_hashes():
        raise ValueError('Frozen Stage17 code/config changed')
    base,audit,datasets,provenance,_=load_inputs(config,output)
    checkpoints,networks,paths={},{},{}
    for arm in config['arms']:
        mode=arm['name'];path=output/'models'/mode/'final.pt'
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        if (checkpoint['step']!=250 or checkpoint['receipt']['source_sha256']!=producer_hashes()
                or checkpoint['receipt']['input_audit_sha256']!=sha256(output/'input-audit.json')):
            raise ValueError('Final model/source/input binding changed')
        model=CausalCNNGRU(checkpoint['config']).to(device)
        model.load_state_dict(checkpoint['model']);model.eval()
        checkpoints[mode]=checkpoint;networks[mode]=model;paths[mode]=path
    score={k:read(ROOT/'configs/stage1.json')[k] for k in
           ['score_seconds','frame_rate','onset_tolerance_seconds','offset_tolerance_seconds','offset_ratio']}
    warm,_=load_audio(ROOT/datasets['validation'][0]['entry']['audio'],1,base)
    for mode in MODES:pcm_replay(networks[mode],warm,checkpoints[mode],config['decoder_parameters'],device)
    records=[]
    for index,item in enumerate(datasets['validation']):
        entry=item['entry'];audio,duration=load_audio(ROOT/entry['audio'],12,base)
        relative=Path('validation/clips')/(entry['id']+'.wav')
        (output/relative).parent.mkdir(parents=True,exist_ok=True)
        sf.write(output/relative,audio,16000,subtype='FLOAT')
        reference=reference_notes(ROOT/entry['annotation'])
        for mode in MODES[index%2:]+MODES[:index%2]:
            notes,events,timing,probability=pcm_replay(networks[mode],audio,checkpoints[mode],config['decoder_parameters'],device,True)
            probability_path=output/'validation/probabilities'/mode/(entry['id']+'.npz')
            probability_path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(probability_path,**probability)
            heads=aligned_heads(probability,item['targets'],config,base,1000)
            observed,_=activity_states(probability['state'][13:1013],.25)
            valid=np.arange(1000)>=26
            state_pitch=binary_metric(item['targets']['pitch'][:1000][valid].astype(bool),pitch_union(observed)[valid])
            metrics=evaluate(reference,notes,score,duration)
            tabs=tablature_metrics(reference,notes,score,10,duration)
            row={'id':entry['id'],'mode':mode,'notes':notes,'events':events,'reference':reference,
                'head_metrics':heads,'state_derived_pitch':state_pitch,'pitch_metrics':metrics,'tab_metrics':tabs,
                'timing':timing,'latency':measure(reference,notes,score,duration),'relativeaudio':relative.as_posix(),
                'probability_path':probability_path.relative_to(output).as_posix(),'probability_sha256':sha256(probability_path),
                'input_provenance':provenance['inputs']['validation'][index]}
            stem=output/'validation/results'/mode/entry['id']
            export(notes,stem,{'stage':17,'mode':mode,'checkpoint_sha256':sha256(paths[mode]),
                'decoder_parameters':config['decoder_parameters'],'source_delay_seconds':.13,
                'feature_planes':checkpoints[mode]['config']['feature_planes'],
                'clock_scope':'Fresh paired PCM replay; virtual completion includes measured processing'})
            save_json(stem.with_name(stem.name+'-metrics.json'),row)
            save_json(stem.with_name(stem.name+'-events.json'),{'events':events});records.append(row)
            print(mode,entry['id'],'state F1',round(heads['exact_state_f1'],4),'onset F1',round(metrics['onset']['f1'],4),flush=True)
    summary=summarize(records)
    winner=choose(summary['modes'])
    selection={'selected_arm':winner,'eligible_modes':MODES,'rule':config['selection_rule'],
        'scores':{mode:{k:summary['modes'][mode][k] for k in ['exact_state_f1','onset_f1']} for mode in MODES},
        'selected_from_test':False,'production_promoted':False}
    save_json(output/'selection.json',selection);save_json(output/'validation/summary.json',summary)
    save_json(output/'validation/run.json',{'config':config,'scoring_config':score,'selection':selection,
        'new_optimizer_updates':500,'historical_warmstart_updates':0,'final_lifetime_updates_per_arm':250,
        'test_inference_calls':0,'production_promoted':False,
        'checkpoint_paths':{mode:path.relative_to(ROOT).as_posix() for mode,path in paths.items()},
        'checkpoint_sha256':{mode:sha256(path) for mode,path in paths.items()},'source_sha256':producer_hashes(),
        'input_provenance':provenance,'training_summary':read(output/'training-summary.json'),
        'scope':'Fresh matched two-plane feature comparison, same initialization and model capacity; reused validation, no independent test or product promotion'})
    return summary


def summarize(records):
    summary=aggregate(records)
    for mode in MODES:
        rows=[r for r in records if r['mode']==mode]
        for key in ['exact_state_f1','active_fret_accuracy']:
            summary['modes'][mode][key]=float(np.mean([r['head_metrics'][key] for r in rows]))
        summary['modes'][mode]['auxiliary_pitch_f1']=float(np.mean([r['head_metrics']['pitch']['overall']['f1'] for r in rows]))
        summary['modes'][mode]['low_register_pitch_f1']=float(np.mean([r['head_metrics']['pitch']['low_register']['f1'] for r in rows]))
        summary['modes'][mode]['state_derived_pitch_f1']=float(np.mean([r['state_derived_pitch']['f1'] for r in rows]))
    return summary
