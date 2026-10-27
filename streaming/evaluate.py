import gc
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from baseline.download import ROOT,sha256
from baseline.evaluation import evaluate,reference_notes
from cnn_gru.data import load_audio,features as old_features
from cnn_gru.inference import decode,predict_probabilities,tablature_metrics
from cnn_gru.model import GuitarCNNGRU
from .export import export
from .inference import replay
from .latency import measure,quantiles
from .model import CausalCNNGRU


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def load_checkpoint(output,device):
    path=Path(output)/'model/best-inference.pt'
    if not path.exists():path=Path(output)/'model/best.pt'
    checkpoint=torch.load(path,map_location=device,weights_only=True)
    for name,digest in checkpoint['source_sha256'].items():
        if sha256(ROOT/'streaming'/name)!=digest:raise ValueError('Selected causal source changed: '+name)
    if checkpoint['manifest_sha256']!=sha256(ROOT/'data/cnn_gru/manifest.json'):raise ValueError('Source split changed.')
    if checkpoint['legacy_loss_sha256']!=sha256(ROOT/'cnn_gru/model.py'):raise ValueError('Shared loss source changed.')
    model=CausalCNNGRU(checkpoint['config']).to(device);model.load_state_dict(checkpoint['model']);model.eval()
    return path,checkpoint,model


def evaluate_streaming(output,device):
    output=Path(output).resolve();path,checkpoint,model=load_checkpoint(output,device);config=checkpoint['config']
    manifest=json.loads((ROOT/'data/cnn_gru/manifest.json').read_text());score_config=checkpoint['scoring_config']
    directory=output/'evaluation';(directory/'clips').mkdir(parents=True,exist_ok=True)
    warm_entry=manifest['validation'][0];warm_audio,_=load_audio(ROOT/warm_entry['audio'],1,config)
    for packet in config['streaming']['evaluation_packet_ms']:
        replay(model,warm_audio,checkpoint['normalization'],config,checkpoint['decoder_parameters'],device,packet)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    records=[]
    for entry in manifest['test']:
        for field in ['audio','annotation']:
            if sha256(ROOT/entry[field])!=entry[field+'_sha256']:raise ValueError('Original evaluation source changed.')
        audio,duration=load_audio(ROOT/entry['audio'],config['evaluation_audio_seconds'],config)
        relative=Path('evaluation/clips')/(entry['id']+'.wav');sf.write(output/relative,audio,config['feature']['sample_rate'],subtype='FLOAT')
        reference=reference_notes(ROOT/entry['annotation'])
        for packet in config['streaming']['evaluation_packet_ms']:
            mode=f'causal_{packet}ms'
            notes,events,timing=replay(model,audio,checkpoint['normalization'],config,checkpoint['decoder_parameters'],device,packet)
            metrics=evaluate(reference,notes,score_config,duration)
            tab=tablature_metrics(reference,notes,score_config,score_config['score_seconds'],duration)
            latency=measure(reference,notes,score_config,duration)
            record={'id':entry['id'],'mode':mode,'notes':notes,'reference':reference,'events':events,'pitch_metrics':metrics,
                    'tab_metrics':tab,'latency':latency,'timing':timing,'relativeaudio':relative.as_posix()}
            stem=directory/'results'/mode/entry['id']
            export(notes,stem,{'mode':mode,'checkpoint_sha256':sha256(path),'source_time':'Estimated note interval seconds',
                               'delivery_time':'Incremental events use virtual replay processing clock; no hardware capture.'})
            save_json(stem.with_name(stem.name+'-events.json'),{'mode':mode,'events':events})
            save_json(stem.with_name(stem.name+'-metrics.json'),record);records.append(record)
            print(mode,entry['id'],'onset F1',round(metrics['onset']['f1'],3),
                  'matched latency',latency['matched_onset_delivery_p50_ms'],flush=True)
    peak=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None
    del model;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    offline_path=ROOT/'outputs/cnn_gru/best-inference.pt';offline=torch.load(offline_path,map_location=device,weights_only=True)
    for name,digest in offline['source_sha256'].items():
        if sha256(ROOT/'cnn_gru'/name)!=digest:raise ValueError('Frozen offline comparison changed.')
    old_model=GuitarCNNGRU(offline['config']).to(device);old_model.load_state_dict(offline['model']);old_model.eval()
    old_audio,_=load_audio(ROOT/warm_entry['audio'],config['evaluation_audio_seconds'],offline['config'])
    predict_probabilities(old_model,old_features(old_audio,offline['config']),offline['normalization'],offline['config'],device)
    for entry in manifest['test']:
        if device.type=='cuda':torch.cuda.synchronize(device)
        started=time.perf_counter();audio,duration=load_audio(ROOT/entry['audio'],config['evaluation_audio_seconds'],offline['config'])
        probabilities=predict_probabilities(old_model,old_features(audio,offline['config']),offline['normalization'],offline['config'],device)
        notes=decode(probabilities,offline['config'],offline['decoder_parameters'])
        if device.type=='cuda':torch.cuda.synchronize(device)
        seconds=time.perf_counter()-started;reference=reference_notes(ROOT/entry['annotation'])
        record={'id':entry['id'],'mode':'offline_bidir','notes':notes,'reference':reference,'events':[],
                'pitch_metrics':evaluate(reference,notes,score_config,duration),
                'tab_metrics':tablature_metrics(reference,notes,score_config,score_config['score_seconds'],duration),
                'timing':{'processing_seconds':seconds,'input_seconds':duration,'real_time_factor':seconds/duration},
                'latency':{},'relativeaudio':f'evaluation/clips/{entry["id"]}.wav'}
        stem=directory/'results'/'offline_bidir'/entry['id'];export(notes,stem,{'mode':'offline_bidir','checkpoint_sha256':sha256(offline_path)})
        save_json(stem.with_name(stem.name+'-metrics.json'),record);records.append(record)
    summary=aggregate(records);save_json(directory/'summary.json',summary)
    training=json.loads((output/'model/training_summary.json').read_text())
    discarded_path=output/'discarded-initial/run.json'
    discarded=json.loads(discarded_path.read_text()) if discarded_path.exists() else {'optimizer_updates':0}
    run={'config':config,'new_training_epochs':training['epochs_completed'],'optimizer_updates':training['optimizer_steps'],
         'discarded_optimizer_updates':discarded['optimizer_updates'],
         'total_stage_optimizer_updates':training['optimizer_steps']+discarded['optimizer_updates'],
         'model_parameters':checkpoint['model_parameters'],'checkpoint_sha256':sha256(path),
         'offline_checkpoint_sha256':sha256(offline_path),'decoder_parameters':checkpoint['decoder_parameters'],
         'training_summary':training,'peak_gpu_mib':peak,'device':str(device),'lookahead_samples':0,
         'inference_calibration':checkpoint.get('inference_calibration'),
         'input_scope':config['streaming']['input_scope'],'past_window_ms':1000*config['feature']['window_length']/config['feature']['sample_rate'],
         'frame_timestamp':'Frame i available at(i+1)*hop/sample_rate; no incomplete-hop EOF padding.',
         'scoring_config':score_config,'manifest_sha256':sha256(ROOT/'data/cnn_gru/manifest.json'),
         'validation_rule':'Progression-2 note onset F1 then onset+offset F1; thresholds and source onset backdate only.',
         'latency_rule':'Pooled signed delivery delays of one-to-one matched notes. EOF flush and censored reference releases excluded from release feedback.',
         'comparison_limit':'Changed frontend, normalization, recurrence and decoding; not a matched isolated encoder comparison.',
         'source_sha256':{p.name:sha256(p) for p in sorted((ROOT/'streaming').glob('*.py'))}}
    save_json(directory/'run.json',run)
    from .report import write_report
    write_report(output,records,summary,run)
    return summary


def aggregate(records):
    result={'modes':{}}
    for mode in sorted(set(r['mode'] for r in records)):
        rows=[r for r in records if r['mode']==mode]
        on=[v for r in rows for v in r['latency'].get('onset_delivery_samples_ms',[])];off=[v for r in rows for v in r['latency'].get('offset_delivery_samples_ms',[])]
        oq=quantiles(on);fq=quantiles(off)
        result['modes'][mode]={'count':len(rows),'onset_f1':float(np.mean([r['pitch_metrics']['onset']['f1'] for r in rows])),
            'onset_offset_f1':float(np.mean([r['pitch_metrics']['onset_offset']['f1'] for r in rows])),
            'frame_f1':float(np.mean([r['pitch_metrics']['frame']['f1'] for r in rows])),
            'tab_onset_f1':float(np.mean([r['tab_metrics']['onset']['f1'] for r in rows])),
            'real_time_factor':sum(r['timing']['processing_seconds'] for r in rows)/sum(r['timing']['input_seconds'] for r in rows),
            'matched_onset_count':len(on),'matched_offset_count':len(off),
            'matched_onset_delivery_p50_ms':oq['p50'],'matched_onset_delivery_p95_ms':oq['p95'],
            'matched_offset_delivery_p50_ms':fq['p50'],'matched_offset_delivery_p95_ms':fq['p95'],
            'negative_onset_delivery_count':sum(v<0 for v in on),
            'censored_terminal_release_count':sum(r['latency'].get('censored_terminal_release_count',0) for r in rows)}
    return result
