import ast
import csv
import gc
import json
import shutil
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from baseline.__main__ import environment
from baseline.download import ROOT,sha256
from baseline.evaluation import evaluate,reference_notes
from baseline.report import summarize
from cnn_gru.data import CropDataset,cache_path,features,load_audio
from cnn_gru.inference import decode,predict_probabilities,read_string_midi,save_midi,tablature_metrics
from cnn_gru.model import GuitarCNNGRU
from .control import common_config,load_controlled_data,source_hashes,verify_sources
from .model import GuitarCNNTransformer


def audit_control(config,manifest,audit,gru_checkpoint):
    reference=common_config(config)
    if gru_checkpoint['config']!=reference or gru_checkpoint['normalization']!=audit['normalization']:
        raise ValueError('Reference checkpoint/configuration/normalization mismatch.')
    if gru_checkpoint['manifest_sha256']!=sha256(ROOT/'data/cnn_gru/manifest.json'):
        raise ValueError('The GRU was trained on a different manifest.')
    for name,digest in gru_checkpoint['source_sha256'].items():
        if sha256(ROOT/'cnn_gru'/name)!=digest: raise ValueError(f'GRU source changed: {name}')
    torch.manual_seed(config['seed']);original=GuitarCNNGRU(reference)
    torch.manual_seed(config['seed']);candidate=GuitarCNNTransformer(config)
    original_state,candidate_state=original.state_dict(),candidate.state_dict()
    shared=('cnn.','projection.','state_head.','onset_head.','offset_head.','pitch_head.')
    shared_keys=[key for key in original_state if key.startswith(shared)]
    if not all(torch.equal(original_state[key],candidate_state[key]) for key in shared_keys):
        raise ValueError('Shared modules did not start at identical weights.')
    def epoch_loop(path):
        module=ast.parse(path.read_text())
        function=next(node for node in module.body if isinstance(node,ast.FunctionDef) and node.name=='train')
        loop=next(node for node in function.body if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='epoch')
        return ast.dump(loop,include_attributes=False)
    if epoch_loop(ROOT/'cnn_gru/training.py')!=epoch_loop(ROOT/'cnn_transformer/training.py'):
        raise ValueError('Optimization/validation loop differs from the GRU baseline.')
    baseline_data=CropDataset(manifest['train'],reference,audit['normalization'])
    candidate_data=CropDataset(manifest['train'],config,audit['normalization'])
    for epoch in [0,4]:
        baseline_data.epoch=candidate_data.epoch=epoch
        for index in [0,23,399]:
            left,left_targets=baseline_data[index];right,right_targets=candidate_data[index]
            if not torch.equal(left,right) or not all(torch.equal(left_targets[key],right_targets[key]) for key in left_targets):
                raise ValueError('Crop schedule, gain augmentation or labels differ.')
    # Validate the same cached features/labels and raw sources used by training.
    for split,entries in manifest.items():
        if split not in ['train','validation','test']: continue
        seconds=config['train_audio_seconds'] if split=='train' else config['evaluation_audio_seconds']
        for entry in entries:
            for key in ['audio','annotation']:
                if sha256(ROOT/entry[key])!=entry[key+'_sha256']: raise ValueError('Source data changed.')
            prefix=cache_path(entry,config,seconds)
            info=json.loads(prefix.with_suffix('.json').read_text())
            for suffix,key in [('.npy','features_sha256'),('.npz','targets_sha256')]:
                if sha256(prefix.with_suffix(suffix))!=info[key]: raise ValueError('Training cache changed.')
    counts={'cnn-gru':sum(p.numel() for p in original.parameters()),
            'cnn-transformer':sum(p.numel() for p in candidate.parameters())}
    return {'same_non_temporal_configuration':True,'identical_shared_initial_weights':True,
            'identical_epoch_loop':True,'same_checked_crops_and_targets':True,'raw_and_cache_checksums_valid':True,
            'same_train_only_normalization':True,'parameter_counts':counts,
            'parameter_difference_percent':100*(counts['cnn-transformer']/counts['cnn-gru']-1),
            'temporal_precision':'float32 for both; shared CNN/projection/heads use the same CUDA AMP policy',
            'seed':config['seed'],'multiple_seeds_run':False}


def compare(config,output,device):
    output=Path(output).resolve()
    manifest,audit=load_controlled_data(config)
    reference_path=ROOT/'outputs/cnn_gru/best-inference.pt'
    gru_checkpoint=torch.load(reference_path,map_location='cpu',weights_only=True)
    transformer_path=output/'best.pt'
    transformer_checkpoint=torch.load(transformer_path,map_location='cpu',weights_only=True)
    if transformer_checkpoint['config']!=config: raise ValueError('Transformer checkpoint config differs.')
    verify_sources(transformer_checkpoint['source_sha256'])
    if transformer_checkpoint['normalization']!=gru_checkpoint['normalization']:
        raise ValueError('Model normalization differs.')
    control=audit_control(config,manifest,audit,gru_checkpoint)
    directory=output/'comparison';directory.mkdir(exist_ok=True)
    (directory/'clips').mkdir(exist_ok=True)
    evaluation_config=json.loads((ROOT/'configs/stage1.json').read_text())
    fps=config['feature']['sample_rate']/config['feature']['hop_length']
    all_records={};provenance={}
    for name,checkpoint,builder,checkpoint_path in [
        ('cnn-gru',gru_checkpoint,GuitarCNNGRU,reference_path),
        ('cnn-transformer',transformer_checkpoint,GuitarCNNTransformer,transformer_path)]:
        model=builder(checkpoint['config']).to(device);model.load_state_dict(checkpoint['model']);model.eval()
        model_config=checkpoint['config'];parameters=checkpoint['decoder_parameters']
        destination=directory/name;destination.mkdir(exist_ok=True)
        # Warm the full input/window-batch shape, not just one short window.
        first_audio,_=load_audio(ROOT/manifest['test'][0]['audio'],config['evaluation_audio_seconds'],config)
        first_features=features(first_audio,config)
        warm_probability=predict_probabilities(model,first_features,checkpoint['normalization'],model_config,device)
        warm_notes=decode(warm_probability,model_config,parameters)
        save_midi(warm_notes,destination/'warmup.mid')
        read_string_midi(destination/'warmup.mid')
        if device.type=='cuda': torch.cuda.synchronize()
        rows=[]
        for index,entry in enumerate(manifest['test']):
            if device.type=='cuda': torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            started=time.perf_counter()
            audio,duration=load_audio(ROOT/entry['audio'],config['evaluation_audio_seconds'],config)
            x=features(audio,config)
            if device.type=='cuda': torch.cuda.synchronize()
            network_started=time.perf_counter()
            probability=predict_probabilities(model,x,checkpoint['normalization'],model_config,device)
            if device.type=='cuda': torch.cuda.synchronize()
            network_times=[time.perf_counter()-network_started]
            predicted=decode(probability,model_config,parameters)
            midi_path=destination/(entry['id']+'.mid');save_midi(predicted,midi_path)
            if device.type=='cuda': torch.cuda.synchronize()
            elapsed=time.perf_counter()-started
            peak=torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None
            # Two extra fixed-weight inference passes stabilize the network measurement.
            for repeat in range(2):
                if device.type=='cuda': torch.cuda.synchronize()
                network_started=time.perf_counter()
                predict_probabilities(model,x,checkpoint['normalization'],model_config,device)
                if device.type=='cuda': torch.cuda.synchronize()
                network_times.append(time.perf_counter()-network_started)
            predicted=read_string_midi(midi_path)
            reference=reference_notes(ROOT/entry['annotation'])
            row=evaluate(reference,predicted,evaluation_config,duration)
            prefix=cache_path(entry,config,config['evaluation_audio_seconds'])
            with np.load(prefix.with_suffix('.npz')) as target:
                tab=tablature_metrics(reference,predicted,evaluation_config,evaluation_config['score_seconds'],duration,probability,target,fps)
            row.update({'id':entry['id'],'model':name,'tablature':tab,'processing_seconds':elapsed,
                        'input_duration_seconds':duration,'real_time_factor':elapsed/duration,'drum_notes':0,
                        'gpu_peak_allocated_mib':peak,'network_seconds_repeats':network_times,
                        'median_network_seconds':statistics.median(network_times)})
            (destination/(entry['id']+'.json')).write_text(json.dumps(row,indent=2),encoding='utf-8')
            np.savez_compressed(destination/(entry['id']+'.npz'),**probability)
            with (destination/(entry['id']+'-tab.csv')).open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=['start','end','string','fret','pitch','program'])
                writer.writeheader();writer.writerows(predicted)
            shutil.copyfile(ROOT/'outputs/stage1/clips'/(entry['id']+'.wav'),directory/'clips'/(entry['id']+'.wav'))
            rows.append(row)
            print(f'{name} {index+1}/12: onset F1={row["onset"]["f1"]:.3f}, '
                  f'pipeline={elapsed:.3f}s, network median={row["median_network_seconds"]:.3f}s',flush=True)
        all_records[name]=rows
        provenance[name]={'checkpoint_sha256':sha256(checkpoint_path),'checkpoint_path':str(checkpoint_path),
                          'selected_epoch':checkpoint['history'][-1]['epoch'],'trained_steps':checkpoint['step'],
                          'validation_onset_f1':checkpoint.get('decoder_validation',checkpoint['history'][-1]['validation'])['onset_f1'],
                          'decoder_parameters':parameters,'parameter_count':checkpoint['parameter_count']}
        del model,warm_probability,probability
        gc.collect()
        if device.type=='cuda': torch.cuda.empty_cache()
    summary=summarize(all_records)
    for name,rows in all_records.items():
        summary[name]['tab_onset_f1']=float(np.mean([row['tablature']['onset']['f1'] for row in rows]))
        summary[name]['tab_onset_offset_f1']=float(np.mean([row['tablature']['onset_offset']['f1'] for row in rows]))
        summary[name]['median_network_seconds_total']=sum(row['median_network_seconds'] for row in rows)
    (directory/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    run={'selected_ids':[entry['id'] for entry in manifest['test']],'control_checks':control,
         'models':provenance,'environment':environment(),'device':str(device),
         'manifest_sha256':sha256(ROOT/'data/cnn_gru/manifest.json'),'config':config,
         'common_source_sha256':source_hashes(),'comparison_source_sha256':sha256(ROOT/'cnn_transformer/compare.py'),
         'timing_scope':'Same process/device and full-shape warmup. Pipeline includes fresh file read, resampling, CQT, inference, decoder and MIDI write. Network medians include normalization, window batching, host/device transfers and probability stitching; three passes per excerpt.',
         'optimizer_updates_added_by_comparison':0}
    (directory/'run.json').write_text(json.dumps(run,indent=2),encoding='utf-8')
    from .report import write_report
    write_report(output,all_records,summary,run)
