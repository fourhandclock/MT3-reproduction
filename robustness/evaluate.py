import ast
import copy
import csv
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from baseline.__main__ import environment
from baseline.download import ROOT,sha256
from baseline.evaluation import evaluate,reference_notes
from cnn_gru.data import cache_path,features,load_audio
from cnn_gru.inference import decode,predict_probabilities,read_string_midi,save_midi,tablature_metrics
from cnn_gru.model import GuitarCNNGRU
from .control import source_hashes,validate_prepared
from .data import effect_cache
from .validation import validate


def condition_summary(rows):
    result={}
    for condition in sorted({row['condition'] for row in rows}):
        group=[row for row in rows if row['condition']==condition]
        result[condition]={'count':len(group),'onset_f1':float(np.mean([row['onset']['f1'] for row in group])),
                           'onset_offset_f1':float(np.mean([row['onset_offset']['f1'] for row in group])),
                           'frame_f1':float(np.mean([row['frame']['f1'] for row in group])),
                           'tab_onset_f1':float(np.mean([row['tablature']['onset']['f1'] for row in group])),
                           'fixed_baseline_threshold_onset_f1':float(np.mean([row['fixed_baseline_threshold_onset_f1'] for row in group]))}
        if condition=='real_electric':
            result[condition]['onset_f1_100ms_diagnostic']=float(np.mean([row['onset_f1_100ms_diagnostic'] for row in group]))
    return result


def optimization_loop(path):
    module=ast.parse(path.read_text())
    function=next(node for node in module.body if isinstance(node,ast.FunctionDef) and node.name=='train')
    return ast.dump(next(node for node in function.body if isinstance(node,ast.For) and isinstance(node.target,ast.Name)
                         and node.target.id=='epoch'),include_attributes=False)


def evaluate_models(config,output,device,only_baseline=False):
    import soundfile as sf
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    manifest,audit=validate_prepared(config)
    evaluation_config=json.loads((ROOT/'configs/stage1.json').read_text())
    baseline_path=ROOT/'outputs/cnn_gru/best-inference.pt'
    baseline=torch.load(baseline_path,map_location='cpu',weights_only=True)
    for name,digest in baseline['source_sha256'].items():
        if sha256(ROOT/'cnn_gru'/name)!=digest: raise ValueError('Baseline source changed.')
    if baseline['normalization']!=audit['normalization']: raise ValueError('Normalization changed.')
    external_path=ROOT/'data/robustness/egset12/manifest.json'
    external=json.loads(external_path.read_text()) if external_path.exists() else None
    fps=config['feature']['sample_rate']/config['feature']['hop_length']
    destination=output/('baseline_stress' if only_baseline else 'evaluation')
    destination.mkdir(exist_ok=True);(destination/'clips').mkdir(exist_ok=True)
    models=[('clean-trained',baseline,baseline_path)]
    if not only_baseline:
        path=output/'augmented/best.pt';augmented=torch.load(path,map_location='cpu',weights_only=True)
        if augmented['config']!=config or augmented['source_sha256']!=source_hashes(): raise ValueError('Augmented checkpoint/source mismatch.')
        if augmented['normalization']!=baseline['normalization']: raise ValueError('Trial changed normalization.')
        if optimization_loop(ROOT/'cnn_gru/training.py')!=optimization_loop(ROOT/'robustness/training.py'):
            raise ValueError('The optimization loop changed beyond data augmentation/model selection.')
        models.append(('effect-augmented',augmented,path))
    all_records={};provenance={};baseline_parameters=None
    for name,checkpoint,path in models:
        model=GuitarCNNGRU(config).to(device);model.load_state_dict(checkpoint['model']);model.eval()
        # Give the frozen baseline the same validation conditions/grid as the trial.
        validation=validate(model,manifest,audit,config,evaluation_config,device)
        parameters=validation['parameters']
        if name=='clean-trained': baseline_parameters=parameters.copy()
        provenance[name]={'checkpoint_sha256':sha256(path),'checkpoint_path':str(path),'selected_epoch':checkpoint['history'][-1]['epoch'],
                          'trained_steps':checkpoint['step'],'robust_validation':validation,
                          'original_decoder_parameters':checkpoint['decoder_parameters']}
        folder=destination/name;folder.mkdir(exist_ok=True)
        rows=[]
        for entry in manifest['test']:
            base=cache_path(entry,config,config['evaluation_audio_seconds'])
            duration=json.loads(base.with_suffix('.json').read_text())['duration']
            reference=reference_notes(ROOT/entry['annotation'])
            for condition in config['robustness']['test_conditions']:
                prefix=base if condition=='clean' else effect_cache(entry,config,'test',condition)
                x=np.load(prefix.with_suffix('.npy'),mmap_mode='r')
                probability=predict_probabilities(model,x,audit['normalization'],config,device)
                notes=decode(probability,config,parameters)
                stem=entry['id']+'__'+condition;midi_path=folder/(stem+'.mid')
                save_midi(notes,midi_path);notes=read_string_midi(midi_path)
                row=evaluate(reference,notes,evaluation_config,duration)
                tab=tablature_metrics(reference,notes,evaluation_config,evaluation_config['score_seconds'],duration)
                fixed=evaluate(reference,decode(probability,config,baseline_parameters),evaluation_config,duration)['onset']['f1']
                row.update({'id':entry['id'],'condition':condition,'model':name,'tablature':tab,
                            'fixed_baseline_threshold_onset_f1':fixed,'data_kind':'processed acoustic recording' if condition!='clean' else 'original acoustic recording'})
                (folder/(stem+'.json')).write_text(json.dumps(row,indent=2),encoding='utf-8')
                with (folder/(stem+'-tab.csv')).open('w',newline='',encoding='utf-8') as stream:
                    writer=csv.DictWriter(stream,fieldnames=['start','end','string','fret','pitch','program'])
                    writer.writeheader();writer.writerows(notes)
                waveform=ROOT/'outputs/stage1/clips'/(entry['id']+'.wav') if condition=='clean' else effect_cache(entry,config,'test',condition).with_suffix('.wav')
                shutil.copyfile(waveform,destination/'clips'/(stem+'.wav'))
                rows.append(row)
            print('Scored',name,entry['id'],'all synthetic conditions',flush=True)
        if external:
            for entry in external['excerpts']:
                for key in ['audio','annotation']:
                    if sha256(ROOT/entry[key])!=entry[key+'_sha256']: raise ValueError('External data checksum changed.')
                audio,duration=load_audio(ROOT/entry['audio'],config['robustness']['external_clip_seconds'],config)
                probability=predict_probabilities(model,features(audio,config),audit['normalization'],config,device)
                notes=decode(probability,config,parameters)
                stem=entry['id']+'__real_electric';midi_path=folder/(stem+'.mid')
                save_midi(notes,midi_path);notes=read_string_midi(midi_path)
                reference=reference_notes(ROOT/entry['annotation'])
                row=evaluate(reference,notes,evaluation_config,duration)
                loose=copy.deepcopy(evaluation_config);loose['onset_tolerance_seconds']=0.1
                diagnostic=evaluate(reference,notes,loose,duration)['onset']['f1']
                tab=tablature_metrics(reference,notes,evaluation_config,evaluation_config['score_seconds'],duration)
                fixed=evaluate(reference,decode(probability,config,baseline_parameters),evaluation_config,duration)['onset']['f1']
                row.update({'id':entry['id'],'condition':'real_electric','model':name,'tablature':tab,
                            'fixed_baseline_threshold_onset_f1':fixed,'onset_f1_100ms_diagnostic':diagnostic,
                            'data_kind':'original external electric-guitar performance'})
                (folder/(stem+'.json')).write_text(json.dumps(row,indent=2),encoding='utf-8')
                with (folder/(stem+'-tab.csv')).open('w',newline='',encoding='utf-8') as stream:
                    writer=csv.DictWriter(stream,fieldnames=['start','end','string','fret','pitch','program'])
                    writer.writeheader();writer.writerows(notes)
                sf.write(destination/'clips'/(stem+'.wav'),audio,config['feature']['sample_rate'],subtype='FLOAT')
                rows.append(row);print('Scored',name,entry['id'],'real electric',flush=True)
        all_records[name]=rows
        del model,probability;gc.collect()
        if device.type=='cuda': torch.cuda.empty_cache()
    summary={name:condition_summary(rows) for name,rows in all_records.items()}
    (destination/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    run={'config':config,'models':provenance,'environment':environment(),'external_manifest_sha256':sha256(external_path) if external else None,
         'base_manifest_sha256':sha256(ROOT/'data/cnn_gru/manifest.json'),'source_sha256':source_hashes(),
         'evaluation_source_sha256':sha256(ROOT/'robustness/evaluate.py'),'optimizer_updates_added_by_evaluation':0,
         'external_used_for_training_or_selection':False,'synthetic_timing_policy':'No added fixed delay; labels unchanged.',
         'scored_prefix_seconds':evaluation_config['score_seconds'],'input_seconds':config['evaluation_audio_seconds']}
    (destination/'run.json').write_text(json.dumps(run,indent=2),encoding='utf-8')
    print('Saved robustness scores:',destination/'summary.json',flush=True)
    if not only_baseline:
        from .report import write_report
        write_report(output,all_records,summary,run)
