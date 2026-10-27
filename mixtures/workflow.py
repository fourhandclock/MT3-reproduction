import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from baseline.download import ROOT, sha256
from baseline.evaluation import evaluate, reference_notes
from cnn_gru.data import load_audio, features, validate_splits
from cnn_gru.inference import decode, predict_probabilities, save_midi, tablature_metrics
from cnn_gru.model import GuitarCNNGRU
from .audio import make_accompaniment, mix
from .metrics import separation_metrics


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def ratio_suffix(ratio):
    return f'{ratio:+g}' if ratio>0 else f'{ratio:g}'


def validate_sources(source):
    for entry in source['test']:
        for field in ['audio','annotation']:
            if sha256(ROOT/entry[field]) != entry[field+'_sha256']:
                raise ValueError('Original source audio/annotation changed.')


def validate_coverage(entries,source,conditions):
    keys=[(e['id'],e['condition']) for e in entries]
    expected={(e['id'],condition) for e in source['test'] for condition in conditions}
    if len(keys)!=len(set(keys)) or set(keys)!=expected:
        raise ValueError('Incomplete, duplicate or unexpected evaluation conditions.')


def prepare_data(config, output):
    output = Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    source_path = ROOT/'data/cnn_gru/manifest.json'; source = json.loads(source_path.read_text()); validate_splits(source)
    validate_sources(source)
    renderer_sha256=sha256(ROOT/'mixtures/audio.py')
    base = json.loads((ROOT/'configs/cnn_gru.json').read_text())
    if config['sample_rate'] != base['feature']['sample_rate']: raise ValueError('Diagnostic render sample rate must match acoustic model.')
    path = output/'data-manifest.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['config'] != config or saved['source_manifest_sha256'] != sha256(source_path):
            raise ValueError('Existing output belongs to a different fixed mixture recipe.')
        if saved.get('renderer_sha256')!=renderer_sha256: raise ValueError('Cached waveform renderer source differs.')
        rendered_conditions=[c for c in config['conditions'] if not c.startswith('separated_')]
        validate_coverage(saved['inputs'],source,rendered_conditions)
        for entry in saved['inputs']:
            if sha256(output/entry['audio']) != entry['audio_sha256']: raise ValueError('Rendered input changed.')
        return saved
    entries = []; render_metadata = []
    for source_entry in source['test']:
        for field in ['audio','annotation']:
            if sha256(ROOT/source_entry[field]) != source_entry[field+'_sha256']: raise ValueError('Source input changed.')
        guitar,duration = load_audio(ROOT/source_entry['audio'],config['input_seconds'],base)
        jams = json.loads((ROOT/source_entry['annotation']).read_text())
        beats = next(a['data'] for a in jams['annotations'] if a['namespace']=='beat_position')
        # First chord namespace is the instructed lead sheet; performed chords partly derive from guitar-note annotations.
        chords = next(a['data'] for a in jams['annotations'] if a['namespace']=='chord')
        seed = config['seed'] + int(hashlib.sha256(source_entry['id'].encode()).hexdigest()[:8],16)
        backing, metadata = make_accompaniment(duration,config['sample_rate'],beats,chords,seed)
        variants = [('clean',None,guitar,None)]
        mixes = {}
        for ratio in config['guitar_to_accompaniment_db']:
            suffix = ratio_suffix(ratio); value = mix(guitar,backing,ratio); mixes[suffix]=value
            variants += [('target_'+suffix,ratio,value['target'],value['metadata']),
                         ('mixture_'+suffix,ratio,value['mixture'],value['metadata'])]
        if '0' not in mixes: raise ValueError('Zero-dB mixture required for accompaniment-only control.')
        variants.append(('accompaniment_only',None,mixes['0']['accompaniment'],mixes['0']['metadata']))
        for condition,ratio,audio,mixing in variants:
            relative = Path('audio')/condition/(source_entry['id']+'.wav'); target=output/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            sf.write(target,audio,config['sample_rate'],subtype='FLOAT')
            entries.append({'id':source_entry['id'],'condition':condition,'ratio_db':ratio,
                            'source':source_entry,'audio':relative.as_posix(),'audio_sha256':sha256(target),
                            'duration_seconds':duration,'mixing':mixing})
        render_metadata.append({'id':source_entry['id'],'metadata':metadata})
        print('Rendered controlled mixtures:',source_entry['id'],flush=True)
    saved = {'dataset':'GuitarSet + procedural backing, controlled synthetic diagnostic',
             'config':config,'source_manifest_sha256':sha256(source_path),'renderer_sha256':renderer_sha256,'inputs':entries,
             'synthesis':render_metadata,'selection':'All twelve progression-3 excerpts; recipe prespecified; no selection on results.',
             'truth_use':'Instructed beat/chord lead sheet creates synthetic backing. Guitar note/string labels are not used for rendering or model inference.',
             'training_updates':0,'sample_alignment':'Only scalar guitar gain; exact original resampled sample count; no label shifting.'}
    write_json(path,saved); return saved


def separate_data(config,output,device):
    import torch
    from .separator import Separator
    output=Path(output).resolve(); manifest=prepare_data(config,output)
    started=time.perf_counter(); separator=Separator(config,device); load_seconds=time.perf_counter()-started
    base=json.loads((ROOT/'configs/cnn_gru.json').read_text())
    source=json.loads((ROOT/'data/cnn_gru/manifest.json').read_text())
    warm_source=source['validation'][0]
    warm_audio,_=load_audio(ROOT/warm_source['audio'],config['input_seconds'],base)
    _,warm_timing=separator.separate(warm_audio,config['sample_rate'])
    inputs=[e for e in manifest['inputs'] if e['condition']=='clean' or e['condition'].startswith('mixture_')]
    rendered=[]; reused=0
    for index,entry in enumerate(inputs,1):
        condition='separated_clean' if entry['condition']=='clean' else entry['condition'].replace('mixture_','separated_')
        relative=Path('audio')/condition/(entry['id']+'.wav'); path=output/relative
        log_path=output/'separation'/condition/(entry['id']+'.json')
        expected={'input_audio_sha256':entry['audio_sha256'],'separator_sha256':separator.provenance['sha256'],
                  'config_fingerprint':fingerprint(config),'source_condition':entry['condition'],
                  'separator_wrapper_sha256':sha256(ROOT/'mixtures/separator.py')}
        if log_path.exists():
            log=json.loads(log_path.read_text())
            if any(log.get(k)!=v for k,v in expected.items()) or sha256(path)!=log['audio_sha256']:
                raise ValueError('Cached separation does not match frozen inputs/settings.')
            reused+=1
        else:
            audio,sr=sf.read(output/entry['audio'],dtype='float32')
            guitar,timing=separator.separate(audio,sr)
            path.parent.mkdir(parents=True,exist_ok=True); sf.write(path,guitar,sr,subtype='FLOAT')
            target_condition='clean' if entry['condition']=='clean' else entry['condition'].replace('mixture_','target_')
            target=next(e for e in manifest['inputs'] if e['id']==entry['id'] and e['condition']==target_condition)
            target_audio,_=sf.read(output/target['audio'],dtype='float32')
            metrics=separation_metrics(target_audio,guitar,sr)
            # Mixture against the exact gain-matched guitar target is a useful signal reference.
            input_metrics=separation_metrics(target_audio,audio,sr)
            log={**expected,'id':entry['id'],'condition':condition,'ratio_db':entry['ratio_db'],
                 'audio':relative.as_posix(),'audio_sha256':sha256(path),'timing':timing,
                 'separation_metrics':metrics,'input_waveform_metrics':input_metrics,
                 'target_audio_sha256':target['audio_sha256']}
            write_json(log_path,log)
        rendered.append({**entry,'condition':condition,'audio':relative.as_posix(),'audio_sha256':log['audio_sha256'],
                         'separation':log})
        print(f'Separated {index}/{len(inputs)} {entry["id"]} {entry["condition"]}: '
              f'{log["timing"]["separation_seconds"]:.2f}s',flush=True)
    result={'entries':rendered,'model_provenance':separator.provenance,'model_parameters':separator.parameters,
            'config_fingerprint':fingerprint(config),'separator_wrapper_sha256':sha256(ROOT/'mixtures/separator.py'),
            'model_sources':list(separator.model.sources),'guitar_source_index':separator.guitar_index,
            'native_segment_seconds':float(separator.original_segment),'cold_load_seconds':load_seconds,
            'warmup':warm_timing,'warmup_source_id':warm_source['id'],'peak_gpu_mib':separator.peak_memory/2**20,
            'device':str(device),'cache_reused_count':reused}
    write_json(output/'separation-run.json',result)
    del separator; gc.collect()
    if torch.device(device).type=='cuda': torch.cuda.empty_cache()
    return manifest,result


def transcribe_data(config,output,device):
    import torch
    output=Path(output).resolve(); manifest=prepare_data(config,output)
    separated=json.loads((output/'separation-run.json').read_text())
    from .separator import verify
    _,provenance=verify(config)
    if separated['model_provenance']!=provenance or separated.get('config_fingerprint')!=fingerprint(config):
        raise ValueError('Separation run does not match model/settings.')
    if separated.get('separator_wrapper_sha256')!=sha256(ROOT/'mixtures/separator.py'):
        raise ValueError('Separation wrapper source changed.')
    source=json.loads((ROOT/'data/cnn_gru/manifest.json').read_text())
    validate_coverage(manifest['inputs']+separated['entries'],source,config['conditions'])
    checkpoint_path=ROOT/'outputs/cnn_gru/best-inference.pt'
    load_start=time.perf_counter(); checkpoint=torch.load(checkpoint_path,map_location=device,weights_only=True)
    if checkpoint['manifest_sha256']!=sha256(ROOT/'data/cnn_gru/manifest.json'): raise ValueError('Acoustic dataset manifest changed.')
    for name,digest in checkpoint['source_sha256'].items():
        if sha256(ROOT/'cnn_gru'/name)!=digest: raise ValueError('Frozen acoustic source changed.')
    base=checkpoint['config']; model=GuitarCNNGRU(base).to(device);model.load_state_dict(checkpoint['model']);model.eval()
    if torch.device(device).type=='cuda': torch.cuda.synchronize(device)
    load_seconds=time.perf_counter()-load_start
    score_config=json.loads((ROOT/'configs/stage1.json').read_text())
    if score_config['score_seconds']!=config['score_seconds']: raise ValueError('Scoring duration mismatch.')
    source=json.loads((ROOT/'data/cnn_gru/manifest.json').read_text())
    warm_audio,_=load_audio(ROOT/source['validation'][0]['audio'],config['input_seconds'],base)
    predict_probabilities(model,features(warm_audio,base),checkpoint['normalization'],base,torch.device(device))
    entries=manifest['inputs']+separated['entries']; records=[]; peak=0
    from fingering.export import export
    for index,entry in enumerate(entries,1):
        if sha256(output/entry['audio'])!=entry['audio_sha256']: raise ValueError('Evaluation audio changed.')
        if torch.device(device).type=='cuda': torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
        started=time.perf_counter()
        audio,sr=sf.read(output/entry['audio'],dtype='float32')
        if sr!=base['feature']['sample_rate']: raise ValueError('Rendered audio sample-rate mismatch.')
        probability=predict_probabilities(model,features(audio,base),checkpoint['normalization'],base,torch.device(device))
        notes=decode(probability,base,checkpoint['decoder_parameters'])
        if torch.device(device).type=='cuda':
            torch.cuda.synchronize(device);peak=max(peak,torch.cuda.max_memory_allocated(device)/2**20)
        elapsed=time.perf_counter()-started
        reference=[] if entry['condition']=='accompaniment_only' else reference_notes(ROOT/entry['source']['annotation'])
        pitch=evaluate(reference,notes,score_config,entry['duration_seconds'])
        tab=tablature_metrics(reference,notes,score_config,config['score_seconds'],entry['duration_seconds'])
        log=entry.get('separation'); separation_seconds=log['timing']['separation_seconds'] if log else 0.0
        record={'id':entry['id'],'condition':entry['condition'],'ratio_db':entry['ratio_db'],'audio':entry['audio'],
                'notes':notes,'reference':reference,'pitch_metrics':pitch,'tab_metrics':tab,
                'timing':{'transcription_seconds':elapsed,'separation_seconds':separation_seconds,
                          'total_measured_seconds':elapsed+separation_seconds,'input_seconds':entry['duration_seconds'],
                          'real_time_factor':(elapsed+separation_seconds)/entry['duration_seconds']},
                'separation_metrics':log['separation_metrics'] if log else None,
                'input_waveform_metrics':log['input_waveform_metrics'] if log else None,
                'false_positive_count':pitch['onset']['predicted_count'] if entry['condition']=='accompaniment_only' else None,
                'mixing':entry.get('mixing'),'audio_sha256':entry['audio_sha256']}
        stem=output/'results'/entry['condition']/entry['id']
        export(notes,stem,{'mode':'frozen acoustic raw string/fret outputs','condition':entry['condition'],
                           'checkpoint_sha256':sha256(checkpoint_path),'audio':entry['audio']})
        write_json(stem.with_name(stem.name+'-metrics.json'),record);records.append(record)
        print(f'Transcribed {index}/{len(entries)} {entry["id"]} {entry["condition"]}: onset F1 {pitch["onset"]["f1"]:.3f}',flush=True)
    summary=aggregate(records);write_json(output/'summary.json',summary)
    run={'config':config,'model_provenance':separated['model_provenance'],'separator_parameters':separated['model_parameters'],
         'model_sources':separated['model_sources'],'guitar_source_index':separated['guitar_source_index'],
         'native_segment_seconds':separated['native_segment_seconds'],'device':str(device),
         'peak_separator_gpu_mib':separated['peak_gpu_mib'],'peak_acoustic_gpu_mib':peak,
         'separator_cold_load_seconds':separated['cold_load_seconds'],'acoustic_cold_load_seconds':load_seconds,
         'separator_warmup':separated['warmup'],'cache_reused_count':separated['cache_reused_count'],
         'acoustic_checkpoint_sha256':sha256(checkpoint_path),'acoustic_decoder':checkpoint['decoder_parameters'],
         'acoustic_normalization':'Frozen, train-only original normalization; no mixture recalibration.',
         'optimizer_updates':0,'new_training_epochs':0,'source_manifest_sha256':manifest['source_manifest_sha256'],
         'scoring_config':score_config,'scoring_config_sha256':sha256(ROOT/'configs/stage1.json'),
         'data_manifest_sha256':sha256(output/'data-manifest.json'),
         'source_hashes':{p.name:sha256(p) for p in sorted((ROOT/'mixtures').glob('*.py'))},
         'timing_scope':'Warm separation includes resampling and all stem extraction work. Warm transcription includes float WAV read, fresh CQT, normalization, network and note decode; excludes model loading, synthesis, disk exports, metrics, fingering beam search, beat/chord/notation.',
         'model_execution':'Separator phase ends and releases its model before acoustic phase; no simultaneous models.',
         'fixed_recipe_no_test_selection':True,'synthetic_diagnostic_only':True}
    write_json(output/'run.json',run)
    from .report import write_report
    write_report(output,records,summary,run)
    return summary


def aggregate(records):
    result={'conditions':{}}
    for condition in sorted(set(r['condition'] for r in records)):
        rows=[r for r in records if r['condition']==condition]
        sdr=[r['separation_metrics']['si_sdr_db'] for r in rows if r['separation_metrics'] and r['separation_metrics']['si_sdr_db'] is not None]
        result['conditions'][condition]={'count':len(rows),
            'onset_f1':float(np.mean([r['pitch_metrics']['onset']['f1'] for r in rows])),
            'onset_offset_f1':float(np.mean([r['pitch_metrics']['onset_offset']['f1'] for r in rows])),
            'frame_f1':float(np.mean([r['pitch_metrics']['frame']['f1'] for r in rows])),
            'tab_onset_f1':float(np.mean([r['tab_metrics']['onset']['f1'] for r in rows])),
            'total_separation_seconds':sum(r['timing']['separation_seconds'] for r in rows),
            'total_transcription_seconds':sum(r['timing']['transcription_seconds'] for r in rows),
            'mean_si_sdr_db':float(np.mean(sdr)) if sdr else None,'defined_si_sdr_count':len(sdr),
            'silent_estimate_count':sum(bool(r['separation_metrics'] and r['separation_metrics'].get('estimate_silent')) for r in rows),
            'false_positive_notes':sum(r['false_positive_count'] or 0 for r in rows),
            'real_time_factor':sum(r['timing']['total_measured_seconds'] for r in rows)/sum(r['timing']['input_seconds'] for r in rows)}
    return result
