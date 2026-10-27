"""Load verified saved probabilities and source labels, without a model call."""

from pathlib import Path
import numpy as np
from baseline.download import ROOT,sha256
from baseline.evaluation import reference_notes
from decoder14.workflow import read
from diagnosis9.oracles import validate_cached_target_identity

MODELS=['short_short','short_long']
COMBINATIONS={
    'control':('short_short','short_short'),
    'fused':('short_long','short_long'),
    'control_state_fused_onset':('short_short','short_long'),
    'fused_state_control_onset':('short_long','short_short')}


def validate_config(config):
    expected={'stage':18,'source_root':'outputs/multires17','models':MODELS,
        'source_delay_frames':13,'source_frames':1000,'leading_mask_frames':26,
        'input_frames':1200,'input_seconds':12.,'packet_ms':20,
        'decoder_parameters':{'onset_threshold':.7,'activity_threshold':.25,'onset_backdate_seconds':0.},
        'auxiliary_pitch_threshold':.5,'nearby_onset_seconds':.2,'fragment_minimum_overlap_seconds':.02,
        'repeated_note_gap_seconds':.06,'new_optimizer_updates':0,'neural_inference_calls':0,
        'calibration_search_allowed':False,'test_inference_allowed':False,'production_promotion':False}
    if set(config)!=set(expected)|{'comparison_rule','clock_rule'} or any(config[k]!=v for k,v in expected.items()):
        raise ValueError('The fixed zero-training Stage18 protocol changed')
    if any(not isinstance(config[k],str) or not config[k] for k in ['comparison_rule','clock_rule']):
        raise ValueError('Explicit comparison/clock scope is required')


def verified_prior(root):
    verification=read(root/'verification.json')
    for group,base in [('artifact_sha256',root),('code_sha256',ROOT)]:
        for name,digest in verification[group].items():
            if sha256(base/name)!=digest:raise ValueError('Verified Stage17 evidence changed: '+name)
    run=read(root/'validation/run.json')
    for model,path in run['checkpoint_paths'].items():
        if sha256(ROOT/path)!=verification['checkpoint_sha256'][model]:
            raise ValueError('Frozen Stage17 checkpoint changed')
    if not read(root/'audit.json')['passed'] or not read(root/'viewer-validation.json')['passed']:
        raise ValueError('Prior verification gate failed')
    return run


def load_inputs(config):
    validate_config(config);root=ROOT/config['source_root'];run=verified_prior(root)
    base=read(ROOT/'configs/streaming.json');score=run['scoring_config']
    if run['config']['decoder_parameters']!=config['decoder_parameters']:
        raise ValueError('Decoder differs from the saved comparison')
    audit=read(root/'input-audit.json');manifest_path=ROOT/'data/cnn_gru/manifest.json'
    if sha256(manifest_path)!=audit['manifest_sha256']:raise ValueError('Split manifest changed')
    manifest=read(manifest_path);entries=manifest['validation']
    if len(entries)!=12 or {e['id'] for e in entries}&{e['id'] for e in manifest['train']}:
        raise ValueError('Validation split is incomplete or overlaps training')
    infos=audit['inputs']['validation']
    if [i['id'] for i in infos]!=[e['id'] for e in entries]:raise ValueError('Input order changed')
    bindings={(root/'verification.json').relative_to(ROOT).as_posix():sha256(root/'verification.json'),
              manifest_path.relative_to(ROOT).as_posix():sha256(manifest_path)}
    inputs=[]
    for entry,info in zip(entries,infos):
        for field in ['audio','annotation']:
            path=ROOT/entry[field]
            if sha256(path)!=entry[field+'_sha256']:raise ValueError('Original validation source changed')
            bindings[path.relative_to(ROOT).as_posix()]=sha256(path)
        reference=reference_notes(ROOT/entry['annotation'])
        path=(ROOT/info['prefix']).with_suffix('.npz')
        if sha256(path)!=info['targets_sha256']:raise ValueError('Source target cache changed')
        with np.load(path,allow_pickle=False) as archive:targets={k:archive[k] for k in archive.files}
        conflicts=validate_cached_target_identity(reference,targets,1200,base,12.)
        bindings[path.relative_to(ROOT).as_posix()]=sha256(path)
        rows,probabilities={},{}
        for model in MODELS:
            path=root/'validation/results'/model/(entry['id']+'-metrics.json');row=read(path)
            if row['id']!=entry['id'] or row['mode']!=model or row['reference']!=reference or row['input_provenance']!=info:
                raise ValueError('Recorded source/reference provenance differs')
            bindings[path.relative_to(ROOT).as_posix()]=sha256(path)
            path=root/row['probability_path']
            if sha256(path)!=row['probability_sha256']:raise ValueError('Recorded probability file changed')
            with np.load(path,allow_pickle=False) as archive:values={k:archive[k] for k in archive.files}
            expected={'state':(1200,6,26),'onset':(1200,6),'offset':(1200,6),'pitch':(1200,49)}
            if set(values)!=set(expected) or any(values[k].shape!=shape or values[k].dtype!=np.float32
                    or not np.isfinite(values[k]).all() or np.any(values[k]<0) or np.any(values[k]>1)
                    for k,shape in expected.items()):raise ValueError('Probability interface/frame grid differs')
            if not np.allclose(values['state'].sum(-1),1.,atol=2e-6,rtol=0):raise ValueError('State probabilities are not normalized')
            for value in values.values():value.flags.writeable=False
            bindings[path.relative_to(ROOT).as_posix()]=sha256(path)
            probabilities[model]=values;rows[model]=row
        audio=root/rows[MODELS[0]]['relativeaudio']
        bindings[audio.relative_to(ROOT).as_posix()]=sha256(audio)
        inputs.append({'id':entry['id'],'reference':reference,'targets':targets,'probabilities':probabilities,
                       'prior':rows,'audio':audio,'target_conflicts':conflicts})
    return base,score,inputs,bindings


def cross_heads(probabilities,state_source,onset_source):
    if state_source not in MODELS or onset_source not in MODELS or set(probabilities)!=set(MODELS):
        raise ValueError('Both verified models and valid source identifiers are required')
    # Each array is copied so a consumer cannot mutate a different intervention.
    return {key:value.copy() for key,value in probabilities[state_source].items() if key!='onset'} | {
            'onset':probabilities[onset_source]['onset'].copy()}
