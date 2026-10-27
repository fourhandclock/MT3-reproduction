"""Immutable Stage18 proposal, Stage17 parent and matched sampled epochs6-10."""

from pathlib import Path
import torch
from acoustic10.data import load_inputs as frozen_inputs
from baseline.download import ROOT,sha256
from decoder14.workflow import read
from optimization13.data import build_schedule
from streaming.data import fingerprint
from streaming.evaluate import save_json


def validate_config(config):
    old=read(ROOT/'configs/multires17.json')
    expected={'stage':19,'seed':42,'data_epoch_start':5,
        'parent_checkpoint':'outputs/multires17/models/short_short/final.pt',
        'parent_checkpoint_sha256':'bb97909aa37503aa4c7321e474d501551a5adbf2acc0d0a6c5780cb45d154522',
        'total_optimizer_update_cap':500,
        'arms':[{'name':'control','conditional_fret_coefficient':0.},{'name':'conditional_fret','conditional_fret_coefficient':.5}],
        'common_target_delay_frames':13,'common_leading_mask_frames':26,
        'training':{**old['training'],'learning_rate':.0001},'decoder_parameters':old['decoder_parameters'],
        'validation_packet_ms':20,'pitch_diagnostics':old['pitch_diagnostics'],
        'sanity_reference':'outputs/acoustic10/sanity.json','test_inference_allowed':False,'production_promotion':False}
    proposal=read(ROOT/'outputs/headswap18/next-experiment.json')
    if set(config)!=set(expected)|{'selection_rule'} or any(config[k]!=v for k,v in expected.items()) or config['selection_rule']!=proposal['selection_rule']:
        raise ValueError('The fixed Stage19 fret-loss protocol changed')


def load_inputs(config,output):
    validate_config(config);root=ROOT/'outputs/headswap18';verification=read(root/'verification.json')
    for group,basepath in [('artifact_sha256',root),('code_sha256',ROOT)]:
        for name,digest in verification[group].items():
            if sha256(basepath/name)!=digest:raise ValueError('Verified Stage18 evidence changed: '+name)
    if not read(root/'audit.json')['passed'] or sha256(ROOT/'outputs/multires17/verification.json')!=verification['prior_stage17_verification_sha256']:
        raise ValueError('Prior experiment verification changed')
    proposal=read(root/'next-experiment.json')
    if proposal['status']!='proposed_not_trained' or proposal['training_executed'] or proposal['parent_checkpoint']!=config['parent_checkpoint']:
        raise ValueError('The Stage18 proposal is incompatible')
    base,cache,datasets,provenance=frozen_inputs(read(ROOT/'configs/acoustic10.json'))
    path=ROOT/config['parent_checkpoint']
    if sha256(path)!=config['parent_checkpoint_sha256']:raise ValueError('The shared parent changed')
    parent=torch.load(path,map_location='cpu',weights_only=True)
    for name,digest in parent['receipt']['source_sha256'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Parent producer changed: '+name)
    if (parent['step']!=250 or parent['receipt']['optimizer_steps']!=250 or parent['config'].get('feature_planes')!=[2048,2048]
            or parent['normalization']!={k:[v,v] for k,v in cache['normalization'].items()}
            or parent['positive_weights']!=cache['positive_weights'] or parent['config']['feature']!=base['feature']
            or parent['scheduler']['last_epoch']!=250 or parent['optimizer']['param_groups'][0]['lr']!=.0001):
        raise ValueError('Parent step, representation, normalization or ended schedule differs')
    if any(parent['config']['training'][k]!=v for k,v in {**config['training'],'learning_rate':.001}.items()):
        raise ValueError('Parent training recipe differs')
    for key in ['manifest_sha256','cache_audit_sha256']:
        if parent['receipt'][key]!=provenance[key]:raise ValueError('Parent data identity changed')
    audit={'normalization':parent['normalization'],'positive_weights':cache['positive_weights']}
    reference={'checkpoint_path':config['parent_checkpoint'],'checkpoint_sha256':sha256(path),
        'historical_unique_updates':250,'lifetime_updates_per_child_before_training':250,
        'ended_source_scheduler':parent['scheduler'],'scheduler_used_for_continuation':False}
    bindings={str((root/name).relative_to(ROOT)).replace('\\','/'):sha256(root/name) for name in
              ['verification.json','next-experiment.json','audit.json','summary.json']}
    input_audit={**provenance,**audit,'parent':reference,'proposal_bindings':bindings,'config_sha256':fingerprint(config),
        'features_rewritten':False,'targets_rewritten':False,'new_optimizer_updates_for_preparation':0}
    target=Path(output)/'input-audit.json'
    if target.exists() and read(target)!=input_audit:raise ValueError('Frozen Stage19 inputs changed')
    if not target.exists():save_json(target,input_audit)
    provenance.update(parent=reference,proposal_bindings=bindings,input_audit_sha256=sha256(target))
    return base,audit,datasets,provenance,parent,build_schedule(config,datasets['train'])
