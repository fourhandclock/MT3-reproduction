"""Exact frozen features/targets, parent lineage and matched epoch11-15 crops."""

from pathlib import Path

import torch

from acoustic10.data import load_inputs as frozen_inputs
from baseline.download import ROOT,sha256
from decoder14.workflow import read
from optimization13.data import build_schedule
from streaming.data import fingerprint
from streaming.evaluate import save_json


def validate_config(config):
    old=read(ROOT/'configs/optimization13.json')
    expected={'stage':16,'seed':42,'data_epoch_start':10,
        'parent_checkpoint':'outputs/optimization13/models/group8/final.pt',
        'parent_checkpoint_sha256':'964ba2cc85422dee6b74bd6e0470ed323b16ec7e36c084c87caf935223f22c0b',
        'total_optimizer_update_cap':500,'arms':[{'name':'control','state_pitch_coefficient':0.},{'name':'state_pitch','state_pitch_coefficient':.5}],
        'common_target_delay_frames':13,'common_leading_mask_frames':26,'training':old['training'],
        'decoder_parameters':{'onset_threshold':.7,'activity_threshold':.25,'onset_backdate_seconds':0.},
        'validation_packet_ms':20,'pitch_diagnostics':old['pitch_diagnostics'],
        'sanity_reference':'outputs/acoustic10/sanity.json','test_inference_allowed':False,'production_promotion':False}
    if set(config)!=set(expected)|{'selection_rule'} or any(config[key]!=value for key,value in expected.items()):
        raise ValueError('The matched Stage16 loss-only protocol changed.')
    if not isinstance(config['selection_rule'],str) or not config['selection_rule']:
        raise ValueError('A declared final-only selection rule is required.')


def load_inputs(config,output):
    validate_config(config)
    prior=ROOT/'outputs/diagnosis15'
    verification=read(prior/'verification.json')
    bindings={}
    for name in ['next-experiment.json','next-experiment-details.md','recommendation.json','audit.json']:
        path=prior/name
        if sha256(path)!=verification['artifact_sha256'][name]:
            raise ValueError('The verified Stage15 proposal changed: '+name)
        bindings[path.relative_to(ROOT).as_posix()]=sha256(path)
    for name,digest in verification['code_sha256'].items():
        if sha256(ROOT/name)!=digest:
            raise ValueError('The verified Stage15 implementation changed: '+name)
    if not read(prior/'recommendation.json')['supported_by_declared_rule'] or not read(prior/'audit.json')['passed']:
        raise ValueError('The Stage15 diagnostic gate did not pass.')
    bindings[(prior/'verification.json').relative_to(ROOT).as_posix()]=sha256(prior/'verification.json')
    base,audit,datasets,provenance=frozen_inputs(read(ROOT/'configs/acoustic10.json'))
    path=ROOT/config['parent_checkpoint']
    if sha256(path)!=config['parent_checkpoint_sha256']:
        raise ValueError('The shared trained parent checkpoint changed.')
    parent=torch.load(path,map_location='cpu',weights_only=True)
    if parent['step']!=500 or parent['receipt']['optimizer_steps']!=500:
        raise ValueError('Expected the actual final500 parent.')
    for name,digest in parent['receipt']['source_sha256'].items():
        if sha256(ROOT/name)!=digest:
            raise ValueError('A source underlying the trained parent changed: '+name)
    if (parent['config']['feature']!=base['feature'] or parent['config']['model'].get('cnn_norm_groups',8)!=8
            or parent['normalization']!=audit['normalization'] or parent['positive_weights']!=audit['positive_weights']
            or any(parent['config']['training'][key]!=value for key,value in config['training'].items())):
        raise ValueError('Parent data/model/normalization/training controls changed.')
    for key in ['manifest_sha256','cache_audit_sha256']:
        if parent['receipt'][key]!=provenance[key]:
            raise ValueError('Original training input identity changed.')
    reference={'checkpoint_path':config['parent_checkpoint'],'checkpoint_sha256':sha256(path),
               'historical_unique_updates':500,'lifetime_updates_per_child_before_training':500,
               'scope':'One shared historical parent, restored into two matched new arms.'}
    input_audit={**provenance,'normalization':audit['normalization'],'positive_weights':audit['positive_weights'],
                 'parent':reference,'proposal_bindings':bindings,'config_sha256':fingerprint(config),
                 'new_optimizer_updates_for_preparation':0,'features_rewritten':False,'targets_rewritten':False}
    save_json(Path(output)/'input-audit.json',input_audit)
    provenance.update(parent=reference,proposal_bindings=bindings,input_audit_sha256=sha256(Path(output)/'input-audit.json'))
    return base,audit,datasets,provenance,parent,build_schedule(config,datasets['train'])
