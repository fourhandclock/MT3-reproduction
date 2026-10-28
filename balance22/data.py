"""Frozen Stage21 proposal, control500 parent and shared sampled epochs11-13."""

from pathlib import Path
import numpy as np
import torch
from acoustic10.data import load_inputs as frozen_inputs
from baseline.download import ROOT, sha256
from decoder14.workflow import read
from optimization13.data import build_schedule
from streaming.data import fingerprint
from streaming.evaluate import save_json
from .loss import class_weights

PROPOSAL = 'outputs/prior21/next-experiment.json'
PROPOSAL_SHA = 'd150e9798b6c39bb5b0db32e9317c0069ee0ab968ed273a3fe183ff111a3b292'
VERIFICATION_SHA = '9c0b4e6b11875a60daba31b5ea7f7c720ef01cd7417aad8d9b3e49354213bed9'


def validate_config(config):
    if sha256(ROOT / PROPOSAL) != PROPOSAL_SHA:
        raise ValueError('The original Stage22 proposal changed')
    proposal = read(ROOT / PROPOSAL)
    old = read(ROOT / 'configs/fret19.json')
    expected = {'stage': 22, 'seed': 42, 'data_epoch_start': 10, 'total_optimizer_update_cap': 450,
        'proposal': PROPOSAL, 'proposal_sha256': PROPOSAL_SHA, 'source_verification_sha256': VERIFICATION_SHA,
        'unexposed_available_policy': 'fixed_cap4_before_training',
        'parent_checkpoint': proposal['parent_checkpoint'], 'parent_checkpoint_sha256': proposal['parent_checkpoint_sha256'],
        'arms': proposal['arms'], 'common_target_delay_frames': 13, 'common_leading_mask_frames': 26,
        'training': {**old['training'], 'epochs': 3, 'max_steps': 150},
        'decoder_parameters': proposal['decoder_parameters'], 'validation_packet_ms': 20,
        'pitch_diagnostics': old['pitch_diagnostics'], 'sanity_reference': old['sanity_reference'],
        'selection_rule': proposal['selection_rule'], 'test_inference_allowed': False, 'production_promotion': False}
    if config != expected:
        raise ValueError('The fixed 150+150+150 continuation protocol changed')


def load_inputs(config, output):
    validate_config(config)
    output = Path(output)
    bindings = {}

    def bind(name, expected):
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != expected:
            raise ValueError('Frozen predecessor changed: ' + name)
        if name in bindings and bindings[name] != expected:
            raise ValueError('Conflicting predecessor hashes')
        bindings[name] = expected

    for name in ('outputs/prior21/verification.json', 'experiments/stage21.snapshot.json'):
        bind(name, VERIFICATION_SHA)
    bind(PROPOSAL, PROPOSAL_SHA)
    prior = read(ROOT / 'outputs/prior21/verification.json')
    for field, prefix in [('artifact_sha256', 'outputs/prior21/'), ('code_sha256', ''), ('input_sha256', '')]:
        for name, digest in prior[field].items():
            bind(prefix + name, digest)
    base, cache, datasets, provenance = frozen_inputs(read(ROOT / 'configs/acoustic10.json'))
    path = ROOT / config['parent_checkpoint']
    if sha256(path) != config['parent_checkpoint_sha256']:
        raise ValueError('Stage19 control500 checkpoint changed')
    parent = torch.load(path, map_location='cpu', weights_only=True)
    if (parent['step'] != 500 or parent['receipt']['optimizer_steps'] != 500
            or parent['config']['feature_planes'] != [2048, 2048]
            or parent['normalization'] != {k: [v, v] for k, v in cache['normalization'].items()}
            or parent['positive_weights'] != cache['positive_weights'] or parent['config']['feature'] != base['feature']
            or parent['scheduler_used'] or parent['source_scheduler']['last_epoch'] != 250
            or parent['optimizer']['param_groups'][0]['lr'] != .0001):
        raise ValueError('Parent state, representation or archived ended scheduler differs')
    for name, digest in parent['receipt']['source_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError('Parent producer changed: ' + name)
    for key in ('manifest_sha256', 'cache_audit_sha256'):
        if parent['receipt'][key] != provenance[key]:
            raise ValueError('Parent training source differs')
    coverage = read(ROOT / 'outputs/coverage20/coverage.json')
    counts = coverage['histories']['epochs1_10']['unique']
    available = np.zeros((6, 25), np.int64)
    for record in datasets['train']:
        state = record['targets']['state']
        for string in range(6):
            frets = state[:, string]
            available[string] += np.bincount(frets[(frets >= 0) & (frets < 25)], minlength=25)
    if available.tolist() != coverage['histories']['epochs1_10']['available']:
        raise ValueError('Available target support differs from the frozen training-only counts')
    weights = class_weights(counts, available)
    fallback = np.argwhere((np.asarray(counts) == 0) & (available > 0)).tolist()
    weight_record = {'source': 'outputs/coverage20/coverage.json',
        'source_sha256': sha256(ROOT / 'outputs/coverage20/coverage.json'),
        'field': 'histories.epochs1_10.unique', 'counts': counts, 'weights': weights.tolist(),
        'rule': read(ROOT / PROPOSAL)['weight_rule'], 'historically_exposed_positions': int((np.asarray(counts) > 0).sum()),
        'available_positions': int((available > 0).sum()), 'unexposed_available_positions': fallback,
        'pre_training_amendment': {
            'original_rule': 'Zero historical support has weight0 and any active such target fails.',
            'revised_rule': 'Training-available but historically unexposed positions receive the already fixed cap4; labels absent from training remain weight0 and fail if active.',
            'basis': 'Training-only preflight found string4/fret17: 50 available active frames and 76 presentations in fixed epochs11-13.',
            'optimizer_updates_before_amendment': 0, 'validation_used_for_amendment': False}}
    audit = {'normalization': parent['normalization'], 'positive_weights': cache['positive_weights']}
    output.mkdir(parents=True, exist_ok=True)
    for name, value in [('class-weights.json', weight_record), ('input-bindings.json', bindings)]:
        target = output / name
        if target.exists() and read(target) != value:
            raise ValueError('Previously frozen Stage22 inputs differ: ' + name)
        if not target.exists():
            save_json(target, value)
    reference = {'checkpoint_path': config['parent_checkpoint'], 'checkpoint_sha256': sha256(path),
        'historical_unique_updates': 500, 'lifetime_updates_per_child_before_training': 500,
        'ended_source_scheduler': parent['source_scheduler'], 'scheduler_used_for_continuation': False}
    input_audit = {**provenance, **audit, 'parent': reference, 'class_weights_sha256': sha256(output / 'class-weights.json'),
        'input_bindings_sha256': sha256(output / 'input-bindings.json'), 'config_sha256': fingerprint(config),
        'features_rewritten': False, 'targets_rewritten': False, 'new_optimizer_updates_for_preparation': 0}
    target = output / 'input-audit.json'
    if target.exists() and read(target) != input_audit:
        raise ValueError('Frozen Stage22 input audit changed')
    if not target.exists():
        save_json(target, input_audit)
    provenance.update(parent=reference, class_weights_sha256=sha256(output / 'class-weights.json'),
                      input_audit_sha256=sha256(target), input_bindings_sha256=sha256(output / 'input-bindings.json'))
    return base, audit, datasets, provenance, parent, build_schedule(config, datasets['train']), weights
