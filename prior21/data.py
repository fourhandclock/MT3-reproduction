"""Verify Stage20 lineage, then read only cached validation targets and probabilities."""

import sys
import numpy as np

from coverage20.data import ROOT, MODES, HISTORIES, read, save, sha, relative

COUNT_SOURCES = ('presentations', 'unique', 'available')
PREDICTORS = ('model_best_active', 'presentations_top1', 'presentations_distribution',
              'unique_top1', 'unique_distribution', 'available_top1',
              'available_distribution', 'uniform25_distribution')
SOURCE = ROOT / 'outputs/coverage20'


def validate_config(config):
    expected = {
        'stage': 21, 'proposal': 'outputs/coverage20/next-experiment.json',
        'proposal_sha256': 'b8e99f9cbcac4ab68446dd5e956c604a2eac6fa0263b7669a2eaba02867d5222',
        'source_verification_sha256': 'f6d33be4c8a8a2090c01975de10ac6fd94e3e338616d8d5916ab87ac6b1e5717',
        'source_frames': 1000, 'source_delay_frames': 13, 'leading_mask_frames': 26,
        'prior_count_sources': list(COUNT_SOURCES), 'primary_prior': 'presentations_top1',
        'tie_rule': 'lowest_fret', 'include_uniform25': True, 'paired_tie_tolerance': 1e-12}
    if config != expected:
        raise ValueError('The fixed Stage21 priors, alignment or diagnostic scope changed')


def no_neural_runtime():
    if any(name in sys.modules for name in ('torch', 'tensorflow', 'jax')):
        raise ValueError('Stage21 must run without a neural runtime')


def load_inputs(config):
    validate_config(config)
    bindings = {}

    def bind(name, expected):
        if sha(relative(name)) != expected:
            raise ValueError('Frozen Stage21 input changed: ' + name)
        if name in bindings and bindings[name] != expected:
            raise ValueError('Conflicting input identity: ' + name)
        bindings[name] = expected

    for name in ('outputs/coverage20/verification.json', 'experiments/stage20.snapshot.json'):
        bind(name, config['source_verification_sha256'])
    bind(config['proposal'], config['proposal_sha256'])
    verification = read(SOURCE / 'verification.json')
    for field, prefix in [('artifact_sha256', 'outputs/coverage20/'), ('code_sha256', ''), ('input_sha256', '')]:
        for name, expected in verification[field].items():
            bind(prefix + name, expected)
    proposal = read(ROOT / config['proposal'])
    if (proposal['stage'] != 21 or proposal['new_optimizer_updates'] or proposal['neural_inference_calls']
            or proposal['test_inference_allowed'] or proposal['production_promotion'] or proposal['calibration_search_allowed']):
        raise ValueError('Stage21 zero-training diagnosis scope differs')
    coverage = read(SOURCE / 'coverage.json')
    if coverage['model_histories'] != HISTORIES:
        raise ValueError('Model exposure histories differ')
    infos = read(ROOT / 'outputs/fret19/input-audit.json')['inputs']['validation']
    if len(infos) != 12 or len({row['id'] for row in infos}) != 12:
        raise ValueError('Validation identity count differs')
    records = []
    for info in infos:
        name = info['prefix'] + '.npz'
        if bindings.get(name) != info['targets_sha256']:
            raise ValueError('Validation target is not anchored by Stage20')
        with np.load(ROOT / name, allow_pickle=False) as arrays:
            state = arrays['state'].copy()
        if state.shape != (info['frames'], 6):
            raise ValueError('Target source grid differs')
        records.append({'id': info['id'], 'state': state})
    return proposal, coverage, records, bindings
