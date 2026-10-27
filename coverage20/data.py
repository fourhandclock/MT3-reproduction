"""Read hash-bound targets, crop schedules and archived predictions; no model imports."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/fret19'
MODES = ('parent250', 'control500', 'conditional_fret500')
HISTORIES = {'parent250': 'epochs1_5', 'control500': 'epochs1_10', 'conditional_fret500': 'epochs1_10'}
TUNING = (40, 45, 50, 55, 59, 64)
BINS = ('unseen', '1-99', '100-999', '1000+')


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def relative(name):
    path = (ROOT / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(ROOT):
        raise ValueError('Input path escapes workspace: ' + str(name))
    return path


def load_inputs(config):
    expected = read(ROOT / 'configs/coverage20.json')
    if config != expected or set(config) != {'stage', 'proposal', 'proposal_sha256',
                                           'source_verification_sha256', 'parent_verification_sha256'}:
        raise ValueError('Stage 20 uses only the fixed saved proposal')
    bindings = {}

    def bind(name, digest):
        path = relative(name)
        if sha(path) != digest:
            raise ValueError('Frozen input changed: ' + name)
        if name in bindings and bindings[name] != digest:
            raise ValueError('Contradictory input hash: ' + name)
        bindings[name] = digest
        return path

    bind(config['proposal'], config['proposal_sha256'])
    proposal = read(ROOT / config['proposal'])
    if (proposal['stage'] != 20 or proposal['new_optimizer_updates'] != 0
            or proposal['neural_inference_calls'] != 0 or proposal['calibration_search_allowed']
            or proposal['test_inference_allowed'] or proposal['production_promotion']):
        raise ValueError('Zero-training diagnosis scope changed')
    for name, digest in [('outputs/fret19/verification.json', config['source_verification_sha256']),
                         ('outputs/multires17/verification.json', config['parent_verification_sha256']),
                         ('experiments/stage19.snapshot.json', config['source_verification_sha256'])]:
        bind(name, digest)
    for folder in ('fret19', 'multires17'):
        verification = read(ROOT / f'outputs/{folder}/verification.json')
        for group, base in [('artifact_sha256', f'outputs/{folder}/'), ('code_sha256', '')]:
            for name, digest in verification[group].items():
                bind(base + name, digest)
    provenance = read(SOURCE / 'input-audit.json')
    bind('data/cnn_gru/manifest.json', provenance['manifest_sha256'])
    bind('data/streaming/cache_audit.json', provenance['cache_audit_sha256'])
    manifest = read(ROOT / 'data/cnn_gru/manifest.json')
    datasets = {}
    for split, count in [('train', 48), ('validation', 12)]:
        infos = provenance['inputs'][split]
        if len(infos) != count or [e['id'] for e in manifest[split]] != [e['id'] for e in infos]:
            raise ValueError('Frozen split identity/order changed')
        datasets[split] = []
        for entry, info in zip(manifest[split], infos):
            if entry['annotation_sha256'] != info['annotation_sha256']:
                raise ValueError('Annotation identity differs')
            bind(entry['annotation'], entry['annotation_sha256'])
            name = info['prefix'] + '.npz'
            bind(name, info['targets_sha256'])
            with np.load(ROOT / name, allow_pickle=False) as arrays:
                targets = {key: arrays[key].copy() for key in arrays.files}
            if targets['state'].shape != (info['frames'], 6):
                raise ValueError('Target frame grid changed')
            datasets[split].append({'id': entry['id'], 'entry': entry, 'targets': targets})
    if ({r['id'] for r in datasets['train']} & {r['id'] for r in datasets['validation']}
            or len({r['id'] for r in datasets['train']}) != 48):
        raise ValueError('Split overlap or duplicate training identity')
    schedules = []
    for path, epochs in zip(proposal['supervised_exposure_schedules'], [range(1, 6), range(6, 11)]):
        record = read(ROOT / path)
        rows = record['descriptors']
        if (fingerprint(rows) != record['sha256'] or len(rows) != 1000
                or [row['epoch'] for row in rows] != [e for e in epochs for _ in range(200)]):
            raise ValueError('Crop schedule ledger differs')
        for epoch in epochs:
            if sorted(r['sample_index'] for r in rows if r['epoch'] == epoch) != list(range(200)):
                raise ValueError('Missing or repeated crop descriptor within epoch')
        schedules.append(rows)
    return proposal, datasets, schedules, bindings
