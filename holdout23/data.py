"""Write the selection/model lock before reading progression3 audio or annotations."""

import re
from pathlib import Path
import numpy as np
from baseline.download import ROOT, sha256
from baseline.evaluation import reference_notes
from cnn_gru.data import load_audio
from decoder14.workflow import read
from streaming.data import cache_path, fingerprint
from streaming.evaluate import save_json
from streaming.targets import make_targets

PROPOSAL = 'outputs/balance22/next-experiment.json'
PROPOSAL_SHA = '3da0048398deb04862be8300937097aedcbcd4f2e7392c1d8c291d7649318961'
VERIFICATION_SHA = '38f66103d68f1f3390c50feb776349ef69ee46d93ceede5a74885e26314e07ee'
MODES = ['parent500', 'original650', 'uniform_ce650', 'balanced_ce650']


def validate_config(config):
    if sha256(ROOT / PROPOSAL) != PROPOSAL_SHA:
        raise ValueError('The original Stage23 proposal changed')
    proposal = read(ROOT / PROPOSAL)
    expected = {'stage': 23, 'proposal': PROPOSAL, 'proposal_sha256': PROPOSAL_SHA,
        'source_verification_sha256': VERIFICATION_SHA, 'locked_selected_arm': 'uniform_ce650',
        'models': proposal['comparison_models'], 'common_target_delay_frames': 13,
        'common_leading_mask_frames': 26, 'source_frames': 1000,
        'decoder_parameters': proposal['decoder_parameters'],
        'pitch_diagnostics': {'threshold': .5, 'low_register_midi': [40, 51]},
        'clip_seconds': 12.0, 'packet_ms': 20, 'new_optimizer_update_cap': 0,
        'model_selection_allowed': False, 'calibration_search_allowed': False, 'production_promotion': False}
    if config != expected:
        raise ValueError('The locked Stage23 models, alignment, decoder or zero-training scope changed')


def source_hashes():
    paths = [ROOT / 'holdout23' / name for name in ('__init__.py', '__main__.py', 'data.py', 'workflow.py')]
    paths += [ROOT / 'configs/holdout23.json']
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths}


def validate_partition(manifest):
    entries = manifest['test']
    ids = [row['id'] for row in entries]
    if len(ids) != 12 or len(set(ids)) != 12 or any(not re.match(r'^\d\d_[A-Za-z]+3-', name) for name in ids):
        raise ValueError('Expected twelve unique progression3 test IDs')
    if set(ids) & {row['id'] for split in ('train', 'validation') for row in manifest[split]}:
        raise ValueError('Test IDs overlap the training/development IDs')
    players = lambda rows: {row['id'].split('_', 1)[0] for row in rows}
    return {'train_test_shared_players': sorted(players(entries) & players(manifest['train'])),
            'validation_test_shared_players': sorted(players(entries) & players(manifest['validation']))}


def bind(bindings, name, expected):
    path = (ROOT / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(ROOT) or sha256(path) != expected:
        raise ValueError('Locked input changed: ' + name)
    if name in bindings and bindings[name] != expected:
        raise ValueError('Contradictory input binding')
    bindings[name] = expected


def freeze(config, output):
    validate_config(config)
    output = Path(output)
    if (output / 'protocol.json').exists():
        raise ValueError('Stage23 is already locked; do not repeat or alter the evaluation')
    bindings = {}
    for name in ('outputs/balance22/verification.json', 'experiments/stage22.snapshot.json'):
        bind(bindings, name, VERIFICATION_SHA)
    prior = read(ROOT / 'outputs/balance22/verification.json')
    for field, prefix in [('artifact_sha256', 'outputs/balance22/'), ('code_sha256', ''), ('input_sha256', '')]:
        for name, digest in prior[field].items():
            bind(bindings, prefix + name, digest)
    bind(bindings, PROPOSAL, PROPOSAL_SHA)
    run = read(ROOT / 'outputs/balance22/validation/run.json')
    if (run['selection']['selected_arm'] != config['locked_selected_arm']
            or run['checkpoint_paths'] != {mode: config['models'][mode]['checkpoint'] for mode in MODES}
            or run['checkpoint_sha256'] != {mode: config['models'][mode]['sha256'] for mode in MODES}):
        raise ValueError('Development selection or checkpoint identities differ from the proposal')
    manifest = read(ROOT / 'data/cnn_gru/manifest.json')
    overlap = validate_partition(manifest)
    base = read(ROOT / 'configs/streaming.json')
    cache_audit = read(ROOT / 'data/streaming/cache_audit.json')
    if cache_audit['config_sha256'] != fingerprint(base) or cache_audit['manifest_sha256'] != sha256(ROOT / 'data/cnn_gru/manifest.json'):
        raise ValueError('Frozen target-cache recipe or manifest changed')
    bind(bindings, 'configs/streaming.json', sha256(ROOT / 'configs/streaming.json'))
    for entry in manifest['test']:
        for key in ('audio', 'annotation'):
            if not (ROOT / entry[key]).is_file():
                raise ValueError('Missing existing test asset: ' + entry[key])
        if not cache_path(entry, base, base['evaluation_audio_seconds']).with_suffix('.npz').is_file():
            raise ValueError('Missing existing progression3 target cache')
    output.mkdir(parents=True, exist_ok=True)
    selection_bytes = (ROOT / 'outputs/balance22/selection.json').read_bytes()
    (output / 'selection.json').write_bytes(selection_bytes)
    save_json(output / 'predecessor-bindings.json', bindings)
    protocol = {'stage': 23, 'config': config, 'source_sha256': source_hashes(),
        'proposal_sha256': PROPOSAL_SHA, 'source_verification_sha256': VERIFICATION_SHA,
        'predecessor_bindings_sha256': sha256(output / 'predecessor-bindings.json'),
        'selection_sha256': sha256(output / 'selection.json'), 'selection_source': 'outputs/balance22/selection.json',
        'locked_selected_arm': config['locked_selected_arm'], 'models': config['models'],
        'test_ids': [entry['id'] for entry in manifest['test']], 'manifest_sha256': sha256(ROOT / 'data/cnn_gru/manifest.json'),
        'scoring_config': run['scoring_config'], 'player_overlap': overlap,
        'lock_written_before_test_input_reads': True, 'test_audio_or_annotations_read_before_lock': False,
        'new_optimizer_updates': 0, 'optimizer_instantiations': 0, 'normalization_updates': 0,
        'model_selection_allowed': False, 'calibration_search_allowed': False, 'production_promotion': False,
        'test_scored_replays_planned': 48, 'test_audio_warmups_planned': 4,
        'data_scope': read(ROOT / PROPOSAL)['data_scope']}
    save_json(output / 'protocol.json', protocol)
    return protocol


def load_test_inputs(config, output):
    validate_config(config)
    output = Path(output)
    protocol = read(output / 'protocol.json')
    if (protocol['config'] != config or protocol['source_sha256'] != source_hashes()
            or not protocol['lock_written_before_test_input_reads']
            or (output / 'selection.json').read_bytes() != (ROOT / 'outputs/balance22/selection.json').read_bytes()):
        raise ValueError('The pre-test protocol or copied development selection changed')
    bindings = read(output / 'predecessor-bindings.json')
    if sha256(output / 'predecessor-bindings.json') != protocol['predecessor_bindings_sha256']:
        raise ValueError('Pre-test predecessor identity changed')
    manifest = read(ROOT / 'data/cnn_gru/manifest.json')
    if sha256(ROOT / 'data/cnn_gru/manifest.json') != protocol['manifest_sha256']:
        raise ValueError('Manifest changed after lock')
    validate_partition(manifest)
    base = read(ROOT / 'configs/streaming.json')
    infos = {row['id']: row for row in read(ROOT / 'data/streaming/cache_audit.json')['splits']['test']}
    if set(infos) != set(protocol['test_ids']):
        raise ValueError('Existing test cache identities differ')
    records, provenance = [], []
    for entry in manifest['test']:
        for key in ('audio', 'annotation'):
            bind(bindings, entry[key], entry[key + '_sha256'])
        path = cache_path(entry, base, base['evaluation_audio_seconds']).with_suffix('.npz')
        info = infos[entry['id']]
        bind(bindings, path.relative_to(ROOT).as_posix(), info['targets_sha256'])
        audio, duration = load_audio(ROOT / entry['audio'], config['clip_seconds'], base)
        if duration != 12.:
            raise ValueError('Locked test excerpt must supply twelve seconds')
        reference = reference_notes(ROOT / entry['annotation'])
        with np.load(path, allow_pickle=False) as arrays:
            targets = {key: arrays[key].copy() for key in arrays.files}
        count = len(audio) // base['feature']['hop_length']
        rebuilt, conflicts = make_targets(reference, count, base, duration)
        if (count != info['frames'] or conflicts != info['conflict_string_frames']
                or set(targets) != set(rebuilt) or any(targets[key].shape != rebuilt[key].shape
                    or targets[key].dtype != rebuilt[key].dtype or targets[key].tobytes() != rebuilt[key].tobytes() for key in targets)):
            raise ValueError('Pinned target labels do not reproduce the original right-edge grid')
        record = {'entry': entry, 'targets': targets, 'audio': audio, 'duration': duration, 'reference': reference}
        records.append(record)
        provenance.append({'id': entry['id'], 'audio_sha256': entry['audio_sha256'],
            'annotation_sha256': entry['annotation_sha256'], 'targets_path': path.relative_to(ROOT).as_posix(),
            'targets_sha256': info['targets_sha256'], 'duration_seconds': duration, 'frames': count,
            'target_arrays_byte_equal_to_frozen_builder': True, 'conflict_string_frames': conflicts})
    if [r['entry']['id'] for r in records] != protocol['test_ids']:
        raise ValueError('Locked test order differs')
    return base, records, provenance, bindings
