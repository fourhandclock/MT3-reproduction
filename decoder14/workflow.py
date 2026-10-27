"""Validation-only grid over immutable Stage13 probability recordings."""

import json
import shutil
import time
from pathlib import Path

import numpy as np

from baseline.download import ROOT, sha256
from baseline.evaluation import reference_notes
from streaming.evaluate import save_json
from streaming.export import export
from .core import (MODELS, candidates, choose, compact, delayed_oracle, packet_latency,
                   replay, score_case, summarize, validate_config, without_compute)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sources():
    names = ['decoder14/__init__.py', 'decoder14/core.py', 'decoder14/workflow.py',
             'configs/decoder14.json', 'acoustic10/decoder.py', 'acoustic10/targets.py',
             'diagnosis9/oracles.py', 'streaming/decoder.py', 'streaming/targets.py',
             'streaming/export.py', 'fingering/export.py', 'baseline/evaluation.py', 'cnn_gru/inference.py']
    return {name: sha256(ROOT / name) for name in names}


def load_inputs(config):
    validate_config(config)
    root = ROOT / config['source_root']
    verification = read(root / 'verification.json')
    if sha256(root / 'audit.json') != verification['audit_sha256'] or read(root / 'audit.json')['passed'] is not True:
        raise ValueError('Stage13 audit/verification binding changed.')
    run = read(root / 'validation/run.json')
    if any(sha256(ROOT / name) != digest for name, digest in run['source_sha256'].items()):
        raise ValueError('Stage13 producer code changed.')
    manifest = read(ROOT / 'data/cnn_gru/manifest.json')
    if sha256(ROOT / 'data/cnn_gru/manifest.json') != run['input_provenance']['manifest_sha256']:
        raise ValueError('Frozen split manifest changed.')
    if len(manifest['validation']) != 12:
        raise ValueError('Expected the twelve fixed validation recordings.')
    if run['config']['common_target_delay_frames'] != 13 or run['config']['decoder_parameters'] != config['baseline']:
        raise ValueError('Cached predictions do not use the expected label delay/decoder.')
    bindings = {str((root / name).relative_to(ROOT)).replace('\\', '/'): sha256(root / name)
                for name in ['verification.json', 'audit.json', 'validation/run.json', 'protocol.json']}
    bindings['data/cnn_gru/manifest.json'] = sha256(ROOT / 'data/cnn_gru/manifest.json')
    base = read(ROOT / 'configs/streaming.json')
    bindings['configs/streaming.json'] = sha256(ROOT / 'configs/streaming.json')
    score = {key: run['scoring_config'][key] for key in
             ['score_seconds', 'frame_rate', 'onset_tolerance_seconds', 'offset_tolerance_seconds', 'offset_ratio']}
    inputs = []
    for model in MODELS:
        path = ROOT / run['checkpoint_paths'][model]
        if sha256(path) != verification['checkpoint_sha256'][model] or sha256(path) != run['checkpoint_sha256'][model]:
            raise ValueError('Frozen final-500 checkpoint changed.')
        bindings[path.relative_to(ROOT).as_posix()] = sha256(path)
    for entry in manifest['validation']:
        for field in ['audio', 'annotation']:
            path = ROOT / entry[field]
            if sha256(path) != entry[field + '_sha256']:
                raise ValueError('Original validation source changed.')
            bindings[entry[field]] = entry[field + '_sha256']
        reference = reference_notes(ROOT / entry['annotation'])
        for model in MODELS:
            path = root / 'validation/results' / model / (entry['id'] + '-metrics.json')
            row = read(path)
            probability_path = root / row['probability_path']
            if (row['id'] != entry['id'] or row['mode'] != model or row['reference'] != reference
                    or row['input_provenance']['annotation_sha256'] != entry['annotation_sha256']
                    or sha256(probability_path) != row['probability_sha256']):
                raise ValueError('Cached probabilities/reference identity changed.')
            bindings[path.relative_to(ROOT).as_posix()] = sha256(path)
            bindings[probability_path.relative_to(ROOT).as_posix()] = row['probability_sha256']
            audio = root / row['relativeaudio']
            bindings[audio.relative_to(ROOT).as_posix()] = sha256(audio)
            with np.load(probability_path, allow_pickle=False) as archive:
                values = {key: archive[key] for key in archive.files}
            if any(len(array) != 1200 for array in values.values()) or row['timing']['input_seconds'] != 12:
                raise ValueError('Unexpected cached right-edge frame grid or duration.')
            for array in values.values():
                array.flags.writeable = False
            inputs.append({'id': entry['id'], 'model': model, 'previous': row, 'probabilities': values,
                           'audio': audio, 'reference': reference, 'duration': 12.})
    return base, score, inputs, bindings


def identity_check(item, notes, events, case):
    previous = item['previous']
    if (without_compute(previous['notes']) != notes or without_compute(previous['events']) != events
            or without_compute(previous['pitch_metrics']) != case['pitch_metrics']
            or without_compute(previous['tab_metrics']) != case['tab_metrics']):
        raise ValueError('Identity replay differs from Stage13 source notes, events or scores.')


def projection_proof(raw, projected, delay):
    """Independently calculate public clamps and require unchanged packet clocks."""
    if len(raw) != len(projected):
        raise ValueError('Source projection changed acoustic decisions.')
    starts, last_end = {}, [0.] * 6
    for left, right in zip(raw, projected):
        for key in ['type', 'note_id', 'string', 'fret', 'pitch', 'emitted_at_audio_seconds', 'flush']:
            if left[key] != right[key]:
                raise ValueError('Source projection changed event identity or availability.')
        expected = left['source_time'] - delay
        if left['type'] == 'note_on':
            expected = max(0., last_end[left['string']], expected)
            starts[left['note_id']] = expected
        else:
            if expected <= starts[left['note_id']]:
                expected = starts[left['note_id']] + 1e-9
            last_end[left['string']] = expected
        if abs(right['source_time'] - expected) > 1e-12:
            raise ValueError('Known supervision delay was not translated exactly once.')
    return {'events_checked': len(raw), 'packet_clocks_and_decisions_unchanged': True,
            'source_translation_verified': True, 'delay_seconds': delay}


def run(config, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'protocol.json').exists():
        raise ValueError('Stage14 already ran here; use report or audit on the saved experiment.')
    base, score, inputs, bindings = load_inputs(config)
    grid = candidates(config)
    protocol = {'config': config, 'candidates': grid, 'source_sha256': sources(),
                'input_sha256': bindings, 'decoder_config': base, 'scoring_config': score,
                'new_optimizer_updates': 0, 'neural_inference_calls': 0, 'test_inference_calls': 0,
                'historical_lifetime_updates_per_model': 500, 'production_promoted': False,
                'scope': 'Cached validation replay; 27 candidates per model; same twelve recordings select and report.'}
    save_json(output / 'protocol.json', protocol)
    # Grid is fixed on disk before consulting any new decoder scores.
    grid_records = []
    for model in MODELS:
        items = [item for item in inputs if item['model'] == model]
        for index, parameters in enumerate(grid):
            scores = []
            for item in items:
                notes, events, _ = replay(item['probabilities'], base, parameters, item['duration'])
                case = score_case(item['reference'], notes, score, item['duration'])
                if index == 0:
                    identity_check(item, notes, events, case)
                scores.append({'id': item['id'], **compact(case)})
            aggregate = {key: (sum(row[key] for row in scores) if key in
                         ['matched_onsets', 'predicted_notes', 'reference_notes'] else
                         float(np.mean([row[key] for row in scores]))) for key in compact(case)}
            grid_records.append({'model': model, 'candidate': index, 'parameters': parameters,
                                 **aggregate, 'clips': scores})
            if index % 9 == 0:
                print(model, 'candidate', index, 'onset F1', round(aggregate['onset_f1'], 4), flush=True)
    selection = {}
    for model in MODELS:
        rows = [row for row in grid_records if row['model'] == model]
        winner = choose(rows)
        selection[model] = {'candidate': winner, 'parameters': grid[winner], 'baseline': rows[0], 'selected': rows[winner]}
    save_json(output / 'grid.json', grid_records)
    save_json(output / 'selection.json', {'models': selection, 'rule': config['selection_rule'],
                                         'selected_from_test': False, 'production_promoted': False})
    records = []
    for item in inputs:
        model = item['model']
        relative_audio = Path('validation/clips') / (item['id'] + '.wav')
        (output / relative_audio).parent.mkdir(parents=True, exist_ok=True)
        if not (output / relative_audio).exists():
            shutil.copyfile(item['audio'], output / relative_audio)
        for variant, parameters in [('strict', config['baseline']), ('selected', selection[model]['parameters'])]:
            started = time.perf_counter()
            notes, events, clamps = replay(item['probabilities'], base, parameters, item['duration'])
            seconds = time.perf_counter() - started
            row = {'id': item['id'], 'mode': model + '_' + variant, 'model': model, 'variant': variant,
                   'parameters': parameters, 'notes': notes, 'events': events, 'reference': item['reference'],
                   'source_clamps': clamps, 'head_metrics': item['previous']['head_metrics'],
                   'probability_sha256': item['previous']['probability_sha256'],
                   'relativeaudio': relative_audio.as_posix(), 'duration': item['duration'],
                   'decoder_only_seconds': seconds, 'packet_latency': packet_latency(item['reference'], notes, score, item['duration']),
                   **score_case(item['reference'], notes, score, item['duration'])}
            stem = output / 'validation/results' / row['mode'] / item['id']
            export(notes, stem, {'stage': 14, 'mode': row['mode'], 'parameters': parameters,
                                'source_delay_seconds': .13, 'clock_scope': 'Packet availability only; no compute clock.'})
            save_json(stem.with_name(stem.name + '-metrics.json'), row)
            save_json(stem.with_name(stem.name + '-events.json'), {'events': events})
            records.append(row)
    oracles = []
    for item in [item for item in inputs if item['model'] == MODELS[0]]:
        for frames in [0, 13]:
            values, conflicts = delayed_oracle(item['reference'], 1200, base, 12., frames)
            for backdate in [.06, 0.]:
                parameters = {**config['baseline'], 'onset_backdate_seconds': backdate}
                notes, events, clamps = replay(values, base, parameters, 12., frames * .01)
                raw_notes, raw_events, _ = replay(values, base, parameters, 12., 0.)
                proof = projection_proof(raw_events, events, .13) if frames else None
                row = {'id': item['id'], 'delay_frames': frames, 'backdate_seconds': backdate,
                       'notes': notes, 'events': events, 'source_clamps': clamps, 'projection_proof': proof,
                       'conflicts': conflicts, 'privileged_labels': True,
                       **score_case(item['reference'], notes, score, 12.)}
                oracles.append(row)
    save_json(output / 'oracle.json', oracles)
    summary = {'modes': {mode: summarize([row for row in records if row['mode'] == mode])
                         for mode in dict.fromkeys(row['mode'] for row in records)},
               'oracles': {f'delay{frames}_backdate{int(backdate*1000)}': summarize(
                   [row for row in oracles if row['delay_frames'] == frames and row['backdate_seconds'] == backdate])
                   for frames in [0, 13] for backdate in [.06, 0.]},
               'raw_heads_unchanged': True, 'new_optimizer_updates': 0, 'neural_inference_calls': 0,
               'test_inference_calls': 0, 'production_promoted': False,
               'identity_reproductions': 24, 'grid_cases': 648, 'oracle_cases': 48,
               'latency_scope': 'Zero-compute packet availability lower bounds; no end-to-end processing or microphone claim.'}
    save_json(output / 'validation/summary.json', summary)
    if sources() != protocol['source_sha256'] or any(sha256(ROOT / name) != digest for name, digest in bindings.items()):
        raise ValueError('An immutable producer/input changed during the experiment.')
    return summary


def load_records(output):
    return [read(path) for path in sorted((Path(output) / 'validation/results').glob('*/*-metrics.json'))]
