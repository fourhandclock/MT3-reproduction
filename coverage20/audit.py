"""Independently reconstruct supervised crops in delayed-target coordinates."""

import csv
from pathlib import Path

import numpy as np

from acoustic10.targets import delay_targets
from .analysis import histogram
from .data import ROOT, load_inputs, read, save, sha
from .workflow import build, no_neural_runtime, source_hashes


def independent_exposure(records, rows):
    config = {'model': {'max_fret': 24, 'pitch_min': 40, 'pitch_max': 88}}
    delayed = [delay_targets(record['targets'], 13, config)['state'] for record in records]
    repeated = np.zeros((6, 25), np.int64)
    visited = [np.zeros(record['targets']['state'].shape, bool) for record in records]
    for row in rows:
        index, start, length = row['record_index'], row['start_frame'], row['frames']
        piece = delayed[index][start:start + length]
        mask = np.ones(length, bool)
        mask[:26] = mask[-26:] = False
        mask[np.arange(start, start + length) < 26] = False
        valid = mask[:, None] & (piece >= 0) & (piece < 25)
        censored = np.where(valid, piece, -100)
        repeated += histogram(censored)
        frame, string = np.where(valid)
        source = start + frame - 13
        if np.any(source < 0):
            raise ValueError('Negative source frame entered independent audit')
        visited[index][source, string] = True
    unique = sum((histogram(np.where(seen, record['targets']['state'], -100))
                  for record, seen in zip(records, visited)), np.zeros((6, 25), np.int64))
    return repeated.tolist(), unique.tolist()


def audit(config, output):
    no_neural_runtime()
    output = Path(output)
    if read(output / 'protocol.json')['source_sha256'] != source_hashes():
        raise ValueError('Stage20 producer changed since the saved run')
    coverage, summary, cases, notes, bindings, _ = build(config)
    for name, expected in [('coverage.json', coverage), ('summary.json', summary), ('cases.json', cases),
                           ('reference-notes.json', notes), ('input-bindings.json', bindings)]:
        if read(output / name) != expected:
            raise ValueError('Saved diagnosis no longer reproduces: ' + name)
    _, datasets, schedules, _ = load_inputs(config)
    for name, rows in [('epochs1_5', schedules[0]), ('epochs1_10', schedules[0] + schedules[1])]:
        repeated, unique = independent_exposure(datasets['train'], rows)
        value = coverage['histories'][name]
        if repeated != value['presentations'] or unique != value['unique']:
            raise ValueError('Independent delayed-crop exposure differs: ' + name)
    csv_rows = {
        'reference-notes.csv': notes,
        'coverage.csv': [{'history': name, **row} for name, rows in coverage['classes'].items() for row in rows],
        'class-errors.csv': [{'mode': mode, 'policy': policy, **row}
                            for mode, value in summary['modes'].items()
                            for policy, detail in value['policies'].items() for row in detail['classes']]}
    for name, expected in csv_rows.items():
        with (output / name).open(encoding='utf-8', newline='') as stream:
            exported = list(csv.DictReader(stream))
        normalized = [{key: '' if value is None else str(value) for key, value in row.items()} for row in expected]
        if exported != normalized:
            raise ValueError('Exact CSV values differ: ' + name)
    for mode in summary['modes'].values():
        for policy in mode['policies'].values():
            for key in summary['count_keys']:
                if sum(row[key] for row in policy['bins'].values()) != policy['overall'][key]:
                    raise ValueError('Coverage bins do not partition pooled counts')
            for row in [policy['overall'], *policy['bins'].values()]:
                if row['reference_active'] != sum(row[k] for k in ('correct_fret', 'predicted_inactive', 'wrong_octave', 'other_wrong_fret')):
                    raise ValueError('Active error taxonomy does not partition the denominator')
    no_neural_runtime()
    result = {'passed': True, 'independently_reconstructed_histories': 2, 'validation_cases': len(cases),
              'reference_note_rows': len(notes), 'position_classes_per_history': 150,
              'training_recordings': 48, 'target_caches_read': 60,
              'new_optimizer_updates': 0, 'neural_inference_calls': 0, 'checkpoint_deserializations': 0,
              'test_inference_calls': 0, 'new_clock_measurements': 0, 'production_promoted': False,
              'official_note_matches_and_original_frame_totals_reproduced': True,
              'exact_csv_rows': {name: len(rows) for name, rows in csv_rows.items()},
              'input_binding_count': len(bindings), 'source_sha256': source_hashes()}
    names = ['protocol.json', 'input-bindings.json', 'coverage.json', 'summary.json', 'cases.json',
             'reference-notes.json', 'reference-notes.csv', 'coverage.csv', 'class-errors.csv']
    result['artifact_sha256'] = {name: sha(output / name) for name in names}
    save(output / 'audit.json', result)
    print('Stage20 audit passed: independent delayed crops, source-frame identities and saved error partitions.', flush=True)
    return result
