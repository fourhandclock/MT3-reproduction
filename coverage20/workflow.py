"""Count saved supervision and associate it with archived validation errors."""

import csv
from pathlib import Path
import sys

import numpy as np

from baseline.evaluation import match_notes, reference_notes
from .analysis import COUNT_KEYS, class_rows, exposure, frame_counts, grouped_counts, rates
from .data import BINS, HISTORIES, MODES, ROOT, SOURCE, fingerprint, load_inputs, read, save, sha

PITCH_GROUPS = ('position_seen', 'pitch_seen_elsewhere', 'pitch_unseen')


def source_hashes():
    paths = list((ROOT / 'coverage20').glob('*.py')) + [ROOT / 'configs/coverage20.json',
            ROOT / 'baseline/evaluation.py', ROOT / 'acoustic10/targets.py']
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def no_neural_runtime():
    if any(name in sys.modules for name in ('torch', 'tensorflow', 'jax')):
        raise ValueError('Stage 20 must run without importing a neural runtime')


def note_rows(row, classes, score, annotation):
    reference = reference_notes(annotation)
    scored = [note for note in reference if 0 <= note['start'] < score['score_seconds']]
    metrics = row['pitch_metrics']
    if reference != row['reference'] or scored != metrics['reference']:
        raise ValueError('Archived reference notes differ from source annotations')
    predicted = [note for note in row['notes'] if 0 <= note['start'] < score['score_seconds']]
    matches = [[int(i), int(j)] for i, j in match_notes(scored, predicted, score)]
    if (predicted != metrics['predicted'] or matches != metrics['onset_matches']
            or len(matches) != metrics['onset']['true_positives']):
        raise ValueError('Official one-to-one onset matches do not reproduce')
    matched = dict(matches)
    result = []
    for index, note in enumerate(scored):
        string, fret = note['string'], note['fret']
        if not 0 <= string < 6 or not 0 <= fret < 25:
            raise ValueError('Reference lies outside the fixed fretboard')
        coverage = classes[string * 25 + fret]
        if coverage['pitch'] != note['pitch']:
            raise ValueError('Reference pitch/string/fret identity differs')
        prediction_index = matched.get(index)
        same_string = prediction_index is not None and predicted[prediction_index]['string'] == string
        result.append({'id': row['id'], 'mode': row['mode'], 'reference_index': index,
                       **note, 'bin': coverage['bin'], 'presentations': coverage['presentations'],
                       'available': coverage['available'], 'unique': coverage['unique'],
                       'other_string_presentations': coverage['other_string_presentations'],
                       'pitch_coverage': coverage['pitch_coverage'],
                       'official_matched': prediction_index is not None,
                       'official_prediction_index': prediction_index,
                       'official_matched_same_string': bool(same_string)})
    return result


def verify_saved_frame_totals(policy, archived):
    values = np.asarray(policy['class_counts']).sum(0)
    expected = archived['counts']
    pairs = [('reference_active_frames', values[0]), ('correct_fret', values[1]),
             ('predicted_inactive', values[2]), ('wrong_octave_same_string', values[3]),
             ('other_wrong_fret', values[4]), ('active_on_reference_inactive', values[5]),
             ('reference_inactive_frames', policy['reference_inactive_frames']),
             ('valid_string_frames', policy['valid_string_frames'])]
    if any(expected[key] != int(value) for key, value in pairs):
        raise ValueError('Original Stage 19 frame totals no longer reproduce')


def build(config):
    proposal, datasets, schedules, bindings = load_inputs(config)
    histories = {'epochs1_5': exposure(datasets['train'], schedules[0]),
                 'epochs1_10': exposure(datasets['train'], schedules[0] + schedules[1])}
    incremental = exposure(datasets['train'], schedules[1])
    expected = read(SOURCE / 'verification.json')['auxiliary_active_support_per_epoch']
    if list(incremental['per_epoch_active_presentations'].values()) != expected:
        raise ValueError('Epochs6-10 support differs from actual training receipts')
    classes = {name: class_rows(value) for name, value in histories.items()}
    score = read(SOURCE / 'validation/run.json')['scoring_config']
    cases, notes = [], []
    total = {mode: {policy: np.zeros((150, len(COUNT_KEYS)), np.int64)
                    for policy in ('argmax', 'decoder_activity')} for mode in MODES}
    inactive = {mode: 0 for mode in MODES}
    valid = {mode: 0 for mode in MODES}
    for record in datasets['validation']:
        for mode in MODES:
            history = HISTORIES[mode]
            path = SOURCE / 'validation/results' / mode / (record['id'] + '-metrics.json')
            row = read(path)
            if row['id'] != record['id'] or row['mode'] != mode:
                raise ValueError('Archived validation case identity changed')
            probability_path = SOURCE / row['probability_path']
            name = probability_path.relative_to(ROOT).as_posix()
            if bindings.get(name) != row['probability_sha256']:
                raise ValueError('Saved probability recording is not hash-bound')
            with np.load(probability_path, allow_pickle=False) as archive:
                policies = frame_counts(archive['state'], record['targets'])
            reference_rows = note_rows(row, classes[history], score, ROOT / record['entry']['annotation'])
            notes.extend(reference_rows)
            case = {'id': record['id'], 'mode': mode, 'history': history, 'policies': {},
                    'reference_notes': len(reference_rows), 'official_matched_notes': sum(n['official_matched'] for n in reference_rows)}
            note_counts = np.zeros((150, 3), np.int64)
            for note in reference_rows:
                note_counts[note['string'] * 25 + note['fret']] += [1, note['official_matched'], note['official_matched_same_string']]
            for policy, value in policies.items():
                verify_saved_frame_totals(value, row['frame_diagnosis']['policies'][policy])
                counts = np.asarray(value['class_counts'])
                counts[:, 6:] = note_counts
                total[mode][policy] += counts
                value['class_counts'] = counts.tolist()
                value['bins'] = grouped_counts(counts, classes[history], 'bin', BINS)
                value['pitch_coverage'] = grouped_counts(counts, classes[history], 'pitch_coverage', PITCH_GROUPS)
                case['policies'][policy] = value
            inactive[mode] += policies['argmax']['reference_inactive_frames']
            valid[mode] += policies['argmax']['valid_string_frames']
            cases.append(case)
    modes = {}
    for mode in MODES:
        history = HISTORIES[mode]
        modes[mode] = {'history': history, 'policies': {}}
        for policy, counts in total[mode].items():
            overall = rates(counts.sum(0))
            overall.update(reference_inactive_frames=inactive[mode], valid_string_frames=valid[mode],
                           false_activity_rate_on_reference_inactive=float(counts[:, 5].sum() / inactive[mode]))
            bins = grouped_counts(counts, classes[history], 'bin', BINS)
            for value in bins.values():
                value['share_of_all_active_errors'] = value['error_count'] / overall['error_count'] if overall['error_count'] else None
                value['false_activity_fraction_of_all_reference_inactive'] = value['false_active_on_reference_inactive'] / inactive[mode]
            modes[mode]['policies'][policy] = {'overall': overall, 'bins': bins,
                'pitch_coverage': grouped_counts(counts, classes[history], 'pitch_coverage', PITCH_GROUPS),
                'classes': [{**label, **rates(count)} for label, count in zip(classes[history], counts)]}
    coverage = {'histories': histories, 'classes': classes, 'incremental_epochs6_10': incremental,
                'model_histories': HISTORIES, 'shared_child_exposure_counted_once_per_history': True}
    summary = {'stage': 20, 'new_optimizer_updates': 0, 'neural_inference_calls': 0,
               'test_inference_calls': 0, 'checkpoint_deserializations': 0, 'waveforms_loaded': 0,
               'feature_arrays_loaded': 0, 'new_clock_measurements': 0, 'calibration_searches': 0,
               'production_promoted': False, 'model_selection_performed': False,
               'validation_cases': len(cases), 'validation_recordings': 12, 'training_recordings': 48,
               'source_frames': 1000, 'source_delay_frames': 13, 'leading_mask_frames': 26,
               'bins': list(BINS), 'count_keys': list(COUNT_KEYS), 'modes': modes,
               'denominators': {
                 'active_errors': 'GT active string-frames; bin is the GT position\'s own-history repeated presentation count.',
                 'false_activity': 'GT inactive string-frames, grouped separately by PREDICTED position coverage; denominator is ALL valid GT inactive string-frames.',
                 'notes': 'All official reference onsets in [0,10) including those before the 0.26s frame mask; official pitch/onset matching ignores string.',
                 'zero_support': 'Null rates mean zero denominator; zero counts are retained for every fixed bin.',
                 'units': 'Pooled string-frames and pooled reference notes; no macro clip F1 is inferred.'},
               'limitations': proposal['limitations']}
    return coverage, summary, cases, notes, bindings, proposal


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config, output):
    output = Path(output)
    if (output / 'verification.json').exists():
        raise ValueError('Stage 20 is already verified; preserve its completed evidence')
    no_neural_runtime()
    coverage, summary, cases, notes, bindings, proposal = build(config)
    no_neural_runtime()
    output.mkdir(parents=True, exist_ok=True)
    save(output / 'protocol.json', {'config': config, 'proposal': proposal, 'source_sha256': source_hashes(),
         'new_optimizer_updates': 0, 'neural_inference_calls': 0, 'test_inference_allowed': False,
         'production_promotion': False, 'new_clock_measurements': 0})
    save(output / 'input-bindings.json', bindings)
    save(output / 'coverage.json', coverage)
    save(output / 'summary.json', summary)
    save(output / 'cases.json', cases)
    save(output / 'reference-notes.json', notes)
    write_csv(output / 'reference-notes.csv', notes)
    write_csv(output / 'coverage.csv', [{'history': name, **row} for name, rows in coverage['classes'].items() for row in rows])
    write_csv(output / 'class-errors.csv', [{'mode': mode, 'policy': policy, **row}
              for mode, value in summary['modes'].items() for policy, detail in value['policies'].items() for row in detail['classes']])
    print('Stage20 counted 48 training recordings and 36 saved validation cases; training=0, inference=0.', flush=True)
    return summary
