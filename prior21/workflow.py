"""All declared priors are reported; none are selected using validation scores."""

import csv
from pathlib import Path
import numpy as np

from .analysis import aligned, confusion, describe, expected_confusion, priors
from .data import COUNT_SOURCES, HISTORIES, MODES, PREDICTORS, ROOT, SOURCE, load_inputs, no_neural_runtime, read, save, sha


def source_hashes():
    paths = list((ROOT / 'prior21').glob('*.py')) + [ROOT / 'configs/prior21.json']
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def prior_for(name, history):
    if name == 'uniform25_distribution':
        return np.full((6, 25), 1 / 25, np.float64)
    source, kind = name.rsplit('_', 1)
    return np.asarray(history[source]['top1' if kind == 'top1' else 'distribution'])


def build(config):
    proposal, coverage, records, bindings = load_inputs(config)
    training = {history: {source: priors(value[source]) for source in COUNT_SOURCES}
                for history, value in coverage['histories'].items()}
    source_cases = {(r['mode'], r['id']): r for r in read(SOURCE / 'cases.json')}
    matrices = {mode: {name: np.zeros((6, 25, 25), np.float64) for name in PREDICTORS} for mode in MODES}
    references = {mode: np.zeros((6, 25), np.int64) for mode in MODES}
    cases = []
    for record in records:
        for mode in MODES:
            history = training[HISTORIES[mode]]
            archived = read(ROOT / 'outputs/fret19/validation/results' / mode / (record['id'] + '-metrics.json'))
            name = 'outputs/fret19/' + archived['probability_path']
            if bindings.get(name) != archived['probability_sha256']:
                raise ValueError('Unbound saved probability recording')
            with np.load(ROOT / name, allow_pickle=False) as arrays:
                strings, truth, best = aligned(arrays['state'], record['state'], config['source_frames'],
                                              config['source_delay_frames'], config['leading_mask_frames'])
            matrix = confusion(strings, truth, best)
            reference = matrix.sum(2)
            prior_case = source_cases[(mode, record['id'])]
            if reference.reshape(-1).tolist() != [r[0] for r in prior_case['policies']['argmax']['class_counts']]:
                raise ValueError('GT position support differs from Stage20')
            anchor = archived['frame_diagnosis']['best_active_fret_on_reference_active']
            correct = int(np.diagonal(matrix, axis1=1, axis2=2).sum())
            if anchor['count'] != int(reference.sum()) or anchor['correct'] != correct:
                raise ValueError('Saved Stage19 best-active accuracy does not reproduce')
            references[mode] += reference
            row = {'mode': mode, 'id': record['id'], 'history': HISTORIES[mode], 'predictors': {}}
            for predictor in PREDICTORS:
                value = matrix if predictor == 'model_best_active' else expected_confusion(reference, prior_for(predictor, history))
                matrices[mode][predictor] += value
                detail = describe(value, reference, history['presentations']['modal_frets'])
                row['predictors'][predictor] = {k: v for k, v in detail.items() if k not in ('strings', 'classes')}
            cases.append(row)
    if any(not np.array_equal(references[mode], references[MODES[0]]) for mode in MODES):
        raise ValueError('Models were not evaluated on identical GT-active support')
    modes = {}
    for mode in MODES:
        history = training[HISTORIES[mode]]
        detail = {name: describe(matrix, references[mode], history['presentations']['modal_frets'])
                  for name, matrix in matrices[mode].items()}
        model, primary = detail['model_best_active'], detail[config['primary_prior']]
        differences = []
        for case in cases:
            if case['mode'] == mode:
                left = case['predictors']['model_best_active']['accuracy_frame_weighted']
                right = case['predictors'][config['primary_prior']]['accuracy_frame_weighted']
                differences.append({'id': case['id'], 'difference': left - right if left is not None and right is not None else None})
        compared = [r['difference'] for r in differences if r['difference'] is not None]
        tolerance = config['paired_tie_tolerance']
        modes[mode] = {'history': HISTORIES[mode], 'predictors': detail, 'primary_comparison': {
            'prior': config['primary_prior'],
            'model_minus_prior_frame_accuracy': model['accuracy_frame_weighted'] - primary['accuracy_frame_weighted'],
            'model_minus_prior_equal_position_accuracy': model['accuracy_equal_supported_position'] - primary['accuracy_equal_supported_position'],
            'paired_clips': differences, 'compared_clips': len(compared),
            'model_wins': sum(v > tolerance for v in compared), 'ties': sum(abs(v) <= tolerance for v in compared),
            'model_losses': sum(v < -tolerance for v in compared),
            'mean_paired_clip_difference': float(np.mean(compared)) if compared else None}}
    summary = {'stage': 21, 'primary_prior': config['primary_prior'], 'predictors': list(PREDICTORS), 'modes': modes,
        'validation_cases': len(cases), 'validation_recordings': 12, 'new_optimizer_updates': 0,
        'neural_inference_calls': 0, 'test_inference_calls': 0, 'checkpoint_deserializations': 0,
        'waveforms_loaded': 0, 'feature_arrays_loaded': 0, 'new_clock_measurements': 0,
        'model_selection_performed': False, 'production_promoted': False, 'calibration_searches': 0,
        'alignment': {key: config[key] for key in ('source_frames', 'source_delay_frames', 'leading_mask_frames')},
        'denominators': {
          'support': 'Only GT-active, nonconflicting string-frames; source0:1000, predictions13:1013, first26 source frames masked.',
          'model': 'Argmax over frets0-24 even if inactive25 has the greatest full-state probability. GT activity is oracle support.',
          'frame_weighted': 'Sum of diagonal confusion mass / all supported active frames.',
          'equal_position': 'Mean recall over string/fret positions with GT support; unsupported positions are excluded and have null recall.',
          'equal_string': 'Mean frame accuracy over strings with GT-active support.',
          'distribution': 'Analytic expected confusions from training frequencies; fractional expected mass, no sampled predictions.',
          'independent_marginals': 'Within-string independence expectation using that predictor\'s own output marginal and GT marginal. Diagnostic association, no shuffled model run or significance test.',
          'concentration': 'Training-modal mass uses presentation-count priors. Pooled top share sums each string\'s largest prediction mass before dividing by all active support.'},
        'limitations': proposal['limitations']}
    serial = {mode: {name: matrix.tolist() for name, matrix in values.items()} for mode, values in matrices.items()}
    return training, summary, cases, serial, bindings, proposal


def csv_rows(training, summary, cases, matrices):
    prior_rows = []
    for history, sources in training.items():
        for source, value in sources.items():
            for string in range(6):
                for fret in range(25):
                    prior_rows.append({'history': history, 'count_source': source, 'string': string, 'fret': fret,
                        'count': value['counts'][string][fret], 'probability': value['distribution'][string][fret],
                        'training_modal_fret': value['modal_frets'][string]})
    classes, strings, case_rows, confusions = [], [], [], []
    for mode, value in summary['modes'].items():
        for predictor, metrics in value['predictors'].items():
            classes.extend({'mode': mode, 'predictor': predictor, **row} for row in metrics['classes'])
            strings.extend({'mode': mode, 'predictor': predictor, **row} for row in metrics['strings'])
            kind = ('saved_model_decisions' if predictor == 'model_best_active' else
                    'expected_distribution' if predictor.endswith('_distribution') else 'deterministic_prior')
            for row in metrics['classes']:
                if row['reference_active_frames']:
                    s, gt = row['string'], row['fret']
                    for predicted in range(25):
                        confusions.append({'mode': mode, 'predictor': predictor, 'string': s, 'gt_fret': gt,
                            'predicted_fret': predicted, 'kind': kind, 'reference_active_frames': row['reference_active_frames'],
                            'value': matrices[mode][predictor][s][gt][predicted]})
    for case in cases:
        case_rows.extend({'mode': case['mode'], 'id': case['id'], 'predictor': predictor, **metrics}
                         for predictor, metrics in case['predictors'].items())
    return {'priors.csv': prior_rows, 'classes.csv': classes, 'strings.csv': strings,
            'cases.csv': case_rows, 'confusions.csv': confusions}


def run(config, output):
    output = Path(output)
    if (output / 'verification.json').exists():
        raise ValueError('Stage21 is already verified; preserve its evidence')
    no_neural_runtime()
    training, summary, cases, matrices, bindings, proposal = build(config)
    no_neural_runtime()
    output.mkdir(parents=True, exist_ok=True)
    save(output / 'protocol.json', {'config': config, 'proposal': proposal, 'source_sha256': source_hashes(),
         'new_optimizer_updates': 0, 'neural_inference_calls': 0, 'new_clock_measurements': 0})
    for name, value in [('priors.json', training), ('summary.json', summary), ('cases.json', cases),
                        ('confusions.json', matrices), ('input-bindings.json', bindings)]:
        save(output / name, value)
    for name, rows in csv_rows(training, summary, cases, matrices).items():
        with (output / name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print('Stage21: 36 saved cases, all eight predictors; training=0, inference=0.', flush=True)
    return summary
