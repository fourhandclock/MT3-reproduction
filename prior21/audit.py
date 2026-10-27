"""Reconstruct best-active choices with scalar comparisons and expected priors with loops."""

import csv
from pathlib import Path
import numpy as np

from .data import HISTORIES, MODES, PREDICTORS, ROOT, SOURCE, load_inputs, no_neural_runtime, read, save, sha
from .workflow import build, csv_rows, source_hashes


def audit(config, output):
    no_neural_runtime()
    output = Path(output)
    if read(output / 'protocol.json')['source_sha256'] != source_hashes():
        raise ValueError('Stage21 producers changed')
    training, summary, cases, matrices, bindings, _ = build(config)
    for name, value in [('priors.json', training), ('summary.json', summary), ('cases.json', cases),
                        ('confusions.json', matrices), ('input-bindings.json', bindings)]:
        if read(output / name) != value:
            raise ValueError('Saved Stage21 result does not reproduce: ' + name)
    _, coverage, records, _ = load_inputs(config)
    independent = {mode: np.zeros((6, 25, 25), np.int64) for mode in MODES}
    for record in records:
        for mode in MODES:
            path = ROOT / 'outputs/fret19/validation/probabilities' / mode / (record['id'] + '.npz')
            with np.load(path, allow_pickle=False) as arrays:
                probability = arrays['state']
            for frame in range(26, 1000):
                for string in range(6):
                    truth = int(record['state'][frame, string])
                    if 0 <= truth < 25:
                        # max returns the first fret on ties, independently of NumPy argmax.
                        best = max(range(25), key=lambda fret: float(probability[frame + 13, string, fret]))
                        independent[mode][string, truth, best] += 1
    for mode in MODES:
        actual = independent[mode]
        if not np.array_equal(actual, matrices[mode]['model_best_active']):
            raise ValueError('Scalar best-active confusion differs')
        reference = actual.sum(2)
        history = coverage['histories'][HISTORIES[mode]]
        for predictor in PREDICTORS[1:]:
            expected = np.zeros((6, 25, 25), np.float64)
            for string in range(6):
                if predictor == 'uniform25_distribution':
                    weights = [1 / 25] * 25
                else:
                    source, kind = predictor.rsplit('_', 1)
                    counts = history[source][string]
                    total = sum(counts)
                    if not total and int(reference[string].sum()):
                        raise ValueError('Unsupported training string')
                    modal = max(range(25), key=lambda fret: counts[fret]) if total else None
                    weights = [float(fret == modal) if kind == 'top1' else counts[fret] / total if total else 0
                               for fret in range(25)]
                for gt in range(25):
                    for predicted in range(25):
                        expected[string, gt, predicted] = int(reference[string, gt]) * weights[predicted]
            if not np.allclose(expected, matrices[mode][predictor], rtol=0, atol=1e-8):
                raise ValueError('Independent expected-prior confusion differs')
        source_support = sum(row['reference_active'] for row in read(SOURCE / 'summary.json')['modes'][mode]['policies']['argmax']['classes'])
        if int(actual.sum()) != source_support:
            raise ValueError('GT-active support changed')
    exports = csv_rows(training, summary, cases, matrices)
    for name, expected in exports.items():
        with (output / name).open(encoding='utf-8', newline='') as stream:
            observed = list(csv.DictReader(stream))
        normalized = [{key: '' if value is None else str(value) for key, value in row.items()} for row in expected]
        if observed != normalized:
            raise ValueError('Exact CSV values differ: ' + name)
    no_neural_runtime()
    names = ['protocol.json', 'priors.json', 'summary.json', 'cases.json', 'confusions.json', 'input-bindings.json', *exports]
    result = {'passed': True, 'validation_cases': len(cases), 'predictors_per_model': 8,
        'scalar_confusion_reconstructions': 3, 'prior_confusion_reconstructions': 21,
        'active_frames_per_model': int(independent[MODES[0]].sum()),
        'exact_csv_rows': {name: len(rows) for name, rows in exports.items()},
        'new_optimizer_updates': 0, 'neural_inference_calls': 0, 'test_inference_calls': 0,
        'new_clock_measurements': 0, 'production_promoted': False,
        'source_sha256': source_hashes(), 'input_binding_count': len(bindings),
        'artifact_sha256': {name: sha(output / name) for name in names}}
    save(output / 'audit.json', result)
    print('Stage21 audit passed: scalar model choices, expected priors and exact CSV values.', flush=True)
    return result
