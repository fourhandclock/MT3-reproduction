"""Full within-string confusions and distinct frame/equal-position denominators."""

import numpy as np

from coverage20.analysis import validate_states


def priors(counts):
    values = np.asarray(counts)
    if (values.shape != (6, 25) or not np.issubdtype(values.dtype, np.integer) or np.any(values < 0)):
        raise ValueError('Training counts must be nonnegative integer [6,25]')
    totals = values.sum(1)
    probability = np.divide(values, totals[:, None], out=np.zeros((6, 25), np.float64), where=totals[:, None] > 0)
    modal = [int(row.argmax()) if total else None for row, total in zip(values, totals)]
    top1 = np.zeros((6, 25), np.float64)
    for string, fret in enumerate(modal):
        if fret is not None:
            top1[string, fret] = 1
    return {'counts': values.tolist(), 'totals': totals.tolist(), 'modal_frets': modal,
            'distribution': probability.tolist(), 'top1': top1.tolist()}


def aligned(probabilities, state, count=1000, delay=13, leading=26):
    truth = validate_states(state)[:count]
    value = np.asarray(probabilities)[delay:delay + count]
    if (len(truth) != count or value.shape != (count, 6, 26)
            or not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1)
            or not np.allclose(value.sum(-1), 1, atol=2e-5)):
        raise ValueError('Malformed or insufficient saved state probabilities')
    active = (np.arange(count) >= leading)[:, None] & (truth >= 0) & (truth < 25)
    frame, string = np.where(active)
    best = value[..., :25].argmax(-1)
    return string, truth[frame, string].astype(np.int64), best[frame, string]


def confusion(strings, truth, predicted):
    arrays = [np.asarray(a) for a in (strings, truth, predicted)]
    if (any(a.ndim != 1 or not np.issubdtype(a.dtype, np.integer) for a in arrays)
            or len({len(a) for a in arrays}) != 1
            or any(np.any((a < 0) | (a >= limit)) for a, limit in zip(arrays, (6, 25, 25)))):
        raise ValueError('Invalid string/fret observation vectors')
    s, t, p = arrays
    return np.bincount(s * 625 + t * 25 + p, minlength=3750).reshape(6, 25, 25)


def expected_confusion(reference_counts, prior):
    reference = np.asarray(reference_counts)
    probability = np.asarray(prior, np.float64)
    if (reference.shape != (6, 25) or not np.issubdtype(reference.dtype, np.integer)
            or np.any(reference < 0) or probability.shape != (6, 25)
            or not np.isfinite(probability).all() or np.any(probability < 0)):
        raise ValueError('Invalid reference counts or training prior')
    totals = probability.sum(1)
    if np.any((reference.sum(1) > 0) & ~np.isclose(totals, 1, atol=1e-12, rtol=0)):
        raise ValueError('A GT-supported string has no normalized training prior')
    if np.any(~np.isclose(totals, 0, atol=1e-12, rtol=0) & ~np.isclose(totals, 1, atol=1e-12, rtol=0)):
        raise ValueError('Prior rows must sum to zero or one')
    return reference[:, :, None] * probability[:, None, :]


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def describe(matrix, reference_counts, training_modal):
    matrix = np.asarray(matrix, np.float64)
    reference = np.asarray(reference_counts)
    if (matrix.shape != (6, 25, 25) or reference.shape != (6, 25)
            or not np.issubdtype(reference.dtype, np.integer) or np.any(reference < 0)
            or not np.isfinite(matrix).all() or np.any(matrix < 0)
            or not np.allclose(matrix.sum(2), reference, rtol=0, atol=1e-8)):
        raise ValueError('Confusion does not preserve exact reference support')
    diagonal = np.diagonal(matrix, axis1=1, axis2=2)
    prediction = matrix.sum(1)
    total = int(reference.sum())
    per_string, classes = [], []
    independent_correct = modal_mass = peak_mass = 0.
    for string in range(6):
        n = int(reference[string].sum())
        correct = float(diagonal[string].sum())
        modal = training_modal[string]
        peak = int(prediction[string].argmax()) if n else None
        independent = float(np.dot(reference[string], prediction[string] / n)) if n else 0.
        on_modal = float(prediction[string, modal]) if n and modal is not None else 0.
        top = float(prediction[string].max()) if n else 0.
        independent_correct += independent
        modal_mass += on_modal
        peak_mass += top
        per_string.append({'string': string, 'reference_active_frames': n, 'correct_mass': correct,
            'accuracy': ratio(correct, n), 'training_modal_fret': modal, 'top_predicted_fret': peak,
            'top_prediction_share': ratio(top, n), 'fraction_on_training_modal': ratio(on_modal, n),
            'independent_marginal_expected_accuracy': ratio(independent, n),
            'accuracy_minus_independent_marginals': ratio(correct - independent, n)})
        for fret in range(25):
            support = int(reference[string, fret])
            classes.append({'string': string, 'fret': fret, 'reference_active_frames': support,
                'correct_mass': float(diagonal[string, fret]), 'accuracy': ratio(diagonal[string, fret], support),
                'predicted_mass': float(prediction[string, fret])})
    supported = [r['accuracy'] for r in classes if r['reference_active_frames']]
    strings_supported = [r['accuracy'] for r in per_string if r['reference_active_frames']]
    correct = float(diagonal.sum())
    return {'reference_active_frames': total, 'correct_mass': correct,
        'accuracy_frame_weighted': ratio(correct, total),
        'accuracy_equal_supported_position': float(np.mean(supported)) if supported else None,
        'accuracy_equal_supported_string': float(np.mean(strings_supported)) if strings_supported else None,
        'supported_positions': len(supported), 'supported_strings': len(strings_supported),
        'fraction_on_training_modal': ratio(modal_mass, total),
        'pooled_per_string_top_prediction_share': ratio(peak_mass, total),
        'independent_marginal_expected_accuracy': ratio(independent_correct, total),
        'accuracy_minus_independent_marginals': ratio(correct - independent_correct, total),
        'strings': per_string, 'classes': classes}
