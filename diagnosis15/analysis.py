"""One-to-one error associations and source-aligned acoustic evidence.

Association categories partition scored notes; they do not establish causes.
Overlap/fragmentation and repeated-note tags are separate, nonexclusive views.
"""

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from baseline.evaluation import STANDARD_TUNING, match_notes
from diagnosis9.head_audit import binary_metric, runs


CATEGORIES = ['matched_same_string', 'matched_other_string', 'nearby_same_pitch',
              'nearby_octave', 'nearby_other_pitch', 'missed']


def associate(reference, predicted, reference_indices, predicted_indices, category, onset_tolerance):
    """Maximum-cardinality bipartite pairing, then minimum onset discrepancy."""
    if not reference_indices or not predicted_indices:
        return []
    nr, ne = len(reference_indices), len(predicted_indices)
    unmatched = float(nr + ne + 1)
    cost = np.full((nr, ne + nr), unmatched, dtype=np.float64)
    cost[:, :ne] = unmatched * 2
    allowed = np.zeros((nr, ne), bool)
    for a, i in enumerate(reference_indices):
        for b, j in enumerate(predicted_indices):
            left, right = reference[i], predicted[j]
            delta = abs(right['pitch'] - left['pitch'])
            separation = abs(right['start'] - left['start'])
            pitch_ok = (delta == 0 if category == 'nearby_same_pitch' else
                        delta == 12 if category == 'nearby_octave' else delta not in [0, 12])
            if pitch_ok and separation <= onset_tolerance + 1e-9:
                allowed[a, b] = True
                cost[a, b] = separation / max(onset_tolerance, 1e-9)
    rr, cc = linear_sum_assignment(cost)
    return [(reference_indices[a], predicted_indices[b]) for a, b in zip(rr, cc) if b < ne and allowed[a, b]]


def taxonomy(reference, predicted, score, config):
    end = score['score_seconds']
    ref = [dict(n) for n in reference if 0 <= n['start'] < end]
    est = [dict(n) for n in predicted if 0 <= n['start'] < end]
    pairs = [(int(i), int(j)) for i, j in match_notes(ref, est, score)]
    assignments = {i: (j, 'matched_same_string' if ref[i]['string'] == est[j]['string'] else 'matched_other_string')
                   for i, j in pairs}
    used = {j for _, j in pairs}
    for category, tolerance in [('nearby_same_pitch', config['nearby_onset_seconds']),
                                ('nearby_octave', score['onset_tolerance_seconds']),
                                ('nearby_other_pitch', score['onset_tolerance_seconds'])]:
        available_ref = [i for i in range(len(ref)) if i not in assignments]
        available_est = [j for j in range(len(est)) if j not in used]
        for i, j in associate(ref, est, available_ref, available_est, category, tolerance):
            assignments[i] = (j, category)
            used.add(j)
    repeated = set()
    for string in range(6):
        ordered = sorted([i for i, n in enumerate(ref) if n['string'] == string],
                         key=lambda i: (ref[i]['start'], ref[i]['end'], i))
        for before, now in zip(ordered, ordered[1:]):
            if (ref[before]['pitch'] == ref[now]['pitch'] and
                    ref[now]['start'] - ref[before]['end'] <= config['repeated_note_gap_seconds'] + 1e-9):
                repeated.add(now)
    reference_rows, prediction_rows = [], []
    for i, note in enumerate(ref):
        j, category = assignments.get(i, (None, 'missed'))
        fragments = []
        for k, prediction in enumerate(est):
            overlap = min(note['end'], prediction['end'], end) - max(note['start'], prediction['start'], 0.)
            if (note['pitch'] == prediction['pitch'] and note['string'] == prediction['string']
                    and overlap >= config['fragment_minimum_overlap_seconds'] - 1e-9):
                fragments.append(k)
        reference_rows.append({'reference_index': i, 'note': note, 'category': category, 'prediction_index': j,
                               'onset_error_ms': 1000 * (est[j]['start'] - note['start']) if j is not None else None,
                               'repeated_note': i in repeated, 'overlapping_same_string_pitch_predictions': fragments,
                               'multiple_overlaps': len(fragments) >= 2})
    reverse = {j: (i, category) for i, (j, category) in assignments.items()}
    for j, note in enumerate(est):
        i, category = reverse.get(j, (None, 'extra'))
        prediction_rows.append({'prediction_index': j, 'note': note, 'reference_index': i, 'category': category})
    counts = {category: sum(row['category'] == category for row in reference_rows) for category in CATEGORIES}
    return {'reference_rows': reference_rows, 'prediction_rows': prediction_rows, 'reference_counts': counts,
            'extra_predictions': sum(row['category'] == 'extra' for row in prediction_rows),
            'matched_onsets': len(pairs), 'reference_count': len(ref), 'prediction_count': len(est),
            'repeated_reference_count': len(repeated),
            'repeated_reference_matched': sum(i in repeated for i, _ in pairs),
            'reference_notes_with_multiple_overlaps': sum(row['multiple_overlaps'] for row in reference_rows)}


def activity_states(probabilities, threshold, inactive=25):
    best = probabilities[..., :inactive].max(-1)
    activity = best / (best + probabilities[..., inactive] + 1e-8)
    frets = probabilities[..., :inactive].argmax(-1)
    return np.where(activity >= threshold, frets, inactive), frets


def pitch_union(states, pitch_min=40, pitch_max=88, inactive=25):
    result = np.zeros((len(states), pitch_max - pitch_min + 1), bool)
    for string, tuning in enumerate(STANDARD_TUNING):
        rows = np.flatnonzero((states[:, string] >= 0) & (states[:, string] < inactive))
        pitches = states[rows, string] + tuning - pitch_min
        if np.any(pitches < 0) or np.any(pitches >= result.shape[1]):
            raise ValueError('State pitch lies outside the declared pitch range.')
        result[rows, pitches] = True
    return result


def source_arrays(probabilities, targets, config, activity_threshold):
    count, delay = config['source_frames'], config['source_delay_frames']
    if len(probabilities['state']) < count + delay or len(targets['state']) < count:
        raise ValueError('Insufficient future input context for the common original source span.')
    state = probabilities['state'][delay:delay + count]
    observed, best_fret = activity_states(state, activity_threshold)
    mask = np.arange(count) >= config['leading_mask_frames']
    return {'target': targets['state'][:count], 'target_pitch': targets['pitch'][:count].astype(bool),
            'observed': observed, 'best_fret': best_fret, 'mask': mask,
            'state_pitch': pitch_union(observed),
            'auxiliary_pitch': probabilities['pitch'][delay:delay + count] >= config['auxiliary_pitch_threshold']}


def longest_true_run(values):
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def describe_lengths(lengths):
    return {'count': len(lengths), 'p50_frames': float(np.median(lengths)) if lengths else None,
            'fraction_below_confirmation': float(np.mean(np.asarray(lengths) < 4)) if lengths else None}


def frame_diagnosis(arrays):
    truth, observed, mask = arrays['target'], arrays['observed'], arrays['mask']
    valid = mask[:, None] & (truth != -100)
    active = valid & (truth != 25)
    predicted_active = observed != 25
    correct = active & predicted_active & (truth == observed)
    inactive = active & ~predicted_active
    wrong = active & predicted_active & (truth != observed)
    octave = wrong & (np.abs(truth - observed) == 12)
    counts = {'correct_fret': int(correct.sum()), 'predicted_inactive': int(inactive.sum()),
              'wrong_octave_same_string': int(octave.sum()), 'other_wrong_fret': int((wrong & ~octave).sum()),
              'reference_active_frames': int(active.sum()),
              'best_active_fret_correct': int((active & (truth == arrays['best_fret'])).sum()),
              'reference_inactive_frames': int((valid & ~active).sum()),
              'active_on_reference_inactive': int((valid & ~active & predicted_active).sum())}
    errors = Counter(int(value) for value in (observed - truth)[wrong])
    target_runs = runs(np.where(valid, truth, -100), 25)
    observed_runs = runs(np.where(mask[:, None], observed, -100), 25)
    return {'counts': counts, 'wrong_fret_delta_histogram': {str(key): value for key, value in sorted(errors.items())},
            'state_derived_pitch': binary_metric(arrays['target_pitch'][mask], arrays['state_pitch'][mask]),
            'auxiliary_pitch': binary_metric(arrays['target_pitch'][mask], arrays['auxiliary_pitch'][mask]),
            'target_runs': describe_lengths(target_runs), 'predicted_runs': describe_lengths(observed_runs),
            'target_run_lengths': target_runs, 'predicted_run_lengths': observed_runs}


def add_note_evidence(taxonomy_result, arrays, period=.01):
    times = (np.arange(len(arrays['mask'])) + 1) * period
    for row in taxonomy_result['reference_rows']:
        note = row['note']
        valid = (arrays['mask'] & (times + 1e-9 >= note['start']) & (times < note['end'] - 1e-9)
                 & (arrays['target'][:, note['string']] == note['fret']))
        count = int(valid.sum())
        correct = arrays['observed'][:, note['string']] == note['fret']
        row['evidence'] = {'valid_source_frames': count,
                           'same_string_fret_fraction': float(correct[valid].mean()) if count else None,
                           'any_string_pitch_fraction': float(arrays['state_pitch'][valid, note['pitch'] - 40].mean()) if count else None,
                           'auxiliary_pitch_fraction': float(arrays['auxiliary_pitch'][valid, note['pitch'] - 40].mean()) if count else None,
                           'longest_correct_state_run_frames': longest_true_run(correct & valid)}
    return taxonomy_result
