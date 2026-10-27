"""Explicit source-frame identities, fixed coverage bins and conditional error rates."""

import numpy as np

from .data import BINS, TUNING

COUNT_KEYS = ('reference_active', 'correct_fret', 'predicted_inactive',
              'wrong_octave', 'other_wrong_fret', 'false_active_on_reference_inactive',
              'reference_notes', 'official_matched_notes', 'official_matched_same_string')


def validate_states(state):
    state = np.asarray(state)
    if (state.ndim != 2 or state.shape[1] != 6 or not np.issubdtype(state.dtype, np.integer)
            or np.any((state != -100) & ((state < 0) | (state > 25)))):
        raise ValueError('Expected integer [frames,6] labels: frets0-24, inactive25, conflict-100')
    return state


def bin_index(count):
    if count < 0:
        raise ValueError('Negative exposure count')
    return 0 if count == 0 else 1 if count < 100 else 2 if count < 1000 else 3


def histogram(state, weights=None):
    state = validate_states(state)
    active = (state >= 0) & (state < 25)
    indices = np.broadcast_to(np.arange(6)[None, :] * 25, state.shape) + state
    values = np.ones(state.shape, np.int64) if weights is None else np.broadcast_to(weights[:, None], state.shape)
    return np.bincount(indices[active], weights=values[active], minlength=150).reshape(6, 25).astype(np.int64)


def exposure(records, rows, delay=13, leading=26, edge=26):
    """Accumulate delayed-source range multiplicity before grouping fixed string labels."""
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in (delay, leading, edge)):
        raise ValueError('Delay and masks must be nonnegative integers')
    states = [validate_states(r['targets']['state']) for r in records]
    difference = [np.zeros(len(state) + 1, np.int64) for state in states]
    per_epoch = {}
    for row in rows:
        index, start, length = row['record_index'], row['start_frame'], row['frames']
        if (not all(isinstance(v, int) and not isinstance(v, bool) for v in (index, start, length))
                or not 0 <= index < len(records) or row['id'] != records[index]['id']
                or start < 0 or length <= 0 or start + length > len(states[index])):
            raise ValueError('Crop identity or bounds differ')
        # In delayed-target coordinates, then map to original recording coordinates.
        left = max(start + edge, leading, delay)
        right = start + length - edge
        source_left, source_right = left - delay, right - delay
        epoch = str(row['epoch'])
        per_epoch.setdefault(epoch, 0)
        if source_right > source_left:
            difference[index][source_left] += 1
            difference[index][source_right] -= 1
            selected = states[index][source_left:source_right]
            per_epoch[epoch] += int(((selected >= 0) & (selected < 25)).sum())
    available = np.zeros((6, 25), np.int64)
    repeated = available.copy()
    unique = available.copy()
    per_record = []
    for record, state, diff in zip(records, states, difference):
        multiplicity = np.cumsum(diff[:-1])
        a, r, u = histogram(state), histogram(state, multiplicity), histogram(state, multiplicity > 0)
        available += a
        repeated += r
        unique += u
        per_record.append({'id': record['id'], 'available': int(a.sum()),
                           'presentations': int(r.sum()), 'unique': int(u.sum())})
    if not np.all(unique <= available) or not np.all(unique <= repeated):
        raise ValueError('Unique/available/repeated exposure invariant failed')
    if int(repeated.sum()) != sum(per_epoch.values()):
        raise ValueError('Epoch and range accumulation disagree')
    return {'available': available.tolist(), 'presentations': repeated.tolist(), 'unique': unique.tolist(),
            'per_epoch_active_presentations': per_epoch, 'per_record': per_record, 'crop_count': len(rows)}


def class_rows(coverage):
    arrays = {key: np.asarray(coverage[key], np.int64) for key in ('available', 'presentations', 'unique')}
    if any(value.shape != (6, 25) for value in arrays.values()):
        raise ValueError('Coverage shape differs')
    pitches = np.asarray(TUNING)[:, None] + np.arange(25)[None, :]
    result = []
    for string in range(6):
        for fret in range(25):
            pitch = int(pitches[string, fret])
            row = {'string': string, 'fret': fret, 'pitch': pitch}
            for key, value in arrays.items():
                row[key] = int(value[string, fret])
                row['other_string_' + key] = int(value[pitches == pitch].sum() - value[string, fret])
            row['bin'] = BINS[bin_index(row['presentations'])]
            row['pitch_coverage'] = ('position_seen' if row['presentations'] else
                                     'pitch_seen_elsewhere' if row['other_string_presentations'] else 'pitch_unseen')
            result.append(row)
    return result


def frame_counts(state_probability, targets, count=1000, delay=13, leading=26, threshold=.25):
    truth = validate_states(targets['state'])[:count]
    probability = np.asarray(state_probability)[delay:delay + count]
    if (len(truth) != count or probability.shape != (count, 6, 26)
            or not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1)
            or not np.allclose(probability.sum(-1), 1, atol=2e-5)):
        raise ValueError('Insufficient or malformed saved source-aligned state probabilities')
    valid = (np.arange(count) >= leading)[:, None] & (truth != -100)
    active = valid & (truth < 25)
    inactive = valid & (truth == 25)
    best = probability[..., :25].argmax(-1)
    top = probability[..., :25].max(-1)
    activity = top / (top + probability[..., 25] + 1e-8)
    result = {}
    for policy, predicted in [('argmax', probability.argmax(-1)),
                              ('decoder_activity', np.where(activity >= threshold, best, 25))]:
        counts = np.zeros((150, len(COUNT_KEYS)), np.int64)
        correct = active & (predicted == truth)
        missing = active & (predicted == 25)
        wrong = active & (predicted != truth) & (predicted < 25)
        octave = wrong & (np.abs(predicted - truth) == 12)
        labels = np.broadcast_to(np.arange(6)[None, :] * 25, truth.shape) + truth
        for column, mask in enumerate([active, correct, missing, octave, wrong & ~octave]):
            counts[:, column] = np.bincount(labels[mask], minlength=150)
        false = inactive & (predicted < 25)
        predicted_labels = np.broadcast_to(np.arange(6)[None, :] * 25, truth.shape) + predicted
        counts[:, 5] = np.bincount(predicted_labels[false], minlength=150)
        result[policy] = {'class_counts': counts.tolist(), 'reference_inactive_frames': int(inactive.sum()),
                          'valid_string_frames': int(valid.sum())}
    return result


def rates(counts):
    values = {key: int(value) for key, value in zip(COUNT_KEYS, counts)}
    n = values['reference_active']
    correct = values['correct_fret']
    values['error_count'] = n - correct
    values['error_rate_on_reference_active'] = (n - correct) / n if n else None
    predicted_active = n - values['predicted_inactive']
    values['conditional_fret_accuracy'] = correct / predicted_active if predicted_active else None
    notes = values['reference_notes']
    values['note_miss_rate'] = (notes - values['official_matched_notes']) / notes if notes else None
    return values


def grouped_counts(counts, classes, field, labels):
    counts = np.asarray(counts, np.int64)
    return {label: rates(counts[[row[field] == label for row in classes]].sum(0)) for label in labels}
