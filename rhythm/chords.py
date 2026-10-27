"""Duration-weighted pitch-class templates and Viterbi smoothing; no learned chord model."""
import re

import numpy as np

ROOTS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
INTERVALS = {'maj': [0, 4, 7], 'min': [0, 3, 7], '7': [0, 4, 7, 10],
             'maj7': [0, 4, 7, 11], 'min7': [0, 3, 7, 10]}


def estimate(notes, grid, config):
    duration = grid['duration_seconds']
    boundaries = sorted(set([0.0, duration] + [t for t in grid['beats'] if 0 < t < duration]))
    labels = ['N'] + [f'{root}:{quality}' for root in ROOTS for quality in config['chord_vocabulary']]
    templates = np.zeros((len(labels), 12))
    for i, label in enumerate(labels[1:], 1):
        root, quality = label.split(':')
        templates[i, [(ROOTS.index(root)+v) % 12 for v in INTERVALS[quality]]] = 1
    norm_templates = templates / np.maximum(1, np.linalg.norm(templates, axis=1, keepdims=True))
    emission = []; chromas = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        window = (end-start)*(config['chord_context_beats']-1)/2
        a, b = max(0, start-window), min(duration, end+window)
        chroma = np.zeros(12)
        for note in notes:
            chroma[int(note['pitch']) % 12] += max(0, min(b, note['end'])-max(a, note['start']))
        chromas.append(chroma.tolist())
        if chroma.sum() < config['chord_minimum_duration']:
            score = np.full(len(labels), -10.0); score[0] = 0
        else:
            score = norm_templates @ (chroma / np.linalg.norm(chroma))
            # Unplayed extensions are penalized so triads do not automatically become sevenths.
            score -= .08 * ((templates > 0) & (chroma[None, :] < .05*chroma.max())).sum(axis=1)
            score[0] = -.25
        emission.append(score)
    emission = np.asarray(emission); costs = emission[0].copy(); paths = []
    transition = config['chord_change_penalty'] * (1-np.eye(len(labels)))
    for score in emission[1:]:
        values = costs[:, None]-transition
        previous = values.argmax(axis=0); paths.append(previous)
        costs = score+values[previous, np.arange(len(labels))]
    states = [int(costs.argmax())]
    for previous in reversed(paths): states.append(int(previous[states[-1]]))
    states.reverse(); result = []
    for i, state in enumerate(states):
        sorted_scores = np.sort(emission[i])
        result.append({'start': boundaries[i], 'end': boundaries[i+1], 'label': labels[state],
                       'local_margin': float(sorted_scores[-1]-sorted_scores[-2]), 'chroma': chromas[i]})
    return result


def reduced(label):
    """Explicit major/minor-root vocabulary; inversions/extensions ignored, others masked."""
    if label == 'N': return 'N'
    match = re.match(r'^([A-G])([#b]*):([^/]+)', label)
    if not match: return None
    letter, accidentals, quality = match.groups()
    root = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}[letter]
    root = (root+accidentals.count('#')-accidentals.count('b')) % 12
    if quality.startswith('min'): basic = 'min'
    elif quality.startswith('maj') or quality == '7': basic = 'maj'
    else: return None
    return f'{root}:{basic}'


def accuracy(reference, predicted, duration):
    supported = correct = root_correct = 0.0
    for ref in reference:
        label = reduced(ref['label'])
        if label is None: continue
        a, b = max(0, ref['start']), min(duration, ref['end'])
        supported += max(0, b-a)
        for est in predicted:
            overlap = max(0, min(b, est['end'])-max(a, est['start']))
            other = reduced(est['label'])
            if label == other: correct += overlap
            if other is not None and label.split(':')[0] == other.split(':')[0]: root_correct += overlap
    return {'major_minor_duration_accuracy': correct/supported if supported else None,
            'root_duration_accuracy': root_correct/supported if supported else None,
            'supported_seconds': supported, 'coverage_fraction': supported/duration,
            'vocabulary': 'Root + major/minor/N; inversions and extensions ignored; unsupported qualities masked.'}
