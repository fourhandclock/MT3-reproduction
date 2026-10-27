"""Packet-only replay: these clocks exclude neural, device and microphone cost."""

import itertools
import math

import numpy as np

from acoustic10.decoder import SourceTimeDecoder
from acoustic10.targets import delay_targets
from baseline.evaluation import evaluate
from cnn_gru.inference import tablature_metrics
from diagnosis9.oracles import oracle_probabilities


BASELINE = {'onset_threshold': .55, 'activity_threshold': .35, 'onset_backdate_seconds': .06}
MODELS = ['group8_500', 'group1_500']


def validate_config(config):
    expected = {'stage': 14, 'source_root': 'outputs/optimization13', 'models': MODELS,
                'source_delay_frames': 13, 'packet_ms': 20, 'baseline': BASELINE,
                'grid': {'onset_threshold': [.55, .4, .7], 'activity_threshold': [.35, .25, .5],
                         'onset_backdate_seconds': [.06, 0., .03]},
                'new_optimizer_updates': 0, 'neural_inference_calls': 0,
                'test_inference_allowed': False, 'production_promotion': False}
    if set(config) != set(expected) | {'selection_rule'}:
        raise ValueError('Unexpected or missing Stage14 configuration fields.')
    if any(config[key] != value for key, value in expected.items()):
        raise ValueError('The prespecified zero-training decoder protocol changed.')
    if not isinstance(config['selection_rule'], str) or not config['selection_rule']:
        raise ValueError('A declared selection rule is required.')


def candidates(config):
    validate_config(config)
    names = list(BASELINE)
    result = [dict(zip(names, values)) for values in itertools.product(*(config['grid'][name] for name in names))]
    if len(result) != 27 or result[0] != BASELINE:
        raise ValueError('Candidate zero must preserve the original decoder.')
    return result


def choose(rows):
    """Stable identity-first tie-breaking; never reads head metrics or test data."""
    if not rows or [row['candidate'] for row in rows] != list(range(len(rows))):
        raise ValueError('Selection needs unique candidates in their prespecified order.')
    for row in rows:
        for key in ['onset_f1', 'onset_offset_f1']:
            if not math.isfinite(row[key]) or not 0 <= row[key] <= 1:
                raise ValueError('Selection score must be finite and in [0,1].')
    return max(rows, key=lambda row: (row['onset_f1'], row['onset_offset_f1']))['candidate']


def replay(probabilities, config, parameters, duration, source_delay_seconds=.13, packet_ms=20):
    period = config['feature']['hop_length'] / config['feature']['sample_rate']
    block = packet_ms / 1000 / period
    if not math.isfinite(block) or block < 1 or abs(block - round(block)) > 1e-9:
        raise ValueError('Packet must contain a positive integer number of hops.')
    if not math.isfinite(duration) or duration < 0:
        raise ValueError('Duration must be finite and nonnegative.')
    if set(probabilities) != {'state', 'onset', 'offset', 'pitch'}:
        raise ValueError('Exactly the four cached heads are required.')
    stream = SourceTimeDecoder(config, parameters, source_delay_seconds)
    # Validate all heads before producing any events, including empty inputs.
    _, count = stream._validate_probabilities(probabilities)
    if count * period > duration + 1e-9:
        raise ValueError('Cached frames extend beyond observed audio.')
    block = round(block)
    for start in range(0, count, block):
        end = min(count, start + block)
        # Match PCMStream's integer-sample arrival arithmetic byte for byte.
        arrival = min(duration, (start + block) * config['feature']['hop_length'] / config['feature']['sample_rate'])
        stream.push({key: value[start:end] for key, value in probabilities.items()}, arrival)
    stream.finish(duration, available_at_audio_seconds=duration)
    # Deliberately omit delivered_at_seconds. Only packet availability is known.
    return stream.notes, stream.events, stream.clamp_counts


def delayed_oracle(reference, count, config, duration, delay_frames):
    """Shift complete ideal label arrays before decoding, including conflict masks."""
    from streaming.targets import make_targets
    labels, _ = make_targets(reference, count, config, duration)
    labels = delay_targets(labels, delay_frames, config)
    inactive = int(config['model']['max_fret']) + 1
    states = labels['state']
    values = {'state': np.eye(inactive + 1, dtype=np.float32)[np.where(states == -100, inactive, states)],
              **{key: labels[key].astype(np.float32) for key in ['onset', 'offset', 'pitch']}}
    _, conflicts = oracle_probabilities(reference, count, config, duration)
    return values, conflicts


def score_case(reference, notes, score, duration):
    return {'pitch_metrics': evaluate(reference, notes, score, duration),
            'tab_metrics': tablature_metrics(reference, notes, score, score['score_seconds'], duration)}


def compact(row):
    metrics = row['pitch_metrics']
    return {'onset_f1': metrics['onset']['f1'], 'onset_offset_f1': metrics['onset_offset']['f1'],
            'frame_f1': metrics['frame']['f1'], 'tab_onset_f1': row['tab_metrics']['onset']['f1'],
            'matched_onsets': metrics['onset']['true_positives'],
            'predicted_notes': metrics['onset']['predicted_count'],
            'reference_notes': metrics['onset']['reference_count']}


def summarize(rows):
    if not rows:
        raise ValueError('Cannot summarize an empty comparison.')
    values = [compact(row) for row in rows]
    result = {'count': len(rows)}
    for key in values[0]:
        result[key] = (sum(value[key] for value in values) if key in
                       ['matched_onsets', 'predicted_notes', 'reference_notes'] else
                       float(np.mean([value[key] for value in values])))
    return result


def without_compute(value):
    if isinstance(value, dict):
        return {key: without_compute(item) for key, item in value.items()
                if key not in ['delivered_at_seconds', 'on_delivered_at_seconds', 'off_delivered_at_seconds']}
    if isinstance(value, list):
        return [without_compute(item) for item in value]
    return value


def packet_latency(reference, notes, score, duration):
    """Availability minus original matched onset; a zero-compute lower bound."""
    from baseline.evaluation import match_notes
    limit = min(duration, score['score_seconds'])
    ref = [note for note in reference if 0 <= note['start'] < limit]
    est = [note for note in notes if 0 <= note['start'] < limit]
    samples = [1000 * (est[j]['on_emitted_at_audio_seconds'] - ref[i]['start'])
               for i, j in match_notes(ref, est, score)]
    return {'samples_ms': samples, 'count': len(samples),
            'p50_ms': float(np.quantile(samples, .5)) if samples else None,
            'p95_ms': float(np.quantile(samples, .95)) if samples else None,
            'scope': 'Packet availability only; excludes all processing, device and microphone cost. Matched onsets only.'}
