"""Common note/frame evaluation and explicit crop-boundary handling."""

import json
import warnings
from pathlib import Path

import mir_eval
import numpy as np

STANDARD_TUNING = [40, 45, 50, 55, 59, 64]  # low E to high E


def reference_notes(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    notes = []
    for annotation in data['annotations']:
        if annotation['namespace'] != 'note_midi':
            continue
        string = int(annotation['annotation_metadata']['data_source'])
        for event in annotation['data']:
            start, duration = float(event['time']), float(event['duration'])
            pitch = int(np.rint(float(event['value'])))
            if duration > 0 and 0 <= pitch <= 127:
                notes.append({'start': start, 'end': start + duration, 'pitch': pitch,
                              'string': string, 'fret': pitch - STANDARD_TUNING[string]})
    return sorted(notes, key=lambda note: (note['start'], note['pitch'], note['end']))


def midi_notes(path):
    import pretty_midi
    midi = pretty_midi.PrettyMIDI(str(path))
    notes, drums = [], 0
    for instrument in midi.instruments:
        if instrument.is_drum:
            drums += len(instrument.notes)
            continue
        for note in instrument.notes:
            if note.end > note.start:
                notes.append({'start': float(note.start), 'end': float(note.end), 'pitch': int(note.pitch),
                              'program': int(instrument.program)})
    return sorted(notes, key=lambda note: (note['start'], note['pitch'], note['end'])), drums


def note_arrays(notes):
    intervals = np.array([[note['start'], note['end']] for note in notes], dtype=float).reshape(-1, 2)
    frequencies = mir_eval.util.midi_to_hz(np.array([note['pitch'] for note in notes], dtype=float))
    return intervals, frequencies


def match_notes(reference, predicted, config, offsets=False):
    ref_intervals, ref_pitch = note_arrays(reference)
    est_intervals, est_pitch = note_arrays(predicted)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return mir_eval.transcription.match_notes(
            ref_intervals, ref_pitch, est_intervals, est_pitch,
            onset_tolerance=config['onset_tolerance_seconds'], pitch_tolerance=50.0,
            offset_ratio=config['offset_ratio'] if offsets else None,
            offset_min_tolerance=config['offset_tolerance_seconds'], strict=False,
        )


def scores(matches, reference_count, predicted_count):
    precision = len(matches) / predicted_count if predicted_count else 0.0
    recall = len(matches) / reference_count if reference_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'true_positives': len(matches),
            'reference_count': reference_count, 'predicted_count': predicted_count}


def piano_roll(notes, duration, fps):
    roll = np.zeros((int(np.ceil(duration * fps)), 128), dtype=bool)
    for note in notes:
        start = max(0, int(np.ceil(note['start'] * fps)))
        end = min(len(roll), int(np.ceil(note['end'] * fps)))
        if start < end:
            roll[start:end, note['pitch']] = True
    return roll


def evaluate(reference, predicted, config, clip_duration):
    score_duration = min(config['score_seconds'], clip_duration)
    # Later audio is context only. Do not count note-onsets beyond the scored prefix.
    ref = [dict(note) for note in reference if 0 <= note['start'] < score_duration]
    est = [dict(note) for note in predicted if 0 <= note['start'] < score_duration]
    # Preserve uncropped annotation offsets to identify notes whose release was not audible.
    censored = [index for index, note in enumerate(ref) if note['end'] >= clip_duration]
    onset_matches = match_notes(ref, est, config)
    # For the offset metric, remove predictions assigned to censored reference notes by
    # one-to-one onset matching; preserve other predictions as potential false positives.
    censored_est = {j for i, j in onset_matches if i in censored}
    offset_ref = [note for i, note in enumerate(ref) if i not in censored]
    offset_est = [note for j, note in enumerate(est) if j not in censored_est]
    offset_matches = match_notes(offset_ref, offset_est, config, offsets=True)
    ref_roll = piano_roll(reference, score_duration, config['frame_rate'])
    est_roll = piano_roll(predicted, score_duration, config['frame_rate'])
    frame_tp = int(np.logical_and(ref_roll, est_roll).sum())
    frame_ref, frame_est = int(ref_roll.sum()), int(est_roll.sum())
    frame_precision = frame_tp / frame_est if frame_est else 0.0
    frame_recall = frame_tp / frame_ref if frame_ref else 0.0
    frame_f1 = 2 * frame_precision * frame_recall / (frame_precision + frame_recall) if frame_precision + frame_recall else 0.0
    matched_ref, matched_est = {i for i, _ in onset_matches}, {j for _, j in onset_matches}
    missed = [note for i, note in enumerate(ref) if i not in matched_ref]
    extra = [note for j, note in enumerate(est) if j not in matched_est]
    octave_candidates = sum(any(abs(note['pitch'] - target['pitch']) == 12 and
                                abs(note['start'] - target['start']) <= config['onset_tolerance_seconds']
                                for target in missed) for note in extra)
    timing_candidates = sum(any(note['pitch'] == target['pitch'] and
                                config['onset_tolerance_seconds'] < abs(note['start'] - target['start']) <= 0.2
                                for target in missed) for note in extra)
    return {
        'onset': scores(onset_matches, len(ref), len(est)),
        'onset_offset': scores(offset_matches, len(offset_ref), len(offset_est)),
        'frame': {'precision': frame_precision, 'recall': frame_recall, 'f1': frame_f1,
                  'true_positives': frame_tp, 'reference_count': frame_ref, 'predicted_count': frame_est},
        'score_duration': score_duration, 'right_censored_reference_notes': len(censored),
        'errors': {'missed_notes': len(missed), 'extra_notes': len(extra),
                   'octave_error_candidates': octave_candidates, 'timing_error_candidates': timing_candidates,
                   'median_onset_error_ms': float(np.median([abs(ref[i]['start'] - est[j]['start']) * 1000
                                                           for i, j in onset_matches])) if onset_matches else None,
                   'median_offset_error_ms': float(np.median([abs(ref[i]['end'] - est[j]['end']) * 1000
                                                            for i, j in onset_matches if i not in censored]))
                   if any(i not in censored for i, _ in onset_matches) else None},
        'onset_matches': [[int(i), int(j)] for i, j in onset_matches],
        'reference': ref, 'predicted': est,
    }
