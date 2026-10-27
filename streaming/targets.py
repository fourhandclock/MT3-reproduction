"""Reference labels on the causal frontend's right-edge time grid."""

import math

import numpy as np


def make_targets(notes, count, config, duration):
    """Label frame i at (i + 1) * hop / sample_rate using original note times.

    Note occupancy is half-open [start, end). Boundaries at or beyond the
    observed audio duration are censored before their Gaussian support can
    reach earlier frames. An attack before the first complete frame is assigned
    to that earliest observable frame.
    """
    if not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError("Frame count must be a nonnegative integer")
    duration = float(duration)
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("Audio duration must be nonnegative and finite")
    feature, model = config["feature"], config["model"]
    fps = float(feature["sample_rate"]) / float(feature["hop_length"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Frame rate must be positive and finite")
    if count / fps > duration + 1e-9:
        raise ValueError("Frame count extends beyond the observed audio")
    fret_count = int(model["max_fret"]) + 1
    pitch_min, pitch_max = int(model["pitch_min"]), int(model["pitch_max"])
    states = np.full((count, 6), fret_count, dtype=np.int16)
    occupancy = np.zeros((count, 6, fret_count), dtype=bool)
    onsets = np.zeros((count, 6), dtype=np.float32)
    offsets = np.zeros((count, 6), dtype=np.float32)
    pitch = np.zeros((count, pitch_max - pitch_min + 1), dtype=np.uint8)

    def grid_index(seconds):
        # Tolerance is in frame units and only resolves floating-point noise
        # around exact hop multiples (e.g. 0.10 - 0.01).
        return int(np.ceil(seconds * fps - 1.0 - 1e-9))

    def mark_boundary(target, seconds, string, attack=False):
        if not 0.0 <= seconds < duration:
            return
        center = seconds * fps - 1.0
        if attack:
            center = max(0.0, center)
        first = max(0, int(np.ceil(center - 3.0 - 1e-9)))
        last = min(count, int(np.floor(center + 3.0 + 1e-9)) + 1)
        if first >= last:
            return
        indices = np.arange(first, last)
        values = np.exp(-0.5 * (indices - center) ** 2).astype(np.float32)
        target[first:last, string] = np.maximum(target[first:last, string], values)

    for note in notes:
        start_time, end_time = float(note["start"]), float(note["end"])
        if not math.isfinite(start_time) or not math.isfinite(end_time):
            raise ValueError("Reference note boundaries must be finite")
        # Filter source times, before either occupancy indexing or Gaussian
        # boundary construction, so unheard attacks never become labels.
        if start_time >= duration or end_time <= 0:
            continue
        if end_time <= start_time:
            raise ValueError("Reference notes must have positive duration")
        string, fret, midi_pitch = note["string"], note["fret"], note["pitch"]
        if (not 0 <= string < 6 or not 0 <= fret < fret_count
                or not pitch_min <= midi_pitch <= pitch_max):
            raise ValueError(f"Note outside supported standard-tuning fretboard: {note}")
        start = min(count, max(0, grid_index(start_time)))
        end = min(count, max(0, grid_index(end_time)))
        occupancy[start:end, string, fret] = True
        pitch[start:end, midi_pitch - pitch_min] = 1
        mark_boundary(onsets, start_time, string, attack=True)
        mark_boundary(offsets, end_time, string)
    active = occupancy.any(axis=-1)
    states[active] = occupancy.argmax(axis=-1)[active]
    conflicts = occupancy.sum(axis=-1) > 1
    states[conflicts] = -100
    return {"state": states, "onset": onsets.astype(np.float16),
            "offset": offsets.astype(np.float16), "pitch": pitch}, int(conflicts.sum())
