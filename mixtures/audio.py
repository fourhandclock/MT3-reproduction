"""Deterministic diagnostic accompaniment and level-controlled mono mixtures.

Beat/chord annotations are used only to render the synthetic backing track.
These waveforms are simple additive synthesizers, not recordings of instruments.
No original guitar note annotations are consumed or changed by this module.
"""
from __future__ import annotations

import math
import re

import numpy as np
from scipy.signal import butter, sosfilt


RENDERER_VERSION = "additive-backing-v1"
_ROOTS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_QUALITIES = {
    "maj": (0, 4, 7), "min": (0, 3, 7), "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11), "min7": (0, 3, 7, 10),
    "dim": (0, 3, 6), "dim7": (0, 3, 6, 9), "hdim7": (0, 3, 6, 10),
    "aug": (0, 4, 8), "sus2": (0, 2, 7), "sus4": (0, 5, 7),
}


def _rms(audio):
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def _chord_pitches(label):
    """Return root and a basic voicing; slash bass and added tensions are ignored."""
    if label in {"N", "X", ""}:
        return None
    match = re.fullmatch(r"([A-G])([#b]*):([^/]+)(?:/.*)?", label)
    if match is None:
        return None
    letter, accidental, quality = match.groups()
    # The renderer deliberately uses a documented basic quality. The original
    # label is retained in metadata, including inversions and added tensions.
    quality = quality.split("(", 1)[0]
    if quality not in _QUALITIES:
        return None
    root = (_ROOTS[letter] + accidental.count("#") - accidental.count("b")) % 12
    return root, [60 + root + interval for interval in _QUALITIES[quality]]


def _schedule_inputs(beats, chords, duration):
    pulses = []
    for index, beat in enumerate(beats):
        if isinstance(beat, dict):
            time = float(beat.get("time", beat.get("start", math.nan)))
            value = beat.get("value", beat)
            position = int(value.get("position", index % 4 + 1)) if isinstance(value, dict) else index % 4 + 1
        else:
            time, position = float(beat), index % 4 + 1
        if not math.isfinite(time):
            raise ValueError("Beat times must be finite.")
        if 0 <= time < duration:
            pulses.append({"time": time, "position": position})
    pulses.sort(key=lambda value: value["time"])
    if any(a["time"] == b["time"] for a, b in zip(pulses, pulses[1:])):
        raise ValueError("Beat times must be distinct.")

    segments = []
    unsupported = []
    for chord in chords:
        start = float(chord.get("start", chord.get("time", math.nan)))
        end = float(chord["end"]) if "end" in chord else start + float(chord.get("duration", math.nan))
        label = str(chord.get("label", chord.get("value", "")))
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError("Chord intervals must be finite and ordered.")
        start, end = max(0.0, start), min(duration, end)
        if end <= start:
            continue
        parsed = _chord_pitches(label)
        if parsed is None and label not in {"N", "X", ""}:
            unsupported.append(label)
        segments.append({"start": start, "end": end, "label": label, "parsed": parsed})
    segments.sort(key=lambda value: value["start"])
    if any(a["end"] > b["start"] + 1e-9 for a, b in zip(segments, segments[1:])):
        raise ValueError("Accompaniment requires non-overlapping chord segments.")
    return pulses, segments, sorted(set(unsupported))


def _envelope(length, sample_rate, attack, release, decay=0.0):
    time = np.arange(length, dtype=np.float64) / sample_rate
    remaining = (length - 1 - np.arange(length)) / sample_rate
    return np.minimum(1, time / attack) * np.minimum(1, np.maximum(0, remaining) / release) * np.exp(-decay * time)


def _pitched(length, sample_rate, pitch, instrument, rng):
    time = np.arange(length, dtype=np.float64) / sample_rate
    frequency = 440 * 2 ** ((pitch - 69) / 12)
    parameters = {
        "bass": ([1, .30, .12], .008, .055, 1.8),
        "keyboard": ([1, .35, .18, .09, .05], .006, .06, 3.5),
        "pad": ([1, .22, .10, .04], .05, .10, .12),
    }
    harmonics, attack, release, decay = parameters[instrument]
    waveform = np.zeros(length, dtype=np.float64)
    phase = rng.uniform(0, 2 * np.pi)
    for harmonic, level in enumerate(harmonics, 1):
        if frequency * harmonic >= .45 * sample_rate:
            continue
        waveform += level * np.sin(2 * np.pi * frequency * harmonic * time + harmonic * phase)
    waveform /= sum(harmonics)
    return waveform * _envelope(length, sample_rate, attack, release, decay)


def _percussion(length, sample_rate, instrument, rng):
    time = np.arange(length, dtype=np.float64) / sample_rate
    noise = rng.standard_normal(length)
    if instrument == "kick":
        # Integral of a falling frequency gives a phase-continuous kick.
        phase = 2 * np.pi * (48 * time + 75 * .025 * (1 - np.exp(-time / .025)))
        sound = np.sin(phase) * np.exp(-time / .09) + .08 * noise * np.exp(-time / .008)
    elif instrument == "snare":
        sound = .72 * noise * np.exp(-time / .05) + .28 * np.sin(2 * np.pi * 180 * time) * np.exp(-time / .055)
    else:
        # The cutoff adapts to tiny sample rates used in tests.
        high_pass = butter(2, min(6000.0, sample_rate * .30), "highpass", fs=sample_rate, output="sos")
        sound = sosfilt(high_pass, noise) * np.exp(-time / .025)
    return sound * _envelope(length, sample_rate, .001, .01)


def make_accompaniment(duration, sr, beats, chords, seed):
    """Render bass, keyboard/pad, and drums from supplied lead-sheet events.

    ``beats`` accepts times or JAMS beat_position data dictionaries. ``chords``
    accepts {start, end, label} or JAMS {time, duration, value} dictionaries.
    All synthesized note/percussion events are returned as diagnostic labels.
    Basic chord quality is rendered; slash inversions and added tensions are
    retained in the input label but do not alter the synthetic voicing.
    """
    duration, sr = float(duration), int(sr)
    if not math.isfinite(duration) or duration <= 0 or sr <= 0:
        raise ValueError("Duration and sample rate must be positive.")
    length = int(round(duration * sr))
    if length < 1:
        raise ValueError("Duration must contain at least one sample.")
    actual_duration = length / sr
    pulses, segments, unsupported = _schedule_inputs(beats, chords, actual_duration)
    rng = np.random.default_rng(int(seed))
    track = np.zeros(length, dtype=np.float64)
    events = []

    def add(instrument, start, end, amplitude, pitch=None, chord=None):
        a, b = max(0, int(round(start * sr))), min(length, int(round(end * sr)))
        if b <= a:
            return
        sound = (_pitched(b-a, sr, pitch, instrument, rng) if pitch is not None
                 else _percussion(b-a, sr, instrument, rng))
        track[a:b] += amplitude * sound
        event = {"instrument": instrument, "start": a / sr, "end": b / sr,
                 "amplitude": float(amplitude)}
        if pitch is not None:
            event["pitch"] = int(pitch)
        if chord is not None:
            event["source_chord"] = chord
        events.append(event)

    # Pads fill only explicitly annotated chord spans, with no inferred chords.
    for chord in segments:
        if chord["parsed"] is None:
            continue
        _, pitches = chord["parsed"]
        for pitch in pitches:
            add("pad", chord["start"], chord["end"], .10 / len(pitches), pitch, chord["label"])
        attacks = [chord["start"]] + [beat["time"] for beat in pulses
                   if chord["start"] < beat["time"] < chord["end"] and beat["position"] in {1, 3}]
        for start in attacks:
            for pitch in pitches:
                add("keyboard", start, min(start + .42, chord["end"]), .22 / len(pitches), pitch, chord["label"])

    intervals = np.diff([beat["time"] for beat in pulses])
    fallback = float(np.median(intervals)) if len(intervals) else .5
    for index, beat in enumerate(pulses):
        start, position = beat["time"], beat["position"]
        interval = pulses[index+1]["time"] - start if index+1 < len(pulses) else fallback
        chord = next((value for value in segments if value["start"] <= start < value["end"]), None)
        if chord is not None and chord["parsed"] is not None:
            root, _ = chord["parsed"]
            bass_pitch = 36 + root
            if bass_pitch > 43:
                bass_pitch -= 12
            if position == 3:
                bass_pitch += 7
            add("bass", start, min(start + .8 * interval, chord["end"]), .28, bass_pitch, chord["label"])
        add("kick" if position % 2 else "snare", start, min(start + .20, actual_duration), .26)
        add("hihat", start, min(start + .08, actual_duration), .055)
        if start + interval / 2 < actual_duration:
            add("hihat", start + interval / 2, min(start + interval / 2 + .08, actual_duration), .035)

    waveform = track.astype(np.float32)
    metadata = {
        "renderer": RENDERER_VERSION, "seed": int(seed), "sample_rate": sr,
        "samples": length, "duration_seconds": actual_duration,
        "events": sorted(events, key=lambda event: (event["start"], event["instrument"], event.get("pitch", -1))),
        "unsupported_chord_labels": unsupported,
        "rms": _rms(waveform), "peak": float(np.max(np.abs(waveform))),
        "annotation_use": "Supplied beat/chord lead-sheet events render backing only; no guitar note annotations used.",
        "limitations": "Additive synthetic timbres; basic qualities only; inversions and added tensions ignored. Events are diagnostic synthetic labels, not recorded instrument annotations.",
    }
    return waveform, metadata


def mix(guitar, accompaniment, guitar_to_accompaniment_db):
    """Set the RMS source ratio, then apply one common peak-limiting scalar.

    The returned target is exactly the original guitar multiplied by one scalar.
    No shift, stretch, filter, source-dependent limiter, or clipping is applied.
    """
    guitar = np.asarray(guitar, dtype=np.float32)
    accompaniment = np.asarray(accompaniment, dtype=np.float32)
    ratio_db = float(guitar_to_accompaniment_db)
    if guitar.ndim != 1 or accompaniment.ndim != 1 or guitar.shape != accompaniment.shape or guitar.size == 0:
        raise ValueError("Sources must be non-empty, equal-length mono arrays.")
    if not np.isfinite(guitar).all() or not np.isfinite(accompaniment).all() or not math.isfinite(ratio_db):
        raise ValueError("Sources and requested RMS ratio must be finite.")
    guitar_rms, backing_rms = _rms(guitar), _rms(accompaniment)
    if guitar_rms == 0 or backing_rms == 0:
        raise ValueError("Both sources need nonzero RMS to set a finite source ratio.")
    try:
        ratio = 10 ** (ratio_db / 20)
    except OverflowError as error:
        raise ValueError("Requested source ratio is outside the supported numeric range.") from error
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("Requested source ratio is outside the supported numeric range.")
    backing_gain = guitar_rms / (backing_rms * ratio)
    raw_mix = guitar.astype(np.float64) + backing_gain * accompaniment.astype(np.float64)
    raw_peak = float(np.max(np.abs(raw_mix)))
    common_gain = min(1.0, .98 / raw_peak) if raw_peak else 1.0
    target = (guitar.astype(np.float64) * common_gain).astype(np.float32)
    backing = (accompaniment.astype(np.float64) * backing_gain * common_gain).astype(np.float32)
    mixture = target + backing
    achieved = 20 * math.log10(_rms(target) / _rms(backing))
    metadata = {
        "requested_guitar_to_accompaniment_db": ratio_db,
        "achieved_guitar_to_accompaniment_db": achieved,
        "input_guitar_rms": guitar_rms, "input_accompaniment_rms": backing_rms,
        "target_rms": _rms(target), "accompaniment_rms": _rms(backing),
        "guitar_gain": common_gain, "accompaniment_gain": backing_gain * common_gain,
        "accompaniment_gain_before_peak_limit": backing_gain,
        "common_peak_gain": common_gain, "raw_mixture_peak": raw_peak,
        "mixture_peak": float(np.max(np.abs(mixture))), "peak_limit": .98,
        "samples": int(guitar.size),
        "alignment": "Original guitar multiplied by a scalar only; no time shift or length change.",
    }
    return {"mixture": mixture, "target": target, "accompaniment": backing, "metadata": metadata}
