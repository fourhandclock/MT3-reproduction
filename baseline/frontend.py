"""Magnitude/log-mel frontend following MT3's TensorFlow spectral_ops.

The frozen third-party implementation is left untouched. This wrapper changes
only its acoustic frontend, retaining the port's model and token decoder.
"""

import numpy as np
from scipy.fft import rfft


def tensorflow_mel_weights(bins=512, fft_size=2048, sample_rate=16000, lower_hz=20.0, upper_hz=7600.0):
    def to_mel(hz):
        hz = np.asarray(hz, dtype=np.float32)
        return np.float32(1127.0) * np.log1p(hz / np.float32(700.0))
    # TensorFlow excludes DC, and constructs triangle slopes in mel coordinates.
    frequencies = np.linspace(0, sample_rate / 2, fft_size // 2 + 1, dtype=np.float32)
    mel_frequencies = to_mel(frequencies[1:])[:, None]
    edges = np.linspace(to_mel(lower_hz), to_mel(upper_hz), bins + 2, dtype=np.float32)
    lower, center, upper = edges[:-2], edges[1:-1], edges[2:]
    rising = (mel_frequencies - lower) / (center - lower)
    falling = (upper - mel_frequencies) / (upper - center)
    weights = np.maximum(np.float32(0.0), np.minimum(rising, falling))
    return np.pad(weights, ((1, 0), (0, 0))).astype(np.float32)


def compute_logmel(samples, weights=None):
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    fft_size, hop = 2048, 128
    frame_count = int(np.ceil(len(samples) / hop))
    if frame_count == 0:
        raise ValueError('Cannot compute features for empty audio.')
    total_length = (frame_count - 1) * hop + fft_size
    padded = np.pad(samples, (0, total_length - len(samples)))
    frames = np.lib.stride_tricks.sliding_window_view(padded, fft_size)[::hop]
    phase = np.arange(fft_size, dtype=np.float32) * np.float32(2 * np.pi / fft_size)
    window = np.float32(0.5) - np.float32(0.5) * np.cos(phase)
    magnitude = np.abs(rfft(frames * window, axis=-1, workers=1))
    mel = magnitude @ (tensorflow_mel_weights() if weights is None else weights)
    # MT3 replaces non-positive values; it does not add epsilon to every bin.
    return np.log(np.where(mel <= 0, np.float32(1e-5), mel)).astype(np.float32)


class ReferenceSpectrogramProcessor:
    def __init__(self, wrapped):
        self.wrapped, self.config = wrapped, wrapped.config
        self.weights = tensorflow_mel_weights()

    def split_audio(self, samples):
        return self.wrapped.split_audio(samples)

    def flatten_frames(self, frames):
        return self.wrapped.flatten_frames(frames)

    def compute_spectrogram(self, samples):
        if hasattr(samples, 'detach'):
            samples = samples.detach().cpu().numpy()
        return compute_logmel(samples, self.weights)
