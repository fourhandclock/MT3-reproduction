"""Past-only log-frequency STFT features for 16 kHz mono PCM.

The timestamp of frame zero is one hop, rather than zero: each frame uses
the samples immediately before its right edge. There is no end padding or
resampling in this streaming contract.
"""

import numpy as np


class CausalFrontend:
    def __init__(self, config):
        specification = config.get("feature", config)
        self.sample_rate = int(specification["sample_rate"])
        self.hop_length = int(specification["hop_length"])
        self.window_length = int(specification.get("window_length", 2048))
        self.n_fft = int(specification.get("n_fft", self.window_length))
        self.n_bins = int(specification["n_bins"])
        self.log_floor = float(specification["log_floor"])
        if not 0 < self.hop_length <= self.window_length <= self.n_fft:
            raise ValueError("Require 0 < hop_length <= window_length <= n_fft")
        if self.sample_rate != 16000:
            raise ValueError("The streaming input contract is 16 kHz mono PCM")
        if self.n_bins < 1 or self.log_floor <= 0:
            raise ValueError("Feature bin count and log floor must be positive")
        minimum = 440.0 * 2.0 ** ((float(specification["fmin_midi"]) - 69.0) / 12.0)
        frequencies = minimum * 2.0 ** (
            np.arange(self.n_bins) / float(specification["bins_per_octave"])
        )
        if frequencies[-1] > self.sample_rate / 2:
            raise ValueError("Feature frequencies exceed the Nyquist frequency")
        positions = frequencies * self.n_fft / self.sample_rate
        self._lower = np.floor(positions).astype(np.int64)
        self._upper = np.minimum(self._lower + 1, self.n_fft // 2)
        self._fraction = positions - self._lower
        # Periodic Hann, matching the common FFT analysis-window convention.
        self._window = np.hanning(self.window_length + 1)[:-1].astype(np.float32)
        self.reset()

    def reset(self):
        self.received_samples = 0
        self.frames_emitted = 0
        self._buffer_start = 0
        self._buffer = np.empty(0, dtype=np.float32)

    @property
    def cached_samples(self):
        return len(self._buffer)

    @property
    def last_frame_sample(self):
        return self.frames_emitted * self.hop_length

    def push(self, audio):
        samples = np.asarray(audio, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError("Expected one-dimensional mono PCM samples")
        if not np.isfinite(samples).all():
            raise ValueError("Audio contains non-finite samples")
        if len(samples):
            self._buffer = np.concatenate((self._buffer, samples))
            self.received_samples += len(samples)
        target_frames = self.received_samples // self.hop_length
        count = target_frames - self.frames_emitted
        result = np.empty((count, self.n_bins), dtype=np.float32)
        for index in range(count):
            right = (self.frames_emitted + index + 1) * self.hop_length
            left = right - self.window_length
            window = np.zeros(self.window_length, dtype=np.float32)
            available_left = max(0, left)
            source_start = available_left - self._buffer_start
            source_end = right - self._buffer_start
            destination_start = available_left - left
            window[destination_start:] = self._buffer[source_start:source_end]
            magnitude = np.abs(np.fft.rfft(window * self._window, n=self.n_fft))
            interpolated = (
                magnitude[self._lower] * (1.0 - self._fraction)
                + magnitude[self._upper] * self._fraction
            )
            result[index] = np.log(np.maximum(interpolated, self.log_floor))
        self.frames_emitted = target_frames
        # Retain only the samples required by the next complete frame. The
        # steady-state cache is strictly smaller than one analysis window.
        next_left = max(0, (target_frames + 1) * self.hop_length - self.window_length)
        trim = next_left - self._buffer_start
        if trim > 0:
            self._buffer = self._buffer[trim:].copy()
            self._buffer_start = next_left
        return result


def features(audio, config):
    """Return exactly floor(len(audio) / hop_length) causal feature frames."""
    return CausalFrontend(config).push(audio)
