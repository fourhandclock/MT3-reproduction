"""Past-only state smoothing ablation for the immutable stage-8 decoder.

This changes probabilities, confirmation lengths and source backdating without
changing any acoustic weights. A fresh instance resets both EMA and decoding
history. The first observation initializes the EMA; subsequent observations are
convex averages. Onset, offset and pitch heads pass through unchanged.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from streaming.decoder import OnlineDecoder


class SmoothedDecoder(OnlineDecoder):
    """A bounded six-string EMA followed by the original append-only decoder.

    ``state_alpha=1`` delegates directly to the original decoder with unmodified
    probability values, guaranteeing the identity ablation. Block validation is
    performed before changing smoothing or note history, including clock checks.
    """

    def __init__(self, config, parameters):
        local_config = copy.deepcopy(config)
        local_parameters = copy.deepcopy(parameters)
        local_config['streaming']['confirm_frames'] = local_parameters.get('confirm_frames', 4)
        local_config['streaming']['release_confirm_frames'] = local_parameters.get('release_confirm_frames', 2)
        self.state_alpha = float(local_parameters.get('state_alpha', 1.0))
        if not math.isfinite(self.state_alpha) or not 0 < self.state_alpha <= 1:
            raise ValueError('State smoothing alpha must be finite and in (0, 1].')
        self._ema = None
        super().__init__(local_config, local_parameters)

    @property
    def smoothed_state(self):
        """Copy of the latest six-string EMA, or None before any frame arrives."""
        return None if self._ema is None else self._ema.copy()

    def _validate_availability(self, length, available_at_audio_seconds):
        # Match the immutable parent's clock rule for the entire input block.
        # Checking only one frame at a time could update EMA before rejecting
        # availability earlier than the final frame of that same block.
        tolerance = 1e-9
        block_end = (self._framecount + length) * self.frame_period
        if available_at_audio_seconds is not None:
            available = float(available_at_audio_seconds)
            if not math.isfinite(available) or available + tolerance < max(block_end, self._last_available):
                raise ValueError('Block availability must follow all its frames and prior availability.')
        elif (self._framecount + 1) * self.frame_period + tolerance < self._last_available:
            raise ValueError('Frame-by-frame arrival cannot precede prior block availability.')

    def push(self, probabilities, available_at_audio_seconds=None):
        if self._finished:
            raise RuntimeError('Cannot push probabilities after finish.')
        arrays, length = self._validate_probabilities(probabilities)
        if not length:
            return []
        self._validate_availability(length, available_at_audio_seconds)
        if self.state_alpha == 1:
            # The original decoder already consumes every frame in order. This
            # path keeps the original dtype and every probability value intact.
            events = super().push(arrays, available_at_audio_seconds)
            self._ema = arrays['state'][-1].astype(np.float32, copy=True)
            return events
        events = []
        alpha = np.float32(self.state_alpha)
        previous_weight = np.float32(1 - self.state_alpha)
        for frame in range(length):
            current = arrays['state'][frame].astype(np.float32, copy=False)
            if self._ema is None:
                self._ema = current.copy()
            else:
                self._ema = alpha * current + previous_weight * self._ema
                # Preserve the probability validation contract under rounding.
                np.clip(self._ema, 0, 1, out=self._ema)
            evidence = {key: value[frame:frame + 1] for key, value in arrays.items()}
            evidence['state'] = self._ema[None]
            events.extend(super().push(evidence, available_at_audio_seconds))
        return events
