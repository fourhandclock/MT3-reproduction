"""Right-edge-aligned 128/128 or 128/256 ms features, without future PCM."""

import copy
import numpy as np
from streaming.frontend import CausalFrontend


class MultiResolutionFrontend:
    def __init__(self, config):
        self.windows = tuple(config['feature_planes'])
        if self.windows not in [(2048, 2048), (2048, 4096)]:
            raise ValueError('Only short/short and short/long planes are supported')
        self.frontends = {}
        for window in dict.fromkeys(self.windows):
            item = copy.deepcopy(config)
            item['feature'].update(window_length=window, n_fft=window)
            self.frontends[window] = CausalFrontend(item)
        self.n_bins = int(config['feature']['n_bins'])

    @property
    def received_samples(self):
        return next(iter(self.frontends.values())).received_samples

    @property
    def cached_samples(self):
        return sum(frontend.cached_samples for frontend in self.frontends.values())

    @property
    def last_frame_sample(self):
        return next(iter(self.frontends.values())).last_frame_sample

    def reset(self):
        for frontend in self.frontends.values():
            frontend.reset()

    def push(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError('Expected finite mono PCM')
        blocks = {window: frontend.push(audio) for window, frontend in self.frontends.items()}
        if len({block.shape for block in blocks.values()}) != 1:
            raise ValueError('Spectral planes lost their common frame grid')
        return np.stack([blocks[window] for window in self.windows], axis=1)
