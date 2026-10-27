import importlib.metadata
import json
import time

import numpy as np
import soundfile as sf

from baseline.download import ROOT, sha256


def verify(config):
    checkpoint = ROOT / config['beat_checkpoint']
    provenance = json.loads((checkpoint.parent/'provenance.json').read_text())
    if sha256(checkpoint) != provenance['sha256']: raise ValueError('Beat checkpoint checksum mismatch.')
    import beat_this
    from pathlib import Path
    package = Path(beat_this.__file__).parent
    for name, digest in provenance['package_source_sha256'].items():
        if sha256(package/name) != digest: raise ValueError('Installed beat model source changed.')
    for name, version in provenance['packages'].items():
        if importlib.metadata.version(name) != version: raise ValueError('Beat model package versions changed.')
    return checkpoint, provenance


class Tracker:
    def __init__(self, config, device):
        import torch
        from beat_this.inference import Audio2Frames
        from beat_this.model.postprocessor import Postprocessor
        path, self.provenance = verify(config)
        self.model = Audio2Frames(str(path), device=device, float16=False)
        self.postprocessor = Postprocessor(type='minimal')
        self.device = device
        self.parameters = sum(p.numel() for p in self.model.model.parameters())
        self.peak_memory = 0
        self.torch = torch

    def predict(self, audio_path, seconds):
        info = sf.info(audio_path)
        signal, sr = sf.read(audio_path, frames=int(seconds*info.samplerate), dtype='float32')
        if str(self.device).startswith('cuda'):
            self.torch.cuda.synchronize(); self.torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        beat_logits, downbeat_logits = self.model(signal, sr)
        beats, downbeats = self.postprocessor(beat_logits, downbeat_logits)
        if str(self.device).startswith('cuda'):
            self.torch.cuda.synchronize()
            self.peak_memory = max(self.peak_memory, self.torch.cuda.max_memory_allocated())
        elapsed = time.perf_counter()-start
        return {'beats': beats.tolist(), 'downbeats': downbeats.tolist(),
                'duration_seconds': len(signal)/sr, 'processing_seconds': elapsed,
                'timing_scope': 'Loaded waveform through frontend, model and minimal postprocessing; excludes disk read.',
                'beat_probability': beat_logits.sigmoid().cpu().numpy(),
                'downbeat_probability': downbeat_logits.sigmoid().cpu().numpy()}
