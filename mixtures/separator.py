import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np

from baseline.download import ROOT, sha256


def verify(config):
    path = ROOT/config['separator']['checkpoint']
    metadata = json.loads((path.parent/'provenance.json').read_text())
    if sha256(path) != metadata['sha256']: raise ValueError('Separator checkpoint changed.')
    import demucs
    package = Path(demucs.__file__).parent
    for name, digest in metadata['package_source_sha256'].items():
        if sha256(package/name) != digest: raise ValueError('Separator source changed.')
    for name, version in metadata['packages'].items():
        if importlib.metadata.version(name) != version: raise ValueError('Separator dependencies changed.')
    return path, metadata


class Separator:
    def __init__(self, config, device):
        import torch
        from demucs.states import load_model
        path, self.provenance = verify(config)
        # This official, checksum-verified legacy checkpoint contains model classes.
        # Keep permissive loading local to this trusted file; do not change torch globally.
        package = torch.load(path, map_location='cpu', weights_only=False)
        self.model = load_model(package, strict=True).eval(); del package
        if 'guitar' not in self.model.sources or len(self.model.sources) != 6:
            raise ValueError('Checkpoint is not the expected six-source guitar model.')
        self.guitar_index = self.model.sources.index('guitar')
        self.config = config; self.device = torch.device(device); self.torch = torch
        self.peak_memory = 0
        self.parameters = sum(p.numel() for p in self.model.parameters())
        self.original_segment = self.model.segment

    def separate(self, audio, sample_rate):
        import librosa
        from demucs.apply import apply_model
        torch = self.torch
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device); torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1: stereo = np.stack([audio, audio])
        elif audio.ndim == 2 and audio.shape[1] == 2: stereo = audio.T
        else: raise ValueError('Expected mono or samples-by-two-channel waveform.')
        if not np.isfinite(stereo).all() or stereo.shape[1] < 2: raise ValueError('Input waveform needs at least two finite samples.')
        length = stereo.shape[1]
        if sample_rate != self.model.samplerate:
            stereo = librosa.resample(stereo, orig_sr=sample_rate, target_sr=self.model.samplerate, res_type='soxr_hq')
        tensor = torch.from_numpy(np.ascontiguousarray(stereo))
        reference = tensor.mean(0)
        mean, std = reference.mean(), reference.std()
        if std < 1e-8:
            return np.zeros(length, np.float32), {'separation_seconds':time.perf_counter()-started,
                                               'silent_input':True, 'stem':'guitar', 'sample_rate':sample_rate}
        specification = self.config['separator']
        # apply_model moves one chunk at a time to CUDA; input/output accumulation stays on CPU.
        normalized = (tensor-mean)/std
        with torch.inference_mode():
            stems = apply_model(self.model, normalized[None], device=self.device,
                                shifts=specification['shifts'], split=specification['split'],
                                overlap=specification['overlap'], segment=specification['segment'],
                                num_workers=specification['num_workers'], progress=False)[0]
        guitar = (stems[self.guitar_index]*std+mean).mean(0).cpu().numpy(); del stems, normalized
        if sample_rate != self.model.samplerate:
            guitar = librosa.resample(guitar, orig_sr=self.model.samplerate, target_sr=sample_rate, res_type='soxr_hq')
        guitar = np.pad(guitar[:length], (0,max(0,length-len(guitar)))).astype(np.float32)
        if not np.isfinite(guitar).all(): raise ValueError('Separator produced nonfinite samples.')
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device); self.peak_memory = max(self.peak_memory,torch.cuda.max_memory_allocated(self.device))
        return guitar, {'separation_seconds':time.perf_counter()-started, 'silent_input':False,
                        'stem':'guitar', 'stem_index':self.guitar_index, 'sources':list(self.model.sources),
                        'model_sample_rate':self.model.samplerate, 'model_segment_seconds':float(self.original_segment),
                        'sample_rate':sample_rate, 'original_sample_count':length, 'output_sample_count':len(guitar),
                        'normalization_mean':float(mean),'normalization_std':float(std),
                        'input_channels':1 if audio.ndim==1 else 2,
                        'timing_scope':'Resampling, input normalization, split model, stem restore/mono average and resampling back; disk IO and loading excluded.'}
