"""Run with .venv-tf-check Python, not the normal inference environment.

Downloads the official MT3 spectral_ops at a recorded Git revision and executes
its compute_logmel on fixed signals to create numerical reference fixtures.
"""

import importlib.util
import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.download import open_url
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)
source_dir = ROOT / '.cache/reference'
source_dir.mkdir(parents=True, exist_ok=True)
recorded_source = ROOT / 'tests/fixtures/mt3_frontend.json'
if recorded_source.exists():
    commit = json.loads(recorded_source.read_text())['official_source_url'].split('/')[5]
else:
    request = urllib.request.Request('https://api.github.com/repos/magenta/mt3/commits/main', headers={'User-Agent': 'guitar-baseline'})
    with open_url(request) as response:
        commit = json.load(response)['sha']
url = f'https://raw.githubusercontent.com/magenta/mt3/{commit}/mt3/spectral_ops.py'
request = urllib.request.Request(f'https://api.github.com/repos/magenta/mt3/contents/mt3/spectral_ops.py?ref={commit}',
                                 headers={'User-Agent': 'guitar-baseline'})
with open_url(request) as response:
    body = base64.b64decode(json.load(response)['content'])
source = source_dir / 'spectral_ops.py'
source.write_bytes(body)
spec = importlib.util.spec_from_file_location('reference_mt3_spectral_ops', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
existing_fixture = ROOT / 'tests/fixtures/mt3_frontend.npz'
if existing_fixture.exists():
    with np.load(existing_fixture) as saved:
        guitar = saved['guitar_audio'].copy()
else:
    guitar = np.load(ROOT / 'tmp/frontend-guitar.npy')
signals = {'silence': np.zeros(512, dtype=np.float32),
           'noise': np.random.default_rng(42).normal(0, 0.1, 4096).astype(np.float32),
           'impulse': np.zeros(4096, dtype=np.float32),
           'guitar': guitar}
signals['impulse'][1024] = 0.8
fixtures = {}
for name, signal in signals.items():
    fixtures[name + '_audio'] = signal
    fixtures[name + '_logmel'] = module.compute_logmel(signal, lo_hz=20.0,
                                                      bins=512, fft_size=2048, overlap=0.9375,
                                                      pad_end=True, sample_rate=16000).numpy()
    print(name, fixtures[name + '_logmel'].shape, flush=True)
output = ROOT / 'tests/fixtures'
output.mkdir(parents=True, exist_ok=True)
np.savez_compressed(output / 'mt3_frontend.npz', **fixtures)
import hashlib
provenance = {'official_source_url': url, 'official_source_sha256': hashlib.sha256(body).hexdigest(),
              'tensorflow': tf.__version__, 'numpy': np.__version__, 'cpu_threads': 4,
              'configuration': {'sample_rate': 16000, 'fft_size': 2048, 'hop': 128, 'mel_bins': 512, 'lower_hz': 20, 'upper_hz': 7600},
              'guitar_source': 'first 4096 resampled samples of GuitarSet 00_Funk3-98-A_solo microphone recording'}
(output / 'mt3_frontend.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
print('Saved numerical fixtures:', output, flush=True)
