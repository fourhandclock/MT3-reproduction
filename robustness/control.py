import json

from baseline.download import ROOT,sha256
from .data import fingerprint,inputs


def source_hashes():
    return {'shared/'+name:sha256(ROOT/'cnn_gru'/name) for name in ['model.py','data.py','inference.py','training.py']} | {
        'robustness/'+name:sha256(ROOT/'robustness'/name) for name in ['effects.py','data.py','validation.py','training.py','control.py']}


def validate_prepared(config):
    manifest,audit=inputs(config)
    effects=json.loads((ROOT/'data/robustness/cache_audit.json').read_text())
    if effects['config_sha256']!=fingerprint(config): raise ValueError('Effect cache configuration differs.')
    if effects['base_manifest_sha256']!=sha256(ROOT/'data/cnn_gru/manifest.json'): raise ValueError('Base manifest changed.')
    return manifest,audit
