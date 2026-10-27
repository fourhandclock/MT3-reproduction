import json

from baseline.download import ROOT,sha256
from cnn_gru.data import fingerprint,validate_splits


SHARED_FILES=['data.py','model.py','inference.py','training.py']


def common_config(config):
    return {key:value for key,value in config.items() if key!='transformer'}


def load_controlled_data(config):
    reference=json.loads((ROOT/'configs/cnn_gru.json').read_text())
    if common_config(config)!=reference:
        raise ValueError('Matched comparison requires unchanged GRU data, feature, model-head, training and decoder settings.')
    manifest=json.loads((ROOT/'data/cnn_gru/manifest.json').read_text())
    audit=json.loads((ROOT/'data/cnn_gru/cache_audit.json').read_text())
    if manifest['config']!=reference or audit['config_sha256']!=fingerprint(reference):
        raise ValueError('Prepared dataset differs from the reference configuration.')
    validate_splits(manifest)
    return manifest,audit


def source_hashes():
    return {'shared/'+name:sha256(ROOT/'cnn_gru'/name) for name in SHARED_FILES} | {
            'transformer/'+name:sha256(ROOT/'cnn_transformer'/name) for name in ['model.py','training.py','control.py']}


def verify_sources(recorded):
    if source_hashes()!=recorded:
        raise ValueError('Source changed since the transformer checkpoint was created.')
