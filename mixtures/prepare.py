import importlib.metadata
import json
from pathlib import Path

from baseline.download import ROOT, open_url, sha256

URL = 'https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th'
AUTHOR_HASH_PREFIX = '34c22ccb'
CHECKSUM = '34c22ccb381c6f9fdbf324f04e1e2fe21aaaf293f5ded163a162697ff9a02ddd'
PACKAGES = ['demucs', 'dora-search', 'julius', 'lameenc', 'omegaconf', 'openunmix',
            'retrying', 'submitit', 'treetable', 'torch', 'torchaudio', 'einops', 'numpy']


def prepare(config):
    if importlib.metadata.version('demucs') != '4.0.1': raise ValueError('Expected pinned Demucs 4.0.1.')
    target = ROOT/config['separator']['checkpoint']; target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix('.download')
        with open_url(URL) as response, temporary.open('wb') as stream:
            while chunk := response.read(1024*1024): stream.write(chunk)
        if sha256(temporary) != CHECKSUM:
            raise ValueError('Official Demucs file does not match the authors\' published hash prefix.')
        temporary.replace(target)
    if sha256(target) != CHECKSUM: raise ValueError('Demucs checkpoint checksum mismatch.')
    import demucs
    package = Path(demucs.__file__).parent
    metadata = {'name':'htdemucs_6s', 'url':URL, 'sha256':sha256(target), 'bytes':target.stat().st_size,
                'authors_published_sha256_prefix':AUTHOR_HASH_PREFIX,
                'packages':{name:importlib.metadata.version(name) for name in PACKAGES},
                'package_source_sha256':{str(p.relative_to(package)):sha256(p) for p in sorted(package.rglob('*.py'))},
                'source_version':'v4.0.1', 'upstream_commit':'ef66d254cd6d558e207eeff2c4b8d053db2e77dd',
                'code_license':'MIT', 'sources':['https://github.com/facebookresearch/demucs/tree/v4.0.1',
                                              'https://github.com/facebookresearch/demucs/blob/main/docs/release.md'],
                'scope':'Experimental six-source model. Guitar quality is limited; training overlap with these exact inputs has not been independently audited.'}
    path = target.parent/'provenance.json'
    if path.exists() and json.loads(path.read_text()) != metadata:
        raise ValueError('Recorded separator weights, source or dependencies changed.')
    path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Demucs prepared:', metadata['bytes']/1e6, 'MB;', metadata['sha256'], flush=True)
