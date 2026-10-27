import importlib.metadata
import json
from pathlib import Path

from baseline.download import ROOT, open_url, sha256

URL = 'https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/small0.ckpt'
CHECKSUM = '6074be2c4d490c5f6101fcc374a1ec72ae93456e23bb6019783b849f5dc7d47b'


def prepare():
    folder = ROOT / 'models/beat_this'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / 'small0.ckpt'
    if not target.exists():
        temporary = target.with_suffix('.download')
        with open_url(URL) as response, temporary.open('wb') as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        temporary.replace(target)
    if sha256(target) != CHECKSUM:
        raise ValueError('Official small0 checkpoint does not match the pinned SHA256.')
    import beat_this
    package = Path(beat_this.__file__).parent
    metadata = {'url': URL, 'sha256': sha256(target), 'bytes': target.stat().st_size,
                'packages': {name: importlib.metadata.version(name) for name in
                             ['beat-this', 'rotary-embedding-torch', 'torch', 'torchaudio']},
                'package_source_sha256': {str(path.relative_to(package)): sha256(path)
                                         for path in sorted(package.rglob('*.py'))},
                'license': 'MIT code and weights', 'training_overlap': 'GuitarSet is included in small0 training.',
                'sources': ['https://github.com/CPJKU/beat_this',
                            'https://github.com/CPJKU/beat_this_annotations/tree/main/guitarset']}
    metadata_path = folder / 'provenance.json'
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise ValueError('Beat checkpoint or installed source differs from recorded provenance.')
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Prepared small0:', metadata['bytes'], 'bytes; SHA256:', metadata['sha256'], flush=True)
