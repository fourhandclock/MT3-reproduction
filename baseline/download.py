"""Download only selected ZIP members; ZIP CRC and local SHA256 are checked."""

import hashlib
import io
import json
import re
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD_URL = 'https://zenodo.org/api/records/3371780'
MT3_SHA256 = 'b8a3807ed265059abd25ad7f68142c06c35e8f6144dcaa45bd55946a3745398f'
MT3_WEIGHTS_URL = 'https://huggingface.co/gudgud1014/MR-MT3/resolve/main/mt3.pth'
MT3_CONFIG_URL = 'https://raw.githubusercontent.com/kunato/mt3-pytorch/master/pretrained/config.json'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def open_url(request):
    for attempt in range(4):
        try:
            return urllib.request.urlopen(request, timeout=60)
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
            time.sleep(min(2 ** attempt, 4))
        except urllib.error.URLError:
            if attempt == 3:
                raise
            time.sleep(min(2 ** attempt, 4))


class RangeReader(io.RawIOBase):
    """Seekable HTTP reader. Fails explicitly if partial requests are ignored."""

    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0
        self.cache_start, self.cache = 0, b''
        self.downloaded_bytes = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if position < 0:
            raise ValueError('Cannot seek before the start of the archive.')
        self.position = position
        return position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b''
        if size > 64 * 1024 * 1024:
            raise ValueError('Refusing an unexpectedly large archive read.')
        if not (self.cache_start <= self.position and self.position + size <= self.cache_start + len(self.cache)):
            start = self.position
            end = min(self.size - 1, start + max(size, 256 * 1024) - 1)
            request = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{end}'})
            with open_url(request) as response:
                expected = f'bytes {start}-{end}/{self.size}'
                if response.status != 206 or response.headers.get('Content-Range') != expected:
                    raise RuntimeError('Server did not honor the requested byte range; no full archive was downloaded.')
                self.cache = response.read()
            if len(self.cache) != end - start + 1:
                raise IOError('Incomplete range download.')
            self.cache_start = start
            self.downloaded_bytes += len(self.cache)
        offset = self.position - self.cache_start
        self.position += size
        return self.cache[offset:offset + size]


def load_record():
    target = ROOT / 'data/downloads/guitarset-zenodo.json'
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with open_url(RECORD_URL) as response:
            target.write_bytes(response.read())
    return json.loads(target.read_text(encoding='utf-8'))


def select_ids(names, count, progression, seed):
    import random
    pattern = re.compile(r'^(\d\d)_([A-Za-z]+)([123])-.*_(comp|solo)$')
    groups = {}
    for name in sorted(names):
        clip_id = Path(name).stem
        match = pattern.match(clip_id)
        if match and int(match.group(3)) == progression:
            groups.setdefault((match.group(1), match.group(4)), []).append(clip_id)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected, used_styles = [], set()
    while len(selected) < count:
        changed = False
        for key in sorted(groups):
            group = groups[key]
            if not group or len(selected) >= count:
                continue
            index = next((i for i, value in enumerate(group) if pattern.match(value).group(2) not in used_styles), 0)
            clip_id = group.pop(index)
            selected.append(clip_id)
            used_styles.add(pattern.match(clip_id).group(2))
            changed = True
        if not changed:
            break
    if len(selected) != count:
        raise ValueError(f'Found {len(selected)} usable excerpts, expected {count}.')
    return selected


def prepare_guitarset(config):
    metadata = load_record()
    files = {item['key']: item for item in metadata['files']}
    annotation_info, audio_info = files['annotation.zip'], files['audio_mono-mic.zip']
    readers = [RangeReader(item['links']['self'], item['size']) for item in (annotation_info, audio_info)]
    manifest = {'dataset': 'GuitarSet', 'record_url': RECORD_URL, 'license': metadata['metadata'].get('license'),
                'selection': config, 'archive_checksums': {key: files[key]['checksum'] for key in ('annotation.zip', 'audio_mono-mic.zip')},
                'integrity': 'Selected members validated by ZIP CRC32; SHA256 recorded locally. Full-archive MD5 not checked for partial downloads.',
                'excerpts': []}
    with zipfile.ZipFile(readers[0]) as annotations, zipfile.ZipFile(readers[1]) as audio:
        annotation_names = {Path(name).stem: name for name in annotations.namelist() if name.endswith('.jams') and '__MACOSX' not in name}
        audio_names = {Path(name).name: name for name in audio.namelist() if name.endswith('.wav') and '__MACOSX' not in name}
        selected = select_ids(annotation_names.values(), config['excerpt_count'], config['progression'], config['seed'])
        for clip_id in selected:
            entry = {'id': clip_id}
            for archive, member, folder, field in (
                (annotations, annotation_names[clip_id], 'annotations', 'annotation'),
                (audio, audio_names[clip_id + '_mic.wav'], 'audio', 'audio'),
            ):
                target = ROOT / 'data/guitarset' / folder / Path(member).name
                target.parent.mkdir(parents=True, exist_ok=True)
                crc = archive.getinfo(member).CRC
                if target.exists():
                    import zlib
                    if zlib.crc32(target.read_bytes()) & 0xffffffff != crc:
                        raise IOError(f'Existing file has wrong CRC: {target}')
                else:
                    target.write_bytes(archive.read(member))
                entry[field] = target.relative_to(ROOT).as_posix()
                entry[field + '_sha256'] = sha256(target)
            manifest['excerpts'].append(entry)
            print('Prepared', clip_id, flush=True)
    manifest['downloaded_bytes_this_run'] = sum(reader.downloaded_bytes for reader in readers)
    output = ROOT / 'data/manifest.json'
    output.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Manifest: {output}; downloaded {manifest["downloaded_bytes_this_run"] / 1024**2:.1f} MiB', flush=True)


def prepare_mt3():
    destination = ROOT / 'models/mt3_pytorch'
    destination.mkdir(parents=True, exist_ok=True)
    weights, config = destination / 'mt3.pth', destination / 'config.json'
    if not weights.exists():
        partial = weights.with_suffix('.part')
        with open_url(MT3_WEIGHTS_URL) as response, partial.open('wb') as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        if sha256(partial) != MT3_SHA256:
            raise IOError('MT3 checkpoint SHA256 mismatch.')
        partial.replace(weights)
    if sha256(weights) != MT3_SHA256:
        raise IOError('Existing MT3 checkpoint SHA256 mismatch.')
    if not config.exists():
        with open_url(MT3_CONFIG_URL) as response:
            body = response.read()
        json.loads(body)
        config.write_bytes(body)
    provenance = {'implementation': 'kunato/mt3-pytorch through openmirlab/mt3-infer; unofficial PyTorch port',
                  'adapter_commit': 'd8cbd287470291dee335f84315c5a9b9ff6e6c8a',
                  'weights_url': MT3_WEIGHTS_URL, 'weights_sha256': sha256(weights),
                  'config_url': MT3_CONFIG_URL, 'config_sha256': sha256(config),
                  'auto_filter': False, 'original_google_jax_run': False}
    (destination / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print('Verified MT3-PyTorch checkpoint:', weights, flush=True)
