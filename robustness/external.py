import hashlib
import http.client
import json
import time
from pathlib import Path

from baseline.download import ROOT,open_url,sha256

RECORD_URL='https://zenodo.org/api/records/11406378'


def md5(path):
    digest=hashlib.md5()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def prepare_external():
    folder=ROOT/'data/robustness/egset12';folder.mkdir(parents=True,exist_ok=True)
    record=folder/'record.json'
    if not record.exists():
        with open_url(RECORD_URL) as response: record.write_bytes(response.read())
    metadata=json.loads(record.read_text())
    files={entry['key']:entry for entry in metadata['files']}
    entries=[]
    for index in range(1,13):
        entry={'id':f'EGSet12_{index:02d}','role':'external evaluation only'}
        for suffix,key in [('.wav','audio'),('.jams','annotation')]:
            name=f'{index:02d}{suffix}';info=files[name];target=folder/name
            expected=info['checksum'].split(':',1)[1]
            if not target.exists():
                temporary=target.with_suffix(target.suffix+'.part')
                for attempt in range(4):
                    try:
                        with open_url(info['links']['self']) as response,temporary.open('wb') as stream:
                            while block:=response.read(1024*1024): stream.write(block)
                        if temporary.stat().st_size!=info['size'] or md5(temporary)!=expected:
                            raise IOError('Downloaded size/checksum mismatch.')
                        temporary.replace(target);break
                    except (OSError,http.client.IncompleteRead) as exc:
                        if attempt==3: raise
                        print('Retrying',name,type(exc).__name__,flush=True);time.sleep(min(2**attempt,4))
            if md5(target)!=expected: raise IOError(f'External file checksum mismatch: {name}')
            entry[key]=target.relative_to(ROOT).as_posix();entry[key+'_sha256']=sha256(target)
        entries.append(entry);print('Verified external recording',entry['id'],flush=True)
    manifest={'dataset':'EGSet12','record_url':RECORD_URL,'license':metadata['metadata'].get('license'),
              'scope':'Twelve original electric-guitar recordings, one guitarist and guitar/amp/microphone setup; no additional audio effects.',
              'role':'Never used for training, normalization, threshold selection or early stopping.',
              'annotation_policy':'Provided note_midi/string labels retained unchanged; timing alignment not independently audited.',
              'excerpts':entries}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
