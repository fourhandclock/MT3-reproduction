"""Select an updated decoder on validation only, preserving trained weights."""

import json
from pathlib import Path

import torch

from baseline.download import ROOT,sha256
from .model import GuitarCNNGRU
from .training import save_checkpoint,validate


def calibrate_decoder(config,output,device):
    output=Path(output).resolve()
    source=output/'best.pt'
    checkpoint=torch.load(source,map_location=device,weights_only=True)
    if checkpoint['config']!=config: raise ValueError('Configuration changed.')
    for name in ['data.py','model.py','training.py']:
        if sha256(ROOT/'cnn_gru'/name)!=checkpoint['source_sha256'][name]:
            raise ValueError(f'Training pipeline changed: {name}')
    manifest_path=ROOT/'data/cnn_gru/manifest.json'
    if sha256(manifest_path)!=checkpoint['manifest_sha256']: raise ValueError('Dataset manifest changed.')
    manifest=json.loads(manifest_path.read_text())
    audit=json.loads((ROOT/'data/cnn_gru/cache_audit.json').read_text())
    evaluation_config=json.loads((ROOT/'configs/stage1.json').read_text())
    model=GuitarCNNGRU(config).to(device);model.load_state_dict(checkpoint['model'])
    validation=validate(model,manifest,audit,config,evaluation_config,device)
    checkpoint['training_source_sha256']=checkpoint['source_sha256'].copy()
    checkpoint['source_sha256']['inference.py']=sha256(ROOT/'cnn_gru/inference.py')
    checkpoint['decoder_parameters']=validation['parameters']
    checkpoint['decoder_validation']=validation
    checkpoint['original_checkpoint_sha256']=sha256(source)
    checkpoint['postprocessing_note']='Release intervals clipped at the next note on the same string; decoder calibrated on validation only, no optimizer updates.'
    save_checkpoint(output/'best-inference.pt',checkpoint)
    report={'validation':validation,'optimizer_updates_added':0,'trained_steps':checkpoint['step'],
            'original_checkpoint_sha256':sha256(source),'derived_checkpoint_sha256':sha256(output/'best-inference.pt'),
            'decoder_source_sha256':checkpoint['source_sha256']['inference.py']}
    (output/'decoder_calibration.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Decoder calibrated on validation only:',validation,'; added optimizer updates: 0',flush=True)
