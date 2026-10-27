import json
from pathlib import Path

import torch

from baseline.download import ROOT,sha256
from cnn_gru.data import features,load_audio
from cnn_gru.inference import decode,predict_probabilities
from cnn_gru.model import GuitarCNNGRU
from .decoder import solve
from .export import export
from .metrics import playability,preserved


def transcribe(audio_path,output,config,selection_path,seconds,device):
    selection=json.loads(Path(selection_path).read_text())
    if selection['config']!=config: raise ValueError('Fingering selection/config mismatch.')
    checkpoint_path=ROOT/'outputs/cnn_gru/best-inference.pt'
    if sha256(checkpoint_path)!=selection['checkpoint_sha256']: raise ValueError('Acoustic model changed.')
    checkpoint=torch.load(checkpoint_path,map_location=device,weights_only=True)
    model_config=checkpoint['config']
    model=GuitarCNNGRU(model_config).to(device);model.load_state_dict(checkpoint['model'])
    audio,duration=load_audio(Path(audio_path),seconds,model_config)
    probability=predict_probabilities(model,features(audio,model_config),checkpoint['normalization'],model_config,device)
    original=decode(probability,model_config,checkpoint['decoder_parameters'])
    notes,diagnostic=solve(original,config,probability,selection['acoustic_weight'])
    if not preserved(original,notes): raise RuntimeError('Fingering changed pitch or timing.')
    metadata={'input_audio':str(Path(audio_path).resolve()),'input_duration_seconds':duration,
              'checkpoint_sha256':sha256(checkpoint_path),'fretboard':config,'acoustic_weight':selection['acoustic_weight'],
              'pitch_timing_preserved':True,'solver_diagnostic':diagnostic,'playability_proxies':playability(notes,config),
              'scope':'Offline acoustic-guitar prototype; fingering cannot repair transcription errors.'}
    export(notes,Path(output),metadata)
    print('Exported constrained transcription:',output,'; unresolved:',diagnostic['unassigned'],flush=True)
