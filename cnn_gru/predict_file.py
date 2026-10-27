import json
from pathlib import Path

import torch

from baseline.download import sha256
from .data import features,load_audio
from .inference import decode,predict_probabilities,read_string_midi,save_midi
from .model import GuitarCNNGRU


def transcribe_file(audio_path,checkpoint_path,output_path,device,seconds=30):
    checkpoint_path=Path(checkpoint_path).resolve()
    checkpoint=torch.load(checkpoint_path,map_location=device,weights_only=True)
    config=checkpoint['config']
    torch.set_num_threads(config['training']['cpu_threads'])
    model=GuitarCNNGRU(config).to(device);model.load_state_dict(checkpoint['model'])
    audio,duration=load_audio(Path(audio_path),seconds,config)
    probability=predict_probabilities(model,features(audio,config),checkpoint['normalization'],config,device)
    notes=decode(probability,config,checkpoint['decoder_parameters'])
    output=Path(output_path).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    save_midi(notes,output)
    notes=read_string_midi(output)
    result={'input_audio':str(Path(audio_path).resolve()),'duration_seconds':duration,
            'checkpoint_sha256':sha256(checkpoint_path),'decoder_parameters':checkpoint['decoder_parameters'],
            'decoder_source_sha256':sha256(Path(__file__).parent/'inference.py'),
            'tuning_midi':[40,45,50,55,59,64],'string_index_order':'0 low E through 5 high E',
            'model_scope':'Offline acoustic-guitar prototype; fixed standard tuning; no meter or technique prediction.',
            'notes':notes}
    output.with_suffix('.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(f'Saved {len(notes)} timed string/fret predictions to {output} and {output.with_suffix(".json")}',flush=True)
