import json
from pathlib import Path

from baseline.download import sha256
from cnn_gru.data import load_audio
from .export import export
from .evaluate import load_checkpoint,save_json
from .inference import replay


def predict_file(audio_path,output,checkpoint_root,seconds,packet_ms,device):
    path,checkpoint,model=load_checkpoint(checkpoint_root,device);config=checkpoint['config']
    # File loading/resampling occurs before replay. PCMStream itself accepts only mono16k audio.
    audio,duration=load_audio(Path(audio_path),seconds,config)
    notes,events,timing=replay(model,audio,checkpoint['normalization'],config,checkpoint['decoder_parameters'],device,packet_ms)
    timing['model_warmup_performed']=False
    timing['timing_scope']=timing['timing_scope'].replace('Warm PCM push','PCM push (no explicit model warmup)',1)
    metadata={'input_audio':str(Path(audio_path).resolve()),'checkpoint_sha256':sha256(path),'input_duration_seconds':duration,
              'packet_ms':packet_ms,'lookahead_samples':0,'timing':timing,'decoder_parameters':checkpoint['decoder_parameters'],
              'scope':'Unpaced isolated-guitar replay after offline file read/resampling. No hardware capture, separator, global fingering search, meter or notation in causal loop.'}
    output=Path(output);export(notes,output,metadata)
    save_json(output.with_name(output.name+'-events.json'),{'metadata':metadata,'events':events})
    print('Causal replay exported:',output,';',len(notes),'notes;',len(events),'incremental events',flush=True)
