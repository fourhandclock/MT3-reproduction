import gc
import json
import time
from pathlib import Path

import soundfile as sf
import torch

from baseline.download import ROOT,sha256
from .separator import Separator
from .workflow import write_json


def release(device):
    gc.collect()
    if device.type=='cuda': torch.cuda.empty_cache()


def transcribe(audio_path,output,seconds,include_score,config,device):
    output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    with sf.SoundFile(audio_path) as source:
        audio=source.read(frames=int(seconds*source.samplerate),dtype='float32',always_2d=True)
        sr=source.samplerate
    if audio.shape[1] not in [1,2]: raise ValueError('Audio must have one or two channels.')
    if audio.shape[1]==1: audio=audio[:,0]
    started=time.perf_counter();separator=Separator(config,device);load_seconds=time.perf_counter()-started
    stem_audio,diagnostic=separator.separate(audio,sr)
    provenance=separator.provenance
    guitar_path=output.with_name(output.name+'-guitar.wav')
    sf.write(guitar_path,stem_audio,sr,subtype='FLOAT')
    del separator;release(device)
    from fingering.predict_audio import transcribe as notes_transcribe
    fretboard=json.loads((ROOT/'configs/fingering.json').read_text())
    notes_started=time.perf_counter()
    notes_transcribe(guitar_path,output,fretboard,ROOT/'outputs/fingering/selection.json',seconds,device)
    note_seconds=time.perf_counter()-notes_started;release(device)
    raw_document=json.loads(output.with_suffix('.json').read_text())
    run={'input_audio':str(Path(audio_path).resolve()),'input_audio_sha256':sha256(audio_path),
         'guitar_audio':str(guitar_path),'guitar_audio_sha256':sha256(guitar_path),
         'separator':provenance,'separation':diagnostic,'cold_separator_load_seconds':load_seconds,
         'acoustic_and_fingering_wall_seconds':note_seconds,'models_executed_sequentially':True,'training_updates':0,
         'scope':'Offline prototype; extracted guitar may contain several guitars, other instruments and missing notes.'}
    if include_score:
        from rhythm.tracker import Tracker
        from rhythm.grid import make_grid
        from rhythm.workflow import score_document
        from rhythm.notation import export_score
        rhythm_config=json.loads((ROOT/'configs/rhythm.json').read_text())
        # Full accompaniment can supply beat evidence that disappears in an isolated guitar stem.
        tracker=Tracker(rhythm_config,device);rhythm=tracker.predict(audio_path,seconds)
        rhythm.pop('beat_probability');rhythm.pop('downbeat_probability')
        write_json(output.with_name(output.name+'-beats.json'),rhythm)
        grid=make_grid(rhythm['beats'],rhythm['downbeats'],rhythm['duration_seconds'])
        document=score_document(raw_document['notes'],grid,rhythm_config)
        document['metadata']={'input_audio':str(Path(audio_path).resolve()),'beat_input':'original mixture',
                              'note_input':'separated guitar stem','scope':'Editable draft; correct uncertain notes, beats, meter and chords.'}
        score_stem=output.with_name(output.name+'-score');export_score(document,score_stem,Path(audio_path).stem)
        run['score_file']=str(score_stem.with_suffix('.musicxml'))
        del tracker;release(device)
    write_json(output.with_name(output.name+'-run.json'),run)
    print('Mixed-audio transcription exported:',output,'; guitar stem:',guitar_path,flush=True)
