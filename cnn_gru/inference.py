from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

from baseline.evaluation import STANDARD_TUNING, match_notes, scores


def predict_probabilities(model, x, normalization, config, device):
    import torch
    mean, std = np.asarray(normalization['mean'],dtype=np.float32), np.asarray(normalization['std'],dtype=np.float32)
    x = np.clip((np.asarray(x,dtype=np.float32)-mean)/std,-5,5)
    length = round(config['training']['crop_seconds']*config['feature']['sample_rate']/config['feature']['hop_length'])
    overlap = config['decoder']['inference_overlap_frames']
    stride = length-overlap
    if stride <= 0:
        raise ValueError('Overlap must be shorter than inference window.')
    starts = list(range(0,max(1,len(x)-length+1),stride))
    if len(x)>length and starts[-1]+length<len(x):
        starts.append(len(x)-length)
    sums = {key: None for key in ['state','onset','offset','pitch']}
    weights = np.zeros(len(x),dtype=np.float32)
    taper = np.minimum(np.arange(length)+1,np.arange(length,0,-1)).astype(np.float32)
    taper = np.minimum(taper/max(overlap,1),1)
    model.eval()
    batch_size = config['training']['batch_size']
    with torch.inference_mode():
        for offset in range(0,len(starts),batch_size):
            positions = starts[offset:offset+batch_size]
            windows = [np.pad(x[start:start+length],((0,max(0,start+length-len(x))),(0,0))) for start in positions]
            tensor = torch.from_numpy(np.stack(windows)).to(device)
            with torch.autocast(device_type=device.type,dtype=torch.float16,
                                enabled=device.type=='cuda' and config['training']['mixed_precision']):
                logits = model(tensor)
            outputs = {key: (torch.softmax(value.float(),dim=-1) if key=='state' else torch.sigmoid(value.float())).cpu().numpy()
                       for key,value in logits.items()}
            for batch_index,start in enumerate(positions):
                valid = min(length,len(x)-start)
                weights[start:start+valid] += taper[:valid]
                for key,value in outputs.items():
                    if sums[key] is None:
                        sums[key] = np.zeros((len(x),)+value.shape[2:],dtype=np.float32)
                    shape = (valid,)+(1,)*(value.ndim-2)
                    sums[key][start:start+valid] += value[batch_index,:valid]*taper[:valid].reshape(shape)
    if not np.all(weights>0):
        raise ValueError('Inference left uncovered frames.')
    return {key:value/weights.reshape((len(x),)+(1,)*(value.ndim-1)) for key,value in sums.items()}


def decode(probabilities, config, parameters):
    fps = config['feature']['sample_rate']/config['feature']['hop_length']
    best_fret_probability=probabilities['state'][:,:,:-1].max(axis=-1)
    activity=best_fret_probability/(best_fret_probability+probabilities['state'][:,:,-1]+1e-8)
    frets = probabilities['state'][:,:,:-1].argmax(axis=-1)
    states = np.where(activity>=parameters['activity_threshold'],frets,-1)
    notes = []
    minimum = max(1,round(config['decoder']['minimum_note_seconds']*fps))
    retrigger = max(1,round(config['decoder']['minimum_retrigger_seconds']*fps))
    refinement = round(config['decoder']['offset_refinement_seconds']*fps)
    for string in range(6):
        state = median_filter(states[:,string],size=3,mode='nearest')
        boundaries = np.flatnonzero(np.diff(np.r_[-2,state,-2]))
        onsets,_ = find_peaks(probabilities['onset'][:,string],height=parameters['onset_threshold'],distance=retrigger)
        offsets,_ = find_peaks(probabilities['offset'][:,string],height=config['decoder']['offset_threshold'],distance=retrigger)
        for start,end in zip(boundaries[:-1],boundaries[1:]):
            fret = int(state[start])
            if fret<0 or end-start<minimum:
                continue
            peaks = [int(peak) for peak in onsets if start<=peak<end]
            if peaks and peaks[0]-start<=3:
                start = peaks.pop(0)
            split_points = [start]+[peak for peak in peaks if peak-start>=minimum and end-peak>=minimum]+[end]
            near_offsets = [int(peak) for peak in offsets if end-refinement<=peak<=end+2]
            if near_offsets and near_offsets[-1]-split_points[-2]>=minimum:
                split_points[-1] = min(len(state),near_offsets[-1])
            for left,right in zip(split_points[:-1],split_points[1:]):
                if right-left>=minimum:
                    notes.append({'start':left/fps,'end':right/fps,'string':string,'fret':fret,
                                  'pitch':STANDARD_TUNING[string]+fret,'program':24})
    # Release refinement must not make two frets overlap on the same string.
    for string in range(6):
        ordered=sorted((note for note in notes if note['string']==string),key=lambda note:note['start'])
        for previous,following in zip(ordered,ordered[1:]):
            previous['end']=min(previous['end'],following['start'])
    return sorted(notes,key=lambda note:(note['start'],note['pitch'],note['string']))


def save_midi(notes,path):
    import pretty_midi
    midi = pretty_midi.PrettyMIDI(initial_tempo=120,resolution=480)
    for string in range(6):
        instrument = pretty_midi.Instrument(program=24,name=f'string_{string}')
        for note in notes:
            if note['string']==string:
                instrument.notes.append(pretty_midi.Note(velocity=90,pitch=int(note['pitch']),
                                                        start=float(note['start']),end=float(note['end'])))
        midi.instruments.append(instrument)
    midi.write(str(path))


def read_string_midi(path):
    import pretty_midi
    midi = pretty_midi.PrettyMIDI(str(path))
    notes=[]
    for instrument in midi.instruments:
        string=int(instrument.name.split('_')[-1])
        for note in instrument.notes:
            pitch=int(note.pitch)
            notes.append({'start':float(note.start),'end':float(note.end),'pitch':pitch,
                          'string':string,'fret':pitch-STANDARD_TUNING[string],'program':24})
    return sorted(notes,key=lambda note:(note['start'],note['pitch'],note['string']))


def tablature_metrics(reference,predicted,config,score_seconds,clip_seconds,probabilities=None,target=None,feature_frame_rate=100):
    reference=[note for note in reference if 0<=note['start']<score_seconds]
    predicted=[note for note in predicted if 0<=note['start']<score_seconds]
    tp_onset=tp_offset=offset_ref_count=offset_est_count=0
    for string in range(6):
        ref=[note for note in reference if note['string']==string]
        est=[note for note in predicted if note['string']==string]
        pairs=match_notes(ref,est,config)
        tp_onset+=len(pairs)
        censored={i for i,note in enumerate(ref) if note['end']>=clip_seconds}
        removed_est={j for i,j in pairs if i in censored}
        ref_offset=[note for i,note in enumerate(ref) if i not in censored]
        est_offset=[note for j,note in enumerate(est) if j not in removed_est]
        tp_offset+=len(match_notes(ref_offset,est_offset,config,offsets=True))
        offset_ref_count+=len(ref_offset)
        offset_est_count+=len(est_offset)
    result={'onset':scores(range(tp_onset),len(reference),len(predicted)),
            'onset_offset':scores(range(tp_offset),offset_ref_count,offset_est_count)}
    if probabilities is not None and target is not None:
        fret_count=probabilities['state'].shape[-1]-1
        prediction=probabilities['state'].argmax(axis=-1)
        frames=min(len(prediction),round(score_seconds*feature_frame_rate))
        truth=target['state'][:frames]
        prediction=prediction[:frames]
        valid=truth!=-100
        correct=(truth==prediction)&valid&(truth<fret_count)
        tp=int(correct.sum())
        ref=int(((truth<fret_count)&valid).sum())
        est=int(((prediction<fret_count)&valid).sum())
        result['frame']=scores(range(tp),ref,est)
        result['frame']['accuracy_including_inactive']=float(((truth==prediction)&valid).sum()/max(valid.sum(),1))
    return result
