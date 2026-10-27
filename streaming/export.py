"""Exact note JSON/CSV plus an explicitly quantized, positive-duration MIDI view."""
from pathlib import Path

import mido

from fingering.export import export as interval_export


def export(notes,stem,metadata):
    stem=Path(stem);ticks_per_second=960
    quantized=[(round(n['start']*ticks_per_second),max(round(n['start']*ticks_per_second)+1,round(n['end']*ticks_per_second)),n)
               for n in notes]
    extended=sum(round(n['end']*ticks_per_second)<=start for start,end,n in quantized)
    metadata={**metadata,'midi_time_policy':'120 BPM,480 ticks/quarter; min1 tick per published attack; JSON source intervals unchanged.',
              'midi_minimum_tick_adjustments':extended}
    interval_export(notes,stem,metadata)
    groups={}
    for start,end,note in sorted(quantized,key=lambda row:(row[0],row[2]['pitch'])):
        lanes=groups.setdefault(note.get('string'),[]);pitch=int(note['pitch'])
        lane=next((v for v in lanes if v['ends'].get(pitch,-1)<=start),None)
        if lane is None:lane={'ends':{},'notes':[]};lanes.append(lane)
        lane['ends'][pitch]=end;lane['notes'].append((start,end,pitch))
    lanes=[(string,index,lane) for string,values in sorted(groups.items(),key=lambda row:-1 if row[0] is None else row[0])
           for index,lane in enumerate(values)]
    if len(lanes)>15:raise ValueError('MIDI needs more than15 independent melodic channels; exact JSON is retained.')
    midi=mido.MidiFile(ticks_per_beat=480);tempo=mido.MidiTrack();midi.tracks.append(tempo)
    tempo.append(mido.MetaMessage('set_tempo',tempo=500000,time=0))
    channels=[i for i in range(16) if i!=9]
    for (string,index,lane),channel in zip(lanes,channels):
        track=mido.MidiTrack();midi.tracks.append(track)
        track.append(mido.MetaMessage('track_name',name=f'string_{string}_lane_{index}',time=0))
        track.append(mido.Message('program_change',program=24,channel=channel,time=0))
        events=[]
        for start,end,pitch in lane['notes']:
            events.extend([(start,1,pitch,True),(end,0,pitch,False)])
        previous=0
        for tick,priority,pitch,on in sorted(events):
            track.append(mido.Message('note_on' if on else 'note_off',note=pitch,velocity=90 if on else 0,channel=channel,time=tick-previous))
            previous=tick
    midi.save(stem.with_suffix('.mid'))
