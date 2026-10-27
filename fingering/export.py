import csv
import json
from pathlib import Path


def export(notes,stem,metadata):
    import pretty_midi
    stem=Path(stem);stem.parent.mkdir(parents=True,exist_ok=True)
    document={'metadata':metadata,'notes':notes,'time_unit':'seconds',
              'string_order':'0 low E to 5 high E; null means unresolved',
              'source_of_truth':'JSON intervals; MIDI is quantized to ticks'}
    stem.with_suffix('.json').write_text(json.dumps(document,indent=2),encoding='utf-8')
    with stem.with_suffix('.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['start','end','pitch','string','fret','assignment_status'])
        writer.writerows([[note['start'],note['end'],note['pitch'],note.get('string'),note.get('fret'),note.get('assignment_status','assigned')] for note in notes])
    midi=pretty_midi.PrettyMIDI(initial_tempo=120,resolution=480)
    # Allocate extra lanes only for overlapping repetitions of the same pitch,
    # so MIDI note-off semantics do not silently shorten duplicate voices.
    lanes={}
    for note in sorted(notes,key=lambda value:(value['start'],value['pitch'])):
        string=note.get('string');group=lanes.setdefault(string,[])
        lane=next((value for value in group if value['ends'].get(note['pitch'],-1)<=note['start']),None)
        if lane is None:
            lane={'ends':{},'notes':[]};group.append(lane)
        lane['ends'][note['pitch']]=note['end'];lane['notes'].append(note)
    if sum(len(group) for group in lanes.values())>15:
        document['metadata']['midi_export']='Not exported: more than 15 independent melodic lanes required.'
        stem.with_suffix('.json').write_text(json.dumps(document,indent=2),encoding='utf-8')
        return
    for string,group in lanes.items():
        for lane_index,lane in enumerate(group):
            name=f'string_{string}_lane_{lane_index}' if string is not None else f'unassigned_lane_{lane_index}'
            instrument=pretty_midi.Instrument(program=24,name=name)
            instrument.notes=[pretty_midi.Note(velocity=90,pitch=int(note['pitch']),start=note['start'],end=note['end']) for note in lane['notes']]
            midi.instruments.append(instrument)
    midi.write(str(stem.with_suffix('.mid')))
