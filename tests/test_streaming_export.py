import json

from baseline.evaluation import midi_notes
from streaming.export import export


def test_tiny_published_intervals_keep_midi_attacks_and_exact_json(tmp_path):
    notes=[{'start':.1,'end':.100000001,'pitch':64,'string':5,'fret':0},
           {'start':.100000001,'end':.2,'pitch':64,'string':5,'fret':0}]
    export(notes,tmp_path/'tiny',{'mode':'test'})
    document=json.loads((tmp_path/'tiny.json').read_text())
    assert document['notes']==notes and document['metadata']['midi_minimum_tick_adjustments']==1
    written,_=midi_notes(tmp_path/'tiny.mid')
    assert len(written)==2 and all(n['end']>n['start'] for n in written)
