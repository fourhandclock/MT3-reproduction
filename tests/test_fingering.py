import copy
import json

import numpy as np
import pytest

from fingering.decoder import candidates,finger_groups,lowest_fret,solve
from fingering.metrics import playability,preserved


@pytest.fixture
def config():
    with open('configs/fingering.json') as source: return json.load(source)


def note(pitch,start=0.0,end=1.0): return {'pitch':pitch,'start':start,'end':end}


def test_alternative_positions_and_capo_preserve_pitch(config):
    assert (5,0) in candidates(64,config) and (4,5) in candidates(64,config)
    config=copy.deepcopy(config);config['capo']=2
    assert (5,0) in candidates(66,config)
    assert all(config['tuning_midi'][s]+config['capo']+f==66 for s,f in candidates(66,config))


def test_sustained_notes_do_not_share_a_string(config):
    notes=[note(64,0,2),note(64,0.5,1.5)]
    output,_=solve(notes,config)
    assert preserved(notes,output)
    assert output[0]['string']!=output[1]['string']
    assert playability(output,config)['same_string_overlap_events']==0


def test_barre_proxy_does_not_block_an_open_intermediate_string():
    # A full first-fret barre and a partial third-fret barre give a 3-finger lower bound.
    assert finger_groups({0:1,1:3,2:3,3:2,4:1,5:1})==3
    assert finger_groups({0:2,1:0,2:2})==2
    assert finger_groups({0:2,1:4,2:2})==2


def test_impossible_polyphony_is_explicit_without_dropping_notes(config):
    notes=[note(64) for _ in range(7)]
    output,diagnostic=solve(notes,config)
    assert preserved(notes,output)
    assert diagnostic['unassigned']>=1
    assert any(value['string'] is None for value in output)
    assert playability(output,config)['same_string_overlap_events']==0


def test_acoustic_evidence_can_resolve_equal_pitch_candidates(config):
    probability=np.full((100,6,26),0.001,dtype=np.float32)
    probability[:,4,5]=0.9
    output,_=solve([note(64)],config,{'state':probability},acoustic_weight=2)
    assert output[0]['string']==4 and output[0]['fret']==5


def test_out_of_range_pitch_stays_visible(config):
    notes=[note(100)]
    output,_=solve(notes,config)
    assert preserved(notes,output)
    assert output[0]['assignment_status']=='outside supported fretboard'


def test_note_identity_and_pitch_timing_are_not_replaced_by_reference_fingering(config):
    notes=[dict(note(64),string=0,fret=24)]
    baseline=lowest_fret(notes,config)
    assert baseline[0]['string']==5 and baseline[0]['fret']==0
    output,_=solve(notes,config)
    assert preserved(notes,output)


def test_midi_export_preserves_overlapping_duplicate_pitch_voices(tmp_path,config):
    import pretty_midi
    from fingering.export import export
    notes=lowest_fret([note(64,0,1),note(64,0.5,1.5)],config)
    stem=tmp_path/'duplicate'
    export(notes,stem,{})
    midi=pretty_midi.PrettyMIDI(str(stem.with_suffix('.mid')))
    restored=sorted([(n.pitch,n.start,n.end) for instrument in midi.instruments for n in instrument.notes])
    assert len(restored)==2
    np.testing.assert_allclose(restored,[(64,0,1),(64,0.5,1.5)],atol=0.002)
