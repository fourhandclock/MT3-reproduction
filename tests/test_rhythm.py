import copy
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from rhythm.chords import accuracy, estimate
from rhythm.grid import make_grid, quantize, to_quarters, to_seconds
from rhythm.notation import allocate_voices, export_score, validate_xml
from rhythm.workflow import event_metrics, score_document


@pytest.fixture
def config():
    return json.load(open('configs/rhythm.json'))


def note(pitch=64, start=0, end=1, string=5, fret=0):
    return {'pitch': pitch, 'start': start, 'end': end, 'string': string, 'fret': fret}


def test_variable_tempo_inverse_and_event_matching():
    grid = make_grid([0, .5, 1.2, 2.1, 3], [0], 3)
    assert to_quarters(1.55, grid) == pytest.approx(2+.35/.9)
    for time in [-.1, .1, 1.55, 3.3]:
        assert to_seconds(to_quarters(time, grid), grid) == pytest.approx(time)
    assert event_metrics([0,.5,1], [.01,.49,.51,1.2], 2, .07)['f1'] == pytest.approx(4/7)


def test_pickup_meter_change_and_compound_override():
    grid = make_grid(np.arange(10)*.5, [1, 2.5, 4.5], 4.7)
    assert grid['bars'][0]['pickup'] and grid['bars'][0]['end'] == 0
    assert [b['numerator'] for b in grid['bars'][1:3]] == [3,4]
    compound = make_grid([0, .6, 1.2, 1.8, 2.4], [0, 1.2, 2.4], 2.4,
                         {'numerator':6,'denominator':8,'pulse_quarters':1.5})
    assert compound['bars'][0]['end'] == 3
    assert compound['bars'][0]['numerator'] == 6
    assert to_quarters(.6, compound) == 1.5


def test_invalid_grids_and_sparse_fallback():
    with pytest.raises(ValueError): make_grid([0], [], 1)
    with pytest.raises(ValueError): make_grid([0, float('nan')], [], 1)
    with pytest.raises(ValueError): make_grid([0,1], [float('nan')], 1)
    grid = make_grid([0,.5,1], [], 1)
    assert grid['bars'][0]['numerator'] == 4
    assert any('Insufficient' in warning for warning in grid['warnings'])
    with pytest.raises(ValueError): make_grid([0,1], [], 1, {'numerator':3,'denominator':4,'pulse_quarters':2})


def test_exact_oracle_lattice_and_triplets_preserve_raw(config):
    grid = make_grid(np.arange(9)*.5, [0,2,4], 4)
    notes = [note(start=.125, end=.5), note(start=1, end=2.5)]
    result = score_document(notes, grid, config, subdivision=4)
    assert [n['raw'] for n in result['notes']] == notes
    assert [n['onset_shift_seconds'] for n in result['notes']] == [0,0]
    triplets = [note(start=t, end=t+1/6) for t in [0,1/6,1/3]]
    result = quantize(triplets, grid, config, 3)
    assert [n['start_tick'] for n in result['notes']] == [0,4,8]


def test_short_notes_collisions_and_unresolved_are_flagged_not_dropped(config, tmp_path):
    grid = make_grid(np.arange(9)*.5, [0,2,4], 4)
    notes = [note(start=.02, end=.03), note(start=.04, end=.05), note(60, 1, 2, None, None)]
    original = copy.deepcopy(notes)
    document = score_document(notes, grid, config, subdivision=4)
    assert notes == original and len(document['notes']) == 3
    assert all('minimum written duration' in n['flags'] for n in document['notes'][:2])
    assert all('quantized string overlap' in n['flags'] for n in document['notes'][:2])
    assert len(allocate_voices(document['notes'])) == 3
    export_score(document, tmp_path/'score', 'Short events')
    tree = ET.parse(tmp_path/'score.musicxml')
    attacks = [n for n in tree.findall('./part/measure/note') if n.find('pitch') is not None and
               n.findtext('staff') == '1' and n.find("tie[@type='stop']") is None]
    assert len(attacks) == 3
    unresolved = next(n for n in attacks if n.findtext('pitch/step') == 'C')
    assert unresolved.find('notations/technical/string') is None


def test_musicxml_bar_lengths_ties_strings_and_tuplets(config, tmp_path):
    grid = make_grid(np.arange(17)*.5, [0,2,4,6,8], 8)
    document = score_document([note(start=1.5, end=3), note(60, 0, 1/6, 4, 1)], grid, config, subdivision=3)
    export_score(document, tmp_path/'score', 'Ties')
    tree = ET.parse(tmp_path/'score.musicxml')
    assert validate_xml(tmp_path/'score.musicxml')
    assert tree.find("./part/measure/note/tie[@type='start']") is not None
    assert tree.find("./part/measure/note/tie[@type='stop']") is not None
    assert tree.findtext('./part/measure/note/time-modification/actual-notes') == '3'
    assert any(n.findtext('notations/technical/string') == '1' for n in tree.findall('./part/measure/note'))
    assert tree.findtext('./part/measure/attributes/staff-details/staff-tuning/tuning-step') == 'E'
    for measure in tree.findall('./part/measure'):
        voices = {}
        for n in measure.findall('note'):
            key = (n.findtext('voice'), n.findtext('staff'))
            voices[key] = voices.get(key,0)+int(n.findtext('duration'))
        assert set(voices.values()) == {48}


def test_chord_templates_silence_and_masking(config):
    grid = make_grid(np.arange(13)*.5, [0,2,4,6], 6)
    notes = [note(p,0,2,None,None) for p in [60,64,67]] + [note(p,2,4,None,None) for p in [57,60,64]]
    result = estimate(notes, grid, config)
    assert next(r['label'] for r in result if r['start'] == .5) == 'C:maj'
    assert next(r['label'] for r in result if r['start'] == 2.5) == 'A:min'
    assert result[-1]['label'] == 'N'
    reference = [{'start':0,'end':1,'label':'C:maj7/3'}, {'start':1,'end':2,'label':'D:dim'}]
    assert accuracy(reference, [{'start':0,'end':2,'label':'C:maj'}], 2)['major_minor_duration_accuracy'] == 1
    assert accuracy(reference, result, 2)['coverage_fraction'] == .5


def test_manual_corrected_score_keeps_notes(config, tmp_path):
    from rhythm.__main__ import corrected_score
    corrections = {'beats':[0,.5,1,1.5,2], 'downbeats':[0,2], 'duration_seconds':2,
                   'meter':{'numerator':4,'denominator':4,'pulse_quarters':1},
                   'subdivision':4, 'chords':[{'start':0,'end':2,'label':'E:min'}]}
    path = tmp_path/'corrections.json'; path.write_text(json.dumps(corrections))
    notes = [note(start=.2,end=.7)]
    corrected_score(notes, path, config, tmp_path/'score')
    result = json.loads((tmp_path/'score.json').read_text())
    assert result['notes'][0]['raw'] == notes[0]
    assert result['chords'][0]['label'] == 'E:min'
    corrections['chords'][0]['label'] = 'bad'; path.write_text(json.dumps(corrections))
    with pytest.raises(ValueError): corrected_score(notes, path, config, tmp_path/'bad')
