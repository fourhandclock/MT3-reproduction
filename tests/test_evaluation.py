import json

import numpy as np

from baseline.download import select_ids
from baseline.evaluation import evaluate, midi_notes, piano_roll, reference_notes

CONFIG = {'score_seconds': 2.0, 'frame_rate': 62.5, 'onset_tolerance_seconds': 0.05,
          'offset_tolerance_seconds': 0.05, 'offset_ratio': 0.2}


def note(start, end, pitch=60):
    return {'start': start, 'end': end, 'pitch': pitch}


def test_offsets_are_evaluated_separately_from_onsets():
    result = evaluate([note(0.1, 0.5)], [note(0.12, 1.0)], CONFIG, 3.0)
    assert result['onset']['f1'] == 1.0
    assert result['onset_offset']['f1'] == 0.0
    json.dumps(result)  # mir_eval matching indices must be converted from NumPy scalars.


def test_context_onsets_do_not_pollute_scored_prefix():
    result = evaluate([note(0.1, 0.5), note(2.1, 2.5)],
                      [note(0.1, 0.5), note(2.1, 2.5)], CONFIG, 3.0)
    assert result['onset']['reference_count'] == 1
    assert result['onset']['predicted_count'] == 1
    assert result['onset']['f1'] == 1.0


def test_censored_offset_is_not_treated_as_observed_release():
    result = evaluate([note(0.1, 4.0), note(1.0, 1.5, 64)],
                      [note(0.1, 3.0), note(1.0, 1.5, 64)], CONFIG, 3.0)
    assert result['right_censored_reference_notes'] == 1
    assert result['onset']['reference_count'] == 2
    assert result['onset_offset']['reference_count'] == 1
    assert result['onset_offset']['predicted_count'] == 1
    assert result['onset_offset']['f1'] == 1.0


def test_empty_predictions_are_valid_failures():
    result = evaluate([note(0.1, 0.5)], [], CONFIG, 3.0)
    assert result['onset']['f1'] == 0.0
    assert result['frame']['f1'] == 0.0


def test_frame_grid_uses_intervals_without_inventing_earlier_onset():
    roll = piano_roll([note(0.01, 0.03)], 0.1, 100)
    np.testing.assert_array_equal(np.where(roll[:, 60])[0], [1, 2])


def test_guitarset_microtonal_note_labels_are_rounded_but_strings_preserved(tmp_path):
    path = tmp_path / 'annotation.jams'
    path.write_text(json.dumps({'annotations': [{'namespace': 'note_midi',
        'annotation_metadata': {'data_source': '5'},
        'data': [{'time': 0.1, 'duration': 0.5, 'value': 64.1}]}]}))
    notes = reference_notes(path)
    assert notes[0]['pitch'] == 64
    assert notes[0]['string'] == 5
    assert notes[0]['fret'] == 0


def test_selection_includes_long_style_names_and_is_deterministic():
    names = [f'{player:02d}_{style}3-120-C_{role}.jams'
             for player in range(6) for role in ['comp', 'solo']
             for style in ['BN', 'SS', 'Jazz', 'Rock', 'Funk']]
    first = select_ids(names, 12, 3, 42)
    assert first == select_ids(names, 12, 3, 42)
    assert {value.split('_')[1].split('3-')[0] for value in first} == {'BN', 'SS', 'Jazz', 'Rock', 'Funk'}
    assert len({value[:2] for value in first}) == 6


def test_midi_predictions_and_metrics_can_be_serialized(tmp_path):
    import pretty_midi
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=24)
    instrument.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.1, end=0.5))
    midi.instruments.append(instrument)
    path = tmp_path / 'prediction.mid'
    midi.write(str(path))
    predicted, drums = midi_notes(path)
    assert drums == 0
    assert type(predicted[0]['pitch']) is int
    json.dumps(evaluate([note(0.1, 0.5)], predicted, CONFIG, 3.0))
