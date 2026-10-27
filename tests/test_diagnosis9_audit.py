import copy
import itertools
import json

import mido
import pytest

from baseline.download import sha256
from diagnosis9 import audit as checker
from streaming.export import export


def selection_fixture(tied=False):
    config = {'checkpoint_root': 'outputs/streaming', 'input_seconds': 12.0, 'score_seconds': 10.0,
              'packet_ms': 20, 'selection_split': 'validation', 'optimizer_updates': 0,
              'baseline_parameters': {'onset_threshold': .55, 'activity_threshold': .35, 'onset_backdate_seconds': .06},
              'decoder_grid': {'state_alpha': [1.0, .5, .25], 'confirm_frames': [4, 2],
                               'onset_backdate_seconds': [.06, 0.0, .12]},
              'selection_rule': 'Mean validation onset F1, then onset+offset F1; stable ties.'}
    ids = [f'validation-{index}' for index in range(12)]
    candidates = []
    for index, (alpha, confirm, backdate) in enumerate(itertools.product(
            config['decoder_grid']['state_alpha'], config['decoder_grid']['confirm_frames'],
            config['decoder_grid']['onset_backdate_seconds'])):
        value = .1 if tied else .1 + index * .001
        rows = [{'id': identity, 'onset_f1': value, 'onset_offset_f1': .04, 'tab_onset_f1': .07,
                 'predicted_notes': 4, 'missed_notes': 3, 'extra_notes': 2} for identity in ids]
        candidates.append({'parameters': {**config['baseline_parameters'], 'state_alpha': alpha,
                                           'confirm_frames': confirm, 'onset_backdate_seconds': backdate},
                           'onset_f1': value, 'onset_offset_f1': .04, 'frame_f1': .2, 'clips': rows})
    winner = candidates[0] if tied else candidates[-1]
    selection = {'config': copy.deepcopy(config), 'selected_from_test': False, 'optimizer_updates': 0,
                 'selection_rule': config['selection_rule'], 'candidates': candidates,
                 'selected_validation': copy.deepcopy(winner), 'selected_parameters': copy.deepcopy(winner['parameters']),
                 'strict_validation': copy.deepcopy(candidates[0])}
    return config, selection, ids


def ledger_fixture():
    note = {'note_id': 0, 'string': 0, 'fret': 5, 'pitch': 45, 'program': 24,
            'start': .01, 'end': .05, 'on_emitted_at_audio_seconds': .04,
            'off_emitted_at_audio_seconds': .06, 'on_delivered_at_seconds': .045,
            'off_delivered_at_seconds': .066, 'censored_at_end': False}
    events = []
    for kind, time, availability, delivery in [('note_on', .01, .04, .045), ('note_off', .05, .06, .066)]:
        events.append({'type': kind, 'note_id': 0, 'string': 0, 'fret': 5, 'pitch': 45,
                       'time': time, 'source_time': time, 'emitted_at_audio_seconds': availability,
                       'delivered_at_seconds': delivery, 'flush': False,
                       'start' if kind == 'note_on' else 'end': time})
    return [note], events


def test_grid_has_exact_coverage_and_stable_identity_ties():
    config, selection, ids = selection_fixture(tied=True)
    winner, identity = checker._validate_candidates(config, selection, ids)
    assert winner is identity is selection['candidates'][0]


@pytest.mark.parametrize('mutation', ['missing', 'duplicate_grid', 'duplicate_ids', 'nan', 'wrong_aggregate', 'wrong_winner', 'test_selection'])
def test_selection_rejects_protocol_or_metric_mutations(mutation):
    config, selection, ids = selection_fixture()
    if mutation == 'missing':
        selection['candidates'].pop()
    elif mutation == 'duplicate_grid':
        selection['candidates'][1]['parameters'] = copy.deepcopy(selection['candidates'][0]['parameters'])
    elif mutation == 'duplicate_ids':
        selection['candidates'][0]['clips'][-1]['id'] = ids[0]
    elif mutation == 'nan':
        selection['candidates'][0]['clips'][0]['onset_f1'] = float('nan')
    elif mutation == 'wrong_aggregate':
        selection['candidates'][0]['onset_f1'] += .01
    elif mutation == 'wrong_winner':
        selection['selected_parameters'] = copy.deepcopy(selection['candidates'][0]['parameters'])
    elif mutation == 'test_selection':
        selection['selected_from_test'] = True
    with pytest.raises(ValueError):
        checker._validate_candidates(config, selection, ids)


def test_source_clock_can_be_backdated_but_delivery_cannot_precede_availability():
    notes, events = ledger_fixture()
    result = checker._validate_event_ledger(notes, events, .06, 20, 6)
    assert result['note_count'] == 1 and result['event_count'] == 2
    events[0]['delivered_at_seconds'] = .039
    notes[0]['on_delivered_at_seconds'] = .039
    with pytest.raises(ValueError, match='clock ordering'):
        checker._validate_event_ledger(notes, events, .06, 20, 6)


def test_ledger_rejects_extra_releases_and_crossed_source_aliases():
    notes, events = ledger_fixture()
    with pytest.raises(ValueError, match='exactly one ordered'):
        checker._validate_event_ledger(notes, events + [copy.deepcopy(events[-1])], .06, 20, 6)
    events[1]['end'] = .051
    with pytest.raises(ValueError, match='interval alias'):
        checker._validate_event_ledger(notes, events, .06, 20, 6)


def test_json_comparison_accepts_rounding_but_rejects_nonfinite_and_boolean_counts():
    checker._equal({'f1': .1 + 1e-15, 'count': 12, 'empty': None}, {'f1': .1, 'count': 12, 'empty': None}, 'metrics')
    with pytest.raises(ValueError, match='numeric'):
        checker._equal(True, 1, 'count')
    with pytest.raises(ValueError, match='nonfinite'):
        checker._equal(float('nan'), .1, 'f1')


def test_midi_attacks_and_minimum_one_tick_releases_preserved(tmp_path):
    notes = [{'string': 0, 'fret': 5, 'pitch': 45, 'start': .04, 'end': .040000001},
             {'string': 1, 'fret': 0, 'pitch': 45, 'start': .04, 'end': .08}]
    stem = tmp_path / 'notes'
    export(notes, stem, {'mode': 'repair'})
    assert checker._validate_midi(stem.with_suffix('.mid'), notes) == 2
    midi = mido.MidiFile(stem.with_suffix('.mid'))
    attack = next(message for track in midi.tracks for message in track if message.type == 'note_on')
    attack.note += 1
    release = next(message for track in midi.tracks for message in track if message.type == 'note_off')
    release.note += 1
    midi.save(stem.with_suffix('.mid'))
    with pytest.raises(ValueError, match='preservation'):
        checker._validate_midi(stem.with_suffix('.mid'), notes)


@pytest.fixture
def selection_paths(tmp_path, monkeypatch):
    from diagnosis9 import workflow
    config, selection, ids = selection_fixture()
    monkeypatch.setattr(checker, 'ROOT', tmp_path)
    output = tmp_path / 'outputs/diagnosis9'
    output.mkdir(parents=True)
    checkpoint_path = tmp_path / 'outputs/streaming/model/best-inference.pt'
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b'frozen checkpoint fixture')
    manifest = {'validation': [{'id': identity} for identity in ids],
                'test': [{'id': f'test-{index}'} for index in range(12)]}
    manifest_path = tmp_path / 'data/cnn_gru/manifest.json'
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest))
    head_source = tmp_path / 'diagnosis9/head_audit.py'
    head_source.parent.mkdir(parents=True)
    head_source.write_text('# frozen head producer')
    loss_source = tmp_path / 'cnn_gru/model.py'
    loss_source.parent.mkdir(parents=True)
    loss_source.write_text('# frozen loss')
    head = {'test_used': False, 'new_optimizer_updates': 0,
            'checkpoint_sha256': sha256(checkpoint_path), 'manifest_sha256': sha256(manifest_path),
            'head_audit_sha256': sha256(head_source)}
    head_path = output / 'validation-head-diagnostics.json'
    head_path.write_text(json.dumps(head))
    producer = {'diagnosis9/workflow.py': 'fixture-source-hash'}
    selection.update({'checkpoint_sha256': sha256(checkpoint_path), 'manifest_sha256': sha256(manifest_path),
                      'head_diagnostics_sha256': sha256(head_path), 'production_source_sha256': producer})
    (output / 'selection.json').write_text(json.dumps(selection))
    checkpoint = {'config': {'evaluation_audio_seconds': 12.0}, 'scoring_config': {'score_seconds': 10.0},
                  'legacy_loss_sha256': sha256(loss_source)}
    monkeypatch.setattr(workflow, 'production_hashes', lambda: producer)
    monkeypatch.setattr(workflow, 'load_inputs', lambda config, directory: (checkpoint_path, checkpoint, manifest, []))
    return config, output, head_path


def test_public_selection_check_enforces_stored_head_artifact_sha(selection_paths):
    config, output, head_path = selection_paths
    result = checker.validate_selection(config, output)
    assert result['passed'] and result['candidate_count'] == 18
    assert result['optimizer_updates'] == 0
    head_path.write_text(head_path.read_text() + '\n')
    with pytest.raises(ValueError, match='Frozen head diagnostics SHA'):
        checker.validate_selection(config, output)


def test_public_selection_check_rejects_changed_selected_parameters(selection_paths):
    config, output, _ = selection_paths
    path = output / 'selection.json'
    saved = json.loads(path.read_text())
    saved['selected_parameters']['state_alpha'] = 1
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='Selected parameters'):
        checker.validate_selection(config, output)
