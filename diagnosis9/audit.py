"""Independent read-only checks for stage-9 selection and exported results.

The audit may write its own report; it never trains, changes producer files,
reselects parameters, or rewrites evaluation artifacts. Optional validation
recomputation uses saved validation probabilities on the CPU.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import itertools
import json
import math
from pathlib import Path
import re

import numpy as np

from baseline.download import ROOT, sha256


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _equal(actual, expected, label, tolerance=1e-10):
    """Compare JSON values, rejecting nonfinite numbers and boolean coercions."""
    if isinstance(expected, dict):
        _require(isinstance(actual, dict) and set(actual) == set(expected), label + ': object keys differ.')
        for key in expected:
            _equal(actual[key], expected[key], label + '.' + str(key), tolerance)
    elif isinstance(expected, list):
        _require(isinstance(actual, list) and len(actual) == len(expected), label + ': list length differs.')
        for index, value in enumerate(expected):
            _equal(actual[index], value, label + f'[{index}]', tolerance)
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        _require(type(actual) is type(expected) and actual == expected, label + ': value differs.')
    elif isinstance(expected, (int, float)):
        _require(isinstance(actual, (int, float)) and not isinstance(actual, bool), label + ': numeric value required.')
        _require(math.isfinite(float(actual)) and math.isfinite(float(expected)), label + ': nonfinite number.')
        _require(math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=tolerance), label + ': numeric value differs.')
    else:
        _require(actual == expected, label + ': value differs.')


def _finite(value, label, minimum=None, maximum=None):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
             label + ': finite numeric value required.')
    _require(minimum is None or value >= minimum, label + ': below minimum.')
    _require(maximum is None or value <= maximum, label + ': above maximum.')
    return float(value)


def _integer(value, label, minimum=0):
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
             label + ': nonnegative integer required.')
    return value


def _workspace_path(relative):
    path = (ROOT / relative).resolve()
    _require(path.is_relative_to(ROOT.resolve()), 'Artifact path leaves the workspace.')
    return path


def _validate_candidates(config, selection, validation_ids):
    """Verify all 18 records and the stable validation-only winner."""
    _equal(selection['config'], config, 'Selection config')
    _require(config.get('selection_split') == 'validation', 'Selection split must be validation.')
    _require(selection.get('selected_from_test') is False, 'Test-based selection is forbidden.')
    _integer(config.get('optimizer_updates'), 'Configured optimizer updates')
    _integer(selection.get('optimizer_updates'), 'Selection optimizer updates')
    _require(config['optimizer_updates'] == selection['optimizer_updates'] == 0, 'Stage 9 must use zero updates.')
    _equal(selection['selection_rule'], config['selection_rule'], 'Selection rule')
    grid = config['decoder_grid']
    _equal(grid, {'state_alpha': [1.0, .5, .25], 'confirm_frames': [4, 2],
                  'onset_backdate_seconds': [.06, 0.0, .12]}, 'Stage-9 frozen grid')
    expected = [{**config['baseline_parameters'], 'state_alpha': alpha, 'confirm_frames': confirm,
                 'onset_backdate_seconds': backdate}
                for alpha, confirm, backdate in itertools.product(
                    grid['state_alpha'], grid['confirm_frames'], grid['onset_backdate_seconds'])]
    candidates = selection['candidates']
    _require(len(candidates) == len(expected) == 18, 'Expected exactly 18 decoder candidates.')
    _require(len(validation_ids) == len(set(validation_ids)) == 12, 'Expected 12 unique validation IDs.')
    for index, (candidate, parameters) in enumerate(zip(candidates, expected)):
        _equal(candidate['parameters'], parameters, f'Candidate {index} grid parameters')
        rows = candidate['clips']
        _require(len(rows) == 12 and {row['id'] for row in rows} == set(validation_ids)
                 and len({row['id'] for row in rows}) == len(rows),
                 f'Candidate {index}: incomplete or duplicate validation IDs.')
        for row in rows:
            for key in ['onset_f1', 'onset_offset_f1', 'tab_onset_f1']:
                _finite(row[key], f'Candidate {index}/{row["id"]}/{key}', 0, 1)
            for key in ['predicted_notes', 'missed_notes', 'extra_notes']:
                _integer(row[key], f'Candidate {index}/{row["id"]}/{key}')
            _require(row['extra_notes'] <= row['predicted_notes'], 'Extra notes exceed predicted count.')
        for key in ['onset_f1', 'onset_offset_f1', 'frame_f1']:
            _finite(candidate[key], f'Candidate {index}/{key}', 0, 1)
        for key in ['onset_f1', 'onset_offset_f1']:
            _equal(candidate[key], float(np.mean([row[key] for row in rows])), f'Candidate {index}/{key} aggregate')
    # max is stable: the first identity record wins exact lexicographic ties.
    winner = max(candidates, key=lambda row: (row['onset_f1'], row['onset_offset_f1']))
    _equal(selection['selected_validation'], winner, 'Selected validation winner')
    _equal(selection['selected_parameters'], winner['parameters'], 'Selected parameters')
    identity = next(row for row in candidates if row['parameters']['state_alpha'] == 1
                    and row['parameters']['confirm_frames'] == 4
                    and row['parameters']['onset_backdate_seconds'] == config['baseline_parameters']['onset_backdate_seconds'])
    _require(identity is candidates[0], 'Identity must be first for stable tie breaking.')
    _equal(selection['strict_validation'], identity, 'Strict validation entry')
    return winner, identity


def _recompute_validation(dataset, checkpoint, config, candidate):
    from baseline.evaluation import evaluate
    from cnn_gru.inference import tablature_metrics
    from .workflow import decode
    rows, all_metrics = [], []
    for entry, probability, reference, duration in dataset:
        notes, _ = decode(probability, checkpoint['config'], candidate['parameters'], duration, config['packet_ms'])
        metric = evaluate(reference, notes, checkpoint['scoring_config'], duration)
        tab = tablature_metrics(reference, notes, checkpoint['scoring_config'], config['score_seconds'], duration)
        rows.append({'id': entry['id'], 'onset_f1': metric['onset']['f1'],
                     'onset_offset_f1': metric['onset_offset']['f1'], 'tab_onset_f1': tab['onset']['f1'],
                     'predicted_notes': metric['onset']['predicted_count'], 'missed_notes': metric['errors']['missed_notes'],
                     'extra_notes': metric['errors']['extra_notes']})
        all_metrics.append(metric)
    _equal({row['id']: row for row in candidate['clips']}, {row['id']: row for row in rows},
           'Recomputed validation clip metrics')
    for key, metric_key in [('onset_f1', 'onset'), ('onset_offset_f1', 'onset_offset'), ('frame_f1', 'frame')]:
        _equal(candidate[key], float(np.mean([row[metric_key]['f1'] for row in all_metrics])),
               'Recomputed validation ' + key)


def validate_selection(config, output, recompute=False):
    """Return a verified selection summary, or raise ValueError before replay."""
    from .workflow import load_inputs, production_hashes
    output = Path(output).resolve()
    selection_path = output / 'selection.json'
    selection_digest = sha256(selection_path)
    selection = _read(selection_path)
    head_path = output / 'validation-head-diagnostics.json'
    _equal(sha256(head_path), selection['head_diagnostics_sha256'], 'Frozen head diagnostics SHA')
    _equal(selection['production_source_sha256'], production_hashes(), 'Frozen production source/config SHA')
    path, checkpoint, manifest, dataset = load_inputs(config, output)
    _equal(selection['checkpoint_sha256'], sha256(path), 'Selection checkpoint SHA')
    _equal(selection['manifest_sha256'], sha256(ROOT / 'data/cnn_gru/manifest.json'), 'Selection manifest SHA')
    _require(len(manifest['test']) == 12 and not ({row['id'] for row in manifest['test']}
                                               & {row['id'] for row in manifest['validation']}),
             'Validation and test IDs must be disjoint, with 12 test clips.')
    _equal(config['input_seconds'], checkpoint['config']['evaluation_audio_seconds'], 'Input duration binding')
    _equal(config['score_seconds'], checkpoint['scoring_config']['score_seconds'], 'Scored duration binding')
    head = _read(head_path)
    _require(head.get('test_used') is False and head.get('new_optimizer_updates') == 0,
             'Validation head audit must use no test data or new updates.')
    _equal(head['checkpoint_sha256'], selection['checkpoint_sha256'], 'Head checkpoint binding')
    _equal(head['manifest_sha256'], selection['manifest_sha256'], 'Head manifest binding')
    _equal(sha256(ROOT / 'diagnosis9/head_audit.py'), head['head_audit_sha256'], 'Head producer source SHA')
    _equal(sha256(ROOT / 'cnn_gru/model.py'), checkpoint['legacy_loss_sha256'], 'Frozen shared loss source SHA')
    winner, identity = _validate_candidates(config, selection, [row['id'] for row in manifest['validation']])
    count = 0
    if recompute:
        for candidate in [identity] + ([] if winner is identity else [winner]):
            _recompute_validation(dataset, checkpoint, config, candidate)
            count += 1
    _equal(sha256(selection_path), selection_digest, 'Selection changed during verification')
    _equal(sha256(head_path), selection['head_diagnostics_sha256'], 'Head audit changed during verification')
    _equal(selection['production_source_sha256'], production_hashes(), 'Production changed during verification')
    return {'passed': True, 'selection_sha256': selection_digest,
            'checkpoint_sha256': selection['checkpoint_sha256'], 'manifest_sha256': selection['manifest_sha256'],
            'candidate_count': 18, 'validation_clip_count': 12, 'selected_from_test': False,
            'optimizer_updates': 0, 'selected_parameters': selection['selected_parameters'],
            'selected_validation_onset_f1': winner['onset_f1'], 'strict_validation_onset_f1': identity['onset_f1'],
            'recomputed_validation_candidates': count}


def _validate_event_ledger(notes, events, duration, packet_ms, max_backlog_ms):
    ids = [note['note_id'] for note in notes]
    _require(len(ids) == len(set(ids)) and sorted(ids) == list(range(len(ids))), 'Note IDs are not unique contiguous IDs.')
    notes_by_id = {note['note_id']: note for note in notes}
    pairs, availability_groups = defaultdict(list), {}
    previous_available = previous_delivery = -1.0
    tuning = (40, 45, 50, 55, 59, 64)
    for index, event in enumerate(events):
        _require(event['type'] in ['note_on', 'note_off'], 'Unknown ledger event type.')
        _require(event['note_id'] in notes_by_id, 'Event refers to an unknown note ID.')
        note = notes_by_id[event['note_id']]
        for key in ['string', 'fret', 'pitch']:
            _equal(event[key], note[key], 'Event note identity/' + key)
        source = _finite(event['source_time'], 'Event source time', 0, duration + 1e-9)
        available = _finite(event['emitted_at_audio_seconds'], 'Event availability', 0)
        delivered = _finite(event['delivered_at_seconds'], 'Event delivery', 0)
        _require(available + 1e-9 >= source and delivered + 1e-9 >= available,
                 'Delivery/source clock ordering is invalid.')
        _require(available + 1e-9 >= previous_available and delivered + 1e-9 >= previous_delivery,
                 'Ledger delivery clocks move backwards.')
        previous_available, previous_delivery = available, delivered
        _require(isinstance(event['flush'], bool), 'Flush flag must be boolean.')
        _equal(event['time'], source, 'Event time alias')
        kind = 'start' if event['type'] == 'note_on' else 'end'
        prefix = 'on' if event['type'] == 'note_on' else 'off'
        _equal(event[kind], source, 'Event interval alias')
        _equal(note[kind], source, 'Note source interval binding')
        _equal(note[prefix + '_emitted_at_audio_seconds'], available, 'Note availability binding')
        _equal(note[prefix + '_delivered_at_seconds'], delivered, 'Note delivery binding')
        _require(event['type'] != 'note_on' or event['flush'] is False, 'EOF cannot create a note-on.')
        if event['flush']:
            _equal(source, duration, 'EOF release time')
        else:
            _equal(available / (packet_ms / 1000), round(available / (packet_ms / 1000)), 'Packet availability grid', 1e-8)
            _require(1000 * (delivered - available) <= max_backlog_ms + 1e-6,
                     'Event delivery exceeds saved maximum PCM backlog.')
        group = (round(available, 9), event['flush'])
        if group in availability_groups:
            _equal(delivered, availability_groups[group], 'Events from one PCM packet have different delivery clocks')
        else:
            availability_groups[group] = delivered
        pairs[event['note_id']].append((index, event))
    _require(set(pairs) == set(notes_by_id), 'A note lacks ledger events.')
    strings = defaultdict(list)
    for note in notes:
        _integer(note['note_id'], 'Note ID')
        _integer(note['string'], 'String')
        _integer(note['fret'], 'Fret')
        _require(note['string'] < 6 and note['fret'] <= 24, 'Note position outside supported fretboard.')
        _equal(note['pitch'], tuning[note['string']] + note['fret'], 'Pitch/fret identity')
        _equal(note['program'], 24, 'MIDI instrument program')
        start = _finite(note['start'], 'Note start', 0, duration)
        end = _finite(note['end'], 'Note end', 0, duration + 1e-9)
        _require(end > start, 'Published note duration must be positive.')
        pair = pairs[note['note_id']]
        _require(len(pair) == 2 and [event['type'] for _, event in pair] == ['note_on', 'note_off'],
                 'Every note needs exactly one ordered on/off pair.')
        _require(isinstance(note['censored_at_end'], bool), 'Censoring flag must be boolean.')
        _equal(note['censored_at_end'], pair[1][1]['flush'], 'EOF censoring binding')
        strings[note['string']].append(note)
    for values in strings.values():
        ordered = sorted(values, key=lambda note: (note['start'], note['note_id']))
        _require(all(first['end'] <= second['start'] + 1e-9 for first, second in zip(ordered[:-1], ordered[1:])),
                 'Published notes overlap on one string.')
    return {'note_count': len(notes), 'event_count': len(events),
            'censored_release_count': sum(note['censored_at_end'] for note in notes)}


def _validate_midi(path, notes):
    import mido
    midi = mido.MidiFile(path)
    _equal(midi.ticks_per_beat, 480, 'MIDI tick resolution')
    tempos = [message.tempo for track in midi.tracks for message in track if message.type == 'set_tempo']
    _equal(tempos, [500000], 'MIDI fixed tempo')
    actual = []
    for track in midi.tracks:
        names = [message.name for message in track if message.type == 'track_name']
        musical = any(message.type in ['note_on', 'note_off'] for message in track)
        if not musical:
            continue
        _require(len(names) == 1, 'MIDI voice lacks one track name.')
        match = re.fullmatch(r'string_(\d+)_lane_(\d+)', names[0])
        _require(match is not None, 'MIDI string/voice track name is invalid.')
        string = int(match.group(1))
        tick, active = 0, {}
        for message in track:
            _require(message.time >= 0, 'MIDI delta time is negative.')
            tick += message.time
            if message.type not in ['note_on', 'note_off']:
                continue
            _require(message.channel != 9, 'Guitar note exported on the drum channel.')
            key = (message.channel, message.note)
            if message.type == 'note_on' and message.velocity > 0:
                _require(key not in active, 'Overlapping duplicate MIDI voice attack.')
                active[key] = tick
            else:
                _require(key in active, 'MIDI release has no attack.')
                first = active.pop(key)
                _require(tick > first, 'MIDI note has nonpositive tick duration.')
                actual.append((string, message.note, first, tick))
        _require(not active, 'MIDI notes remain open at EOF.')
    expected = [(note['string'], note['pitch'], round(note['start'] * 960),
                 max(round(note['start'] * 960) + 1, round(note['end'] * 960))) for note in notes]
    _require(Counter(actual) == Counter(expected), 'MIDI pitch/attack/release preservation failed.')
    return len(actual)


def audit(config, output, recompute=False):
    """Verify saved validation and all 24 test cases, then save audit.json."""
    from baseline.evaluation import evaluate, reference_notes
    from cnn_gru.data import load_audio
    from cnn_gru.inference import tablature_metrics
    from streaming.evaluate import aggregate, save_json
    from streaming.latency import measure
    import soundfile as sf
    output = Path(output).resolve()
    validated = validate_selection(config, output, recompute)
    selection = _read(output / 'selection.json')
    run_path, summary_path = output / 'evaluation/run.json', output / 'evaluation/summary.json'
    run, summary = _read(run_path), _read(summary_path)
    _equal(run['config'], config, 'Evaluation config')
    _equal(run['selection_sha256'], validated['selection_sha256'], 'Evaluation frozen selection SHA')
    _equal(run['checkpoint_sha256'], validated['checkpoint_sha256'], 'Evaluation checkpoint SHA')
    for key in ['optimizer_updates', 'new_training_epochs']:
        _integer(run[key], 'Evaluation ' + key)
        _require(run[key] == 0, 'Evaluation contains new training.')
    _require(run['weights_changed'] is False and run['same_features_normalization_model'] is True,
             'Evaluation must retain frozen acoustic weights, features and normalization.')
    _require(run['test_used_for_selection'] is False and run['reused_test_set'] is True,
             'Expected reused-test diagnosis with validation-only selection.')
    _equal(run['selection'], {'parameters': validated['selected_parameters'],
                             'validation_onset_f1': validated['selected_validation_onset_f1'],
                             'strict_validation_onset_f1': validated['strict_validation_onset_f1']}, 'Evaluation selection binding')
    presentation = {}
    for name, digest in run['source_sha256'].items():
        path = _workspace_path('diagnosis9/' + name)
        if name == 'report.py':
            presentation[name] = {'at_evaluation_sha256': digest, 'current_sha256': sha256(path),
                                  'changed_since_evaluation': sha256(path) != digest}
        else:
            _equal(sha256(path), digest, 'Evaluation producer source/' + name)
    for name, digest in run['diagnostic_artifacts_sha256'].items():
        _equal(sha256(output / name), digest, 'Evaluation diagnostic artifact/' + name)
    _equal(run['head_diagnostics_summary'], _read(output / 'validation-head-diagnostics.json'), 'Embedded head audit')
    oracle = _read(output / 'oracle-diagnostics.json')
    _equal(run['oracle_summary'], {'summary': oracle['summary'], 'protocol': oracle['protocol']}, 'Embedded oracle audit')
    _equal(run['hybrid_summary'], _read(output / 'hybrid-head-diagnostics.json').get('summary'), 'Embedded hybrid audit')
    manifest = _read(ROOT / 'data/cnn_gru/manifest.json')
    checkpoint_path = _workspace_path(config['checkpoint_root'] + '/model/best-inference.pt')
    import torch
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    _equal(run['scoring_config'], checkpoint['scoring_config'], 'Frozen scoring configuration')
    entries = {entry['id']: entry for entry in manifest['test']}
    expected_files = {output / 'evaluation/results' / mode / (identity + '-metrics.json')
                      for mode in ['strict', 'repair'] for identity in entries}
    actual_files = set((output / 'evaluation/results').glob('*/*-metrics.json'))
    _require(actual_files == expected_files and len(actual_files) == 24, 'Expected exactly 24 test metric cases.')
    audio_cache, records, file_hashes = {}, [], {}
    note_count = event_count = midi_count = 0
    for path in sorted(expected_files):
        record = _read(path)
        identity, mode = record['id'], record['mode']
        _require(identity in entries and mode in ['strict', 'repair'], 'Unexpected test ID or mode.')
        _require(path == output / 'evaluation/results' / mode / (identity + '-metrics.json'), 'Metric filename identity differs.')
        entry = entries[identity]
        if identity not in audio_cache:
            for field in ['audio', 'annotation']:
                _equal(sha256(_workspace_path(entry[field])), entry[field + '_sha256'], 'Original test source/' + identity + '/' + field)
            audio, duration = load_audio(_workspace_path(entry['audio']), config['input_seconds'], checkpoint['config'])
            audio_cache[identity] = audio, duration
        audio, duration = audio_cache[identity]
        _equal(record['relativeaudio'], 'evaluation/clips/' + identity + '.wav', 'Audio clip identity')
        exported_audio, rate = sf.read(output / record['relativeaudio'], dtype='float32')
        _equal(rate, checkpoint['config']['feature']['sample_rate'], 'Exported audio sample rate')
        _require(exported_audio.ndim == 1 and np.array_equal(exported_audio, audio), 'Exported PCM differs from evaluated source.')
        reference = reference_notes(_workspace_path(entry['annotation']))
        _equal(record['reference'], reference, 'Test reference annotations')
        stem = path.with_name(identity)
        interval_path, event_path, midi_path = stem.with_suffix('.json'), stem.with_name(identity + '-events.json'), stem.with_suffix('.mid')
        interval = _read(interval_path)
        _equal(interval['notes'], record['notes'], 'Exact JSON notes used for scoring')
        _equal(interval['metadata']['mode'], mode, 'Export mode')
        _equal(interval['metadata']['checkpoint_sha256'], validated['checkpoint_sha256'], 'Export checkpoint SHA')
        params = config['baseline_parameters'] if mode == 'strict' else validated['selected_parameters']
        _equal(interval['metadata']['decoder_parameters'], params, 'Export frozen decoder parameters')
        _equal(_read(event_path), {'mode': mode, 'events': record['events']}, 'Exported ledger binding')
        timing = record['timing']
        _equal(timing['input_seconds'], duration, 'Timing audio duration')
        _equal(timing['packet_ms'], config['packet_ms'], 'Timing packet size')
        _equal(timing['mode'], mode, 'Timing mode')
        _equal(timing['lookahead_samples'], 0, 'Timing lookahead')
        _finite(timing['processing_seconds'], 'Processing seconds', 0)
        _equal(timing['real_time_factor'], timing['processing_seconds'] / duration, 'Processing ratio')
        _finite(timing['max_virtual_backlog_ms'], 'Maximum backlog', 0)
        _finite(timing['chunk_processing_p50_ms'], 'Median processing', 0)
        _finite(timing['chunk_processing_p95_ms'], 'P95 processing', timing['chunk_processing_p50_ms'])
        ledger = _validate_event_ledger(record['notes'], record['events'], duration,
                                        config['packet_ms'], timing['max_virtual_backlog_ms'])
        midi_count += _validate_midi(midi_path, record['notes'])
        note_count += ledger['note_count']
        event_count += ledger['event_count']
        minimum_ticks = sum(round(note['end'] * 960) <= round(note['start'] * 960) for note in record['notes'])
        _equal(interval['metadata']['midi_minimum_tick_adjustments'], minimum_ticks, 'MIDI minimum-duration count')
        _equal(record['pitch_metrics'], evaluate(reference, record['notes'], run['scoring_config'], duration), 'Recomputed pitch metrics')
        _equal(record['tab_metrics'], tablature_metrics(reference, record['notes'], run['scoring_config'], config['score_seconds'], duration),
               'Recomputed string metrics')
        _equal(record['latency'], measure(reference, record['notes'], run['scoring_config'], duration), 'Recomputed matched-note latency')
        records.append(record)
        for artifact in [path, interval_path, event_path, midi_path]:
            file_hashes[str(artifact.relative_to(output)).replace('\\', '/')] = sha256(artifact)
    _equal(summary, aggregate(records), 'Recomputed test summary')
    _equal(sha256(output / 'selection.json'), validated['selection_sha256'], 'Selection changed during artifact audit')
    _equal(selection['production_source_sha256'], __import__('diagnosis9.workflow', fromlist=['production_hashes']).production_hashes(),
           'Production changed during artifact audit')
    result = {'passed': True, 'selection': validated, 'test_case_count': len(records),
              'test_modes': {'strict': 12, 'repair': 12}, 'verified_json_notes': note_count,
              'verified_midi_attacks': midi_count, 'verified_ledger_events': event_count,
              'new_optimizer_updates': 0, 'new_training_epochs': 0, 'weights_changed': False,
              'recomputed_test_metrics': True, 'recomputed_test_aggregate': True,
              'source_sha256': sha256(Path(__file__)), 'run_sha256': sha256(run_path),
              'summary_sha256': sha256(summary_path), 'artifact_sha256': file_hashes,
              'presentation_sources': presentation,
              'limits': 'Audits saved file replay. No microphone, unseen-data, confidence-calibration or acoustic-accuracy claim.'}
    save_json(output / 'audit.json', result)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/diagnosis9.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/diagnosis9')
    parser.add_argument('--recompute', action='store_true')
    options = parser.parse_args()
    result = audit(_read(options.config), options.output, options.recompute)
    print(json.dumps({key: value for key, value in result.items() if key != 'artifact_sha256'}, indent=2))
