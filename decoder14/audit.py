"""Reproduce every grid decision and exported interval without neural inference."""

import csv
from collections import Counter
from pathlib import Path

import mido

from baseline.download import ROOT, sha256
from streaming.evaluate import save_json
from .core import (MODELS, compact, delayed_oracle, packet_latency, replay, score_case,
                   summarize, validate_config)
from .workflow import (identity_check, load_inputs, load_records, projection_proof, read, sources)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def midi_intervals(path):
    midi = mido.MidiFile(path)
    intervals = []
    require(midi.ticks_per_beat == 480, 'MIDI tick grid changed.')
    for track in midi.tracks:
        tick, active = 0, {}
        for message in track:
            tick += message.time
            if message.type == 'note_on' and message.velocity > 0:
                key = (message.channel, message.note)
                require(key not in active, 'Overlapping same-pitch MIDI attacks lost a voice.')
                active[key] = tick
            elif message.type == 'note_off' or (message.type == 'note_on' and message.velocity == 0):
                key = (message.channel, message.note)
                require(key in active, 'MIDI release has no corresponding attack.')
                intervals.append((active.pop(key), tick, message.note))
        require(not active, 'MIDI left an unmatched attack.')
    return Counter(intervals)


def verify_exports(stem, row):
    notes = row['notes']
    require(read(stem.with_suffix('.json'))['notes'] == notes, 'Exact JSON notes changed.')
    with stem.with_suffix('.csv').open(newline='', encoding='utf-8') as stream:
        values = list(csv.DictReader(stream))
    expected = [(n['start'], n['end'], n['pitch'], n['string'], n['fret']) for n in notes]
    actual = [(float(n['start']), float(n['end']), int(n['pitch']), int(n['string']), int(n['fret'])) for n in values]
    require(actual == expected, 'CSV source intervals changed.')
    expected_midi = Counter((round(n['start'] * 960), max(round(n['start'] * 960) + 1, round(n['end'] * 960)),
                             n['pitch']) for n in notes)
    require(midi_intervals(stem.with_suffix('.mid')) == expected_midi, 'MIDI lost or moved a quantized attack.')
    require(read(stem.with_name(stem.name + '-events.json'))['events'] == row['events'], 'Exported event ledger changed.')


def audit(output):
    output = Path(output)
    protocol = read(output / 'protocol.json')
    config = protocol['config']
    validate_config(config)
    require(sources() == protocol['source_sha256'], 'Frozen Stage14 producer changed.')
    base, score, inputs, bindings = load_inputs(config)
    require(bindings == protocol['input_sha256'] and base == protocol['decoder_config']
            and score == protocol['scoring_config'], 'Frozen experiment input binding changed.')
    require(protocol['new_optimizer_updates'] == protocol['neural_inference_calls'] == protocol['test_inference_calls'] == 0,
            'Zero-training/cache-only scope changed.')
    grid = read(output / 'grid.json')
    require(len(grid) == 54, 'Expected 27 candidates for each of the two models.')
    # Reconstruct the declared Cartesian product separately from the candidate helper.
    expected_parameters = [{'onset_threshold': onset, 'activity_threshold': activity, 'onset_backdate_seconds': backdate}
                           for onset in [.55, .4, .7] for activity in [.35, .25, .5] for backdate in [.06, 0., .03]]
    require(protocol['candidates'] == expected_parameters, 'Preregistered grid ordering changed.')
    selection = read(output / 'selection.json')
    identities = 0
    cases = 0
    for model in MODELS:
        rows = [row for row in grid if row['model'] == model]
        require([row['candidate'] for row in rows] == list(range(27)), 'Missing/duplicate candidate.')
        for row, parameters in zip(rows, expected_parameters):
            require(row['parameters'] == parameters, 'Candidate parameters changed.')
            recomputed = []
            for item in [item for item in inputs if item['model'] == model]:
                notes, events, _ = replay(item['probabilities'], base, parameters, 12.)
                result = score_case(item['reference'], notes, score, 12.)
                if row['candidate'] == 0:
                    identity_check(item, notes, events, result)
                    identities += 1
                recomputed.append({'id': item['id'], **compact(result)})
                cases += 1
            require(recomputed == row['clips'], 'Per-recording grid scores changed.')
            for key in compact(result):
                values = [value[key] for value in recomputed]
                expected = sum(values) if key in ['matched_onsets', 'predicted_notes', 'reference_notes'] else sum(values) / len(values)
                require(abs(expected - row[key]) < 1e-14, 'Grid macro/pooled aggregate changed.')
        winner = 0
        for index in range(1, len(rows)):
            if (rows[index]['onset_f1'], rows[index]['onset_offset_f1']) > (rows[winner]['onset_f1'], rows[winner]['onset_offset_f1']):
                winner = index
        selected = selection['models'][model]
        require(selected == {'candidate': winner, 'parameters': expected_parameters[winner],
                             'baseline': rows[0], 'selected': rows[winner]}, 'Final per-model selection changed.')
    require(selection['selected_from_test'] is False and selection['production_promoted'] is False
            and selection['rule'] == config['selection_rule'], 'Selection scope changed.')
    records = load_records(output)
    expected_keys = {(item['id'], item['model'] + '_' + variant) for item in inputs for variant in ['strict', 'selected']}
    require(len(records) == 48 and {(r['id'], r['mode']) for r in records} == expected_keys,
            'Missing/duplicate review cases.')
    lookup = {(item['id'], item['model']): item for item in inputs}
    notes_count = events_count = 0
    for row in records:
        item = lookup[(row['id'], row['model'])]
        parameters = config['baseline'] if row['variant'] == 'strict' else selection['models'][row['model']]['parameters']
        notes, events, clamps = replay(item['probabilities'], base, parameters, 12.)
        require(row['notes'] == notes and row['events'] == events and row['parameters'] == parameters
                and row['source_clamps'] == clamps and row['reference'] == item['reference'], 'Saved replay changed.')
        for key, value in score_case(item['reference'], notes, score, 12.).items():
            require(row[key] == value, 'Saved note/tab metric changed.')
        require(row['packet_latency'] == packet_latency(item['reference'], notes, score, 12.), 'Packet-only latency changed.')
        require(row['head_metrics'] == item['previous']['head_metrics']
                and row['probability_sha256'] == item['previous']['probability_sha256'], 'Acoustic-head provenance changed.')
        require(sha256(output / row['relativeaudio']) == sha256(item['audio']), 'Copied review audio changed.')
        require(all('delivered_at_seconds' not in event for event in events), 'A measured compute clock was invented.')
        require(all(event['emitted_at_audio_seconds'] + 1e-9 >= event['source_time'] for event in events),
                'Published source time is later than availability.')
        verify_exports(output / 'validation/results' / row['mode'] / row['id'], row)
        notes_count += len(notes)
        events_count += len(events)
    oracles = read(output / 'oracle.json')
    require(len(oracles) == 48 and len({(r['id'], r['delay_frames'], r['backdate_seconds']) for r in oracles}) == 48,
            'Missing or duplicate delayed-oracle case.')
    for row in oracles:
        item = lookup[(row['id'], MODELS[0])]
        values, conflicts = delayed_oracle(item['reference'], 1200, base, 12., row['delay_frames'])
        parameters = {**config['baseline'], 'onset_backdate_seconds': row['backdate_seconds']}
        notes, events, clamps = replay(values, base, parameters, 12., row['delay_frames'] * .01)
        require(row['notes'] == notes and row['events'] == events and row['source_clamps'] == clamps
                and row['privileged_labels'] is True and row['conflicts'] == conflicts, 'Oracle diagnostic changed.')
        for key, value in score_case(item['reference'], notes, score, 12.).items():
            require(row[key] == value, 'Oracle scoring changed.')
        if row['delay_frames']:
            _, raw, _ = replay(values, base, parameters, 12., 0.)
            require(projection_proof(raw, events, .13) == row['projection_proof'], 'Oracle source projection changed.')
    summary = read(output / 'validation/summary.json')
    require(summary['modes'] == {mode: summarize([row for row in records if row['mode'] == mode])
                                 for mode in dict.fromkeys(row['mode'] for row in records)}, 'Review summary changed.')
    require(summary['oracles'] == {f'delay{frames}_backdate{int(backdate*1000)}': summarize(
        [r for r in oracles if r['delay_frames'] == frames and r['backdate_seconds'] == backdate])
        for frames in [0, 13] for backdate in [.06, 0.]}, 'Oracle summary changed.')
    require(summary['identity_reproductions'] == identities == 24 and summary['grid_cases'] == cases == 648
            and summary['oracle_cases'] == 48 and summary['new_optimizer_updates'] == summary['neural_inference_calls']
            == summary['test_inference_calls'] == 0 and summary['raw_heads_unchanged'] is True
            and summary['production_promoted'] is False, 'Summary budget/scope changed.')
    result = {'passed': True, 'grid_cases_reproduced': cases, 'identity_reproductions': identities,
              'review_cases': 48, 'oracle_cases': 48, 'exact_notes': notes_count, 'exact_events': events_count,
              'new_optimizer_updates': 0, 'neural_inference_calls': 0, 'test_inference_calls': 0,
              'inputs_unchanged': True, 'source_projection_verified': True, 'production_promoted': False,
              'artifact_sha256': {name: sha256(output / name) for name in
                                 ['protocol.json', 'grid.json', 'selection.json', 'oracle.json', 'validation/summary.json']}}
    return result
