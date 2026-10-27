"""Reproduce saved diagnosis and verify its accounting without acoustic inference."""

import csv
from collections import Counter
from pathlib import Path

from baseline.download import ROOT, sha256
from decoder14.core import delayed_oracle, replay, score_case
from decoder14.audit import midi_intervals
from decoder14.workflow import read
from diagnosis9.oracles import replace_heads
from .analysis import CATEGORIES
from .workflow import (INTERVENTIONS, MODELS, aggregate, describe_item, load_inputs,
                       load_records, recommend, sources, validate_config)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_partition(row):
    value = row['taxonomy']
    refs, predictions = value['reference_rows'], value['prediction_rows']
    require([r['reference_index'] for r in refs] == list(range(len(refs))), 'Reference identities changed.')
    require([r['prediction_index'] for r in predictions] == list(range(len(predictions))), 'Prediction identities changed.')
    pairs = [(r['reference_index'], r['prediction_index']) for r in refs if r['prediction_index'] is not None]
    require(len({j for _, j in pairs}) == len(pairs), 'A prediction was used to explain multiple reference notes.')
    require(sum(value['reference_counts'].values()) == value['reference_count'] == len(refs), 'Reference categories do not conserve support.')
    require(set(value['reference_counts']) == set(CATEGORIES), 'Unexpected category in the exclusive partition.')
    require(len(pairs) + value['extra_predictions'] == value['prediction_count'] == len(predictions), 'Prediction partition does not conserve support.')
    for i, j in pairs:
        require(predictions[j]['reference_index'] == i and predictions[j]['category'] == refs[i]['category'], 'Association is not bidirectional.')
    require(value['matched_onsets'] == row['pitch_metrics']['onset']['true_positives'], 'Official onset matching was changed by diagnostic associations.')
    counts = row['frames']['counts']
    require(sum(counts[key] for key in ['correct_fret', 'predicted_inactive', 'wrong_octave_same_string', 'other_wrong_fret'])
            == counts['reference_active_frames'], 'Frame error categories do not conserve support.')
    return len(refs), len(predictions)


def audit(output):
    output = Path(output)
    protocol = read(output / 'protocol.json')
    config = protocol['config']
    validate_config(config)
    require(protocol['source_sha256'] == sources(), 'Frozen Stage15 source changed.')
    base, score, inputs, bindings = load_inputs(config)
    require(protocol['input_sha256'] == bindings and protocol['decoder_config'] == base
            and protocol['scoring_config'] == score, 'Frozen diagnostic input/config changed.')
    require(protocol['new_optimizer_updates'] == protocol['neural_inference_calls'] == protocol['test_inference_calls'] == 0
            and protocol['production_promoted'] is False, 'The zero-training/validation-only scope changed.')
    records = load_records(output)
    lookup = {(r['id'], r['model']): r for r in records}
    require(len(lookup) == len(records) == len(inputs) == 24, 'Missing or duplicated diagnostic case.')
    interventions = read(output / 'interventions.json')
    intervention_lookup = {(r['id'], r['model'], r['variant']): r for r in interventions}
    require(len(intervention_lookup) == len(interventions) == 96, 'Missing or duplicated head intervention.')
    expected_records, expected_interventions = [], []
    reference_count = prediction_count = event_count = 0
    for item in inputs:
        item['score'] = score
        expected = describe_item(item, config)
        row = lookup[(item['id'], item['model'])]
        require(row == expected, 'Saved classification, evidence, frames or calibrated notes changed.')
        nr, ne = verify_partition(row)
        reference_count += nr
        prediction_count += ne
        expected_records.append(expected)
        path = output / 'validation/results' / item['model'] / (item['id'] + '.csv')
        with path.open(newline='', encoding='utf-8') as stream:
            values = list(csv.DictReader(stream))
        require(len(values) == nr, 'Reference CSV omitted a diagnostic row.')
        for raw, saved in zip(values, row['taxonomy']['reference_rows']):
            flattened = {**saved, **saved['note'], **saved['evidence']}
            require(all(text == ('' if flattened[key] is None else str(flattened[key]))
                        for key, text in raw.items()), 'CSV association, interval, tag or evidence changed.')
        require(sha256(output / row['relativeaudio']) == sha256(item['audio']), 'Diagnostic audio copy changed.')
        oracle, conflicts = delayed_oracle(item['reference'], 1200, base, 12., 13)
        for variant, heads in [('predicted', []), ('oracle_state', ['state']), ('oracle_onset', ['onset']), ('oracle_both', ['state','onset'])]:
            probability = replace_heads(item['probabilities'], oracle, heads)
            notes, events, clamps = replay(probability, base, item['parameters'], 12.)
            case = {'id': item['id'], 'model': item['model'], 'variant': variant,
                    'oracle_heads': heads, 'privileged_labels': bool(heads), 'conflicts': conflicts,
                    'notes': notes, 'events': events, 'source_clamps': clamps,
                    **score_case(item['reference'], notes, score, 12.)}
            require(intervention_lookup[(item['id'], item['model'], variant)] == case, 'Intervention source notes, head provenance, events or scores changed.')
            require(len(events) == 2 * len(notes), 'Intervention event ledger lost a boundary.')
            require(all('delivered_at_seconds' not in event for event in events), 'A measured processing clock was invented.')
            if variant == 'predicted':
                require(notes == item['selected']['notes'] and events == item['selected']['events'], 'Stage14 selected source replay changed.')
            expected_interventions.append(case)
            event_count += len(events)
    summary = aggregate(expected_records, expected_interventions)
    require(summary == read(output / 'summary.json'), 'Summary aggregation changed.')
    recommendation = recommend(summary, config)
    require(recommendation == read(output / 'recommendation.json'), 'Declared recommendation rule changed.')
    proposal = read(output / 'next-experiment.json')
    require(proposal['status'] == 'proposed_not_trained' and proposal['supported'] == recommendation['supported_by_declared_rule']
            and proposal['recommendation'] == recommendation['recommendation'], 'A proposal was presented as an executed experiment.')
    require(sum(v['new_updates'] for v in proposal['arms'].values()) == proposal['future_total_new_optimizer_update_cap'] == 500
            and proposal['future_new_epochs_per_arm'] == 5 and proposal['parent_lifetime_optimizer_updates'] == 500
            and proposal['future_lifetime_updates_per_arm'] == 750, 'Future budget/lifetime accounting changed.')
    require(proposal['proposal_source_sha256'] == sha256(ROOT / proposal['proposal_module'])
            and proposal['parent_checkpoint_sha256'] == sha256(ROOT / proposal['parent_checkpoint']), 'Loss proposal or frozen parent changed.')
    # Full-ideal heads must agree between families under the identical decoder.
    for item in [x for x in inputs if x['model'] == MODELS[0]]:
        left = intervention_lookup[(item['id'], MODELS[0], 'oracle_both')]
        right = intervention_lookup[(item['id'], MODELS[1], 'oracle_both')]
        require(left['notes'] == right['notes'] and left['events'] == right['events'], 'Identical ideal inputs changed by model family.')
    exported_notes = 0
    if (output / 'review.html').exists():
        for row in records:
            stem = output / 'validation/notes' / row['model'] / row['id']
            require(read(stem.with_suffix('.json'))['notes'] == row['notes'], 'Review note JSON changed source intervals.')
            with stem.with_suffix('.csv').open(newline='', encoding='utf-8') as stream:
                values = list(csv.DictReader(stream))
            require([(float(v['start']),float(v['end']),int(v['pitch']),int(v['string']),int(v['fret'])) for v in values]
                    == [(n['start'],n['end'],n['pitch'],n['string'],n['fret']) for n in row['notes']], 'Review note CSV changed.')
            ticks = Counter((round(n['start']*960),max(round(n['start']*960)+1,round(n['end']*960)),n['pitch'])
                            for n in row['notes'])
            require(midi_intervals(stem.with_suffix('.mid')) == ticks, 'Review MIDI lost a quantized attack.')
            exported_notes += len(row['notes'])
    return {'passed': True, 'diagnostic_cases': 24, 'intervention_cases': 96, 'identity_reproductions': 24,
            'reference_rows': reference_count, 'prediction_rows': prediction_count, 'intervention_events': event_count,
            'review_note_exports': exported_notes,
            'new_optimizer_updates': 0, 'neural_inference_calls': 0, 'test_inference_calls': 0,
            'frozen_inputs_unchanged': True, 'one_to_one_partitions_verified': True, 'loss_proposal_trained': False,
            'artifact_sha256': {name: sha256(output / name) for name in
                               ['protocol.json', 'summary.json', 'interventions.json', 'recommendation.json', 'next-experiment.json']}}
