"""Diagnose frozen calibrated models without audio/model inference or training."""

import csv
import shutil
from pathlib import Path

import numpy as np

from baseline.download import ROOT, sha256
from decoder14.core import delayed_oracle, replay, score_case, summarize
from decoder14.workflow import load_inputs as prior_inputs, read
from diagnosis9.oracles import replace_heads, validate_cached_target_identity
from streaming.evaluate import save_json
from .analysis import CATEGORIES, add_note_evidence, frame_diagnosis, source_arrays, taxonomy


MODELS = ['group8_500', 'group1_500']
INTERVENTIONS = ['predicted', 'oracle_state', 'oracle_onset', 'oracle_both']


def validate_config(config):
    expected = {'stage': 15, 'source_root': 'outputs/decoder14', 'models': MODELS,
                'source_delay_frames': 13, 'source_frames': 1000, 'leading_mask_frames': 26,
                'auxiliary_pitch_threshold': .5, 'nearby_onset_seconds': .2,
                'fragment_minimum_overlap_seconds': .02, 'repeated_note_gap_seconds': .06,
                'interventions': INTERVENTIONS, 'new_optimizer_updates': 0, 'neural_inference_calls': 0,
                'test_inference_allowed': False, 'production_promotion': False,
                'proposed_state_pitch_loss_weight': .5}
    if set(config) != set(expected) | {'next_intervention_rule'} or any(config[key] != value for key, value in expected.items()):
        raise ValueError('The prespecified Stage15 diagnostic protocol changed.')
    if not isinstance(config['next_intervention_rule'], str) or not config['next_intervention_rule']:
        raise ValueError('A declared intervention recommendation rule is required.')


def sources():
    paths = [ROOT / 'configs/diagnosis15.json'] + [ROOT / 'diagnosis15' / name for name in
            ['__init__.py', 'analysis.py', 'loss.py', 'workflow.py']]
    paths += [ROOT / name for name in ['decoder14/core.py', 'decoder14/workflow.py',
              'diagnosis9/oracles.py', 'diagnosis9/head_audit.py', 'streaming/targets.py',
              'streaming/decoder.py', 'acoustic10/decoder.py', 'acoustic10/targets.py',
              'baseline/evaluation.py', 'cnn_gru/model.py', 'cnn_gru/inference.py']]
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths}


def load_inputs(config):
    validate_config(config)
    root = ROOT / config['source_root']
    verification = read(root / 'verification.json')
    for name, digest in verification['artifact_sha256'].items():
        if sha256(root / name) != digest:
            raise ValueError('The verified Stage14 artifact changed: ' + name)
    if not read(root / 'audit.json')['passed']:
        raise ValueError('The previous stage audit did not pass.')
    for name, digest in verification['code_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError('The verified Stage14 code changed: ' + name)
    protocol = read(root / 'protocol.json')
    base, score, inputs, bindings = prior_inputs(protocol['config'])
    for name in ['verification.json', 'audit.json', 'protocol.json', 'selection.json', 'validation/summary.json']:
        bindings[(root / name).relative_to(ROOT).as_posix()] = sha256(root / name)
    selection = read(root / 'selection.json')
    for item in inputs:
        mode = item['model'] + '_selected'
        path = root / 'validation/results' / mode / (item['id'] + '-metrics.json')
        selected = read(path)
        parameters = selection['models'][item['model']]['parameters']
        if parameters != {'onset_threshold': .7, 'activity_threshold': .25, 'onset_backdate_seconds': 0.}:
            raise ValueError('The fixed Stage14 calibrated decoder changed.')
        if selected['parameters'] != parameters or selected['probability_sha256'] != item['previous']['probability_sha256']:
            raise ValueError('The calibrated recording/cache identity changed.')
        item['selected'], item['parameters'] = selected, parameters
        bindings[path.relative_to(ROOT).as_posix()] = sha256(path)
        prefix = ROOT / item['previous']['input_provenance']['prefix']
        for suffix in ['.npz', '.json']:
            bindings[prefix.with_suffix(suffix).relative_to(ROOT).as_posix()] = sha256(prefix.with_suffix(suffix))
        if sha256(prefix.with_suffix('.npz')) != item['previous']['input_provenance']['targets_sha256']:
            raise ValueError('Original source-aligned cached targets changed.')
        with np.load(prefix.with_suffix('.npz'), allow_pickle=False) as arrays:
            targets = {key: arrays[key] for key in arrays.files}
        validate_cached_target_identity(item['reference'], targets, 1200, base, 12.)
        item['targets'] = targets
    return base, score, inputs, bindings


def describe_item(item, config):
    selected = item['selected']
    classified = taxonomy(item['reference'], selected['notes'], item['score'], config)
    arrays = source_arrays(item['probabilities'], item['targets'], config, item['parameters']['activity_threshold'])
    return {'id': item['id'], 'model': item['model'], 'parameters': item['parameters'],
            'taxonomy': add_note_evidence(classified, arrays), 'frames': frame_diagnosis(arrays),
            'notes': selected['notes'], 'reference': selected['reference'],
            'pitch_metrics': selected['pitch_metrics'], 'tab_metrics': selected['tab_metrics'],
            'head_metrics': selected['head_metrics'], 'probability_sha256': selected['probability_sha256'],
            'relativeaudio': 'validation/clips/' + item['id'] + '.wav'}


def aggregate(records, interventions):
    result = {}
    for model in MODELS:
        rows = [row for row in records if row['model'] == model]
        counts = {key: sum(row['taxonomy']['reference_counts'][key] for row in rows) for key in CATEGORIES}
        frame_counts = {key: sum(row['frames']['counts'][key] for row in rows) for key in rows[0]['frames']['counts']}
        value = {'reference_categories': counts, 'frame_counts': frame_counts,
                 'reference_count': sum(row['taxonomy']['reference_count'] for row in rows),
                 'prediction_count': sum(row['taxonomy']['prediction_count'] for row in rows),
                 'extra_predictions': sum(row['taxonomy']['extra_predictions'] for row in rows),
                 'repeated_reference_count': sum(row['taxonomy']['repeated_reference_count'] for row in rows),
                 'repeated_reference_matched': sum(row['taxonomy']['repeated_reference_matched'] for row in rows),
                 'reference_notes_with_multiple_overlaps': sum(row['taxonomy']['reference_notes_with_multiple_overlaps'] for row in rows),
                 'state_derived_pitch_f1': float(np.mean([row['frames']['state_derived_pitch']['f1'] for row in rows])),
                 'auxiliary_pitch_f1': float(np.mean([row['frames']['auxiliary_pitch']['f1'] for row in rows])),
                 'interventions': {variant: summarize([row for row in interventions if row['model'] == model and row['variant'] == variant])
                                   for variant in INTERVENTIONS}}
        for key in ['target', 'predicted']:
            lengths = [length for row in rows for length in row['frames'][key + '_run_lengths']]
            value[key + '_run_p50_frames'] = float(np.median(lengths)) if lengths else None
        result[model] = value
    return result


def recommend(summary, config):
    checks = {}
    for model, row in summary.items():
        scores = row['interventions']
        baseline = scores['predicted']['onset_f1']
        state_gain = scores['oracle_state']['onset_f1'] - baseline
        onset_gain = scores['oracle_onset']['onset_f1'] - baseline
        checks[model] = {'oracle_state_gain': state_gain, 'oracle_onset_gain': onset_gain,
                         'state_gain_exceeds_onset_gain': state_gain > onset_gain,
                         'auxiliary_pitch_beats_state_union': row['auxiliary_pitch_f1'] > row['state_derived_pitch_f1']}
    supported = all(row['state_gain_exceeds_onset_gain'] and row['auxiliary_pitch_beats_state_union'] for row in checks.values())
    return {'rule': config['next_intervention_rule'], 'support_checks': checks,
            'recommendation': 'state_pitch_auxiliary_supervision' if supported else 'collect_more_diagnostic_evidence',
            'supported_by_declared_rule': supported, 'training_executed': False,
            'coefficient': config['proposed_state_pitch_loss_weight'],
            'scope': 'Hypothesis for a future controlled comparison; this diagnostic does not show that the proposed loss improves accuracy.'}


def write_csv(path, taxonomy_result):
    headers = ['reference_index', 'start', 'end', 'pitch', 'string', 'fret', 'category',
               'prediction_index', 'onset_error_ms', 'repeated_note', 'multiple_overlaps',
               'valid_source_frames', 'same_string_fret_fraction', 'any_string_pitch_fraction',
               'auxiliary_pitch_fraction', 'longest_correct_state_run_frames']
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        for row in taxonomy_result['reference_rows']:
            flattened = {**row, **row['note'], **row['evidence']}
            writer.writerow({key: flattened.get(key) for key in headers})


def run(config, output):
    validate_config(config)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'protocol.json').exists():
        raise ValueError('Diagnosis already started here; use report or audit on the saved output.')
    base, score, inputs, bindings = load_inputs(config)
    protocol = {'config': config, 'source_sha256': sources(), 'input_sha256': bindings,
                'decoder_config': base, 'scoring_config': score,
                'interventions': INTERVENTIONS, 'new_optimizer_updates': 0, 'neural_inference_calls': 0,
                'test_inference_calls': 0, 'production_promoted': False,
                'category_rule': 'Official one-to-one 50ms pitch matches, then remaining one-to-one 200ms same-pitch, 50ms octave, 50ms other-pitch associations; residual reference misses and prediction extras. These are associations, not causes.',
                'head_rule': 'Source p13:1013 against original targets0:1000, first26 source frames excluded. State pitch uses fixed selected activity odds0.25; auxiliary pitch uses0.5.',
                'note_rule': 'Onsets in original0:10s, following input through12s; note metrics do not use the frame leading mask. Repetition and multiple-overlap tags are separate from the exclusive association partition.'}
    save_json(output / 'protocol.json', protocol)
    records, interventions = [], []
    for item in inputs:
        item['score'] = score
        row = describe_item(item, config)
        directory = output / 'validation/results' / item['model']
        path = directory / (item['id'] + '.json')
        save_json(path, row)
        write_csv(path.with_suffix('.csv'), row['taxonomy'])
        audio = output / row['relativeaudio']
        audio.parent.mkdir(parents=True, exist_ok=True)
        if not audio.exists():
            shutil.copyfile(item['audio'], audio)
        records.append(row)
        oracle, conflicts = delayed_oracle(item['reference'], 1200, base, 12., 13)
        for variant, heads in [('predicted', []), ('oracle_state', ['state']),
                               ('oracle_onset', ['onset']), ('oracle_both', ['state', 'onset'])]:
            probabilities = replace_heads(item['probabilities'], oracle, heads)
            notes, events, clamps = replay(probabilities, base, item['parameters'], 12.)
            case = score_case(item['reference'], notes, score, 12.)
            if variant == 'predicted':
                if notes != row['notes'] or events != item['selected']['events'] or any(case[key] != row[key] for key in case):
                    raise ValueError('Stage14 calibrated source notes/events/metrics no longer reproduce exactly.')
            interventions.append({'id': item['id'], 'model': item['model'], 'variant': variant,
                                  'oracle_heads': heads, 'privileged_labels': bool(heads), 'conflicts': conflicts,
                                  'notes': notes, 'events': events, 'source_clamps': clamps, **case})
        print(item['model'], item['id'], 'matches', row['taxonomy']['matched_onsets'],
              'state/aux pitch F1', round(row['frames']['state_derived_pitch']['f1'], 3),
              round(row['frames']['auxiliary_pitch']['f1'], 3), flush=True)
    save_json(output / 'interventions.json', interventions)
    summary = aggregate(records, interventions)
    save_json(output / 'summary.json', summary)
    recommendation = recommend(summary, config)
    save_json(output / 'recommendation.json', recommendation)
    save_json(output / 'next-experiment.json', {'status': 'proposed_not_trained',
        'recommendation': recommendation['recommendation'], 'supported': recommendation['supported_by_declared_rule'],
        'parent_checkpoint': 'outputs/optimization13/models/group8/final.pt',
        'parent_lifetime_optimizer_updates': 500, 'parent_checkpoint_sha256': sha256(ROOT / 'outputs/optimization13/models/group8/final.pt'),
        'arms': {'control': {'new_updates': 250, 'loss': 'unchanged legacy multi-head loss'},
                 'state_pitch': {'new_updates': 250, 'loss': 'legacy total plus0.5 times direct state-derived pitch BCE'}},
        'future_total_new_optimizer_update_cap': 500, 'future_new_epochs_per_arm': 5,
        'future_lifetime_updates_per_arm': 750, 'learning_rate': .0001,
        'proposal_module': 'diagnosis15/loss.py', 'proposal_source_sha256': sha256(ROOT / 'diagnosis15/loss.py'),
        'controls': 'Both arms restore the same frozen model/AdamW/scaler/RNG parent, use the same new epoch11-15 crop/gain descriptors and fixed Stage14 decoder; no new architecture parameters.',
        'new_term': 'Softmax state probabilities map through standard tuning to49 MIDI occupancies; independent-string union collapses unisons. BCE targets the original delayed pitch labels; crop/leading masks and any-string-conflict exclusion apply to this new term.',
        'scope': 'A reviewable next-stage proposal and tested loss module. No acoustic training or benefit claim in Stage15.'})
    if sources() != protocol['source_sha256'] or any(sha256(ROOT / name) != digest for name, digest in bindings.items()):
        raise ValueError('An immutable source/input changed during Stage15.')
    return summary


def load_records(output):
    return [read(path) for path in sorted((Path(output) / 'validation/results').glob('*/*.json'))]
