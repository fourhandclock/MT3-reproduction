"""Verify the actual update budget, matched support, restoration and saved replays."""

from pathlib import Path
import numpy as np
import torch
from acoustic10.data import prepare_targets
from baseline.download import ROOT, sha256
from baseline.evaluation import evaluate, reference_notes
from cnn_gru.inference import tablature_metrics
from decoder14.audit import verify_exports
from decoder14.core import replay, without_compute
from decoder14.workflow import read
from headswap18.analysis import state_diagnosis
from headswap18.audit import verify_frame_partition
from optimization13.optimizer import _validate_group, recursive_digest
from spectral11.workflow import aligned_heads
from streaming.data import fingerprint
from streaming.evaluate import save_json
from streaming.latency import measure
from .data import load_inputs, validate_config
from .evaluation import MODES, check_parent, choose, fret_confusion, frame_config, summarize
from .loss import class_weights
from .training import producer_hashes


def require(value, message):
    if not value:
        raise ValueError(message)


def audit(config, output):
    validate_config(config)
    output = Path(output)
    protocol = read(output / 'protocol.json')
    require(protocol['config'] == config and protocol['source_sha256'] == producer_hashes(), 'Actual training source changed')
    require(protocol['total_new_optimizer_update_cap'] == 450
            and protocol['planned_new_optimizer_updates'] == {arm['name']: 150 for arm in config['arms']}
            and protocol['historical_unique_parent_updates'] == 500 and protocol['new_sanity_updates'] == 0
            and not protocol['test_inference_allowed'] and not protocol['production_promotion'], 'Protocol budget/scope differs')
    base, cache, datasets, provenance, parent, schedule, weights = load_inputs(config, output)
    saved_schedule = read(output / 'crop-schedule.json')
    require(saved_schedule['descriptors'] == schedule and saved_schedule['sha256'] == fingerprint(schedule)
            and len(schedule) == 600 and {r['epoch'] for r in schedule} == {11, 12, 13}, 'Matched crop schedule changed')
    unique = np.asarray(read(ROOT / 'outputs/coverage20/coverage.json')['histories']['epochs1_10']['unique'])
    available = np.asarray(read(ROOT / 'outputs/coverage20/coverage.json')['histories']['epochs1_10']['available'])
    expected_weights = np.zeros((6, 25), np.float32)
    for string in range(6):
        for fret in range(25):
            n = int(unique[string, fret])
            expected_weights[string, fret] = min(4., (int(unique[string].max()) / n) ** .5) if n else 4. if available[string, fret] else 0.
    require(np.array_equal(weights, expected_weights), 'Independent training-only class weights differ')
    weight_record = read(output / 'class-weights.json')
    require(weight_record['pre_training_amendment']['optimizer_updates_before_amendment'] == 0
            and not weight_record['pre_training_amendment']['validation_used_for_amendment']
            and weight_record['unexposed_available_positions'] == [[4, 17]], 'Recorded pre-training amendment differs')
    targets = prepare_targets(datasets['train'], 13, config, base)
    support = [np.zeros((6, 25), np.int64) for _ in range(3)]
    for row in schedule:
        start, length = row['start_frame'], row['frames']
        values = targets[row['record_index']]
        state = values['state'][start:start + length]
        mask = values['mask'][start:start + length].copy()
        mask[:26] = mask[-26:] = 0
        for string in range(6):
            valid = (mask > 0) & (state[:, string] >= 0) & (state[:, string] < 25)
            source = np.flatnonzero(valid) + start - 13
            require(np.all(source >= 0) and np.array_equal(state[valid, string],
                datasets['train'][row['record_index']]['targets']['state'][source, string]), 'Delayed source-frame identity differs')
            support[row['epoch'] - 11][string] += np.bincount(state[valid, string], minlength=25)
    require(sum(int(value[4, 17]) for value in support) == 76, 'Amended training-only support changed')
    training = read(output / 'training-summary.json')
    require(training['new_optimizer_updates'] == 450 and training['historical_unique_parent_updates'] == 500
            and training['final_lifetime_updates_per_arm'] == 650 and training['new_sanity_updates'] == 0
            and not training['validation_used_during_training'] and training['test_inference_calls'] == 0, 'Training accounting differs')
    require(training['source_sha256'] == producer_hashes() and training['protocol_sha256'] == sha256(output / 'protocol.json'), 'Training provenance changed')
    parent_states = {key: recursive_digest(parent[key]) for key in ('model', 'optimizer', 'scaler', 'rng_cpu', 'rng_cuda')}
    final_rng, paths = [], {'parent500': ROOT / config['parent_checkpoint']}
    for arm in config['arms']:
        directory = output / 'models' / arm['name']
        snapshot = torch.load(directory / 'resume-state.pt', map_location='cpu', weights_only=True)
        checkpoint = torch.load(directory / 'final.pt', map_location='cpu', weights_only=True)
        receipt = read(directory / 'training-receipt.json')
        require(receipt == checkpoint['receipt'] == training['arms'][arm['name']], 'Receipt/final/training ledger differs')
        require(snapshot['step'] == 500 and all(recursive_digest(snapshot[k]) == v for k, v in parent_states.items()), 'Full pre-update state differs from control500')
        require(snapshot['config'] == parent['config'] and snapshot['normalization'] == parent['normalization']
                and snapshot['positive_weights'] == parent['positive_weights'] and snapshot['source_scheduler'] == parent['source_scheduler'], 'Pre-update metadata differs')
        proof = receipt['restoration_proof']
        require(proof == snapshot['restoration_proof'] and proof['parent_step'] == 500 and proof['new_optimizer_updates'] == 0
                and proof['parent_state_sha256'] == proof['restored_state_sha256'] == parent_states, 'Restoration proof differs')
        require(receipt['new_optimizer_steps'] == 150 and receipt['parent_optimizer_steps'] == 500
                and receipt['optimizer_steps'] == checkpoint['step'] == 650 and receipt['selected_step'] == 650, 'Actual update count differs')
        require(receipt['active_fret_ce_coefficient'] == arm['active_fret_ce_coefficient'] and receipt['weighting'] == arm['weighting'], 'Loss arm changed')
        _validate_group(checkpoint['optimizer'], 34, .0001)
        require(receipt['actual_optimizer_step_counters'] == {'minimum': 650, 'maximum': 650, 'parameter_states': 34}, 'Recorded final counters differ')
        for index, (name, value) in enumerate(checkpoint['model'].items()):
            require(value.shape == parent['model'][name].shape and value.dtype == parent['model'][name].dtype
                    and torch.isfinite(value).all(), 'Invalid final model tensor')
            state = checkpoint['optimizer']['state'][index]
            require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'} and state['step'].dtype == torch.float32
                    and state['step'].shape == torch.Size([]) and state['step'].item() == 650, 'Actual final AdamW counter differs')
            for field in ('exp_avg', 'exp_avg_sq'):
                require(state[field].shape == value.shape and state[field].dtype == value.dtype
                        and torch.isfinite(state[field]).all(), 'Invalid final AdamW moment')
            require((state['exp_avg_sq'] >= 0).all(), 'Negative second moment')
        require(checkpoint['scaler'] == {**parent['scaler'], '_growth_tracker': parent['scaler']['_growth_tracker'] + 150}, 'AMP history differs')
        require(not checkpoint['scheduler_used'] and checkpoint['source_scheduler'] == parent['source_scheduler']
                and not receipt['scheduler_used'] and not receipt['early_stopping_used']
                and not receipt['validation_used_during_training'] and receipt['test_inference_calls'] == 0, 'Scheduling/selection scope changed')
        require(receipt['source_sha256'] == producer_hashes() and receipt['config_sha256'] == fingerprint(config)
                and receipt['input_audit_sha256'] == sha256(output / 'input-audit.json')
                and receipt['class_weights_sha256'] == sha256(output / 'class-weights.json')
                and receipt['resume_state_sha256'] == sha256(directory / 'resume-state.pt'), 'Source/input/snapshot binding differs')
        require(receipt['crop_schedule_sha256'] == fingerprint(schedule)
                and receipt['gain_schedule_sha256'] == fingerprint([row['gain_log'] for row in schedule]), 'Matched crop/gain history differs')
        require(receipt['model_parameters'] == 931361 and checkpoint['config'] == receipt['model_config']
                and checkpoint['config']['feature_planes'] == [2048, 2048]
                and checkpoint['normalization'] == parent['normalization'] and checkpoint['positive_weights'] == parent['positive_weights'], 'Capacity/frontend/balancing changed')
        history = receipt['history']
        require([r['data_epoch'] for r in history] == [11, 12, 13]
                and [r['new_optimizer_steps'] for r in history] == [50, 100, 150]
                and [r['optimizer_steps'] for r in history] == [550, 600, 650], 'Epoch ledger differs')
        for row, expected in zip(history, support):
            require(row['active_class_presentations'] == expected.tolist() and row['valid_active_string_frames'] == int(expected.sum()), 'Actual supervised class presentations differ')
            mass = float((expected * weights).sum()) if arm['weighting'] == 'clipped_inverse_sqrt' else float(expected.sum())
            require(abs(row['class_weighted_target_mass'] - mass) < 1e-8, 'Weighted target mass differs')
            require(row['actual_optimizer_step_counters'] == {'minimum': row['optimizer_steps'], 'maximum': row['optimizer_steps'], 'parameter_states': 34}
                    and row['learning_rate'] == .0001, 'Block counters/rate differ')
            loss = row['training_losses']
            total = loss['state'] + loss['onset'] + .5 * loss['offset'] + .5 * loss['pitch']
            if arm['active_fret_ce_coefficient']: total += .5 * loss['active_fret_ce']
            require(('active_fret_ce' in loss) == bool(arm['active_fret_ce_coefficient']) and abs(total - loss['total']) < 1e-5, 'Loss decomposition differs')
        final_rng.append({k: recursive_digest(checkpoint[k]) for k in ('rng_cpu', 'rng_cuda')})
        paths[arm['name'] + '650'] = directory / 'final.pt'
    require(final_rng[0] == final_rng[1] == final_rng[2], 'Paired final random streams differ')
    run = read(output / 'validation/run.json')
    expected_sources = {**producer_hashes(), 'balance22/evaluation.py': sha256(ROOT / 'balance22/evaluation.py')}
    require(run['checkpoint_sha256'] == {m: sha256(p) for m, p in paths.items()}
            and run['source_sha256'] == expected_sources and run['test_inference_calls'] == 0 and not run['production_promoted'], 'Validation source/model scope differs')
    score = run['scoring_config']
    records = [read(p) for p in sorted((output / 'validation/results').glob('*/*-metrics.json'))]
    lookup = {(r['id'], r['mode']): r for r in records}
    require(len(records) == len(lookup) == 48, 'Validation case count differs')
    ordered, note_count, event_count, bit_identical = [], 0, 0, 0
    for index, item in enumerate(datasets['validation']):
        entry = item['entry']
        reference = reference_notes(ROOT / entry['annotation'])
        for mode in MODES[index % 4:] + MODES[:index % 4]:
            row = lookup[(entry['id'], mode)]
            path = output / row['probability_path']
            require(sha256(path) == row['probability_sha256'], 'Cached probability recording changed')
            with np.load(path, allow_pickle=False) as arrays: probability = {key: arrays[key] for key in arrays.files}
            notes, events, _ = replay(probability, base, config['decoder_parameters'], 12.)
            require(without_compute(row['notes']) == notes and without_compute(row['events']) == events, 'Exact cached source replay differs')
            require(all(e['delivered_at_seconds'] + 1e-9 >= e['emitted_at_audio_seconds'] for e in row['events']), 'Completion precedes availability')
            heads = aligned_heads(probability, item['targets'], config, base, 1000)
            diagnosis, _ = state_diagnosis(probability, item['targets'], frame_config(config))
            fret = fret_confusion(probability, item['targets'])
            require(row['head_metrics'] == heads and row['frame_diagnosis'] == diagnosis
                    and row['state_derived_pitch'] == diagnosis['state_derived_pitch'] and row['best_active_confusion'] == fret, 'Saved head/frame/fret metrics differ')
            verify_frame_partition(diagnosis)
            require(row['pitch_metrics'] == evaluate(reference, row['notes'], score, 12.)
                    and row['tab_metrics'] == tablature_metrics(reference, row['notes'], score, 10, 12.)
                    and row['latency'] == measure(reference, row['notes'], score, 12.), 'Saved note/string/completion metrics differ')
            if mode == 'parent500':
                proof = check_parent(item, probability, row['notes'], row['events'], heads, diagnosis, row['pitch_metrics'], row['tab_metrics'], fret)
                require(proof == row['parent_reproduction'], 'Parent reproduction proof differs')
                bit_identical += int(proof['probabilities_bit_identical'])
            stem = output / 'validation/results' / mode / entry['id']
            verify_exports(stem, row)
            require(len(row['events']) == 2 * len(row['notes']), 'Event ledger lost a boundary')
            require(row['timing']['lookahead_samples'] == 0 and row['timing']['feature_windows_samples'] == [2048, 2048]
                    and row['timing']['unique_frontend_fft_count'] == 1, 'Frontend timing/capacity differs')
            ordered.append(row)
            note_count += len(row['notes'])
            event_count += len(row['events'])
    summary = summarize(ordered)
    require(summary == read(output / 'validation/summary.json'), 'Summary aggregation differs')
    selection = read(output / 'selection.json')
    winner = choose({m: summary['modes'][m] for m in MODES[1:]})
    require(selection == run['selection'] and selection['selected_arm'] == winner and selection['eligible_modes'] == MODES[1:]
            and not selection['selected_from_test'] and not selection['production_promoted'], 'Fixed-final selection differs')
    result = {'passed': True, 'new_optimizer_updates': 450, 'historical_unique_parent_updates': 500,
        'final_lifetime_updates_per_arm': 650, 'new_sanity_updates': 0, 'validation_cases': 48,
        'parent_reproductions': 12, 'parent_probability_sets_bit_identical': bit_identical,
        'exact_notes': note_count, 'exact_events': event_count, 'paired_final_rng_identical': True,
        'test_inference_calls': 0, 'new_optimizer_updates_for_audit': 0, 'neural_inference_calls_for_audit': 0,
        'active_supervised_support_per_epoch': [int(value.sum()) for value in support],
        'pre_training_amendment': weight_record['pre_training_amendment'], 'amended_class_presentations': 76,
        'selected_arm': winner, 'production_promoted': False,
        'artifact_sha256': {name: sha256(output / name) for name in ['protocol.json', 'input-audit.json',
            'input-bindings.json', 'class-weights.json', 'crop-schedule.json', 'training-summary.json',
            'selection.json', 'validation/run.json', 'validation/summary.json']}}
    save_json(output / 'audit.json', result)
    print('Stage22 audit passed: 450 actual updates, matched exposure, complete restoration and 48 saved replays.', flush=True)
    return result
