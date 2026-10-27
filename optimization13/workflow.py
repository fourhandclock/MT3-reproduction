"""Four-view continuation comparison: immutable parents and final step500 models."""

import gc
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from acoustic10.workflow import pcm_replay
from baseline.download import ROOT, sha256
from baseline.evaluation import evaluate, reference_notes
from cnn_gru.data import load_audio
from cnn_gru.inference import tablature_metrics
from normalization12.model import CausalCNNGRU
from spectral11.workflow import aligned_heads
from streaming.data import fingerprint
from streaming.evaluate import aggregate, save_json
from streaming.export import export
from streaming.latency import measure
from .data import load_inputs, validate_config
from .training import producer_hashes, train_continuation, verified_sanity


def run(config, output, device):
    validate_config(config)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'protocol.json').exists():
        raise ValueError('Continuation already started; reuse saved report/audit/evaluate without new updates.')
    base, audit, datasets, provenance, parents, schedule = load_inputs(config, output)
    sanity = verified_sanity(config)
    protocol = {'config': config, 'source_sha256': producer_hashes(), 'total_new_update_cap': 500,
                'planned_new_optimizer_updates': {'group8': 250, 'group1': 250},
                'historical_parent_updates': 500, 'parents': provenance['parents'], 'new_sanity_updates': 0,
                'sanity_reference': sanity, 'selection_rule': config['selection_rule'],
                'test_inference_allowed': False, 'exact_uninterrupted_resume': False}
    save_json(output / 'protocol.json', protocol)
    save_json(output / 'crop-schedule.json', {'descriptors': schedule, 'sha256': fingerprint(schedule)})
    for name, reference in provenance['parents'].items():
        save_json(output / 'models' / name / 'parent-reference.json', reference)
    receipts = {}
    for arm in config['arms']:
        name = arm['name']
        receipts[name] = train_continuation(arm, config, base, audit, datasets['train'], provenance,
                                           parents[name], schedule, output, device)
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    if sum(row['new_optimizer_steps'] for row in receipts.values()) != 500:
        raise ValueError('The actual stage new-update budget does not equal its fixed500 plan.')
    summary = {'new_optimizer_updates': 500, 'historical_parent_updates': 500,
               'final_lifetime_updates_per_model': 500, 'new_sanity_updates': 0,
               'additional_epochs_per_arm': 5, 'lifetime_epoch_blocks_per_model': 10,
               'arms': receipts, 'parents': provenance['parents'], 'sanity_reference': sanity,
               'test_inference_calls': 0, 'validation_used_during_training': False,
               'config': config, 'source_sha256': producer_hashes(),
               'protocol_sha256': sha256(output / 'protocol.json'), 'input_audit_sha256': sha256(output / 'input-audit.json')}
    save_json(output / 'training-summary.json', summary)
    del datasets
    gc.collect()
    return evaluate_validation(config, output, device)


def without_delivery(value):
    if isinstance(value, dict):
        return {key: without_delivery(item) for key, item in value.items()
                if key not in ['on_delivered_at_seconds', 'off_delivered_at_seconds']}
    if isinstance(value, list):
        return [without_delivery(item) for item in value]
    return value


def check_parent_replay(name, clip_id, probability, notes, heads, metrics, tabs):
    root = ROOT / 'outputs/normalization12'
    path = root / 'validation/results' / name / (clip_id + '-metrics.json')
    previous = json.loads(path.read_text(encoding='utf-8'))
    probability_path = root / previous['probability_path']
    if sha256(probability_path) != previous['probability_sha256']:
        raise ValueError('Frozen parent probability cache changed.')
    exact, maximum = True, 0.
    with np.load(probability_path, allow_pickle=False) as arrays:
        if set(arrays.files) != set(probability):
            raise ValueError('Parent replay heads differ from their recorded interface.')
        for key in arrays.files:
            if arrays[key].shape != probability[key].shape:
                raise ValueError('Parent replay feature grid changed.')
            difference = float(np.max(np.abs(arrays[key] - probability[key])))
            exact = exact and np.array_equal(arrays[key], probability[key])
            maximum = max(maximum, difference)
            if difference > 1e-6:
                raise ValueError('Parent neural replay differs beyond the prescribed numerical tolerance.')
    if (without_delivery(notes) != without_delivery(previous['notes'])
            or without_delivery(metrics) != without_delivery(previous['pitch_metrics'])
            or without_delivery(tabs) != without_delivery(previous['tab_metrics'])):
        raise ValueError('Parent source notes/decisions changed; only measured delivery may differ.')
    for key in ['exact_state', 'active_fret_accuracy', 'pitch']:
        if heads[key] != previous['head_metrics'][key]:
            raise ValueError('Parent head decisions changed.')
    return {'probabilities_bit_identical': bool(exact), 'max_probability_absolute_difference': maximum,
            'source_notes_identical': True, 'source_scores_identical': True, 'head_decisions_identical': True,
            'historical_metrics_path': str(path.relative_to(ROOT)).replace('\\', '/'),
            'historical_metrics_sha256': sha256(path), 'historical_probability_sha256': previous['probability_sha256']}


def evaluate_validation(config, output, device):
    output = Path(output).resolve()
    protocol = json.loads((output / 'protocol.json').read_text(encoding='utf-8'))
    if protocol['config'] != config or protocol['source_sha256'] != producer_hashes():
        raise ValueError('The frozen continuation source/config identity changed.')
    base, audit, datasets, provenance, parents, _ = load_inputs(config, output)
    if protocol['parents'] != provenance['parents']:
        raise ValueError('The original parent checkpoint identity changed.')
    paths, checkpoints, networks = {}, {}, {}
    for arm in config['arms']:
        name = arm['name']
        for lifetime in [250, 500]:
            mode = name + '_' + str(lifetime)
            path = ROOT / arm['parent_checkpoint'] if lifetime == 250 else output / 'models' / name / 'final.pt'
            checkpoint = parents[name] if lifetime == 250 else torch.load(path, map_location='cpu', weights_only=True)
            if checkpoint['step'] != lifetime:
                raise ValueError('Checkpoint lifetime count differs from its evaluation mode.')
            if lifetime == 500 and (checkpoint['receipt']['source_sha256'] != producer_hashes()
                    or checkpoint['receipt']['config_sha256'] != fingerprint(config)
                    or checkpoint['receipt']['input_audit_sha256'] != sha256(output / 'input-audit.json')):
                raise ValueError('Continuation checkpoint differs from its fixed inputs/production recipe.')
            model = CausalCNNGRU(checkpoint['config']).to(device)
            model.load_state_dict(checkpoint['model'])
            model.eval()
            networks[mode], paths[mode], checkpoints[mode] = model, path, checkpoint
    score = json.loads((ROOT / 'configs/stage1.json').read_text(encoding='utf-8'))
    score['selection_role'] = 'Stage13 continuation-dose development comparison; only final500 models select, no test inference or promotion.'
    directory = output / 'validation'
    (directory / 'clips').mkdir(parents=True, exist_ok=True)
    warm, _ = load_audio(ROOT / datasets['validation'][0]['entry']['audio'], 1, base)
    for mode in networks:
        pcm_replay(networks[mode], warm, checkpoints[mode], config['decoder_parameters'], device, 20, .13)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    modes = ['group8_250', 'group8_500', 'group1_250', 'group1_500']
    records = []
    for index, input_record in enumerate(datasets['validation']):
        entry = input_record['entry']
        audio, duration = load_audio(ROOT / entry['audio'], 12, base)
        relative = Path('validation/clips') / (entry['id'] + '.wav')
        sf.write(output / relative, audio, 16000, subtype='FLOAT')
        reference = reference_notes(ROOT / entry['annotation'])
        # Rotate the first mode across excerpts to reduce fixed timing-order effects.
        order = modes[index % 4:] + modes[:index % 4]
        for mode in order:
            name, lifetime_text = mode.rsplit('_', 1)
            lifetime = int(lifetime_text)
            notes, events, timing, probability = pcm_replay(networks[mode], audio, checkpoints[mode],
                                               config['decoder_parameters'], device, 20, .13, capture=True)
            timing.update(window_seconds=.128, hann_peak_age_seconds=.064, fft_bin_spacing_hz=7.8125,
                          cnn_norm_groups=8 if name == 'group8' else 1,
                          lifetime_optimizer_steps=lifetime, parent_reused=lifetime == 250)
            probability_path = directory / 'probabilities' / mode / (entry['id'] + '.npz')
            probability_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(probability_path, **probability)
            heads = aligned_heads(probability, input_record['targets'], config, base, 1000)
            metrics = evaluate(reference, notes, score, duration)
            tabs = tablature_metrics(reference, notes, score, 10, duration)
            parent_check = check_parent_replay(name, entry['id'], probability, notes, heads, metrics, tabs) if lifetime == 250 else None
            row = {'id': entry['id'], 'mode': mode, 'notes': notes, 'reference': reference, 'events': events,
                   'head_metrics': heads, 'pitch_metrics': metrics, 'tab_metrics': tabs,
                   'latency': measure(reference, notes, score, duration), 'timing': timing,
                   'relativeaudio': relative.as_posix(), 'input_provenance': provenance['inputs']['validation'][index],
                   'probability_path': str(probability_path.relative_to(output)).replace('\\', '/'),
                   'probability_sha256': sha256(probability_path), 'parent_reproduction': parent_check}
            stem = directory / 'results' / mode / entry['id']
            export(notes, stem, {'mode': mode, 'checkpoint_sha256': sha256(paths[mode]),
                                'decoder_parameters': config['decoder_parameters'], 'source_delay_seconds': .13,
                                'cnn_norm_groups': 8 if name == 'group8' else 1, 'lifetime_optimizer_steps': lifetime})
            save_json(stem.with_name(stem.name + '-metrics.json'), row)
            save_json(stem.with_name(stem.name + '-events.json'), {'mode': mode, 'events': events})
            records.append(row)
            print('Validation', mode, entry['id'], 'state F1', round(heads['exact_state_f1'], 3),
                  'note onset F1', round(metrics['onset']['f1'], 3), flush=True)
    summary = aggregate(records)
    for mode in modes:
        rows = [row for row in records if row['mode'] == mode]
        summary['modes'][mode]['exact_state_f1'] = float(np.mean([row['head_metrics']['exact_state_f1'] for row in rows]))
        summary['modes'][mode]['active_fret_accuracy'] = float(np.mean([row['head_metrics']['active_fret_accuracy'] for row in rows]))
        for field, key in [('auxiliary_pitch_f1', 'overall'), ('low_register_pitch_f1', 'low_register')]:
            summary['modes'][mode][field] = float(np.mean([row['head_metrics']['pitch'][key]['f1'] for row in rows]))
    finals = ['group8_500', 'group1_500']
    winner = max(finals, key=lambda mode: (summary['modes'][mode]['exact_state_f1'], summary['modes'][mode]['onset_f1']))
    selection = {'selected_arm': winner, 'rule': config['selection_rule'], 'eligible_modes': finals,
                 'scores': {mode: {key: summary['modes'][mode][key] for key in ['exact_state_f1', 'onset_f1']} for mode in finals},
                 'checkpoint_sha256': {mode: sha256(path) for mode, path in paths.items()},
                 'selected_from_test': False, 'production_promoted': False,
                 'scope': 'Final500-only development selection; parents are within-family dose references, not eligible winners.'}
    save_json(output / 'selection.json', selection)
    save_json(directory / 'summary.json', summary)
    training = json.loads((output / 'training-summary.json').read_text(encoding='utf-8'))
    run = {'config': config, 'scoring_config': score, 'selection': selection,
           'checkpoint_sha256': selection['checkpoint_sha256'],
           'checkpoint_paths': {mode: str(path.relative_to(ROOT)).replace('\\', '/') for mode, path in paths.items()},
           'source_sha256': producer_hashes(), 'new_optimizer_updates': 500, 'historical_parent_updates': 500,
           'new_sanity_updates': 0, 'test_used': False, 'test_inference_calls': 0,
           'training_summary': training, 'input_provenance': provenance, 'parents': provenance['parents'],
           'sanity_reference': protocol['sanity_reference'], 'input_audit_sha256': sha256(output / 'input-audit.json'),
           'protocol_sha256': sha256(output / 'protocol.json'),
           'peak_gpu_mib': torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None,
           'scope': 'Validation-only continuation-dose protocol with final500-only selection; no exact historical RNG resume or product promotion.'}
    save_json(directory / 'run.json', run)
    from .report import write_report
    write_report(output, records, summary, run)
    return summary
