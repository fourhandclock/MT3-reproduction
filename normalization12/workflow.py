"""Within-frame normalization comparison, reusing an exact historical control."""

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
from spectral11.workflow import aligned_heads
from streaming.data import fingerprint
from streaming.evaluate import aggregate, save_json
from streaming.export import export
from streaming.latency import measure
from .data import load_inputs, validate_config
from .model import CausalCNNGRU
from .training import producer_hashes, train_treatment, verified_sanity


def run(config, output, device):
    validate_config(config)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'protocol.json').exists():
        raise ValueError('Stage already started; report/evaluate/audit reuse saved weights without new training.')
    base, audit, datasets, provenance, control, schedule = load_inputs(config, output)
    sanity = verified_sanity(config)
    protocol = {'config': config, 'source_sha256': producer_hashes(), 'total_new_update_cap': 500,
                'planned_new_optimizer_updates': {'group8': 0, 'group1': 250},
                'historical_optimizer_updates': {'group8': 250}, 'new_sanity_updates': 0,
                'historical_control': provenance['control_reference'], 'sanity_reference': sanity,
                'selection_rule': config['selection_rule'], 'test_inference_allowed': False}
    save_json(output / 'protocol.json', protocol)
    save_json(output / 'models/group8/reference.json', provenance['control_reference'])
    save_json(output / 'crop-schedule.json', {'descriptors': schedule, 'sha256': fingerprint(schedule)})
    treatment = train_treatment(config, base, audit, datasets['train'], provenance, schedule, output, device)
    summary = {'new_optimizer_updates': treatment['optimizer_steps'], 'reused_optimizer_updates': 250,
               'new_sanity_updates': 0, 'epochs_per_arm': 5,
               'control_reused': True, 'control_reference': provenance['control_reference'],
               'arms': {'group8': control['receipt'], 'group1': treatment},
               'test_inference_calls': 0, 'validation_used_during_training': False,
               'config': config, 'source_sha256': producer_hashes(), 'sanity_reference': sanity,
               'protocol_sha256': sha256(output / 'protocol.json'),
               'input_audit_sha256': sha256(output / 'input-audit.json')}
    save_json(output / 'training-summary.json', summary)
    del datasets
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return evaluate_validation(config, output, device)


def check_control_replay(clip_id, probability, notes, heads, pitch_metrics, tab_metrics):
    """Require the new control path to reproduce the frozen original behavior."""
    stem = ROOT / 'outputs/spectral11/validation/results/short128' / clip_id
    previous = json.loads(stem.with_name(stem.name + '-metrics.json').read_text(encoding='utf-8'))
    probability_path = ROOT / 'outputs/spectral11' / previous['probability_path']
    if sha256(probability_path) != previous['probability_sha256']:
        raise ValueError('Frozen historical control probabilities changed.')
    exact, maximum = True, 0.
    with np.load(probability_path, allow_pickle=False) as arrays:
        if set(probability) != set(arrays.files):
            raise ValueError('Historical probability heads changed.')
        for key in arrays.files:
            if probability[key].shape != arrays[key].shape:
                raise ValueError('Control probability grid differs from the historical run.')
            exact = exact and np.array_equal(probability[key], arrays[key])
            difference = float(np.max(np.abs(probability[key] - arrays[key])))
            maximum = max(maximum, difference)
            if difference > 1e-6:
                raise ValueError('Current control probabilities do not reproduce the historical model.')
    def source_only(items):
        return [{key: value for key, value in note.items()
                 if key not in ['on_delivered_at_seconds', 'off_delivered_at_seconds']} for note in items]
    def without_measured_delivery(value):
        if isinstance(value, dict):
            return {key: without_measured_delivery(item) for key, item in value.items()
                    if key not in ['on_delivered_at_seconds', 'off_delivered_at_seconds']}
        if isinstance(value, list):
            return [without_measured_delivery(item) for item in value]
        return value
    if source_only(notes) != source_only(previous['notes']):
        raise ValueError('Current control source notes differ from the historical model.')
    if (without_measured_delivery(pitch_metrics) != without_measured_delivery(previous['pitch_metrics'])
            or without_measured_delivery(tab_metrics) != without_measured_delivery(previous['tab_metrics'])):
        raise ValueError('Current control note scores differ from the historical model.')
    # Loss can reflect insignificant floating-point differences, but decisions must agree.
    for key in ['exact_state', 'active_fret_accuracy', 'pitch']:
        if heads[key] != previous['head_metrics'][key]:
            raise ValueError('Current control acoustic decisions differ from historical scores.')
    return {'probabilities_bit_identical': bool(exact), 'max_probability_absolute_difference': maximum,
            'source_notes_identical': True, 'note_and_tab_scores_identical': True,
            'head_decisions_identical': True, 'historical_metrics_sha256': sha256(stem.with_name(stem.name + '-metrics.json')),
            'historical_probability_sha256': previous['probability_sha256']}


def evaluate_validation(config, output, device):
    output = Path(output).resolve()
    protocol = json.loads((output / 'protocol.json').read_text(encoding='utf-8'))
    if protocol['config'] != config or protocol['source_sha256'] != producer_hashes():
        raise ValueError('The frozen normalization experiment sources/config changed.')
    base, audit, datasets, provenance, control, _ = load_inputs(config, output)
    if protocol['historical_control'] != provenance['control_reference']:
        raise ValueError('Historical reference changed after treatment training.')
    paths = {'group8': ROOT / config['control_checkpoint'], 'group1': output / 'models/group1/final.pt'}
    treatment = torch.load(paths['group1'], map_location='cpu', weights_only=True)
    if (treatment['step'] != 250 or treatment['receipt']['source_sha256'] != producer_hashes()
            or treatment['receipt']['config_sha256'] != fingerprint(config)
            or treatment['receipt']['input_audit_sha256'] != sha256(output / 'input-audit.json')):
        raise ValueError('Final treatment differs from the saved input/recipe bindings.')
    checkpoints = {'group8': control, 'group1': treatment}
    networks = {}
    for name, checkpoint in checkpoints.items():
        network = CausalCNNGRU(checkpoint['config']).to(device)
        network.load_state_dict(checkpoint['model'])
        network.eval()
        expected = 8 if name == 'group8' else 1
        if any(layer.normalization.num_groups != expected for layer in network.normalizations):
            raise ValueError('Loaded model does not use its declared normalization groups.')
        networks[name] = network
    score = json.loads((ROOT / 'configs/stage1.json').read_text(encoding='utf-8'))
    score['selection_role'] = 'Stage12 validation-only normalization comparison; exact reused control, no test inference or promotion.'
    directory = output / 'validation'
    (directory / 'clips').mkdir(parents=True, exist_ok=True)
    warm, _ = load_audio(ROOT / datasets['validation'][0]['entry']['audio'], 1, base)
    for name in networks:
        pcm_replay(networks[name], warm, checkpoints[name], config['decoder_parameters'], device, 20, .13)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    records = []
    for index, input_record in enumerate(datasets['validation']):
        entry = input_record['entry']
        audio, duration = load_audio(ROOT / entry['audio'], base['evaluation_audio_seconds'], base)
        relative = Path('validation/clips') / (entry['id'] + '.wav')
        sf.write(output / relative, audio, 16000, subtype='FLOAT')
        reference = reference_notes(ROOT / entry['annotation'])
        order = ['group8', 'group1'] if index % 2 == 0 else ['group1', 'group8']
        for name in order:
            notes, events, timing, probability = pcm_replay(networks[name], audio, checkpoints[name],
                                               config['decoder_parameters'], device, 20, .13, capture=True)
            timing.update(window_seconds=.128, hann_peak_age_seconds=.064, fft_bin_spacing_hz=7.8125,
                          cnn_norm_groups=8 if name == 'group8' else 1, control_reused=name == 'group8')
            probability_path = directory / 'probabilities' / name / (entry['id'] + '.npz')
            probability_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(probability_path, **probability)
            heads = aligned_heads(probability, input_record['targets'], config, base, round(score['score_seconds'] * 100))
            metrics = evaluate(reference, notes, score, duration)
            tabs = tablature_metrics(reference, notes, score, score['score_seconds'], duration)
            control_check = check_control_replay(entry['id'], probability, notes, heads, metrics, tabs) if name == 'group8' else None
            row = {'id': entry['id'], 'mode': name, 'notes': notes, 'reference': reference, 'events': events,
                   'head_metrics': heads, 'pitch_metrics': metrics, 'tab_metrics': tabs,
                   'latency': measure(reference, notes, score, duration), 'timing': timing,
                   'relativeaudio': relative.as_posix(), 'input_provenance': provenance['inputs']['validation'][index],
                   'probability_path': str(probability_path.relative_to(output)).replace('\\', '/'),
                   'probability_sha256': sha256(probability_path), 'historical_control_reproduction': control_check}
            stem = directory / 'results' / name / entry['id']
            export(notes, stem, {'mode': name, 'checkpoint_sha256': sha256(paths[name]),
                                'decoder_parameters': config['decoder_parameters'], 'source_delay_seconds': .13,
                                'cnn_norm_groups': 8 if name == 'group8' else 1, 'control_reused': name == 'group8'})
            save_json(stem.with_name(stem.name + '-metrics.json'), row)
            save_json(stem.with_name(stem.name + '-events.json'), {'mode': name, 'events': events})
            records.append(row)
            print('Validation', name, entry['id'], 'state F1', round(heads['exact_state_f1'], 3),
                  'note onset F1', round(metrics['onset']['f1'], 3), flush=True)
    summary = aggregate(records)
    for name in ['group8', 'group1']:
        rows = [row for row in records if row['mode'] == name]
        summary['modes'][name]['exact_state_f1'] = float(np.mean([row['head_metrics']['exact_state_f1'] for row in rows]))
        summary['modes'][name]['active_fret_accuracy'] = float(np.mean([row['head_metrics']['active_fret_accuracy'] for row in rows]))
        for field, key in [('auxiliary_pitch_f1', 'overall'), ('low_register_pitch_f1', 'low_register')]:
            summary['modes'][name][field] = float(np.mean([row['head_metrics']['pitch'][key]['f1'] for row in rows]))
    winner = max(['group8', 'group1'], key=lambda name:
                 (summary['modes'][name]['exact_state_f1'], summary['modes'][name]['onset_f1']))
    selection = {'selected_arm': winner, 'rule': config['selection_rule'],
                 'scores': {name: {key: summary['modes'][name][key] for key in ['exact_state_f1', 'onset_f1']}
                            for name in ['group8', 'group1']},
                 'checkpoint_sha256': {name: sha256(path) for name, path in paths.items()},
                 'selected_from_test': False, 'production_promoted': False,
                 'scope': 'Development comparison on the displayed selection set; no independent-test claim.'}
    save_json(output / 'selection.json', selection)
    save_json(directory / 'summary.json', summary)
    training_summary = json.loads((output / 'training-summary.json').read_text(encoding='utf-8'))
    run = {'config': config, 'scoring_config': score, 'selection': selection,
           'checkpoint_sha256': selection['checkpoint_sha256'],
           'checkpoint_paths': {name: str(path.relative_to(ROOT)).replace('\\', '/') for name, path in paths.items()},
           'source_sha256': producer_hashes(), 'new_optimizer_updates': 250, 'reused_optimizer_updates': 250,
           'new_sanity_updates': 0, 'test_used': False, 'test_inference_calls': 0,
           'training_summary': training_summary, 'input_provenance': provenance,
           'historical_control': provenance['control_reference'], 'sanity_reference': protocol['sanity_reference'],
           'input_audit_sha256': sha256(output / 'input-audit.json'), 'protocol_sha256': sha256(output / 'protocol.json'),
           'peak_gpu_mib': torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None,
           'scope': 'Validation-only normalization group comparison; no product promotion.'}
    save_json(directory / 'run.json', run)
    from .report import write_report
    write_report(output, records, summary, run)
    return summary
