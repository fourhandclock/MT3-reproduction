"""Matched window-length trial with fixed supervision/source clocks."""

import gc
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from acoustic10.data import build_schedule
from acoustic10.training import head_metrics, tensor_digest
from acoustic10.workflow import pcm_replay
from baseline.download import ROOT, sha256
from baseline.evaluation import evaluate, reference_notes
from cnn_gru.data import load_audio
from cnn_gru.inference import tablature_metrics
from cnn_gru.training import save_checkpoint
from diagnosis9.head_audit import binary_metric
from streaming.data import fingerprint
from streaming.evaluate import aggregate, save_json
from streaming.export import export
from streaming.latency import measure
from streaming.model import CausalCNNGRU
from .data import prepare, validate_config
from .training import model_configuration, producer_hashes, train_arm, verified_sanity


def validate_fixed_settings(config):
    validate_config(config)
    if config['decoder_parameters'] != {'onset_threshold': .55, 'activity_threshold': .35,
                                        'onset_backdate_seconds': .06}:
        raise ValueError('The matched decoder parameters must remain fixed.')
    if config['pitch_diagnostics'] != {'threshold': .5, 'low_register_midi': [40, 51]}:
        raise ValueError('Pitch diagnostics must use the prespecified threshold and register.')
    if config['production_promotion'] is not False:
        raise ValueError('This validation experiment does not authorize product promotion.')


def run(config, output, device):
    validate_fixed_settings(config)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'protocol.json').exists():
        raise ValueError('This stage has already started; reuse report/audit/evaluate without more updates.')
    sanity = verified_sanity(config)
    protocol = {'config': config, 'source_sha256': producer_hashes(), 'total_update_cap': 500,
                'planned_optimizer_updates': {'short128': 250, 'long256': 250},
                'new_sanity_updates': 0, 'sanity_reference': sanity,
                'test_inference_allowed': False, 'selection_rule': config['selection_rule']}
    save_json(output / 'protocol.json', protocol)
    models, audit, datasets, provenance = prepare(config, output)
    schedule = build_schedule(config, datasets['short128']['train'])
    save_json(output / 'crop-schedule.json', {'descriptors': schedule, 'sha256': fingerprint(schedule)})
    torch.manual_seed(config['seed'])
    initial_model = CausalCNNGRU(model_configuration(models['short128'], config))
    initial = {'model': initial_model.state_dict(), 'seed': config['seed'],
               'weights_sha256': tensor_digest(initial_model.state_dict()), 'origin': 'fresh random initialization',
               'sanity_warmstart': False}
    save_checkpoint(output / 'initial-weights.pt', initial)
    del initial_model
    receipts = {}
    for arm in config['arms']:
        name = arm['name']
        receipts[name] = train_arm(arm, config, models[name], audit, datasets[name]['train'], provenance,
                                   schedule, output / 'initial-weights.pt', output, device)
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    if sum(row['optimizer_steps'] for row in receipts.values()) != 500:
        raise ValueError('Actual updates do not reproduce the fixed stage budget.')
    summary = {'new_optimizer_updates': 500, 'new_sanity_updates': 0, 'sanity_reference': sanity,
               'arms': receipts, 'epochs_per_arm': 5, 'test_inference_calls': 0,
               'validation_used_during_training': False, 'config': config, 'source_sha256': producer_hashes(),
               'protocol_sha256': sha256(output / 'protocol.json'), 'cache_audit_sha256': sha256(output / 'cache-audit.json')}
    save_json(output / 'training-summary.json', summary)
    del datasets
    gc.collect()
    return evaluate_validation(config, output, device)


def aligned_heads(probability, targets, config, model_config, score_frames):
    delay = config['common_target_delay_frames']
    count = min(score_frames, len(targets['state']), len(probability['state']) - delay)
    mask = np.ones(count, np.float32)
    mask[:config['common_leading_mask_frames']] = 0
    metrics = head_metrics(probability['state'][delay:delay + count], targets['state'][:count], mask, 25)
    target_pitch = targets['pitch'][:count].astype(bool)
    estimate_pitch = probability['pitch'][delay:delay + count] >= config['pitch_diagnostics']['threshold']
    start, end = config['pitch_diagnostics']['low_register_midi']
    offset = model_config['model']['pitch_min']
    low = slice(start - offset, end - offset + 1)
    valid = np.asarray(mask, bool)
    metrics['pitch'] = {'overall': binary_metric(target_pitch[valid], estimate_pitch[valid]),
                        'low_register': binary_metric(target_pitch[valid, low], estimate_pitch[valid, low]),
                        **config['pitch_diagnostics']}
    return {**metrics, 'source_frames_scored': count, 'prediction_start_frame': delay,
            'source_grid': 'Both arms predict frames13:1013 against original source targets0:1000; '
                           'original first26 source frames masked. Same original 0–10s span.'}


def evaluate_validation(config, output, device):
    validate_fixed_settings(config)
    output = Path(output).resolve()
    protocol = json.loads((output / 'protocol.json').read_text(encoding='utf-8'))
    if protocol['config'] != config or protocol['source_sha256'] != producer_hashes():
        raise ValueError('Prespecified config or protected production sources changed.')
    models, audit, datasets, provenance = prepare(config, output)
    score = json.loads((ROOT / 'configs/stage1.json').read_text(encoding='utf-8'))
    score['selection_role'] = 'Stage11 validation-only window-length comparison; no test inference or promotion.'
    directory = output / 'validation'
    (directory / 'clips').mkdir(parents=True, exist_ok=True)
    checkpoints, paths, networks = {}, {}, {}
    for arm in config['arms']:
        name = arm['name']
        path = output / 'models' / name / 'final.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        receipt = checkpoint['receipt']
        if (checkpoint['step'] != 250 or receipt['source_sha256'] != producer_hashes()
                or receipt['config_sha256'] != fingerprint(config)
                or receipt['manifest_sha256'] != provenance['manifest_sha256']
                or receipt['spectral_cache_receipts_sha256'] != sha256(output / 'cache-audit.json')):
            raise ValueError('Final matched checkpoint differs from its cache/source protocol.')
        network = CausalCNNGRU(checkpoint['config']).to(device)
        network.load_state_dict(checkpoint['model'])
        network.eval()
        networks[name], paths[name], checkpoints[name] = network, path, checkpoint
    warm_record = datasets['short128']['validation'][0]
    warm, _ = load_audio(ROOT / warm_record['entry']['audio'], 1, models['short128'])
    delay_seconds = config['common_target_delay_frames'] / 100
    for name in networks:
        pcm_replay(networks[name], warm, checkpoints[name], config['decoder_parameters'], device,
                   config['validation_packet_ms'], delay_seconds)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    records = []
    for index, short_record in enumerate(datasets['short128']['validation']):
        entry = short_record['entry']
        audio, duration = load_audio(ROOT / entry['audio'], models['short128']['evaluation_audio_seconds'], models['short128'])
        relative = Path('validation/clips') / (entry['id'] + '.wav')
        sf.write(output / relative, audio, 16000, subtype='FLOAT')
        reference = reference_notes(ROOT / entry['annotation'])
        order = config['arms'] if index % 2 == 0 else list(reversed(config['arms']))
        for arm in order:
            name = arm['name']
            input_record = datasets[name]['validation'][index]
            if input_record['entry']['id'] != entry['id']:
                raise ValueError('Paired validation order changed.')
            notes, events, timing, probability = pcm_replay(networks[name], audio, checkpoints[name],
                                               config['decoder_parameters'], device, config['validation_packet_ms'],
                                               delay_seconds, capture=True)
            timing.update(window_seconds=arm['window_length'] / 16000,
                          hann_peak_age_seconds=arm['window_length'] / 32000,
                          fft_bin_spacing_hz=16000 / arm['n_fft'])
            probability_path = directory / 'probabilities' / name / (entry['id'] + '.npz')
            probability_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(probability_path, **probability)
            heads = aligned_heads(probability, short_record['targets'], config, models[name], round(score['score_seconds'] * 100))
            row = {'id': entry['id'], 'mode': name, 'notes': notes, 'reference': reference, 'events': events,
                   'head_metrics': heads, 'pitch_metrics': evaluate(reference, notes, score, duration),
                   'tab_metrics': tablature_metrics(reference, notes, score, score['score_seconds'], duration),
                   'latency': measure(reference, notes, score, duration), 'timing': timing,
                   'relativeaudio': relative.as_posix(), 'input_provenance': input_record['info'],
                   'probability_path': str(probability_path.relative_to(output)).replace('\\', '/'),
                   'probability_sha256': sha256(probability_path)}
            stem = directory / 'results' / name / entry['id']
            export(notes, stem, {'mode': name, 'checkpoint_sha256': sha256(paths[name]),
                                'decoder_parameters': config['decoder_parameters'], 'source_delay_seconds': delay_seconds})
            save_json(stem.with_name(stem.name + '-metrics.json'), row)
            save_json(stem.with_name(stem.name + '-events.json'), {'mode': name, 'events': events})
            records.append(row)
            print('Validation', name, entry['id'], 'state F1', round(heads['exact_state_f1'], 3),
                  'low pitch F1', round(heads['pitch']['low_register']['f1'], 3),
                  'note onset F1', round(row['pitch_metrics']['onset']['f1'], 3), flush=True)
    summary = aggregate(records)
    for name in ['short128', 'long256']:
        rows = [row for row in records if row['mode'] == name]
        summary['modes'][name]['exact_state_f1'] = float(np.mean([row['head_metrics']['exact_state_f1'] for row in rows]))
        summary['modes'][name]['active_fret_accuracy'] = float(np.mean([row['head_metrics']['active_fret_accuracy'] for row in rows]))
        for field, metric in [('auxiliary_pitch_f1', 'overall'), ('low_register_pitch_f1', 'low_register')]:
            summary['modes'][name][field] = float(np.mean([row['head_metrics']['pitch'][metric]['f1'] for row in rows]))
    winner = max(['short128', 'long256'], key=lambda name:
                 (summary['modes'][name]['exact_state_f1'], summary['modes'][name]['onset_f1']))
    selection = {'selected_arm': winner, 'rule': config['selection_rule'],
                 'scores': {name: {key: summary['modes'][name][key] for key in ['exact_state_f1', 'onset_f1']}
                            for name in ['short128', 'long256']},
                 'checkpoint_sha256': {name: sha256(path) for name, path in paths.items()},
                 'selected_from_test': False, 'production_promoted': False,
                 'scope': 'Development comparison under a fixed shared timing contract; displayed validation clips also select the model.'}
    save_json(output / 'selection.json', selection)
    save_json(directory / 'summary.json', summary)
    training_summary = json.loads((output / 'training-summary.json').read_text(encoding='utf-8'))
    run = {'config': config, 'scoring_config': score, 'selection': selection,
           'checkpoint_sha256': selection['checkpoint_sha256'], 'source_sha256': producer_hashes(),
           'new_optimizer_updates': 500, 'new_sanity_updates': 0, 'sanity_reference': protocol['sanity_reference'],
           'test_used': False, 'test_inference_calls': 0, 'training_summary': training_summary,
           'cache_audit': json.loads((output / 'cache-audit.json').read_text(encoding='utf-8')),
           'input_provenance': provenance, 'protocol_sha256': sha256(output / 'protocol.json'),
           'peak_gpu_mib': torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None,
           'scope': 'Validation-only matched window-length comparison. No promotion or independent-test claim.'}
    save_json(directory / 'run.json', run)
    from .report import write_report
    write_report(output, records, summary, run)
    return summary
