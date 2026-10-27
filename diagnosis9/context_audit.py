"""Validation-only recurrence length diagnostic using frozen weights."""

import json
from pathlib import Path

import numpy as np
import torch

from baseline.download import ROOT, sha256
from baseline.evaluation import evaluate, reference_notes
from streaming.data import cache_path
from streaming.evaluate import load_checkpoint, save_json
from streaming.inference import decode_probabilities
from .head_audit import binary_metric, exact_state_metric, pooled


def reset_predictions(model, features, normalization, device, chunk_frames=200):
    mean = np.asarray(normalization['mean'], np.float32)
    std = np.asarray(normalization['std'], np.float32)
    result = {}
    with torch.inference_mode():
        for start in range(0, len(features), chunk_frames):
            x = np.clip((np.asarray(features[start:start + chunk_frames], np.float32) - mean) / std, -5, 5)
            # Deliberately reset both convolution and GRU state to crop-training conditions.
            logits, _ = model.forward_chunk(torch.from_numpy(x[None]).to(device), None)
            for key, value in logits.items():
                probability = torch.softmax(value, dim=-1) if key == 'state' else torch.sigmoid(value)
                result.setdefault(key, []).append(probability[0].cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in result.items()}


def state_to_pitch(states, config):
    inactive = config['model']['max_fret'] + 1
    lower, upper = config['model']['pitch_min'], config['model']['pitch_max']
    result = np.zeros((len(states), upper - lower + 1), dtype=bool)
    for string, tuning in enumerate((40, 45, 50, 55, 59, 64)):
        active = (states[:, string] >= 0) & (states[:, string] < inactive)
        times = np.flatnonzero(active)
        pitches = states[times, string] + tuning - lower
        good = (pitches >= 0) & (pitches < result.shape[1])
        result[times[good], pitches[good]] = True
    return result


def score_heads(probability, target, pitch_target, config):
    inactive = config['model']['max_fret'] + 1
    state = probability['state']
    best = state[:, :, :-1].max(axis=-1)
    odds = best / (best + state[:, :, -1] + 1e-8)
    states = np.where(odds >= .35, state[:, :, :-1].argmax(axis=-1), inactive)
    return {'string_fret_activity_035': exact_state_metric(target, states, inactive),
            'state_derived_pitch_activity_035': binary_metric(pitch_target, state_to_pitch(states, config)),
            'auxiliary_pitch_05': binary_metric(pitch_target, probability['pitch'] >= .5)}


def main():
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path, checkpoint, model = load_checkpoint(ROOT / 'outputs/streaming', device)
    config, scoring = checkpoint['config'], checkpoint['scoring_config']
    audit = json.loads((ROOT / 'outputs/diagnosis9/validation-head-diagnostics.json').read_text())
    if audit['checkpoint_sha256'] != sha256(checkpoint_path):
        raise ValueError('Head audit checkpoint differs.')
    records = []
    manifest = json.loads((ROOT / 'data/cnn_gru/manifest.json').read_text())
    for entry, old in zip(manifest['validation'], audit['clips']):
        if entry['id'] != old['id']:
            raise ValueError('Validation ordering changed.')
        prefix = cache_path(entry, config, config['evaluation_audio_seconds'])
        features = np.load(prefix.with_suffix('.npy'), mmap_mode='r')
        if sha256(prefix.with_suffix('.npy')) != old['features_sha256']:
            raise ValueError('Validation features changed.')
        if sha256(prefix.with_suffix('.npz')) != old['targets_sha256']:
            raise ValueError('Validation labels changed.')
        if sha256(ROOT / old['probability_path']) != old['probability_sha256']:
            raise ValueError('Continuous probabilities changed.')
        with np.load(ROOT / old['probability_path']) as arrays:
            continuous = {key: arrays[key].copy() for key in arrays.files}
        reset = reset_predictions(model, features, checkpoint['normalization'], device)
        with np.load(prefix.with_suffix('.npz')) as arrays:
            targets = arrays['state'].copy()
            pitch_targets = arrays['pitch'].astype(bool)
        row = {'id': entry['id'], 'modes': {}}
        for name, probability in (('continuous_12s', continuous), ('reset_every_2s', reset)):
            count = min(len(features), 1000)
            shortened = {key: value[:count] for key, value in probability.items()}
            metrics = score_heads(shortened, targets[:count], pitch_targets[:count], config)
            segments = []
            for start in range(0, count, 200):
                end = min(count, start + 200)
                result = score_heads({key: value[start:end] for key, value in probability.items()},
                                     targets[start:end], pitch_targets[start:end], config)
                segments.append({'start_seconds': start / 100, 'end_seconds': end / 100, 'metrics': result})
            notes, _ = decode_probabilities(probability, config, checkpoint['decoder_parameters'], len(features) / 100)
            evaluation = evaluate(reference_notes(ROOT / entry['annotation']), notes, scoring, len(features) / 100)
            row['modes'][name] = {'heads': metrics, 'segments': segments,
                                  'note_onset': evaluation['onset'], 'note_frame': evaluation['frame']}
        records.append(row)
        print(entry['id'], 'continuous/reset strict onset',
              round(row['modes']['continuous_12s']['note_onset']['f1'], 3),
              round(row['modes']['reset_every_2s']['note_onset']['f1'], 3), flush=True)
    summary = {}
    for name in records[0]['modes']:
        rows = [record['modes'][name] for record in records]
        summary[name] = {'heads': {key: {'pooled': pooled([row['heads'][key] for row in rows]),
                                               'mean_clip_f1': float(np.mean([row['heads'][key]['f1'] for row in rows]))}
                                   for key in rows[0]['heads']},
                         'note_onset': {'pooled': pooled([row['note_onset'] for row in rows]),
                                        'mean_clip_f1': float(np.mean([row['note_onset']['f1'] for row in rows]))},
                         'note_frame_mean_f1': float(np.mean([row['note_frame']['f1'] for row in rows])),
                         'segments': []}
        for index in range(5):
            summary[name]['segments'].append({'start_seconds': index * 2,
                'metrics': {key: {'pooled': pooled([row['segments'][index]['metrics'][key] for row in rows]),
                    'mean_clip_f1': float(np.mean([row['segments'][index]['metrics'][key]['f1'] for row in rows]))}
                    for key in rows[0]['heads']}})
    result = {'scope': 'Validation only; first 10 seconds scored with 12 seconds input.',
              'new_optimizer_updates': 0, 'test_used': False,
              'checkpoint_sha256': sha256(checkpoint_path), 'source_sha256': sha256(Path(__file__)),
              'parameter_selection': False,
              'reset_limit': 'Fixed 2-second reset is diagnostic. It resets convolution caches and recurrent hidden state; no lookahead or weight updates. Boundary discontinuities can create extra notes.',
              'summary': summary, 'clips': records}
    save_json(ROOT / 'outputs/diagnosis9/validation-context-diagnostics.json', result)
    text = ['# Frozen causal context audit', '', result['scope'], '', result['reset_limit'], '',
            '| Mode | String/fret frame F1 | State pitch frame F1 | Auxiliary pitch frame F1 | Strict note onset F1 |',
            '|---|---:|---:|---:|---:|']
    for name, row in summary.items():
        head = row['heads']
        text.append(f"| {name} | {head['string_fret_activity_035']['mean_clip_f1']:.4f} | "
                    f"{head['state_derived_pitch_activity_035']['mean_clip_f1']:.4f} | "
                    f"{head['auxiliary_pitch_05']['mean_clip_f1']:.4f} | {row['note_onset']['mean_clip_f1']:.4f} |")
    text += ['', '| Continuous segment start | String/fret F1 | Pitch F1 |', '|---|---:|---:|']
    for row in summary['continuous_12s']['segments']:
        text.append(f"| {row['start_seconds']} s | {row['metrics']['string_fret_activity_035']['mean_clip_f1']:.4f} | "
                    f"{row['metrics']['state_derived_pitch_activity_035']['mean_clip_f1']:.4f} |")
    (ROOT / 'outputs/diagnosis9/validation-context-diagnostics.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
