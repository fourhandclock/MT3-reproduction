"""Locked 48-case PCM inference, without an optimizer or test-driven selection."""

from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from baseline.download import ROOT, sha256
from baseline.evaluation import evaluate
from cnn_gru.inference import tablature_metrics
from headswap18.analysis import state_diagnosis
from multires17.model import CausalCNNGRU
from multires17.workflow import pcm_replay
from spectral11.workflow import aligned_heads
from streaming.evaluate import save_json
from streaming.export import export
from streaming.latency import measure
from balance22.evaluation import frame_config, fret_confusion, summarize
from .data import MODES, freeze, load_test_inputs, read, source_hashes


def locked_selection(config):
    return {'selected_arm': config['locked_selected_arm'], 'selection_stage': 22,
            'model_selection_performed': False, 'selected_from_test': False, 'production_promoted': False}


def contrasts(test_summary):
    validation = read(ROOT / 'outputs/balance22/validation/summary.json')
    selected = 'uniform_ce650'
    result = {'locked_selected_arm': selected, 'comparison_scope': 'Descriptive test comparisons; no model reselection.',
              'per_mode': {}, 'locked_selected_vs_original': {}, 'locked_selected_vs_parent': {}}
    keys = ('exact_state_f1', 'onset_f1', 'onset_offset_f1', 'tab_onset_f1', 'frame_f1',
            'best_active_fret_accuracy', 'best_active_equal_position_accuracy')
    for mode in MODES:
        result['per_mode'][mode] = {key: {'development': validation['modes'][mode][key],
            'test': test_summary['modes'][mode][key],
            'test_minus_development': test_summary['modes'][mode][key] - validation['modes'][mode][key]} for key in keys}
    for name, baseline in [('locked_selected_vs_original', 'original650'), ('locked_selected_vs_parent', 'parent500')]:
        result[name] = {key: test_summary['modes'][selected][key] - test_summary['modes'][baseline][key] for key in keys}
    return result


def run(config, output, device):
    output = Path(output)
    protocol = freeze(config, output)
    base, records, provenance, bindings = load_test_inputs(config, output)
    save_json(output / 'input-bindings.json', bindings)
    save_json(output / 'test-inputs.json', provenance)
    networks, checkpoints, identities = {}, {}, {}
    for mode in MODES:
        item = config['models'][mode]
        path = ROOT / item['checkpoint']
        if sha256(path) != item['sha256']:
            raise ValueError('Locked checkpoint changed before inference')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        expected_step = 500 if mode == 'parent500' else 650
        if (checkpoint['step'] != expected_step or checkpoint['config']['feature_planes'] != [2048, 2048]
                or checkpoint['config']['feature'] != base['feature']):
            raise ValueError('Locked checkpoint step or representation differs')
        model = CausalCNNGRU(checkpoint['config']).to(device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        model.requires_grad_(False)
        networks[mode], checkpoints[mode] = model, checkpoint
        identities[mode] = {'checkpoint': item['checkpoint'], 'sha256_before': item['sha256'],
                            'step': expected_step, 'normalization_recomputed': False}
    for mode in MODES:
        pcm_replay(networks[mode], records[0]['audio'][:16000], checkpoints[mode], config['decoder_parameters'], device)
    rows = []
    for index, item in enumerate(records):
        entry, audio, duration, reference = item['entry'], item['audio'], item['duration'], item['reference']
        relative = Path('test/clips') / (entry['id'] + '.wav')
        (output / relative).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output / relative, audio, 16000, subtype='FLOAT')
        for mode in MODES[index % 4:] + MODES[:index % 4]:
            notes, events, timing, probability = pcm_replay(networks[mode], audio, checkpoints[mode], config['decoder_parameters'], device, True)
            path = output / 'test/probabilities' / mode / (entry['id'] + '.npz')
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **probability)
            heads = aligned_heads(probability, item['targets'], config, base, config['source_frames'])
            diagnosis, _ = state_diagnosis(probability, item['targets'], frame_config(config))
            metrics = evaluate(reference, notes, protocol['scoring_config'], duration)
            tabs = tablature_metrics(reference, notes, protocol['scoring_config'], 10, duration)
            row = {'id': entry['id'], 'mode': mode, 'notes': notes, 'events': events, 'reference': reference,
                'head_metrics': heads, 'frame_diagnosis': diagnosis, 'state_derived_pitch': diagnosis['state_derived_pitch'],
                'best_active_confusion': fret_confusion(probability, item['targets']),
                'pitch_metrics': metrics, 'tab_metrics': tabs, 'timing': timing,
                'latency': measure(reference, notes, protocol['scoring_config'], duration),
                'relativeaudio': relative.as_posix(), 'probability_path': path.relative_to(output).as_posix(),
                'probability_sha256': sha256(path), 'input_provenance': provenance[index]}
            stem = output / 'test/results' / mode / entry['id']
            export(notes, stem, {'stage': 23, 'split': 'existing_progression3', 'mode': mode,
                'checkpoint_sha256': config['models'][mode]['sha256'], 'decoder_parameters': config['decoder_parameters'],
                'source_delay_seconds': .13, 'selection_source': 'Stage22 development; not reselected on test',
                'clock_scope': 'Fresh mono16k PCM replay, measured virtual completion; no capture-hardware latency.'})
            save_json(stem.with_name(stem.name + '-metrics.json'), row)
            save_json(stem.with_name(stem.name + '-events.json'), {'events': events})
            rows.append(row)
            print(mode, entry['id'], 'state F1', round(heads['exact_state_f1'], 4), 'onset F1', round(metrics['onset']['f1'], 4), flush=True)
    for mode in MODES:
        path = ROOT / config['models'][mode]['checkpoint']
        identities[mode]['sha256_after'] = sha256(path)
        if identities[mode]['sha256_after'] != identities[mode]['sha256_before']:
            raise ValueError('A locked checkpoint changed during inference')
    summary = summarize(rows)
    summary.update(stage=23, **locked_selection(config))
    save_json(output / 'test/summary.json', summary)
    save_json(output / 'comparison.json', contrasts(summary))
    save_json(output / 'model-integrity.json', identities)
    save_json(output / 'test/run.json', {'config': config, 'scoring_config': protocol['scoring_config'],
        'selection': read(output / 'selection.json'), 'selection_source': 'Stage22 development, copied before test reads',
        'locked_selection': locked_selection(config), 'new_optimizer_updates': 0, 'optimizer_instantiations': 0,
        'normalization_updates': 0, 'test_scored_replays': 48, 'test_audio_warmups': 4,
        'checkpoint_paths': {mode: config['models'][mode]['checkpoint'] for mode in MODES},
        'checkpoint_sha256': {mode: config['models'][mode]['sha256'] for mode in MODES},
        'source_sha256': source_hashes(), 'input_provenance': provenance, 'device': str(device),
        'data_scope': protocol['data_scope'], 'production_promoted': False})
    print('Stage23 complete: 48 scored replays plus four warmups; new optimizer updates=0; selection unchanged.', flush=True)
    return summary
