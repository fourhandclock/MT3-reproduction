"""Prespecified train-only sanity and matched validation acoustic experiment."""

import gc
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from baseline.download import ROOT, sha256
from baseline.evaluation import evaluate, reference_notes
from cnn_gru.data import load_audio
from cnn_gru.inference import tablature_metrics
from cnn_gru.training import save_checkpoint
from streaming.data import fingerprint
from streaming.evaluate import aggregate, save_json
from streaming.export import export
from streaming.inference import PCMStream
from streaming.latency import measure
from streaming.model import CausalCNNGRU
from .data import build_schedule, load_inputs, validate_config
from .decoder import SourceTimeDecoder
from .training import head_metrics, model_configuration, producer_hashes, run_sanity, tensor_digest, train_arm


def run(config, output, device):
    validate_config(config)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'protocol.json').exists():
        raise ValueError('This stage has already started. Use report/audit/evaluate; training is not repeated.')
    # This immutable marker prevents retrying an interrupted run with an uncounted fresh budget.
    protocol = {'config': config, 'source_sha256': producer_hashes(), 'total_update_cap': 500,
                'planned_optimizer_updates': {'sanity': 100, 'baseline': 200, 'delay60': 200},
                'test_inference_allowed': False, 'selection_rule': config['selection_rule']}
    save_json(output / 'protocol.json', protocol)
    base, audit, datasets, provenance = load_inputs(config, include_validation=False)
    sanity = run_sanity(config, base, audit, datasets, provenance, output, device)
    if not sanity['passed']:
        summary = {'new_optimizer_updates': 100, 'sanity_passed': False, 'arms_trained': [],
                   'reason': 'The fixed 100-update training-batch sanity gate did not pass. '
                             'No additional acoustic training is authorized by this conditional protocol.',
                   'test_inference_calls': 0, 'config': config, 'source_sha256': producer_hashes()}
        save_json(output / 'training-summary.json', summary)
        from .report import write_report
        write_report(output, [], {'modes': {}}, {'config': config, 'sanity': sanity,
                     'training_summary': summary, 'new_optimizer_updates': 100, 'scope': summary['reason']})
        return summary
    del datasets
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    base, audit, datasets, provenance = load_inputs(config)
    schedule = build_schedule(config, datasets['train'])
    save_json(output / 'crop-schedule.json', {'descriptors': schedule, 'sha256': fingerprint(schedule)})
    torch.manual_seed(config['seed'])
    model = CausalCNNGRU(model_configuration(base, config))
    initial = {'model': model.state_dict(), 'seed': config['seed'], 'origin': 'fresh random initialization',
               'weights_sha256': tensor_digest(model.state_dict()), 'sanity_warmstart': False}
    save_checkpoint(output / 'initial-weights.pt', initial)
    del model
    receipts = {}
    for arm in config['arms']:
        receipts[arm['name']] = train_arm(arm, config, base, audit, datasets, provenance,
                                         schedule, output / 'initial-weights.pt', output, device)
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    total = sanity['optimizer_steps'] + sum(row['optimizer_steps'] for row in receipts.values())
    if total != 500:
        raise ValueError('Actual stage update accounting differs from the fixed budget.')
    summary = {'new_optimizer_updates': total, 'sanity_passed': True, 'arms': receipts,
               'sanity_updates': sanity['optimizer_steps'], 'epochs_per_arm': 5,
               'test_inference_calls': 0, 'validation_used_during_training': False,
               'config': config, 'source_sha256': producer_hashes(), 'protocol_sha256': sha256(output / 'protocol.json')}
    save_json(output / 'training-summary.json', summary)
    return evaluate_validation(config, output, device)


class RecordedDecoder:
    """Keep the exact acoustic probabilities consumed by each causal packet."""
    def __init__(self, decoder):
        self.decoder, self.blocks = decoder, []

    def push(self, probabilities, available_at_audio_seconds=None):
        self.blocks.append({key: value.copy() for key, value in probabilities.items()})
        return self.decoder.push(probabilities, available_at_audio_seconds)

    def __getattr__(self, name):
        return getattr(self.decoder, name)

    def probabilities(self):
        return {key: np.concatenate([block[key] for block in self.blocks]) for key in self.blocks[0]} if self.blocks else {}


def pcm_replay(model, audio, checkpoint, parameters, device, packet_ms, source_delay_seconds, capture=False):
    stream = PCMStream(model, checkpoint['normalization'], checkpoint['config'], parameters, device)
    stream.decoder = SourceTimeDecoder(checkpoint['config'], parameters, source_delay_seconds)
    if capture:
        stream.decoder = RecordedDecoder(stream.decoder)
    rate = checkpoint['config']['feature']['sample_rate']
    packet = round(packet_ms * rate / 1000)
    for start in range(0, len(audio), packet):
        stream.push(audio[start:start + packet])
    stream.finish()
    times = np.asarray(stream.packet_times)
    duration = len(audio) / rate
    timing = {'processing_seconds': float(times.sum()), 'input_seconds': duration,
              'real_time_factor': float(times.sum()) / duration if duration else None,
              'chunk_processing_p50_ms': float(np.quantile(times, .5) * 1000) if len(times) else None,
              'chunk_processing_p95_ms': float(np.quantile(times, .95) * 1000) if len(times) else None,
              'packet_ms': packet_ms, 'lookahead_samples': 0,
              'max_virtual_backlog_ms': max(stream.packet_backlogs or [0]) * 1000,
              'source_delay_compensation_seconds': source_delay_seconds,
              'source_clamps': stream.decoder.clamp_counts,
              'timing_scope': 'Warm isolated mono16k PCM: causal frontend, frozen train normalization, '
                              'carried model state and incremental decoder/source projection. Loading/resampling excluded.',
              'delivery_clock': 'max(previous finish, packet arrival)+measured processing; '
                                'source shift never changes availability or delivery.'}
    timing['exact_probability_recording_included'] = capture
    return stream.notes, stream.events, timing, stream.decoder.probabilities() if capture else None


def aligned_head_metrics(probabilities, targets, config, delay_frames, score_frames):
    count = min(score_frames, len(targets['state']), len(probabilities['state']) - delay_frames)
    mask = np.ones(count, np.float32)
    mask[:config['common_leading_mask_frames']] = 0
    metrics = head_metrics(probabilities['state'][delay_frames:delay_frames + count],
                           targets['state'][:count], mask, 25)
    return {**metrics, 'source_frames_scored': count,
            'prediction_start_frame': delay_frames,
            'source_grid': 'Identical original 0–10 s source frame span in both arms; '
                           'delayed predictions use frames6–1005, immediate uses0–999; original first6 source frames masked.'}


def evaluate_validation(config, output, device):
    output = Path(output).resolve()
    protocol = json.loads((output / 'protocol.json').read_text(encoding='utf-8'))
    if protocol['config'] != config or protocol['source_sha256'] != producer_hashes():
        raise ValueError('Prespecified configuration or production sources changed.')
    sanity = json.loads((output / 'sanity.json').read_text(encoding='utf-8'))
    if not sanity['passed']:
        raise ValueError('The sanity gate failed; paired evaluation is not available.')
    base, audit, datasets, provenance = load_inputs(config)
    score = json.loads((ROOT / 'configs/stage1.json').read_text(encoding='utf-8'))
    score['selection_role'] = 'Stage10 validation-only development comparison. No test inference or fresh generalization claim.'
    directory = output / 'validation'
    (directory / 'clips').mkdir(parents=True, exist_ok=True)
    checkpoints, paths, models = {}, {}, {}
    for arm in config['arms']:
        name = arm['name']
        path = output / 'models' / name / 'final.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if (checkpoint['step'] != 200 or checkpoint['receipt']['source_sha256'] != producer_hashes()
                or checkpoint['receipt']['config_sha256'] != fingerprint(config)
                or checkpoint['receipt']['manifest_sha256'] != provenance['manifest_sha256']):
            raise ValueError('Final matched acoustic checkpoint differs from its protocol.')
        model = CausalCNNGRU(checkpoint['config']).to(device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        models[name], paths[name], checkpoints[name] = model, path, checkpoint
    # Warm each model on validation audio, without inspecting a score or changing weights.
    warm, _ = load_audio(ROOT / datasets['validation'][0]['entry']['audio'], 1, base)
    for arm in config['arms']:
        name = arm['name']
        pcm_replay(models[name], warm, checkpoints[name], config['decoder_parameters'], device,
                   config['validation_packet_ms'], arm['target_delay_frames'] / 100)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    records = []
    for index, record in enumerate(datasets['validation']):
        entry = record['entry']
        audio, duration = load_audio(ROOT / entry['audio'], base['evaluation_audio_seconds'], base)
        relative = Path('validation/clips') / (entry['id'] + '.wav')
        sf.write(output / relative, audio, base['feature']['sample_rate'], subtype='FLOAT')
        reference = reference_notes(ROOT / entry['annotation'])
        order = config['arms'] if index % 2 == 0 else list(reversed(config['arms']))
        for arm in order:
            name, delay = arm['name'], arm['target_delay_frames']
            notes, events, timing, probability = pcm_replay(models[name], audio, checkpoints[name],
                                              config['decoder_parameters'], device,
                                              config['validation_packet_ms'], delay / 100, capture=True)
            probability_path = directory / 'probabilities' / name / (entry['id'] + '.npz')
            probability_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(probability_path, **probability)
            heads = aligned_head_metrics(probability, record['targets'], config, delay, round(score['score_seconds'] * 100))
            row = {'id': entry['id'], 'mode': name, 'notes': notes, 'reference': reference, 'events': events,
                   'head_metrics': heads, 'pitch_metrics': evaluate(reference, notes, score, duration),
                   'tab_metrics': tablature_metrics(reference, notes, score, score['score_seconds'], duration),
                   'latency': measure(reference, notes, score, duration), 'timing': timing,
                   'relativeaudio': relative.as_posix(), 'input_provenance': provenance['inputs']['validation'][index],
                   'probability_path': str(probability_path.relative_to(output)).replace('\\', '/'),
                   'probability_sha256': sha256(probability_path)}
            stem = directory / 'results' / name / entry['id']
            export(notes, stem, {'mode': name, 'checkpoint_sha256': sha256(paths[name]),
                                'decoder_parameters': config['decoder_parameters'], 'source_delay_seconds': delay / 100})
            save_json(stem.with_name(stem.name + '-metrics.json'), row)
            save_json(stem.with_name(stem.name + '-events.json'), {'mode': name, 'events': events})
            records.append(row)
            print('Validation', name, entry['id'], 'head F1', round(heads['exact_state_f1'], 3),
                  'note onset F1', round(row['pitch_metrics']['onset']['f1'], 3), flush=True)
    summary = aggregate(records)
    for name in ['baseline', 'delay60']:
        rows = [record for record in records if record['mode'] == name]
        summary['modes'][name]['exact_state_f1'] = float(np.mean([row['head_metrics']['exact_state_f1'] for row in rows]))
        summary['modes'][name]['active_fret_accuracy'] = float(np.mean([row['head_metrics']['active_fret_accuracy'] for row in rows]))
    winner = max(['baseline', 'delay60'], key=lambda name:
                 (summary['modes'][name]['exact_state_f1'], summary['modes'][name]['onset_f1']))
    selection = {'selected_arm': winner, 'rule': config['selection_rule'],
                 'scores': {name: {key: summary['modes'][name][key] for key in ['exact_state_f1', 'onset_f1']}
                            for name in ['baseline', 'delay60']},
                 'checkpoint_sha256': {name: sha256(path) for name, path in paths.items()},
                 'selected_from_test': False, 'production_promoted': False,
                 'scope': 'Development comparison only; all displayed clips were used for model selection.'}
    save_json(output / 'selection.json', selection)
    save_json(directory / 'summary.json', summary)
    run_metadata = {'config': config, 'scoring_config': score, 'selection': selection,
                    'checkpoint_sha256': selection['checkpoint_sha256'], 'source_sha256': producer_hashes(),
                    'new_optimizer_updates': 500, 'test_used': False, 'test_inference_calls': 0,
                    'peak_gpu_mib': torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None,
                    'sanity': sanity, 'training_summary': json.loads((output / 'training-summary.json').read_text()),
                    'input_provenance': provenance, 'protocol_sha256': sha256(output / 'protocol.json'),
                    'scope': 'Validation-only acoustic ablation with fixed final step200 models. No product promotion.'}
    save_json(directory / 'run.json', run_metadata)
    from .report import write_report
    write_report(output, records, summary, run_metadata)
    return summary
