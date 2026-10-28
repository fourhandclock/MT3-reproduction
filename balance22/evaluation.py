"""Four fresh PCM replays per clip; fixed final selection and parent reproduction."""

from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from baseline.download import ROOT, sha256
from baseline.evaluation import evaluate, reference_notes
from cnn_gru.data import load_audio
from cnn_gru.inference import tablature_metrics
from decoder14.core import without_compute
from decoder14.workflow import read
from diagnosis9.head_audit import pooled
from headswap18.analysis import state_diagnosis
from multires17.model import CausalCNNGRU
from multires17.workflow import pcm_replay
from prior21.analysis import aligned, confusion
from spectral11.workflow import aligned_heads
from streaming.evaluate import aggregate, save_json
from streaming.export import export
from streaming.latency import measure
from .data import load_inputs
from .training import producer_hashes

MODES = ['parent500', 'original650', 'uniform_ce650', 'balanced_ce650']


def choose(summaries):
    if set(summaries) != set(MODES[1:]):
        raise ValueError('Only all three fixed step650 finals may be selected')
    scores = {m: tuple(summaries[m][k] for k in ('exact_state_f1', 'onset_f1')) for m in MODES[1:]}
    if any(not np.isfinite(v) or not 0 <= v <= 1 for values in scores.values() for v in values):
        raise ValueError('Selection metrics must be finite and in [0,1]')
    return max(MODES[1:], key=lambda m: scores[m])


def frame_config(config):
    return {'source_frames': 1000, 'source_delay_frames': 13, 'leading_mask_frames': 26,
            'auxiliary_pitch_threshold': .5, 'decoder_parameters': config['decoder_parameters']}


def fret_confusion(probability, targets):
    string, truth, best = aligned(probability['state'], targets['state'])
    return confusion(string, truth, best).tolist()


def fret_metrics(counts):
    counts = np.asarray(counts, np.int64)
    support = counts.sum(2)
    correct = np.diagonal(counts, axis1=1, axis2=2)
    valid = support > 0
    return {'correct': int(correct.sum()), 'count': int(support.sum()),
        'accuracy': float(correct.sum() / support.sum()) if support.sum() else None,
        'equal_supported_position_accuracy': float(np.mean(correct[valid] / support[valid])) if valid.any() else None,
        'supported_positions': int(valid.sum())}


def check_parent(item, probability, notes, events, heads, diagnosis, metrics, tabs, fret):
    source = ROOT / 'outputs/fret19'
    prior = read(source / 'validation/results/control500' / (item['entry']['id'] + '-metrics.json'))
    exact, maximum = True, 0.
    path = source / prior['probability_path']
    if sha256(path) != prior['probability_sha256']:
        raise ValueError('Original control500 probabilities changed')
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != set(probability):
            raise ValueError('Parent head interface differs')
        for key in arrays.files:
            if arrays[key].shape != probability[key].shape:
                raise ValueError('Parent probability grid differs')
            exact = exact and np.array_equal(arrays[key], probability[key])
            maximum = max(maximum, float(np.max(np.abs(arrays[key] - probability[key]))))
    if (maximum > 1e-6 or without_compute(notes) != without_compute(prior['notes'])
            or without_compute(events) != without_compute(prior['events']) or heads != prior['head_metrics']
            or diagnosis != prior['frame_diagnosis'] or without_compute(metrics) != without_compute(prior['pitch_metrics'])
            or without_compute(tabs) != without_compute(prior['tab_metrics'])
            or fret_metrics(fret)['correct'] != prior['frame_diagnosis']['best_active_fret_on_reference_active']['correct']):
        raise ValueError('Parent decisions, raw heads or scores do not reproduce')
    return {'probabilities_bit_identical': bool(exact), 'max_absolute_probability_difference': maximum,
        'source_decisions_identical': True, 'source_scores_identical': True,
        'historical_probability_sha256': prior['probability_sha256']}


def summarize(records):
    result = aggregate(records)
    for mode in MODES:
        rows = [r for r in records if r['mode'] == mode]
        value = result['modes'][mode]
        for key in ('exact_state_f1', 'active_fret_accuracy'):
            value[key] = float(np.mean([r['head_metrics'][key] for r in rows]))
        for field, key in [('auxiliary_pitch_f1', 'overall'), ('low_register_pitch_f1', 'low_register')]:
            value[field] = float(np.mean([r['head_metrics']['pitch'][key]['f1'] for r in rows]))
        value['state_derived_pitch_f1'] = float(np.mean([r['state_derived_pitch']['f1'] for r in rows]))
        value['frame_policies'] = {}
        for policy in ('argmax', 'decoder_activity'):
            metrics = [r['frame_diagnosis']['policies'][policy] for r in rows]
            correct = sum(r['fret_given_reference_and_prediction_active']['correct'] for r in metrics)
            count = sum(r['fret_given_reference_and_prediction_active']['count'] for r in metrics)
            value['frame_policies'][policy] = {
                'exact_state_mean_clip_f1': float(np.mean([r['exact_state']['f1'] for r in metrics])),
                'exact_state_pooled': pooled([r['exact_state'] for r in metrics]),
                'activity_pooled': pooled([r['activity'] for r in metrics]),
                'counts': {k: sum(r['counts'][k] for r in metrics) for k in metrics[0]['counts']},
                'conditional_fret_correct': correct, 'conditional_fret_count': count,
                'conditional_fret_accuracy': correct / count if count else None}
        value['conditional_fret_accuracy'] = value['frame_policies']['decoder_activity']['conditional_fret_accuracy']
        combined = np.sum([r['best_active_confusion'] for r in rows], axis=0)
        value['best_active_fret'] = fret_metrics(combined)
        value['best_active_fret_accuracy'] = value['best_active_fret']['accuracy']
        value['best_active_equal_position_accuracy'] = value['best_active_fret']['equal_supported_position_accuracy']
    return result


def evaluate_validation(config, output, device):
    output = Path(output)
    if (output / 'verification.json').exists():
        raise ValueError('Completed Stage22 evidence must not be regenerated')
    protocol = read(output / 'protocol.json')
    if protocol['config'] != config or protocol['source_sha256'] != producer_hashes():
        raise ValueError('The actual training producers/config changed')
    base, audit, datasets, provenance, parent, _, _ = load_inputs(config, output)
    paths = {'parent500': ROOT / config['parent_checkpoint'],
             **{arm['name'] + '650': output / 'models' / arm['name'] / 'final.pt' for arm in config['arms']}}
    checkpoints, networks = {}, {}
    for mode, path in paths.items():
        checkpoint = parent if mode == 'parent500' else torch.load(path, map_location='cpu', weights_only=True)
        if mode != 'parent500' and (checkpoint['step'] != 650 or checkpoint['receipt']['source_sha256'] != producer_hashes()
                or checkpoint['receipt']['input_audit_sha256'] != sha256(output / 'input-audit.json')):
            raise ValueError('Final checkpoint or training provenance differs')
        model = CausalCNNGRU(checkpoint['config']).to(device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        checkpoints[mode], networks[mode] = checkpoint, model
    score = {k: read(ROOT / 'configs/stage1.json')[k] for k in
             ('score_seconds', 'frame_rate', 'onset_tolerance_seconds', 'offset_tolerance_seconds', 'offset_ratio')}
    warm, _ = load_audio(ROOT / datasets['validation'][0]['entry']['audio'], 1, base)
    for mode in MODES:
        pcm_replay(networks[mode], warm, checkpoints[mode], config['decoder_parameters'], device)
    records = []
    for index, item in enumerate(datasets['validation']):
        entry = item['entry']
        audio, duration = load_audio(ROOT / entry['audio'], 12, base)
        relative = Path('validation/clips') / (entry['id'] + '.wav')
        (output / relative).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output / relative, audio, 16000, subtype='FLOAT')
        reference = reference_notes(ROOT / entry['annotation'])
        for mode in MODES[index % 4:] + MODES[:index % 4]:
            notes, events, timing, probability = pcm_replay(networks[mode], audio, checkpoints[mode], config['decoder_parameters'], device, True)
            path = output / 'validation/probabilities' / mode / (entry['id'] + '.npz')
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **probability)
            heads = aligned_heads(probability, item['targets'], config, base, 1000)
            diagnosis, _ = state_diagnosis(probability, item['targets'], frame_config(config))
            fret = fret_confusion(probability, item['targets'])
            metrics = evaluate(reference, notes, score, duration)
            tabs = tablature_metrics(reference, notes, score, 10, duration)
            proof = check_parent(item, probability, notes, events, heads, diagnosis, metrics, tabs, fret) if mode == 'parent500' else None
            row = {'id': entry['id'], 'mode': mode, 'notes': notes, 'events': events, 'reference': reference,
                'head_metrics': heads, 'frame_diagnosis': diagnosis, 'state_derived_pitch': diagnosis['state_derived_pitch'],
                'best_active_confusion': fret, 'pitch_metrics': metrics, 'tab_metrics': tabs, 'timing': timing,
                'latency': measure(reference, notes, score, duration), 'relativeaudio': relative.as_posix(),
                'probability_path': path.relative_to(output).as_posix(), 'probability_sha256': sha256(path),
                'input_provenance': provenance['inputs']['validation'][index], 'parent_reproduction': proof}
            stem = output / 'validation/results' / mode / entry['id']
            export(notes, stem, {'stage': 22, 'mode': mode, 'checkpoint_sha256': sha256(paths[mode]),
                'decoder_parameters': config['decoder_parameters'], 'source_delay_seconds': .13,
                'clock_scope': 'Fresh mono16k PCM replay; virtual completion includes measured processing'})
            save_json(stem.with_name(stem.name + '-metrics.json'), row)
            save_json(stem.with_name(stem.name + '-events.json'), {'events': events})
            records.append(row)
            print(mode, entry['id'], 'state F1', round(heads['exact_state_f1'], 4), 'onset F1', round(metrics['onset']['f1'], 4), flush=True)
    summary = summarize(records)
    winner = choose({mode: summary['modes'][mode] for mode in MODES[1:]})
    selection = {'selected_arm': winner, 'eligible_modes': MODES[1:], 'rule': config['selection_rule'],
        'scores': {mode: {key: summary['modes'][mode][key] for key in ('exact_state_f1', 'onset_f1')} for mode in MODES[1:]},
        'selected_from_test': False, 'production_promoted': False}
    save_json(output / 'selection.json', selection)
    save_json(output / 'validation/summary.json', summary)
    save_json(output / 'validation/run.json', {'config': config, 'scoring_config': score, 'selection': selection,
        'new_optimizer_updates': 450, 'historical_unique_parent_updates': 500, 'final_lifetime_updates_per_arm': 650,
        'test_inference_calls': 0, 'production_promoted': False,
        'checkpoint_paths': {mode: path.relative_to(ROOT).as_posix() for mode, path in paths.items()},
        'checkpoint_sha256': {mode: sha256(path) for mode, path in paths.items()},
        'source_sha256': {**producer_hashes(), 'balance22/evaluation.py': sha256(Path(__file__))},
        'input_provenance': provenance, 'training_summary': read(output / 'training-summary.json'),
        'scope': 'Matched 150+150+150 continuations, fixed final state/note selection on reused development; no independent test or production promotion.'})
    return summary
