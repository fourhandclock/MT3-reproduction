import json
import time
from pathlib import Path

import mir_eval
import numpy as np

from baseline.download import ROOT, sha256
from baseline.evaluation import evaluate, reference_notes
from .chords import accuracy, estimate
from .grid import make_grid, quantize, to_seconds
from .notation import export_score


def references(path):
    document = json.loads(Path(path).read_text())
    beat = next(a for a in document['annotations'] if a['namespace'] == 'beat_position')
    rows = beat['data']
    chords = [a for a in document['annotations'] if a['namespace'] == 'chord']
    return {'beats': [float(r['time']) for r in rows],
            'downbeats': [float(r['time']) for r in rows if r['value']['position'] == 1],
            'meters': sorted(set((int(r['value']['num_beats']), int(r['value']['beat_units'])) for r in rows)),
            'instructed_chords': [{'start': r['time'], 'end': r['time']+r['duration'], 'label': r['value']} for r in chords[0]['data']],
            'performed_chords': [{'start': r['time'], 'end': r['time']+r['duration'], 'label': r['value']} for r in chords[-1]['data']],
            'performed_chord_provenance': chords[-1]['annotation_metadata']}


def event_metrics(reference, predicted, duration, tolerance):
    ref = np.asarray([t for t in reference if 0 <= t < duration])
    est = np.asarray([t for t in predicted if 0 <= t < duration])
    matches = mir_eval.util.match_events(ref, est, tolerance)
    precision = len(matches)/len(est) if len(est) else 0
    recall = len(matches)/len(ref) if len(ref) else 0
    return {'precision': precision, 'recall': recall, 'f1': 2*precision*recall/(precision+recall) if precision+recall else 0,
            'reference_count': len(ref), 'predicted_count': len(est), 'matches': len(matches),
            'mean_matched_absolute_error_ms': float(np.mean([abs(ref[a]-est[b])*1000 for a, b in matches])) if matches else None}


def score_document(notes, grid, config, chords=None, subdivision=None):
    document = quantize(notes, grid, config, subdivision)
    document['tuning_midi'] = config['tuning_midi']
    document['chords'] = chords if chords is not None else estimate(notes, grid, config)
    document['chord_method'] = 'Fixed pitch-class templates + Viterbi; local margins are not calibrated confidence.'
    return document


def quantized_notes(document):
    grid = document['grid']; result = []
    for note in document['notes']:
        value = dict(note['raw'])
        value.update(start=max(0, to_seconds(note['start_tick']/12, grid)),
                     end=max(0, to_seconds(note['end_tick']/12, grid)))
        result.append(value)
    return result


def score_metrics(document):
    notes = document['notes']
    return {'note_count': len(notes), 'raw_note_fields_preserved': True,
            'mean_absolute_onset_shift_ms': float(np.mean([abs(n['onset_shift_seconds'])*1000 for n in notes])) if notes else 0,
            'mean_absolute_offset_shift_ms': float(np.mean([abs(n['offset_shift_seconds'])*1000 for n in notes])) if notes else 0,
            'large_shift_notes': sum('large quantization shift' in n['flags'] for n in notes),
            'quantized_string_overlap_notes': sum('quantized string overlap' in n['flags'] for n in notes),
            'minimum_duration_notes': sum('minimum written duration' in n['flags'] for n in notes),
            'unresolved_positions': sum(n['raw'].get('string') is None for n in notes)}


def evaluate_rhythm(config, output, device):
    from .tracker import Tracker
    import torch
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT/'data/cnn_gru/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    score_config = json.loads((ROOT/'configs/stage1.json').read_text())
    tracker = Tracker(config, device)
    # Warm the same twelve-second shape once; per-clip timing is recorded separately.
    tracker.predict(ROOT/manifest['validation'][0]['audio'], config['input_seconds'])
    records = []; audits = []; started = time.perf_counter()
    for split in ['validation', 'test']:
        for entry in manifest[split]:
            for field in ['audio', 'annotation']:
                if sha256(ROOT/entry[field]) != entry[field+'_sha256']: raise ValueError('Dataset input changed.')
            ref = references(ROOT/entry['annotation']); audits.append({'id': entry['id'], 'split': split, 'meters': ref['meters']})
            prediction = tracker.predict(ROOT/entry['audio'], config['input_seconds'])
            folder = output/'beats'/split; folder.mkdir(parents=True, exist_ok=True)
            arrays = {k: prediction.pop(k) for k in ['beat_probability', 'downbeat_probability']}
            np.savez_compressed(folder/(entry['id']+'.npz'), **arrays)
            prediction['audio_sha256'] = entry['audio_sha256']
            prediction['checkpoint_sha256'] = tracker.provenance['sha256']
            (folder/(entry['id']+'.json')).write_text(json.dumps(prediction, indent=2), encoding='utf-8')
            duration = prediction['duration_seconds']; score_duration = min(duration, config['score_seconds'])
            predicted_grid = make_grid(prediction['beats'], prediction['downbeats'], duration)
            reference_grid = make_grid(ref['beats'], ref['downbeats'], duration)
            beat_metrics = {name: event_metrics(ref[name], prediction[name], score_duration, config['beat_tolerance_seconds'])
                            for name in ['beats', 'downbeats']}
            # Only a diagnostic over the currently all-4/4 reference dataset.
            bars = [b for b in predicted_grid['bars'] if not b['pickup'] and b['start_seconds'] < score_duration]
            meter_correct = sum(max(0, min(score_duration, b['end_seconds'])-max(0, b['start_seconds']))
                                for b in bars if (b['numerator'], b['denominator']) in ref['meters'])
            meter_total = sum(max(0, min(score_duration, b['end_seconds'])-max(0, b['start_seconds'])) for b in bars)
            rhythm_record = {'id': entry['id'], 'split': split, 'beats': beat_metrics, 'processing_seconds': prediction['processing_seconds'],
                             'provisional_meter_duration_accuracy': meter_correct/meter_total if meter_total else 0,
                             'meter_scored_seconds': meter_total, 'proposal': predicted_grid['observed_pulses_per_bar']}
            (folder/(entry['id']+'-metrics.json')).write_text(json.dumps(rhythm_record, indent=2), encoding='utf-8')
            reference = [n for n in reference_notes(ROOT/entry['annotation']) if n['start'] < duration]
            variants = [('oracle_notes_oracle_beats', reference, reference_grid)]
            if split == 'test':
                path = ROOT/'outputs/fingering/evaluation/predicted/constrained_acoustic'/(entry['id']+'.json')
                original = json.loads(path.read_text()); predicted_notes = original['notes']
                variants += [('oracle_notes_predicted_beats', reference, predicted_grid),
                             ('predicted_notes_oracle_beats', predicted_notes, reference_grid),
                             ('predicted_notes_predicted_beats', predicted_notes, predicted_grid)]
            for scenario, notes, grid in variants:
                document = score_document(notes, grid, config)
                if [n['raw'] for n in document['notes']] != notes: raise ValueError('Notation mutated raw notes.')
                document['metadata'] = {'id': entry['id'], 'scenario': scenario, 'split': split,
                                        'audio': str((ROOT/entry['audio']).resolve()),
                                        'training_overlap': 'Pretrained beat model trained on GuitarSet; integration-only rhythm scores.',
                                        'annotation_provenance': ref['performed_chord_provenance']}
                stem = output/'scores'/split/scenario/entry['id']
                export_score(document, stem, entry['id']+' / '+scenario)
                raw_scores = evaluate(reference, notes, score_config, duration)
                snapped_scores = evaluate(reference, quantized_notes(document), score_config, duration)
                record = {'id': entry['id'], 'split': split, 'scenario': scenario, 'rhythm': rhythm_record,
                          'notation': score_metrics(document), 'raw_pitch_onset_f1': raw_scores['onset']['f1'],
                          'quantized_pitch_onset_f1': snapped_scores['onset']['f1'],
                          'chords_performed': accuracy(ref['performed_chords'], document['chords'], score_duration),
                          'chords_instructed': accuracy(ref['instructed_chords'], document['chords'], score_duration),
                          'score_file': str(stem.relative_to(output).with_suffix('.json'))}
                stem.with_name(stem.name+'-metrics.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
                records.append(record)
            print(split, entry['id'], 'beat F1', round(beat_metrics['beats']['f1'], 3),
                  'downbeat F1', round(beat_metrics['downbeats']['f1'], 3), flush=True)
    summary = aggregate(records)
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    run = {'config': config, 'optimizer_updates': 0, 'new_training_epochs': 0, 'beat_model': tracker.provenance,
           'model_parameters': tracker.parameters, 'device': str(device), 'peak_gpu_allocated_bytes': tracker.peak_memory,
           'processing_elapsed_seconds': time.perf_counter()-started, 'manifest_sha256': sha256(manifest_path),
           'fingering_selection_sha256': sha256(ROOT/'outputs/fingering/selection.json'),
           'meter_audit': audits, 'all_parameters_prespecified': True, 'tuned_on_test': False,
           'source_sha256': {p.name: sha256(p) for p in sorted((ROOT/'rhythm').glob('*.py'))}}
    (output/'run.json').write_text(json.dumps(run, indent=2), encoding='utf-8')
    del tracker
    if str(device).startswith('cuda'): torch.cuda.empty_cache()
    from .report import write_report
    write_report(output, records, summary, run)


def aggregate(records):
    result = {}
    for split in ['validation', 'test']:
        rows = [r for r in records if r['split'] == split and r['scenario'] == 'oracle_notes_oracle_beats']
        result[split] = {'count': len(rows),
                        'beat_f1': float(np.mean([r['rhythm']['beats']['beats']['f1'] for r in rows])),
                        'downbeat_f1': float(np.mean([r['rhythm']['beats']['downbeats']['f1'] for r in rows])),
                        'provisional_meter_duration_accuracy': float(np.mean([r['rhythm']['provisional_meter_duration_accuracy'] for r in rows])),
                        'warm_processing_seconds': sum(r['rhythm']['processing_seconds'] for r in rows)}
    result['test_scenarios'] = {}
    for scenario in sorted(set(r['scenario'] for r in records if r['split'] == 'test')):
        rows = [r for r in records if r['split'] == 'test' and r['scenario'] == scenario]
        result['test_scenarios'][scenario] = {'count': len(rows),
            'raw_pitch_onset_f1': float(np.mean([r['raw_pitch_onset_f1'] for r in rows])),
            'quantized_pitch_onset_f1': float(np.mean([r['quantized_pitch_onset_f1'] for r in rows])),
            'chord_performed_accuracy': float(np.mean([r['chords_performed']['major_minor_duration_accuracy'] for r in rows])),
            'chord_instructed_accuracy': float(np.mean([r['chords_instructed']['major_minor_duration_accuracy'] for r in rows])),
            'chord_reference_coverage': float(np.mean([r['chords_performed']['coverage_fraction'] for r in rows])),
            'mean_absolute_onset_shift_ms': float(np.mean([r['notation']['mean_absolute_onset_shift_ms'] for r in rows])),
            'large_shift_notes': sum(r['notation']['large_shift_notes'] for r in rows),
            'quantized_string_overlap_notes': sum(r['notation']['quantized_string_overlap_notes'] for r in rows),
            'minimum_duration_notes': sum(r['notation']['minimum_duration_notes'] for r in rows),
            'notes': sum(r['notation']['note_count'] for r in rows)}
    return result


def regenerate_exports(output):
    """Regenerate notation from saved source events; no model inference or parameter selection."""
    output = Path(output).resolve()
    for path in sorted(output.glob('scores/*/*/*.json')):
        if path.name.endswith('-metrics.json'): continue
        document = json.loads(path.read_text())
        export_score(document, path.with_suffix(''), document['metadata']['id']+' / '+document['metadata']['scenario'])
    run_path = output/'run.json'; run = json.loads(run_path.read_text())
    run.setdefault('source_sha256_at_inference', run['source_sha256'])
    run['source_sha256'] = {p.name:sha256(p) for p in sorted((ROOT/'rhythm').glob('*.py'))}
    run['notation_exports_regenerated_from_saved_notes'] = True
    run_path.write_text(json.dumps(run, indent=2), encoding='utf-8')
    print('Regenerated saved notation exports; no inference or training.', flush=True)
