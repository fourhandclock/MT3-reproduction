"""Readable summary and audio-linked note plots; no server or external assets."""

import csv
import html
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np


def collect(directory):
    metadata = json.loads((directory / 'run.json').read_text())
    records = {}
    for model in metadata['models']:
        records[model] = []
        for clip_id in metadata['selected_ids']:
            path = directory / model / (clip_id + '.json')
            if path.exists():
                records[model].append(json.loads(path.read_text()))
    return metadata, records


def summarize(records):
    output = {}
    for model, rows in records.items():
        if not rows:
            continue
        value = {'excerpt_count': len(rows), 'total_processing_seconds': sum(row['processing_seconds'] for row in rows),
                 'total_input_seconds': sum(row['input_duration_seconds'] for row in rows),
                 'total_missed_notes': sum(row['errors']['missed_notes'] for row in rows),
                 'total_extra_notes': sum(row['errors']['extra_notes'] for row in rows),
                 'drum_notes': sum(row['drum_notes'] for row in rows)}
        value['real_time_factor'] = value['total_processing_seconds'] / value['total_input_seconds']
        value['gpu_peak_allocated_mib'] = max((row['gpu_peak_allocated_mib'] or 0 for row in rows), default=0)
        for metric in ['onset', 'onset_offset', 'frame']:
            mean = float(np.mean([row[metric]['f1'] for row in rows]))
            deviation = float(np.std([row[metric]['f1'] for row in rows]))
            tp = sum(row[metric]['true_positives'] for row in rows)
            ref, est = sum(row[metric]['reference_count'] for row in rows), sum(row[metric]['predicted_count'] for row in rows)
            value[metric] = {'macro_f1': mean, 'std_across_excerpts': deviation,
                             'micro_f1': 2 * tp / (ref + est) if ref + est else 0.0,
                             'true_positives': tp, 'reference_count': ref, 'predicted_count': est}
        output[model] = value
    return output


def roll_svg(reference, predicted, matches, duration, name):
    width, height, left, top = 1000, 235, 45, 20
    notes = reference + predicted
    lowest = min([note['pitch'] for note in notes] + [40]) - 1
    highest = max([note['pitch'] for note in notes] + [76]) + 1
    plot_height = height - 2 * top
    scale = plot_height / (highest - lowest + 1)
    ref_matched, est_matched = {i for i, _ in matches}, {j for _, j in matches}
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(name)} reference and predicted notes">',
             f'<rect width="{width}" height="{height}" fill="#101c2d"/>']
    for second in range(int(duration) + 1):
        x = left + second / duration * (width - left - 10)
        parts.append(f'<path d="M{x:.1f},{top} V{height-top}" stroke="#263448"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-3}" fill="#9aaec9" font-size="11">{second}s</text>')
    for pitch in range(lowest, highest + 1):
        if pitch % 12 == 0:
            y = top + (highest - pitch) * scale
            parts.append(f'<text x="3" y="{y+5:.1f}" fill="#9aaec9" font-size="11">C{pitch//12-1}</text>')
    for layer, matched, is_prediction in [(reference, ref_matched, False), (predicted, est_matched, True)]:
        for index, note in enumerate(layer):
            start, end = max(0, note['start']), min(duration, note['end'])
            if start >= duration or end <= start:
                continue
            x = left + start / duration * (width - left - 10)
            length = max(1, (end - start) / duration * (width - left - 10))
            y = top + (highest - note['pitch']) * scale + (scale * 0.45 if is_prediction else 0)
            color = ('#58d9a2' if index in matched else '#ff7b7b') if is_prediction else ('#739cff' if index in matched else '#c8a1ff')
            label = f'{"Prediction" if is_prediction else "Reference"}: MIDI {note["pitch"]}, {note["start"]:.3f}-{note["end"]:.3f}s'
            if not is_prediction:
                label += f', string {note.get("string", "?")}, fret {note.get("fret", "?")}'
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{length:.1f}" height="{max(1,scale*0.42):.1f}" fill="{color}"><title>{html.escape(label)}</title></rect>')
    parts.append('</svg>')
    return ''.join(parts)


def write_report(directory):
    directory = Path(directory)
    metadata, records = collect(directory)
    summary = summarize(records)
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    lines = ['# Stage 1: pretrained guitar transcription baseline', '',
             f'Selected excerpts: {len(metadata["selected_ids"])}. Training performed: no.', '',
             'This is a small diagnostic experiment, not a reproduction of the full MT3 benchmark.', '',
             '| Baseline | Completed | Onset F1 | Onset+offset F1 | Frame F1 | Processing/audio | Device |',
             '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    table = []
    for model, value in summary.items():
        device = metadata['models'][model]['device']
        fields = [model, str(value['excerpt_count']), f'{value["onset"]["macro_f1"]:.3f}',
                  f'{value["onset_offset"]["macro_f1"]:.3f}', f'{value["frame"]["macro_f1"]:.3f}',
                  f'{value["real_time_factor"]:.3f}', device]
        lines.append('| ' + ' | '.join(fields) + ' |')
        table.append('<tr>' + ''.join(f'<td>{html.escape(field)}</td>' for field in fields) + '</tr>')
    lines += ['', 'F1 values are means across completed excerpts. Processing/audio is total measured warm processing time divided by input duration. '
              'Lower values mean faster processing. CPU/GPU differences prevent interpreting this as a controlled architecture speed comparison.', '',
              '## Evaluation protocol', '',
              f'Input: first {metadata["config"]["clip_seconds"]:g} seconds. Score note onsets and frames in the first '
              f'{metadata["config"]["score_seconds"]:g} seconds; later audio provides release context.',
              'Pitch matching: 50 cents after rounding reference MIDI labels. Onset tolerance: 50 ms. '
              'Offset tolerance: max(50 ms, 20% of reference duration). Frame rate: 62.5 Hz.',
              'Notes whose reference offsets extend beyond the input are censored from offset evaluation; '
              'their onset-matched predictions are removed from that metric too. Counts are recorded per excerpt.',
              'All non-drum predicted pitches are evaluated, regardless of instrument label. '
              'Drum hallucinations and MT3 program labels are reported separately. No oracle guitar-program filtering.', '',
              '## Model and data provenance', '',
              'The MT3 backend is the unofficial kunato port through a pinned openmirlab adapter. Automatic instrument filtering is disabled. '
              'The pretrained checkpoint is SHA256-verified. The default mt3-reference replaces the centered-power frontend with '
              'a magnitude/end-padded frontend checked against numerical fixtures from Google\'s actual TensorFlow spectral_ops. '
              'Full original Google JAX model/decoder parity has not been independently established. '
              'An optional mt3-pytorch run retains the unmodified third-party frontend.',
              'Basic Pitch uses its released ONNX model on CPU, with default decoding thresholds and a 127.7 ms minimum note length. '
              'The session uses four CPU threads.',
              'Selection follows the progression-3 validation convention described in the MT3 paper. '
              'GuitarSet pretraining overlap, particularly for Basic Pitch, is not independently resolved. '
              'These numbers describe this diagnostic subset; they do not establish unseen-electric-guitar generalization.',
              'Raw filenames, checksums, selected IDs, configuration, software versions, load/warmup times and measured timing scope '
              'are preserved in data/manifest.json and this directory\'s run.json.', '',
              '## Error review', '',
              'Octave/timing candidate counts are heuristics among unmatched notes, not mutually exclusive diagnoses. '
              'Inspect synchronized audio and note plots before assigning causes.', '',
              '| Model | Excerpt | Onset F1 | Onset+offset F1 | Missed | Extra | Octave candidates | Timing candidates |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for model, rows in records.items():
        for row in sorted(rows, key=lambda item: item['onset']['f1']):
            errors = row['errors']
            lines.append(f'| {model} | {row["id"]} | {row["onset"]["f1"]:.3f} | {row["onset_offset"]["f1"]:.3f} | '
                         f'{errors["missed_notes"]} | {errors["extra_notes"]} | {errors["octave_error_candidates"]} | {errors["timing_error_candidates"]} |')
    lines += ['', '## Manual review checklist', '',
              'For the lowest-scoring examples, listen for missed attacks, spurious harmonics/octaves, repeated notes merged into one, '
              'incorrect sustain/release, and string-unison ambiguity. These are hypotheses, not automatic conclusions.',
              'Use review.html for audio, MIDI links and overlaid note intervals. JSON files retain matching pairs and reference string/fret labels.', '',
              '## Next experiment', '',
              'Use this pipeline to evaluate a small CNN-GRU on the same inputs. The initial planned training budget is '
              '5 epochs, at most 500 optimizer steps, batch size 4 and 2-second crops, with validation and early stopping. '
              'These caps are recorded preferences; no training implementation or training run is included in stage 1.', '']
    (directory / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    with (directory / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'id', 'onset_f1', 'onset_offset_f1', 'frame_f1', 'processing_seconds',
                         'real_time_factor', 'missed_notes', 'extra_notes', 'gpu_peak_allocated_mib'])
        for model, rows in records.items():
            for row in rows:
                writer.writerow([model, row['id'], row['onset']['f1'], row['onset_offset']['f1'], row['frame']['f1'],
                                 row['processing_seconds'], row['real_time_factor'], row['errors']['missed_notes'],
                                 row['errors']['extra_notes'], row['gpu_peak_allocated_mib']])
    sections = []
    for clip_id in metadata['selected_ids']:
        url_id = quote(clip_id, safe='')
        body = [f'<section><h2>{html.escape(clip_id)}</h2><audio controls preload="none" src="clips/{url_id}.wav"></audio>']
        for model, rows in records.items():
            row = next((record for record in rows if record['id'] == clip_id), None)
            if row is None:
                continue
            body.append(f'<h3>{html.escape(model)} &middot; onset F1 {row["onset"]["f1"]:.3f} &middot; onset+offset F1 {row["onset_offset"]["f1"]:.3f}</h3>')
            body.append(roll_svg(row['reference'], row['predicted'], row['onset_matches'], row['score_duration'], model))
            body.append(f'<p><a href="{model}/{url_id}.mid">Download MIDI</a> &middot; '
                        f'<a href="{model}/{url_id}.json">Notes and metrics</a></p>')
        body.append('</section>')
        sections.append(''.join(body))
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    page += '<title>Guitar transcription baseline review</title><style>body{background:#0b1421;color:#e8eef8;font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 22px;line-height:1.5}'
    page += 'h1{font-size:32px}h2{font-size:21px}h3{font-size:16px;color:#c3d1e6}section{margin:32px 0;padding:22px;background:#142135;border-radius:12px}svg{width:100%;height:auto}a{color:#94bdff}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:10px;border-bottom:1px solid #33435a}audio{width:100%;max-width:600px}.legend{color:#b6c6db}</style>'
    page += '<h1>Guitar transcription: stage 1</h1><p>Pretrained models on a small acoustic-guitar diagnostic subset. No training.</p>'
    page += '<p>F1 is accuracy on these excerpts; processing/audio measures warm offline throughput. MT3 uses GPU; Basic Pitch uses CPU. '
    page += 'These runs do not establish live latency or unseen-electric-guitar accuracy.</p><table><thead><tr>'
    page += ''.join('<th>' + value + '</th>' for value in ['Model', 'Clips', 'Onset F1', 'Onset+offset F1', 'Frame F1', 'Processing/audio', 'Device'])
    page += '</tr></thead><tbody>' + ''.join(table) + '</tbody></table>'
    page += '<p class="legend">Upper bars: reference, blue when onset-matched and purple when missed. Lower bars: predictions, green when matched and red when extra. '
    page += 'Hover for pitch/timing. Matching colors concern onsets; differing bar lengths reveal offset errors. Reference string indices run from 0 (low E) to 5 (high E).</p>'
    page += ''.join(sections) + '<p>Full protocol and limitations: <a href="report.md">report.md</a>. <a href="metrics.csv">Per-excerpt CSV</a>.</p></html>'
    (directory / 'review.html').write_text(page, encoding='utf-8')
    print('Reports:', directory / 'report.md', 'and', directory / 'review.html', flush=True)
