import csv
import html
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np

from baseline.download import ROOT
from baseline.report import roll_svg,summarize


def write_report(output,rows,run):
    destination=output/'evaluation'
    summary=summarize({'cnn-gru':rows})['cnn-gru']
    summary['tab_onset_f1']=float(np.mean([row['tablature']['onset']['f1'] for row in rows]))
    summary['tab_onset_offset_f1']=float(np.mean([row['tablature']['onset_offset']['f1'] for row in rows]))
    summary['tab_frame_f1']=float(np.mean([row['tablature']['frame']['f1'] for row in rows]))
    (destination/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    baseline=json.loads((ROOT/'outputs/stage1/summary.json').read_text())
    baseline['cnn-gru']=summary
    training=json.loads((output/'training_summary.json').read_text())
    history=json.loads((output/'history.json').read_text())
    table=[]
    lines=['# CNN-GRU: limited-budget guitar transcription run','',
           f'Parameters: {training["parameter_count"]:,}. Optimizer steps: {training["optimizer_steps"]}. '
           f'Epochs completed: {training["epochs_completed"]}. Selected epoch: {run["selected_epoch"]}.',
           f'Training plus validation: {training["training_and_validation_seconds"]:.1f} seconds. '
           f'Peak training CUDA tensor allocation: {training["gpu_peak_allocated_mib"] or 0:.1f} MiB. Stop reason: {training["stop_reason"]}.','',
           '## Same 12-excerpt evaluation','',
           '| Model | Mean onset F1 | Mean onset+offset F1 | Mean frame F1 | Processing 144 seconds of input |',
           '| --- | ---: | ---: | ---: | ---: |']
    for model,value in baseline.items():
        fields=[model,f'{value["onset"]["macro_f1"]:.3f}',f'{value["onset_offset"]["macro_f1"]:.3f}',
                f'{value["frame"]["macro_f1"]:.3f}',f'{value["total_processing_seconds"]:.2f}s']
        lines.append('| '+' | '.join(fields)+' |')
        table.append('<tr>'+''.join('<td>'+html.escape(field)+'</td>' for field in fields)+'</tr>')
    lines+=['','Mean F1 is the macro mean across excerpts. The CNN-GRU was trained from random initialization '
            'on a small subset with a short budget; the other two systems are pretrained. This is not an equal-training-resource architecture comparison.',
            'CNN-GRU and MT3 use CUDA; Basic Pitch uses CPU. Timing includes fresh acoustic features and note decoding, '
            'after warmup. It does not establish live latency.','',
            '## Tablature prediction','',
            f'Mean string-aware onset F1: {summary["tab_onset_f1"]:.3f}. '
            f'Mean string-aware onset+offset F1: {summary["tab_onset_offset_f1"]:.3f}. '
            f'Raw fret-class frame F1: {summary["tab_frame_f1"]:.3f}.',
            'String-aware matching requires the reference string as well as pitch/timing. String indices run from 0 (low E) '
            'to 5 (high E). Frets are 0-24 in standard tuning. Every decoded pitch follows open-string pitch + fret. '
            'This guarantees fretboard pitch consistency; it does not optimize hand span or recover bends.',
            'MIDI files have one named track per string. The matching -tab.csv files provide note timing, string, fret and pitch.','',
            '## Training and model selection','',
            '| Epoch | Steps | Mean train loss | Validation onset F1 | Validation string-aware onset F1 |',
            '| --- | ---: | ---: | ---: | ---: |']
    for entry in history:
        lines.append(f'| {entry["epoch"]} | {entry["optimizer_steps"]} | {entry["training_losses"]["total"]:.3f} | '
                     f'{entry["validation"]["onset_f1"]:.3f} | {entry["validation"]["tab_onset_f1"]:.3f} |')
    lines+=['','48 training recordings use progression 1; 12 validation recordings use progression 2; '
            'the unchanged stage-1 evaluation recordings use progression 3. Players are shared across splits, so this is '
            'not a held-out-player experiment. Normalization and class-balancing statistics use training data only. '
            'Checkpoint selection, early stopping and decoder thresholds use validation only.',
            f'Selected decoder thresholds: {run["decoder_parameters"]}. Test recordings did not select them.',
            'Release refinement is clipped at the next note on each string. The original training checkpoint is '
            'preserved; a derived inference checkpoint was calibrated on validation only after this decoding fix, '
            'without additional optimizer updates.',
            'Each epoch samples 400 random two-second crops, giving 100 batches at batch size 4. '
            'The run stops by 5 epochs or 500 steps, with early-stopping patience 2.','',
            '## Architecture and features','',
            '216 CQT bins, 36 bins/octave starting at E2, 16 kHz audio and a 160-sample hop (100 frames/second). '
            'Three CNN blocks (24/48/96 channels) pool frequency only; a 128-dimensional projection feeds two bidirectional '
            'GRU layers with 128 units per direction. Outputs are six 26-class fret distributions, six onsets, six offsets, '
            'and 49 auxiliary pitch activations. Frame labels preserve sample timestamps.',
            'CNN arithmetic can use float16 AMP on CUDA; recurrent arithmetic remains float32. '
            'Conflicting simultaneous frets on one string are masked from fret classification. '
            'Onset/offset labels use short Gaussian peaks, and crop edges are masked during training.',
            'Inference uses overlapping two-second windows and blends their predictions. Centered CQT and the bidirectional '
            'GRU use future context, so this implementation is offline.',
            'Cached features use float32. Timing recomputes CQT on each final evaluation recording rather than reusing that cache.','',
            '## Review and provenance','',
            '[Open audio and note review](evaluation/review.html). Check the note JSON and MIDI alongside the tablature CSV '
            'before treating predictions as playable transcriptions. No time signatures or score durations are inferred.',
            'Files preserve the configuration, split manifest/checksums, normalization, loss history, model source hashes, '
            'best/last checkpoints, selected thresholds and software versions. The pretrained references can overlap GuitarSet '
            'through their original training. These acoustic results do not establish unseen-electric-guitar robustness.','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    sections=[]
    for row in rows:
        name,url_name=row['id'],quote(row['id'],safe='')
        body=f'<section><h2>{html.escape(name)}</h2><audio controls preload="none" src="clips/{url_name}.wav"></audio>'
        body+=f'<p>Pitch onset F1 {row["onset"]["f1"]:.3f}; string-aware onset F1 {row["tablature"]["onset"]["f1"]:.3f}</p>'
        body+=roll_svg(row['reference'],row['predicted'],row['onset_matches'],row['score_duration'],'CNN-GRU')
        body+=f'<p><a href="cnn-gru/{url_name}.mid">MIDI</a> &middot; '
        body+=f'<a href="cnn-gru/{url_name}-tab.csv">Timed tablature CSV</a> &middot; <a href="cnn-gru/{url_name}.json">Notes and metrics</a></p></section>'
        sections.append(body)
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CNN-GRU guitar review</title>'
    page+='<style>body{background:#0b1421;color:#e8eef8;font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 22px;line-height:1.5}section{margin:30px 0;padding:22px;background:#142135;border-radius:12px}svg{width:100%;height:auto}a{color:#94bdff}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:10px;border-bottom:1px solid #33435a}audio{width:100%;max-width:600px}h2{font-size:21px}</style>'
    page+='<h1>CNN-GRU guitar transcription</h1><p>Initial training under a five-epoch / 500-step cap. '
    page+='Acoustic-guitar evaluation; this is an offline model with future context.</p><table><tr>'
    page+=''.join('<th>'+name+'</th>' for name in ['Model','Onset F1','Onset+offset F1','Frame F1','Processing time'])
    page+='</tr>'+''.join(table)+'</table><p>Models have different training resources and devices. '
    page+='Upper bars show reference notes; lower bars show predictions. Blue/green are onset matches, purple is missed reference, red is extra prediction. Hover for timing.</p>'
    page+=''.join(sections)+'<p><a href="../report.md">Protocol and training history</a></p></html>'
    (destination/'review.html').write_text(page,encoding='utf-8')
    with (destination/'metrics.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream)
        writer.writerow(['id','onset_f1','onset_offset_f1','frame_f1','tab_onset_f1','tab_onset_offset_f1','processing_seconds'])
        for row in rows:
            writer.writerow([row['id'],row['onset']['f1'],row['onset_offset']['f1'],row['frame']['f1'],
                             row['tablature']['onset']['f1'],row['tablature']['onset_offset']['f1'],row['processing_seconds']])
    print('Saved report:',output/'report.md',flush=True)
