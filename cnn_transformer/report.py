import csv
import html
import json
from pathlib import Path
from urllib.parse import quote

from baseline.download import ROOT
from baseline.report import roll_svg


def write_report(output,records,summary,run):
    directory=output/'comparison'
    training={'cnn-gru':json.loads((ROOT/'outputs/cnn_gru/training_summary.json').read_text()),
              'cnn-transformer':json.loads((output/'training_summary.json').read_text())}
    histories={'cnn-gru':json.loads((ROOT/'outputs/cnn_gru/history.json').read_text()),
               'cnn-transformer':json.loads((output/'history.json').read_text())}
    validation_choice=max(run['models'],key=lambda name:run['models'][name]['validation_onset_f1'])
    text=['# Stage 3: matched CNN-GRU / CNN-transformer comparison','',
          'Both models use the same CQT, CNN, projection, framewise heads, loss, train/validation/evaluation split, '
          'crop schedule, augmentation, optimizer, learning-rate schedule, batch size and update cap. '
          'The temporal encoder is replaced. Neither model uses autoregressive note tokens.','',
          '## Results on the same 12 evaluation excerpts','',
          '| Model | Onset F1 | Onset+offset F1 | String-aware onset F1 | Frame F1 | Pipeline time / 144s audio | Network time / 144s audio |',
          '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    table=[]
    for name,values in summary.items():
        fields=[name,f'{values["onset"]["macro_f1"]:.3f}',f'{values["onset_offset"]["macro_f1"]:.3f}',
                f'{values["tab_onset_f1"]:.3f}',f'{values["frame"]["macro_f1"]:.3f}',
                f'{values["total_processing_seconds"]:.3f}s',f'{values["median_network_seconds_total"]:.3f}s']
        text.append('| '+' | '.join(fields)+' |')
        table.append('<tr>'+''.join('<td>'+html.escape(field)+'</td>' for field in fields)+'</tr>')
    text+=['','F1 values are macro means over excerpts; micro metrics and counts are retained in summary.json. '
           'Both checkpoints were refreshed in the same process on the same device, with a full 12-second warmup '
           'for each model. Pipeline time includes fresh audio read, resampling, CQT, normalization, network, '
           'probability stitching, note decoding and MIDI saving.',
           'Network time sums the per-excerpt median of three inference passes. It includes normalization, window batching, '
           'host/device transfers and stitching; it excludes CQT and note/MIDI decoding. These are warm offline '
           'throughput measurements, not live onset latency. Models run sequentially; residual load/order effects remain.','',
           '## Budget, validation and memory','',
           '| Model | Parameters | Epochs / updates | Best validation onset F1 | Recorded training+validation time | Peak allocated training tensors |',
           '| --- | ---: | --- | ---: | ---: | ---: |']
    for name,value in training.items():
        text.append(f'| {name} | {value["parameter_count"]:,} | {value["epochs_completed"]} / {value["optimizer_steps"]} | '
                    f'{run["models"][name]["validation_onset_f1"]:.3f} | {value["training_and_validation_seconds"]:.2f}s | '
                    f'{value["gpu_peak_allocated_mib"] or 0:.1f} MiB |')
    text+=['','Training costs come from separate runs on the same laptop, so their wall-time ratio is approximate. '
           'Allocated CUDA tensors exclude CUDA context and other runtime/process memory.',
           f'The existing validation-only selection rule favors **{validation_choice}** for the next experiment. '
           'This is a practical choice under the recorded short budget, not a general architecture ranking. '
           'Use the evaluation scores as diagnostic evidence; no settings were tuned on those excerpts.','',
           '## Learning under the cap','',
           '| Epoch | GRU training loss | GRU validation onset F1 | Transformer training loss | Transformer validation onset F1 |',
           '| --- | ---: | ---: | ---: | ---: |']
    for epoch in range(max(len(history) for history in histories.values())):
        values=[]
        for name in ['cnn-gru','cnn-transformer']:
            history=histories[name]
            values.extend([f'{history[epoch]["training_losses"]["total"]:.3f}',f'{history[epoch]["validation"]["onset_f1"]:.3f}']
                          if epoch<len(history) else ['-','-'])
        text.append('| '+str(epoch+1)+' | '+' | '.join(values)+' |')
    control=run['control_checks']
    text+=['','The training histories do not demonstrate convergence. One seed and 12 excerpts are insufficient '
           'for a robust claim that one temporal architecture is universally better. Longer-budget or multiple-seed '
           'comparisons are separate experiments; none were added here.','',
           '## Control checks','',
           f'Parameter counts differ by {control["parameter_difference_percent"]:.3f}%. Shared CNN, projection and '
           'all output-head initial weights are bit-identical for seed 42. The transformer constructor recreates '
           'the baseline initialization order, discards the untrained GRU, and builds its temporal encoder. '
           'No trained GRU weights are transferred.',
           'The epoch optimization/validation loop is checked for AST equivalence. Dataset recipe and normalization '
           'are identical; selected crop/label tensors were compared across early and late epochs. Source and cache '
           'checksums are validated. These audit results are stored in run.json.',
           'Training/validation/evaluation counts are 48/12/12, using progressions 1/2/3. Players occur in all splits. '
           'Both model selection and decoder threshold selection use progression-2 validation. Both models use '
           'the corrected decoder that prevents same-string note overlaps.',
           'Temporal arithmetic is float32 in both models. The common CNN/projection/heads share the same CUDA '
           'AMP policy. Decoder threshold grids are equal, but validation may select different values for each model.','',
           '## Transformer architecture','',
           'The common CNN produces a 128-dimensional frame sequence. Sinusoidal positions feed four bidirectional '
           'transformer encoder blocks, with four attention heads, feed-forward width 192, pre-normalization and '
           'dropout 0.1. A learned 128-to-256 projection preserves the exact output-head dimensions of the GRU. '
           'The feed-forward width was chosen to match total parameter counts closely. Encoder layers are '
           'initialized independently rather than retaining identical cloned initial values.',
           'Inference uses the same overlapping two-second windows. Attention is bidirectional, CQT is centered, '
           'and this is an offline prototype. No causal/streaming claim is made.','',
           '## Artifacts and next stage','',
           '[Audio and paired note plots](comparison/review.html). Each model has MIDI, timed string/fret CSV, '
           'probability arrays and per-excerpt note/error JSON. The original GRU outputs/checkpoints remain unchanged.',
           'Next, investigate held-out guitar tones/effects with the selected baseline, retaining an explicit '
           'accuracy/cost tradeoff. Current results are acoustic GuitarSet measurements and do not establish '
           'robustness to electric guitar, commercial full-band mixtures or unseen recording setups.','']
    (output/'report.md').write_text('\n'.join(text),encoding='utf-8')
    sections=[]
    for clip_id in run['selected_ids']:
        encoded=quote(clip_id,safe='')
        body=f'<section><h2>{html.escape(clip_id)}</h2><audio controls preload="none" src="clips/{encoded}.wav"></audio>'
        for name,rows in records.items():
            row=next(value for value in rows if value['id']==clip_id)
            body+=f'<h3>{name}: onset F1 {row["onset"]["f1"]:.3f}; string-aware F1 {row["tablature"]["onset"]["f1"]:.3f}</h3>'
            body+=roll_svg(row['reference'],row['predicted'],row['onset_matches'],row['score_duration'],name)
            body+=f'<p><a href="{name}/{encoded}.mid">MIDI</a> &middot; <a href="{name}/{encoded}-tab.csv">Timed tab</a> '
            body+=f'&middot; <a href="{name}/{encoded}.json">Notes and metrics</a></p>'
        sections.append(body+'</section>')
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GRU versus transformer guitar comparison</title>'
    page+='<style>body{background:#0b1421;color:#e8eef8;font:16px system-ui;max-width:1150px;margin:40px auto;padding:0 22px;line-height:1.5}section{margin:30px 0;padding:22px;background:#142135;border-radius:12px}svg{width:100%;height:auto}a{color:#94bdff}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:9px;border-bottom:1px solid #33435a}audio{width:100%;max-width:600px}h2{font-size:21px}h3{font-size:16px}.table{overflow-x:auto}</style>'
    page+='<h1>CNN-GRU versus CNN-transformer</h1><p>Same features, shared starting weights, data and 500-update cap. '
    page+='One seed; offline evaluation. F1 means across 12 excerpts.</p><div class="table"><table><tr>'
    page+=''.join('<th>'+value+'</th>' for value in ['Model','Onset F1','Onset+offset F1','String-aware F1','Frame F1','Pipeline time','Network time'])
    page+='</tr>'+''.join(table)+'</table></div><p>Upper bars: reference; lower bars: predictions. '
    page+='Blue/green are onset matches, purple is missed reference, red is extra prediction. Hover for timing.</p>'
    page+=''.join(sections)+'<p><a href="../report.md">Full protocol, controls and limitations</a></p></html>'
    (directory/'review.html').write_text(page,encoding='utf-8')
    with (directory/'metrics.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream)
        writer.writerow(['model','id','onset_f1','onset_offset_f1','tab_onset_f1','pipeline_seconds','median_network_seconds'])
        for name,rows in records.items():
            for row in rows:
                writer.writerow([name,row['id'],row['onset']['f1'],row['onset_offset']['f1'],
                                 row['tablature']['onset']['f1'],row['processing_seconds'],row['median_network_seconds']])
    print('Saved matched comparison:',output/'report.md',flush=True)
