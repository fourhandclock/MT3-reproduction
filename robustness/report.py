import csv
import html
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np

from baseline.download import ROOT
from baseline.report import roll_svg


def write_report(output,records,summary,run):
    destination=output/'evaluation'
    training=json.loads((output/'augmented/training_summary.json').read_text())
    history=json.loads((output/'augmented/history.json').read_text())
    baseline,augmented=summary['clean-trained'],summary['effect-augmented']
    conditions=list(run['config']['robustness']['test_conditions'])+(['real_electric'] if 'real_electric' in baseline else [])
    table=[]
    lines=['# Stage 4: tone/effect robustness','',
           'A frozen clean-trained CNN-GRU is compared with one CNN-GRU trained from scratch using mild waveform '
           'augmentation. Stronger analytic effects and a separate real electric-guitar dataset are evaluated. '
           'Synthetic processing and real recordings are reported separately.','',
           '## Note onset F1 by condition','',
           '| Condition | Clean-trained | Effect-augmented | Difference |',
           '| --- | ---: | ---: | ---: |']
    for condition in conditions:
        before,after=baseline[condition]['onset_f1'],augmented[condition]['onset_f1']
        fields=[condition,f'{before:.3f}',f'{after:.3f}',f'{after-before:+.3f}']
        lines.append('| '+' | '.join(fields)+' |')
        table.append('<tr>'+''.join('<td>'+html.escape(value)+'</td>' for value in fields)+'</tr>')
    effects=[name for name in conditions if name not in ['clean','real_electric']]
    effect_before=float(np.mean([baseline[name]['onset_f1'] for name in effects]))
    effect_after=float(np.mean([augmented[name]['onset_f1'] for name in effects]))
    clean_delta=augmented['clean']['onset_f1']-baseline['clean']['onset_f1']
    preferred=max(run['models'],key=lambda name:run['models'][name]['robust_validation']['onset_f1'])
    lines+=['',f'Mean onset F1 over the five processed-acoustic conditions: {effect_before:.3f} before, '
            f'{effect_after:.3f} after. Clean-acoustic change: {clean_delta:+.3f}.',
            'Values are macro means across twelve excerpts per condition. They are not whole-dataset or commercial-song accuracy. '
            'This is one seed and one small-budget trial; it does not establish a general robustness guarantee.','',
            '## Decision from this trial','',
            f'The common validation criterion favors **{preferred}**. The synthetic-effect mean changes by '
            f'{effect_after-effect_before:+.4f}; individual-condition gains and losses should be read alongside that average.',
            ('Primary real-electric onset F1 changes by '
             f'{augmented["real_electric"]["onset_f1"]-baseline["real_electric"]["onset_f1"]:+.4f}. '
             'The result does not justify claiming that this augmentation recipe solved acoustic-to-electric transfer.'
             if 'real_electric' in baseline else 'Independent real-electric evaluation is not available in this report.'),
            'The external score gap changes recording setup, performer, musical material and annotation provenance '
            'together. It does not isolate guitar/amp timbre as the sole cause. A future fine-tuning or augmentation '
            'experiment should use separate real-electric training/validation data and keep this evaluation set independent.','',
            '## String positions, durations and pitch activity','',
            '| Condition | Clean-trained onset+offset F1 | Augmented onset+offset F1 | Clean-trained string-aware onset F1 | Augmented string-aware onset F1 |',
            '| --- | ---: | ---: | ---: | ---: |']
    for condition in conditions:
        left,right=baseline[condition],augmented[condition]
        lines.append(f'| {condition} | {left["onset_offset_f1"]:.3f} | {right["onset_offset_f1"]:.3f} | '
                     f'{left["tab_onset_f1"]:.3f} | {right["tab_onset_f1"]:.3f} |')
    if 'real_electric' in baseline:
        lines+=['','## Independent real recording test','',
                'EGSet12 contains twelve original electric-guitar recordings from one guitarist using a Sire T7 guitar, '
                'Yamaha B15 amplifier and microphone capture. No additional audio effects were used in that dataset. '
                '[Dataset and attribution](https://zenodo.org/records/11406378).',
                'We use its first 12 seconds per recording and score the first 10 seconds, matching the prefix protocol. '
                'The WAV/JAMS files are MD5-verified against the published record and SHA256-recorded locally. '
                'These files do not participate in training, normalization, early stopping or threshold selection.',
                'Provided note/string annotations are retained without time shifting. Alignment has not been independently '
                'audited; the following preset 100 ms onset diagnostic supplements the primary 50 ms result and is not used '
                'for model selection.',
                f'100 ms diagnostic onset F1: clean-trained {baseline["real_electric"]["onset_f1_100ms_diagnostic"]:.3f}; '
                f'augmented {augmented["real_electric"]["onset_f1_100ms_diagnostic"]:.3f}.',
                f'Primary real-electric frame F1: clean-trained {baseline["real_electric"]["frame_f1"]:.3f}; '
                f'augmented {augmented["real_electric"]["frame_f1"]:.3f}.',
                'One performer and recording setup cannot establish performance on arbitrary guitars, amps, styles or full mixes.']
    lines+=['','## Effects and label policy','',
            'EQ splits the full polyphonic waveform into low/high components around a crossover and applies gains. '
            'Overdrive uses a tanh waveshaper with 4x oversampling before downsampling, followed by optional low-pass '
            'cabinet-style coloration. Effects process the complete guitar signal rather than independent string tones.',
            'RMS is matched to the source, with a final 0.99 peak ceiling if necessary. Filters are zero-phase and '
            'resampling is centered, preserving sample count and avoiding a fixed added delay. Onsets/offsets and '
            'string/fret labels are unchanged; these operations can still change perceptual attack/release clarity.',
            'These are analytic offline timbre simulations on acoustic recordings, not calibrated amp/cabinet models '
            'or new real electric-guitar performances. Delay/reverb repetitions and pitch shifting are excluded.',
            'Train EQ parameters are randomized per source recording at mild levels; drive ranges from 6-16 dB '
            'with a 4-7 kHz low-pass cutoff. Training samples are 50% clean, 25% EQ and 25% overdrive. '
            'Test EQ/drive strengths extend beyond those ranges, and the combined effect chain is absent from training.','',
            '## Budget and validation controls','',
            f'The augmented model has {training["parameter_count"]:,} parameters and completed '
            f'{training["epochs_completed"]} epochs / {training["optimizer_steps"]} updates. '
            f'Training plus validation took {training["training_and_validation_seconds"]:.2f} seconds; '
            f'peak allocated training CUDA tensors were {training["gpu_peak_allocated_mib"] or 0:.1f} MiB.',
            'Architecture, random initialization seed, source recordings, crop positions, gain augmentation, labels, '
            'optimizer, loss, cosine schedule, clean-training normalization and update cap are unchanged. '
            'A separate RNG selects effect views without changing source/crop/gain draws. The optimization loop '
            'is checked for AST equivalence. No weights are fine-tuned or copied from the trained baseline.',
            'The augmented checkpoint is selected on progression-2 validation averaged across clean, mild EQ and '
            'moderate-drive conditions. The frozen baseline decoder is recalibrated on the same conditions/grid '
            'without optimizer updates. Original baseline checkpoints and prior experiment outputs remain intact.',
            'The baseline weights were originally selected on clean validation; the original best and final budget '
            'checkpoint are both epoch 5. This differs from the augmented selection objective and limits the ablation.',
            'The next table holds the baseline-calibrated decoder thresholds fixed for both models, helping distinguish '
            'training gains from threshold recalibration.','',
            '| Condition | Baseline with common thresholds | Augmented with baseline thresholds |',
            '| --- | ---: | ---: |']
    for condition in conditions:
        lines.append(f'| {condition} | {baseline[condition]["fixed_baseline_threshold_onset_f1"]:.3f} | '
                     f'{augmented[condition]["fixed_baseline_threshold_onset_f1"]:.3f} |')
    lines+=['','| Epoch | Training loss | Mixed validation onset F1 | Clean validation onset F1 | EQ validation onset F1 | Drive validation onset F1 |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in history:
        validation=row['validation'];condition=validation['condition_onset_f1']
        lines.append(f'| {row["epoch"]} | {row["training_losses"]["total"]:.3f} | {validation["onset_f1"]:.3f} | '
                     f'{condition["clean"]:.3f} | {condition["eq"]:.3f} | {condition["overdrive"]:.3f} |')
    lines+=['','## Review and scope','',
            '[Listen and inspect selected paired examples](evaluation/review.html). Full per-condition/per-record '
            'note metrics, MIDI and timed tablature CSVs are in evaluation/. Cached acoustic features are used for '
            'the stress evaluation, so its execution time is not an end-to-end deployment benchmark.',
            'Effect-based augmentation is established prior work. This experiment tests a specific cheap augmentation '
            'recipe under the project budget rather than claiming a new augmentation method. '
            '[DAFx authors and related work](https://robust-guitar-tabs.github.io/).','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    selected=['00_Rock3-148-C_comp','00_Funk3-98-A_solo','02_BN3-119-G_comp','EGSet12_01','EGSet12_06','EGSet12_09']
    sections=[]
    for clip_id in selected:
        for condition in conditions:
            first=next((row for row in records['clean-trained'] if row['id']==clip_id and row['condition']==condition),None)
            if first is None: continue
            stem=clip_id+'__'+condition;encoded=quote(stem,safe='')
            body=f'<section data-condition="{condition}"><h2>{html.escape(clip_id)} · {html.escape(condition)}</h2>'
            body+=f'<audio controls preload="none" src="clips/{encoded}.wav"></audio>'
            for model,rows in records.items():
                row=next(value for value in rows if value['id']==clip_id and value['condition']==condition)
                body+=f'<h3>{html.escape(model)} · onset F1 {row["onset"]["f1"]:.3f} · string-aware F1 {row["tablature"]["onset"]["f1"]:.3f}</h3>'
                body+=roll_svg(row['reference'],row['predicted'],row['onset_matches'],row['score_duration'],model)
                body+=f'<p><a href="{model}/{encoded}.mid">MIDI</a> · <a href="{model}/{encoded}-tab.csv">Timed tab</a> · '
                body+=f'<a href="{model}/{encoded}.json">Notes and errors</a></p>'
            sections.append(body+'</section>')
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Guitar tone robustness review</title>'
    page+='<style>body{background:#0b1421;color:#e8eef8;font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 22px;line-height:1.5}section{margin:28px 0;padding:20px;background:#142135;border-radius:12px}svg{width:100%;height:auto}a{color:#94bdff}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:10px;border-bottom:1px solid #33435a}audio{width:100%;max-width:600px}h2{font-size:21px}h3{font-size:16px}select{padding:9px;background:#142135;color:white;border:1px solid #556b8b}.table{overflow-x:auto}</style>'
    page+='<h1>Tone and effect robustness</h1><p>Controlled processing of acoustic guitar and independent real electric recordings are separate tests. '
    page+='One small-budget augmentation trial; no universal accuracy claim.</p><div class="table"><table><tr><th>Condition</th><th>Clean-trained onset F1</th><th>Augmented onset F1</th><th>Difference</th></tr>'
    page+=''.join(table)+'</table></div><p>Show condition: <select id="condition"><option value="all">All</option>'
    page+=''.join(f'<option value="{name}">{html.escape(name)}</option>' for name in conditions)+'</select></p>'
    page+='<p>Blue/green bars are reference/prediction onset matches; purple is missed reference and red is extra prediction. Hover for timing. '
    page+='Selected examples below; full scores are in the CSV/JSON outputs.</p>'+''.join(sections)
    page+='<p><a href="../report.md">Protocol, training history and limitations</a></p><script>document.getElementById("condition").addEventListener("change",function(){document.querySelectorAll("section").forEach(s=>s.hidden=this.value!=="all"&&s.dataset.condition!==this.value)});</script></html>'
    (destination/'review.html').write_text(page,encoding='utf-8')
    with (destination/'metrics.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['model','id','condition','onset_f1','onset_offset_f1','frame_f1','tab_onset_f1','fixed_baseline_threshold_onset_f1'])
        for model,rows in records.items():
            for row in rows:
                writer.writerow([model,row['id'],row['condition'],row['onset']['f1'],row['onset_offset']['f1'],row['frame']['f1'],
                                 row['tablature']['onset']['f1'],row['fixed_baseline_threshold_onset_f1']])
    print('Saved robustness report:',output/'report.md',flush=True)
