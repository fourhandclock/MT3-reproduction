"""Local listening comparison backed by exact replay and export artifacts."""

import html
import json
from pathlib import Path
from urllib.parse import quote
from baseline.download import ROOT,sha256
from baseline.report import roll_svg
from decoder14.report import TEMPLATE as STYLE_SOURCE
from decoder14.workflow import read
from streaming.evaluate import save_json
from streaming.report import _table
from supervision16.report import TEMPLATE as PLAYER_SOURCE

LABELS={'short_short':'Short + short / 128 + 128 ms','short_long':'Short + long / 128 + 256 ms'}


def write_report(output):
    output=Path(output);run=read(output/'validation/run.json');summary=read(output/'validation/summary.json')
    records=[read(p) for p in sorted((output/'validation/results').glob('*/*-metrics.json'))]
    modes=summary['modes'];training=run['training_summary'];selected=run['selection']['selected_arm']
    control,treatment=modes['short_short'],modes['short_long']
    state_delta=treatment['exact_state_f1']-control['exact_state_f1'];onset_delta=treatment['onset_f1']-control['onset_f1']
    outcome=(f"Adding the long-window plane changes exact string/fret frame F1 by {state_delta:+.5f} and note-onset F1 by {onset_delta:+.5f}. "
             f"The declared development rule selects {LABELS[selected].lower()}.")
    headers=['Model','State F1','Note-onset F1','Onset + offset F1','String F1','Frame F1']
    score_rows=[[label]+[f"{modes[mode][key]:.4f}" for key in
                ['exact_state_f1','onset_f1','onset_offset_f1','tab_onset_f1','frame_f1']] for mode,label in LABELS.items()]
    pitch_rows=[[label]+[f"{modes[mode][key]:.4f}" for key in
                ['state_derived_pitch_f1','auxiliary_pitch_f1','low_register_pitch_f1']] for mode,label in LABELS.items()]
    cost_rows=[];latency_rows=[];history_rows=[]
    for mode,label in LABELS.items():
        receipt=training['arms'][mode]
        cost_rows.append([label,receipt['model_parameters'],receipt['new_optimizer_steps'],
            f"{receipt['training_seconds']:.1f}",f"{receipt['peak_gpu_mib']:.1f}",f"{modes[mode]['real_time_factor']:.3f}"])
        latency_rows.append([label,modes[mode]['matched_onset_count'],f"{modes[mode]['matched_onset_delivery_p50_ms']:.1f}",
            f"{modes[mode]['matched_onset_delivery_p95_ms']:.1f}",sum(r['pitch_metrics']['errors']['extra_notes'] for r in records if r['mode']==mode)])
        for row in receipt['history']:
            loss=row['training_losses']
            history_rows.append([mode,row['epoch'],row['optimizer_steps'],f"{loss['state']:.4f}",f"{loss['total']:.4f}"])
    tables={'scores':_table(headers,score_rows),
        'pitch':_table(['Model','State-derived pitch F1','Separate pitch head F1','MIDI 40–51 F1'],pitch_rows),
        'cost':_table(['Model','Parameters','New / lifetime updates','Training seconds','Peak tensor MiB','Processing / audio'],cost_rows),
        'latency':_table(['Model','Matched onsets','Completion p50 ms','Completion p95 ms','Official extra notes'],latency_rows),
        'history':_table(['Arm','Sampled epoch','New updates','State loss','Total loss'],history_rows)}
    payload=[]
    for row in records:
        mode,clip=row['mode'],row['id'];metrics=row['pitch_metrics'];stem=Path('validation/results')/mode/clip
        svg=roll_svg(metrics['reference'],metrics['predicted'],metrics['onset_matches'],10.,clip)
        svg=svg.replace('</svg>','<line id="source-playhead" x1="45" x2="45" y1="20" y2="215" stroke="white" stroke-width="2"/></svg>')
        payload.append({'id':clip,'mode':mode,'label':LABELS[mode],'notes':row['notes'],'events':row['events'],
            'svg':svg,'audio':quote(row['relativeaudio'],safe='/'),'f1':metrics['onset']['f1'],
            'json':(output/stem.with_suffix('.json')).read_bytes().decode('utf-8'),
            'links':{name:quote(stem.as_posix()+ext,safe='/') for name,ext in
                     [('JSON','.json'),('CSV','.csv'),('MIDI','.mid'),('Metrics','-metrics.json'),('Events','-events.json')]}})
    # Reuse the verified three-clock player; only its two comparison identifiers change.
    script=PLAYER_SOURCE.split('<script>',1)[1].split('</script>',1)[0]
    script=script.replace('state_pitch750','short_long').replace('control750','short_short')
    page=TEMPLATE.replace('__SCRIPT__',script)
    values={**tables,'outcome':html.escape(outcome),'payload':json.dumps(payload).replace('<','\\u003c'),
        'labels':json.dumps(LABELS),'selection':html.escape(json.dumps(run['selection'],indent=2)),
        'css':STYLE_SOURCE.split('<style>',1)[1].split('</style>',1)[0]}
    for key,value in values.items():page=page.replace('__'+key.upper()+'__',value)
    (output/'review.html').write_text(page,encoding='utf-8')
    lines=['# Stage17: matched short/long spectral fusion','',outcome,'',
        '500 new updates total. Two fresh models, 250 lifetime updates each, five sampled-crop epochs per arm. No warmstart, extra sanity optimization, test inference or default promotion.','',
        '| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    lines+=['| '+' | '.join(row)+' |' for row in score_rows]
    lines+=['','Both arms have931,361 parameters and start with identical tensors, empty AdamW state, AMP scaler and CPU/CUDA RNG state. The control repeats its normalized128ms plane; the treatment combines independently normalized128/256ms planes. Same1000 crop/gain descriptors, legacy loss, cosine schedule, delayed labels and fixed Stage14 decoder.','',
        'Both planes use216 logarithmic frequency bins,16kHz mono PCM and a10ms right-edge grid. They are past-only STFT features, not constant-Q transforms. Long-window Fourier spacing is finer, but the window also includes older information and blurs temporal detail; this comparison does not isolate spectral resolution from temporal context.','',
        'The neural architecture/capacity is matched. The control computes its duplicated FFT once; the treatment computes two distinct FFTs. End-to-end timing includes both frontends. There is zero algorithmic future lookahead and zero EOF feature padding; even the long plane emits its first zero-left-padded frame after10ms.','',
        'Normalization was recomputed using all frames of the48 training clips only and exactly reproduced the archived stage11 statistics. All24 two-plane validation cases reproduce full-input,20ms-packet and frozen-cache features byte for byte.','',
        'These are reused development clips with a decoder previously selected on this validation set. One seed and250 updates per model cannot establish generalization, significance, electric-guitar robustness or a general architecture ranking. These fresh250-update models are not a matched comparison with the750-update Stage16 continuations.','',
        'Source correction remains130ms globally, with zero additional onset backdate. Completion delay is conditional on correctly matched notes and excludes missed notes. Timings measure warm file-PCM replays, not microphone/sound-card latency.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save_json(output/'presentation-provenance.json',{'report_source_sha256':sha256(Path(__file__)),
        'borrowed_source_sha256':{name:sha256(ROOT/name) for name in ['supervision16/report.py','decoder14/report.py','baseline/report.py','streaming/report.py']},
        'review_html_sha256':sha256(output/'review.html'),'report_md_sha256':sha256(output/'report.md'),
        'numeric_artifact_sha256':{name:sha256(output/name) for name in ['protocol.json','input-audit.json','training-summary.json',
            'selection.json','validation/run.json','validation/summary.json']},
        'scope':'Browser QA must bind the exact final HTML. Rendering alone does not confirm QA.'})


TEMPLATE=r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 17 · Short and long spectra</title><style>__CSS__</style></head><body><main>
<div class="kicker">Stage 17 · 500 new updates total</div><h1>Do short and long spectra work better together?</h1><p>__OUTCOME__</p>
<p class="notice">Two fresh models with identical starting weights and capacity. The same twelve development clips select and report these results. The existing default stays unchanged.</p>
<section><h2>Matched final-model comparison</h2>__SCORES__<p>Exact string/fret frame F1 is primary; original-source note-onset F1 breaks ties. Compare the two 250-update finals. Short + short wins exact ties. These fresh models have a different training history from the Stage 16 750-update continuations.</p></section>
<section><h2>What changed at the input?</h2><p>The control uses two copies of the same 128 ms features. The treatment uses 128 ms and 256 ms features at the same 216 log-frequency bins and 10 ms right-edge timestamps. Each plane uses its own training-only normalization. These are log-frequency STFT features.</p><p>The long plane offers finer Fourier spacing and older audio context together. Its longer window can smear transients. Both emit their first zero-left-padded frame after 10 ms and use no future PCM; window duration is not a mandatory lookahead delay.</p>__PITCH__<p>State-derived pitch uses activity odds 0.25; the separate pitch head uses probability 0.5. The low-register diagnostic covers MIDI 40–51 and does not select the winner.</p></section>
<section><h2>Budget and measured cost</h2>__COST__<p>500 new updates total: 250 per arm over five sampled-crop epochs, with zero historical warmstart or new sanity updates. Each neural network has 931,361 parameters. Short + short computes its duplicated FFT once; short + long computes two distinct FFTs. End-to-end processing includes all feature planes.</p><p>Peak tensors exclude other GPU allocations. Training seconds exclude setup and evaluation. Current timings are warm unpaced PCM replays, including the decoder and probability recording. Processing/audio covers packet push calls; EOF flush processing enters the completion clock but is excluded from that ratio. Hardware capture and resampling are excluded.</p></section>
<section><details><summary>Training losses and matching checks</summary>__HISTORY__<p>Both arms start with identical model tensors, empty AdamW state, AMP scale 1024 and CPU/CUDA RNG state. They use the same 1,000 crop/gain descriptors from sampled epochs 1–5, the original loss and a 250-step cosine schedule from 0.001 to 0.0001. No validation-driven early stopping.</p><p>Target delay remains 13 frames, with 26-frame leading/crop masks. The decoder is fixed at onset 0.70, activity odds 0.25, global source correction 130 ms and additional onset backdate zero. No Stage16 auxiliary loss is added.</p><p><a href="models/short_short/training-receipt.json">Short + short receipt</a><a href="models/short_long/training-receipt.json">Short + long receipt</a><a href="frontend-parity.json">Feature parity</a></p></details></section>
<section><h2>Listen and compare</h2><label>Recording<br><select id="clip"></select></label><label>Model<br><select id="mode"></select></label><button id="compare">Compare short / fused features</button>
<audio id="audio" controls preload="metadata"></audio><p id="currentinfo"></p><p id="clock">0.000 s</p>
<p class="legend"><span style="color:#739cff">Reference / matched</span><span style="color:#c8a1ff">Reference / missed</span><span style="color:#58d9a2">Prediction / matched</span><span style="color:#ff7b7b">Prediction / extra</span></p><div id="roll"></div>
<div id="exports"></div><button id="download-json">Download exact note JSON</button><h3>Source → packet → virtual completion</h3><p>White marks estimated source time, blue marks packet availability, and green marks completion after measured processing. End-of-input closures are tagged as flushes. Source correction never advances packet or completion clocks.</p><div id="events"></div>
<label><input id="showall" type="checkbox"> Show all events, including those after playback</label><p id="eventcount"></p><div id="eventwrap"><table><thead><tr><th>Event</th><th>Pitch</th><th>Source (s)</th><th>Packet (s)</th><th>Completion (s)</th></tr></thead><tbody id="eventrows"></tbody></table></div></section>
<section><h2>When did matched onsets finish processing?</h2>__LATENCY__<p>Conditional virtual completion minus performed reference onset. Misses have no finite delay, and matched subsets differ. These file replays do not measure sound-card or microphone capture latency.</p></section>
<section><details><summary>Selection and limitations</summary><pre>__SELECTION__</pre><p>One seed, short training and reused validation limit this result. The decoder was previously tuned on these clips. No independent test, new electric-guitar robustness or general architecture ranking is established. The long window changes spectral detail and temporal age together. Only this matched pair supports a direct comparison.</p></details></section>
<p><a href="report.md">Report</a><a href="findings.md">Findings</a><a href="audit.json">Reproduction audit</a><a href="viewer-validation.json">Viewer checks</a><a href="presentation-provenance.json">Presentation provenance</a></p>
</main><script>__SCRIPT__</script></body></html>'''
