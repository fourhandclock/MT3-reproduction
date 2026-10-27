"""Matched-loss results with original/decoder-policy scores and measured clocks."""

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

LABELS={'parent250':'Parent / 250 updates','control500':'Original loss / 500 updates',
        'conditional_fret500':'Extra fret loss / 500 updates'}


def pct(value):return '—' if value is None else f'{100*value:.2f}%'


def write_report(output):
    output=Path(output);run=read(output/'validation/run.json');summary=read(output/'validation/summary.json')
    rows=[read(p) for p in sorted((output/'validation/results').glob('*/*-metrics.json'))]
    modes=summary['modes'];training=run['training_summary'];selected=run['selection']['selected_arm']
    control,treatment=modes['control500'],modes['conditional_fret500']
    state_delta=treatment['exact_state_f1']-control['exact_state_f1'];onset_delta=treatment['onset_f1']-control['onset_f1']
    outcome=(f"Extra fret supervision changes argmax string/fret frame F1 by {state_delta:+.5f} and note-onset F1 by {onset_delta:+.5f} "
             f"versus the matched original-loss control. The declared development rule selects {LABELS[selected].lower()}.")
    headers=['Model','Argmax state F1','Note-onset F1','Onset + offset F1','String F1','Frame F1']
    scores=[[label]+[f"{modes[m][k]:.4f}" for k in ['exact_state_f1','onset_f1','onset_offset_f1','tab_onset_f1','frame_f1']]
            for m,label in LABELS.items()]
    evidence=[];activity=[];cost=[];latency=[];history=[]
    for mode,label in LABELS.items():
        value=modes[mode];decoder=value['frame_policies']['decoder_activity']
        evidence.append([label,pct(value['best_active_fret_accuracy']),pct(value['conditional_fret_accuracy']),
            f"{decoder['exact_state_mean_clip_f1']:.4f}",f"{decoder['exact_state_pooled']['f1']:.4f}"])
        activity.append([label,pct(decoder['activity_pooled']['precision']),pct(decoder['activity_pooled']['recall']),
            f"{value['state_derived_pitch_f1']:.4f}",f"{value['auxiliary_pitch_f1']:.4f}"])
        receipt=None if mode=='parent250' else training['arms'][mode.removesuffix('500')]
        cost.append([label,0 if receipt is None else receipt['new_optimizer_steps'],250 if receipt is None else 500,
            '—' if receipt is None else f"{receipt['training_seconds']:.1f}",
            '—' if receipt is None else f"{receipt['peak_gpu_mib']:.1f}",f"{value['real_time_factor']:.3f}"])
        latency.append([label,value['matched_onset_count'],f"{value['matched_onset_delivery_p50_ms']:.1f}",
            f"{value['matched_onset_delivery_p95_ms']:.1f}",sum(r['pitch_metrics']['errors']['extra_notes'] for r in rows if r['mode']==mode)])
    for name,receipt in training['arms'].items():
        for block in receipt['history']:
            loss=block['training_losses']
            history.append([name,block['data_epoch'],block['optimizer_steps'],f"{loss['state']:.4f}",
                '—' if 'conditional_fret' not in loss else f"{loss['conditional_fret']:.4f}",f"{loss['total']:.4f}"])
    tables={'scores':_table(headers,scores),
        'evidence':_table(['Model','Best fret on reference-active frames','Correct fret when both active','Activity-rule state F1 / mean clip','Activity-rule state F1 / pooled'],evidence),
        'activity':_table(['Model','Activity precision / pooled','Activity recall / pooled','State-union pitch F1','Separate pitch F1'],activity),
        'cost':_table(['Model','New updates','Lifetime updates','New training seconds','Peak tensor MiB','Processing / audio'],cost),
        'latency':_table(['Model','Matched onsets','Completion p50 ms','Completion p95 ms','Official extra notes'],latency),
        'history':_table(['Arm','Data epoch','Lifetime updates','State loss','Extra fret term','Total loss'],history)}
    payload=[]
    for row in rows:
        mode,clip=row['mode'],row['id'];metrics=row['pitch_metrics'];stem=Path('validation/results')/mode/clip
        svg=roll_svg(metrics['reference'],metrics['predicted'],metrics['onset_matches'],10.,clip)
        svg=svg.replace('</svg>','<line id="source-playhead" x1="45" x2="45" y1="20" y2="215" stroke="white" stroke-width="2"/></svg>')
        payload.append({'id':clip,'mode':mode,'label':LABELS[mode],'notes':row['notes'],'events':row['events'],
            'svg':svg,'audio':quote(row['relativeaudio'],safe='/'),'f1':metrics['onset']['f1'],
            'json':(output/stem.with_suffix('.json')).read_bytes().decode('utf-8'),
            'links':{name:quote(stem.as_posix()+ext,safe='/') for name,ext in
                [('JSON','.json'),('CSV','.csv'),('MIDI','.mid'),('Metrics','-metrics.json'),('Events','-events.json')]}})
    script=PLAYER_SOURCE.split('<script>',1)[1].split('</script>',1)[0]
    for old,new in [('state_pitch750','conditional_fret500'),('control750','control500'),('parent500','parent250')]:script=script.replace(old,new)
    page=TEMPLATE.replace('__SCRIPT__',script)
    values={**tables,'outcome':html.escape(outcome),'payload':json.dumps(payload).replace('<','\\u003c'),
        'labels':json.dumps(LABELS),'selection':html.escape(json.dumps(run['selection'],indent=2)),
        'css':STYLE_SOURCE.split('<style>',1)[1].split('</style>',1)[0]}
    for key,value in values.items():page=page.replace('__'+key.upper()+'__',value)
    (output/'review.html').write_text(page,encoding='utf-8')
    lines=['# Stage19: active-only conditional-fret supervision','',outcome,'',
        '500 new updates total: 250 per arm over five new sampled-crop epochs. Both final models have500 lifetime updates; their one shared parent has250 historical updates. No new sanity optimization, decoder search, test inference or default promotion.','',
        '| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    lines+=['| '+' | '.join(map(str,r))+' |' for r in scores]
    lines+=['','Both arms restore the same model, ordered AdamW moments/options/counters, AMP scaler and CPU/CUDA RNG. They use identical sampled epochs6–10, constant stored-end learning rate0.0001, masks and delayed labels. The archived cosine scheduler is preserved as provenance and is not restarted or stepped.','',
        'Treatment adds0.5 times cross entropy over the existing25 active fret logits, averaged over valid active string-frame targets. The term runs in float32 outside autocast; inactive/conflict targets and masked frames contribute nothing. Batches without active targets return a differentiable zero. No model parameters are added. There is no direct auxiliary gradient on the inactive logit, but active logits/shared weights can still affect occupancy.','',
        'Argmax mean string/fret F1 remains the primary metric, then note-onset F1; the parent is a dose reference and is ineligible for final selection. Best-fret accuracy evaluates all reference-active frames. Conditional fret accuracy evaluates only frames where both reference and prediction are active. Activity-rule mean and pooled F1 use different aggregation. Do not treat these denominators as equivalent.','',
        'All models use the same931,361-parameter two-plane causal network and duplicated128ms features, training-only normalization and fixed Stage14 decoder. First frames arrive at10ms with zero future lookahead; no EOF feature padding. Source correction remains130ms with no additional onset backdate. Onset spikes only retrigger same-fret activity; new notes/fret changes use four stable state frames.','',
        'Timing is freshly measured for all three models and includes packet feature/model/decoder work and probability recording. EOF flush processing enters completion clocks but is excluded from processing/audio. Identical inference architecture means timing variation cannot show that the loss speeds inference. Conditional matched-note delays omit misses and compare different matched subsets. No microphone or sound-card latency is measured.','',
        'These are single-parent, single-coefficient results on reused development clips, with a decoder previously tuned on them. Independent electric and distorted-guitar recordings are still needed for generalization or product-quality claims.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save_json(output/'presentation-provenance.json',{'report_source_sha256':sha256(Path(__file__)),
        'borrowed_source_sha256':{p:sha256(ROOT/p) for p in ['supervision16/report.py','decoder14/report.py','baseline/report.py','streaming/report.py']},
        'review_html_sha256':sha256(output/'review.html'),'report_md_sha256':sha256(output/'report.md'),
        'numeric_artifact_sha256':{p:sha256(output/p) for p in ['protocol.json','input-audit.json','training-summary.json','selection.json','validation/run.json','validation/summary.json']},
        'scope':'Browser QA must bind the final HTML. Rendering alone does not confirm QA.'})


TEMPLATE=r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 19 · Fret supervision</title><style>__CSS__</style></head><body><main>
<div class="kicker">Stage 19 · 500 new updates total</div><h1>Does extra fret supervision help?</h1><p>__OUTCOME__</p>
<p class="notice">One restored parent, two matched continuations, one additional loss term. The same twelve development clips select and report the results. The current default stays unchanged.</p>
<section><h2>Matched final-model comparison</h2>__SCORES__<p>Argmax string/fret frame F1 is primary; note-onset F1 breaks ties. Only the two fixed 500-update finals are eligible. Original loss wins exact ties. The 250-update parent is a training-dose reference.</p></section>
<section><h2>Does fret choice improve?</h2>__EVIDENCE__<p>Best-fret accuracy uses every reference-active string frame and the most likely active fret, independent of the activity gate. Conditional fret accuracy uses only frames where both reference and prediction are active. Mean clip F1 weights clips equally; pooled F1 counts all scored string frames. Different denominators answer different questions.</p>__ACTIVITY__<p>The decoder activity rule stays at best-active / (best-active + inactive + 1e-8) ≥ 0.25. Separate pitch uses 0.5. Frame comparisons use source targets 0:1000 versus recorded predictions 13:1013, with the first 26 source frames and per-string conflicts excluded from state metrics.</p></section>
<section><h2>Budget and current cost</h2>__COST__<p>500 new updates total: 250 each over five new sampled-crop epochs. Each child reaches 500 lifetime updates from the same 250-update parent. Count that shared parent once; no new sanity optimization. All three networks have 931,361 parameters and the same duplicated 128 ms frontend.</p><p>Peak tensors exclude other GPU allocations. Training seconds exclude restoration and validation. Processing/audio covers warm packet pushes, including probability recording; EOF flush processing enters the completion clock but is excluded from that ratio. The inference graph is identical, so runtime variation does not establish a causal speed gain.</p></section>
<section><details><summary>Loss, restoration and matching controls</summary>__HISTORY__<p>The treatment adds 0.5 times active-only conditional-fret cross entropy over the existing 25 active logits. The original loss stays intact. The extra term uses existing crop/leading masks, ignores inactive and per-string conflict labels, averages over active string frames and returns differentiable zero when there are none. It runs in float32 outside autocast and adds no parameters.</p><p>The term has no direct gradient on the inactive logit. Changes to active logits and shared layers can still affect activity probabilities. Both arms restore the same full model/AdamW/scaler/RNG state and use identical sampled epochs 6–10 at constant 0.0001. The ended parent scheduler is archived, with no restart or new scheduler steps.</p><p><a href="models/control/training-receipt.json">Control receipt</a><a href="models/conditional_fret/training-receipt.json">Fret-loss receipt</a><a href="training-summary.json">Training summary</a></p></details></section>
<section><h2>Listen and compare</h2><label>Recording<br><select id="clip"></select></label><label>Model<br><select id="mode"></select></label><button id="compare">Compare original / extra fret loss</button>
<audio id="audio" controls preload="metadata"></audio><p id="currentinfo"></p><p id="clock">0.000 s</p>
<p class="legend"><span style="color:#739cff">Reference / matched</span><span style="color:#c8a1ff">Reference / missed</span><span style="color:#58d9a2">Prediction / matched</span><span style="color:#ff7b7b">Prediction / extra</span></p><div id="roll"></div>
<div id="exports"></div><button id="download-json">Download exact note JSON</button><h3>Source → packet → virtual completion</h3><p>White marks estimated source time, blue marks packet availability and green marks completion after measured processing. End-of-input closures are tagged as flushes. Source correction never advances packet or completion clocks.</p><div id="events"></div>
<label><input id="showall" type="checkbox"> Show all events, including those after playback</label><p id="eventcount"></p><div id="eventwrap"><table><thead><tr><th>Event</th><th>Pitch</th><th>Source (s)</th><th>Packet (s)</th><th>Completion (s)</th></tr></thead><tbody id="eventrows"></tbody></table></div></section>
<section><h2>When did matched onsets finish processing?</h2>__LATENCY__<p>Conditional virtual completion minus performed reference onset. Misses have no finite delay; matched subsets differ. These file replays do not measure microphone or sound-card capture latency.</p></section>
<section><details><summary>Selection and limitations</summary><pre>__SELECTION__</pre><p>The parent is reproduced from saved evidence. Single-parent, single-coefficient results on reused validation clips cannot establish independent generalization or a general architecture ranking. The decoder was previously tuned on these clips. Onset peaks only retrigger same-fret activity; new notes/fret changes open after four stable state frames. No new electric-guitar robustness or product promotion is claimed.</p></details></section>
<p><a href="report.md">Report</a><a href="findings.md">Findings</a><a href="audit.json">Reproduction audit</a><a href="viewer-validation.json">Viewer checks</a><a href="presentation-provenance.json">Presentation provenance</a></p>
</main><script>__SCRIPT__</script></body></html>'''
