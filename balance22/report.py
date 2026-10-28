"""Four-model audio review with the amended training-only rule and measured clocks."""

import html
import json
from pathlib import Path
from urllib.parse import quote
from baseline.download import ROOT, sha256
from baseline.report import roll_svg
from decoder14.report import TEMPLATE as STYLE_SOURCE
from decoder14.workflow import read
from streaming.evaluate import save_json
from streaming.report import _table
from supervision16.report import TEMPLATE as PLAYER_SOURCE
from .evaluation import MODES

LABELS = {'parent500': 'Parent / 500 updates', 'original650': 'Original loss / 650 updates',
          'uniform_ce650': 'Uniform fret CE / 650 updates', 'balanced_ce650': 'Weighted fret CE / 650 updates'}


def pct(value):
    return '—' if value is None else f'{100 * value:.2f}%'


def findings(output):
    output = Path(output)
    run = read(output / 'validation/run.json')
    modes = read(output / 'validation/summary.json')['modes']
    selected = run['selection']['selected_arm']
    original, uniform, balanced = (modes[m] for m in MODES[1:])
    text = (f"Weighted versus uniform fret CE changes argmax state F1 by {balanced['exact_state_f1'] - uniform['exact_state_f1']:+.5f} "
            f"and note-onset F1 by {balanced['onset_f1'] - uniform['onset_f1']:+.5f}. "
            f"The fixed final-development rule selects {LABELS[selected].lower()}.")
    headers = ['Model', 'Argmax state F1', 'Note-onset F1', 'Onset + offset F1', 'String F1', 'Decoded frame F1']
    scores = [[LABELS[m]] + [f"{modes[m][k]:.4f}" for k in
              ['exact_state_f1', 'onset_f1', 'onset_offset_f1', 'tab_onset_f1', 'frame_f1']] for m in MODES]
    lines = ['# Stage22: matched active-fret class weighting', '', text, '',
        '450 new optimizer updates total:150 per arm over three shared sampled-crop epochs11-13. Each final has650 lifetime updates from the same500-update parent. Count the one shared historical parent once. No new sanity optimization, decoder search, independent-test inference or default promotion.', '',
        '## Pre-training amendment', '',
        'The frozen proposal assigned weight0 to historically unexposed classes and rejected any active such target. Training-only preflight found string4/fret17:50 available active source frames, no supervised exposure in epochs1-10, and76 presentations in the fixed epochs11-13 crops. No optimizer update had run. The amended rule gives training-available but historically unexposed positions the existing cap4; labels absent from training remain weight0 and fail if active. All crops, arms, coefficients, learning rate and450-update cap remain as proposed. The original proposal file remains byte-identical; class-weights.json records the deviation and its training-only basis.', '',
        '## Fixed final comparison', '', '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    lines += ['| ' + ' | '.join(row) + ' |' for row in scores]
    lines += ['', '| Model | Best active fret / pooled GT frames | Equal-supported-position fret accuracy | Fret accuracy when both active |',
              '| --- | ---: | ---: | ---: |']
    for mode in MODES:
        value = modes[mode]
        lines.append(f"| {LABELS[mode]} | {pct(value['best_active_fret_accuracy'])} | {pct(value['best_active_equal_position_accuracy'])} | {pct(value['conditional_fret_accuracy'])} |")
    lines += ['', 'Primary selection uses mean clip argmax string/fret frame F1, then official note-onset F1. Only the three fixed step650 finals are eligible; exact ties prefer original, uniform, then weighted. The500-update parent is a dose reference and cannot be selected. Conditional and macro fret scores have different denominators and do not override primary selection.', '',
        'All arms restore identical saved model weights, ordered AdamW options/moments/counters, AMP scaler and CPU/CUDA RNG. They share600 crop/gain descriptors, thirteen-frame full-recording target delay, delayed-grid leading26 and crop-edge26 masks, batch4 and constant learning rate0.0001. The ended parent scheduler is archived and never restarted or stepped.', '',
        'Original loss is unchanged. Uniform CE adds0.5 times float32 CE over the25 active logits on valid active targets. Weighted CE uses the same coefficient and active support; fixed weights are min(4,sqrt(max unique count within string / class unique count)), with the recorded available/unexposed fallback. It divides sum(weight*CE) by sum(weight) within each batch. Empty active support returns differentiable zero. No auxiliary inactive-logit gradient or new model parameters are introduced; active logits and shared weights can still affect occupancy.', '',
        'All four inference networks have931,361 parameters, the same duplicated past-only128ms feature planes and training-only normalization. Decoder onset0.7, activity0.25, no additional onset backdate and130ms source correction remain fixed. New notes/fret changes require four stable state frames; onset peaks only retrigger same-fret activity.', '',
        'All timings are fresh file-PCM replays and include probability recording. Processing/audio excludes EOF flush processing; the virtual completion clock includes that cost. Identical inference graphs mean runtime variation cannot establish a speed benefit from class weighting. Matched-onset completion delays omit missed notes and use different matched subsets. No microphone or sound-card latency was measured.', '',
        'These single-parent results reuse twelve development recordings and a decoder previously tuned on them. They cannot establish independent electric-guitar generalization, causation beyond this controlled comparison or product readiness. The offline default is preserved.', '']
    return '\n'.join(lines)


def write_report(output):
    output = Path(output)
    run = read(output / 'validation/run.json')
    summary = read(output / 'validation/summary.json')
    modes = summary['modes']
    rows = [read(p) for p in sorted((output / 'validation/results').glob('*/*-metrics.json'))]
    scores, evidence, activity, cost, latency, history = [], [], [], [], [], []
    for mode in MODES:
        value = modes[mode]
        scores.append([LABELS[mode]] + [f"{value[k]:.4f}" for k in ['exact_state_f1', 'onset_f1', 'onset_offset_f1', 'tab_onset_f1', 'frame_f1']])
        decoder = value['frame_policies']['decoder_activity']
        evidence.append([LABELS[mode], pct(value['best_active_fret_accuracy']), pct(value['best_active_equal_position_accuracy']),
                         pct(value['conditional_fret_accuracy']), f"{decoder['exact_state_mean_clip_f1']:.4f}", f"{decoder['exact_state_pooled']['f1']:.4f}"])
        activity.append([LABELS[mode], pct(decoder['activity_pooled']['precision']), pct(decoder['activity_pooled']['recall']),
                         f"{value['state_derived_pitch_f1']:.4f}", f"{value['auxiliary_pitch_f1']:.4f}"])
        receipt = None if mode == 'parent500' else run['training_summary']['arms'][mode.removesuffix('650')]
        cost.append([LABELS[mode], 0 if receipt is None else 150, 500 if receipt is None else 650,
                     '—' if receipt is None else f"{receipt['training_seconds']:.1f}",
                     '—' if receipt is None else f"{receipt['peak_gpu_mib']:.1f}", f"{value['real_time_factor']:.3f}"])
        latency.append([LABELS[mode], value['matched_onset_count'], f"{value['matched_onset_delivery_p50_ms']:.1f}",
                        f"{value['matched_onset_delivery_p95_ms']:.1f}", sum(r['pitch_metrics']['errors']['extra_notes'] for r in rows if r['mode'] == mode)])
    for name, receipt in run['training_summary']['arms'].items():
        for block in receipt['history']:
            loss = block['training_losses']
            history.append([name, block['data_epoch'], block['optimizer_steps'], f"{loss['state']:.4f}",
                            '—' if 'active_fret_ce' not in loss else f"{loss['active_fret_ce']:.4f}",
                            f"{loss['total']:.4f}", block['valid_active_string_frames']])
    tables = {'scores': _table(['Model', 'Argmax state F1', 'Note-onset F1', 'Onset + offset F1', 'String F1', 'Decoded frame F1'], scores),
        'evidence': _table(['Model', 'Best fret / active GT', 'Equal-position fret', 'Fret / both active', 'Activity-rule mean F1', 'Activity-rule pooled F1'], evidence),
        'activity': _table(['Model', 'Activity precision', 'Activity recall', 'State-union pitch F1', 'Separate pitch F1'], activity),
        'cost': _table(['Model', 'New updates', 'Lifetime updates', 'Training seconds', 'Peak tensor MiB', 'Processing / audio'], cost),
        'latency': _table(['Model', 'Matched onsets', 'Completion p50 ms', 'Completion p95 ms', 'Official extra notes'], latency),
        'history': _table(['Arm', 'Data epoch', 'Lifetime updates', 'State loss', 'Active fret CE', 'Total loss', 'Active presentations'], history)}
    payload = []
    for row in rows:
        mode, clip = row['mode'], row['id']
        metrics = row['pitch_metrics']
        stem = Path('validation/results') / mode / clip
        svg = roll_svg(metrics['reference'], metrics['predicted'], metrics['onset_matches'], 10., clip)
        svg = svg.replace('</svg>', '<line id="source-playhead" x1="45" x2="45" y1="20" y2="215" stroke="white" stroke-width="2"/></svg>')
        payload.append({'id': clip, 'mode': mode, 'label': LABELS[mode], 'notes': row['notes'], 'events': row['events'],
            'svg': svg, 'audio': quote(row['relativeaudio'], safe='/'), 'f1': metrics['onset']['f1'],
            'json': (output / stem.with_suffix('.json')).read_bytes().decode('utf-8'),
            'links': {name: quote(stem.as_posix() + extension, safe='/') for name, extension in
                      [('JSON', '.json'), ('CSV', '.csv'), ('MIDI', '.mid'), ('Metrics', '-metrics.json'), ('Events', '-events.json')]}})
    script = PLAYER_SOURCE.split('<script>', 1)[1].split('</script>', 1)[0]
    old = "$('compare').onclick=()=>{$('mode').value=current.mode==='state_pitch750'?'control750':'state_pitch750';render();};"
    if old not in script:
        raise ValueError('Frozen player cycle control changed')
    script = script.replace(old, "$('compare').onclick=()=>{const keys=Object.keys(LABELS);$('mode').value=keys[(keys.indexOf(current.mode)+1)%keys.length];render();};")
    text = findings(output)
    outcome = text.split('\n\n')[1]
    page = TEMPLATE.replace('__SCRIPT__', script)
    values = {**tables, 'outcome': html.escape(outcome), 'payload': json.dumps(payload).replace('<', '\\u003c'),
        'labels': json.dumps(LABELS), 'selection': html.escape(json.dumps(run['selection'], indent=2)),
        'css': STYLE_SOURCE.split('<style>', 1)[1].split('</style>', 1)[0]}
    for key, value in values.items():
        page = page.replace('__' + key.upper() + '__', value)
    (output / 'review.html').write_text(page, encoding='utf-8')
    for name in ('report.md', 'findings.md'):
        (output / name).write_text(text, encoding='utf-8')
    save_json(output / 'presentation-provenance.json', {'report_source_sha256': sha256(Path(__file__)),
        'borrowed_source_sha256': {name: sha256(ROOT / name) for name in ['supervision16/report.py', 'decoder14/report.py', 'baseline/report.py', 'streaming/report.py']},
        'review_html_sha256': sha256(output / 'review.html'), 'report_md_sha256': sha256(output / 'report.md'),
        'numeric_artifact_sha256': {name: sha256(output / name) for name in ['protocol.json', 'input-audit.json', 'class-weights.json',
            'training-summary.json', 'selection.json', 'validation/run.json', 'validation/summary.json']},
        'scope': 'Final HTML requires browser QA and visual inspection; rendering alone does not establish those gates.'})
    selected = run['selection']['selected_arm']
    save_json(output / 'next-experiment.json', {
        'stage': 23, 'status': 'proposed_not_evaluated', 'kind': 'locked_progression3_evaluation',
        'question': 'Does the locked Stage22 selection retain its development gains on the existing reserved progression3 recordings?',
        'source_root': 'outputs/balance22', 'new_optimizer_updates': 0,
        'locked_selected_arm': selected, 'locked_selected_checkpoint': run['checkpoint_paths'][selected],
        'locked_selected_checkpoint_sha256': run['checkpoint_sha256'][selected],
        'comparison_models': {mode: {'checkpoint': run['checkpoint_paths'][mode], 'sha256': run['checkpoint_sha256'][mode]} for mode in MODES},
        'test_manifest': 'data/cnn_gru/manifest.json', 'test_split': 'test', 'expected_recordings': 12,
        'data_scope': 'Existing progression3 recordings were used in earlier historical evaluations. They are held out from the current training/development protocol, but are not an untouched new independent dataset; players occur across splits.',
        'model_selection_allowed': False, 'calibration_search_allowed': False, 'training_allowed': False,
        'test_inference_allowed': True, 'production_promotion': False,
        'alignment': {'source_frames': 1000, 'source_delay_frames': 13, 'leading_mask_frames': 26},
        'decoder_parameters': run['config']['decoder_parameters'], 'global_source_correction_seconds': .13,
        'scope': 'Freeze all model identities and the selected arm before any new progression3 scores. Report every locked model on the same excerpts; no test-driven reselection, thresholds or normalization changes. No electric-guitar or fresh-independent-generalization claim.'})
    print('Stage22 review saved:', output / 'review.html', flush=True)


TEMPLATE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Stage22 - Fret class weighting</title><style>__CSS__</style></head><body><main>
<div class="kicker">Stage 22 · 450 new updates total</div><h1>Does class weighting improve fret choices?</h1><p>__OUTCOME__</p>
<p class="notice">Three matched continuations from the same control500 state. These twelve development recordings select and report the results; no independent generalization or default promotion is claimed.</p>
<section><h2>Recorded pre-training amendment</h2><p>Training-only preflight found string4/fret17:50 available active frames, no prior supervised exposure, and76 presentations in the fixed new crops. The original zero-weight/fail rule would stop the trial. Before any optimizer update, available-but-unexposed positions were assigned the already fixed cap4. No crops, arms, coefficients or budget changed.</p><p><a href="class-weights.json">Exact counts, weights and amendment</a> · <a href="preflight-initial.log">Initial preflight stop</a></p></section>
<section><h2>Fixed step650 comparison</h2>__SCORES__<p>Mean clip argmax string/fret state F1 is primary; official note-onset F1 breaks ties. Only the three650-update finals are eligible. Exact ties prefer original, then uniform CE, then weighted CE. The500-update parent is a dose reference.</p></section>
<section><h2>Fret choice and occupancy</h2>__EVIDENCE__<p>Best-active fret accuracy uses all valid GT-active string-frames. Equal-position accuracy averages recall across supported positions after pooling clips. Conditional fret accuracy uses only frames where both GT and prediction are active. Activity-rule mean and pooled F1 have different aggregation.</p>__ACTIVITY__<p>Original-source targets0:1000 align with predictions13:1013, with first26 source frames and per-string conflicts masked. Decoder activity remains best-active/(best-active+inactive+1e-8)≥0.25.</p></section>
<section><h2>Matched budget and current cost</h2>__COST__<p>150 updates per arm over the same epochs11-13, reaching650 lifetime updates. One shared500-update historical parent is counted once. No new sanity training or scheduler restart. All networks have931,361 parameters and duplicated past-only128ms features.</p><p>Processing/audio includes packet feature/model/decoder work and probability recording, excluding EOF flush processing. Flush cost enters virtual completion. Identical inference graphs prevent a causal speed claim from runtime differences.</p></section>
<section><details><summary>Loss and restoration details</summary>__HISTORY__<p>Uniform CE adds0.5 times float32 cross entropy over the25 active logits on valid active targets. Weighted CE uses fixed within-string min(4,sqrt(max unique count/class unique count)) weights, the recorded available/unexposed fallback, and sum(weight*CE)/sum(weight). Empty active support returns differentiable zero.</p><p>No new parameters or direct auxiliary inactive-logit gradient are added. Shared weights and active logits can still affect occupancy. Full model, ordered AdamW moments/options/counters, AMP scaler and CPU/CUDA RNG are restored identically. The archived ended scheduler is not restarted or stepped; learning rate remains0.0001.</p><p><a href="training-summary.json">All receipts</a><a href="crop-schedule.json">Shared crop schedule</a><a href="protocol.json">Actual protocol</a></p></details></section>
<section><h2>Listen and compare</h2><label>Recording<br><select id="clip"></select></label><label>Model<br><select id="mode"></select></label><button id="compare">Next model, keep playback</button><audio id="audio" controls preload="metadata"></audio><p id="currentinfo"></p><p id="clock">0.000 s</p><p class="legend"><span style="color:#739cff">Reference / matched</span><span style="color:#c8a1ff">Reference / missed</span><span style="color:#58d9a2">Prediction / matched</span><span style="color:#ff7b7b">Prediction / extra</span></p><div id="roll"></div><div id="exports"></div><button id="download-json">Download exact note JSON</button><h3>Source → packet → virtual completion</h3><p>White marks source time, blue marks packet availability, green marks completion after measured processing. EOF closures are tagged as flushes. Source correction never advances packet or completion time.</p><div id="events"></div><label><input id="showall" type="checkbox"> Show all events, including after playback</label><p id="eventcount"></p><div id="eventwrap"><table><thead><tr><th>Event</th><th>Pitch</th><th>Source (s)</th><th>Packet (s)</th><th>Completion (s)</th></tr></thead><tbody id="eventrows"></tbody></table></div></section>
<section><h2>When did matched attacks finish processing?</h2>__LATENCY__<p>Completion minus performed reference onset, conditional on correct matches. Misses have no finite matched delay; matched subsets differ. These file replays do not measure microphone or sound-card latency.</p></section>
<section><details><summary>Selection and limits</summary><pre>__SELECTION__</pre><p>Single-parent, single-weight-rule results on reused development clips cannot establish independent electric-guitar generalization. New notes/fret changes require four stable state frames; onset peaks only retrigger same-fret notes. No new calibration or product promotion occurred.</p></details></section><p><a href="report.md">Report</a><a href="findings.md">Findings</a><a href="audit.json">Audit</a><a href="viewer-validation.json">Viewer checks</a><a href="presentation-provenance.json">Presentation provenance</a></p>
</main><script>__SCRIPT__</script></body></html>'''
