"""Portable local review of saved decoder decisions and their packet clocks."""

import csv
import html
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np

from baseline.download import ROOT, sha256
from baseline.report import roll_svg
from streaming.evaluate import save_json
from streaming.report import _table
from .core import MODELS
from .workflow import load_records, read


LABELS = {model + '_' + variant: ('Eight groups' if model == MODELS[0] else 'One group') +
          (' / original decoder' if variant == 'strict' else ' / selected decoder')
          for model in MODELS for variant in ['strict', 'selected']}


def write_report(output):
    output = Path(output)
    records = load_records(output)
    summary = read(output / 'validation/summary.json')
    selection = read(output / 'selection.json')
    protocol = read(output / 'protocol.json')
    rows = []
    for mode in LABELS:
        values = summary['modes'][mode]
        rows.append([LABELS[mode]] + [f'{values[key]:.3f}' for key in
                                     ['onset_f1', 'onset_offset_f1', 'frame_f1', 'tab_onset_f1']] +
                    [values['matched_onsets'], values['predicted_notes']])
    score_headers = ['Model / decoder', 'Onset F1', 'Onset + offset F1', 'Frame F1', 'String F1', 'Matched', 'Predicted']
    parameter_rows, outcomes = [], []
    for model in MODELS:
        selected = selection['models'][model]
        pars = selected['parameters']
        parameter_rows.append([model, selected['candidate'], pars['onset_threshold'], pars['activity_threshold'],
                               int(pars['onset_backdate_seconds'] * 1000)])
        original = summary['modes'][model + '_strict']['onset_f1']
        final = summary['modes'][model + '_selected']['onset_f1']
        outcomes.append(f'{"Eight-group" if model == MODELS[0] else "One-group"} onset F1: {original:.3f} to {final:.3f}.')
    outcome = ' '.join(outcomes)
    oracle_rows = [[key, f"{value['onset_f1']:.3f}", f"{value['onset_offset_f1']:.3f}", value['matched_onsets']]
                   for key, value in summary['oracles'].items()]
    head_rows = []
    latency_rows = []
    for model in MODELS:
        group = [r for r in records if r['model'] == model and r['variant'] == 'strict']
        head_rows.append([model, f"{np.mean([r['head_metrics']['exact_state_f1'] for r in group]):.3f}",
                         f"{np.mean([r['head_metrics']['pitch']['overall']['f1'] for r in group]):.3f}",
                         f"{np.mean([r['head_metrics']['pitch']['low_register']['f1'] for r in group]):.3f}"])
    for mode in LABELS:
        samples = [sample for row in records if row['mode'] == mode for sample in row['packet_latency']['samples_ms']]
        latency_rows.append([LABELS[mode], len(samples), f'{np.quantile(samples, .5):.1f}' if samples else '-',
                             f'{np.quantile(samples, .95):.1f}' if samples else '-'])
    tables = {'scores': _table(score_headers, rows),
              'parameters': _table(['Model', 'Candidate', 'Onset threshold', 'Activity threshold', 'Extra onset backdate (ms)'], parameter_rows),
              'oracle': _table(['Ideal label experiment', 'Onset F1', 'Onset + offset F1', 'Matched'], oracle_rows),
              'heads': _table(['Frozen model', 'Exact state F1', 'Auxiliary pitch F1', 'MIDI 40-51 F1'], head_rows),
              'latency': _table(['Model / decoder', 'Matched onsets', 'Packet p50 (ms)', 'Packet p95 (ms)'], latency_rows)}
    payload = []
    for row in records:
        metrics = row['pitch_metrics']
        stem = Path('validation/results') / row['mode'] / row['id']
        svg = roll_svg(metrics['reference'], metrics['predicted'], metrics['onset_matches'],
                       metrics['score_duration'], row['id'])
        svg = svg.replace('</svg>', '<line id="source-playhead" x1="45" x2="45" y1="20" y2="215" stroke="white" stroke-width="2"/></svg>')
        # Preserve CRLF bytes from Windows exports in the downloadable Blob.
        exact_json = (output / stem.with_suffix('.json')).read_bytes().decode('utf-8')
        payload.append({'id': row['id'], 'mode': row['mode'], 'label': LABELS[row['mode']], 'notes': row['notes'],
                        'events': row['events'], 'svg': svg, 'audio': quote(row['relativeaudio'], safe='/'),
                        'f1': metrics['onset']['f1'], 'parameters': row['parameters'], 'json': exact_json,
                        'links': {key: quote(stem.as_posix() + suffix, safe='/') for key, suffix in
                                  [('JSON', '.json'), ('CSV', '.csv'), ('MIDI', '.mid'),
                                   ('Metrics', '-metrics.json'), ('Events', '-events.json')]}})
    page = TEMPLATE
    for key, value in {**tables, 'outcome': html.escape(outcome), 'payload': json.dumps(payload).replace('<', '\\u003c'),
                       'labels': json.dumps(LABELS), 'selection': html.escape(json.dumps(selection, indent=2))}.items():
        page = page.replace('__' + key.upper() + '__', value)
    (output / 'review.html').write_text(page, encoding='utf-8')
    lines = ['# Stage 14: frozen decoder calibration', '', outcome, '',
             'Zero new training, zero neural inference, zero test inference. Both frozen models retain 500 lifetime updates.', '',
             'Each model selects from 27 prespecified candidates using the same twelve validation recordings that are reported.', '',
             '| ' + ' | '.join(score_headers) + ' |', '| ' + ' | '.join(['---'] * len(score_headers)) + ' |']
    lines.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)
    lines += ['', '## Timing controls', '',
              'The known 13-frame label delay is subtracted once from all published boundaries. Extra onset backdating is a separate candidate parameter; confirmation and release guards stay fixed.', '',
              'Ideal-label experiments translate full arrays with inactive leading frames and truncated tails. Conflicting states become inactive. These annotation-only inputs contain privileged information and are not acoustic predictions.', '',
              '| Oracle | Onset F1 | Onset + offset F1 | Matched |', '| --- | --- | --- | --- |']
    lines.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in oracle_rows)
    lines += ['', 'Only packet availability is replayed. Processing and microphone latency are absent. Conditional latency samples differ across settings.', '',
              'Raw acoustic heads are unchanged. Calibration uses repeatedly reused development data; no fresh generalization result or production promotion is claimed.', '',
              'Next: inspect note misses by acoustic error type and test one bounded representation or data change before spending more training compute.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    with (output / 'validation/metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'mode', 'onset_f1', 'onset_offset_f1', 'frame_f1', 'tab_onset_f1'])
        for row in records:
            writer.writerow([row['id'], row['mode'], row['pitch_metrics']['onset']['f1'],
                             row['pitch_metrics']['onset_offset']['f1'], row['pitch_metrics']['frame']['f1'],
                             row['tab_metrics']['onset']['f1']])
    save_json(output / 'presentation-provenance.json', {'report_source_sha256': sha256(Path(__file__)),
              'review_html_sha256': sha256(output / 'review.html'), 'report_md_sha256': sha256(output / 'report.md'),
              'numeric_artifact_sha256': {name: sha256(output / name) for name in
                  ['protocol.json', 'selection.json', 'grid.json', 'oracle.json', 'validation/summary.json']},
              'browser_qa_record': 'viewer-validation.json',
              'scope': 'Browser verification must match this exact HTML hash; render alone does not confirm QA.'})


TEMPLATE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 14 · Frozen decoder calibration</title><style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#09111e;color:#e6eefb}*{box-sizing:border-box}body{margin:0}main{max-width:1160px;margin:38px auto;padding:0 24px 40px}h1{font-size:36px;margin-bottom:8px}h2{font-size:22px;margin:0 0 16px}p{line-height:1.65;color:#b7c8df}section{background:#101c2d;border:1px solid #28394f;border-radius:14px;padding:24px;margin:22px 0}.kicker{color:#75e6c4;letter-spacing:.08em;font-size:13px;text-transform:uppercase}.notice{border-left:3px solid #f1bb6c;padding-left:15px}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:12px 10px;border-bottom:1px solid #29384b;white-space:nowrap}th{color:#9eb5d2;font-size:12px}td:first-child{font-weight:600}.table-scroll{overflow:auto}select,button{background:#1a3048;color:#e6eefb;border:1px solid #3d5572;padding:10px;border-radius:7px}label{display:inline-block;margin:0 12px 12px 0;font-size:13px}button{cursor:pointer}audio{width:100%;margin:12px 0}svg{width:100%;height:auto;display:block}a{color:#8ec8ff;margin-right:18px}#exports{margin:16px 0}#eventwrap{max-height:260px;overflow:auto}#eventwrap table{font-size:12px}pre{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere;color:#a7bbd7}#clock{font-variant-numeric:tabular-nums}small{color:#a0b7d2}summary{cursor:pointer;color:#a9d4ff}#currentinfo{color:#75e6c4}.legend{font-size:13px}.legend span{margin-right:22px}details p{max-width:95ch}@media(max-width:650px){main{padding:0 12px}section{padding:16px}h1{font-size:28px}}
</style></head><body><main>
<div class="kicker">Stage 14 · zero additional training</div><h1>Frozen decoder calibration</h1>
<p>__OUTCOME__</p><p class="notice">Development results on the same twelve validation recordings used for selection. The acoustic models are frozen. The existing default remains unchanged.</p>
<section><h2>How many notes did the decoder recover?</h2>__SCORES__<p>Macro F1 averages recordings equally. Correct pitches must match within 50 ms; string F1 also requires the reference string. Two seconds of following audio provide context for the scored first ten seconds.</p></section>
<section><h2>Which settings were selected?</h2>__PARAMETERS__<p>Each model searches the same 27 onset / activity / extra-backdate combinations. Onset F1 selects, then onset-plus-offset F1; the original setting wins exact ties. Four-frame attack confirmation and two-frame release confirmation remain fixed.</p><p>The known <strong>130 ms supervision delay</strong> is compensated once for every boundary. The extra onset backdate above is independent of that correction.</p></section>
<section><h2>What happens with ideal labels?</h2>__ORACLE__<p>Privileged annotation-only inputs check decoder behavior. The delayed experiment translates complete label arrays by thirteen frames, including Gaussian boundary support. Leading state is inactive; the end is truncated. These scores do not measure model accuracy.</p><p>The source-time check independently verifies every delayed public event against its raw event: identity and packet availability stay unchanged, source time loses 130 ms once, and declared zero / previous-release / positive-duration clamps apply.</p></section>
<section><h2>The acoustic evidence is unchanged</h2>__HEADS__<p>These are the frozen Stage 13 head metrics, bound to the exact cached probability hashes. Changing decoder settings does not improve framewise pitch or string/fret predictions.</p></section>
<section><h2>Compare a recording</h2><label>Recording<br><select id="clip"></select></label><label>Model and decoder<br><select id="mode"></select></label><button id="compare">Switch original / selected</button>
<audio id="audio" controls preload="metadata"></audio><p id="currentinfo"></p><p id="clock">0.000 s</p>
<p class="legend"><span style="color:#739cff">Reference / matched</span><span style="color:#c8a1ff">Reference / missed</span><span style="color:#58d9a2">Prediction / matched</span><span style="color:#ff7b7b">Prediction / extra</span></p><div id="roll"></div>
<div id="exports"></div><button id="download-json">Download exact note JSON</button>
<h3>Source time → packet availability</h3><p>Each line joins an estimated source boundary to the packet where the decision became available. White is source time; cyan is packet availability. End-of-input closures are tagged as flushes.</p><div id="events"></div>
<label><input id="showall" type="checkbox"> Show all events, including those after playback</label><p id="eventcount"></p><div id="eventwrap"><table><thead><tr><th>Event</th><th>Pitch</th><th>String / fret</th><th>Source (s)</th><th>Packet (s)</th></tr></thead><tbody id="eventrows"></tbody></table></div></section>
<section><h2>When were matched onsets available?</h2>__LATENCY__<p class="notice">Packet availability only: a zero-processing lower bound. This stage does not rerun the neural network or measure processing, microphone, sound-card or live-player latency. Missed notes have no finite delay; matched subsets change with settings.</p></section>
<section><details><summary>Protocol, scope and saved artifacts</summary><p>24 unchanged-decoder replays reproduce the Stage 13 source notes, packet events and scores exactly after removing only measured completion timestamps. There are 648 grid cases, 48 ideal-label cases, and 48 original/selected review views. Both models retain 500 historical lifetime updates; this stage adds zero updates and performs zero neural or test inference calls.</p><p>Repeated development selection can overstate gains. No fresh electric-guitar or held-out-player claim is supported. Ideal labels mask ambiguous string states as inactive. EOF closures are censored and cannot recover the truncated delayed tail.</p><pre>__SELECTION__</pre><p><a href="protocol.json">Frozen protocol</a><a href="grid.json">Full grid</a><a href="oracle.json">Ideal-label records</a><a href="audit.json">Reproduction audit</a><a href="viewer-validation.json">Viewer checks</a></p></details></section>
<p><a href="report.md">Report</a><a href="findings.md">Findings</a><a href="validation/metrics.csv">Per-recording scores</a><a href="presentation-provenance.json">Presentation provenance</a></p>
</main><script>
'use strict';const DATA=__PAYLOAD__,LABELS=__LABELS__;let current;const $=id=>document.getElementById(id),audio=$('audio');
const ids=[...new Set(DATA.map(r=>r.id))];for(const id of ids)$('clip').add(new Option(id,id));for(const [id,name]of Object.entries(LABELS))$('mode').add(new Option(name,id));
function eventSvg(events){let out='<svg viewBox="0 0 1000 210" aria-label="Source and packet availability">';for(let s=0;s<=12;s++){let x=45+s/12*945;out+=`<path d="M${x},15V185" stroke="#29384b"/><text x="${x}" y="205" fill="#a1b7d3" font-size="11">${s}s</text>`;}for(let s=0;s<6;s++)out+=`<text x="3" y="${31+s*27}" fill="#a1b7d3" font-size="11">S${s}</text>`;for(const e of events){let x=45+e.source_time/12*945,z=45+e.emitted_at_audio_seconds/12*945,y=27+e.string*27+(e.type==='note_off'?6:0);out+=`<g class="event" data-source="${e.source_time}" data-packet="${e.emitted_at_audio_seconds}"><title>${e.type} MIDI ${e.pitch}; source ${e.source_time.toFixed(3)}s; packet ${e.emitted_at_audio_seconds.toFixed(3)}s${e.flush?' (flush)':''}</title><path d="M${x},${y}H${z}" stroke="#3b5b79"/><circle cx="${x}" cy="${y}" r="2" fill="#fff"/><circle cx="${z}" cy="${y}" r="2" fill="#75e6c4"/></g>`;}return out+'<line id="packet-playhead" x1="45" x2="45" y1="15" y2="185" stroke="#f1bb6c" stroke-width="2"/></svg>';}
function render(){const next=DATA.find(r=>r.id===$('clip').value&&r.mode===$('mode').value),same=current&&current.id===next.id;current=next;if(!same){audio.src=current.audio;audio.load();}$('roll').innerHTML=current.svg;$('events').innerHTML=eventSvg(current.events);$('currentinfo').textContent=current.label+' · onset F1 '+current.f1.toFixed(3)+' · '+JSON.stringify(current.parameters);$('exports').replaceChildren();for(const [name,link]of Object.entries(current.links)){const a=document.createElement('a');a.textContent=name;a.href=link;a.download='';$('exports').appendChild(a);}update();}
function update(){if(!current)return;const t=audio.currentTime||0;$('clock').textContent=t.toFixed(3)+' s';const x=45+Math.min(t,10)/10*945,p=45+Math.min(t,12)/12*945;for(const [id,v]of [['source-playhead',x],['packet-playhead',p]]){const el=$(id);if(el){el.setAttribute('x1',v);el.setAttribute('x2',v);}}const events=$('showall').checked?current.events:current.events.filter(e=>e.emitted_at_audio_seconds<=t);$('eventcount').textContent=events.length+' of '+current.events.length+' events available';$('eventrows').replaceChildren();for(const e of events.slice(-40)){const tr=document.createElement('tr');for(const value of [e.type+(e.flush?' (flush)':''),e.pitch,e.string+' / '+e.fret,e.source_time.toFixed(3),e.emitted_at_audio_seconds.toFixed(3)]){const td=document.createElement('td');td.textContent=value;tr.appendChild(td);}$('eventrows').appendChild(tr);}}
$('clip').onchange=render;$('mode').onchange=render;$('compare').onclick=()=>{$('mode').value=current.mode.endsWith('_strict')?current.mode.replace(/_strict$/,'_selected'):current.mode.replace(/_selected$/,'_strict');render();};$('showall').onchange=update;audio.addEventListener('timeupdate',update);audio.addEventListener('seeked',update);$('download-json').onclick=()=>{const u=URL.createObjectURL(new Blob([current.json],{type:'application/json'}));const a=document.createElement('a');a.href=u;a.download=current.id+'-'+current.mode+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);};render();
</script></body></html>'''
