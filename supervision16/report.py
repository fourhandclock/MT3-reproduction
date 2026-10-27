"""Matched-loss study: parent reference and both fixed-final continuations."""

import html
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np

from baseline.download import ROOT,sha256
from baseline.report import roll_svg
from decoder14.report import TEMPLATE as PARENT_TEMPLATE
from decoder14.workflow import read
from streaming.evaluate import save_json
from streaming.report import _table


LABELS={'parent500':'Parent / 500 updates','control750':'Original loss / 750 updates','state_pitch750':'Added pitch loss / 750 updates'}


def write_report(output):
    output=Path(output)
    run=read(output/'validation/run.json');summary=read(output/'validation/summary.json')
    records=[read(path) for path in sorted((output/'validation/results').glob('*/*-metrics.json'))]
    modes=summary['modes'];training=run['training_summary'];selected=run['selection']['selected_arm']
    control,treatment=modes['control750'],modes['state_pitch750']
    state_delta=treatment['exact_state_f1']-control['exact_state_f1']
    onset_delta=treatment['onset_f1']-control['onset_f1']
    outcome=(f"The added loss changes exact state F1 by {state_delta:+.5f} and note-onset F1 by {onset_delta:+.5f} versus the matched original-loss control. "
             f"The declared rule selects {LABELS[selected].lower()} for development.")
    headers=['Model','State F1','Note-onset F1','Onset + offset F1','String F1','Frame F1']
    score_rows=[[label]+[f"{modes[mode][key]:.4f}" for key in ['exact_state_f1','onset_f1','onset_offset_f1','tab_onset_f1','frame_f1']]
                for mode,label in LABELS.items()]
    pitch_rows=[[label]+[f"{modes[mode][key]:.4f}" for key in ['state_derived_pitch_f1','auxiliary_pitch_f1','low_register_pitch_f1']]
                for mode,label in LABELS.items()]
    cost_rows=[];latency_rows=[]
    for mode,label in LABELS.items():
        receipt=training['arms'].get('control' if mode=='control750' else 'state_pitch') if mode!='parent500' else None
        cost_rows.append([label,0 if receipt is None else receipt['new_optimizer_steps'],500 if receipt is None else 750,
                          '—' if receipt is None else f"{receipt['training_seconds']:.1f}",
                          '—' if receipt is None else f"{receipt['peak_gpu_mib']:.1f}",f"{modes[mode]['real_time_factor']:.3f}"])
        latency_rows.append([label,modes[mode]['matched_onset_count'],f"{modes[mode]['matched_onset_delivery_p50_ms']:.1f}",
                             f"{modes[mode]['matched_onset_delivery_p95_ms']:.1f}",
                             sum(r['pitch_metrics']['errors']['extra_notes'] for r in records if r['mode']==mode)])
    history_rows=[]
    for arm,receipt in training['arms'].items():
        for block in receipt['history']:
            losses=block['training_losses']
            history_rows.append([arm,block['data_epoch'],block['optimizer_steps'],f"{losses['state']:.4f}",
                                  '—' if 'state_pitch' not in losses else f"{losses['state_pitch']:.4f}",f"{losses['total']:.4f}"])
    tables={'scores':_table(headers,score_rows),'pitch':_table(['Model','State-derived pitch F1','Separate pitch head F1','MIDI 40–51 F1'],pitch_rows),
            'cost':_table(['Model','New updates','Lifetime updates','New training seconds','Peak tensor MiB','Current processing / audio'],cost_rows),
            'latency':_table(['Model','Matched onsets','Completion p50 ms','Completion p95 ms','Official extra notes'],latency_rows),
            'history':_table(['Arm','Data epoch','Lifetime updates','State loss','New pitch term','Total loss'],history_rows)}
    payload=[]
    for row in records:
        mode,clip=row['mode'],row['id'];metrics=row['pitch_metrics']
        stem=Path('validation/results')/mode/clip
        svg=roll_svg(metrics['reference'],metrics['predicted'],metrics['onset_matches'],10.,clip)
        svg=svg.replace('</svg>','<line id="source-playhead" x1="45" x2="45" y1="20" y2="215" stroke="white" stroke-width="2"/></svg>')
        payload.append({'id':clip,'mode':mode,'label':LABELS[mode],'notes':row['notes'],'events':row['events'],
            'svg':svg,'audio':quote(row['relativeaudio'],safe='/'),'f1':metrics['onset']['f1'],
            'json':(output/stem.with_suffix('.json')).read_bytes().decode('utf-8'),
            'links':{name:quote(stem.as_posix()+ext,safe='/') for name,ext in
                     [('JSON','.json'),('CSV','.csv'),('MIDI','.mid'),('Metrics','-metrics.json'),('Events','-events.json')]}})
    page=TEMPLATE
    values={**tables,'outcome':html.escape(outcome),'payload':json.dumps(payload).replace('<','\\u003c'),
            'labels':json.dumps(LABELS),'selection':html.escape(json.dumps(run['selection'],indent=2)),
            'css':PARENT_TEMPLATE.split('<style>',1)[1].split('</style>',1)[0]}
    for key,value in values.items():page=page.replace('__'+key.upper()+'__',value)
    (output/'review.html').write_text(page,encoding='utf-8')
    lines=['# Stage16: direct state-to-pitch supervision','',outcome,'',
        '500 new optimizer updates total:250 per arm, five new sampled-crop epoch blocks each. Both final models have750 lifetime updates. The one shared parent has500 historical unique updates; its history is not spent again.','',
        '| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    lines += ['| '+' | '.join(map(str,row))+' |' for row in score_rows]
    lines += ['', 'Both arms restore identical parent model/AdamW/AMP/CPU-CUDA RNG state and use the same1000 epoch11–15 crop/gain descriptors, constant0.0001 learning rate and fixed Stage14 decoder. Treatment adds only0.5-weight direct state-derived pitch BCE. This probability-space term runs in float32 outside CUDA autocast.','',
        'The parent is reproduced on all twelve validation clips. These are reused development results from one shared parent; there is no independent-test result or automatic default promotion.','',
        'State-derived pitch uses fixed activity odds0.25, auxiliary pitch uses0.5, and the low-register diagnostic covers MIDI40–51. Compare all metrics rather than treating primary selection as a universal improvement.','',
        'Current inference timing is freshly measured for all three models. Their inference architecture is identical; runtime differences do not establish a causal speed improvement. Completion delays use correctly matched notes only and omit misses; matched subsets differ.','',
        'Source compensation remains130ms once for every boundary, with no extra onset backdate. The event timeline separates source estimates, packet availability and measured virtual completion. No native microphone/audio-device timing is measured.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save_json(output/'presentation-provenance.json',{'report_source_sha256':sha256(Path(__file__)),
        'review_html_sha256':sha256(output/'review.html'),'report_md_sha256':sha256(output/'report.md'),
        'numeric_artifact_sha256':{name:sha256(output/name) for name in ['protocol.json','input-audit.json','training-summary.json',
                                   'selection.json','validation/run.json','validation/summary.json']},
        'scope':'Browser QA must match the exact final HTML hash. Rendering alone does not confirm QA.'})


TEMPLATE=r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 16 · Direct pitch supervision</title><style>__CSS__</style></head><body><main>
<div class="kicker">Stage 16 · 500 new updates total</div><h1>Does direct pitch supervision help?</h1><p>__OUTCOME__</p>
<p class="notice">The same twelve validation clips select and report these results. One shared parent, two matched continuations, one new loss term. The existing default stays unchanged.</p>
<section><h2>Matched final-model comparison</h2>__SCORES__<p>Exact string/fret frame F1 is primary; original-source note-onset F1 breaks ties. Only the two 750-update finals are eligible. The original-loss control wins exact ties. The 500-update parent is a dose reference.</p></section>
<section><h2>What happened to pitch evidence?</h2>__PITCH__<p>State-derived pitch uses activity odds 0.25; the separate pitch head uses probability 0.5. These thresholds stay fixed. Low-register pitch covers MIDI 40–51 and is not a selection metric.</p></section>
<section><h2>Budget and current cost</h2>__COST__<p>Each arm adds 250 updates over five new sampled-crop epoch blocks, reaching 750 lifetime updates. The one shared parent contributes 500 historical unique updates. The new stage spends 500 updates total, with zero new sanity optimization.</p><p>Peak tensors exclude other GPU allocations. Training seconds exclude setup/restoration/validation. All processing/audio timings are newly measured warm replays. The inference graph is identical across these models, so timing variation does not establish that the training loss makes inference faster.</p></section>
<section><details><summary>Training losses and restored-state controls</summary>__HISTORY__<p>Both arms restore the same model tensors, ordered AdamW moments/counters, AMP scaler and CPU/CUDA random states. Identical new crops/gains come from data epochs 11–15. Stored-end learning rate 0.0001 remains constant, with no early stopping or scheduler restart.</p><p>The original loss remains unchanged. Treatment adds 0.5 times direct pitch BCE through the existing string/fret probabilities, with unisons merged by an independent-string union. Crop/leading masks and any-string-conflict exclusion apply to the new term. It runs in float32 outside CUDA autocast and adds no model parameters.</p><p><a href="models/control/training-receipt.json">Control receipt</a><a href="models/state_pitch/training-receipt.json">Pitch-loss receipt</a><a href="training-summary.json">Full training summary</a></p></details></section>
<section><h2>Listen and compare</h2><label>Recording<br><select id="clip"></select></label><label>Model<br><select id="mode"></select></label><button id="compare">Compare original / added loss</button>
<audio id="audio" controls preload="metadata"></audio><p id="currentinfo"></p><p id="clock">0.000 s</p>
<p class="legend"><span style="color:#739cff">Reference / matched</span><span style="color:#c8a1ff">Reference / missed</span><span style="color:#58d9a2">Prediction / matched</span><span style="color:#ff7b7b">Prediction / extra</span></p><div id="roll"></div>
<div id="exports"></div><button id="download-json">Download exact note JSON</button><h3>Source → packet → virtual completion</h3>
<p>White marks estimated source time, blue marks packet availability, and green marks completion after measured processing. End-of-input closures are tagged as flushes. The fixed 130 ms source correction never advances packet or completion clocks.</p><div id="events"></div>
<label><input id="showall" type="checkbox"> Show all events, including those after playback</label><p id="eventcount"></p><div id="eventwrap"><table><thead><tr><th>Event</th><th>Pitch</th><th>Source (s)</th><th>Packet (s)</th><th>Completion (s)</th></tr></thead><tbody id="eventrows"></tbody></table></div></section>
<section><h2>When did matched onsets finish processing?</h2>__LATENCY__<p>Conditional virtual completion minus performed reference onset. Misses have no finite delay, and matched subsets differ. These file replays do not measure sound-card or microphone capture latency.</p></section>
<section><details><summary>Selection and limitations</summary><pre>__SELECTION__</pre><p>Parent probability and source-decision reproduction is recorded per clip. The same reused validation set is used for final selection and reporting. This single-parent, single-coefficient comparison cannot establish a general architecture ranking, a sole training cause, or new electric-guitar robustness.</p></details></section>
<p><a href="report.md">Report</a><a href="findings.md">Findings</a><a href="audit.json">Reproduction audit</a><a href="viewer-validation.json">Viewer checks</a><a href="presentation-provenance.json">Presentation provenance</a></p>
</main><script>
'use strict';const DATA=__PAYLOAD__,LABELS=__LABELS__;let current;const $=id=>document.getElementById(id),audio=$('audio');
for(const id of [...new Set(DATA.map(r=>r.id))])$('clip').add(new Option(id,id));for(const [key,value]of Object.entries(LABELS))$('mode').add(new Option(value,key));
function eventSvg(events){const duration=Math.max(12,...events.map(e=>e.delivered_at_seconds));let out=`<svg viewBox="0 0 1000 210" data-duration="${duration}" aria-label="Source, packet and measured virtual completion">`;for(let s=0;s<=Math.floor(duration);s++){let x=45+s/duration*945;out+=`<path d="M${x},15V185" stroke="#29384b"/><text x="${x}" y="205" fill="#a1b7d3" font-size="11">${s}s</text>`;}for(let s=0;s<6;s++)out+=`<text x="3" y="${31+s*27}" fill="#a1b7d3" font-size="11">S${s}</text>`;for(const e of events){let x=45+e.source_time/duration*945,p=45+e.emitted_at_audio_seconds/duration*945,z=45+e.delivered_at_seconds/duration*945,y=27+e.string*27+(e.type==='note_off'?6:0);out+=`<g class="event" data-source="${e.source_time}" data-packet="${e.emitted_at_audio_seconds}" data-completion="${e.delivered_at_seconds}"><title>${e.type} MIDI ${e.pitch}; source ${e.source_time.toFixed(3)}s; packet ${e.emitted_at_audio_seconds.toFixed(3)}s; completion ${e.delivered_at_seconds.toFixed(3)}s${e.flush?' (flush)':''}</title><path d="M${x},${y}H${z}" stroke="#3b5b79"/><circle cx="${x}" cy="${y}" r="2" fill="white"/><circle cx="${p}" cy="${y}" r="2" fill="#739cff"/><circle cx="${z}" cy="${y}" r="2" fill="#75e6c4"/></g>`;}return out+'<line id="delivery-playhead" x1="45" x2="45" y1="15" y2="185" stroke="#f1bb6c" stroke-width="2"/></svg>';}
function render(){const next=DATA.find(r=>r.id===$('clip').value&&r.mode===$('mode').value),same=current&&current.id===next.id;current=next;if(!same){audio.src=current.audio;audio.load();}$('roll').innerHTML=current.svg;$('events').innerHTML=eventSvg(current.events);$('currentinfo').textContent=current.label+' · onset F1 '+current.f1.toFixed(4);$('exports').replaceChildren();for(const [name,link]of Object.entries(current.links)){const a=document.createElement('a');a.textContent=name;a.href=link;a.download='';$('exports').appendChild(a);}update();}
function update(){if(!current)return;const time=audio.currentTime||0;$('clock').textContent=time.toFixed(3)+' s';const duration=Number(document.querySelector('#events svg').dataset.duration);for(const [id,x]of [['source-playhead',45+Math.min(time,10)/10*945],['delivery-playhead',45+Math.min(time,duration)/duration*945]]){for(const name of ['x1','x2'])$(id).setAttribute(name,x);}const visible=$('showall').checked?current.events:current.events.filter(e=>e.delivered_at_seconds<=time);$('eventcount').textContent=visible.length+' of '+current.events.length+' events completed';$('eventrows').replaceChildren();for(const e of visible.slice(-40)){const tr=document.createElement('tr');for(const value of [e.type+(e.flush?' (flush)':''),e.pitch,e.source_time.toFixed(3),e.emitted_at_audio_seconds.toFixed(3),e.delivered_at_seconds.toFixed(3)]){const td=document.createElement('td');td.textContent=value;tr.appendChild(td);}$('eventrows').appendChild(tr);}}
$('clip').onchange=render;$('mode').onchange=render;$('compare').onclick=()=>{$('mode').value=current.mode==='state_pitch750'?'control750':'state_pitch750';render();};$('showall').onchange=update;audio.addEventListener('timeupdate',update);audio.addEventListener('seeked',update);$('download-json').onclick=()=>{const url=URL.createObjectURL(new Blob([current.json],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=current.id+'-'+current.mode+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};render();
</script></body></html>'''
