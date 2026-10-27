"""Four head combinations, two packet clocks, and inspectable note evidence."""

import html
import json
from pathlib import Path
from urllib.parse import quote
from baseline.download import ROOT,sha256
from baseline.report import roll_svg
from decoder14.report import TEMPLATE as STYLE_SOURCE
from decoder14.workflow import read
from diagnosis15.analysis import CATEGORIES
from streaming.report import _table
from .workflow import load_cases,save

MODEL_LABELS={'short_short':'Short + short','short_long':'Short + long'}
LABELS={'matched_same_string':'Matched pitch and string','matched_other_string':'Matched pitch, other string',
    'nearby_same_pitch':'Same pitch within 200 ms','nearby_octave':'Octave within 50 ms',
    'nearby_other_pitch':'Other pitch within 50 ms','missed':'No paired prediction'}


def percent(value):return '—' if value is None else f'{100*value:.1f}%'


def write_report(output):
    output=Path(output);summary=read(output/'summary.json');cases=load_cases(output)
    modes=summary['modes'];frames=summary['frames'];delta=summary['contrasts']
    onset_gains=[delta['onset_swap_at_control_state'],delta['onset_swap_at_fused_state']]
    state_gains=[delta['state_swap_at_control_onset'],delta['state_swap_at_fused_onset']]
    outcome=(f"Changing the state head changes note-onset F1 by {min(state_gains):+.5f} to {max(state_gains):+.5f}; "
             f"changing only the onset head changes it by {min(onset_gains):+.5f} to {max(onset_gains):+.5f}.")
    cross_headers=['State source','Onset source','Note-onset F1','Onset + offset F1','String F1','Matched','Official extra notes']
    cross_rows=[[MODEL_LABELS[r['state_source']],MODEL_LABELS[r['onset_source']],f"{r['onset_f1']:.4f}",
        f"{r['onset_offset_f1']:.4f}",f"{r['string_onset_f1']:.4f}",r['matched_onsets'],r['official_extra_notes']] for r in modes.values()]
    policy_rows=[];activity_rows=[];partition_rows=[];pitch_rows=[];category_rows=[]
    for model,label in MODEL_LABELS.items():
        row=frames[model];argmax=row['policies']['argmax'];activity=row['policies']['decoder_activity'];counts=activity['counts']
        policy_rows.append([label,f"{argmax['exact_state_mean_clip_f1']:.4f}",f"{activity['exact_state_mean_clip_f1']:.4f}",
            f"{activity['exact_state_pooled']['f1']:.4f}"])
        activity_rows.append([label,percent(activity['activity_pooled']['precision']),percent(activity['activity_pooled']['recall']),
            percent(row['best_active_fret_on_reference_active']['accuracy']),
            percent(activity['fret_given_reference_and_prediction_active']['accuracy'])])
        partition_rows.append([label,counts['reference_active_frames']]+[str(counts[k])+f" ({100*counts[k]/counts['reference_active_frames']:.1f}%)"
            for k in ['correct_fret','predicted_inactive','wrong_octave_same_string','other_wrong_fret']])
        pitch_rows.append([label,f"{row['state_derived_pitch']['mean_clip_f1']:.4f}",f"{row['auxiliary_pitch']['mean_clip_f1']:.4f}"])
    for mode,row in modes.items():
        category_rows.append([MODEL_LABELS[row['state_source']]+' / '+MODEL_LABELS[row['onset_source']]]+
            [row['reference_categories'][k] for k in CATEGORIES]+[row['unpaired_after_associations']])
    contrast_rows=[[label,f'{delta[key]:+.5f}'] for key,label in [
        ('onset_swap_at_control_state','Change onset; short + short state fixed'),
        ('onset_swap_at_fused_state','Change onset; short + long state fixed'),
        ('state_swap_at_control_onset','Change state; short + short onset fixed'),
        ('state_swap_at_fused_onset','Change state; short + long onset fixed'),('interaction','Interaction between the two changes')]]
    changes=summary['original_note_changes']
    values={'scores':_table(cross_headers,cross_rows),
        'contrasts':_table(['Fixed-decoder comparison','Change in macro note-onset F1'],contrast_rows),
        'policies':_table(['State model','Argmax state F1 / mean clip','Activity-rule state F1 / mean clip','Activity-rule state F1 / pooled'],policy_rows),
        'activity':_table(['State model','Activity precision / pooled','Activity recall / pooled','Best fret on reference-active frames','Correct fret when both active'],activity_rows),
        'partition':_table(['State model','Reference-active string frames','Correct fret','Predicted inactive','Wrong octave on same string','Other wrong fret'],partition_rows),
        'pitch':_table(['State model','Pitch union from states / mean F1','Separate pitch head / mean F1'],pitch_rows),
        'categories':_table(['State / onset source']+[LABELS[k] for k in CATEGORIES]+['Predictions unpaired after associations'],category_rows),
        'changes':_table(['Original-model note match status','Reference notes'],[[label,changes[key]] for key,label in
            [('matched_both','Matched by both'),('fused_only','Matched by combined features only'),
             ('control_only','Matched by short features only'),('missed_both','Matched by neither')]])}
    payload=[]
    for row in cases:
        clip,mode=row['id'],row['mode'];metrics=row['pitch_metrics'];stem=Path('validation/results')/mode/clip
        svg=roll_svg(metrics['reference'],metrics['predicted'],metrics['onset_matches'],10.,clip)
        svg=svg.replace('</svg>','<line id="source-playhead" x1="45" x2="45" y1="20" y2="215" stroke="white" stroke-width="2"/></svg>')
        payload.append({'id':clip,'mode':mode,'state_source':row['state_source'],'onset_source':row['onset_source'],
            'original':row['original_combination'],'notes':row['notes'],'events':row['events'],
            'reference_rows':row['taxonomy']['reference_rows'],'categories':row['taxonomy']['reference_counts'],
            'svg':svg,'audio':quote(row['relativeaudio'],safe='/'),'f1':metrics['onset']['f1'],
            'json':(output/stem.with_suffix('.json')).read_bytes().decode('utf-8'),
            'links':{name:quote(stem.as_posix()+ext,safe='/') for name,ext in
                [('Notes JSON','.json'),('Notes CSV','.csv'),('MIDI','.mid'),('Metrics','-metrics.json'),
                 ('Events','-events.json'),('Evidence CSV','-evidence.csv')]}})
    values.update(outcome=html.escape(outcome),payload=json.dumps(payload).replace('<','\\u003c'),
        model_labels=json.dumps(MODEL_LABELS),labels=json.dumps(LABELS),
        css=STYLE_SOURCE.split('<style>',1)[1].split('</style>',1)[0])
    page=TEMPLATE
    for key,value in values.items():page=page.replace('__'+key.upper()+'__',value)
    (output/'review.html').write_text(page,encoding='utf-8')
    lines=['# Stage18: frozen state/onset head diagnosis','',outcome,'',
        'Zero new optimizer updates, zero neural inference and zero test inference. All24 original model/clip cases reproduce their Stage17 source decisions and scores exactly before24 crossed cases are evaluated.','',
        '| '+' | '.join(cross_headers)+' |','| '+' | '.join(['---']*len(cross_headers))+' |']
    lines+=['| '+' | '.join(map(str,row))+' |' for row in cross_rows]
    lines+=['','All four combinations use the same decoder: onset0.70, activity odds0.25, global source correction130ms and no extra onset backdate. New notes and fret changes open after four stable state frames; onset spikes only retrigger an active note at the same fret. Small onset contrasts therefore do not establish that onset prediction is unimportant under another decoder. Offset/pitch heads follow the state source and do not drive this decoder. Crossed heads are synthetic diagnostics with possible calibration mismatch; no hybrid is selected or promoted.','',
        '## Frame policies and aggregation','',
        '| State model | Argmax mean F1 | Decoder-rule mean F1 | Decoder-rule pooled F1 |','| --- | ---: | ---: | ---: |']
    lines+=['| '+' | '.join(map(str,row))+' |' for row in policy_rows]
    lines+=['','Argmax is the unchanged Stage17 selection metric. The activity rule is the existing best-active/(best-active+inactive+1e-8)>=0.25 gate, not a newly tuned threshold. Frame diagnoses omit the first26 source frames and per-string conflicts; note scoring retains the original0–10s source span and12s input.','',
        'Equal-clip means and pooled counts weight the data differently. Combined features slightly raise activity-rule mean state F1 but lower pooled exact-state F1. They raise activity recall while lowering conditional fret accuracy; this is not a uniform improvement or a reason to rewrite the prior selection.','',
        'Reference-note categories are one-to-one descriptive associations. Nearby wrong-pitch/octave pairings are not official matches or proven error causes. Predictions left unpaired after these associations differ from official extra-note counts.','',
        'Source and packet clocks are available from cached replay. No processing completion, real-time factor, microphone latency or hybrid speed is measured. Parent completion times are deliberately omitted.','',
        'The next proposed controlled question is active-only conditional-fret auxiliary supervision, with the existing joint state loss retained. It targets observed fret-choice errors without assuming that a loss change will help. Any future comparison needs matched controls, an explicit budget, frozen decoding and fresh independent recordings before a product-quality claim.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(output/'presentation-provenance.json',{'report_source_sha256':sha256(Path(__file__)),
        'borrowed_source_sha256':{name:sha256(ROOT/name) for name in ['decoder14/report.py','baseline/report.py','streaming/report.py']},
        'review_html_sha256':sha256(output/'review.html'),'report_md_sha256':sha256(output/'report.md'),
        'numeric_artifact_sha256':{name:sha256(output/name) for name in ['protocol.json','run.json','summary.json']},
        'scope':'Browser QA must bind the exact final HTML; cached replay has no measured completion clock.'})


TEMPLATE=r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 18 · State and onset diagnosis</title><style>__CSS__
.tablescroll{max-height:430px;overflow:auto}#events svg{width:100%;min-width:600px}#events{overflow:auto}button{margin:6px 8px 6px 0}td button{white-space:nowrap}.footnote{color:#a1b7d3}</style></head><body><main>
<div class="kicker">Stage 18 · Zero training · Cached outputs only</div><h1>What changes when the heads are crossed?</h1><p>__OUTCOME__</p>
<p class="notice">All 24 original model/clip cases reproduce Stage 17 exactly. The other 24 cases combine frozen outputs for diagnosis. No model or threshold is selected, and the existing default remains unchanged.</p>
<section><h2>Four fixed combinations</h2>__SCORES__<p>Two original models and two crossed combinations. New notes and fret changes open after four stable state frames. Onset spikes only retrigger an active note at the same fret. This limited role matters: small onset contrasts do not establish that onset prediction is unimportant with another decoder. Offset and pitch heads follow the state source and do not drive the current decoder. All scores use the same twelve reused development clips.</p>__CONTRASTS__<p>These are differences of mean clip note-onset F1 under the fixed decoder. Cross-head calibration mismatch prevents treating the differences as pure causal attribution or a deployable improvement.</p></section>
<section><h2>Why frame and note scores can disagree</h2>__POLICIES__<p>Stage 17 uses argmax state F1 for selection. The note decoder uses a fixed, more permissive activity rule: best active probability divided by best active plus inactive probability, with a small numerical stabilizer, must reach 0.25. We measure both policies without retuning.</p><p>Mean clip F1 gives equal weight to each clip; pooled F1 counts all scored string frames together. Their rankings differ here. The small decoder-rule mean gain does not overturn the previous argmax-based decision.</p>__ACTIVITY__<p>Activity asks whether a string is sounding, regardless of fret. Best-fret accuracy evaluates the most likely active fret on every reference-active frame. The last column evaluates fret choice only when both reference and prediction are active. Different denominators must not be compared as equivalent accuracies.</p></section>
<section><h2>What happens on sounding strings?</h2>__PARTITION__<p>These categories partition reference-active string frames using the decoder activity rule. Wrong-octave counts here mean a 12-fret error on the same string. They differ from nearby-octave note associations across strings.</p>__PITCH__<p>Frames compare recorded predictions 13:1013 against original targets 0:1000, excluding the first 26 source frames and per-string state conflicts. The pitch head threshold stays 0.5. String/fret and pitch-union metrics measure different tasks.</p></section>
<section><h2>Which reference notes change?</h2>__CHANGES__<p>Official one-to-one pitch/onset matches on the same 753 reference notes. Recovered and lost notes can coexist; a net gain does not imply every passage improved.</p><details><summary>Nearby associations and residual predictions</summary>__CATEGORIES__<p>After official matches, remaining notes are paired one-to-one in this order: same pitch within 200 ms, octave within 50 ms, other pitch within 50 ms. These are descriptive associations. Predictions unpaired after associations are different from official extra notes.</p></details></section>
<section><h2>Listen and inspect</h2><label>Recording<br><select id="clip"></select></label><label>State source<br><select id="state"></select></label><label>Onset source<br><select id="onset"></select></label>
<div><button id="swap-state">Switch state source</button><button id="swap-onset">Switch onset source</button></div>
<audio id="audio" controls preload="metadata"></audio><p id="currentinfo"></p><p id="clock">0.000 s</p>
<p class="legend"><span style="color:#739cff">Reference / matched</span><span style="color:#c8a1ff">Reference / missed</span><span style="color:#58d9a2">Prediction / matched</span><span style="color:#ff7b7b">Prediction / extra</span></p><div id="roll"></div>
<div id="exports"></div><button id="download-json">Download exact note JSON</button>
<h3>Estimated source → packet availability</h3><p>White marks source time; blue marks the first packet containing the decision. End-of-input closures are tagged as flushes. No processing completion or microphone delay is measured in this cached replay.</p><div id="events"></div>
<label><input id="showall" type="checkbox"> Show all packet events, including after playback</label><p id="eventcount"></p><div class="tablescroll"><table><thead><tr><th>Event</th><th>Pitch</th><th>Source (s)</th><th>Packet (s)</th></tr></thead><tbody id="eventrows"></tbody></table></div>
<h3>Reference-note evidence</h3><label>Association<br><select id="filter"><option value="all">All reference notes</option></select></label><p id="rowcount"></p>
<div class="tablescroll"><table><thead><tr><th>Seek</th><th>Pitch / string / fret</th><th>Association</th><th>Valid frames</th><th>Same string + fret</th><th>Any-string pitch</th><th>Separate pitch head</th></tr></thead><tbody id="noterows"></tbody></table></div><p class="footnote">Evidence uses the selected state source over each reference note's valid frames. A dash means no valid frame evidence; nearby associations are not official matches.</p></section>
<section><h2>Next controlled question</h2><p>The observed errors support testing active-only conditional-fret auxiliary supervision while retaining the existing joint state loss. This is a future hypothesis, not an improvement demonstrated by this diagnosis. The proposed matched continuation has a 500-new-update cap; it has not run.</p><p>One seed, short training and reused validation limit the evidence. Independent electric and distorted-guitar recordings remain necessary. The Stage 17 selection and current default stay unchanged.</p></section>
<p><a href="report.md">Report</a><a href="findings.md">Findings</a><a href="summary.json">Full diagnosis</a><a href="audit.json">Reproduction audit</a><a href="viewer-validation.json">Viewer checks</a><a href="next-experiment.json">Proposed next experiment</a></p>
</main><script>
'use strict';const DATA=__PAYLOAD__,MODELS=__MODEL_LABELS__,LABELS=__LABELS__;const $=id=>document.getElementById(id),audio=$('audio');let current;
for(const id of [...new Set(DATA.map(r=>r.id))])$('clip').add(new Option(id,id));for(const [k,v]of Object.entries(MODELS)){for(const id of ['state','onset'])$(id).add(new Option(v,k));}for(const[k,v]of Object.entries(LABELS))$('filter').add(new Option(v,k));
function eventSvg(events){let out='<svg viewBox="0 0 1000 210" aria-label="Estimated source and known packet availability" data-duration="12">';for(let s=0;s<=12;s++){const x=45+s/12*945;out+=`<path d="M${x},15V185" stroke="#29384b"/><text x="${x}" y="205" fill="#a1b7d3" font-size="11">${s}s</text>`;}for(let s=0;s<6;s++)out+=`<text x="3" y="${31+s*27}" fill="#a1b7d3" font-size="11">S${s}</text>`;for(const e of events){const x=45+e.source_time/12*945,p=45+e.emitted_at_audio_seconds/12*945,y=27+e.string*27+(e.type==='note_off'?6:0);out+=`<g class="event" data-source="${e.source_time}" data-packet="${e.emitted_at_audio_seconds}"><title>${e.type} MIDI ${e.pitch}; source ${e.source_time.toFixed(3)}s; packet ${e.emitted_at_audio_seconds.toFixed(3)}s${e.flush?' (flush)':''}</title><path d="M${x},${y}H${p}" stroke="#3b5b79"/><circle cx="${x}" cy="${y}" r="2" fill="white"/><circle cx="${p}" cy="${y}" r="2" fill="#739cff"/></g>`;}return out+'<line id="packet-playhead" x1="45" x2="45" y1="15" y2="185" stroke="#f1bb6c" stroke-width="2"/></svg>';}
function renderNotes(){const rows=current.reference_rows.filter(r=>$('filter').value==='all'||r.category===$('filter').value);$('rowcount').textContent=rows.length+' of '+current.reference_rows.length+' reference notes';$('noterows').replaceChildren();for(const r of rows){const tr=document.createElement('tr');tr.dataset.referenceIndex=r.reference_index;tr.dataset.category=r.category;const td=document.createElement('td'),b=document.createElement('button');b.textContent=r.note.start.toFixed(3)+' s';b.onclick=()=>{audio.currentTime=r.note.start;};td.appendChild(b);tr.appendChild(td);const evidence=r.evidence;for(const value of [`${r.note.pitch} / S${r.note.string} / ${r.note.fret}`,LABELS[r.category],evidence.valid_source_frames,...['same_string_fret_fraction','any_string_pitch_fraction','auxiliary_pitch_fraction'].map(k=>evidence[k]===null?'—':(100*evidence[k]).toFixed(0)+'%')]){const cell=document.createElement('td');cell.textContent=value;tr.appendChild(cell);}$('noterows').appendChild(tr);}}
function render(){const next=DATA.find(r=>r.id===$('clip').value&&r.state_source===$('state').value&&r.onset_source===$('onset').value),same=current&&current.id===next.id;current=next;if(!same){audio.src=current.audio;audio.load();}$('roll').innerHTML=current.svg;$('events').innerHTML=eventSvg(current.events);$('currentinfo').textContent=(current.original?'Original model':'Crossed-head diagnostic')+' · state: '+MODELS[current.state_source]+' · onset: '+MODELS[current.onset_source]+' · onset F1 '+current.f1.toFixed(4);$('exports').replaceChildren();for(const [name,link]of Object.entries(current.links)){const a=document.createElement('a');a.textContent=name;a.href=link;a.download='';$('exports').appendChild(a);}renderNotes();update();}
function update(){if(!current)return;const time=audio.currentTime||0;$('clock').textContent=time.toFixed(3)+' s';for(const[id,x]of [['source-playhead',45+Math.min(time,10)/10*945],['packet-playhead',45+Math.min(time,12)/12*945]])for(const k of ['x1','x2'])$(id).setAttribute(k,x);const rows=$('showall').checked?current.events:current.events.filter(e=>e.emitted_at_audio_seconds<=time);$('eventcount').textContent=rows.length+' of '+current.events.length+' events available by packet';$('eventrows').replaceChildren();for(const e of rows.slice(-40)){const tr=document.createElement('tr');for(const value of [e.type+(e.flush?' (flush)':''),e.pitch,e.source_time.toFixed(3),e.emitted_at_audio_seconds.toFixed(3)]){const td=document.createElement('td');td.textContent=value;tr.appendChild(td);}$('eventrows').appendChild(tr);}}
for(const id of ['clip','state','onset'])$(id).onchange=render;for(const [button,head]of [['swap-state','state'],['swap-onset','onset']])$(button).onclick=()=>{$(head).value=$(head).value==='short_short'?'short_long':'short_short';render();};$('filter').onchange=renderNotes;$('showall').onchange=update;audio.addEventListener('timeupdate',update);audio.addEventListener('seeked',update);$('download-json').onclick=()=>{const url=URL.createObjectURL(new Blob([current.json],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=current.id+'-'+current.mode+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};render();
</script></body></html>'''
