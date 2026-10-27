"""Inspectable residual-note evidence with audio and exact local exports."""

import html
import json
from pathlib import Path
from urllib.parse import quote

from baseline.download import ROOT, sha256
from baseline.report import roll_svg
from decoder14.report import TEMPLATE as PARENT_TEMPLATE
from decoder14.workflow import read
from streaming.evaluate import save_json
from streaming.export import export
from streaming.report import _table
from .analysis import CATEGORIES
from .workflow import MODELS, load_records


LABELS = {'matched_same_string': 'Correct pitch and string', 'matched_other_string': 'Correct pitch, different string',
          'nearby_same_pitch': 'Same pitch within 200 ms', 'nearby_octave': 'Octave within 50 ms',
          'nearby_other_pitch': 'Other pitch within 50 ms', 'missed': 'No paired prediction'}
MODEL_LABELS = {MODELS[0]: 'Eight normalization groups', MODELS[1]: 'One normalization group'}


def write_report(output):
    output = Path(output)
    records = load_records(output)
    summary = read(output / 'summary.json')
    recommendation = read(output / 'recommendation.json')
    intervention_rows, category_rows, frame_rows, pitch_rows, repeat_rows = [], [], [], [], []
    for model in MODELS:
        value = summary[model]
        for variant, score in value['interventions'].items():
            intervention_rows.append([MODEL_LABELS[model], variant.replace('_', ' '), f"{score['onset_f1']:.3f}",
                                      f"{score['onset_offset_f1']:.3f}", score['matched_onsets'], score['predicted_notes']])
        category_rows.append([MODEL_LABELS[model]] + [value['reference_categories'][key] for key in CATEGORIES])
        counts = value['frame_counts']
        frame_rows.append([MODEL_LABELS[model], counts['reference_active_frames']] + [
            str(counts[key]) + f" ({100*counts[key]/counts['reference_active_frames']:.1f}%)" for key in
            ['correct_fret', 'predicted_inactive', 'wrong_octave_same_string', 'other_wrong_fret']])
        pitch_rows.append([MODEL_LABELS[model], f"{value['state_derived_pitch_f1']:.3f}",
                           f"{value['auxiliary_pitch_f1']:.3f}", value['target_run_p50_frames'], value['predicted_run_p50_frames']])
        matched = value['reference_categories']['matched_same_string'] + value['reference_categories']['matched_other_string']
        repeat_rows.append([MODEL_LABELS[model], f"{value['repeated_reference_matched']} / {value['repeated_reference_count']}",
                            value['reference_notes_with_multiple_overlaps'], value['prediction_count'] - matched,
                            value['extra_predictions']])
    tables = {'interventions': _table(['Frozen model', 'Head inputs', 'Onset F1', 'Onset + offset F1', 'Matched', 'Predicted'], intervention_rows),
              'categories': _table(['Frozen model'] + [LABELS[key] for key in CATEGORIES], category_rows),
              'frames': _table(['Frozen model', 'Reference active string-frames', 'Correct fret', 'Predicted inactive', 'Wrong octave', 'Other wrong fret'], frame_rows),
              'pitch': _table(['Frozen model', 'Pitch union from states F1', 'Separate pitch head F1', 'Label run median (frames)', 'Predicted run median (frames)'], pitch_rows),
              'repeats': _table(['Frozen model', 'Repeated notes matched / total', 'References with multiple overlaps', 'Official unmatched predictions', 'Unpaired after associations'], repeat_rows)}
    payload = []
    for row in records:
        model, clip = row['model'], row['id']
        stem = Path('validation/results') / model / clip
        note_stem = Path('validation/notes') / model / clip
        export(row['notes'], output / note_stem, {'stage': 15, 'model': model,
               'source': 'Unchanged calibrated Stage14 source notes, copied from verified saved records.',
               'no_new_training': True, 'source_delay_seconds': .13})
        metrics = row['pitch_metrics']
        svg = roll_svg(metrics['reference'], metrics['predicted'], metrics['onset_matches'], 10., clip)
        svg = svg.replace('</svg>', '<line id="source-playhead" x1="45" x2="45" y1="20" y2="215" stroke="white" stroke-width="2"/></svg>')
        payload.append({'id': clip, 'model': model, 'svg': svg, 'audio': quote(row['relativeaudio'], safe='/'),
                        'notes': row['notes'], 'reference_rows': row['taxonomy']['reference_rows'],
                        'categories': row['taxonomy']['reference_counts'], 'f1': metrics['onset']['f1'],
                        'json': (output / stem.with_suffix('.json')).read_bytes().decode('utf-8'),
                        'links': {'Diagnosis JSON': quote(stem.with_suffix('.json').as_posix(), safe='/'),
                                  'Evidence CSV': quote(stem.with_suffix('.csv').as_posix(), safe='/'),
                                  **{name: quote(note_stem.as_posix() + extension, safe='/') for name, extension in
                                     [('Notes JSON', '.json'), ('Notes CSV', '.csv'), ('MIDI', '.mid')]}}})
    outcome = 'Supplying ideal string/fret predictions helps far more than supplying ideal onsets. The existing models and their measured accuracy are unchanged.'
    proposed = ('The declared checks support a controlled test of direct pitch supervision for the existing string/fret logits. '
                'A loss module is implemented and tested; no acoustic training has been performed with it.'
                if recommendation['supported_by_declared_rule'] else
                'The declared checks do not yet support the proposed pitch-supervision experiment. No acoustic training has been performed with it.')
    values = {**tables, 'outcome': html.escape(outcome), 'proposed': html.escape(proposed),
              'payload': json.dumps(payload).replace('<', '\\u003c'), 'labels': json.dumps(LABELS),
              'model_labels': json.dumps(MODEL_LABELS), 'recommendation': html.escape(json.dumps(recommendation, indent=2)),
              'css': PARENT_TEMPLATE.split('<style>', 1)[1].split('</style>', 1)[0]}
    page = TEMPLATE
    for key, value in values.items():
        page = page.replace('__' + key.upper() + '__', value)
    (output / 'review.html').write_text(page, encoding='utf-8')
    lines = ['# Stage 15: residual acoustic errors', '', outcome, '',
             'Zero new optimizer updates, zero neural inference, zero test inference. This stage reuses development recordings.', '',
             '## Fixed-head interventions', '',
             '| Model | Inputs | Onset F1 | Onset + offset F1 | Matched | Predicted |',
             '| --- | --- | --- | --- | --- | --- |']
    lines += ['| ' + ' | '.join(map(str, row)) + ' |' for row in intervention_rows]
    lines += ['', 'These privileged-label substitutions use the same calibrated decoder (0.70 onset / 0.25 activity / zero extra backdate) and fixed 130 ms source compensation throughout. The all-ideal scores therefore use different thresholds from the Stage 14 oracle experiment.', '',
              '## Acoustic frame evidence', '',
              '| Model | Reference active frames | Correct fret | Inactive | Wrong octave | Other wrong fret |',
              '| --- | --- | --- | --- | --- | --- |']
    lines += ['| ' + ' | '.join(map(str, row)) + ' |' for row in frame_rows]
    lines += ['', 'Frames use source0:1000 against predictions13:1013, with the first26 source frames masked. Pitch unions use state activity odds0.25 and the separate pitch head uses0.5. The comparison is threshold-dependent; neither threshold is retuned here.', '',
              'Reference-note associations are exclusive and one-to-one in the declared order. Nearby wrong-pitch associations are not proof that a specific note caused a miss. Multiple overlaps and repetition are separate tags.', '',
              '## Next controlled question', '', proposed, '',
              'The proposal restores the same 500-update eight-group parent into both arms, gives each 250 new updates at0.0001 over five additional sampled-crop epochs, and changes only one loss term. Both arms would reach750 lifetime updates; total future new budget500. This proposal has not run.', '',
              'The new term maps string/fret probabilities through standard tuning to pitch occupancy, collapses unisons with an independent-string union, and compares with delayed pitch labels using BCE. Crop/leading masks and any-string-conflict exclusion apply to this term. It adds no model parameters and does not infer which equivalent string fingering is preferred.', '',
              'No model, decoder default, fresh-data result, processing latency or training benefit is promoted.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    save_json(output / 'presentation-provenance.json', {'report_source_sha256': sha256(Path(__file__)),
              'review_html_sha256': sha256(output / 'review.html'), 'report_md_sha256': sha256(output / 'report.md'),
              'numeric_artifact_sha256': {name: sha256(output / name) for name in
                  ['protocol.json', 'summary.json', 'interventions.json', 'recommendation.json', 'next-experiment.json']},
              'scope': 'The browser record must match this exact HTML hash; generation alone does not confirm browser QA.'})


TEMPLATE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 15 · Residual acoustic errors</title><style>__CSS__
#note-list{max-height:440px;overflow:auto}#note-list table{font-size:12px}#note-list th{position:sticky;top:0;background:#17263b}#note-list td{white-space:normal}#note-list button{padding:5px 8px;font-size:11px}.counts{display:flex;gap:10px;flex-wrap:wrap}.pill{padding:9px 12px;border:1px solid #35516c;border-radius:8px;color:#b5cee9;font-size:12px}#currentinfo{margin:8px 0}.muted{color:#90a7c3}
</style></head><body><main><div class="kicker">Stage 15 · zero additional training</div>
<h1>Where are the notes going wrong?</h1><p>__OUTCOME__</p>
<p class="notice">These are reused validation recordings. Ideal-label interventions are diagnostic upper bounds with privileged information. No model was trained, and the existing defaults stay unchanged.</p>
<section><h2>Which output is limiting note detection?</h2>__INTERVENTIONS__<p>Every row uses the fixed Stage 14 selected decoder: onset 0.70, activity 0.25, zero extra onset backdate, and the same 130 ms supervision-delay correction. Only the named state/onset input is replaced. Offset and auxiliary pitch heads stay frozen and are unused by this decoder.</p><p>These oracle thresholds differ from the earlier Stage 14 ideal-label check. Compare within this table to isolate the named head substitution.</p></section>
<section><h2>What do the string/fret decisions predict?</h2>__FRAMES__<p>Counts are over reference-active string-frames, not whole-note scores. Source 0–10 s is aligned once to the 13-frame delayed predictions; the first 26 source frames are excluded. Ambiguous string targets are masked.</p></section>
<section><h2>Pitch evidence and short state runs</h2>__PITCH__<p>Both pitch scores compare the same 49 MIDI occupancies. The state union uses the fixed 0.25 activity odds; the separate pitch head uses 0.5. Their difference is threshold-dependent. A frame is 10 ms. Short state runs suggest unstable evidence, without proving one cause.</p></section>
<section><h2>How are note misses associated?</h2>__CATEGORIES__<p>Each row totals 753 reference attacks. Only the first two categories count as correct 50 ms pitch/onset matches. The remaining categories associate unmatched references with remaining predictions, in order, without reusing either note. Dense chords can produce coincidental associations; these labels do not identify causes.</p>__REPEATS__<p>Repeated notes are consecutive same-string, same-pitch reference notes separated by at most 60 ms, including overlap. Multiple overlaps require at least 20 ms of overlap with two or more same-string/pitch predictions. These tags are separate from the exclusive categories.</p></section>
<section><h2>Inspect the evidence for a note</h2><label>Recording<br><select id="clip"></select></label><label>Model<br><select id="model"></select></label><button id="compare">Switch model</button>
<audio id="audio" controls preload="metadata"></audio><p id="currentinfo"></p><p id="clock">0.000s</p>
<p class="legend"><span style="color:#739cff">Reference / matched</span><span style="color:#c8a1ff">Reference / missed</span><span style="color:#58d9a2">Prediction / matched</span><span style="color:#ff7b7b">Prediction / extra</span></p><div id="roll"></div>
<div id="exports"></div><button id="download-json">Download exact diagnosis JSON</button><div id="counts" class="counts"></div>
<p>Frame evidence is the fraction of valid, source-aligned frames inside that reference note. A dash means the note has no frames after the leading/conflict mask; it does not mean zero pitch confidence.</p>
<label>Association filter<br><select id="filter"><option value="all">All reference notes</option></select></label><p id="rowcount"></p>
<div id="note-list"><table><thead><tr><th>Source time</th><th>Note / position</th><th>Association</th><th>Repeated?</th><th>Correct string/fret</th><th>Pitch on any string</th><th>Separate pitch head</th></tr></thead><tbody id="noterows"></tbody></table></div></section>
<section><h2>One concrete training experiment to test next</h2><p>__PROPOSED__</p><p>The proposed term teaches the existing state logits to agree with pitch labels after standard-tuning projection. It merges equivalent unison pitches, adds no model parameters and cannot resolve every fingering ambiguity. The existing state, onset, offset and pitch losses remain unchanged.</p><p>Future matched comparison: restore the same 500-update eight-group parent into both arms, add 250 updates per arm over five sampled-crop epochs, and change only the added 0.5-weight pitch loss. Total future budget: 500 new updates; each arm would finish at 750 lifetime updates. This stage has spent none of that budget.</p><p><a href="next-experiment.json">Proposed protocol</a><a href="recommendation.json">Recommendation checks</a></p></section>
<section><details><summary>Reproducibility and limitations</summary><p>24 calibrated cases reproduce the Stage 14 notes, packet events and scores. 96 fixed-head intervention cases are recorded. Original checkpoints, cached probability arrays, annotations and source targets remain untouched. No test inference or fresh-domain result is claimed.</p><p>Nearby note associations, oracle gains and repeated development clips do not prove a training cause or guarantee that the proposed loss will help. There is no end-to-end latency claim.</p><pre>__RECOMMENDATION__</pre></details></section>
<p><a href="report.md">Report</a><a href="findings.md">Findings</a><a href="audit.json">Reproduction audit</a><a href="viewer-validation.json">Viewer checks</a><a href="presentation-provenance.json">Presentation provenance</a></p>
</main><script>
'use strict';const DATA=__PAYLOAD__,LABELS=__LABELS__,MODELS=__MODEL_LABELS__;let current;const $=id=>document.getElementById(id),audio=$('audio');
for(const id of [...new Set(DATA.map(r=>r.id))])$('clip').add(new Option(id,id));for(const [key,value]of Object.entries(MODELS))$('model').add(new Option(value,key));for(const [key,value]of Object.entries(LABELS))$('filter').add(new Option(value,key));
function fraction(value){return value===null?'—':(100*value).toFixed(0)+'%';}
function pitchName(pitch){return ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][pitch%12]+(Math.floor(pitch/12)-1);}
function renderRows(){const choice=$('filter').value,rows=current.reference_rows.filter(row=>choice==='all'||row.category===choice);$('rowcount').textContent=rows.length+' of '+current.reference_rows.length+' reference notes';$('noterows').replaceChildren();for(const row of rows){const tr=document.createElement('tr');tr.dataset.referenceIndex=row.reference_index;tr.dataset.category=row.category;const first=document.createElement('td'),seek=document.createElement('button');seek.textContent=row.note.start.toFixed(3)+'s';seek.setAttribute('aria-label','Seek to reference note '+row.reference_index);seek.onclick=()=>{audio.currentTime=row.note.start;};first.appendChild(seek);tr.appendChild(first);const ev=row.evidence;for(const value of [pitchName(row.note.pitch)+' · '+['Low E','A','D','G','B','High E'][row.note.string]+' fret'+row.note.fret,LABELS[row.category],row.repeated_note?'Yes':'—',fraction(ev.same_string_fret_fraction),fraction(ev.any_string_pitch_fraction),fraction(ev.auxiliary_pitch_fraction)]){const td=document.createElement('td');td.textContent=value;tr.appendChild(td);}$('noterows').appendChild(tr);}}
function render(){const next=DATA.find(r=>r.id===$('clip').value&&r.model===$('model').value),same=current&&current.id===next.id;current=next;if(!same){audio.src=current.audio;audio.load();}$('roll').innerHTML=current.svg;$('currentinfo').textContent=MODELS[current.model]+' · unchanged onset F1 '+current.f1.toFixed(3);$('exports').replaceChildren();for(const [name,link]of Object.entries(current.links)){const a=document.createElement('a');a.textContent=name;a.href=link;a.download='';$('exports').appendChild(a);}$('counts').replaceChildren();for(const [key,value]of Object.entries(current.categories)){const span=document.createElement('span');span.className='pill';span.textContent=LABELS[key]+': '+value;$('counts').appendChild(span);}renderRows();update();}
function update(){if(!current)return;const t=audio.currentTime||0;$('clock').textContent=t.toFixed(3)+'s';const x=45+Math.min(t,10)/10*945;for(const name of ['x1','x2'])$('source-playhead').setAttribute(name,x);}
$('clip').onchange=render;$('model').onchange=render;$('filter').onchange=renderRows;$('compare').onclick=()=>{const models=Object.keys(MODELS);$('model').value=models.find(m=>m!==current.model);render();};audio.addEventListener('timeupdate',update);audio.addEventListener('seeked',update);$('download-json').onclick=()=>{const url=URL.createObjectURL(new Blob([current.json],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=current.id+'-'+current.model+'-diagnosis.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};render();
</script></body></html>'''
