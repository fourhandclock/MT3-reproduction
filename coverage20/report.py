"""Local review of coverage denominators and archived errors, with exact exports."""

import base64
import json
from pathlib import Path
from urllib.parse import quote

from .data import BINS, MODES, read


def percentage(value):
    return 'n/a (zero support)' if value is None else f'{100 * value:.2f}%'


def findings(coverage, summary):
    control = summary['modes']['control500']['policies']['argmax']
    well = control['bins']['1000+']
    unseen = control['bins']['unseen']
    lines = ['# Stage 20: training coverage and saved errors', '',
             'Zero new optimizer updates, zero neural inference calls, no test inference and no model selection.', '',
             f"Every validation GT position had supervised exposure: the unseen bin has {unseen['reference_active']} "
             f"active string-frames and {unseen['reference_notes']} reference notes. This dataset therefore provides no unseen-position error-rate estimate.", '',
             f"For control500 argmax, the 1000+ presentation bin contains {well['reference_active']:,} of "
             f"{control['overall']['reference_active']:,} valid active validation string-frames and "
             f"{well['error_count']:,} errors ({percentage(well['share_of_all_active_errors'])} of all active errors). "
             f"Its within-bin error rate is {percentage(well['error_rate_on_reference_active'])}.", '',
             'This is an association with recorded exposure, not evidence that exposure caused an error or that more data cannot help.', '',
             '## Recorded supervision', '',
             '| History | Available active source frames | Repeated active presentations | Unique supervised source-frame/string identities | Positions presented |',
             '| --- | ---: | ---: | ---: | ---: |']
    for name, value in coverage['histories'].items():
        rows = coverage['classes'][name]
        totals = [sum(row[key] for row in rows) for key in ('available', 'presentations', 'unique')]
        lines.append(f"| {name} | {totals[0]:,} | {totals[1]:,} | {totals[2]:,} | {sum(r['presentations'] > 0 for r in rows)}/150 |")
    lines += ['', 'The parent uses epochs 1-5. Each child uses epochs 1-10. The two children share the same exposure history; it is not counted twice as new data.', '',
              'Available counts include all valid active labels in the 48 original target caches. Actual exposure uses full-recording delay13, delayed-grid leading26 and each crop\'s 26-frame edge masks. Delayed frame i maps to original frame i-13. Recording identity and string remain part of each unique source identity; overlapping/repeated crops add presentations, not unique frames.', '',
              '## Pooled development errors', '',
              '| Model / policy | GT-position exposure bin | Active GT frames | Incorrect | Error rate | Official reference notes | Matched pitch/onset notes | Note miss rate |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for mode in MODES:
        for policy, value in summary['modes'][mode]['policies'].items():
            for label in BINS:
                row = value['bins'][label]
                lines.append(f"| {mode} / {policy} | {label} | {row['reference_active']:,} | {row['error_count']:,} | "
                             f"{percentage(row['error_rate_on_reference_active'])} | {row['reference_notes']} | "
                             f"{row['official_matched_notes']} | {percentage(row['note_miss_rate'])} |")
    lines += ['', 'Frame policies use the same GT frame mask but different state decisions: ordinary argmax versus the fixed best-active/(best-active+inactive+1e-8) >= 0.25 activity rule. The latter is a raw-state diagnostic, not a new temporally decoded note stream.', '',
              'Reference onsets use official one-to-one pitch/onset matches from [0,10), including early notes before the 0.26s frame mask. Pitch/onset matching ignores string; the exports separately flag whether each official match uses the reference string. The same saved notes are used for both frame policies, so their note columns repeat by design.', '',
              'False activity is grouped separately by the predicted position\'s exposure, with all valid inactive GT string-frames as denominator. It is not assigned to a nonexistent active GT fret. Empty bins keep zero counts and null rates. These are pooled frame errors and note misses, not macro clip F1.', '',
              '## Pitch versus position', '',
              'Each exact position is tagged as presented, unpresented with the same pitch presented on another string, or pitch unpresented anywhere. Available, repeated and unique counts remain separate. See coverage.csv for all 150 positions in both histories and summary.json for each model\'s GT-position and pitch-group diagnostics.', '',
              '## Scope and next question', '',
              summary['limitations'], '',
              'A class with 1000 repeated presentations may still have few unique examples or limited recording diversity. Neighbouring frames and repeated crops are correlated. Bin associations also mix strings, frets, durations, musical contexts and model histories; no confidence or causal claim follows from the frame count alone.', '',
              'The next proposed check compares saved within-string fret decisions with simple training-frequency priors on reference-active frames. It asks whether errors on supported positions mainly reflect a frequent-fret prior. It uses existing targets/probabilities, zero new training and zero neural inference, and cannot select a production model. This follow-up is proposed, not executed.', '',
              'No new speed, latency or processing-clock measurement was made. Earlier saved clocks remain archived evidence. The offline default and Stage19 research selection are preserved.', '']
    return '\n'.join(lines)


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 20 - Coverage audit</title><style>
:root{font-family:Segoe UI,Arial,sans-serif;color:#e3edf5;background:#101a25;font-size:15px}body{max-width:1180px;margin:auto;padding:30px}h1{font-size:31px;margin-bottom:10px}h2{font-size:21px}.muted{color:#acbfce;line-height:1.6}.badge{display:inline-block;background:#20394b;padding:7px 12px;border-radius:20px;margin:0 7px 12px 0}a{color:#8bd8ec}label{display:inline-block;margin:0 18px 12px 0}select,button{background:#233647;color:#eff9ff;border:1px solid #53718a;border-radius:5px;padding:8px;max-width:100%}section{background:#172635;border:1px solid #324b60;border-radius:10px;padding:20px;margin-top:20px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.card{border-left:3px solid #66c5be;padding:8px 14px;background:#1a3040}.number{font-size:25px;display:block;margin:6px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:right;padding:9px;border-bottom:1px solid #334b5d;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{color:#a7dce9;background:#1e3344;position:sticky;top:0}#class-scroll{max-height:430px}.bar{height:6px;background:#d99766;margin-top:5px;border-radius:3px}.exports{display:flex;flex-wrap:wrap;gap:14px}.legend{font-size:13px;color:#b9ccd7}footer{margin-top:22px;line-height:1.6;color:#acbfce}@media(max-width:700px){body{padding:16px}.cards{grid-template-columns:1fr}h1{font-size:25px}section{padding:14px}}
</style><h1>Stage 20: is fret coverage the gap?</h1>
<p class="muted">A diagnosis of saved predictions against recorded training exposure. All twelve clips are reused development recordings. Every ground-truth validation position had already received supervision; the unseen bin has no GT support.</p>
<span class="badge">0 training updates</span><span class="badge">0 neural inference calls</span><span class="badge">48 training recordings</span><span class="badge">No new model selection</span>
<section><label>Saved model <select id="model"><option>parent250</option><option selected>control500</option><option>conditional_fret500</option></select></label>
<label>Raw state decision <select id="policy"><option value="argmax">Argmax</option><option value="decoder_activity">Fixed decoder activity rule</option></select></label>
<p id="history" class="muted"></p><div class="cards"><div class="card">Repeated active presentations<strong class="number" id="presentations"></strong><span class="legend">Counts repeated crops</span></div><div class="card">Unique supervised string-frames<strong class="number" id="unique"></strong><span class="legend">Recording + source frame + string</span></div><div class="card">Errors in the 1000+ bin<strong class="number" id="share"></strong><span class="legend">Share of all active-frame errors</span></div></div></section>
<section><h2>Every exposure bin, with its denominator</h2><p class="muted">Bins use repeated presentations of the ground-truth position in this model's own history. Error rate = incorrect / active GT frames. A dash means zero support.</p><div class="scroll"><table id="bins-table"><thead><tr><th>Presentations</th><th>Active GT frames</th><th>Incorrect</th><th>Error rate</th><th>Reference notes</th><th>Matched notes</th><th>Note miss rate</th></tr></thead><tbody></tbody></table></div><p class="legend">Note matches use saved pitch/onset decisions and ignore string. Notes begin in [0,10); frame scores mask the first 0.26s. Note values repeat between frame policies.</p></section>
<section><h2>Inspect string/fret positions</h2><label>GT exposure <select id="bin"><option value="all">All bins</option><option>unseen</option><option>1-99</option><option>100-999</option><option>1000+</option></select></label><label>String <select id="string"><option value="all">All strings</option><option value="0">0 - low E</option><option value="1">1 - A</option><option value="2">2 - D</option><option value="3">3 - G</option><option value="4">4 - B</option><option value="5">5 - high E</option></select></label><label><input id="all-positions" type="checkbox"> Include positions without validation support</label>
<p id="class-count" class="legend"></p><div class="scroll" id="class-scroll"><table id="classes-table"><thead><tr><th>String / fret / pitch</th><th>Available</th><th>Presentations</th><th>Unique</th><th>GT frames</th><th>Error rate</th><th>Notes</th><th>Pitch coverage</th></tr></thead><tbody></tbody></table></div></section>
<section><h2>False activity uses a separate denominator</h2><p id="false-activity" class="muted"></p><p class="legend">Grouped by the predicted position's exposure. This is a fraction of all inactive GT frames, not a per-bin GT fret error rate.</p><div class="scroll"><table id="false-table"><thead><tr><th>Predicted-position bin</th><th>False active string-frames</th><th>Fraction of all inactive GT</th></tr></thead><tbody></tbody></table></div></section>
<section><h2>Reference-note examples for the current filters</h2><p id="note-count" class="legend"></p><div class="scroll"><table id="notes-table"><thead><tr><th>Clip / source onset</th><th>String / fret</th><th>GT presentations</th><th>Official match</th><th>Same string in that match</th></tr></thead><tbody></tbody></table></div></section>
<section><h2>Evidence and exports</h2><div class="exports">__LINKS__<button id="download-notes">Download exact note JSON</button></div><p class="muted"><a href="../fret19/review.html">Original Stage 19 audio and notes</a> · <a href="report.md">Methods, results and limits</a></p></section>
<footer>Available labels, repeated presentations and unique identities measure different things. A frequent label may still have few independent examples. Coverage associations do not establish causation or independent electric-guitar generalization. No new processing clock was measured.</footer>
<script>const data=__DATA__, noteBytes='__NOTES__';const $=id=>document.getElementById(id),fmt=n=>n.toLocaleString('en-US'),pct=v=>v===null?'—':(100*v).toFixed(2)+'%';window.current=null;
function render(){const mode=$('model').value,policy=$('policy').value,model=data.summary.modes[mode],detail=model.policies[policy],history=data.coverage.histories[model.history],bin=$('bin').value,string=$('string').value,all=$('all-positions').checked;
 $('history').textContent='Training history: '+model.history.replaceAll('_','–')+'. Children share the same epochs 1–10 exposure; each model is evaluated against its own history.';
 $('presentations').textContent=fmt(history.per_record.reduce((a,r)=>a+r.presentations,0));$('unique').textContent=fmt(history.per_record.reduce((a,r)=>a+r.unique,0));$('share').textContent=pct(detail.bins['1000+'].share_of_all_active_errors);
 $('bins-table').tBodies[0].innerHTML=Object.entries(detail.bins).map(([b,r])=>`<tr data-bin="${b}"><td>${b}</td><td>${fmt(r.reference_active)}</td><td>${fmt(r.error_count)}</td><td>${pct(r.error_rate_on_reference_active)}${r.error_rate_on_reference_active===null?'':`<div class="bar" style="width:${100*r.error_rate_on_reference_active}%"></div>`}</td><td>${r.reference_notes}</td><td>${r.official_matched_notes}</td><td>${pct(r.note_miss_rate)}</td></tr>`).join('');
 const rows=detail.classes.filter(r=>(bin==='all'||r.bin===bin)&&(string==='all'||r.string===Number(string))&&(all||r.reference_active||r.reference_notes));window.current={mode,policy,detail,rows};$('class-count').textContent=rows.length+' positions shown. String 0 is low E; string 5 is high E.';
 $('classes-table').tBodies[0].innerHTML=rows.map(r=>`<tr><td>${r.string} / ${r.fret} / MIDI ${r.pitch}</td><td>${fmt(r.available)}</td><td>${fmt(r.presentations)}</td><td>${fmt(r.unique)}</td><td>${fmt(r.reference_active)}</td><td>${pct(r.error_rate_on_reference_active)}</td><td>${r.reference_notes}</td><td>${r.pitch_coverage.replaceAll('_',' ')}</td></tr>`).join('');
 const overall=detail.overall;$('false-activity').textContent=fmt(overall.false_active_on_reference_inactive)+' false active string-frames / '+fmt(overall.reference_inactive_frames)+' inactive GT string-frames ('+pct(overall.false_activity_rate_on_reference_inactive)+').';
 $('false-table').tBodies[0].innerHTML=Object.entries(detail.bins).map(([b,r])=>`<tr><td>${b}</td><td>${fmt(r.false_active_on_reference_inactive)}</td><td>${pct(r.false_activity_fraction_of_all_reference_inactive)}</td></tr>`).join('');
 const notes=data.notes.filter(r=>r.mode===mode&&(bin==='all'||r.bin===bin)&&(string==='all'||r.string===Number(string)));$('note-count').textContent=notes.length+' reference notes; showing the first 20. Download the complete note ledger below.';
 $('notes-table').tBodies[0].innerHTML=notes.slice(0,20).map(n=>`<tr><td><a href="../fret19/validation/results/${encodeURIComponent(mode)}/${encodeURIComponent(n.id)}-metrics.json">${n.id}</a> / ${n.start.toFixed(3)}s</td><td>${n.string} / ${n.fret}</td><td>${n.presentations}</td><td>${n.official_matched?'matched':'missed'}</td><td>${n.official_matched?(n.official_matched_same_string?'yes':'no'):'—'}</td></tr>`).join('');}
for(const id of ['model','policy','bin','string','all-positions'])$(id).addEventListener('change',render);
$('download-notes').addEventListener('click',()=>{const bytes=Uint8Array.from(atob(noteBytes),c=>c.charCodeAt(0)),url=URL.createObjectURL(new Blob([bytes],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='reference-notes.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});render();</script></html>'''


def report(output):
    output = Path(output)
    coverage, summary, notes = (read(output / name) for name in ('coverage.json', 'summary.json', 'reference-notes.json'))
    text = findings(coverage, summary)
    (output / 'report.md').write_text(text, encoding='utf-8')
    (output / 'findings.md').write_text(text, encoding='utf-8')
    export_names = ['summary.json', 'coverage.json', 'cases.json', 'reference-notes.json',
                    'coverage.csv', 'class-errors.csv', 'reference-notes.csv', 'input-bindings.json', 'protocol.json']
    links = ''.join(f'<a href="{quote(name, safe="/")}">{name}</a>' for name in export_names)
    payload = json.dumps({'summary': summary, 'coverage': coverage, 'notes': notes}, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    encoded = base64.b64encode((output / 'reference-notes.json').read_bytes()).decode('ascii')
    html = HTML.replace('__LINKS__', links).replace('__DATA__', payload).replace('__NOTES__', encoded)
    (output / 'review.html').write_text(html, encoding='utf-8')
    save_proposal = {
        'stage': 21, 'status': 'proposed_not_executed', 'kind': 'training_prior_confusion_diagnosis',
        'question': 'On GT-active frames, do saved fret choices improve on per-string active-fret frequency priors?',
        'source_root': 'outputs/coverage20', 'models': list(MODES),
        'new_optimizer_updates': 0, 'neural_inference_calls': 0, 'test_inference_allowed': False,
        'production_promotion': False, 'calibration_search_allowed': False,
        'scope': 'Use each history\'s training counts and saved state probabilities. Compare best-active fret accuracy, full supported-class confusions and top-fret concentration, including equal-label and frequency-weighted views. GT-active conditioning is diagnostic oracle support, not a deployable decoder. No prior baseline or model selection from development results.',
        'limitations': summary['limitations']}
    from .data import save
    save(output / 'next-experiment.json', save_proposal)
    print('Stage20 report saved:', output / 'review.html', flush=True)
