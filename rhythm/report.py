import html
import json
from pathlib import Path


def write_report(output, records, summary, run):
    output = Path(output)
    test = summary['test']; scenarios = summary['test_scenarios']
    lines = ['# Stage 6: rhythm, chords and editable score', '',
             'No new training: the official Beat This small0 checkpoint supplies beats/downbeats. '
             'The existing frozen clean CNN-GRU and selected constrained fingering supply note events.', '',
             '**Training overlap:** small0 was trained on GuitarSet, including this evaluation domain. '
             'These beat/downbeat results are integration diagnostics, not unseen-data generalization.', '',
             f"Model: {run['model_parameters']:,} parameters; {run['beat_model']['bytes']/1e6:.2f} MB checkpoint; "
             f"peak GPU allocated tensors {run['peak_gpu_allocated_bytes']/2**20:.1f} MiB. No DBN, AMP or model fine-tuning.", '',
             '## Rhythm integration', '',
             '| Split | Clips | Beat F1 | Downbeat F1 | Provisional meter duration agreement | Warm beat pipeline seconds |',
             '|---|---:|---:|---:|---:|---:|']
    for split in ['validation', 'test']:
        row = summary[split]
        lines.append(f"| {split} | {row['count']} | {row['beat_f1']:.3f} | {row['downbeat_f1']:.3f} | "
                     f"{row['provisional_meter_duration_accuracy']:.3f} | {row['warm_processing_seconds']:.3f} |")
    lines += ['', 'Events in [0,10 s) are matched one-to-one within 70 ms, with no phase/double/half-tempo allowance. '
              'Inference uses 12 s of context. First/last beats remain scored. Twelve-second inference is warmed once. '
              'Timing includes frontend, network and minimal postprocessing on a loaded waveform; excludes disk read, '
              'CNN-GRU, fingering and notation. This is offline throughput, not live latency.', '',
              'All 24 validation/test excerpts have 4/4 reference labels. Meter is only a pulse-group proposal; '
              'counting beats between downbeats does not establish the denominator or distinguish 6/8 from 3/4. '
              'Automatic irregular groups are retained for inspection, not certified meter changes. '
              'Manual meter, pulse length, beat/downbeat and straight/triplet grid corrections are supported. '
              'Tempo follows piecewise-linear beat intervals. Pickups and meter changes are covered by synthetic tests.', '',
              '## Oracle and predicted-input ablation', '',
              '| Notes / beats | Raw onset F1 | Quantized onset F1 | Mean onset movement (ms) | Chord accuracy, performed | Chord accuracy, instructed |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for scenario, row in scenarios.items():
        lines.append(f"| {scenario.replace('_',' ')} | {row['raw_pitch_onset_f1']:.3f} | {row['quantized_pitch_onset_f1']:.3f} | "
                     f"{row['mean_absolute_onset_shift_ms']:.1f} | {row['chord_performed_accuracy']:.3f} | {row['chord_instructed_accuracy']:.3f} |")
    lines += ['', 'The oracle-note/oracle-beat condition tests conversion of provided note intervals. It does not '
              'measure intended written rhythm: GuitarSet provides performed attacks/releases and a lead-sheet beat grid, '
              'not a ground-truth engraved score. Onset F1 after snapping is a timing diagnostic against the performed notes. '
              'Lower values can reflect removal of performance timing as well as wrong quantization.', '',
              'Every source note, pitch, string/fret and original onset/offset is retained in score JSON. '
              'Score tick intervals are separate derived fields; short notes receive a flagged minimum duration. '
              'Large shifts (>80 ms), crop-boundary releases and newly created string overlaps are flagged. '
              'Overlaps use separate MusicXML voices rather than silently shortening or dropping notes. '
              'Unresolved positions remain on the pitch staff and in JSON, with no invented tab number. '
              'All exported voices fill their bars with notes/rests; sustained events use ties across bars and duration fragments. '
              'A six-line tab staff uses MusicXML string 1 for high E; original JSON string 0 is low E.', '',
              'Quantization selects a sixteenth-note or eighth-triplet lattice per bar using squared onset movement '
              'and a prespecified 0.002-quarter-squared triplet penalty. There is no score-language model. '
              'Swing, arbitrary tuplets, enharmonic/key spelling, articulation and intended release duration remain unsupported. '
              'Both staves are in concert pitch, without guitar octave transposition.', '',
              '## Chord baseline and provenance', '',
              'Chord inference uses duration-weighted pitch classes over two tracker pulses, five qualities '
              '(major/minor/dominant7/major7/minor7), an explicit silence class and Viterbi change smoothing. '
              'It uses only supplied note events and beats, never reference chord roots/boundaries. '
              'It is a fixed template baseline, not a trained chord classifier. Local margins are not calibrated confidence. '
              'It estimates the guitar voicing; an isolated melodic solo may not specify full-song harmony.', '',
              'Duration accuracy scores the first 10 s after reducing roots and major/minor quality; inversions/extensions '
              'are ignored, unsupported qualities masked, and coverage is reported in JSON. Performed GuitarSet chords '
              'are partly derived from string-note transcriptions with lead-sheet roots and segmentation, then manually '
              'verified. They are not independent chord ground truth. Instructed and performed labels are reported separately.', '',
              '## Use and next decision', '',
              'Open `review.html` for synchronized audio/tabs, bar grids, exports and a correction-file editor. '
              'MusicXML can be imported into a score editor. Corrected beat/meter/chord JSON can be passed to '
              '`python -m rhythm score` to regenerate the score while retaining raw notes.', '',
              'Treat automatic notation as a review draft. Before expanding to full mixtures or live use, '
              'collect an independent rhythm test with compound/changing meters and musician-verified written rhythms. '
              'Improve chord inference only after inspecting failures in both oracle and predicted notes.', '',
              '## Sources', '',
              '- [Official Beat This code and model documentation](https://github.com/CPJKU/beat_this).',
              '- [Beat This GuitarSet training annotations](https://github.com/CPJKU/beat_this_annotations/tree/main/guitarset).',
              '- [GuitarSet authors](https://guitarset.weebly.com/).',
              '- [MusicXML 4.0 specification](https://www.w3.org/2021/06/musicxml40/).', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    payload = []
    for record in records:
        if record['split'] != 'test': continue
        doc = json.loads((output/record['score_file']).read_text())
        # Use existing twelve-second audio clips; no new remote resources in the viewer.
        payload.append({'id': record['id'], 'scenario': record['scenario'], 'document': doc,
                        'metrics': record, 'audio': '../fingering/evaluation/clips/'+record['id']+'.wav',
                        'stem': str(Path(record['score_file']).with_suffix('')).replace('\\','/')})
    # Present a chordal example first; all solos remain available in the selector.
    payload.sort(key=lambda item: (not item['id'].startswith('00_Rock'), item['id'], item['scenario']))
    template = VIEWER.replace('__DATA__', json.dumps(payload).replace('</', '<\\/'))
    (output/'review.html').write_text(template, encoding='utf-8')
    print('Report:', output/'report.md', '; viewer:', output/'review.html', flush=True)


VIEWER = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Guitar rhythm lab</title><style>
*{box-sizing:border-box}body{margin:0;background:#101924;color:#e4edf4;font:16px/1.5 system-ui}main{max-width:1180px;margin:auto;padding:28px}
h1{font-size:30px;margin:0}h2{font-size:21px}p{color:#bac9d6}section{background:#192737;padding:22px;margin:20px 0;border-radius:12px}
select,input,textarea,button{font:inherit;border:1px solid #50708a;border-radius:6px;background:#0f1b27;color:#e4edf4;padding:8px}
label{display:inline-block;margin:6px 12px 6px 0}select{max-width:100%}audio{width:100%;margin:18px 0}a{color:#77d7be;margin-right:18px}
#bars{display:flex;gap:10px;overflow:auto;padding:8px 0}.bar{min-width:160px;border:1px solid #506273;border-radius:7px;padding:10px;cursor:pointer}.active{border-color:#70e3c3;background:#245145}
svg{width:100%;min-width:650px;background:#101924;border-radius:6px}.scroll{overflow:auto}textarea{width:100%;height:100px}#status{color:#7ad8bd}
.warning{border-left:4px solid #dbad67;padding-left:14px;color:#efd3a9}.metrics{font-variant-numeric:tabular-nums}small{color:#b1c4d5}button{cursor:pointer;background:#25624f}input[type=number]{width:85px}
</style><main><h1>Guitar rhythm lab</h1><p>Stage 6 · audio, bars, chord drafts and editable score</p>
<p class="warning">Beat model training includes GuitarSet. These examples test integration. Meter and chord labels need musician review.</p>
<section><label>Excerpt <select id="clip"></select></label><label>Inputs <select id="scenario"></select></label>
<audio id="audio" controls></audio><div id="metrics" class="metrics"></div><div id="bars"></div>
<p><a id="xml">MusicXML</a><a id="json">Score JSON</a><a id="csv">Tab CSV</a><a id="raw">Raw note JSON for corrections</a></p>
<small>Click a bar or fret to seek. Tabs below follow original audio times; bar cards follow the proposed score grid. Amber frets carry review flags.</small>
<div class="scroll"><svg id="tab" viewBox="0 0 1080 230" aria-label="Timed guitar tablature"></svg></div></section>
<section><h2>Correct the grid</h2><p>Edit beat/downbeat seconds and meter, then download a correction file. Regenerate MusicXML with the command in the README. Existing note times remain available.</p>
<label>Numerator <input id="num" type="number" min="1" value="4"></label><label>Denominator <select id="den"><option>4</option><option>8</option><option>2</option><option>16</option></select></label>
<label>Quarters per detected pulse <input id="pulse" type="number" min="0.125" step="0.125" value="1"></label>
<label>Subdivision <select id="sub"><option value="auto">Automatic per bar</option><option value="4">Straight sixteenths</option><option value="3">Eighth triplets</option></select></label>
<p>For 6/8 tracked as two dotted-quarter pulses, enter numerator 6, denominator 8 and pulse length 1.5. The denominator is a notation choice.</p>
<label style="width:100%">Beat seconds<textarea id="beats"></textarea></label><label style="width:100%">Downbeat seconds<textarea id="downbeats"></textarea></label>
<label style="width:100%">Chord events (JSON; start/end seconds and label)<textarea id="chords"></textarea></label>
<button id="download">Download corrections</button><p id="status" role="status"></p></section>
<p><a href="report.md">Experiment report</a>All source notes are preserved. Quantized overlaps and large timing shifts are flagged; written articulation and swing are not inferred.</p></main>
<script>const data=__DATA__;const $=id=>document.getElementById(id);let current;
const clips=[...new Set(data.map(x=>x.id))];clips.forEach(id=>$('clip').add(new Option(id,id)));
const modes=[...new Set(data.map(x=>x.scenario))];modes.forEach(id=>$('scenario').add(new Option(id.replaceAll('_',' '),id)));
function load(){current=data.find(x=>x.id===$('clip').value&&x.scenario===$('scenario').value);const d=current.document,g=d.grid,m=current.metrics;
const source=current.audio;if($('audio').getAttribute('src')!==source){$('audio').src=source;}
['xml','json','csv'].forEach((id,i)=>{$(id).href=current.stem+['.musicxml','.json','.csv'][i];$(id).download='';});
$('raw').onclick=e=>{e.preventDefault();save({notes:d.notes.map(n=>n.raw)},current.id+'-raw.json');};$('raw').href='#';
$('metrics').textContent=`${d.notes.length} notes · raw pitch F1 ${m.raw_pitch_onset_f1.toFixed(3)} · quantized F1 ${m.quantized_pitch_onset_f1.toFixed(3)} · ${m.notation.large_shift_notes} large shifts · ${m.notation.quantized_string_overlap_notes} notes with written string overlaps`;
$('bars').replaceChildren();g.bars.forEach(b=>{let v=document.createElement('div');v.className='bar';v.dataset.start=b.start_seconds;v.dataset.end=b.end_seconds;
v.textContent=`Bar ${b.number}${b.pickup?' · pickup':''} · ${b.numerator}/${b.denominator} · ${b.subdivision===3?'triplet':'straight'} grid`;v.onclick=()=>{$('audio').currentTime=Math.max(0,b.start_seconds);};$('bars').append(v);});
$('num').value=g.bars.find(b=>!b.pickup)?.numerator||4;$('den').value=g.bars.find(b=>!b.pickup)?.denominator||4;$('pulse').value=g.pulse_quarters;
$('beats').value=g.beats.filter(t=>t<=g.duration_seconds).map(t=>t.toFixed(5)).join(', ');$('downbeats').value=g.downbeats.filter(t=>t<=g.duration_seconds).map(t=>t.toFixed(5)).join(', ');
$('chords').value=JSON.stringify(d.chords.map(({start,end,label})=>({start,end,label})),null,2);$('status').textContent='';
const svg=$('tab'),ns='http://www.w3.org/2000/svg';svg.replaceChildren();function el(name,attrs,text){let e=document.createElementNS(ns,name);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text!==undefined)e.textContent=text;svg.append(e);return e;}
for(let s=0;s<6;s++){let y=38+(5-s)*28;el('line',{x1:45,x2:1060,y1:y,y2:y,stroke:'#496074'});el('text',{x:12,y:y+5,fill:'#b9d5e6'},['E','A','D','G','B','e'][s]);}
g.beats.filter(t=>0<=t&&t<=g.duration_seconds).forEach(t=>el('line',{x1:45+t/g.duration_seconds*1000,x2:45+t/g.duration_seconds*1000,y1:22,y2:189,stroke:'#294538'}));
d.notes.forEach(n=>{const r=n.raw;if(r.string===null||r.string===undefined)return;let x=45+r.start/g.duration_seconds*1000,y=38+(5-r.string)*28;
el('line',{x1:x,x2:45+Math.min(r.end,g.duration_seconds)/g.duration_seconds*1000,y1:y-4,y2:y-4,stroke:n.flags.length?'#d5a666':'#70cbb1','stroke-width':3});
let text=el('text',{x,y:y+5,fill:n.flags.length?'#ffd294':'#e0fff2','font-size':13,'text-anchor':'middle',style:'cursor:pointer;paint-order:stroke;stroke:#101924;stroke-width:5px'},r.fret);let title=document.createElementNS(ns,'title');title.textContent=`Pitch ${r.pitch}, ${r.start.toFixed(3)}–${r.end.toFixed(3)} s. ${n.flags.join('; ')}`;text.append(title);text.onclick=()=>{$('audio').currentTime=r.start;};});
el('line',{id:'playhead',x1:45,x2:45,y1:15,y2:201,stroke:'#ec7680','stroke-width':2});el('text',{x:45,y:222,fill:'#b1c4d5'},'0 seconds');el('text',{x:990,y:222,fill:'#b1c4d5'},g.duration_seconds+' seconds');}
$('clip').onchange=load;$('scenario').onchange=load;
$('audio').ontimeupdate=()=>{const t=$('audio').currentTime;document.querySelectorAll('.bar').forEach(b=>b.classList.toggle('active',t>=+b.dataset.start&&t<+b.dataset.end));let p=$('playhead');if(p){let x=45+t/current.document.grid.duration_seconds*1000;p.setAttribute('x1',x);p.setAttribute('x2',x);}};
function save(value,name){const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),30000);}
$('download').onclick=()=>{try{const parse=id=>$(id).value.split(/[\s,]+/).filter(Boolean).map(Number);let beats=parse('beats'),downbeats=parse('downbeats');if(beats.length<2||beats.some((v,i)=>!Number.isFinite(v)||(i&&v<=beats[i-1])))throw Error('Beats must be finite and strictly increasing.');if(downbeats.some((v,i)=>!Number.isFinite(v)||(i&&v<=downbeats[i-1])))throw Error('Downbeats must be finite and strictly increasing.');if(!Number.isInteger(+$('num').value)||+$('num').value<=0||!Number.isFinite(+$('pulse').value)||+$('pulse').value<=0)throw Error('Check meter and pulse length.');let chords=JSON.parse($('chords').value);
if(!Array.isArray(chords)||chords.some(c=>!Number.isFinite(c.start)||!Number.isFinite(c.end)||c.end<=c.start||!(/^(N|[A-G]#?:(maj|min|7|maj7|min7))$/).test(c.label)))throw Error('Check chord intervals and labels.');
let result={beats,downbeats,duration_seconds:current.document.grid.duration_seconds,meter:{numerator:+$('num').value,denominator:+$('den').value,pulse_quarters:+$('pulse').value},chords};if($('sub').value!=='auto')result.subdivision=+$('sub').value;
save(result,current.id+'-corrections.json');$('status').textContent='Correction file downloaded. Use the raw note JSON link and rhythm score command to regenerate exports.';
}catch(e){$('status').textContent=e.message;}};load();</script></html>'''
