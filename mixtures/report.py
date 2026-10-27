"""Portable report for the frozen separator/transcriber mixture diagnostic."""

import csv
import html
import json
import math
from pathlib import Path
from urllib.parse import quote

from baseline.report import roll_svg


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _fmt(value, signed=False):
    value = _number(value)
    return '—' if value is None else format(value, '+.3f' if signed else '.3f')


def _pitch(record):
    value = record.get('pitch_metrics')
    return value if isinstance(value, dict) else record


def _score(record, metric='onset'):
    value = _pitch(record).get(metric, {})
    return _number(value.get('f1')) if isinstance(value, dict) else None


def _conditions(summary, records):
    supplied = summary.get('conditions', summary)
    output = {key: dict(value) for key, value in supplied.items() if isinstance(value, dict)}
    for condition in dict.fromkeys(record['condition'] for record in records):
        rows = [record for record in records if record['condition'] == condition]
        value = output.setdefault(condition, {})
        value.setdefault('count', len(rows))
        for metric, field in [('onset', 'onset_f1'), ('onset_offset', 'onset_offset_f1'), ('frame', 'frame_f1')]:
            numbers = [_score(record, metric) for record in rows]
            numbers = [number for number in numbers if number is not None]
            value.setdefault(field, sum(numbers) / len(numbers) if numbers else None)
        tab = []
        for record in rows:
            metrics = record.get('tab_metrics', {})
            candidate = metrics.get('onset', metrics.get('string_onset', {}))
            number = _number(candidate.get('f1')) if isinstance(candidate, dict) else None
            if number is None:
                number = _number(metrics.get('onset_f1'))
            if number is not None:
                tab.append(number)
        value.setdefault('tab_onset_f1', sum(tab) / len(tab) if tab else None)
        for stage in ['separation', 'transcription']:
            value.setdefault('total_' + stage + '_seconds', sum(
                _number(record.get('timing', {}).get(stage + '_seconds')) or 0 for record in rows))
        sdr = [_number((record.get('separation_metrics') or {}).get('si_sdr_db')) for record in rows]
        sdr = [number for number in sdr if number is not None]
        value.setdefault('mean_si_sdr_db', sum(sdr) / len(sdr) if sdr else None)
    return output


def _paired(records, before, after):
    left = {record['id']: record for record in records if record['condition'] == before}
    right = {record['id']: record for record in records if record['condition'] == after}
    deltas = [_score(right[key]) - _score(left[key]) for key in sorted(left.keys() & right.keys())
              if _score(right[key]) is not None and _score(left[key]) is not None]
    return (sum(deltas) / len(deltas) if deltas else None), len(deltas)


def _url(path):
    return quote(str(path).replace('\\', '/'), safe='/')


def _table(headers, rows):
    return '<div class="table-scroll"><table><thead><tr>' + ''.join(
        '<th>' + html.escape(str(value)) + '</th>' for value in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + html.escape(str(value)) + '</td>' for value in row) + '</tr>'
        for row in rows) + '</tbody></table></div>'


def write_report(output, records, summary, run):
    """Write report.md, per-excerpt CSV, SVG plots and an offline review.html."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plots = output / 'plots'
    plots.mkdir(exist_ok=True)
    conditions = _conditions(summary, records)
    headers = ['Target/accompaniment', 'Target gain control', 'Mixture', 'Separated guitar',
               'Mixture − target', 'Separated − mixture', 'Paired clips', 'Separated SI-SDR (dB)']
    primary = []
    paired_changes = []
    for ratio in [6, 0, -6]:
        suffix = f'{ratio:+d}' if ratio else '0'
        names = ['target_' + suffix, 'mixture_' + suffix, 'separated_' + suffix]
        if not any(name in conditions for name in names):
            continue
        target, mixture, separated = [conditions.get(name, {}) for name in names]
        mixing, mixing_count = _paired(records, names[0], names[1])
        separation, separation_count = _paired(records, names[1], names[2])
        primary.append([f'{ratio:+d} dB', _fmt(target.get('onset_f1')), _fmt(mixture.get('onset_f1')),
                        _fmt(separated.get('onset_f1')), _fmt(mixing, True), _fmt(separation, True),
                        f'{mixing_count}/{separation_count}', _fmt(separated.get('mean_si_sdr_db'))])
        if separation is not None:
            paired_changes.append(separation)
    artifact_delta, artifact_count = _paired(records, 'clean', 'separated_clean')
    clean = conditions.get('clean', {})
    separated_clean = conditions.get('separated_clean', {})
    controls = [['Original clean guitar', _fmt(clean.get('onset_f1')), clean.get('count', 0)],
                ['Clean guitar passed through separator', _fmt(separated_clean.get('onset_f1')),
                 separated_clean.get('count', 0)],
                ['Separator-only paired change', _fmt(artifact_delta, True), artifact_count]]
    false_rows = [record for record in records if record['condition'] == 'accompaniment_only']
    false_counts = [_pitch(record).get('onset', {}).get('predicted_count') for record in false_rows]
    false_counts = [int(number) for number in false_counts if number is not None]
    false_total = sum(false_counts)
    false_mean = false_total / len(false_counts) if false_counts else None
    change = sum(paired_changes) / len(paired_changes) if paired_changes else None
    all_rows = []
    for condition, value in conditions.items():
        empty = condition == 'accompaniment_only'
        all_rows.append([condition, value.get('count', 0),
                         'n/a' if empty else _fmt(value.get('onset_f1')),
                         'n/a' if empty else _fmt(value.get('onset_offset_f1')),
                         'n/a' if empty else _fmt(value.get('frame_f1')),
                         'n/a' if empty else _fmt(value.get('tab_onset_f1')),
                         _fmt(value.get('total_separation_seconds')),
                         _fmt(value.get('total_transcription_seconds')),
                         _fmt(value.get('mean_si_sdr_db'))])
    all_headers = ['Condition', 'Clips', 'Onset F1', 'Onset+offset F1', 'Frame F1', 'String onset F1',
                   'Separator seconds', 'Transcriber seconds', 'SI-SDR (dB)']
    config = run.get('config', {})
    evaluation = config.get('evaluation', config)
    score_seconds = evaluation.get('score_seconds', config.get('score_seconds', 10))
    clip_seconds = evaluation.get('clip_seconds', evaluation.get('input_seconds',
                         config.get('clip_seconds', config.get('input_seconds', 12))))
    lines = ['# Stage 7: guitar transcription in controlled mixtures', '',
             'The existing frozen CNN-GRU transcribes acoustic guitar, procedurally mixed accompaniment, '
             'and the guitar output of the official frozen `htdemucs_6s` separator. No model training or '
             'threshold tuning is performed in this stage.', '',
             '## Paired note onset results', '', '| ' + ' | '.join(headers) + ' |',
             '| ' + ' | '.join(['---'] + ['---:'] * (len(headers) - 1)) + ' |']
    lines.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in primary)
    lines += ['', 'F1 values are macro means over completed excerpts. Differences use matching excerpt IDs '
              'within each condition pair; the paired-clips column lists mixing/separation pair counts. '
              'The target gain control uses the same guitar component and gain as its corresponding mixture, '
              'so the comparison separates interference from input-level changes.', '',
              'This stage scores raw JSON note seconds before MIDI serialization. Clean F1 is 0.525 versus '
              '0.522 in the earlier MIDI-round-trip report. The original cached acoustic probabilities decode '
              'to the same raw note events on the inspected discrepancy; small tick rounding shifts can cross '
              'the strict 50 ms matching boundary. All stage-7 conditions use the same raw-event protocol.', '',
              f'Mean of the three ratio-level paired separator changes: **{_fmt(change, True)} onset F1**. '
              'This small controlled result is a diagnostic; it is not a selection benchmark for multiple separators.', '',
              '## Separator artifact control', '', '| Condition | Onset F1 / paired change | Clips |',
              '| --- | ---: | ---: |']
    lines.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in controls)
    lines += ['', 'Passing clean guitar through the separator can change attacks, sustains and harmonics even '
              'when no accompaniment needs removal. Read this control alongside the mixture gains/losses.', '',
              '## Accompaniment-only false positives', '',
              f'No target guitar is present in {len(false_rows)} accompaniment-only inputs. '
              f'The transcriber emits **{false_total} scored-prefix notes**, '
              f'**{_fmt(false_mean)} per completed excerpt**. F1 is not informative when the target reference '
              'contains zero notes; these detections count as guitar false positives.', '',
              '## All conditions', '', '| ' + ' | '.join(all_headers) + ' |',
              '| ' + ' | '.join(['---'] + ['---:'] * (len(all_headers) - 1)) + ' |']
    lines.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in all_rows)
    lines += ['', 'Times are measured offline processing scopes recorded in run.json. They are not microphone-to-tab '
              'latency. Model loading, file handling and review/export may fall outside these scopes; '
              'GPU peak allocations are model tensor allocations, not total system/video-memory usage.', '',
              '## Protocol and controls', '',
              f'Audio input uses the first {clip_seconds} seconds and note/frame scoring uses the first '
              f'{score_seconds} seconds, retaining later audio for release context. Guitar pitches, strings '
              'and note intervals remain unchanged in the reference labels. The same frozen transcription '
              'checkpoint, frontend, decoder thresholds and note-matching rules are used in every condition.',
              'Onset matching uses a 50 ms tolerance and 50-cent pitch tolerance. Offset matching uses '
              'max(50 ms, 20% of reference duration). Right-censored notes follow the existing baseline protocol.',
              'The accompaniment contains procedural bass, keyboard and drums. It is synthesized using '
              'lead-sheet rhythm/harmony annotations to make musically coherent interference; those labels '
              'are supplied only to the renderer and never to the separator or transcriber. This coherence '
              'is a construction confound, and the accompaniment is less diverse than real recordings.',
              'SI-SDR and SDR compare separated audio with the known scaled clean target. SI-SDR permits '
              'one fitted scalar gain and is sensitive to distortion; ordinary SDR also penalizes gain errors. '
              'Normalized correlation, gain projection and diagnostic lag are retained per excerpt. '
              'Diagnostic lag is reported without shifting note predictions or reference labels.', '',
              '## Model provenance and limits', '',
              '[Official Demucs v4.0.1 code and model documentation](https://github.com/facebookresearch/demucs/tree/v4.0.1) '
              'documents `htdemucs_6s` with drums, bass, vocals, other, guitar and piano outputs. The authors '
              'describe the added guitar output as weaker than the main stems. The guitar stem can lose '
              'target notes or retain piano/bass harmonics; it is not guaranteed to be an isolated guitar performance.',
              'The original small-budget CNN-GRU is frozen. Previous stages showed a gap on independent '
              'electric-guitar recordings; this synthetic acoustic-guitar experiment does not remove that gap. '
              'Demucs training overlap with these source recordings is not independently audited.',
              'The recordings are short GuitarSet excerpts. Procedural mixtures do not establish full-song, '
              'commercial-master, arbitrary-instrument, time-signature or live-stream performance. '
              'No new training uses these test mixtures.', '',
              f'Device: `{run.get("device", "not recorded")}`. '
              f'Peak separator GPU tensor allocation: {_fmt(run.get("peak_separator_gpu_mib"))} MiB; '
              f'peak acoustic model allocation: {_fmt(run.get("peak_acoustic_gpu_mib"))} MiB. '
              f'Optimizer updates: {run.get("optimizer_updates", 0)}.',
              f'Acoustic checkpoint SHA256: `{run.get("acoustic_checkpoint_sha256", "not recorded")}`.',
              'Pinned code/checkpoint identifiers, source-file hashes, renderer settings, software versions '
              'and measured timing scopes are retained in `run.json` and the data manifest.', '',
              '## Review and next decision', '',
              'Open `review.html` to listen to the gain control, mixture and separated output for the same '
              'excerpt, compare the synchronized note plot, and download note JSON, MIDI and raw string/fret CSV.',
              'Inspect paired failures before adopting separation as the default path. A useful next experiment '
              'is independent, musician-checked real mixtures with isolated guitar stems and disjoint songs; '
              'keep the clean/no-guitar controls and the frozen transcription protocol.', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'condition', 'ratio_db', 'onset_f1', 'onset_offset_f1', 'frame_f1',
                         'predicted_count', 'missed_notes', 'extra_notes', 'si_sdr_db', 'sdr_db',
                         'separation_seconds', 'transcription_seconds'])
        for record in records:
            metrics, timing = _pitch(record), record.get('timing', {})
            separation = record.get('separation_metrics') or {}
            errors = metrics.get('errors', {})
            writer.writerow([record['id'], record['condition'], record.get('ratio_db'),
                             _score(record), _score(record, 'onset_offset'), _score(record, 'frame'),
                             metrics.get('onset', {}).get('predicted_count'), errors.get('missed_notes'),
                             errors.get('extra_notes'), separation.get('si_sdr_db'), separation.get('sdr_db'),
                             timing.get('separation_seconds'), timing.get('transcription_seconds')])
    payload = []
    for record in records:
        metrics = _pitch(record)
        condition, clip_id = record['condition'], record['id']
        reference = metrics.get('reference', record.get('reference', []))
        notes = metrics.get('predicted', record.get('notes', []))
        duration = _number(metrics.get('score_duration')) or _number(score_seconds) or 10
        svg = roll_svg(reference, notes, metrics.get('onset_matches', []), duration,
                       clip_id + ' · ' + condition)
        plot_name = _url(condition + '__' + clip_id) + '.svg'
        (plots / plot_name).write_text(svg, encoding='utf-8')
        stem = _url('results/' + condition + '/' + clip_id)
        payload.append({'id': clip_id, 'condition': condition, 'ratio': record.get('ratio_db'),
                        'audio': _url(record.get('audio', 'audio/' + condition + '/' + clip_id + '.wav')),
                        'svg': svg, 'duration': duration, 'stem': stem,
                        'onset': _score(record), 'offset': _score(record, 'onset_offset'),
                        'predictedCount': metrics.get('onset', {}).get('predicted_count', len(notes)),
                        'errors': metrics.get('errors', {}),
                        'separation': record.get('separation_metrics'), 'timing': record.get('timing', {}),
                        'noteJson': (output / 'results' / condition / (clip_id + '.json')).read_text(encoding='utf-8')})
    payload.sort(key=lambda value: (not value['id'].startswith('00_Rock'), value['id']))
    introduction = '<p class="result">Average paired separator change: <strong>' + _fmt(change, True) + \
        ' onset F1</strong>. Clean-guitar separator change: <strong>' + _fmt(artifact_delta, True) + '</strong>.</p>'
    page = VIEWER.replace('__PRIMARY__', _table(headers, primary)).replace('__INTRO__', introduction)
    page = page.replace('__ALL__', _table(all_headers, all_rows))
    page = page.replace('__FALSE__', html.escape(f'{false_total} false-positive notes across {len(false_counts)} '
                                               f'completed accompaniment-only excerpts; {_fmt(false_mean)} per excerpt.'))
    page = page.replace('__DATA__', json.dumps(payload, ensure_ascii=True, allow_nan=False).replace('</', '<\\/'))
    (output / 'review.html').write_text(page, encoding='utf-8')
    print('Report:', output / 'report.md', '; viewer:', output / 'review.html', flush=True)


VIEWER = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Guitar mixture lab</title><style>
*{box-sizing:border-box}body{margin:0;background:#101924;color:#e4edf4;font:16px/1.5 system-ui}main{max-width:1180px;margin:auto;padding:28px}
h1{font-size:30px;margin:0}h2{font-size:21px}p{color:#bac9d6}section{background:#192737;padding:22px;margin:20px 0;border-radius:12px}
select,button{font:inherit;border:1px solid #50708a;border-radius:6px;background:#0f1b27;color:#e4edf4;padding:8px}label{display:inline-block;margin:6px 12px 6px 0}
select{max-width:100%}audio{width:100%;margin:18px 0}a{color:#77d7be;margin-right:18px}.scroll,.table-scroll{overflow:auto}svg{width:100%;min-width:650px;display:block}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:9px 11px;text-align:right;border-bottom:1px solid #425267;white-space:nowrap}th:first-child,td:first-child{text-align:left}
.warning{border-left:4px solid #dbad67;padding-left:14px;color:#efd3a9}.metrics{font-variant-numeric:tabular-nums}small{color:#b1c4d5}#status{color:#efb37f}#roll{cursor:crosshair}
.result{padding:14px;background:#153b35;border-radius:8px}button{cursor:pointer;margin-right:8px}.tip{font-size:14px}details summary{cursor:pointer}
</style><main><h1>Guitar mixture lab</h1><p>Stage 7 · frozen separation and transcription · no new training</p>
<p class="warning">These are synthetic mixtures of short acoustic-guitar excerpts. Six-source Demucs can remove guitar details and retain accompaniment. Scores do not establish real-song or live performance.</p>
__INTRO__<section><h2>Does separation help?</h2><p>Note onset F1, from 0 to 1. The gain control matches the guitar level in the mixture. Paired differences compare the same excerpts.</p>__PRIMARY__</section>
<section><h2>Listen and inspect</h2><label>Excerpt <select id="clip"></select></label><label>Condition <select id="condition"></select></label>
<div><button id="compare">Compare with gain control</button><button id="separate">Compare mixture / separated</button></div>
<audio id="audio" controls preload="metadata"></audio><p id="status" role="status"></p><div id="metrics" class="metrics"></div>
<p><a id="json">Note JSON</a><a id="midi">MIDI</a><a id="csv">Raw string/fret CSV</a></p><div class="scroll" id="roll"></div>
<p class="tip">Click the note plot to seek. The plot covers the scored prefix; audio includes release context. Upper bars are reference notes (blue matched, purple missed); lower bars are predictions (green matched, red extra). Hover for pitch and timing. Colors use onset matching.</p></section>
<section><h2>No-guitar control</h2><p>__FALSE__ F1 is not meaningful with an empty target reference. Select accompaniment only to hear what caused these detections.</p></section>
<section><details><summary>All conditions and processing times</summary>__ALL__<p class="tip">Times are offline processing scopes, not end-to-end live latency. SI-SDR uses the known scaled target.</p></details></section>
<p>Accompaniment is synthesized from lead-sheet annotations. These labels guide only the renderer; they are never model inputs. Original guitar note labels remain unchanged.</p>
<p><a href="report.md">Protocol and limitations</a><a href="metrics.csv">Per-excerpt metrics</a><a href="run.json">Run provenance</a></p></main>
<script>const data=__DATA__;const $=id=>document.getElementById(id);let current;const labels={clean:'Clean guitar',separated_clean:'Clean guitar → separator',accompaniment_only:'Accompaniment only (no guitar)'};
function label(name){if(labels[name])return labels[name];return name.replace(/^target_/,'Target gain control · ').replace(/^mixture_/,'Mixture · ').replace(/^separated_/,'Separated guitar · ')+' dB';}
const clips=[...new Set(data.map(x=>x.id))];clips.forEach(id=>$('clip').add(new Option(id,id)));const fmt=v=>v===null||v===undefined?'—':Number(v).toFixed(3);
function conditionMenu(){const previous=$('condition').value;const choices=data.filter(x=>x.id===$('clip').value);$('condition').replaceChildren();choices.forEach(x=>$('condition').add(new Option(label(x.condition),x.condition)));if(choices.some(x=>x.condition===previous))$('condition').value=previous;load();}
function load(){const previous=current;current=data.find(x=>x.id===$('clip').value&&x.condition===$('condition').value);if(!current)return;const time=previous&&previous.id===current.id?$('audio').currentTime:0;
$('audio').pause();$('audio').src=current.audio;$('audio').onloadedmetadata=()=>{$('audio').currentTime=Math.min(time,$('audio').duration||time);update();};$('status').textContent='';
const empty=current.condition==='accompaniment_only';$('metrics').textContent=empty?`${current.predictedCount} false-positive notes in the scored prefix (no target guitar).`:`Onset F1 ${fmt(current.onset)} · onset+offset F1 ${fmt(current.offset)} · missed ${current.errors.missed_notes??'—'} · extra ${current.errors.extra_notes??'—'}`;
if(current.separation)$('metrics').textContent+=` · SI-SDR ${fmt(current.separation.si_sdr_db)} dB`;
['json','midi','csv'].forEach((id,i)=>{$(id).href=current.stem+['.json','.mid','.csv'][i];$(id).download='';});$('roll').innerHTML=current.svg;
$('json').onclick=event=>{event.preventDefault();const url=URL.createObjectURL(new Blob([current.noteJson],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=current.id+'-'+current.condition+'.json';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),30000);};
const svg=$('roll').querySelector('svg'),line=document.createElementNS('http://www.w3.org/2000/svg','line');line.id='playhead';line.setAttribute('y1','15');line.setAttribute('y2','215');line.setAttribute('stroke','#fff4b5');line.setAttribute('stroke-width','2');svg.append(line);update();
$('compare').disabled=empty||current.condition==='clean'||current.condition==='separated_clean';$('separate').disabled=empty||current.condition==='clean'||current.condition==='separated_clean';}
function update(){if(!current)return;const line=$('playhead');if(line){const x=45+Math.min(Math.max($('audio').currentTime,0),current.duration)/current.duration*945;line.setAttribute('x1',x);line.setAttribute('x2',x);}}
function selectCondition(condition){if(data.some(x=>x.id===current.id&&x.condition===condition)){$('condition').value=condition;load();}}
$('clip').onchange=conditionMenu;$('condition').onchange=load;$('audio').ontimeupdate=update;$('audio').onerror=()=>{$('status').textContent='The local audio file could not be loaded. Keep review.html beside its audio and results folders.';};
$('compare').onclick=()=>{const suffix=current.condition.replace(/^(target|mixture|separated)_/,'');selectCondition(current.condition.startsWith('target_')?'mixture_'+suffix:'target_'+suffix);};
$('separate').onclick=()=>{const suffix=current.condition.replace(/^(target|mixture|separated)_/,'');selectCondition(current.condition.startsWith('separated_')?'mixture_'+suffix:'separated_'+suffix);};
$('roll').onclick=event=>{const svg=$('roll').querySelector('svg');if(!svg||!current)return;const bounds=svg.getBoundingClientRect(),x=(event.clientX-bounds.left)/bounds.width*1000;$('audio').currentTime=Math.min(current.duration,Math.max(0,(x-45)/945*current.duration));update();};
conditionMenu();</script></html>'''
