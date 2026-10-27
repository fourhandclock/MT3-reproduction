"""Portable accuracy/latency review for the stage-8 causal replay experiment."""

import csv
import html
import json
import math
from pathlib import Path
from urllib.parse import quote

from baseline.report import roll_svg


LABELS = {
    'causal_20ms': 'Causal · 20 ms packets',
    'causal_40ms': 'Causal · 40 ms packets',
    'causal_80ms': 'Causal · 80 ms packets',
    'offline_bidir': 'Offline CNN-GRU',
}


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def _fmt(value, places=3):
    number = _number(value)
    return '—' if number is None else f'{number:.{places}f}'


def _url(path):
    return quote(str(path).replace('\\', '/'), safe='/')


def _clean(value):
    """JSON used in a script must not contain nonfinite Python float literals."""
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _score(record, name='onset'):
    value = record.get('pitch_metrics', {}).get(name, {})
    return _number(value.get('f1')) if isinstance(value, dict) else None


def _tab_score(record):
    value = record.get('tab_metrics', {})
    return _number(value.get('onset', {}).get('f1', value.get('onset_f1')))


def _mean(values):
    values = [number for number in map(_number, values) if number is not None]
    return sum(values) / len(values) if values else None


def _modes(records, summary):
    output = {name: dict(value) for name, value in summary.get('modes', {}).items()
              if isinstance(value, dict)}
    for mode in dict.fromkeys(record['mode'] for record in records):
        rows = [record for record in records if record['mode'] == mode]
        value = output.setdefault(mode, {})
        value.setdefault('count', len(rows))
        for metric in ['onset', 'onset_offset', 'frame']:
            value.setdefault(metric + '_f1', _mean(_score(row, metric) for row in rows))
        value.setdefault('tab_onset_f1', _mean(_tab_score(row) for row in rows))
        processing = sum(_number(row.get('timing', {}).get('processing_seconds')) or 0 for row in rows)
        audio = sum(_number(row.get('timing', {}).get('input_seconds')) or 0 for row in rows)
        value.setdefault('real_time_factor', processing / audio if audio else None)
        value.setdefault('matched_onset_count', sum(row.get('latency', {}).get('matched_onset_count', 0) or 0
                                                   for row in rows))
        value.setdefault('matched_offset_count', sum(row.get('latency', {}).get('matched_offset_count', 0) or 0
                                                    for row in rows))
        # Do not invent pooled latency quantiles by averaging per-clip quantiles.
        for edge in ['onset', 'offset']:
            for quantile in ['p50', 'p95']:
                value.setdefault(f'matched_{edge}_delivery_{quantile}_ms', None)
    return output


def _table(headers, rows):
    return '<div class="table-scroll"><table><thead><tr>' + ''.join(
        '<th>' + html.escape(str(value)) + '</th>' for value in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + html.escape(str(value)) + '</td>' for value in row) + '</tr>'
        for row in rows) + '</tbody></table></div>'


def _events(record):
    output = []
    for event in record.get('events', []):
        note = event.get('note', {}) if isinstance(event.get('note'), dict) else {}
        kind = str(event.get('type', event.get('kind', event.get('event', 'event'))))
        onset = kind in ('onset', 'note_on', 'on', 'start')
        source = next((_number(event.get(key)) for key in
                       ['source_time', 'source_seconds', 'source_timestamp', 'time', 'timestamp']
                       if _number(event.get(key)) is not None), None)
        if source is None:
            source = _number(event.get('start' if onset else 'end', note.get('start' if onset else 'end')))
        delivery = next((_number(event.get(key)) for key in
                         ['delivered_at_seconds', 'delivery_time', 'delivery_seconds', 'delivered_at',
                          'delivered_seconds', 'delivery_timestamp', 'emitted_at_audio_seconds',
                          'emission_time', 'emitted_at']
                         if _number(event.get(key)) is not None), None)
        output.append({'type': 'onset' if onset else 'offset' if kind in
                       ('offset', 'note_off', 'off', 'end') else kind,
                       'source': source, 'delivery': delivery,
                       'pitch': event.get('pitch', note.get('pitch')),
                       'string': event.get('string', note.get('string')),
                       'fret': event.get('fret', note.get('fret')),
                       'flush': bool(event.get('flush', event.get('censored_at_end', event.get('censored', False)))),
                       'note_id': event.get('note_id', event.get('id'))})
    return output


def _event_svg(events, duration, name):
    """Connect a backdated source timestamp to the virtual delivery timestamp."""
    width, height, left, right = 1000, 295, 60, 15
    xmax = max([duration] + [event['delivery'] for event in events if event['delivery'] is not None])
    xmax = max(1, xmax)
    x = lambda seconds: left + seconds / xmax * (width - left - right)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="{html.escape(name)} source and delivery event timeline" data-duration="{xmax}">',
             '<rect width="1000" height="295" fill="#101c2d"/>']
    for second in range(math.ceil(xmax) + 1):
        parts.append(f'<path d="M{x(second):.2f},20 V260" stroke="#263448"/>')
        parts.append(f'<text x="{x(second):.2f}" y="282" fill="#9aaec9" font-size="12">{second}s</text>')
    for string in range(6):
        y = 35 + string * 40
        parts.append(f'<path d="M60,{y} H985" stroke="#40516b"/>')
        parts.append(f'<text x="4" y="{y+4}" fill="#a8bad0" font-size="12">S{string}</text>')
    valid = 0
    for event in events:
        if event['source'] is None or event['delivery'] is None:
            continue
        valid += 1
        string = _number(event['string'])
        y = 35 + (min(5, max(0, int(string))) if string is not None else 5) * 40
        y += -7 if event['type'] == 'onset' else 7
        start, delivered = x(max(0, event['source'])), x(max(0, event['delivery']))
        color = '#61dfb0' if event['type'] == 'onset' else '#f4b473'
        delay = 1000 * (event['delivery'] - event['source'])
        label = (f'{event["type"]}: MIDI {event["pitch"]}, string {event["string"]}, fret {event["fret"]}; '
                 f'source {event["source"]:.3f}s, delivered {event["delivery"]:.3f}s, delay {delay:.1f}ms'
                 + ('; end-of-input flush' if event['flush'] else ''))
        parts.append(f'<g><title>{html.escape(label)}</title>'
                     f'<path d="M{start:.2f},{y} H{delivered:.2f}" stroke="{color}" stroke-opacity=".7"/>'
                     f'<circle cx="{start:.2f}" cy="{y}" r="3" stroke="{color}" fill="#101c2d"/>')
        if event['type'] == 'onset':
            parts.append(f'<circle cx="{delivered:.2f}" cy="{y}" r="3.5" fill="{color}"/>')
        else:
            parts.append(f'<path d="M{delivered:.2f},{y-4} l4,4 -4,4 -4,-4 Z" fill="{color}"/>')
        parts.append('</g>')
    if not valid:
        parts.append('<text x="90" y="140" fill="#b9c8d8" font-size="16">'
                     'No incremental delivery events for this mode.</text>')
    parts.append('</svg>')
    return ''.join(parts)


def write_report(output, records, summary, run):
    """Write Markdown, per-excerpt metrics, note/event SVGs and a local HTML viewer."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plots = output / 'evaluation' / 'plots'
    plots.mkdir(parents=True, exist_ok=True)
    modes = _modes(records, summary)
    ordered = [mode for mode in LABELS if mode in modes] + [mode for mode in modes if mode not in LABELS]
    score_headers = ['Mode', 'Clips', 'Onset F1', 'Offset F1', 'String F1', 'Processing/audio']
    score_rows = [[LABELS.get(mode, mode), modes[mode].get('count', 0),
                   _fmt(modes[mode].get('onset_f1')), _fmt(modes[mode].get('onset_offset_f1')),
                   _fmt(modes[mode].get('tab_onset_f1')), _fmt(modes[mode].get('real_time_factor'))]
                  for mode in ordered]
    latency_headers = ['Packet size', 'Matched onsets', 'Onset p50 / p95 ms',
                       'Matched releases', 'Release p50 / p95 ms']
    latency_rows = []
    for mode in ordered:
        if mode == 'offline_bidir':
            continue
        value = modes[mode]
        latency_rows.append([LABELS.get(mode, mode).replace('Causal · ', ''),
                             value.get('matched_onset_count', '—'),
                             _fmt(value.get('matched_onset_delivery_p50_ms'), 1) + ' / ' +
                             _fmt(value.get('matched_onset_delivery_p95_ms'), 1),
                             value.get('matched_offset_count', '—'),
                             _fmt(value.get('matched_offset_delivery_p50_ms'), 1) + ' / ' +
                             _fmt(value.get('matched_offset_delivery_p95_ms'), 1)])
    config = run.get('config', {})
    evaluation = config.get('evaluation', {})
    score_seconds = _number(evaluation.get('score_seconds', config.get('score_seconds'))) or 10
    input_seconds = _number(evaluation.get('clip_seconds', config.get('clip_seconds'))) or 12
    train = run.get('training_summary', {})
    epochs = run.get('new_training_epochs', train.get('epochs_completed', train.get('epochs', 'recorded in run.json')))
    updates = run.get('optimizer_updates', train.get('optimizer_updates', 'recorded in run.json'))
    params = _number(run.get('model_parameters'))
    lines = ['# Stage 8: causal guitar transcription prototype', '',
             'A new small CNN-GRU processes isolated guitar PCM in incoming packets, keeps recurrent state, '
             'and emits note starts/releases as audio arrives. The existing bidirectional CNN-GRU remains '
             'a frozen offline quality reference.', '',
             '**This is a measured file replay, not a microphone demo.** The input is already 16 kHz PCM. '
             'Packet arrivals and a virtual processing clock expose when each event would become available. '
             'Audio capture, a causal sample-rate converter, drivers, UI scheduling, source separation and '
             'rhythm/score inference are outside this experiment.', '',
             '## Accuracy and processing', '', '| ' + ' | '.join(score_headers) + ' |',
             '| ' + ' | '.join(['---'] + ['---:'] * (len(score_headers) - 1)) + ' |']
    lines.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in score_rows)
    lines += ['', 'F1 is the macro mean over completed excerpts. "Offset F1" requires both a correct '
              'onset and release; "String F1" also requires the correct guitar string. Processing/audio '
              'is measured computation divided by input duration. Below 1 means adequate average '
              'throughput in this replay; it does not guarantee that every packet meets its deadline.', '',
              '## Delivery delay for correctly matched events', '',
              '| ' + ' | '.join(latency_headers) + ' |',
              '| ' + ' | '.join(['---'] + ['---:'] * (len(latency_headers) - 1)) + ' |']
    lines.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in latency_rows)
    lines += ['', 'Delivery delay is the virtual delivery timestamp minus the matching reference onset or '
              'release, in milliseconds. It includes packet waiting and measured processing. A source '
              'timestamp can be backdated by the validation-selected decoder; backdating does not make '
              'the event available sooner. Negative values mean the estimate arrived before the reference '
              'boundary. Quantiles apply only to correctly matched notes; read their counts alongside '
              'F1 and missed/extra counts. Missed notes have no finite delivery time. End-of-input '
              'flush releases and censored offsets are excluded from the matched-release latency.', '',
              '## Causal contract and comparison limits', '',
              'The frontend uses a 2048-sample right-edge STFT window (128 ms at 16 kHz), a 160-sample '
              '(10 ms) hop and no future audio. Warmup/padding is recorded by the implementation. '
              'The CNN uses left-only temporal padding; normalization operates within each frame. '
              'A two-layer unidirectional GRU with hidden size 200 retains its hidden state across packets. '
              'Decoding uses only observed frames and already-emitted events. Packet-size invariance and '
              'prefix causality are implementation checks, not acoustic accuracy guarantees.', '',
              'The offline reference uses a centered CQT, a bidirectional GRU, its original training '
              'normalization and retrospective note decoding. The new model has a different frontend '
              'and normalization as well as a different recurrent direction. This is a prototype '
              'comparison, not a controlled claim that one architecture is better.', '',
              f'The new run uses {epochs} epochs and {updates} optimizer updates within the small '
              'training budget. Such a short run does not establish convergence or a quality ceiling. '
              'Reference labels are provided only to training/evaluation; decoder parameters are '
              'selected on the validation split and then frozen for the test excerpts.', '',
              f'The initial {run.get("discarded_optimizer_updates", 0)} updates were discarded after target-grid/EOF '
              'edge-case repairs; the corrected model starts from scratch. Total stage updates are '
              f'{run.get("total_stage_optimizer_updates", updates)}, within the original 500-update cap. '
              'Post-training decoder calibration expands the validation-only threshold/timestamp grid with zero '
              'optimizer updates. Backdating adjusts source estimates and never the time of delivery.', '',
              'The 40 ms minimum-note setting guards retriggers; it is not a minimum duration for every published '
              'note. Backdated fret changes can create very short intervals, which remain in the event ledger. '
              'EOF releases are file-finalization events rather than observed note releases.', '',
              f'Inference receives up to {input_seconds:g} seconds; pitch/string note and frame scoring '
              f'uses the first {score_seconds:g} seconds with later release context. Onset tolerance is '
              '50 ms with 50-cent pitch tolerance. Offset tolerance is max(50 ms, 20% of reference '
              'duration). GuitarSet players are shared across the existing progression split, so this '
              'is not an unseen-player or real electric-guitar result.', '',
              '## Reproduce and inspect', '',
              'Open `review.html` to listen, compare source note times with reference notes, and inspect '
              'source-to-delivery event links. The event list follows the replay delivery clock. '
              'JSON preserves estimated source intervals and separate delivery timestamps; MIDI and '
              'string/fret CSV remain useful for offline inspection.', '',
              'Device: `' + str(run.get('device', 'not recorded')) + '`. Model parameters: ' +
              (f'{int(params):,}' if params is not None else 'not recorded') + '. Peak GPU allocated '
              'tensors: ' + _fmt(run.get('peak_gpu_mib'), 1) + ' MiB. This allocation is not total '
              'GPU/system memory consumption.', '',
              'Causal checkpoint SHA256: `' + str(run.get('checkpoint_sha256', 'not recorded')) + '`.',
              'Configuration, checkpoint/source hashes, decoder parameters, training history and '
              'timing scope are preserved in `run.json` and the evaluation result files.', '',
              'The next decision is whether the observed accuracy/delivery tradeoff is useful enough '
              'for further causal-model work. In this run, onset F1 0.116 is too weak to replace the offline '
              'default; improve acoustic/decoder quality before a microphone product. Independently test capture/resampling and '
              'packet deadlines before presenting this as live transcription. The earlier offline '
              'separator and beat tracker do not become causal by attaching this note decoder.', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    csv_headers = ['id', 'mode', 'onset_f1', 'onset_offset_f1', 'frame_f1', 'tab_onset_f1',
                   'missed_notes', 'extra_notes', 'processing_seconds', 'real_time_factor',
                   'chunk_processing_p50_ms', 'chunk_processing_p95_ms',
                   'matched_onset_count', 'matched_onset_delivery_p50_ms', 'matched_onset_delivery_p95_ms',
                   'matched_offset_count', 'matched_offset_delivery_p50_ms', 'matched_offset_delivery_p95_ms']
    with (output / 'evaluation' / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(csv_headers)
        for record in records:
            timing, latency = record.get('timing', {}), record.get('latency', {})
            errors = record.get('pitch_metrics', {}).get('errors', {})
            writer.writerow([record['id'], record['mode'], _score(record), _score(record, 'onset_offset'),
                             _score(record, 'frame'), _tab_score(record), errors.get('missed_notes'),
                             errors.get('extra_notes')] + [timing.get(key) for key in csv_headers[8:12]] +
                            [latency.get(key) for key in csv_headers[12:]])
    payload = []
    for record in records:
        mode, clip_id = record['mode'], record['id']
        metrics = record.get('pitch_metrics', {})
        notes = record.get('notes', metrics.get('predicted', []))
        reference = record.get('reference', metrics.get('reference', []))
        duration = _number(metrics.get('score_duration')) or score_seconds
        svg = roll_svg(reference, notes, metrics.get('onset_matches', []), duration,
                       clip_id + ' · ' + LABELS.get(mode, mode))
        events = _events(record)
        event_duration = _number(record.get('timing', {}).get('input_seconds')) or input_seconds
        event_svg = _event_svg(events, event_duration, clip_id + ' · ' + mode)
        stem = Path('evaluation') / 'results' / mode / clip_id
        json_path = output / stem.with_suffix('.json')
        serialized = json_path.read_text(encoding='utf-8') if json_path.exists() else json.dumps(
            {'notes': notes, 'events': record.get('events', [])}, indent=2)
        plot_name = mode + '__' + clip_id
        (plots / (plot_name + '-notes.svg')).write_text(svg, encoding='utf-8')
        (plots / (plot_name + '-events.svg')).write_text(event_svg, encoding='utf-8')
        payload.append({'id': clip_id, 'mode': mode, 'audio': _url(record.get('relativeaudio',
                        record.get('audio', 'evaluation/clips/' + clip_id + '.wav'))),
                        'svg': svg, 'eventSvg': event_svg, 'duration': duration, 'events': events,
                        'onset': _score(record), 'offset': _score(record, 'onset_offset'),
                        'tab': _tab_score(record), 'errors': metrics.get('errors', {}),
                        'onsetCount': metrics.get('onset', {}).get('predicted_count', len(notes)),
                        'latency': record.get('latency', {}), 'timing': record.get('timing', {}),
                        'stem': _url(stem), 'noteJson': serialized,
                        'serializedSource': json_path.exists()})
    payload.sort(key=lambda item: (not item['id'].startswith('00_Rock'), item['id'],
                                   ordered.index(item['mode']) if item['mode'] in ordered else len(ordered)))
    page = VIEWER.replace('__SCORES__', _table(score_headers, score_rows))
    page = page.replace('__LATENCY__', _table(latency_headers, latency_rows))
    page = page.replace('__BUDGET__', html.escape(f'{epochs} epochs · {updates} updates'))
    page = page.replace('__DATA__', json.dumps(_clean(payload), ensure_ascii=True, allow_nan=False).replace('</', '<\\/'))
    page = page.replace('__LABELS__', json.dumps(LABELS))
    (output / 'review.html').write_text(page, encoding='utf-8')
    print('Report:', output / 'report.md', '; viewer:', output / 'review.html', flush=True)


VIEWER = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Causal guitar lab</title><style>
*{box-sizing:border-box}body{margin:0;background:#101924;color:#e4edf4;font:16px/1.5 system-ui}main{max-width:1180px;margin:auto;padding:28px}
h1{font-size:30px;margin:0}h2{font-size:21px}p{color:#bac9d6}section{background:#192737;padding:22px;margin:20px 0;border-radius:12px}
select,button{font:inherit;border:1px solid #50708a;border-radius:6px;background:#0f1b27;color:#e4edf4;padding:8px}label{display:inline-block;margin:6px 12px 6px 0}
select{max-width:100%}audio{width:100%;margin:18px 0}a{color:#77d7be;margin-right:18px}.scroll,.table-scroll{overflow:auto}svg{width:100%;min-width:650px;display:block}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:9px 11px;text-align:right;border-bottom:1px solid #425267;white-space:nowrap}th:first-child,td:first-child{text-align:left}
.warning{border-left:4px solid #dbad67;padding-left:14px;color:#efd3a9}.metrics{font-variant-numeric:tabular-nums}small{color:#b1c4d5}#status{color:#efb37f}#roll,#events{cursor:crosshair}
.tip{font-size:14px}button{cursor:pointer;margin-right:8px}#eventlog{max-height:360px;overflow:auto}#eventlog td{padding:6px 9px}input[type=checkbox]{accent-color:#65d9b0}details summary{cursor:pointer}
</style><main><h1>Causal guitar lab</h1><p>Stage 8 · isolated guitar packet replay · __BUDGET__</p>
<p class="warning">This is a file replay of 16 kHz audio, with a virtual delivery clock. Microphone capture, resampling, hardware latency, live separation and live rhythm detection have not been tested.</p>
<section><h2>Accuracy and throughput</h2><p>Onset F1 scores attacks; offset F1 also scores releases; string F1 also requires the right guitar string. Processing/audio measures average computation.</p>__SCORES__
<p class="tip">The offline CQT model and the new causal STFT model have different frontends, normalization and training. These results do not isolate an architectural advantage.</p></section>
<section><h2>When did matched notes arrive?</h2>__LATENCY__<p class="tip">Delay is delivery time minus reference event time, only for correctly matched notes. Misses have no finite delay. Review matched counts and accuracy together. End-of-input flush releases are excluded.</p></section>
<section><h2>Listen and inspect</h2><label>Excerpt <select id="clip"></select></label><label>Mode <select id="mode"></select></label><button id="compare">Compare causal / offline</button>
<audio id="audio" controls preload="metadata"></audio><p id="status" role="status"></p><p id="metrics" class="metrics"></p><p id="timing" class="metrics"></p>
<p><a id="json">Notes + delivery times</a><a id="midi">MIDI</a><a id="csv">String/fret CSV</a></p><h3>Source note intervals</h3><div class="scroll" id="roll"></div>
<p class="tip">Click to seek. Reference: blue matched, purple missed. Predictions: green matched, red extra. These are source timestamps, which can be backdated; they are not event delivery times. Hover for note details.</p>
<h3>Source → delivery events</h3><div class="scroll" id="events"></div><p class="tip">Hollow markers show source time; filled markers show delivery time. Green circles: note starts. Orange diamonds: releases. Lines connect the two clocks. Strings 0–5 run low E to high E. Hover for source and delivery times.</p>
<label><input type="checkbox" id="showall"> Show all events, including events not yet delivered at the playhead</label><p id="eventcount" class="metrics" aria-live="polite"></p>
<div id="eventlog" class="table-scroll"><table><thead><tr><th>Event</th><th>Pitch</th><th>String/fret</th><th>Source s</th><th>Delivery s</th><th>Delay ms</th></tr></thead><tbody id="eventrows"></tbody></table></div>
<p class="tip">The list follows virtual delivery time. It contains all detections, including false positives; the summary latency table contains only matched events. Showing the final 40 available events keeps the view readable. End-of-input flush events are labeled.</p></section>
<section><details><summary>What this prototype establishes</summary><p>The new network has no future-audio lookahead. A right-edge 128 ms STFT window, left-padded CNN and unidirectional recurrent state process incoming packets. Prefix causality and packet-size consistency test the implementation. The short training run, shared GuitarSet players and pre-resampled input limit conclusions about real musicians, electric guitar and microphones.</p>
<p>A backdated source onset can improve timestamp alignment while the actual event still arrives later. Average throughput below 1 does not prove every packet deadline was met. Check processing tails, missed events and release errors before building a live player.</p></details></section>
<p><a href="report.md">Protocol and limitations</a><a href="evaluation/metrics.csv">Per-excerpt metrics</a><a href="evaluation/run.json">Run provenance</a></p></main>
<script>const data=__DATA__,labels=__LABELS__;const $=id=>document.getElementById(id);let current,jsonUrl,lastEventKey='';const fmt=(v,n=3)=>v===null||v===undefined?'—':Number(v).toFixed(n);
const clips=[...new Set(data.map(x=>x.id))];clips.forEach(id=>$('clip').add(new Option(id,id)));
function modeMenu(){const old=$('mode').value,choices=data.filter(x=>x.id===$('clip').value);$('mode').replaceChildren();choices.forEach(x=>$('mode').add(new Option(labels[x.mode]||x.mode,x.mode)));if(choices.some(x=>x.mode===old))$('mode').value=old;load();}
function addPlayhead(container,id,height){const svg=$(container).querySelector('svg');if(!svg)return;const line=document.createElementNS('http://www.w3.org/2000/svg','line');line.id=id;line.setAttribute('y1','15');line.setAttribute('y2',height);line.setAttribute('stroke','#fff4b5');line.setAttribute('stroke-width','2');svg.append(line);}
function load(){const previous=current;current=data.find(x=>x.id===$('clip').value&&x.mode===$('mode').value);if(!current)return;const time=previous&&previous.id===current.id?$('audio').currentTime:0;$('audio').pause();
if(!previous||previous.audio!==current.audio){$('audio').src=current.audio;$('audio').onloadedmetadata=()=>{$('audio').currentTime=Math.min(time,$('audio').duration||time);update();};}else $('audio').currentTime=time;
$('status').textContent='';$('metrics').textContent=`Onset F1 ${fmt(current.onset)} · offset F1 ${fmt(current.offset)} · string F1 ${fmt(current.tab)} · missed ${current.errors.missed_notes??'—'} · extra ${current.errors.extra_notes??'—'}`;
const t=current.timing,l=current.latency;$('timing').textContent=`Processing/audio ${fmt(t.real_time_factor)} · packet processing p50 / p95 ${fmt(t.chunk_processing_p50_ms,1)} / ${fmt(t.chunk_processing_p95_ms,1)} ms`;
if(current.mode!=='offline_bidir')$('timing').textContent+=` · matched onset delivery p50 / p95 ${fmt(l.matched_onset_delivery_p50_ms,1)} / ${fmt(l.matched_onset_delivery_p95_ms,1)} ms`;
['json','midi','csv'].forEach((id,i)=>{$(id).href=current.stem+['.json','.mid','.csv'][i];$(id).download='';});$('roll').innerHTML=current.svg;$('events').innerHTML=current.eventSvg;addPlayhead('roll','source-playhead',215);addPlayhead('events','delivery-playhead',260);lastEventKey='';update();
$('compare').disabled=!data.some(x=>x.id===current.id&&x.mode===(current.mode==='offline_bidir'?'causal_40ms':'offline_bidir'));}
function eventList(){if(!current)return;const all=$('showall').checked,now=$('audio').currentTime;const available=current.events.filter(e=>all||(e.delivery!==null&&e.delivery<=now));const shown=available.slice(-40);const key=current.id+current.mode+all+available.length;
if(key===lastEventKey)return;lastEventKey=key;$('eventcount').textContent=current.mode==='offline_bidir'?'The offline decoder has no incremental delivery events.':`${available.length} of ${current.events.length} events ${all?'shown':'delivered by '+fmt(now)+' s'} · displaying the latest ${shown.length}`;
$('eventrows').replaceChildren();shown.forEach(e=>{const tr=document.createElement('tr');const values=[e.type+(e.flush?' (flush)':''),e.pitch??'—',(e.string??'—')+'/'+(e.fret??'—'),fmt(e.source),fmt(e.delivery),e.source===null||e.delivery===null?'—':fmt(1000*(e.delivery-e.source),1)];values.forEach(value=>{const td=document.createElement('td');td.textContent=value;tr.append(td);});$('eventrows').append(tr);});}
function update(){if(!current)return;const time=$('audio').currentTime;const a=$('source-playhead');if(a){const x=45+Math.min(Math.max(time,0),current.duration)/current.duration*945;a.setAttribute('x1',x);a.setAttribute('x2',x);}const b=$('delivery-playhead'),svg=$('events').querySelector('svg');if(b&&svg){const duration=Number(svg.dataset.duration)||12,x=60+Math.min(Math.max(time,0),duration)/duration*925;b.setAttribute('x1',x);b.setAttribute('x2',x);}eventList();}
function seek(event,container,left,width,duration){const svg=$(container).querySelector('svg');if(!svg||!current)return;const bounds=svg.getBoundingClientRect(),x=(event.clientX-bounds.left)/bounds.width*1000;$('audio').currentTime=Math.min($('audio').duration||duration,Math.max(0,(x-left)/width*duration));update();}
$('clip').onchange=modeMenu;$('mode').onchange=load;$('audio').ontimeupdate=update;$('audio').onseeked=update;$('showall').onchange=()=>{lastEventKey='';eventList();};$('audio').onerror=()=>{$('status').textContent='The local audio file could not be loaded. Keep review.html beside the evaluation folder.';};
$('roll').onclick=e=>seek(e,'roll',45,945,current.duration);$('events').onclick=e=>seek(e,'events',60,925,Number($('events').querySelector('svg')?.dataset.duration)||12);
$('compare').onclick=()=>{const mode=current.mode==='offline_bidir'?'causal_40ms':'offline_bidir';if(data.some(x=>x.id===current.id&&x.mode===mode)){$('mode').value=mode;load();}};
$('json').onclick=e=>{e.preventDefault();if(jsonUrl)URL.revokeObjectURL(jsonUrl);jsonUrl=URL.createObjectURL(new Blob([current.noteJson],{type:'application/json'}));const a=document.createElement('a');a.href=jsonUrl;a.download=current.id+'-'+current.mode+'.json';document.body.append(a);a.click();a.remove();};window.addEventListener('beforeunload',()=>{if(jsonUrl)URL.revokeObjectURL(jsonUrl);});if(data.length)modeMenu();</script></html>'''
