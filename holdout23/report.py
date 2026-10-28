"""Held-out audio review which labels the development choice without test ranking."""

import html
import json
from pathlib import Path
from urllib.parse import quote
from baseline.download import ROOT, sha256
from baseline.report import roll_svg
from decoder14.report import TEMPLATE as STYLE_SOURCE
from streaming.evaluate import save_json
from streaming.report import _table
from supervision16.report import TEMPLATE as PLAYER_SOURCE
from .data import MODES, read

LABELS = {'parent500': 'Parent / 500 updates', 'original650': 'Original loss / 650 updates',
          'uniform_ce650': 'Uniform CE / selected on development', 'balanced_ce650': 'Weighted CE / 650 updates'}


def number(value, digits=4):
    return '—' if value is None else f'{value:.{digits}f}'


def pct(value):
    return '—' if value is None else f'{100 * value:.2f}%'


def findings(output):
    output = Path(output)
    summary = read(output / 'test/summary.json')
    comparison = read(output / 'comparison.json')
    selected = summary['modes']['uniform_ce650']
    original = summary['modes']['original650']
    lines = ['# Stage23: locked progression3 evaluation', '',
        f"The preselected uniform_ce650 model reaches test state F1 {selected['exact_state_f1']:.5f} and note-onset F1 "
        f"{selected['onset_f1']:.5f}. Versus the locked original650 comparator, the differences are "
        f"{selected['exact_state_f1'] - original['exact_state_f1']:+.5f} and {selected['onset_f1'] - original['onset_f1']:+.5f}. "
        'The Stage22 development selection is preserved; no model is reselected from these scores.', '',
        'Zero training updates, zero optimizer instantiations and no normalization/calibration changes. '
        'All four checkpoint identities, the decoder and the exact copied development selection were locked before test audio/annotations were read. '
        'There are48 scored twelve-second replays plus four one-second test-audio warmups.', '',
        '## Data scope', '', read(output / 'protocol.json')['data_scope'], '',
        'The twelve test IDs are disjoint from training/development IDs. All six players occur across splits. '
        'These data therefore do not establish fresh independent-player or electric-guitar generalization. '
        'No scores here may change the selected model or tune a threshold.', '',
        '| Locked model | Development state F1 | Test state F1 | Development onset F1 | Test onset F1 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for mode in MODES:
        value = comparison['per_mode'][mode]
        lines.append(f"| {LABELS[mode]} | {value['exact_state_f1']['development']:.4f} | {value['exact_state_f1']['test']:.4f} | "
                     f"{value['onset_f1']['development']:.4f} | {value['onset_f1']['test']:.4f} |")
    lines += ['', '| Locked model | Onset+offset F1 | String-note F1 | Decoded pitch-frame F1 | Best active fret | Equal-supported-position fret |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for mode in MODES:
        value = summary['modes'][mode]
        lines.append(f"| {LABELS[mode]} | {value['onset_offset_f1']:.4f} | {value['tab_onset_f1']:.4f} | {value['frame_f1']:.4f} | "
                     f"{pct(value['best_active_fret_accuracy'])} | {pct(value['best_active_equal_position_accuracy'])} |")
    lines += ['', 'State/note F1 values are means over the same twelve clips. Best-active fret accuracy pools valid GT-active string-frames; equal-position accuracy averages recalls over test-supported string/fret positions. '
        'Conditional decoder-activity scores use another denominator. Test-supported positions and musical content differ from development, so differences are descriptive and do not identify a cause.', '',
        'Original test waveforms/annotations and existing target caches are SHA-bound. Cached targets match the frozen right-edge target builder byte-for-byte. '
        'The observed span is12 seconds, with scored reference onsets in [0,10). Frame targets0:1000 align with model predictions13:1013; first26 source frames and per-string conflicts are masked.', '',
        'Each model retains its checkpoint normalization. All use the same931,361-parameter causal two-plane graph, duplicated past-only128ms frontend, four stable state frames for a new note/fret, '
        'onset0.7/activity0.25, no additional backdate and130ms global source correction. Onset peaks only retrigger same-fret notes. No optimizer, gradient update or test-derived calibration is used.', '',
        'Fresh processing/audio includes packet frontend/model/decoder work and exact probability recording but excludes EOF flush processing. Virtual completion includes flush cost. '
        'Matched-note completion delays omit misses and use different matched subsets. The inference graph is identical, so timing variation cannot establish a speed gain; no capture-hardware latency is measured.', '',
        'The locked uniform model remains a research checkpoint selected on Stage22 development. This evaluation does not promote it to the default offline workflow or establish product-quality transcription.', '']
    return '\n'.join(lines)


def write_report(output):
    output = Path(output)
    if (output / 'verification.json').exists():
        raise ValueError('Completed Stage23 presentation must not be regenerated')
    summary = read(output / 'test/summary.json')
    comparison = read(output / 'comparison.json')
    rows = [read(path) for path in sorted((output / 'test/results').glob('*/*-metrics.json'))]
    scores, fret, latency = [], [], []
    for mode in MODES:
        value, paired = summary['modes'][mode], comparison['per_mode'][mode]
        scores.append([LABELS[mode], number(paired['exact_state_f1']['development']), number(value['exact_state_f1']),
                       number(paired['onset_f1']['development']), number(value['onset_f1'])])
        fret.append([LABELS[mode], pct(value['best_active_fret_accuracy']), pct(value['best_active_equal_position_accuracy']),
                     pct(value['conditional_fret_accuracy']), number(value['onset_offset_f1']), number(value['tab_onset_f1'])])
        latency.append([LABELS[mode], number(value['real_time_factor'], 3), value['matched_onset_count'],
                        number(value['matched_onset_delivery_p50_ms'], 1), number(value['matched_onset_delivery_p95_ms'], 1),
                        sum(row['pitch_metrics']['errors']['extra_notes'] for row in rows if row['mode'] == mode)])
    tables = {'scores': _table(['Locked model', 'Dev state F1', 'Test state F1', 'Dev onset F1', 'Test onset F1'], scores),
        'fret': _table(['Model', 'Best fret / active GT', 'Equal-position fret', 'Fret / both active', 'Onset+offset F1', 'String-note F1'], fret),
        'latency': _table(['Model', 'Processing/audio', 'Matched attacks', 'Completion p50 ms', 'Completion p95 ms', 'Extra notes'], latency)}
    payload = []
    for row in rows:
        mode, clip, metrics = row['mode'], row['id'], row['pitch_metrics']
        stem = Path('test/results') / mode / clip
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
        raise ValueError('Frozen player cycle changed')
    script = script.replace(old, "$('compare').onclick=()=>{const keys=Object.keys(LABELS);$('mode').value=keys[(keys.indexOf(current.mode)+1)%keys.length];render();};")
    text = findings(output)
    values = {**tables, 'css': STYLE_SOURCE.split('<style>', 1)[1].split('</style>', 1)[0],
        'payload': json.dumps(payload).replace('<', '\\u003c'), 'labels': json.dumps(LABELS),
        'outcome': html.escape(text.split('\n\n')[1]), 'script': script,
        'selection': html.escape(json.dumps(read(output / 'selection.json'), indent=2))}
    page = TEMPLATE.replace('__SCRIPT__', script)
    for key, value in values.items(): page = page.replace('__' + key.upper() + '__', value)
    if '__PAYLOAD__' in page or '__LABELS__' in page or '__SCRIPT__' in page:
        raise ValueError('Unresolved player template data')
    (output / 'review.html').write_text(page, encoding='utf-8')
    for name in ('report.md', 'findings.md'): (output / name).write_text(text, encoding='utf-8')
    save_json(output / 'presentation-provenance.json', {'report_source_sha256': sha256(Path(__file__)),
        'borrowed_source_sha256': {name: sha256(ROOT / name) for name in ['supervision16/report.py', 'decoder14/report.py', 'baseline/report.py', 'streaming/report.py']},
        'review_html_sha256': sha256(output / 'review.html'), 'report_md_sha256': sha256(output / 'report.md'),
        'numeric_artifact_sha256': {name: sha256(output / name) for name in ['protocol.json', 'selection.json', 'comparison.json',
            'model-integrity.json', 'test-inputs.json', 'test/run.json', 'test/summary.json']}})
    save_json(output / 'next-experiment.json', {'stage': 24, 'status': 'proposed_not_implemented',
        'kind': 'explicit_research_inference_cli', 'new_optimizer_updates': 0,
        'question': 'Can the locked research model be used through one local file-to-notes CLI without changing selection, normalization or decoding?',
        'source_root': 'outputs/holdout23', 'locked_checkpoint': 'outputs/balance22/models/uniform_ce/final.pt',
        'locked_checkpoint_sha256': read(output / 'test/run.json')['checkpoint_sha256']['uniform_ce650'],
        'scope': 'Explicit research opt-in; retain the existing offline default. Reuse exact frozen PCM/decoder behavior, export source-timed JSON/CSV/MIDI and a simple timed string/fret text view. Preserve model hashes and timing scope; no meter prediction, threshold search or model-quality promotion.',
        'verification_data': 'Use an existing development excerpt and synthetic boundary fixtures for engineering parity, not progression3 scores for tuning.',
        'model_selection_allowed': False, 'calibration_search_allowed': False, 'training_allowed': False,
        'production_promotion': False})
    print('Stage23 review saved:', output / 'review.html', flush=True)


TEMPLATE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Stage23 - Locked progression3 evaluation</title><style>__CSS__</style></head><body><main>
<div class="kicker">Stage 23 · zero training · locked models</div><h1>Do the locked development gains carry to progression 3?</h1><p>__OUTCOME__</p>
<p class="notice">The selected model is fixed from Stage22 development. These recordings had earlier historical evaluations and share players across splits; this is not a fresh independent-player or electric-guitar test. No test-driven reselection is allowed.</p>
<section><h2>Development and held-out scores</h2>__SCORES__<p>All values average the same twelve clips within their split. The selected-on-development label is preserved regardless of test rankings. Four checkpoint hashes, scoring rules, normalization and decoder settings were locked before test input reads.</p></section>
<section><h2>Fret and note diagnostics</h2>__FRET__<p>Best-fret accuracy uses all valid GT-active string-frames. Equal-position accuracy averages recalls after pooling test clips. Both-active fret accuracy uses only frames where GT and prediction are active. These denominators and class supports differ from mean clip state F1.</p></section>
<section><h2>Frozen scope</h2><p>0 optimizer updates, 0 optimizer instantiations and 0 normalization updates. There are48 scored twelve-second replays and four one-second test-audio warmups. Twelve existing target caches reproduce the frozen right-edge label builder byte-for-byte. Test IDs are disjoint from train/development IDs; all six players are shared.</p><p><a href="protocol.json">Pre-test protocol</a><a href="selection.json">Exact copied development selection</a><a href="model-integrity.json">Unchanged checkpoint files</a><a href="comparison.json">Descriptive contrasts</a></p><p>Onset0.7/activity0.25, zero additional backdate and130ms source correction stay fixed. All four graphs have931,361 parameters and duplicated past-only128ms features.</p></section>
<section><h2>Listen and compare the locked models</h2><label>Recording<br><select id="clip"></select></label><label>Model<br><select id="mode"></select></label><button id="compare">Next locked model, keep playback</button><audio id="audio" controls preload="metadata"></audio><p id="currentinfo"></p><p id="clock">0.000 s</p><p class="legend"><span style="color:#739cff">Reference / matched</span><span style="color:#c8a1ff">Reference / missed</span><span style="color:#58d9a2">Prediction / matched</span><span style="color:#ff7b7b">Prediction / extra</span></p><div id="roll"></div><div id="exports"></div><button id="download-json">Download exact note JSON</button><h3>Source → packet → virtual completion</h3><p>White marks estimated source time, blue marks packet availability and green marks completion after measured processing. EOF closures are tagged; source correction cannot advance availability or completion.</p><div id="events"></div><label><input id="showall" type="checkbox"> Show all events, including after playback</label><p id="eventcount"></p><div id="eventwrap"><table><thead><tr><th>Event</th><th>Pitch</th><th>Source (s)</th><th>Packet (s)</th><th>Completion (s)</th></tr></thead><tbody id="eventrows"></tbody></table></div></section>
<section><h2>Measured file-replay completion</h2>__LATENCY__<p>Processing/audio includes probability recording and excludes EOF flush processing; virtual completion includes flush cost. Matched-note delays omit misses and compare different subsets. Identical graphs do not establish a speed benefit, and no microphone or sound-card latency is measured.</p></section>
<section><details><summary>Copied development decision and limits</summary><pre>__SELECTION__</pre><p>These are the original Stage22 development scores, not a new test selection. Previously evaluated progression3 audio, shared players and a reused decoder limit generalization claims. The offline default remains separate from this research checkpoint.</p></details></section><p><a href="report.md">Report</a><a href="findings.md">Findings</a><a href="audit.json">Cached replay audit</a><a href="viewer-validation.json">Viewer checks</a><a href="presentation-provenance.json">Presentation provenance</a></p>
</main><script>__SCRIPT__</script></body></html>'''
