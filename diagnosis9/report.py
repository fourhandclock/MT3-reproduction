"""Portable review of frozen-model decoder diagnosis and causal repair."""

import csv
import hashlib
import html
import json
import math
from pathlib import Path
from urllib.parse import quote

from baseline.report import roll_svg
from streaming.report import _event_svg, _events


LABELS = {"strict": "Stage 8 strict decoder", "repair": "Validation-selected repair"}


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def _fmt(value, places=3):
    value = _number(value)
    return "—" if value is None else f"{value:.{places}f}"


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _url(path):
    return quote(str(path).replace("\\", "/"), safe="/")


def _table(headers, rows):
    return '<div class="table-scroll"><table><thead><tr>' + "".join(
        "<th>" + html.escape(str(value)) + "</th>" for value in headers) + "</tr></thead><tbody>" + "".join(
        "<tr>" + "".join("<td>" + html.escape(str(value)) + "</td>" for value in row) + "</tr>"
        for row in rows) + "</tbody></table></div>"


def _markdown_table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + [
        "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in rows]


def _score(record, name="onset"):
    return _number(record.get("pitch_metrics", {}).get(name, {}).get("f1"))


def _tab_score(record):
    value = record.get("tab_metrics", {})
    return _number(value.get("onset", {}).get("f1", value.get("onset_f1")))


def _mean(values):
    values = [number for number in map(_number, values) if number is not None]
    return sum(values) / len(values) if values else None


def _modes(records, summary):
    result = {key: dict(value) for key, value in summary.get("modes", {}).items() if isinstance(value, dict)}
    for mode in dict.fromkeys(row["mode"] for row in records):
        rows = [row for row in records if row["mode"] == mode]
        value = result.setdefault(mode, {})
        value.setdefault("count", len(rows))
        for metric in ("onset", "onset_offset", "frame"):
            value.setdefault(metric + "_f1", _mean(_score(row, metric) for row in rows))
        value.setdefault("tab_onset_f1", _mean(_tab_score(row) for row in rows))
        for edge in ("onset", "offset"):
            value.setdefault("matched_" + edge + "_count", sum(
                row.get("latency", {}).get("matched_" + edge + "_count", 0) or 0 for row in rows))
            # Pooled quantiles cannot be reconstructed by averaging clip quantiles.
            for quantile in ("p50", "p95"):
                value.setdefault(f"matched_{edge}_delivery_{quantile}_ms", None)
        processing = sum(_number(row.get("timing", {}).get("processing_seconds")) or 0 for row in rows)
        duration = sum(_number(row.get("timing", {}).get("input_seconds")) or 0 for row in rows)
        value.setdefault("real_time_factor", processing / duration if duration else None)
    return result


def _diagnostic(output, run, key, filenames):
    value = run.get(key)
    if isinstance(value, dict):
        return value.get("summary", value)
    candidates = [value] if isinstance(value, (str, Path)) else []
    candidates.extend(filenames)
    for name in candidates:
        if not name:
            continue
        path = Path(name)
        paths = [path] if path.is_absolute() else [output / path, Path.cwd() / path]
        for candidate in paths:
            if candidate.is_file():
                data = json.loads(candidate.read_text(encoding="utf-8"))
                return data.get("summary", data)
    return {}


def _head_rows(head):
    rows = []
    labels = {"state_argmax": "Exact string/fret · argmax",
              "state_activity_035": "Exact string/fret · activity ≥ 0.35",
              "activity_argmax": "String active/inactive · argmax",
              "activity_035": "String active/inactive · activity ≥ 0.35"}
    for key, label in labels.items():
        value = head.get(key, {})
        if value:
            rows.append([label, _fmt(value.get("mean_clip_f1")),
                         _fmt(value.get("pooled", {}).get("precision")),
                         _fmt(value.get("pooled", {}).get("recall"))])
    for key, value in head.get("pitch", {}).items():
        rows.append(["Pitch/frame · probability ≥ " + key, _fmt(value.get("mean_clip_f1")),
                     _fmt(value.get("pooled", {}).get("precision")),
                     _fmt(value.get("pooled", {}).get("recall"))])
    return rows


def _hybrid_rows(hybrid):
    """Label the factorial interventions without implying deployable oracle heads."""
    variants = [
        ("oracle_state_oracle_onset", "Oracle", "Oracle", 0),
        ("oracle_state_oracle_onset_selected_backdate", "Oracle", "Oracle", 60),
        ("oracle_state_predicted_onset", "Oracle", "Predicted", 0),
        ("oracle_state_predicted_onset_selected_backdate", "Oracle", "Predicted", 60),
        ("predicted_state_oracle_onset_zero_backdate", "Predicted", "Oracle", 0),
        ("predicted_state_oracle_onset", "Predicted", "Oracle", 60),
        ("predicted_state_predicted_onset_zero_backdate", "Predicted", "Predicted", 0),
        ("predicted_state_predicted_onset", "Predicted", "Predicted", 60),
    ]
    return [[state, onset, str(backdate), _fmt(value.get("note_onset_f1")),
             _fmt(value.get("note_onset_offset_f1")), _fmt(value.get("string_onset_f1"))]
            for key, state, onset, backdate in variants for value in [hybrid.get(key, {})] if value]


def write_report(output, records, summary, run):
    """Write report, paired audio/event viewer, metrics CSV and reusable plots."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # A presentation-only regeneration must not imply the full browser check was rerun.
    previous_page = output / "review.html"
    validation_path = output / "viewer-validation.json"
    previous_provenance = output / "presentation-provenance.json"
    prior_viewer_qa = {}
    if previous_provenance.is_file():
        prior_viewer_qa = json.loads(previous_provenance.read_text(encoding="utf-8")).get("prior_viewer_qa", {})
    if previous_page.is_file() and validation_path.is_file() and validation_path.stat().st_mtime_ns >= previous_page.stat().st_mtime_ns:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        previous_html = previous_page.read_text(encoding="utf-8")
        script = previous_html.split("<script>", 1)[-1].rsplit("</script>", 1)[0]
        prior_viewer_qa = {"passed": validation.get("passed"), "completed_at": validation.get("completed_at"),
                           "review_html_sha256": hashlib.sha256(previous_page.read_bytes()).hexdigest(),
                           "interactive_script_sha256": hashlib.sha256(script.encode("utf-8")).hexdigest(),
                           "validation_json_sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
                           "combinations": validation.get("combinations"), "export_link_checks": validation.get("export_link_checks")}
    plots = output / "evaluation/plots"
    plots.mkdir(parents=True, exist_ok=True)
    modes = _modes(records, summary)
    ordered = [mode for mode in LABELS if mode in modes] + [mode for mode in modes if mode not in LABELS]
    headers = ["Decoder", "Clips", "Onset F1", "Onset+offset F1", "Frame F1", "String F1", "Processing/audio"]
    rows = [[LABELS.get(mode, mode), modes[mode].get("count", 0)] + [
        _fmt(modes[mode].get(key)) for key in ("onset_f1", "onset_offset_f1", "frame_f1", "tab_onset_f1", "real_time_factor")]
        for mode in ordered]
    latency_headers = ["Decoder", "Matched onsets", "Onset p50 / p95 ms", "Matched releases", "Release p50 / p95 ms"]
    latency_rows = [[LABELS.get(mode, mode), value.get("matched_onset_count", 0),
                     _fmt(value.get("matched_onset_delivery_p50_ms"), 1) + " / " + _fmt(value.get("matched_onset_delivery_p95_ms"), 1),
                     value.get("matched_offset_count", 0),
                     _fmt(value.get("matched_offset_delivery_p50_ms"), 1) + " / " + _fmt(value.get("matched_offset_delivery_p95_ms"), 1)]
                    for mode in ordered for value in [modes[mode]]]
    head = _diagnostic(output, run, "head_diagnostics_summary", ["validation-head-diagnostics.json"])
    oracle = _diagnostic(output, run, "oracle_summary", ["validation-oracle.json", "oracle-summary.json", "oracle.json"])
    hybrid = _diagnostic(output, run, "hybrid_summary", ["hybrid-head-diagnostics.json"])
    context = _diagnostic(output, run, "context_summary", ["validation-context-diagnostics.json"])
    hybrid_headers = ["State/fret head", "Onset head", "Backdate ms", "Onset F1", "Onset+offset F1", "String F1"]
    hybrid_rows = _hybrid_rows(hybrid)
    context_headers = ["Validation context", "Exact string/fret frame F1", "Note-onset F1"]
    context_rows = [[label, _fmt(value.get("heads", {}).get("string_fret_activity_035", {}).get("mean_clip_f1"), 4),
                     _fmt(value.get("note_onset", {}).get("mean_clip_f1"), 6)]
                    for key, label in [("continuous_12s", "Continuous 12 s"), ("reset_every_2s", "Reset caches/state every 2 s")]
                    for value in [context.get(key, {})] if value]
    head_headers = ["Validation measurement", "Mean clip F1", "Pooled precision", "Pooled recall"]
    head_rows = _head_rows(head)
    selection = run.get("selection", [])
    selection_text = json.dumps(_clean(selection), indent=2, allow_nan=False)
    oracle_text = json.dumps(_clean(oracle), indent=2, allow_nan=False)
    config = run.get("config", {})
    score_config = run.get("scoring_config", config.get("evaluation", {}))
    score_seconds = _number(score_config.get("score_seconds", config.get("score_seconds"))) or 10.0
    input_seconds = _number(config.get("input_seconds", config.get("evaluation_audio_seconds", config.get("clip_seconds")))) or 12.0
    strict = _number(modes.get("strict", {}).get("onset_f1"))
    repair = _number(modes.get("repair", {}).get("onset_f1"))
    outcome = (f"On these reused test excerpts, onset F1 changes from {_fmt(strict)} to {_fmt(repair)} "
               f"({repair-strict:+.3f}) when only the decoder changes." if strict is not None and repair is not None
               else "Read the completed strict/repair rows below; unavailable metrics are shown as dashes.")
    decision = ("The validation-selected smoothing is not promoted: its small validation gain did not carry over "
                "to the reused test excerpts. Keep the strict decoder as the causal reference and the offline "
                "CNN-GRU as the product default." if strict is not None and repair is not None and repair <= strict
                else "These reused excerpts alone do not justify promoting a production decoder; keep the offline product default.")
    interpretation = []
    if hybrid_rows:
        exact = hybrid.get("oracle_state_oracle_onset", {})
        state_oracle = hybrid.get("oracle_state_predicted_onset", {})
        onset_oracle = hybrid.get("predicted_state_oracle_onset", {})
        predicted_zero = hybrid.get("predicted_state_predicted_onset_zero_backdate", {})
        predicted_selected = hybrid.get("predicted_state_predicted_onset", {})
        onset_oracle_zero = hybrid.get("predicted_state_oracle_onset_zero_backdate", {})
        state_oracle_selected = hybrid.get("oracle_state_predicted_onset_selected_backdate", {})
        interpretation = [
            "The factorial checks point to the state/fret head as the main bottleneck in this frozen run. "
            f"Holding backdate at 0 ms, the predicted-head baseline is {_fmt(predicted_zero.get('note_onset_f1'))} onset F1: "
            f"replacing only state raises it to {_fmt(state_oracle.get('note_onset_f1'))}, while replacing only onset raises it "
            f"to {_fmt(onset_oracle_zero.get('note_onset_f1'))}. Holding backdate at 60 ms, the baseline is "
            f"{_fmt(predicted_selected.get('note_onset_f1'))}: replacing state reaches {_fmt(state_oracle_selected.get('note_onset_f1'))}, "
            f"while replacing onset reaches {_fmt(onset_oracle.get('note_onset_f1'))}. These contrasts hold the lag correction fixed. "
            "The state intervention replaces occupancy, fret and duration together; "
            "it does not isolate fret identity from those other state-head errors.",
            f"Both oracle heads reach {_fmt(exact.get('note_onset_f1'))} onset F1 at zero backdate. "
            "That is performance on privileged constructed reference targets, not an acoustic accuracy result or a universal ceiling. "
            "Reference occupancy has no learned acoustic lag. Applying the selected 60 ms backdate to these exact targets "
            "moves source estimates early by more than the 50 ms onset tolerance; the lower stress-check score is therefore "
            "not evidence that perfect acoustic predictions should use the same correction. Backdating never advances actual delivery.",
            "Onset and repetition errors remain after idealizing state. Among "
            f"{state_oracle.get('repeated_note_reference_count', '—')} consecutive same-string, same-pitch reference notes, "
            f"{state_oracle.get('repeated_note_missed_count', '—')} are missed with predicted onsets versus "
            f"{exact.get('repeated_note_missed_count', '—')} with oracle onsets at zero backdate. These string-aware "
            "structural counts concern references separated by at most the retrigger guard; they are not the overall note-F1 denominator.",
        ]
    context_text = ("Resetting convolution caches and recurrent state every 2 seconds gives no meaningful note-onset gain "
                    "here and slightly reduces exact string/fret frame F1. This diagnostic changes boundary context as well "
                    "as recurrent history; it does not rule out every long-context training problem. No reset strategy is promoted."
                    if context_rows else "Context diagnostics are recorded separately when available.")
    budget = str(run.get("optimizer_updates", 0)) + " new optimizer updates"
    protocol = [
        "The stage-8 causal acoustic checkpoint is frozen. Strict and repaired inference use the same frontend, "
        "feature frames, normalization, source sample rate, recurrent context and supported pitch/string bounds. "
        "This comparison isolates decoder behavior; it does not add model training.",
        "Decoder choices use progression-2 validation labels and are frozen before progression-3 test replay. "
        "The 12 test excerpts have already appeared in stages 1–8. This is an engineering diagnosis on reused "
        "examples, not fresh evidence of generalization. Players are shared across the existing split.",
        "Raw head scores measure individual frames, frame/string/fret classes or pitch/frame cells. Assembled note F1 "
        "measures complete detected attacks. Different counting units and boundary requirements mean these values "
        "cannot be compared as if they were the same metric. Diagnostic thresholds and lag sweeps describe the "
        "frozen model; they are not production corrections unless explicitly selected in run.json.",
        "Oracle inputs explicitly use ground-truth labels to test whether decoder logic can turn ideal head values "
        "into notes. Oracle performance diagnoses implementation behavior; no oracle values enter production "
        "inference, validation head predictions or test replay.",
        "Audio is already mono 16 kHz PCM. Packet arrivals and measured processing define a virtual delivery clock. "
        "This remains a file replay; microphone capture, device latency, resampling, live separation and rhythm "
        "inference are outside the measured path.",
        "A source timestamp may be backdated after evidence arrives. Actual delivery remains later and is recorded "
        "separately. Delay is delivery minus matched reference onset/release, including packet waiting and processing. "
        "Latency quantiles include only correctly matched events: misses have no finite delivery time. EOF flush "
        "releases and censored offsets are excluded from matched-release delay. Negative values mean early estimates.",
        f"Inference receives up to {input_seconds:g} seconds; scoring uses the first {score_seconds:g} seconds with "
        "later release context. Onset tolerance is 50 ms, pitch tolerance 50 cents, and offset tolerance "
        "max(50 ms, 20% of reference duration). These primary scores are macro means across completed excerpts.",
        "Processing/audio measures average computation divided by input duration. Below 1 shows sufficient average "
        "throughput for this replay and does not guarantee every packet deadline. GPU allocated tensors do not "
        "represent total device or system memory.",
    ]
    lines = ["# Stage 9: frozen causal-model diagnosis and decoder repair", "", outcome, "", decision, "", budget + ".", "",
             "## Paired note accuracy and processing", ""]
    lines += _markdown_table(headers, rows)
    lines += ["", "String F1 additionally requires the reference guitar string. Onset+offset F1 requires both attack and release.",
              "", "## Delivery delay for correctly matched events", ""]
    lines += _markdown_table(latency_headers, latency_rows)
    lines += ["", "Read matched counts alongside F1 and missed/extra notes. A dash means unavailable or no matched events; it is not zero delay.",
              "", "## Validation acoustic-head diagnosis", ""]
    lines += _markdown_table(head_headers, head_rows) if head_rows else ["Detailed head diagnostics are recorded in the linked JSON artifact."]
    lines += ["", "See [head diagnostics](validation-head-diagnostics.json) for per-string distributions, active-run fragmentation and signed lag sweeps.",
              "", "## Factorial head interventions", "", "Validation only. Oracle inputs explicitly contain privileged reference labels; no new inference or training.", ""]
    lines += _markdown_table(hybrid_headers, hybrid_rows) if hybrid_rows else ["No factorial summary was supplied."]
    lines += ["", "See [hybrid diagnostics](hybrid-head-diagnostics.json) and [oracle checks](oracle-diagnostics.json) for the full records.", ""]
    for paragraph in interpretation:
        lines.extend([paragraph, ""])
    lines += ["## Context reset check", ""]
    lines += _markdown_table(context_headers, context_rows) if context_rows else ["No context summary was supplied."]
    lines += ["", context_text, "", "[Context audit](validation-context-diagnostics.json) · [Readable context audit](validation-context-diagnostics.md)",
              "", "## Oracle decoder details", "", "Ground-truth head values are diagnostic inputs only.", "", "```json", oracle_text, "```",
              "", "## Frozen validation selection", "", "```json", selection_text, "```", "", "## Protocol and interpretation", ""]
    for paragraph in protocol:
        lines.extend([paragraph, ""])
    lines += ["The offline CNN-GRU remains the product default unless a separate quality decision is recorded. "
              "The earlier offline reference onset F1 was 0.525; these decoder-only diagnostics do not make that "
              "model, source separator or beat tracker causal.", "",
              "Checkpoint SHA256: `" + str(run.get("checkpoint_sha256", "not recorded")) + "`.", "",
              "Device: `" + str(run.get("device", "not recorded")) + "`. Configuration, source/checkpoint hashes, "
              "selection metrics and timing scope are preserved in [run.json](evaluation/run.json).", "",
              "Presentation was regenerated without changing the evaluation run. Current report-source and rendered-artifact hashes "
              "are recorded separately in [presentation-provenance.json](presentation-provenance.json).", "",
              "Open [review.html](review.html) to switch decoders while preserving the excerpt playhead, inspect "
              "source-versus-delivery events, and download exact saved JSON, MIDI and string/fret CSV.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    csv_headers = ["id", "mode", "onset_f1", "onset_offset_f1", "frame_f1", "tab_onset_f1", "missed_notes", "extra_notes",
                   "processing_seconds", "real_time_factor", "chunk_processing_p50_ms", "chunk_processing_p95_ms",
                   "matched_onset_count", "matched_onset_delivery_p50_ms", "matched_onset_delivery_p95_ms",
                   "matched_offset_count", "matched_offset_delivery_p50_ms", "matched_offset_delivery_p95_ms"]
    with (output / "evaluation/metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(csv_headers)
        for record in records:
            errors = record.get("pitch_metrics", {}).get("errors", {})
            writer.writerow([record["id"], record["mode"], _score(record), _score(record, "onset_offset"), _score(record, "frame"),
                             _tab_score(record), errors.get("missed_notes"), errors.get("extra_notes")] +
                            [record.get("timing", {}).get(key) for key in csv_headers[8:12]] +
                            [record.get("latency", {}).get(key) for key in csv_headers[12:]])
    payload = []
    for record in records:
        mode, clip_id = record["mode"], record["id"]
        metrics = record.get("pitch_metrics", {})
        notes = record.get("notes", metrics.get("predicted", []))
        reference = record.get("reference", metrics.get("reference", []))
        duration = _number(metrics.get("score_duration")) or score_seconds
        svg = roll_svg(reference, notes, metrics.get("onset_matches", []), duration, clip_id + " · " + LABELS.get(mode, mode))
        events = _events(record)
        event_svg = _event_svg(events, _number(record.get("timing", {}).get("input_seconds")) or input_seconds, clip_id + " · " + mode)
        stem = Path("evaluation/results") / mode / clip_id
        source = output / stem.with_suffix(".json")
        # Preserve exact saved JSON bytes/text rather than reconstructing a reduced export.
        serialized = source.read_text(encoding="utf-8") if source.is_file() else None
        plot_name = mode + "__" + clip_id
        (plots / (plot_name + "-notes.svg")).write_text(svg, encoding="utf-8")
        (plots / (plot_name + "-events.svg")).write_text(event_svg, encoding="utf-8")
        payload.append({"id": clip_id, "mode": mode, "audio": _url(record.get("relativeaudio", "evaluation/clips/" + clip_id + ".wav")),
                        "svg": svg, "eventSvg": event_svg, "duration": duration, "events": events,
                        "onset": _score(record), "offset": _score(record, "onset_offset"), "frame": _score(record, "frame"),
                        "tab": _tab_score(record), "errors": metrics.get("errors", {}), "latency": record.get("latency", {}),
                        "timing": record.get("timing", {}), "stem": _url(stem), "noteJson": serialized})
    payload.sort(key=lambda item: (not item["id"].startswith("00_Rock"), item["id"], ordered.index(item["mode"])))
    page = VIEWER.replace("__SCORES__", _table(headers, rows)).replace("__LATENCY__", _table(latency_headers, latency_rows))
    page = page.replace("__HEAD__", _table(head_headers, head_rows) if head_rows else "<p>See the saved head diagnostics artifact.</p>")
    page = page.replace("__ORACLE__", html.escape(oracle_text)).replace("__SELECTION__", html.escape(selection_text))
    page = page.replace("__OUTCOME__", html.escape(outcome)).replace("__BUDGET__", html.escape(budget))
    page = page.replace("__DECISION__", html.escape(decision))
    page = page.replace("__HYBRID__", _table(hybrid_headers, hybrid_rows) if hybrid_rows else "<p>No factorial summary was supplied.</p>")
    page = page.replace("__INTERPRETATION__", "".join("<p>" + html.escape(paragraph) + "</p>" for paragraph in interpretation))
    page = page.replace("__CONTEXT__", _table(context_headers, context_rows) if context_rows else "<p>No context summary was supplied.</p>")
    page = page.replace("__CONTEXT_TEXT__", html.escape(context_text))
    page = page.replace("__PROTOCOL__", "".join("<p>" + html.escape(paragraph) + "</p>" for paragraph in protocol))
    page = page.replace("__DATA__", json.dumps(_clean(payload), ensure_ascii=True, allow_nan=False).replace("</", "<\\/"))
    page = page.replace("__LABELS__", json.dumps(LABELS))
    (output / "review.html").write_text(page, encoding="utf-8")
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    provenance = {"presentation_only": True, "new_optimizer_updates": 0, "evaluation_run_changed": False,
                  "report_source_sha256": digest(__file__),
                  "original_report_source_sha256": run.get("source_sha256", {}).get("report.py"),
                  "rendered_sha256": {name: digest(output / name) for name in ("report.md", "review.html")}}
    if prior_viewer_qa:
        script = page.split("<script>", 1)[-1].rsplit("</script>", 1)[0]
        current_script_sha256 = hashlib.sha256(script.encode("utf-8")).hexdigest()
        provenance["prior_viewer_qa"] = prior_viewer_qa
        provenance["current_interactive_script_sha256"] = current_script_sha256
        provenance["interactive_script_unchanged_since_qa"] = current_script_sha256 == prior_viewer_qa.get("interactive_script_sha256")
        provenance["current_render_received_full_browser_qa"] = digest(output / "review.html") == prior_viewer_qa.get("review_html_sha256")
        provenance["qa_scope"] = "Earlier full viewer QA applies to the recorded earlier HTML hash. Current prose was regenerated; numerical exports are unchanged."
    for name in ("run.json", "summary.json"):
        source_path = output / "evaluation" / name
        if source_path.is_file():
            provenance["evaluation_" + name.removesuffix(".json") + "_sha256"] = digest(source_path)
    (output / "presentation-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print("Report:", output / "report.md", "; viewer:", output / "review.html", flush=True)


VIEWER = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Causal decoder diagnosis</title><style>
*{box-sizing:border-box}body{margin:0;background:#101924;color:#e4edf4;font:16px/1.5 system-ui}main{max-width:1180px;margin:auto;padding:28px}
h1{font-size:30px;margin:0}h2{font-size:21px}p{color:#bac9d6}section{background:#192737;padding:22px;margin:20px 0;border-radius:12px}
select,button{font:inherit;border:1px solid #50708a;border-radius:6px;background:#0f1b27;color:#e4edf4;padding:8px}label{display:inline-block;margin:6px 12px 6px 0}
select{max-width:100%}audio{width:100%;margin:18px 0}a{color:#77d7be;margin-right:18px}.scroll,.table-scroll{overflow:auto}svg{width:100%;min-width:650px;display:block}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:9px 11px;text-align:right;border-bottom:1px solid #425267;white-space:nowrap}th:first-child,td:first-child{text-align:left}
.warning{border-left:4px solid #dbad67;padding-left:14px;color:#efd3a9}.metrics{font-variant-numeric:tabular-nums}#status{color:#efb37f}#roll,#events{cursor:crosshair}
.tip{font-size:14px}button{cursor:pointer;margin-right:8px}#eventlog{max-height:360px;overflow:auto}#eventlog td{padding:6px 9px}input[type=checkbox]{accent-color:#65d9b0}details summary{cursor:pointer}
pre{white-space:pre-wrap;overflow-wrap:anywhere;color:#c2d5e7;font-size:13px}#clock{font-variant-numeric:tabular-nums;color:#a9e4cf}
</style><main><h1>Causal decoder diagnosis</h1><p>Stage 9 · frozen stage-8 model · __BUDGET__</p>
<p class="warning">The same test excerpts have been inspected in earlier stages. This is a file replay and decoder diagnosis; it does not establish microphone performance or fresh generalization.</p>
<section><h2>What changed?</h2><p>__OUTCOME__</p><p>__DECISION__</p><p>The acoustic checkpoint, frontend, normalization and recurrent context remain frozen. Parameters are selected on validation before replaying the reused test excerpts.</p>__SCORES__
<p class="tip">Onset F1 scores attacks; onset+offset F1 also scores releases; string F1 requires the right guitar string. The offline model remains the default pending a separate quality decision.</p></section>
<section><h2>When did matched notes arrive?</h2>__LATENCY__<p class="tip">Delay is delivery minus reference time, only for correctly matched notes. Misses have no finite delay. Read matched counts beside accuracy. End-of-input flush releases are excluded; a dash means unavailable, not zero.</p></section>
<section><h2>Why is causal accuracy low?</h2><p>Validation-only factorial checks replace one or both predicted heads with reference labels. The oracle values contain privileged information and never enter production inference or test replay.</p>__HYBRID__
__INTERPRETATION__<p><a href="hybrid-head-diagnostics.json">Full factorial diagnostics</a><a href="oracle-diagnostics.json">Oracle decoder checks</a></p>
<details><summary>Raw heads, context audit and selection</summary><h3>Raw acoustic heads</h3>__HEAD__<p>These count frames, frame/string/fret classes or pitch/frame cells. Assembled note F1 counts detected attacks, so the units differ. Lag sweeps and raw runs are descriptive diagnostics.</p>
<p><a href="validation-head-diagnostics.json">Full head diagnostics</a></p><h3>Ground-truth oracle inputs</h3><p>Reference labels replace acoustic predictions only in explicit diagnostic checks. Oracle values never enter production inference or test replay.</p><pre>__ORACLE__</pre>
<h3>Context reset audit</h3>__CONTEXT__<p>__CONTEXT_TEXT__</p><p><a href="validation-context-diagnostics.json">Full context audit</a><a href="validation-context-diagnostics.md">Readable context audit</a></p>
<h3>Validation selection</h3><pre>__SELECTION__</pre></details></section>
<section><h2>Listen and compare</h2><label>Excerpt <select id="clip"></select></label><label>Decoder <select id="mode"></select></label><button id="compare">Switch strict / repair</button>
<audio id="audio" controls preload="metadata"></audio><p id="status" role="status"></p><p id="clock">Delivery clock 0.000 s</p><p id="metrics" class="metrics"></p><p id="timing" class="metrics"></p>
<p><a id="json">Saved notes JSON</a><a id="midi">MIDI</a><a id="csv">String/fret CSV</a><a id="ledger">Saved event ledger</a><a id="metricfile">Saved metrics</a></p>
<h3>Source note intervals</h3><div class="scroll" id="roll"></div><p class="tip">Click to seek. Reference: blue matched, purple missed. Predictions: green matched, red extra. These intervals use estimated source timestamps, which can be backdated. Hover for pitch and timing.</p>
<h3>Source → delivery events</h3><div class="scroll" id="events"></div><p class="tip">Hollow markers show source time; filled markers show actual virtual delivery. Green circles: starts. Orange diamonds: releases. Strings 0–5 run low E to high E.</p>
<label><input type="checkbox" id="showall"> Show events that arrive after the playhead</label><p id="eventcount" class="metrics" aria-live="polite"></p>
<div id="eventlog" class="table-scroll"><table><thead><tr><th>Event</th><th>Pitch</th><th>String/fret</th><th>Source s</th><th>Delivery s</th><th>Delay ms</th></tr></thead><tbody id="eventrows"></tbody></table></div>
<p class="tip">The ledger includes every emitted detection, including false positives. The summary delay table uses matched detections only. The latest 40 available events are shown.</p></section>
<section><details><summary>Protocol and interpretation</summary>__PROTOCOL__</details></section>
<p><a href="report.md">Full report</a><a href="evaluation/metrics.csv">Per-excerpt metrics</a><a href="evaluation/run.json">Run provenance</a><a href="presentation-provenance.json">Presentation provenance</a></p></main>
<script>const data=__DATA__,labels=__LABELS__;const $=id=>document.getElementById(id);let current,jsonUrl,lastEventKey='';
const fmt=(value,n=3)=>value===null||value===undefined||!Number.isFinite(Number(value))?'—':Number(value).toFixed(n);
const clips=[...new Set(data.map(x=>x.id))];clips.forEach(id=>$('clip').add(new Option(id,id)));
function modeMenu(){const old=$('mode').value,choices=data.filter(x=>x.id===$('clip').value);$('mode').replaceChildren();choices.forEach(x=>$('mode').add(new Option(labels[x.mode]||x.mode,x.mode)));if(choices.some(x=>x.mode===old))$('mode').value=old;load();}
function addPlayhead(container,id,height){const svg=$(container).querySelector('svg');if(!svg)return;const line=document.createElementNS('http://www.w3.org/2000/svg','line');line.id=id;line.setAttribute('y1','15');line.setAttribute('y2',height);line.setAttribute('stroke','#fff4b5');line.setAttribute('stroke-width','2');svg.append(line);}
function load(){const previous=current;current=data.find(x=>x.id===$('clip').value&&x.mode===$('mode').value);if(!current)return;const time=previous&&previous.id===current.id?$('audio').currentTime:0;$('audio').pause();
if(!previous||previous.audio!==current.audio){$('audio').src=current.audio;$('audio').onloadedmetadata=()=>{$('audio').currentTime=Math.min(time,Number.isFinite($('audio').duration)?$('audio').duration:time);update();};}else $('audio').currentTime=time;
$('status').textContent='';$('metrics').textContent=`Onset F1 ${fmt(current.onset)} · onset+offset F1 ${fmt(current.offset)} · frame F1 ${fmt(current.frame)} · string F1 ${fmt(current.tab)} · missed ${current.errors.missed_notes??'—'} · extra ${current.errors.extra_notes??'—'}`;
const t=current.timing,l=current.latency;$('timing').textContent=`Processing/audio ${fmt(t.real_time_factor)} · packet processing p50 / p95 ${fmt(t.chunk_processing_p50_ms,1)} / ${fmt(t.chunk_processing_p95_ms,1)} ms · matched onset delivery p50 / p95 ${fmt(l.matched_onset_delivery_p50_ms,1)} / ${fmt(l.matched_onset_delivery_p95_ms,1)} ms`;
['json','midi','csv','ledger','metricfile'].forEach((id,i)=>{$(id).href=current.stem+['.json','.mid','.csv','-events.json','-metrics.json'][i];$(id).download='';});
$('roll').innerHTML=current.svg;$('events').innerHTML=current.eventSvg;addPlayhead('roll','source-playhead',215);addPlayhead('events','delivery-playhead',260);lastEventKey='';update();$('compare').disabled=!data.some(x=>x.id===current.id&&x.mode===(current.mode==='strict'?'repair':'strict'));}
function eventList(){if(!current)return;const all=$('showall').checked,now=$('audio').currentTime,available=current.events.filter(e=>all||(e.delivery!==null&&e.delivery<=now)),shown=available.slice(-40),key=current.id+current.mode+all+available.length;
if(key===lastEventKey)return;lastEventKey=key;$('eventcount').textContent=`${available.length} of ${current.events.length} events ${all?'shown':'delivered by '+fmt(now)+' s'} · latest ${shown.length}`;$('eventrows').replaceChildren();shown.forEach(e=>{const tr=document.createElement('tr'),values=[e.type+(e.flush?' (flush)':''),e.pitch??'—',(e.string??'—')+'/'+(e.fret??'—'),fmt(e.source),fmt(e.delivery),e.source===null||e.delivery===null?'—':fmt(1000*(e.delivery-e.source),1)];values.forEach(value=>{const td=document.createElement('td');td.textContent=value;tr.append(td);});$('eventrows').append(tr);});}
function update(){if(!current)return;const time=$('audio').currentTime;$('clock').textContent='Delivery clock '+fmt(time)+' s';const a=$('source-playhead');if(a){const x=45+Math.min(Math.max(time,0),current.duration)/current.duration*945;a.setAttribute('x1',x);a.setAttribute('x2',x);}const b=$('delivery-playhead'),svg=$('events').querySelector('svg');if(b&&svg){const duration=Number(svg.dataset.duration)||12,x=60+Math.min(Math.max(time,0),duration)/duration*925;b.setAttribute('x1',x);b.setAttribute('x2',x);}eventList();}
function seek(event,container,left,width,duration){const svg=$(container).querySelector('svg');if(!svg||!current)return;const bounds=svg.getBoundingClientRect(),x=(event.clientX-bounds.left)/bounds.width*1000;$('audio').currentTime=Math.min($('audio').duration||duration,Math.max(0,(x-left)/width*duration));update();}
$('clip').onchange=modeMenu;$('mode').onchange=load;$('audio').ontimeupdate=update;$('audio').onseeked=update;$('showall').onchange=()=>{lastEventKey='';eventList();};$('audio').onerror=()=>{$('status').textContent='The local audio file could not be loaded. Keep review.html beside its evaluation folder.';};
$('roll').onclick=e=>seek(e,'roll',45,945,current.duration);$('events').onclick=e=>seek(e,'events',60,925,Number($('events').querySelector('svg')?.dataset.duration)||12);
$('compare').onclick=()=>{const mode=current.mode==='strict'?'repair':'strict';if(data.some(x=>x.id===current.id&&x.mode===mode)){$('mode').value=mode;load();}};
$('json').onclick=e=>{if(current.noteJson===null)return;e.preventDefault();if(jsonUrl)URL.revokeObjectURL(jsonUrl);jsonUrl=URL.createObjectURL(new Blob([current.noteJson],{type:'application/json'}));const link=document.createElement('a');link.href=jsonUrl;link.download=current.id+'-'+current.mode+'.json';document.body.append(link);link.click();link.remove();};window.addEventListener('beforeunload',()=>{if(jsonUrl)URL.revokeObjectURL(jsonUrl);});if(data.length)modeMenu();</script></html>'''
