"""Portable, validation-only review of the stage-11 causal spectral comparison."""

import csv
import hashlib
import html
import json
from pathlib import Path

from acoustic10.report import _artifact, _budget_text, _compact_training, _json_text
from baseline.report import roll_svg
from diagnosis9.report import (
    _clean, _fmt, _markdown_table, _mean, _modes, _number, _score, _tab_score, _table, _url,
)
from streaming.report import _event_svg, _events
from acoustic10.report import VIEWER as PREVIOUS_VIEWER


LABELS = {"short128": "128 ms past window", "long256": "256 ms past window"}


def _head_rows(records):
    rows = []
    for mode, label in LABELS.items():
        heads = [row.get("head_metrics", {}) for row in records if row["mode"] == mode]
        if heads:
            rows.append([label, len(heads),
                         _fmt(_mean(head.get("exact_state", {}).get("f1") for head in heads)),
                         _fmt(_mean(head.get("active_fret_accuracy") for head in heads)),
                         _fmt(_mean(head.get("state_loss") for head in heads))])
    return rows


def _head_value(records, mode, key):
    values = [row.get("head_metrics", {}).get(key) for row in records if row["mode"] == mode]
    return _mean(value.get("f1") if isinstance(value, dict) else value for value in values)


def _pitch_rows(records):
    rows = []
    for mode, label in LABELS.items():
        heads = [row.get("head_metrics", {}).get("pitch", {}) for row in records if row["mode"] == mode]
        if heads:
            rows.append([label,
                         _fmt(_mean(head.get("overall", {}).get("f1") for head in heads)),
                         _fmt(_mean(head.get("low_register", {}).get("f1") for head in heads))])
    return rows


def _pitch_support(records):
    values = {}
    for mode, label in LABELS.items():
        heads = [row.get("head_metrics", {}).get("pitch", {}).get("low_register", {})
                 for row in records if row["mode"] == mode]
        if heads and all("reference_count" in head for head in heads):
            values[label] = (sum(head["reference_count"] for head in heads),
                             sum(head["reference_count"] == 0 for head in heads), len(heads))
    def sentence(label, value):
        count, empty, clips = value
        return f"{label}: {count:,} reference pitch-frame combinations; {empty}/{clips} clips have no low-band reference."
    if len(values) == 2 and len(set(values.values())) == 1:
        return sentence("Shared low-band support", next(iter(values.values())))
    return " ".join(sentence(label, value) for label, value in values.items()) or "Low-band reference support is unavailable."


def _training_display(training):
    result = _compact_training(training)
    result["new_sanity_updates"] = training.get("new_sanity_updates")
    prior = training.get("sanity_reference", {})
    result["prior_training_only_sanity"] = {key: prior[key] for key in
        ("passed", "prior_optimizer_updates", "new_optimizer_updates", "final_exact_state_f1",
         "final_active_fret_accuracy", "scope") if key in prior}
    for name, receipt in training.get("arms", {}).items():
        row = result["arms"][name]
        row.update({key: receipt[key] for key in
                    ("actual_optimizer_step_counters", "optimizer_parameter_steps", "model_parameters",
                     "peak_gpu_mib", "validation_used_during_training") if key in receipt})
    return result


def _outcome(records, modes):
    short_head = _head_value(records, "short128", "exact_state")
    long_head = _head_value(records, "long256", "exact_state")
    short_note = _number(modes.get("short128", {}).get("onset_f1"))
    long_note = _number(modes.get("long256", {}).get("onset_f1"))
    parts = []
    if short_head is not None and long_head is not None:
        parts.append(f"On these validation excerpts, exact string/fret frame F1 is {_fmt(short_head)} "
                     f"with the 128 ms past window and {_fmt(long_head)} with the 256 ms past window "
                     f"({long_head-short_head:+.3f}).")
    if short_note is not None and long_note is not None:
        parts.append(f"Note-onset F1 is {_fmt(short_note)} and {_fmt(long_note)}, respectively "
                     f"({long_note-short_note:+.3f}).")
    return " ".join(parts) or "Recorded validation results appear below. Missing values remain visible as dashes."


def _cost_rows(modes, training):
    rows = []
    for mode, label in LABELS.items():
        value = modes.get(mode, {})
        receipt = training.get("arms", {}).get(mode, {})
        rows.append([label, "128" if mode == "short128" else "256", "64" if mode == "short128" else "128",
                     _fmt(value.get("real_time_factor")),
                     _fmt(receipt.get("training_seconds"), 1),
                     _fmt(receipt.get("peak_gpu_mib"), 1)])
    return rows


def write_report(output, records, summary, run):
    """Render saved results without acoustic inference, training or new selection."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    validation = output / "validation"
    plots = validation / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    modes = _modes(records, summary)
    ordered = [mode for mode in LABELS if mode in modes] + [mode for mode in modes if mode not in LABELS]
    headers = ["Model", "Clips", "Onset F1", "Onset+offset F1", "Frame F1", "String F1"]
    rows = [[LABELS.get(mode, mode), modes[mode].get("count", 0)] + [
        _fmt(modes[mode].get(key)) for key in
        ("onset_f1", "onset_offset_f1", "frame_f1", "tab_onset_f1")]
        for mode in ordered]
    latency_headers = ["Model", "Matched onsets", "Onset p50 / p95 ms", "Matched releases", "Release p50 / p95 ms"]
    latency_rows = [[LABELS.get(mode, mode), value.get("matched_onset_count", 0),
                     _fmt(value.get("matched_onset_delivery_p50_ms"), 1) + " / " +
                     _fmt(value.get("matched_onset_delivery_p95_ms"), 1),
                     value.get("matched_offset_count", 0),
                     _fmt(value.get("matched_offset_delivery_p50_ms"), 1) + " / " +
                     _fmt(value.get("matched_offset_delivery_p95_ms"), 1)]
                    for mode in ordered for value in [modes[mode]]]
    selection = _artifact(output, run, "selection.json")
    training = _artifact(output, run, "training-summary.json")
    training_display = _training_display(training)
    head_headers = ["Model", "Clips", "Exact string/fret frame F1", "Active-fret accuracy", "State loss"]
    head_rows = _head_rows(records)
    pitch_headers = ["Model", "Auxiliary pitch frame F1", "MIDI 40–51 pitch frame F1"]
    pitch_rows = _pitch_rows(records)
    pitch_support = _pitch_support(records)
    cost_headers = ["Model", "Past window ms", "Window midpoint age ms", "Processing/audio", "Training seconds", "Peak tensor MiB"]
    cost_rows = _cost_rows(modes, training)
    budget = _budget_text(training, run)
    outcome = _outcome(records, modes)
    decision = ("This stage selects on validation only. Accuracy remains weak, and the comparison does not "
                "establish quality on unseen songs. The existing product default remains in place.")
    design = ("The controlled change doubles the past-only spectral window and FFT from 2048 samples "
              "(128 ms) to 4096 samples (256 ms). Both retain 216 log-frequency bins, a 10 ms hop and the "
              "same causal CNN-GRU. This changes spectral resolution, temporal smearing and context age. "
              "The 130 ms supervision delay and decoder settings are held fixed.")
    config = run.get("config", {})
    evaluation = config.get("evaluation", {})
    score_seconds = _number(config.get("score_seconds", evaluation.get("score_seconds"))) or 10.0
    input_seconds = _number(config.get("input_seconds", evaluation.get("input_seconds"))) or 12.0
    protocol = [
        "The two acoustic models start from the same fresh seed-42 weights, with 250 optimizer updates "
        "per model over five epoch blocks. They share the same 1000 crop descriptors and gain schedule. "
        "Normalization is estimated independently from training features for each frontend. The passed "
        "stage-10 tiny overfit check is retained as prior provenance; it adds no new optimizer updates and "
        "does not initialize either model.",
        "Both frontends use only samples already received. The larger window improves nominal spectral "
        "resolution while using older audio context; it adds no future-audio lookahead. The 64/128 ms "
        "window midpoint ages describe context, not measured delivery delay. This matched comparison "
        "holds label delay fixed, so it does not estimate each frontend's optimal latency or label alignment.",
        "Supervision is delayed by 13 frames (130 ms) in both arms. The same 26-frame leading mask applies "
        "to training prediction support, crop edges and validation source support. Acoustic-head scoring "
        "uses prediction frames 13–1012 against original source target frames 0–999, omitting the first "
        "26 source frames in both arms. Exact-state F1 requires the correct string and fret; active-fret "
        "accuracy is conditional on reference-active frames. Note F1 measures a different unit. The "
        "auxiliary pitch head is scored at the predeclared 0.5 threshold. The MIDI 40–51 low-register "
        "band is a predeclared diagnostic, not a selection criterion. These are macro means across all "
        "12 clips, including zero F1 for clips with no low-band reference pitch-frames. The low-band "
        "support counts are shown explicitly; a higher score does not establish a general bass-register improvement.",
        "The decoder is fixed at onset 0.55, activity 0.35 and internal source backdate 60 ms. There is no "
        "decoder calibration. Each public source boundary is mapped back by the common 130 ms "
        "supervision delay exactly once, after the internal decision. Delivery timestamps retain the "
        "measured processing clock. Original annotation times are used for note scoring; references never "
        "enter inference. Edge clamps are counted explicitly.",
        "Development selection uses mean validation exact-state F1, then note-onset F1, with the 128 ms "
        "control winning exact ties. The same validation set is reported here, so selection introduces "
        "optimism. No test clips or new data are evaluated in this stage; a separate held-out dataset is "
        "needed before a quality claim or product promotion.",
        f"Replay receives up to {input_seconds:g} seconds of already loaded, resampled PCM in 20 ms packets "
        f"and scores the first {score_seconds:g} seconds, with later release context. This is file replay, "
        "not microphone capture or resampling timing. Onset tolerance is 50 ms, pitch tolerance 50 cents, "
        "and offset tolerance is max(50 ms, 20% of reference duration).",
        "F1 values are macro means across excerpts. String F1 additionally requires the correct reference "
        "string. Matched-event delay is virtual delivery completion minus reference time, including packet "
        "waiting and measured processing. Missed notes have no finite delay. End-of-input flush releases "
        "are excluded from matched-release delay. Read matched counts beside accuracy.",
        "Processing/audio is total measured computation divided by replayed duration. A value below 1 "
        "does not guarantee each packet deadline. Cached probabilities are exact copies from actual "
        "decoder pushes, and recording overhead is included in timing. No extra acoustic inference is "
        "used for head diagnostics. Timing is hardware-dependent; GPU tensor allocation is not total "
        "device memory. This small run compares frontends under one training and timing protocol; it "
        "does not isolate the source of all remaining transcription errors.",
    ]
    lines = ["# Stage 11: causal spectral-window comparison", "", outcome, "", decision, "", budget + ".",
             "", "## Aligned validation acoustic heads", ""]
    lines += _markdown_table(head_headers, head_rows) if head_rows else ["No paired acoustic-model evaluation was recorded."]
    lines += ["", "## Auxiliary pitch diagnostics", ""] + _markdown_table(pitch_headers, pitch_rows)
    lines += ["", pitch_support]
    lines += ["", "Pitch probabilities use the fixed 0.5 threshold. MIDI 40–51 is a predeclared low-register band; neither pitch diagnostic selects the model. Macro means include all 12 clips, with zero F1 when the reference band is empty. Support counts refer to pitch-frame combinations."]
    lines += ["", "## Paired validation note accuracy", ""] + _markdown_table(headers, rows)
    lines += ["", "## Context and measured cost", ""] + _markdown_table(cost_headers, cost_rows)
    lines += ["", "Window midpoint age is context age, not delivery delay. Peak tensor allocation excludes other device allocations.",
              "", "## Delivery delay for correctly matched events", ""] + _markdown_table(latency_headers, latency_rows)
    lines += ["", "A dash means unavailable or no matched events.", "", "## Validation selection", "",
              "```json", _json_text(selection), "```", "", "## Matched training budget", "",
              "```json", _json_text(training_display), "```", "",
              "[Full training summary](training-summary.json) · [128 ms receipt](models/short128/training-receipt.json) · "
              "[256 ms receipt](models/long256/training-receipt.json)", "", "## Protocol and limitations", ""]
    for paragraph in protocol:
        lines += [paragraph, ""]
    lines += ["[Interactive comparison](review.html) · [Per-excerpt metrics](validation/metrics.csv) · "
              "[Run provenance](validation/run.json) · [Selection](selection.json)", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    csv_headers = ["id", "mode", "onset_f1", "onset_offset_f1", "frame_f1", "tab_onset_f1", "missed_notes", "extra_notes",
                   "processing_seconds", "real_time_factor", "chunk_processing_p50_ms", "chunk_processing_p95_ms",
                   "matched_onset_count", "matched_onset_delivery_p50_ms", "matched_onset_delivery_p95_ms",
                   "matched_offset_count", "matched_offset_delivery_p50_ms", "matched_offset_delivery_p95_ms",
                   "exact_state_f1", "active_fret_accuracy", "state_loss"]
    with (validation / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(csv_headers)
        for record in records:
            errors = record.get("pitch_metrics", {}).get("errors", {})
            head = record.get("head_metrics", {})
            writer.writerow([record["id"], record["mode"], _score(record), _score(record, "onset_offset"),
                             _score(record, "frame"), _tab_score(record), errors.get("missed_notes"), errors.get("extra_notes")] +
                            [record.get("timing", {}).get(key) for key in csv_headers[8:12]] +
                            [record.get("latency", {}).get(key) for key in csv_headers[12:18]] +
                            [head.get("exact_state", {}).get("f1"), head.get("active_fret_accuracy"), head.get("state_loss")])
    payload = []
    for record in records:
        mode, clip_id = record["mode"], record["id"]
        metrics = record.get("pitch_metrics", {})
        notes = record.get("notes", metrics.get("predicted", []))
        reference = record.get("reference", metrics.get("reference", []))
        duration = _number(metrics.get("score_duration")) or score_seconds
        svg = roll_svg(reference, notes, metrics.get("onset_matches", []), duration,
                       clip_id + " · " + LABELS.get(mode, mode))
        events = _events(record)
        event_svg = _event_svg(events, _number(record.get("timing", {}).get("input_seconds")) or input_seconds,
                               clip_id + " · " + mode)
        stem = Path("validation/results") / mode / clip_id
        source = output / stem.with_suffix(".json")
        serialized = source.read_text(encoding="utf-8") if source.is_file() else None
        plot_name = mode + "__" + clip_id
        (plots / (plot_name + "-notes.svg")).write_text(svg, encoding="utf-8")
        (plots / (plot_name + "-events.svg")).write_text(event_svg, encoding="utf-8")
        payload.append({"id": clip_id, "mode": mode,
                        "audio": _url(record.get("relativeaudio", "validation/clips/" + clip_id + ".wav")),
                        "svg": svg, "eventSvg": event_svg, "duration": duration, "events": events,
                        "onset": _score(record), "offset": _score(record, "onset_offset"), "frame": _score(record, "frame"),
                        "tab": _tab_score(record), "errors": metrics.get("errors", {}),
                        "head": record.get("head_metrics", {}),
                        "latency": record.get("latency", {}), "timing": record.get("timing", {}),
                        "stem": _url(stem), "noteJson": serialized})
    payload.sort(key=lambda item: (item["id"], ordered.index(item["mode"])))
    page = VIEWER.replace("__SCORES__", _table(headers, rows)).replace("__LATENCY__", _table(latency_headers, latency_rows))
    page = page.replace("__HEADS__", _table(head_headers, head_rows) if head_rows else "<p>No paired acoustic-model evaluation was recorded.</p>")
    page = page.replace("__COST__", _table(cost_headers, cost_rows))
    page = page.replace("__PITCH__", _table(pitch_headers, pitch_rows))
    page = page.replace("__PITCHSUPPORT__", html.escape(pitch_support))
    page = page.replace("__OUTCOME__", html.escape(outcome)).replace("__DECISION__", html.escape(decision))
    page = page.replace("__DESIGN__", html.escape(design)).replace("__BUDGET__", html.escape(budget))
    page = page.replace("__TRAINING__", html.escape(_json_text(training_display))).replace("__SELECTION__", html.escape(_json_text(selection)))
    page = page.replace("__PROTOCOL__", "".join("<p>" + html.escape(paragraph) + "</p>" for paragraph in protocol))
    page = page.replace("__DATA__", json.dumps(_clean(payload), ensure_ascii=True, allow_nan=False).replace("</", "<\\/"))
    page = page.replace("__LABELS__", json.dumps(LABELS))
    (output / "review.html").write_text(page, encoding="utf-8")
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    provenance = {"presentation_only": True, "new_optimizer_updates_for_report": 0,
                  "evaluation_run_changed": False, "report_source_sha256": digest(__file__),
                  "rendered_sha256": {name: digest(output / name) for name in ("report.md", "review.html")}}
    for path in (validation / "run.json", validation / "summary.json", output / "selection.json",
                 output / "training-summary.json", output / "cache-audit.json"):
        if path.is_file():
            provenance[str(path.relative_to(output)).replace("\\", "/") + "_sha256"] = digest(path)
    qa_path = output / "viewer-validation.json"
    if qa_path.is_file():
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        current_verified = qa.get("passed") is True and qa.get("review_html_sha256") == digest(output / "review.html")
        provenance["current_render_received_full_browser_qa"] = current_verified
        provenance["viewer_qa"] = {key: qa.get(key) for key in
            ("passed", "completed_at", "review_html_sha256", "combinations", "export_link_checks",
             "note_json_download_checks", "virtual_delivery_event_checks")}
        provenance["viewer_qa"]["validation_json_sha256"] = digest(qa_path)
        provenance["viewer_qa_scope"] = ("Browser QA verified this exact rendered HTML." if current_verified
                                         else "Saved browser QA applies to its recorded HTML hash; this render requires verification.")
    (output / "presentation-provenance.json").write_text(_json_text(provenance) + "\n", encoding="utf-8")
    print("Report:", output / "report.md", "; viewer:", output / "review.html", flush=True)


_top = PREVIOUS_VIEWER.split("<main>", 1)[0]
_listen = PREVIOUS_VIEWER.split("<section><h2>Listen and compare</h2>", 1)[1]
_listen = _listen.split("<section><details><summary>Protocol and limitations</summary>", 1)[0]
_script = PREVIOUS_VIEWER.split("<script>", 1)[1]
VIEWER = (_top.replace("Budgeted acoustic-model controls", "Causal spectral-window comparison") + r'''<main>
<h1>Causal spectral-window comparison</h1><p>Stage 11 · __BUDGET__</p>
<p class="warning">Validation-only model development. No test clips or fresh data are scored in this stage. The reported comparison uses the same validation set as model selection.</p>
<section><h2>What changed?</h2><p>__OUTCOME__</p><p>__DECISION__</p><p class="tip">__DESIGN__</p></section>
<section><h2>Aligned acoustic-head comparison</h2>__HEADS__<p class="tip">The primary development comparison is exact string/fret frame F1. Active-fret accuracy is conditional on reference-active frames. Both models use the same original 0–10 s source span and mask its first 26 source frames.</p></section>
<section><h2>Auxiliary pitch diagnostics</h2>__PITCH__<p>__PITCHSUPPORT__</p><p class="tip">The separate pitch head uses the fixed 0.5 probability threshold. MIDI 40–51 is a predeclared low-register diagnostic; neither pitch diagnostic selects the model. Macro F1 includes all 12 clips, with zero F1 for an empty reference band.</p></section>
<section><h2>Paired validation note accuracy</h2>__SCORES__<p class="tip">Note-onset F1 measures detected attacks. String F1 additionally requires the correct reference string. These note metrics measure a different unit from the acoustic-head table.</p></section>
<section><h2>Context and measured cost</h2>__COST__<p class="tip">Past-window midpoint age describes audio context, not measured delivery delay. Both frontends have zero future-audio lookahead. Processing/audio below 1 does not guarantee every packet deadline.</p></section>
<section><h2>When did matched notes arrive?</h2>__LATENCY__<p class="tip">Delay is delivery minus reference time for correctly matched notes only. Misses have no finite delay. End-of-input flush releases are excluded; a dash means unavailable.</p></section>
<section><details><summary>Validation selection and matched training</summary><p>Each frontend receives 250 fresh training updates. Selection uses mean exact-state F1, then note-onset F1, on the same validation set reported here. Decoder and label delay are fixed.</p><pre>__SELECTION__</pre>
<details><summary>Compact training budget and matched bindings</summary><pre>__TRAINING__</pre></details>
<p><a href="training-summary.json">Full training summary</a><a href="models/short128/training-receipt.json">128 ms receipt</a><a href="models/long256/training-receipt.json">256 ms receipt</a></p></details></section>
<section><h2>Listen and compare</h2>''' + _listen + r'''
<section><details><summary>Protocol and limitations</summary>__PROTOCOL__</details></section>
<p><a href="report.md">Full report</a><a href="validation/metrics.csv">Per-excerpt metrics</a><a href="validation/run.json">Run provenance</a><a href="presentation-provenance.json">Presentation provenance</a></p></main>
<script>''' + _script)
VIEWER = VIEWER.replace("Switch immediate / delayed labels", "Switch 128 / 256 ms windows")
VIEWER = VIEWER.replace("current.mode==='baseline'?'delay60':'baseline'", "current.mode==='short128'?'long256':'short128'")
