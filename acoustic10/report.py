"""Portable, validation-only review of the stage-10 acoustic training controls."""

import csv
import hashlib
import html
import json
from pathlib import Path

from baseline.report import roll_svg
from diagnosis9.report import (
    _clean, _fmt, _markdown_table, _mean, _modes, _number, _score, _tab_score,
    _table, _url, VIEWER as PREVIOUS_VIEWER,
)
from streaming.report import _event_svg, _events


LABELS = {"baseline": "Immediate-label control", "delay60": "60 ms delayed-label model"}


def _artifact(output, run, name):
    """Read recorded artifacts; missing values remain visible, never inferred."""
    value = run.get(name.replace("-", "_").removesuffix(".json"))
    if isinstance(value, dict):
        return value
    path = output / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _json_text(value):
    return json.dumps(_clean(value), ensure_ascii=True, indent=2, allow_nan=False)


def _budget_text(training, run):
    updates = _number(training.get("new_optimizer_updates", training.get("optimizer_updates",
                      run.get("new_optimizer_updates", run.get("optimizer_updates")))))
    return (f"{int(updates)} recorded new optimizer updates" if updates is not None
            else "Optimizer counts are recorded in the training artifacts")


def _head_rows(records):
    """Macro frame-head metrics remain separate from assembled-note metrics."""
    rows = []
    for mode, label in LABELS.items():
        heads = [row.get("head_metrics", {}) for row in records if row["mode"] == mode]
        if heads:
            rows.append([label, len(heads),
                         _fmt(_mean(head.get("exact_state", {}).get("f1") for head in heads)),
                         _fmt(_mean(head.get("active_fret_accuracy") for head in heads)),
                         _fmt(_mean(head.get("state_loss") for head in heads))])
    return rows


def _compact_sanity(sanity):
    """Keep the decision and learning evidence visible without full provenance."""
    result = {key: sanity[key] for key in
              ("passed", "optimizer_steps", "blocks_completed", "state_loss_ratio", "coverage") if key in sanity}
    for stage in ("initial", "final"):
        value = sanity.get(stage, {})
        result[stage] = {key: value[key] for key in
                         ("exact_state_f1", "active_fret_accuracy", "state_loss") if key in value}
    criteria = sanity.get("criteria", {})
    result["settings_and_gate"] = {key: criteria[key] for key in
        ("crop_seconds", "dropout", "gain_augmentation", "minimum_active_frames", "minimum_distinct_frets",
         "maximum_state_loss_ratio", "minimum_active_fret_accuracy", "minimum_exact_state_f1") if key in criteria}
    result["fixed_training_crops"] = len(sanity.get("descriptors", []))
    if "training_seconds" in sanity:
        result["training_seconds"] = round(sanity["training_seconds"], 3)
    return result


def _compact_training(training):
    result = {key: training[key] for key in
              ("new_optimizer_updates", "sanity_passed", "sanity_updates", "epochs_per_arm",
               "test_inference_calls", "validation_used_during_training", "reason") if key in training}
    arms = training.get("arms", {})
    result["arms"] = {name: {key: value[key] for key in
        ("optimizer_steps", "selected_step", "target_delay_frames", "common_leading_mask_frames",
         "warmstarted", "early_stopping_used", "warm_started_from_sanity") if key in value}
        for name, value in arms.items()}
    for name, value in arms.items():
        result["arms"][name]["completed_epoch_blocks"] = len(value.get("history", []))
        if "training_seconds" in value:
            result["arms"][name]["training_seconds"] = round(value["training_seconds"], 3)
    # Show each binding once when it is shared; retain differences rather than hiding them.
    bindings = {}
    for key in ("initial_weights_sha256", "crop_schedule_sha256", "gain_schedule_sha256", "common_support_sha256"):
        values = {name: value[key] for name, value in arms.items() if key in value}
        if values:
            bindings[key] = next(iter(values.values())) if len(set(values.values())) == 1 else values
    result["matched_bindings"] = bindings
    return result


def _outcome(modes):
    baseline = _number(modes.get("baseline", {}).get("onset_f1"))
    delayed = _number(modes.get("delay60", {}).get("onset_f1"))
    if baseline is None or delayed is None:
        return "Completed validation results appear below. Missing values are shown as dashes."
    base_head = _number(modes.get("baseline", {}).get("exact_state_f1"))
    delay_head = _number(modes.get("delay60", {}).get("exact_state_f1"))
    head_text = (f"On these validation excerpts, exact string/fret frame F1 changes from {_fmt(base_head)} "
                 f"to {_fmt(delay_head)} ({delay_head-base_head:+.3f}) with delayed labels. "
                 if base_head is not None and delay_head is not None else "On these validation excerpts, ")
    return (head_text + f"Note-onset F1 is {_fmt(baseline)} for immediate labels and "
            f"{_fmt(delayed)} for the 60 ms delayed-label model ({delayed-baseline:+.3f}).")


def write_report(output, records, summary, run):
    """Render saved results without acoustic inference, training or new selection."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    validation = output / "validation"
    plots = validation / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    modes = _modes(records, summary)
    ordered = [mode for mode in LABELS if mode in modes] + [mode for mode in modes if mode not in LABELS]
    headers = ["Model", "Clips", "Onset F1", "Onset+offset F1", "Frame F1", "String F1", "Processing/audio"]
    rows = [[LABELS.get(mode, mode), modes[mode].get("count", 0)] + [
        _fmt(modes[mode].get(key)) for key in
        ("onset_f1", "onset_offset_f1", "frame_f1", "tab_onset_f1", "real_time_factor")]
        for mode in ordered]
    latency_headers = ["Model", "Matched onsets", "Onset p50 / p95 ms", "Matched releases", "Release p50 / p95 ms"]
    latency_rows = [[LABELS.get(mode, mode), value.get("matched_onset_count", 0),
                     _fmt(value.get("matched_onset_delivery_p50_ms"), 1) + " / " +
                     _fmt(value.get("matched_onset_delivery_p95_ms"), 1),
                     value.get("matched_offset_count", 0),
                     _fmt(value.get("matched_offset_delivery_p50_ms"), 1) + " / " +
                     _fmt(value.get("matched_offset_delivery_p95_ms"), 1)]
                    for mode in ordered for value in [modes[mode]]]
    sanity = _artifact(output, run, "sanity.json")
    selection = _artifact(output, run, "selection.json")
    training = _artifact(output, run, "training-summary.json")
    sanity_display = _compact_sanity(sanity)
    training_display = _compact_training(training)
    head_headers = ["Model", "Clips", "Exact string/fret frame F1", "Active-fret accuracy", "State loss"]
    head_rows = _head_rows(records)
    budget = _budget_text(training, run)
    outcome = _outcome(modes)
    decision = ("This stage compares and selects on validation only. These scores are development measurements, "
                "so they do not establish accuracy on unseen songs. The existing product default remains in place.")
    if records and sanity.get("passed") is True:
        decision = ("The training-only sanity check shows that the pipeline can learn the four repeated crops. "
                    "That success does not establish generalization. The paired models still have weak validation "
                    "accuracy; this small development gain does not justify product promotion. The existing "
                    "product default remains in place.")
    gate_failed = sanity.get("passed") is False or training.get("sanity_passed") is False
    if gate_failed:
        outcome = ("The fixed 100-update training-only sanity gate failed. The protocol stopped after those "
                   "100 updates; neither 200-update comparison arm was trained or evaluated.")
        decision = ("Investigate the failed learning/target-alignment check before spending the paired training "
                    "budget. There is no validation model selection from this failed-gate run.")
    design = ("The two matched acoustic models learn from immediate or 60 ms delayed targets. Delayed supervision "
              "does not provide future audio. Original annotation times are preserved for scoring." if not gate_failed
              else "The planned immediate-label and delayed-label models were not run because the training-only sanity gate failed.")
    config = run.get("config", {})
    score_seconds = _number(config.get("score_seconds", config.get("evaluation", {}).get("score_seconds"))) or 10.0
    input_seconds = _number(config.get("input_seconds", config.get("evaluation", {}).get("input_seconds"))) or 12.0
    protocol = [
        "The tiny overfit sanity check uses training examples only. Repeated memorization scores are a plumbing "
        "check, not evidence of generalization. Its checkpoint is not used for the validation comparison.",
        "The matched models start from scratch and have equal update budgets. The intervention delays supervised "
        "targets by 60 ms; it does not grant future audio. Causal features, frame-local normalization and carried "
        "unidirectional recurrent state remain the inference path. Read the configuration for exact matched settings.",
        "Both models use fixed decoder thresholds: onset 0.55, activity 0.35 and internal source backdate 60 ms. "
        "There is no decoder calibration. The development model is selected by mean validation exact-state F1, "
        "then note-onset F1, with the immediate-label control winning exact ties. Selection creates optimism in these "
        "measurements. No test clips or new data are evaluated in this stage; "
        "a separate held-out dataset is needed before a quality claim or product promotion.",
        "References retain their original source times. Supervision delay and decoder source backdating are "
        "different operations. The delayed model maps every public source timestamp back by its declared 60 ms "
        "supervision delay exactly once; delivery times retain the measured processing clock. Edge clamps are "
        "counted explicitly. No reference labels enter inference.",
        "Raw head metrics compare the same original-source 0–10 s span: immediate predictions use frames 0–999 "
        "and delayed predictions use frames 6–1005, against original targets 0–999. Both mask the first six "
        "source frames, so the delayed model cannot receive free inactive-prefix credit. Exact-state F1 requires the correct "
        "string and fret; active-fret accuracy is conditional on reference-active frames. These frame metrics and "
        "assembled-note F1 measure different units.",
        f"Replay receives up to {input_seconds:g} seconds of already loaded, resampled PCM and scores the first "
        f"{score_seconds:g} seconds with later release context. This is file replay, not microphone capture or resampling timing. "
        "Onset tolerance is 50 ms, pitch tolerance 50 cents, and offset tolerance is max(50 ms, 20% of reference duration).",
        "F1 values are macro means across excerpts. String F1 additionally requires the correct reference string. "
        "Onset+offset F1 requires attack and release. Matched-event delay uses virtual delivery completion minus "
        "reference time, including packet waiting and measured processing; missed notes have no finite delay. "
        "End-of-input flush releases are excluded from matched-release delay.",
        "Processing/audio is total measured computation divided by replayed duration. Average throughput below 1 "
        "does not guarantee each packet deadline. The cached probabilities are exact copies from the actual "
        "20 ms decoder pushes, and that recording overhead is included in timing. No extra acoustic inference "
        "is used to obtain head diagnostics. Timing is hardware-dependent; GPU tensor allocation is not total device memory.",
    ]
    if gate_failed:
        protocol = [protocol[0], "The planned paired 200-update arms were not run. No validation or test inference "
                    "was performed by this failed-gate branch."]
    lines = ["# Stage 10: budgeted acoustic-model sanity and label-delay control", "", outcome, "", decision,
             "", budget + ".", "", "## Aligned validation acoustic heads", ""]
    lines += _markdown_table(head_headers, head_rows) if head_rows else ["No paired acoustic-model evaluation was recorded."]
    lines += ["", "## Paired validation note accuracy", ""]
    lines += _markdown_table(headers, rows)
    lines += ["", "## Delivery delay for correctly matched events", ""]
    lines += _markdown_table(latency_headers, latency_rows)
    lines += ["", "A dash means unavailable or no matched events. Read matched counts beside accuracy.",
              "", "## Training-only sanity check", "",
              "This repeated training-example check is separate from the paired validation models.", "",
              "```json", _json_text(sanity_display), "```", "", "[Full sanity artifact](sanity.json)",
              "", "## Validation selection", "", "```json", _json_text(selection), "```",
              "", "## Recorded training budget", "", "```json", _json_text(training_display), "```",
              "", "[Full training summary](training-summary.json) · [Immediate-label receipt](models/baseline/training-receipt.json) · "
              "[Delayed-label receipt](models/delay60/training-receipt.json)",
              "", "## Protocol and limitations", ""]
    for paragraph in protocol:
        lines += [paragraph, ""]
    lines += ["[Interactive comparison](review.html) · [Per-excerpt metrics](validation/metrics.csv) · "
              "[Run provenance](validation/run.json) · [Selection](selection.json) · "
              "[Training summary](training-summary.json) · [Sanity check](sanity.json)", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    csv_headers = ["id", "mode", "onset_f1", "onset_offset_f1", "frame_f1", "tab_onset_f1", "missed_notes", "extra_notes",
                   "processing_seconds", "real_time_factor", "chunk_processing_p50_ms", "chunk_processing_p95_ms",
                   "matched_onset_count", "matched_onset_delivery_p50_ms", "matched_onset_delivery_p95_ms",
                   "matched_offset_count", "matched_offset_delivery_p50_ms", "matched_offset_delivery_p95_ms"]
    with (validation / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(csv_headers)
        for record in records:
            errors = record.get("pitch_metrics", {}).get("errors", {})
            writer.writerow([record["id"], record["mode"], _score(record), _score(record, "onset_offset"),
                             _score(record, "frame"), _tab_score(record), errors.get("missed_notes"), errors.get("extra_notes")] +
                            [record.get("timing", {}).get(key) for key in csv_headers[8:12]] +
                            [record.get("latency", {}).get(key) for key in csv_headers[12:]])
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
    page = page.replace("__OUTCOME__", html.escape(outcome)).replace("__DECISION__", html.escape(decision))
    page = page.replace("__DESIGN__", html.escape(design))
    page = page.replace("__BUDGET__", html.escape(budget)).replace("__SANITY__", html.escape(_json_text(sanity_display)))
    page = page.replace("__TRAINING__", html.escape(_json_text(training_display))).replace("__SELECTION__", html.escape(_json_text(selection)))
    page = page.replace("__PROTOCOL__", "".join("<p>" + html.escape(paragraph) + "</p>" for paragraph in protocol))
    page = page.replace("__DATA__", json.dumps(_clean(payload), ensure_ascii=True, allow_nan=False).replace("</", "<\\/"))
    page = page.replace("__LABELS__", json.dumps(LABELS))
    (output / "review.html").write_text(page, encoding="utf-8")
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    provenance = {"presentation_only": True, "new_optimizer_updates_for_report": 0,
                  "evaluation_run_changed": False, "report_source_sha256": digest(__file__),
                  "rendered_sha256": {name: digest(output / name) for name in ("report.md", "review.html")}}
    for path in (validation / "run.json", validation / "summary.json", output / "sanity.json",
                 output / "selection.json", output / "training-summary.json"):
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


# Share the previously verified playhead/event logic, with a new scope and model menu.
_top = PREVIOUS_VIEWER.split("<main>", 1)[0]
_listen = PREVIOUS_VIEWER.split("<section><h2>Listen and compare</h2>", 1)[1]
_listen = _listen.split("<section><details><summary>Protocol and interpretation</summary>", 1)[0]
_script = PREVIOUS_VIEWER.split("<script>", 1)[1]
VIEWER = (_top.replace("Causal decoder diagnosis", "Budgeted acoustic-model controls") + r'''<main>
<h1>Budgeted acoustic-model controls</h1><p>Stage 10 · __BUDGET__</p>
<p class="warning">Training-only sanity check and validation-only model development. No test clips or fresh data are scored in this stage. The reported comparison uses the same validation set as parameter selection.</p>
<section><h2>What changed?</h2><p>__OUTCOME__</p><p>__DECISION__</p>
<p class="tip">__DESIGN__</p></section>
<section><h2>Aligned acoustic-head comparison</h2>__HEADS__<p class="tip">The primary development comparison is exact string/fret frame F1. Active-fret accuracy is conditional on reference-active frames. Predictions are mapped to the same original 0–10 s source span, with the first six source frames masked in both models.</p></section>
<section><h2>Paired validation note accuracy</h2>__SCORES__<p class="tip">Note-onset F1 measures detected attacks. String F1 additionally requires the correct reference string. These note metrics measure a different unit from the acoustic-head table.</p></section>
<section><h2>Training-only sanity check</h2><p>The sanity model repeatedly learns a tiny training subset. This checks learning and target alignment, and is separate from the models compared below.</p>
<details><summary>Compact sanity result and gate</summary><pre>__SANITY__</pre></details><p><a href="sanity.json">Full sanity artifact</a></p>
<details><summary>Compact training budget and matched bindings</summary><pre>__TRAINING__</pre></details>
<p><a href="training-summary.json">Full training summary</a><a href="models/baseline/training-receipt.json">Immediate-label receipt</a><a href="models/delay60/training-receipt.json">Delayed-label receipt</a></p></section>
<section><h2>When did matched notes arrive?</h2>__LATENCY__<p class="tip">Delay is delivery minus reference time for correctly matched notes only. Misses have no finite delay. End-of-input flush releases are excluded; a dash means unavailable.</p></section>
<section><details><summary>Validation selection</summary><p>Decoder parameters are fixed. Development model selection uses mean exact-state F1, then note-onset F1, on the same validation set reported here. These are development results.</p><pre>__SELECTION__</pre></details></section>
<section><h2>Listen and compare</h2>''' + _listen + r'''
<section><details><summary>Protocol and limitations</summary>__PROTOCOL__</details></section>
<p><a href="report.md">Full report</a><a href="validation/metrics.csv">Per-excerpt metrics</a><a href="validation/run.json">Run provenance</a><a href="presentation-provenance.json">Presentation provenance</a></p></main>
<script>''' + _script)
VIEWER = VIEWER.replace("Decoder <select", "Model <select")
VIEWER = VIEWER.replace("Switch strict / repair", "Switch immediate / delayed labels")
VIEWER = VIEWER.replace("current.mode==='strict'?'repair':'strict'", "current.mode==='baseline'?'delay60':'baseline'")
VIEWER = VIEWER.replace("beside its evaluation folder", "beside its validation folder")
VIEWER = VIEWER.replace('<p id="timing" class="metrics"></p>',
                        '<p id="headmetrics" class="metrics"></p><p id="timing" class="metrics"></p>')
VIEWER = VIEWER.replace("const t=current.timing,l=current.latency;",
                        "const h=current.head;$('headmetrics').textContent=`Exact state F1 ${fmt(h.exact_state?.f1)} · "
                        "active-fret accuracy ${fmt(h.active_fret_accuracy)} · state loss ${fmt(h.state_loss)} · "
                        "reference-active frames ${h.active_frames??'—'}`;const t=current.timing,l=current.latency;")
