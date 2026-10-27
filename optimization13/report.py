"""Read-only, portable stage-13 training-dose comparison."""

import csv
import hashlib
import html
import json
from pathlib import Path

from acoustic10.report import _artifact, _json_text, VIEWER as PREVIOUS_VIEWER
from baseline.report import roll_svg
from diagnosis9.report import (
    _clean, _fmt, _markdown_table, _mean, _modes, _number, _score, _tab_score, _table, _url,
)
from streaming.report import _event_svg, _events


LABELS = {
    "group8_250": "Eight groups · 250 updates (parent)",
    "group8_500": "Eight groups · 500 updates (continued)",
    "group1_250": "One group · 250 updates (parent)",
    "group1_500": "One group · 500 updates (continued)",
}


def _head_value(records, mode, key):
    values = [row.get("head_metrics", {}).get(key) for row in records if row["mode"] == mode]
    return _mean(value.get("f1") if isinstance(value, dict) else value for value in values)


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


def _dose_rows(records, modes):
    rows = []
    for group, label in (("group8", "Eight groups"), ("group1", "One group")):
        parent, continued = group + "_250", group + "_500"
        first = _head_value(records, parent, "exact_state")
        last = _head_value(records, continued, "exact_state")
        first_note = _number(modes.get(parent, {}).get("onset_f1"))
        last_note = _number(modes.get(continued, {}).get("onset_f1"))
        change = lambda before, after: f"{after-before:+.3f}" if before is not None and after is not None else "—"
        rows.append([label, _fmt(first), _fmt(last), change(first, last),
                     _fmt(first_note), _fmt(last_note), change(first_note, last_note)])
    return rows


def _outcome(records, modes):
    parts = []
    for group, label in (("group8", "Eight groups"), ("group1", "One group")):
        first = _head_value(records, group + "_250", "exact_state")
        last = _head_value(records, group + "_500", "exact_state")
        before = _number(modes.get(group + "_250", {}).get("onset_f1"))
        after = _number(modes.get(group + "_500", {}).get("onset_f1"))
        if first is not None and last is not None:
            text = f"{label}: exact string/fret frame F1 changes from {_fmt(first)} at 250 lifetime updates to {_fmt(last)} at 500 ({last-first:+.3f})."
            if before is not None and after is not None:
                text += f" Note-onset F1 changes from {_fmt(before)} to {_fmt(after)} ({after-before:+.3f})."
            parts.append(text)
    return " ".join(parts) or "Recorded validation results appear below. Missing values remain visible as dashes."


def _training_display(training):
    result = {key: training[key] for key in
              ("new_optimizer_updates", "historical_parent_updates", "total_optimizer_update_cap",
               "new_epochs_per_arm", "additional_epochs", "additional_epochs_per_arm", "lifetime_epoch_blocks",
               "lifetime_epoch_blocks_per_model", "final_lifetime_updates_per_model", "test_inference_calls",
               "validation_used_during_training") if key in training}
    result["arms"] = {}
    fields = ("name", "mode", "arm", "new_optimizer_updates", "new_optimizer_steps", "optimizer_steps",
              "historical_parent_updates", "parent_optimizer_steps", "lifetime_optimizer_steps",
              "actual_steps", "actual_epochs", "epochs_completed", "additional_epochs_completed", "actual_optimizer_step_counters",
              "optimizer_parameter_steps", "model_parameters", "training_seconds", "peak_gpu_mib",
              "validation_used_during_training", "parent_checkpoint_sha256", "parent_receipt_sha256",
              "initial_weights_sha256", "crop_schedule_sha256", "crop_descriptors_sha256",
              "normalization_sha256", "cnn_norm_groups", "target_delay_frames", "common_leading_mask_frames",
              "gain_schedule_sha256", "common_support_sha256", "learning_rate", "scheduler_policy",
              "scheduler_used", "optimizer_state_restored", "rng_state_restored", "scaler_state_restored",
              "continuation_seed", "torch_seed", "resume_state_path", "resume_state_sha256")
    for name, receipt in training.get("arms", {}).items():
        result["arms"][name] = {key: receipt[key] for key in fields if key in receipt}
    result["scope"] = ("Two frozen parents each have 250 historical updates. Each continued arm receives "
                       "250 new updates; 500 new updates total. Parent evaluation adds no training. "
                       "RNG and AMP scaler are freshly initialized because their parent states were not archived.")
    return result


def _cost_rows(modes, training):
    rows = []
    for mode, label in LABELS.items():
        group, dose = mode.rsplit("_", 1)
        receipt = training.get("arms", {}).get(group, {})
        continued = dose == "500"
        rows.append([label, "8" if group == "group8" else "1", dose,
                     _fmt(modes.get(mode, {}).get("real_time_factor")),
                     _fmt(receipt.get("training_seconds") if continued else 0.0, 1),
                     _fmt(receipt.get("peak_gpu_mib") if continued else None, 1)])
    return rows


def _parent_reproduction_text(records):
    parts = []
    for mode in ("group8_250", "group1_250"):
        checks = [row.get("parent_reproduction") for row in records if row.get("mode") == mode]
        checks = [item for item in checks if isinstance(item, dict)]
        if checks:
            exact = sum(item.get("probabilities_bit_identical") is True for item in checks)
            parts.append(f"{LABELS[mode]}: {exact}/{len(checks)} probability arrays are bit-identical to the saved parent evaluation.")
    return " ".join(parts) or "Parent reproduction checks are recorded in the per-excerpt metrics when available."


def write_report(output, records, summary, run):
    """Render saved records without inference, training or new model selection."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    validation = output / "validation"
    plots = validation / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    modes = _modes(records, summary)
    ordered = [mode for mode in LABELS if mode in modes] + [mode for mode in modes if mode not in LABELS]
    headers = ["Model", "Clips", "Onset F1", "Onset+offset F1", "Frame F1", "String F1"]
    rows = [[LABELS.get(mode, mode), modes[mode].get("count", 0)] + [
        _fmt(modes[mode].get(key)) for key in ("onset_f1", "onset_offset_f1", "frame_f1", "tab_onset_f1")]
        for mode in ordered]
    head_headers = ["Model", "Clips", "Exact string/fret frame F1", "Active-fret accuracy", "State loss"]
    head_rows = _head_rows(records)
    dose_headers = ["CNN normalization", "State F1 · 250", "State F1 · 500", "Change",
                    "Onset F1 · 250", "Onset F1 · 500", "Change"]
    dose_rows = _dose_rows(records, modes)
    latency_headers = ["Model", "Matched onsets", "Onset p50 / p95 ms", "Matched releases", "Release p50 / p95 ms"]
    latency_rows = [[LABELS.get(mode, mode), value.get("matched_onset_count", 0),
                     _fmt(value.get("matched_onset_delivery_p50_ms"), 1) + " / " + _fmt(value.get("matched_onset_delivery_p95_ms"), 1),
                     value.get("matched_offset_count", 0),
                     _fmt(value.get("matched_offset_delivery_p50_ms"), 1) + " / " + _fmt(value.get("matched_offset_delivery_p95_ms"), 1)]
                    for mode in ordered for value in [modes[mode]]]
    pitch_headers = ["Model", "Auxiliary pitch frame F1", "MIDI 40–51 pitch frame F1"]
    pitch_rows = []
    for mode in ordered:
        heads = [row.get("head_metrics", {}).get("pitch", {}) for row in records if row["mode"] == mode]
        pitch_rows.append([LABELS.get(mode, mode), _fmt(_mean(head.get("overall", {}).get("f1") for head in heads)),
                           _fmt(_mean(head.get("low_register", {}).get("f1") for head in heads))])
    support = []
    support_values = []
    for mode in ordered:
        heads = [row.get("head_metrics", {}).get("pitch", {}).get("low_register", {}) for row in records if row["mode"] == mode]
        if heads and all("reference_count" in head for head in heads):
            support_values.append((sum(head['reference_count'] for head in heads),
                                   sum(head['reference_count'] == 0 for head in heads), len(heads)))
            support.append(f"{LABELS.get(mode, mode)}: {sum(head['reference_count'] for head in heads):,} reference pitch-frame combinations; "
                           f"{sum(head['reference_count'] == 0 for head in heads)}/{len(heads)} clips have no low-band reference.")
    pitch_support = " ".join(support) or "Low-band reference support is unavailable."
    if len(support_values) == len(ordered) and len(set(support_values)) == 1:
        count, empty, clips = support_values[0]
        pitch_support = (f"Shared low-band support across all {len(ordered)} modes: {count:,} reference "
                         f"pitch-frame combinations; {empty}/{clips} clips have no low-band reference.")
    pitch_outcomes = []
    for group, label in (("group8", "Eight groups"), ("group1", "One group")):
        values = []
        for dose in ("250", "500"):
            heads = [row.get("head_metrics", {}).get("pitch", {}).get("low_register", {})
                     for row in records if row["mode"] == group + "_" + dose]
            values.append(_mean(head.get("f1") for head in heads))
        if all(value is not None for value in values):
            before, after = values
            direction = "declines" if after < before else "increases" if after > before else "stays unchanged"
            pitch_outcomes.append(f"{label}: low-register pitch F1 {direction} from {_fmt(before)} to {_fmt(after)}.")
    pitch_outcome = " ".join(pitch_outcomes)
    selection = _artifact(output, run, "selection.json")
    training = _artifact(output, run, "training-summary.json")
    training_display = _training_display(training)
    cost_headers = ["Model", "CNN groups", "Lifetime updates", "Processing/audio", "New training seconds", "New training peak tensor MiB"]
    cost_rows = _cost_rows(modes, training)
    budget = f"{int(training['new_optimizer_updates'])} recorded new optimizer updates" if _number(training.get("new_optimizer_updates")) is not None else "Optimizer counts are recorded in the training artifacts"
    outcome = _outcome(records, modes)
    decision = ("This stage selects between the two 500-update candidates on validation only. The 250-update "
                "parents are diagnostic comparisons. These development results do not establish quality on "
                "unseen songs; the existing product default remains in place.")
    design = ("Continue each verified 250-update parent for 250 additional updates, restoring its model and "
              "AdamW state. Hold the parent's final learning rate at 0.0001. The eight-group and one-group "
              "models retain 931,145 parameters, the same 128 ms past spectral window, training-only feature "
              "statistics, 130 ms supervision delay, loss and fixed decoder.")
    config = run.get("config", {})
    evaluation = config.get("evaluation", {})
    score_seconds = _number(config.get("score_seconds", evaluation.get("score_seconds"))) or 10.0
    input_seconds = _number(config.get("input_seconds", evaluation.get("input_seconds"))) or 12.0
    protocol = [
        _parent_reproduction_text(records),
        "There are four replay modes: each normalization family's immutable 250-update parent and its "
        "500-update continuation. Both continued arms receive exactly 250 new optimizer updates, for 500 "
        "new updates total. The two parents contribute 500 historical updates in total. Their replay adds "
        "zero new training. No test clips, fresh songs or additional tiny-overfit training are used here.",
        "Continuation restores model weights and AdamW moments/counters from the verified parent. The "
        "parent's final learning rate of 0.0001 is held constant, with no scheduler reset. Parent RNG and "
        "AMP scaler states were not archived, so both arms use a fresh common seed-43 training RNG and "
        "AMP scale 1024. This is a specified continuation experiment, not a bit-identical uninterrupted "
        "run. The same fresh crop/gain schedule uses seed 42, epoch indices 5–9, 1000 crop descriptors, "
        "batch size four and two-second crops: five additional epoch blocks, ten lifetime epoch blocks.",
        "Within each family, the controlled intervention adds optimization updates and the corresponding "
        "training-example exposure. Normalization groups, architecture, frontend, targets, feature statistics "
        "and decoder stay fixed. Between families, CNN normalization uses eight channel-frequency groups "
        "or one group separately within each frame. Neither pools across time or batch. Both use a 128 ms "
        "past window, 216 log-frequency bins, a 10 ms hop and zero future-audio lookahead. This does not "
        "separate the effects of extra updates from extra crop exposure, or prove the cause of every acoustic error.",
        "Supervision is delayed by 13 frames (130 ms). The common 26-frame leading mask applies to crop "
        "and validation source support. Acoustic-head scoring uses prediction frames 13–1012 against "
        "original source target frames 0–999, omitting the first 26 source frames in every mode. Exact-state "
        "F1 requires the correct string and fret. Active-fret accuracy is conditional on reference-active "
        "frames. The auxiliary pitch head uses the predeclared 0.5 threshold; MIDI 40–51 is a predeclared "
        "low-register diagnostic. Neither pitch metric selects the model. Macro means include all 12 clips, "
        "including zero F1 when the reference band is empty; reference support counts are shown explicitly.",
        "The decoder stays at onset 0.55, activity 0.35 and internal source backdate 60 ms, with no "
        "calibration. Public source boundaries are mapped back by the common 130 ms supervision delay "
        "exactly once after the internal decision. Delivery timestamps retain the measured processing clock. "
        "Original annotation times are used for scoring; references never enter inference. Edge clamps "
        "are counted explicitly.",
        "Development selection considers only group8_500 and group1_500, by mean validation exact-state "
        "F1, then note-onset F1, with eight groups winning exact ties. The parent modes do not compete for "
        "this stage's selection. The same validation set is used for selection and reporting, so the result "
        "is optimistic and does not constitute a fresh generalization estimate. The small difference between "
        "final candidates comes from one seed and does not establish a universal normalization ranking. An independent dataset is "
        "needed before a quality claim or product promotion.",
        f"Replay receives up to {input_seconds:g} seconds of already loaded, resampled PCM in 20 ms packets "
        f"and scores the first {score_seconds:g} seconds, with later release context. It is file replay; "
        "microphone capture and resampling costs are outside this measurement. Onset tolerance is 50 ms, "
        "pitch tolerance 50 cents and offset tolerance max(50 ms, 20% of reference duration).",
        "F1 values are macro means across excerpts. String F1 additionally requires the correct reference "
        "string. Matched-event delay is virtual delivery completion minus reference time, including packet "
        "waiting and measured processing. Missed notes have no finite delay. End-of-input flush releases "
        "are excluded from matched-release delay. Read matched counts beside accuracy.",
        "Processing/audio is measured computation divided by replayed duration; a value below one does "
        "not guarantee every packet deadline. All four modes are warmed and timed again in this stage. "
        "Cached probabilities copy actual decoder pushes, and recording overhead is included in timing. "
        "Head diagnostics require no extra acoustic inference. Timing is hardware-dependent. "
        "The architecture and inference algorithm are unchanged within each family; timing differences "
        "do not establish an algorithmic speed improvement. Training peak tensor allocation is reported only "
        "for the two new continuations and excludes other device "
        "allocations. The parent modes receive zero new training seconds.",
    ]
    training_links = ("[Full training summary](training-summary.json) · "
                      "[Eight-group parent](models/group8/parent-reference.json) · [One-group parent](models/group1/parent-reference.json) · "
                      "[Eight-group continuation receipt](models/group8/training-receipt.json) · [One-group continuation receipt](models/group1/training-receipt.json)")
    lines = ["# Stage 13: causal optimization-dose continuation", "", outcome, "", decision, "", budget + ".",
             "", "## Paired effect of 250 additional updates", ""] + _markdown_table(dose_headers, dose_rows)
    lines += ["", "## Aligned validation acoustic heads", ""] + _markdown_table(head_headers, head_rows)
    lines += ["", "## Auxiliary pitch diagnostics", ""] + _markdown_table(pitch_headers, pitch_rows)
    lines += ["", pitch_support, "", pitch_outcome,
              "", "Pitch diagnostics do not select the model. Macro means include zero F1 for an empty reference band."]
    lines += ["", "## Validation note accuracy", ""] + _markdown_table(headers, rows)
    lines += ["", "## Lifetime dose and current measured cost", ""] + _markdown_table(cost_headers, cost_rows)
    lines += ["", "Parent replay receives zero new training seconds. All four inference measurements are current. A dash is unavailable.",
              "", "## Delivery delay for correctly matched events", ""] + _markdown_table(latency_headers, latency_rows)
    lines += ["", "A dash means unavailable or no matched events.", "", "## Final-candidate validation selection", "",
              "```json", _json_text(selection), "```", "", "## Recorded continuation budget", "", "```json",
              _json_text(training_display), "```", "", training_links, "", "## Protocol and limitations", ""]
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
        svg = roll_svg(reference, notes, metrics.get("onset_matches", []), duration, clip_id + " · " + LABELS.get(mode, mode))
        events = _events(record)
        event_svg = _event_svg(events, _number(record.get("timing", {}).get("input_seconds")) or input_seconds, clip_id + " · " + mode)
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
                        "tab": _tab_score(record), "errors": metrics.get("errors", {}), "head": record.get("head_metrics", {}),
                        "latency": record.get("latency", {}), "timing": record.get("timing", {}),
                        "stem": _url(stem), "noteJson": serialized})
    payload.sort(key=lambda item: (item["id"], ordered.index(item["mode"])))
    substitutions = {"__SCORES__": _table(headers, rows), "__LATENCY__": _table(latency_headers, latency_rows),
                     "__HEADS__": _table(head_headers, head_rows), "__DOSE__": _table(dose_headers, dose_rows),
                     "__COST__": _table(cost_headers, cost_rows), "__PITCH__": _table(pitch_headers, pitch_rows),
                     "__PITCHSUPPORT__": html.escape(pitch_support), "__PITCHOUTCOME__": html.escape(pitch_outcome), "__OUTCOME__": html.escape(outcome),
                     "__DECISION__": html.escape(decision), "__DESIGN__": html.escape(design), "__BUDGET__": html.escape(budget),
                     "__TRAINING__": html.escape(_json_text(training_display)), "__SELECTION__": html.escape(_json_text(selection)),
                     "__PROTOCOL__": "".join("<p>" + html.escape(paragraph) + "</p>" for paragraph in protocol),
                     "__DATA__": json.dumps(_clean(payload), ensure_ascii=True, allow_nan=False).replace("</", "<\\/"),
                     "__LABELS__": json.dumps(LABELS)}
    page = VIEWER
    for key, value in substitutions.items():
        page = page.replace(key, value)
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
        provenance["viewer_qa"] = {key: qa.get(key) for key in ("passed", "completed_at", "review_html_sha256", "combinations", "export_link_checks", "note_json_download_checks", "virtual_delivery_event_checks")}
        provenance["viewer_qa"]["validation_json_sha256"] = digest(qa_path)
        provenance["viewer_qa_scope"] = "Browser QA verified this exact rendered HTML." if current_verified else "Saved browser QA applies to its recorded HTML hash; this render requires verification."
    (output / "presentation-provenance.json").write_text(_json_text(provenance) + "\n", encoding="utf-8")
    print("Report:", output / "report.md", "; viewer:", output / "review.html", flush=True)


_top = PREVIOUS_VIEWER.split("<main>", 1)[0]
_listen = PREVIOUS_VIEWER.split("<section><h2>Listen and compare</h2>", 1)[1]
_listen = _listen.split("<section><details><summary>Protocol and limitations</summary>", 1)[0]
_script = PREVIOUS_VIEWER.split("<script>", 1)[1]
VIEWER = (_top.replace("Budgeted acoustic-model controls", "Causal optimization-dose continuation") + r'''<main>
<h1>Causal optimization-dose continuation</h1><p>Stage 13 · __BUDGET__</p>
<p class="warning">Validation-only development. No test clips or fresh data are scored. Final model selection and reporting use the same validation set.</p>
<section><h2>What changed?</h2><p>__OUTCOME__</p><p>__DECISION__</p><p class="tip">__DESIGN__</p></section>
<section><h2>Did another 250 updates help?</h2>__DOSE__<p class="tip">Each row compares a frozen parent with its own continuation. Positive and negative changes are reported. Only the two 500-update candidates compete in the final development selection.</p></section>
<section><h2>Aligned acoustic-head comparison</h2>__HEADS__<p class="tip">Exact string/fret frame F1 is the primary selection metric. Active-fret accuracy is conditional on reference-active frames. Every mode scores the same original 0–10 s source span with its first 26 frames masked.</p></section>
<section><h2>Auxiliary pitch diagnostics</h2>__PITCH__<p>__PITCHSUPPORT__</p><p>__PITCHOUTCOME__</p><p class="tip">The separate pitch head uses threshold 0.5. MIDI 40–51 is a predeclared diagnostic. Neither pitch metric selects the model. Macro means include zero F1 for an empty reference band.</p></section>
<section><h2>Validation note accuracy</h2>__SCORES__<p class="tip">Note F1 measures a different unit from acoustic-head F1. String F1 additionally requires the correct reference string.</p></section>
<section><h2>Lifetime dose and current measured cost</h2>__COST__<p class="tip">Parent replay adds zero training. All four processing/audio measurements are current. New training peak tensor allocation covers the continuations only and excludes other device allocations. A dash is unavailable.</p></section>
<section><h2>When did matched notes arrive?</h2>__LATENCY__<p class="tip">Delay is virtual delivery minus reference time for correctly matched notes. Misses have no finite delay. End-of-input flush releases are excluded.</p></section>
<section><details><summary>Final-candidate selection and continuation budget</summary><p>Selection considers the two 500-update candidates only, by exact-state F1 then onset F1, with eight groups winning exact ties. Both 250-update parents remain frozen. Continuations restore AdamW state and hold the final learning rate at 0.0001. RNG and AMP scaler states are fresh, so this is not bit-identical uninterrupted training.</p><pre>__SELECTION__</pre><details><summary>Compact continuation budget</summary><pre>__TRAINING__</pre></details>
<p><a href="training-summary.json">Full training summary</a><a href="models/group8/parent-reference.json">Eight-group parent</a><a href="models/group1/parent-reference.json">One-group parent</a><a href="models/group8/training-receipt.json">Eight-group continuation receipt</a><a href="models/group1/training-receipt.json">One-group continuation receipt</a></p></details></section>
<section><h2>Listen and compare</h2>''' + _listen + r'''
<section><details><summary>Protocol and limitations</summary>__PROTOCOL__</details></section>
<p><a href="report.md">Full report</a><a href="validation/metrics.csv">Per-excerpt metrics</a><a href="validation/run.json">Run provenance</a><a href="presentation-provenance.json">Presentation provenance</a></p></main>
<script>''' + _script)
VIEWER = VIEWER.replace("Switch immediate / delayed labels", "Switch 250 / 500 updates in this family")
VIEWER = VIEWER.replace("current.mode==='baseline'?'delay60':'baseline'", "current.mode.replace(/_(250|500)$/,(_,dose)=>dose==='250'?'_500':'_250')")
