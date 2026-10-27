"""Independent saved-artifact checks for matched continuation dose.

Audits restore receipts, optimizer tensors and CPU probability ledgers. They
perform no acoustic inference, audio preparation or optimizer updates.
"""

from __future__ import annotations

from collections import Counter
import copy
import csv
import hashlib
import json
import math
from pathlib import Path

from acoustic10.audit import (_check_inputs, _equal, _fingerprint, _integer,
                              _number, _read, _relative, _require,
                              _source_bindings, audit_note_ledger, weight_digest)
from normalization12.audit import without_delivery_clocks
from spectral11.audit import aligned_heads, aligned_pitch, validate_normalization


ARMS = ("group8", "group1")
MODES = ("group8_250", "group8_500", "group1_250", "group1_500")


def optimizer_digest(state):
    """Canonical hash of all optimizer hyperparameters, moments and counters."""
    import torch

    digest = hashlib.sha256()

    def feed(value):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            header = json.dumps(["tensor", str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode()
            payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            for item in (header, payload):
                digest.update(len(item).to_bytes(8, "big"))
                digest.update(item)
        elif isinstance(value, dict):
            digest.update(b"dict")
            for key in sorted(value, key=lambda item: (type(item).__name__, str(item))):
                feed(key)
                feed(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(b"list" if isinstance(value, list) else b"tuple")
            digest.update(len(value).to_bytes(8, "big"))
            for item in value:
                feed(item)
        else:
            _require(value is None or isinstance(value, (str, bool, int, float)), "Unsupported optimizer metadata")
            _require(not isinstance(value, float) or math.isfinite(value), "Nonfinite optimizer metadata")
            item = json.dumps([type(value).__name__, value], separators=(",", ":")).encode()
            digest.update(len(item).to_bytes(8, "big"))
            digest.update(item)

    _require(isinstance(state, dict) and set(state) == {"state", "param_groups"}, "Invalid optimizer state dictionary")
    feed(state)
    return digest.hexdigest()


def optimizer_counters(checkpoint):
    """Read actual AdamW parameter counters and validate moment tensors."""
    import torch

    optimizer = checkpoint["optimizer"]
    states = optimizer["state"]
    _require(isinstance(states, dict) and bool(states), "Optimizer parameter states are missing")
    group_ids = [index for group in optimizer["param_groups"] for index in group["params"]]
    _require(len(group_ids) == len(set(group_ids)) == len(states) and set(group_ids) == set(states),
             "Optimizer parameter IDs are duplicated or missing")
    steps = []
    for state in states.values():
        value = state.get("step")
        _require(isinstance(value, torch.Tensor) and value.numel() == 1 and bool(torch.isfinite(value)),
                 "Invalid saved optimizer step counter")
        numeric = float(value.item())
        _require(numeric.is_integer() and numeric >= 0, "Saved optimizer step must be a nonnegative integer")
        steps.append(int(numeric))
        mean, square = state.get("exp_avg"), state.get("exp_avg_sq")
        _require(isinstance(mean, torch.Tensor) and isinstance(square, torch.Tensor)
                 and mean.shape == square.shape and mean.dtype == square.dtype
                 and bool(torch.isfinite(mean).all()) and bool(torch.isfinite(square).all())
                 and bool((square >= 0).all()), "Invalid AdamW moment tensors")
    return {"minimum": min(steps), "maximum": max(steps), "parameter_states": len(steps)}


def validate_budget(receipts, require_complete=True):
    """Count new updates separately from each frozen parent's 250 updates."""
    _require(isinstance(receipts, dict) and bool(receipts) and set(receipts).issubset(ARMS),
             "Missing or unbudgeted continuation arms")
    if require_complete:
        _require(set(receipts) == set(ARMS), "Both matched continuation arms are required")
    new, historical, blocks = 0, 0, {}
    for name, row in receipts.items():
        parent = _integer(row["parent_optimizer_steps"], name + " parent updates", 1)
        added = _integer(row["new_optimizer_steps"], name + " new updates", 1)
        final = _integer(row["optimizer_steps"], name + " lifetime updates", 1)
        _require(parent == 250 and 0 < added <= 250 and final == parent + added,
                 name + " parent/new/lifetime update accounting changed")
        if require_complete:
            _require(added == 250 and final == 500 and row["selected_step"] == 500,
                     name + " must finish and retain its fixed step-500 model")
        history = row["history"]
        _require(isinstance(history, list) and 0 < len(history) <= 5, "Invalid continuation block ledger")
        _require([_integer(block["optimizer_steps"], "Block lifetime steps", 1) for block in history]
                 == list(range(300, final + 1, 50))
                 and [_integer(block["new_optimizer_steps"], "Block new steps", 1) for block in history]
                 == list(range(50, added + 1, 50)), "Continuation blocks do not follow fixed update boundaries")
        counters = row["actual_optimizer_step_counters"]
        _require(counters["minimum"] == counters["maximum"] == final
                 and _integer(counters["parameter_states"], "Parameter state count", 1) == 34,
                 "Loop count differs from actual optimizer parameter updates")
        for block in history:
            if "actual_optimizer_step_counters" in block:
                actual = block["actual_optimizer_step_counters"]
                _require(actual["minimum"] == actual["maximum"] == block["optimizer_steps"]
                         and actual["parameter_states"] == 34, "Block optimizer counters changed")
        _require(row.get("test_used", False) is False and row.get("test_inference_calls", 0) == 0
                 and row.get("validation_used_during_training", False) is False
                 and row.get("early_stopping_used", False) is False,
                 "Validation/test learning or early stopping entered continuation training")
        new += added
        historical += parent
        blocks[name] = len(history)
    _require(new <= 500, "Stage13 exceeded its new-update cap")
    return {"historical_optimizer_updates": historical, "new_optimizer_updates": new,
            "lifetime_optimizer_updates": historical + new, "new_sanity_updates": 0,
            "blocks": blocks, "complete": set(receipts) == set(ARMS)
            and all(row["new_optimizer_steps"] == 250 for row in receipts.values())}


def choose_arm(summaries):
    """Select only final-500 arms; exact-state first, onset second, group8 tie."""
    _require(set(summaries) == set(ARMS), "Selection requires exactly the two final-500 arms")
    scores = {name: (_number(summaries[name]["exact_state_f1"], name + " state F1", 0, 1),
                     _number(summaries[name]["onset_f1"], name + " onset F1", 0, 1)) for name in ARMS}
    return "group1" if scores["group1"] > scores["group8"] else "group8"


def paired_deltas(records):
    """Recompute within-recording 500-minus-250 score differences."""
    import numpy as np

    _require(isinstance(records, list) and bool(records), "Paired validation records are missing")
    keys = [(row["id"], row["mode"]) for row in records]
    _require(len(keys) == len(set(keys)) and {mode for _, mode in keys} == set(MODES),
             "Paired comparison has duplicate or unexpected views")
    expected_ids = {row["id"] for row in records}
    _require(set(keys) == {(clip_id, mode) for clip_id in expected_ids for mode in MODES},
             "Paired comparison is missing a recording/dose view")
    lookup = {(row["id"], row["mode"]): row for row in records}
    result = {}
    for arm in ARMS:
        pairs = []
        for clip_id in sorted(expected_ids):
            before, after = lookup[(clip_id, arm + "_250")], lookup[(clip_id, arm + "_500")]
            values = {"id": clip_id}
            for name, left, right in (("exact_state_f1", before["head_metrics"]["exact_state_f1"], after["head_metrics"]["exact_state_f1"]),
                                      ("onset_f1", before["pitch_metrics"]["onset"]["f1"], after["pitch_metrics"]["onset"]["f1"])):
                _number(left, "Before " + name, 0, 1)
                _number(right, "After " + name, 0, 1)
                values[name + "_250"] = left
                values[name + "_500"] = right
                values[name + "_delta"] = right - left
            pairs.append(values)
        result[arm] = {"records": pairs,
                       "exact_state_f1_delta": float(np.mean([row["exact_state_f1_delta"] for row in pairs])),
                       "onset_f1_delta": float(np.mean([row["onset_f1_delta"] for row in pairs]))}
    return result


def verify_schedule(schedule, config, train_inputs):
    """Regenerate crops and gain for historical data epochs five through nine."""
    import numpy as np
    import torch

    spec = config["training"]
    _require(spec["epochs"] == 5 and spec["samples_per_epoch"] == 200 and spec["batch_size"] == 4
             and spec["crop_seconds"] == 2 and config["data_epoch_start"] == 5 and config["seed"] == 42,
             "Matched continuation exposure changed")
    expected = []
    for epoch in range(5, 10):
        generator = torch.Generator().manual_seed(42 + epoch)
        torch.empty((), dtype=torch.int64).random_(generator=generator)
        order = torch.randperm(200, generator=generator).tolist()
        for index in order:
            rng = np.random.default_rng(42 + epoch * 100000 + index)
            record_index = int(rng.integers(len(train_inputs)))
            record = train_inputs[record_index]
            start = int(rng.integers(record["frames"] - 200 + 1))
            gain = float(rng.normal(0, spec["gain_augmentation_log_std"]))
            expected.append({"epoch": epoch + 1, "sample_index": index, "record_index": record_index,
                             "id": record["id"], "start_frame": start, "frames": 200, "gain_log": gain})
    _equal(schedule, expected, "Prespecified continuation crop/gain schedule", tolerance=0)
    return {"crop_schedule_sha256": _fingerprint(expected),
            "gain_schedule_sha256": _fingerprint([row["gain_log"] for row in expected])}


def verify_restoration(parent, restored, final, receipt):
    """Require actual restored tensors/moments, not a receipt's resume claim."""
    for kind in ("model", "optimizer"):
        digest = weight_digest if kind == "model" else optimizer_digest
        _require(digest(parent[kind]) == digest(restored[kind]), "Restored " + kind + " differs from frozen parent")
    _require(parent["step"] == restored["step"] == 250 and final["step"] == 500,
             "Parent/restored/final checkpoint steps changed")
    _equal(optimizer_counters(parent), {"minimum": 250, "maximum": 250, "parameter_states": 34}, "Frozen parent counters")
    _equal(optimizer_counters(restored), optimizer_counters(parent), "Restored optimizer counters")
    _equal(optimizer_counters(final), {"minimum": 500, "maximum": 500, "parameter_states": 34}, "Final optimizer counters")
    _equal(receipt["actual_optimizer_step_counters"], optimizer_counters(final), "Final receipt counters")
    groups = parent["optimizer"]["param_groups"]
    _require(len(groups) == 1 and groups[0]["lr"] == .0001, "Parent did not store the fixed terminal learning rate")
    _equal(final["optimizer"]["param_groups"], groups, "Unchanged AdamW hyperparameters and fixed learning rate")
    _require(groups[0]["params"] == list(range(34)) and len(parent["model"]) == 34,
             "Parent model and optimizer parameter order changed")
    _require(weight_digest(final["model"]) != weight_digest(parent["model"]), "Continuation model weights did not change")
    _require(optimizer_digest(final["optimizer"]) != optimizer_digest(parent["optimizer"]), "Continuation optimizer state did not change")
    _require(list(parent["model"]) == list(final["model"]), "Continuation model parameter keys/order changed")
    import torch
    for index, (key, value) in enumerate(final["model"].items()):
        expected = parent["model"][key]
        _require(value.shape == expected.shape and value.dtype == expected.dtype
                 and bool(torch.isfinite(value).all()), "Invalid continued model tensor")
        for checkpoint in (parent, final):
            state = checkpoint["optimizer"]["state"][index]
            _require(set(state) == {"step", "exp_avg", "exp_avg_sq"}
                     and state["step"].shape == torch.Size([]) and state["step"].dtype == torch.float32,
                     "Canonical AdamW state fields or step dtype changed")
            for moment in ("exp_avg", "exp_avg_sq"):
                _require(state[moment].shape == expected.shape and state[moment].dtype == expected.dtype,
                         "Optimizer moments differ from their corresponding model parameter")
    return {"parent_weights_sha256": weight_digest(parent["model"]),
            "parent_optimizer_sha256": optimizer_digest(parent["optimizer"]),
            "final_weights_sha256": weight_digest(final["model"]),
            "final_optimizer_sha256": optimizer_digest(final["optimizer"])}


def validate_parent_references(references, workspace):
    """Bind both frozen parents to the completed historical verification."""
    import torch
    from baseline.download import sha256
    from normalization12.audit import matched_arms

    workspace = Path(workspace).resolve()
    _require(set(references) == set(ARMS), "Both frozen parent references are required")
    paths = {"group8": "outputs/spectral11/models/short128/final.pt",
             "group1": "outputs/normalization12/models/group1/final.pt"}
    verification_path = workspace / "outputs/normalization12/verification.json"
    verification = _read(verification_path)
    old_audit_path = workspace / "outputs/normalization12/audit.json"
    old_audit = _read(old_audit_path)
    _require(verification["audit_sha256"] == sha256(old_audit_path)
             and old_audit["passed"] is True and old_audit["complete"] is True
             and old_audit["recomputed_saved_probabilities"] is True
             and verification["pytest"]["failed"] == 0,
             "The preserved parent verification did not pass or its audit changed")
    _equal(old_audit["checkpoint_sha256"], verification["checkpoint_sha256"], "Historical verified parent hashes")
    parents = {}
    for name in ARMS:
        reference = references[name]
        path = _relative(workspace, reference["checkpoint_path"])
        _require(path == workspace / paths[name] and reference["checkpoint_sha256"] == sha256(path)
                 == verification["checkpoint_sha256"][name], "Frozen parent checkpoint bytes changed")
        _require(_relative(workspace, reference["verification_path"]) == verification_path
                 and reference["verification_sha256"] == sha256(verification_path), "Historical verification bytes changed")
        parent = torch.load(path, map_location="cpu", weights_only=True)
        receipt = parent["receipt"]
        _equal(reference["historical_receipt"], receipt, "Original frozen parent receipt")
        _source_bindings(receipt["source_sha256"], workspace)
        _require(parent["step"] == receipt["optimizer_steps"] == receipt["selected_step"] == 250
                 and reference["historical_optimizer_steps"] == 250, "Frozen parent update accounting changed")
        _equal(optimizer_counters(parent), {"minimum": 250, "maximum": 250, "parameter_states": 34}, "Actual parent counters")
        _equal(parent["config"], receipt["model_config"], "Frozen parent architecture")
        _equal(parent["normalization"], receipt["normalization"], "Frozen parent normalization")
        _equal(parent["positive_weights"], receipt["positive_weights"], "Frozen parent positive loss weights")
        _require(parent["config"]["model"].get("cnn_norm_groups", 8) == reference["cnn_norm_groups"]
                 == (8 if name == "group8" else 1), "Frozen parent normalization grouping changed")
        _require(parent["config"]["feature"]["window_length"] == parent["config"]["feature"]["n_fft"] == 2048,
                 "Frozen parent must retain the common 128-ms analysis window")
        parents[name] = parent
    identities = matched_arms(parents["group8"]["receipt"], parents["group1"]["receipt"])
    _require(weight_digest(parents["group8"]["model"]) != weight_digest(parents["group1"]["model"]),
             "Different normalization arms were rebound to one parent")
    return parents, identities


def audit_validation_records(output, run, checkpoints, checkpoint_hashes, manifest, inputs, recompute=True):
    """Bind all forty-eight recording/dose exports to saved packet probabilities."""
    import mido
    import numpy as np
    from acoustic10.decoder import SourceTimeDecoder
    from baseline.download import ROOT, sha256
    from baseline.evaluation import evaluate, reference_notes
    from cnn_gru.inference import tablature_metrics
    from streaming.evaluate import aggregate
    from streaming.latency import measure

    output, workspace = Path(output).resolve(), ROOT.resolve()
    config = run["config"]
    expected_ids = {row["id"] for row in manifest["validation"]}
    records = [_read(path) for path in sorted((output / "validation/results").glob("*/*-metrics.json"))]
    _require(len(records) == 48 and {(row["id"], row["mode"]) for row in records}
             == {(clip_id, name) for clip_id in expected_ids for name in MODES},
             "Expected twelve unique validation recordings for each of four dose views")
    expected_probability_paths = {output / "validation/probabilities" / mode / (clip_id + ".npz")
                                  for mode in MODES for clip_id in expected_ids}
    _require(set((output / "validation/probabilities").glob("*/*.npz")) == expected_probability_paths,
             "Saved probability files have missing or unexpected recording/dose views")
    original_score = _read(workspace / "configs/stage1.json")
    for key, value in original_score.items():
        if key != "selection_role":
            _equal(run["scoring_config"][key], value, "Original source scoring configuration " + key)
    _require(run["scoring_config"]["selection_role"] ==
             "Stage13 continuation-dose development comparison; only final500 models select, no test inference or promotion.",
             "Validation score selection role changed")
    source_rows = {row["id"]: row for row in inputs["validation"]}
    note_count, event_count, parent_records = 0, 0, 0
    for row in records:
        mode, clip_id = row["mode"], row["id"]
        arm, dose = mode.rsplit("_", 1)
        expected_groups = 8 if arm == "group8" else 1
        source = source_rows[clip_id]
        for key in ("id", "frames", "duration", "features_sha256", "targets_sha256"):
            _equal(row["input_provenance"][key], source[key], "Matched frozen per-recording input " + key)
        stem = output / "validation/results" / mode / clip_id
        exported = _read(stem.with_suffix(".json"))
        _equal(exported["notes"], row["notes"], "Exact source-note JSON export")
        metadata = exported["metadata"]
        _require(metadata["checkpoint_sha256"] == checkpoint_hashes[mode] and metadata["mode"] == mode,
                 "Export checkpoint/dose identity changed")
        _equal(metadata["decoder_parameters"], config["decoder_parameters"], "Export fixed decoder")
        _require(metadata["source_delay_seconds"] == .13 and metadata["cnn_norm_groups"] == expected_groups,
                 "Export source compensation or normalization group count changed")
        _require(metadata["lifetime_optimizer_steps"] == int(dose), "Export lifetime dose changed")
        _equal(_read(stem.with_name(stem.name + "-events.json"))["events"], row["events"], "Exact published event export")
        ledger = audit_note_ledger(row["notes"], row["events"])
        note_count += ledger["notes"]
        event_count += ledger["events"]
        with stem.with_suffix(".csv").open(encoding="utf-8", newline="") as stream:
            table = list(csv.DictReader(stream))
        _require(len(table) == len(row["notes"]), "CSV dropped or duplicated source notes")
        for note, entry in zip(row["notes"], table):
            _require(float(entry["start"]) == note["start"] and float(entry["end"]) == note["end"]
                     and int(entry["pitch"]) == note["pitch"] and int(entry["string"]) == note["string"]
                     and int(entry["fret"]) == note["fret"], "CSV revised source intervals or assignments")
        midi = mido.MidiFile(stem.with_suffix(".mid"))
        attacks = Counter(message.note for track in midi.tracks for message in track
                          if message.type == "note_on" and message.velocity > 0)
        _require(attacks == Counter(note["pitch"] for note in row["notes"]), "MIDI dropped or revised published attacks")
        probability_path = _relative(output, row["probability_path"])
        _require(probability_path == output / "validation/probabilities" / mode / (clip_id + ".npz")
                 and sha256(probability_path) == row["probability_sha256"], "Saved packet probabilities changed or were rebound")
        timing = row["timing"]
        _require(timing["lookahead_samples"] == 0 and timing["packet_ms"] == config["validation_packet_ms"] == 20
                 and timing["exact_probability_recording_included"] is True
                 and timing["source_delay_compensation_seconds"] == .13, "Measured causal packet contract changed")
        _require(timing["window_seconds"] == .128 and timing["hann_peak_age_seconds"] == .064
                 and timing["fft_bin_spacing_hz"] == 7.8125 and timing["cnn_norm_groups"] == expected_groups
                 and timing["lifetime_optimizer_steps"] == int(dose)
                 and timing["parent_reused"] is (dose == "250"),
                 "Measured frontend or model identity changed")
        for key in ("input_seconds", "processing_seconds", "real_time_factor"):
            _number(timing[key], "Measured " + key, 0)
        if not recompute:
            continue
        with np.load(probability_path, allow_pickle=False) as arrays:
            probability = {key: arrays[key].copy() for key in arrays.files}
        _require(set(probability) == {"state", "onset", "offset", "pitch"}, "Saved acoustic heads changed")
        length, duration = len(probability["state"]), timing["input_seconds"]
        _require(length == math.floor(duration * 100 + 1e-8), "Saved probability frame count changed")
        for key, shape in (("state", (length, 6, 26)), ("onset", (length, 6)),
                           ("offset", (length, 6)), ("pitch", (length, 49))):
            values = probability[key]
            _require(values.shape == shape and values.dtype == np.float32 and np.isfinite(values).all()
                     and np.all((values >= 0) & (values <= 1)), "Invalid saved probability head " + key)
        _require(np.allclose(probability["state"].sum(-1), 1, rtol=0, atol=2e-6), "Saved state distribution is not normalized")
        decoder = SourceTimeDecoder(checkpoints[mode]["config"], config["decoder_parameters"], .13)
        for start in range(0, length, 2):
            end = min(start + 2, length)
            decoder.push({key: value[start:end] for key, value in probability.items()}, end / 100)
        decoder.finish(duration, duration)
        notes = [{key: value for key, value in note.items()
                  if key not in ("on_delivered_at_seconds", "off_delivered_at_seconds")} for note in row["notes"]]
        events = [{key: value for key, value in event.items() if key != "delivered_at_seconds"} for event in row["events"]]
        _equal(notes, decoder.notes, "Recomputed immutable source notes")
        _equal(events, decoder.events, "Recomputed packet publication events")
        _equal(timing["source_clamps"], decoder.clamp_counts, "Source-clock clamp count")
        entry = next(entry for entry in manifest["validation"] if entry["id"] == clip_id)
        reference = reference_notes(_relative(workspace, entry["annotation"]))
        _equal(row["reference"], reference, "Original source references")
        prefix = _relative(workspace, source["prefix"])
        with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as arrays:
            targets = {key: arrays[key].copy() for key in arrays.files}
        heads = aligned_heads(probability, targets)
        _equal({key: row["head_metrics"][key] for key in heads}, heads, "Independent aligned exact-state metrics")
        _equal(row["head_metrics"]["pitch"], aligned_pitch(probability, targets), "Independent fixed-threshold pitch diagnostics")
        _equal(row["pitch_metrics"], evaluate(reference, row["notes"], run["scoring_config"], duration), "Recomputed note metrics")
        _equal(row["tab_metrics"], tablature_metrics(reference, row["notes"], run["scoring_config"],
                                                   run["scoring_config"]["score_seconds"], duration), "Recomputed string metrics")
        _equal(row["latency"], measure(reference, row["notes"], run["scoring_config"], duration), "Recomputed matched-event latency")
        if dose == "250":
            previous_output = workspace / "outputs/normalization12"
            previous_path = previous_output / "validation/results" / arm / (clip_id + "-metrics.json")
            previous = _read(previous_path)
            previous_probability = _relative(previous_output, previous["probability_path"])
            _require(sha256(previous_probability) == previous["probability_sha256"], "Preserved parent probabilities changed")
            with np.load(previous_probability, allow_pickle=False) as arrays:
                _require(set(arrays.files) == set(probability)
                         and all(arrays[key].shape == probability[key].shape for key in probability),
                         "Parent probability heads/grid changed")
                maximum = max(float(np.max(np.abs(arrays[key] - probability[key]))) for key in probability)
                exact = all(np.array_equal(arrays[key], probability[key]) for key in probability)
                _require(maximum <= 1e-6, "Frozen parent probabilities were not reproduced")
            reproduction = row["parent_reproduction"]
            _require(_relative(workspace, reproduction["historical_metrics_path"]) == previous_path
                     and reproduction["historical_metrics_sha256"] == sha256(previous_path)
                     and reproduction["historical_probability_sha256"] == previous["probability_sha256"],
                     "Frozen parent reproduction byte bindings changed")
            _equal(reproduction["max_probability_absolute_difference"], maximum, "Recorded parent probability difference")
            _require(reproduction["probabilities_bit_identical"] is bool(exact)
                     and reproduction["source_notes_identical"] is True
                     and reproduction["source_scores_identical"] is True
                     and reproduction["head_decisions_identical"] is True,
                     "Recorded parent reproduction claims contradict saved probabilities")
            _equal(notes, without_delivery_clocks(previous["notes"]), "Historical parent source-note decisions")
            _equal(without_delivery_clocks(row["pitch_metrics"]), without_delivery_clocks(previous["pitch_metrics"]), "Historical parent note scores")
            _equal(without_delivery_clocks(row["tab_metrics"]), without_delivery_clocks(previous["tab_metrics"]), "Historical parent string scores")
            for key in ("exact_state", "active_fret_accuracy", "pitch"):
                _equal(row["head_metrics"][key], previous["head_metrics"][key], "Historical parent head decisions")
            parent_records += 1
        else:
            _require(row["parent_reproduction"] is None, "A continued model was represented as a parent replay")
    summary = aggregate(records)
    for mode in MODES:
        rows = [row for row in records if row["mode"] == mode]
        summary["modes"][mode]["exact_state_f1"] = float(np.mean([row["head_metrics"]["exact_state_f1"] for row in rows]))
        summary["modes"][mode]["active_fret_accuracy"] = float(np.mean([row["head_metrics"]["active_fret_accuracy"] for row in rows]))
        summary["modes"][mode]["auxiliary_pitch_f1"] = float(np.mean([row["head_metrics"]["pitch"]["overall"]["f1"] for row in rows]))
        summary["modes"][mode]["low_register_pitch_f1"] = float(np.mean([row["head_metrics"]["pitch"]["low_register"]["f1"] for row in rows]))
    _equal(_read(output / "validation/summary.json"), summary, "Independent validation aggregation")
    return records, summary, {"validation_records": len(records), "notes": note_count, "events": event_count,
                              "parent_records_reproduced": parent_records,
                              "recomputed_saved_probabilities": bool(recompute)}


def audit_artifacts(output, recompute=True):
    """Verify historical parents, two continued models and four dose views."""
    import numpy as np
    import torch
    from acoustic10.audit import validate_sanity
    from baseline.download import ROOT, sha256
    from .optimizer import recursive_digest

    output, workspace = Path(output).resolve(), ROOT.resolve()
    config = _read(workspace / "configs/optimization13.json")
    previous_config = _read(workspace / "configs/normalization12.json")
    expected_training = {**previous_config["training"], "learning_rate": .0001}
    _equal(config["training"], expected_training, "Fixed continuation training recipe")
    _require(config["seed"] == 42 and config["torch_continuation_seed"] == 43
             and config["data_epoch_start"] == 5 and config["total_optimizer_update_cap"] == 500
             and config["planned_new_optimizer_updates"] == 500
             and config["amp_scaler_reset"] == {"initial_scale": 1024}
             and config["test_inference_allowed"] is False and config["production_promotion"] is False,
             "Prespecified continuation budget, random state or development scope changed")
    _equal(config["arms"], [{"name": "group8", "cnn_norm_groups": 8,
                             "parent_checkpoint": "outputs/spectral11/models/short128/final.pt"},
                            {"name": "group1", "cnn_norm_groups": 1,
                             "parent_checkpoint": "outputs/normalization12/models/group1/final.pt"}],
           "Two matched immutable parents")
    for key in ("common_target_delay_frames", "common_leading_mask_frames", "decoder_parameters",
                "validation_packet_ms", "pitch_diagnostics"):
        _equal(config[key], previous_config[key], "Unchanged timing/diagnostic recipe " + key)
    protocol = _read(output / "protocol.json")
    _equal(protocol["config"], config, "Prespecified Stage13 configuration")
    _source_bindings(protocol["source_sha256"], workspace)
    _require(protocol["total_new_update_cap"] == 500 and protocol["historical_parent_updates"] == 500
             and protocol["new_sanity_updates"] == 0 and protocol["test_inference_allowed"] is False
             and protocol["exact_uninterrupted_resume"] is False,
             "Prespecified stage accounting or exact-resume claim changed")
    _equal(protocol["planned_new_optimizer_updates"], {"group8": 250, "group1": 250}, "Prespecified new updates")
    _require(protocol["selection_rule"] == config["selection_rule"], "Prespecified final-only selection rule changed")
    input_path = output / "input-audit.json"
    inputs = _read(input_path)
    manifest = _read(workspace / "data/cnn_gru/manifest.json")
    _require(len(manifest["train"]) == 48 and len(manifest["validation"]) == 12,
             "Expected forty-eight training and twelve validation recordings")
    _require(not ({row["id"] for row in manifest["train"]} & {row["id"] for row in manifest["validation"]}),
             "Training and validation recordings overlap")
    _check_inputs(inputs, manifest, workspace, include_validation=True)
    _require(inputs["config_sha256"] == _fingerprint(config)
             and inputs["new_optimizer_updates_for_preparation"] == 0
             and inputs["features_rewritten"] is False and inputs["targets_rewritten"] is False,
             "Input preparation rewrote frozen caches or performed training")
    parents, initial_identities = validate_parent_references(inputs["parents"], workspace)
    _equal(protocol["parents"], inputs["parents"], "Prespecified matched parent references")
    frozen = _read(workspace / "data/streaming/cache_audit.json")
    _equal(inputs["normalization"], frozen["normalization"], "Frozen train-only feature statistics")
    _equal(inputs["positive_weights"], frozen["positive_weights"], "Frozen loss balancing")
    train_records = [{"features": np.load(_relative(workspace, row["prefix"]).with_suffix(".npy"),
                                         mmap_mode="r", allow_pickle=False)} for row in inputs["inputs"]["train"]]
    normalization = validate_normalization(inputs["normalization"], train_records)
    saved_schedule = _read(output / "crop-schedule.json")
    descriptors = saved_schedule["descriptors"]
    _require(saved_schedule["sha256"] == _fingerprint(descriptors), "Saved shared continuation crop schedule changed")
    schedule_identities = verify_schedule(descriptors, config, inputs["inputs"]["train"])
    support = _fingerprint([{"id": row["id"], "frames": row["frames"],
                             "ignored_global_leading_frames": 26, "ignored_crop_edge_frames": 26,
                             "target_delay_frames": 13} for row in inputs["inputs"]["train"]])
    receipts, checkpoints, checkpoint_hashes, restoration = {}, {}, {}, {}
    for name in ARMS:
        parent = parents[name]
        reference = inputs["parents"][name]
        directory = output / "models" / name
        _equal(_read(directory / "parent-reference.json"), reference, "Saved immutable parent reference")
        final_path = directory / "final.pt"
        final = torch.load(final_path, map_location="cpu", weights_only=True)
        receipt = _read(directory / "training-receipt.json")
        _equal(final["receipt"], receipt, "Continued checkpoint training receipt")
        _equal(receipt["config"], config, "Continued prespecified configuration")
        _require(receipt["config_sha256"] == _fingerprint(config), "Continued configuration fingerprint changed")
        _equal(receipt["source_sha256"], protocol["source_sha256"], "Continued producer identities")
        _equal(receipt["inputs"], inputs["inputs"], "Continued frozen train/validation inputs")
        _equal(receipt["parent_reference"], reference, "Continued immutable parent reference")
        _equal(receipt["parents"], inputs["parents"], "Continued matched parent identities")
        _require(receipt["input_audit_sha256"] == sha256(input_path)
                 and receipt["parent_checkpoint_sha256"] == reference["checkpoint_sha256"]
                 and receipt["parent_checkpoint_path"] == reference["checkpoint_path"],
                 "Continued parent/input byte bindings changed")
        _require(receipt["name"] == name and receipt["cnn_norm_groups"] == (8 if name == "group8" else 1)
                 and receipt["restored_from_parent"] is True and receipt["historical_rng_restored"] is False
                 and receipt["historical_scaler_restored"] is False and receipt["torch_continuation_seed"] == 43
                 and receipt["scaler_initial_scale"] == 1024 and receipt["learning_rate"] == .0001
                 and receipt["learning_rate_rule"] == "constant stored-end learning rate"
                 and receipt["scheduler_used"] is False and receipt["model_parameters"] == 931145,
                 "Continuation restoration, random-state or optimizer recipe changed")
        _require(receipt["target_delay_frames"] == receipt["delay_frames"] == 13
                 and receipt["common_leading_mask_frames"] == receipt["crop_edge_mask_frames"] == 26,
                 "Common delayed target/masked support changed")
        _require(receipt["additional_epochs_completed"] == 5,
                 "The continuation did not record five additional sampled-crop blocks")
        _equal(receipt["optimizer_parameter_steps"], {"minimum": 500, "maximum": 500}, "Displayed actual parameter steps")
        _require(receipt["common_support_sha256"] == support == parent["receipt"]["common_support_sha256"]
                 and receipt["original_fresh_initial_weights_sha256"] == initial_identities["initial_weights_sha256"],
                 "Historical fresh initialization or common loss support changed")
        for key, digest in schedule_identities.items():
            _require(receipt[key] == digest, "Matched continuation crop/gain exposure changed")
        expected_config = copy.deepcopy(parent["config"])
        expected_config["training"].update(config["training"])
        expected_config["stage_budget"] = {"max_new_optimizer_updates": 500, "new_per_arm": 250,
                                            "parent_updates": 250, "final_lifetime_updates": 500,
                                            "new_sanity_updates": 0, "purpose": "Stage13 continuation-dose trial."}
        _equal(final["config"], expected_config, "Only continuation training/accounting settings may change")
        _equal(receipt["model_config"], expected_config, "Continued architecture receipt")
        for key in ("normalization", "positive_weights"):
            _equal(final[key], inputs[key], "Continued frozen " + key)
            _equal(receipt[key], inputs[key], "Continued receipt " + key)
        resume_path = _relative(output, receipt["resume_state_path"])
        _require(resume_path == directory / "resume-state.pt" and sha256(resume_path) == receipt["resume_state_sha256"],
                 "Actual restored-state snapshot changed or was rebound")
        restored = torch.load(resume_path, map_location="cpu", weights_only=True)
        for key in ("config", "normalization", "positive_weights"):
            _equal(restored[key], parent[key], "Restored parent metadata " + key)
        proof = receipt["restoration_proof"]
        _equal(restored["restoration_proof"], proof, "Restored-state proof binding")
        restoration[name] = verify_restoration(parent, restored, final, receipt)
        _require(proof["parent_step"] == 250 and proof["new_optimizer_updates"] == 0
                 and proof["held_learning_rate"] == .0001 and proof["model_parameters"] == 931145
                 and proof["parameter_states"] == 34 and proof["model_weights_identical"] is True
                 and proof["optimizer_state_identical"] is True and proof["parent_checkpoint_unmutated"] is True
                 and proof["historical_rng_restored"] is False and proof["historical_amp_scaler_restored"] is False,
                 "Restoration proof contradicts the continuation contract")
        _equal(proof["actual_optimizer_step_counters"], optimizer_counters(parent), "Displayed restored parameter counters")
        _equal(proof["parameter_names"], list(parent["model"]), "Displayed parameter ordering")
        for kind in ("model", "optimizer"):
            _require(proof["parent_" + kind + "_sha256"] == proof["restored_" + kind + "_sha256"]
                     == receipt["restored_" + kind + "_sha256"] == recursive_digest(parent[kind]),
                     "Displayed restored " + kind + " digest is incorrect")
        _require(proof["parent_metadata_sha256"] == recursive_digest({key: parent[key]
                                                                     for key in ("config", "normalization", "positive_weights")}),
                 "Displayed restored metadata digest is incorrect")
        for block_index, block in enumerate(receipt["history"], 1):
            _require(block["learning_rate"] == .0001
                     and block["additional_epoch"] == block_index
                     and block["data_epoch"] == 5 + block_index, "Training block recipe/exposure changed")
            for key, value in block["training_losses"].items():
                _number(value, "Recorded training loss " + key, 0)
        receipts[name] = receipt
        for dose, checkpoint, digest in (("250", parent, reference["checkpoint_sha256"]),
                                         ("500", final, sha256(final_path))):
            checkpoints[name + "_" + dose] = checkpoint
            checkpoint_hashes[name + "_" + dose] = digest
    budget = validate_budget(receipts)
    _require({directory.name for directory in (output / "models").iterdir() if directory.is_dir()} == set(ARMS),
             "An unbudgeted model directory is present")
    training = _read(output / "training-summary.json")
    _equal(training["arms"], receipts, "Matched continuation training-summary receipts")
    _equal(training["source_sha256"], protocol["source_sha256"], "Training-summary producer identities")
    _require(training["new_optimizer_updates"] == training["historical_parent_updates"] == 500
             and training["final_lifetime_updates_per_model"] == 500 and training["new_sanity_updates"] == 0
             and training["additional_epochs_per_arm"] == 5 and training["lifetime_epoch_blocks_per_model"] == 10
             and training["test_inference_calls"] == 0
             and training["validation_used_during_training"] is False,
             "Training-summary budget or development scope changed")
    _equal(training["config"], config, "Training-summary fixed configuration")
    _equal(training["parents"], inputs["parents"], "Training-summary immutable parent references")
    _require(training["input_audit_sha256"] == sha256(input_path)
             and training["protocol_sha256"] == sha256(output / "protocol.json"), "Training protocol/input binding changed")
    prior_sanity = _read(workspace / config["sanity_reference"])
    verdict = validate_sanity(prior_sanity["initial"], prior_sanity["final"], prior_sanity["coverage"], prior_sanity["passed"])
    _require(verdict["passed"] is True, "Preserved training-only sanity evidence failed")
    _source_bindings(prior_sanity["source_sha256"], workspace)
    sanity = protocol["sanity_reference"]
    _require(sanity["sha256"] == sha256(workspace / config["sanity_reference"])
             and sanity["checkpoint_sha256"] == sha256(workspace / "outputs/acoustic10/sanity-checkpoint.pt")
             and sanity["new_optimizer_updates"] == 0 and sanity["passed"] is True, "Preserved prior sanity identity changed")
    _equal(training["sanity_reference"], sanity, "Training preserved prior-sanity identity")
    run = _read(output / "validation/run.json")
    _equal(run["config"], config, "Validation continuation recipe")
    _equal(run["source_sha256"], protocol["source_sha256"], "Validation producer identities")
    _equal(run["training_summary"], training, "Validation continuation training accounting")
    _equal(run["checkpoint_sha256"], checkpoint_hashes, "Validation checkpoint identities")
    _equal(run["input_provenance"]["inputs"], inputs["inputs"], "Validation frozen input identities")
    _equal(run["parents"], inputs["parents"], "Validation immutable parent references")
    _equal(run["sanity_reference"], sanity, "Validation preserved prior-sanity identity")
    _require(run["new_optimizer_updates"] == run["historical_parent_updates"] == 500
             and run["new_sanity_updates"] == 0 and run["test_used"] is False and run["test_inference_calls"] == 0
             and run["input_audit_sha256"] == sha256(input_path)
             and run["protocol_sha256"] == sha256(output / "protocol.json"), "Validation scope or immutable bindings changed")
    records, summary, counts = audit_validation_records(output, run, checkpoints, checkpoint_hashes,
                                                         manifest, inputs["inputs"], recompute)
    selection = _read(output / "selection.json")
    _equal(run["selection"], selection, "Validation final-only selection")
    _equal(selection["checkpoint_sha256"], checkpoint_hashes, "Selection frozen checkpoints")
    _require(selection["selected_from_test"] is False and selection["production_promoted"] is False
             and selection["rule"] == config["selection_rule"], "Selection scope or rule changed")
    _equal(selection["eligible_modes"], ["group8_500", "group1_500"], "Only final-500 models are selection eligible")
    scores = {name: {key: summary["modes"][name + "_500"][key] for key in ("exact_state_f1", "onset_f1")} for name in ARMS}
    _equal(selection["scores"], {name + "_500": row for name, row in scores.items()}, "Independent final-only selection scores")
    _require(selection["selected_arm"] == choose_arm(scores) + "_500", "Selected group contradicts the fixed final-only validation rule")
    return {"passed": True, **budget, **counts, "selected_arm": selection["selected_arm"],
            "checkpoint_sha256": checkpoint_hashes, "protocol_sha256": sha256(output / "protocol.json"),
            "restoration": restoration, "normalization": normalization, "paired_deltas": paired_deltas(records),
            "test_inference_calls": 0,
            "scope": "Read-only saved-tensor audit; zero audio/model inference or optimizer updates."}


def audit(output, recompute=True):
    """Public CLI adapter; optional recomputation consumes saved probabilities."""
    return audit_artifacts(output, recompute=recompute)
