"""Independent read-only checks for the causal normalization comparison.

The control is a frozen historical run. Its 250 updates remain historical;
only the new groups-one run consumes the stage-12 update budget. Validation
probabilities can be decoded again without audio inference or optimization.
"""

from __future__ import annotations

from collections import Counter
import copy
import math
from pathlib import Path

from acoustic10.audit import (_check_inputs, _equal, _fingerprint, _integer,
                              _number, _read, _relative, _require,
                              _source_bindings, _verify_schedule,
                              audit_note_ledger, weight_digest)
from spectral11.audit import (aligned_heads, aligned_pitch,
                              _optimizer_receipt, validate_normalization)


ARMS = ("group8", "group1")


def without_delivery_clocks(value):
    """Compare historical decisions while allowing current service times."""
    if isinstance(value, dict):
        return {key: without_delivery_clocks(item) for key, item in value.items()
                if key not in ("on_delivered_at_seconds", "off_delivered_at_seconds")}
    if isinstance(value, list):
        return [without_delivery_clocks(item) for item in value]
    return value


def validate_budget(receipts, require_complete=True):
    """Validate actual parameter update counters, separately by run origin."""
    _require(isinstance(receipts, dict) and bool(receipts), "Training receipts are missing")
    _require(set(receipts).issubset(ARMS), "An unbudgeted training run is present")
    if require_complete:
        _require(set(receipts) == set(ARMS), "Both historical and new matched arms are required")
    blocks = {}
    for name, row in receipts.items():
        steps = _integer(row["optimizer_steps"], name + " optimizer updates", 1)
        _require(steps <= 250, name + " exceeded its fixed update budget")
        if require_complete:
            _require(steps == 250, name + " did not finish its fixed update budget")
        history = row["history"]
        _require(isinstance(history, list) and 0 < len(history) <= 5,
                 name + " requires one to five recorded training blocks")
        counts = [_integer(block["optimizer_steps"], name + " block update count", 1)
                  for block in history]
        _require(counts == list(range(50, steps + 1, 50)) and counts[-1] == steps,
                 name + " block ledger differs from the fixed update boundaries")
        counters = row["actual_optimizer_step_counters"]
        _integer(counters["minimum"], name + " minimum parameter update count", 1)
        _integer(counters["maximum"], name + " maximum parameter update count", 1)
        _require(counters["minimum"] == counters["maximum"] == steps,
                 name + " loop count differs from actual optimizer parameter updates")
        for block in history:
            if "actual_optimizer_step_counters" in block:
                actual = block["actual_optimizer_step_counters"]
                _require(actual["minimum"] == actual["maximum"] == block["optimizer_steps"],
                         name + " block count differs from actual optimizer updates")
        _require(row.get("test_used", False) is False and row.get("test_inference_calls", 0) == 0,
                 "Test inference entered training")
        blocks[name] = len(history)
    historical = receipts.get("group8", {}).get("optimizer_steps", 0)
    new = receipts.get("group1", {}).get("optimizer_steps", 0)
    return {"historical_optimizer_updates": historical, "new_optimizer_updates": new,
            "compared_optimizer_updates": historical + new, "new_sanity_updates": 0,
            "blocks": blocks,
            "complete": set(receipts) == set(ARMS)
                        and all(row["optimizer_steps"] == 250 for row in receipts.values())}


def matched_arms(control, variant):
    """Allow only the frame-group count to change between matched recipes."""
    keys = ("initial_weights_sha256", "initial_checkpoint_sha256", "crop_schedule_sha256",
            "gain_schedule_sha256", "common_support_sha256", "manifest_sha256", "cache_audit_sha256")
    for key in keys:
        _require(isinstance(control.get(key), str) and len(control[key]) == 64,
                 "Missing matched-arm identity: " + key)
        _require(control[key] == variant.get(key), "Matched arms differ in " + key)
    for name, row in zip(ARMS, (control, variant)):
        _require(row.get("selected_step") == row.get("optimizer_steps") == 250,
                 name + " must use its fixed final step-250 model")
        _require(row.get("warmstarted", False) is False
                 and row.get("warm_started_from_sanity", False) is False,
                 name + " must start from fresh initial weights")
        _require(row.get("target_delay_frames") == 13 and row.get("delay_frames") == 13,
                 "Both arms require the common thirteen-frame target delay")
        _require(row.get("common_leading_mask_frames") == 26
                 and row.get("crop_edge_mask_frames") == 26,
                 "Both arms require the same twenty-six-frame support masks")
        _require(row.get("early_stopping_used", False) is False
                 and row.get("validation_used_during_training", False) is False,
                 name + " used validation learning or early stopping")
    _equal(control["normalization"], variant["normalization"], "Matched train-only feature normalization")
    _equal(control["positive_weights"], variant["positive_weights"], "Matched loss weights")
    left, right = copy.deepcopy(control["model_config"]), copy.deepcopy(variant["model_config"])
    # Accounting metadata changes because the historical control is reused.
    left.pop("stage_budget", None)
    right.pop("stage_budget", None)
    for name, item, expected in (("group8", left, 8), ("group1", right, 1)):
        groups = item["model"].pop("cnn_norm_groups", 8)
        _require(groups == expected and not isinstance(groups, bool), name + " group count changed")
    _equal(left, right, "All model/training settings except normalization group count")
    return {key: control[key] for key in keys}


def choose_arm(summaries):
    """Exact-state F1 first, note-onset F1 second; historical control wins ties."""
    _require(set(summaries) == set(ARMS), "Exactly two validation arms are required")
    scores = {name: (_number(summaries[name]["exact_state_f1"], name + " exact-state F1", 0, 1),
                     _number(summaries[name]["onset_f1"], name + " onset F1", 0, 1)) for name in ARMS}
    return "group1" if scores["group1"] > scores["group8"] else "group8"


def validate_historical_control(reference, workspace):
    """Bind the reused control to its immutable original training evidence."""
    import torch
    from baseline.download import sha256
    from streaming.model import CausalCNNGRU

    workspace = Path(workspace).resolve()
    expected = {"checkpoint": "outputs/spectral11/models/short128/final.pt",
                "receipt": "outputs/spectral11/models/short128/training-receipt.json",
                "initial_checkpoint": "outputs/spectral11/initial-weights.pt",
                "crop_schedule": "outputs/spectral11/crop-schedule.json"}
    for kind, relative in expected.items():
        path = _relative(workspace, reference[kind + "_path"])
        _require(path == workspace / relative, "Historical control " + kind + " path changed")
        _require(sha256(path) == reference[kind + "_sha256"],
                 "Historical control " + kind + " bytes changed")
    receipt = _read(workspace / expected["receipt"])
    checkpoint = torch.load(workspace / expected["checkpoint"], map_location="cpu", weights_only=True)
    _equal(checkpoint["receipt"], receipt, "Unmodified original control receipt")
    _require(receipt["name"] == "short128", "Historical control was rebound to another Stage11 arm")
    _source_bindings(receipt["source_sha256"], workspace)
    _equal(receipt["config"], _read(workspace / "configs/spectral11.json"), "Original control experiment recipe")
    _require(receipt["config_sha256"] == _fingerprint(receipt["config"]),
             "Historical control configuration fingerprint changed")
    _equal(checkpoint["config"], receipt["model_config"], "Original control architecture")
    _equal(checkpoint["normalization"], receipt["normalization"], "Original control feature statistics")
    _require(receipt["model_config"]["feature"]["window_length"] == 2048
             and receipt["model_config"]["feature"]["n_fft"] == 2048,
             "Historical control must use the 128-ms spectral window")
    validate_budget({"group8": receipt}, require_complete=False)
    _require(receipt["selected_step"] == 250 and receipt["warmstarted"] is False
             and receipt["early_stopping_used"] is False
             and receipt["validation_used_during_training"] is False,
             "Historical control used a different training or checkpoint selection rule")
    initial = torch.load(workspace / expected["initial_checkpoint"], map_location="cpu", weights_only=True)
    _require(initial["origin"] == "fresh random initialization" and initial["sanity_warmstart"] is False
             and initial["seed"] == 42, "Historical initial weights were not a fresh seed42 model")
    _require(weight_digest(initial["model"]) == initial["weights_sha256"] == receipt["initial_weights_sha256"],
             "Historical fresh initial tensor bytes changed")
    _require(reference["initial_checkpoint_sha256"] == receipt["initial_checkpoint_sha256"],
             "Historical initial-file receipt changed")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        fresh = CausalCNNGRU(receipt["model_config"])
        _require(weight_digest(fresh.state_dict()) == initial["weights_sha256"],
                 "Historical initial model differs from a fresh seed42 initialization")
        count = len(list(fresh.parameters()))
        _require(sum(value.numel() for value in fresh.parameters()) == 931145,
                 "Historical control parameter count changed")
    _optimizer_receipt(checkpoint, receipt, count)
    schedule = _read(workspace / expected["crop_schedule"])
    _require(schedule["sha256"] == _fingerprint(schedule["descriptors"]) == receipt["crop_schedule_sha256"],
             "Historical crop exposure changed")
    _require(_fingerprint([row["gain_log"] for row in schedule["descriptors"]]) == receipt["gain_schedule_sha256"],
             "Historical gain augmentation exposure changed")
    return receipt, checkpoint, initial, schedule["descriptors"]


def audit_validation_records(output, run, checkpoints, checkpoint_hashes, manifest, inputs, recompute=True):
    """Bind the 24 saved validation exports to exact model probabilities."""
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
    _require(len(records) == 24 and {(row["id"], row["mode"]) for row in records}
             == {(clip_id, name) for clip_id in expected_ids for name in ARMS},
             "Expected twelve unique validation clips per normalization arm")
    original_score = _read(workspace / "configs/stage1.json")
    for key, value in original_score.items():
        if key == "selection_role":
            continue
        _equal(run["scoring_config"][key], value, "Original source scoring configuration " + key)
    _require(run["scoring_config"]["selection_role"] ==
             "Stage12 validation-only normalization comparison; exact reused control, no test inference or promotion.",
             "Validation scoring selection scope changed")
    source_rows = {row["id"]: row for row in inputs["validation"]}
    note_count = event_count = 0
    for row in records:
        name, clip_id = row["mode"], row["id"]
        source = source_rows[clip_id]
        for key in ("id", "frames", "duration", "features_sha256", "targets_sha256"):
            _equal(row["input_provenance"][key], source[key], "Matched per-clip frozen input " + key)
        stem = output / "validation/results" / name / clip_id
        exported = _read(stem.with_suffix(".json"))
        _equal(exported["notes"], row["notes"], "Exact source-note export")
        metadata = exported["metadata"]
        _require(metadata["checkpoint_sha256"] == checkpoint_hashes[name] and metadata["mode"] == name,
                 "Export checkpoint/normalization-arm identity changed")
        _equal(metadata["decoder_parameters"], config["decoder_parameters"], "Export fixed decoder parameters")
        _equal(metadata["source_delay_seconds"], .13, "Public source-clock compensation")
        expected_groups = 8 if name == "group8" else 1
        _require(metadata["cnn_norm_groups"] == expected_groups
                 and metadata["control_reused"] is (name == "group8"),
                 "Export normalization group count or historical origin changed")
        _equal(_read(stem.with_name(stem.name + "-events.json"))["events"], row["events"], "Published event export")
        ledger = audit_note_ledger(row["notes"], row["events"])
        note_count += ledger["notes"]
        event_count += ledger["events"]
        midi = mido.MidiFile(stem.with_suffix(".mid"))
        attacks = Counter(message.note for track in midi.tracks for message in track
                          if message.type == "note_on" and message.velocity > 0)
        _require(attacks == Counter(note["pitch"] for note in row["notes"]),
                 "MIDI dropped or changed published attacks")
        probability_path = _relative(output, row["probability_path"])
        _require(probability_path == output / "validation/probabilities" / name / (clip_id + ".npz"),
                 "Probability file was rebound to another clip or arm")
        _require(sha256(probability_path) == row["probability_sha256"], "Exact packet probabilities changed")
        timing = row["timing"]
        _require(timing["lookahead_samples"] == 0 and timing["packet_ms"] == config["validation_packet_ms"] == 20
                 and timing["exact_probability_recording_included"] is True,
                 "Measured packet contract changed")
        _equal(timing["source_delay_compensation_seconds"], .13, "Measured source compensation")
        _require(timing["window_seconds"] == .128 and timing["hann_peak_age_seconds"] == .064
                 and timing["fft_bin_spacing_hz"] == 7.8125
                 and timing["cnn_norm_groups"] == expected_groups
                 and timing["control_reused"] is (name == "group8"),
                 "Measured model/frontend identity changed")
        for key in ("input_seconds", "processing_seconds", "real_time_factor"):
            if key in timing:
                _number(timing[key], "Measured " + key, 0)
        if recompute:
            with np.load(probability_path, allow_pickle=False) as arrays:
                probability = {key: arrays[key].copy() for key in arrays.files}
            _require(set(probability) == {"state", "onset", "offset", "pitch"}, "Acoustic probability heads changed")
            length, duration = len(probability["state"]), timing["input_seconds"]
            _require(length == math.floor(duration * 100 + 1e-8), "Packet probability frame count changed")
            _require(probability["state"].shape == (length, 6, 26)
                     and np.isfinite(probability["state"]).all()
                     and np.all((probability["state"] >= 0) & (probability["state"] <= 1))
                     and np.allclose(probability["state"].sum(-1), 1, rtol=0, atol=2e-6),
                     "Saved full state probability distribution is invalid")
            for key, width in (("onset", 6), ("offset", 6), ("pitch", 49)):
                _require(probability[key].shape == (length, width)
                         and np.isfinite(probability[key]).all()
                         and np.all((probability[key] >= 0) & (probability[key] <= 1)),
                         "Invalid acoustic probability head")
            decoder = SourceTimeDecoder(checkpoints[name]["config"], config["decoder_parameters"], .13)
            for start in range(0, length, 2):
                end = min(start + 2, length)
                decoder.push({key: value[start:end] for key, value in probability.items()}, end / 100)
            decoder.finish(duration, duration)
            notes = [{key: value for key, value in note.items()
                      if key not in ("on_delivered_at_seconds", "off_delivered_at_seconds")}
                     for note in row["notes"]]
            events = [{key: value for key, value in event.items() if key != "delivered_at_seconds"}
                      for event in row["events"]]
            _equal(notes, decoder.notes, "Recomputed immutable source notes")
            _equal(events, decoder.events, "Recomputed packet publication events")
            _equal(timing["source_clamps"], decoder.clamp_counts, "Source-clock clamp counts")
            entry = next(entry for entry in manifest["validation"] if entry["id"] == clip_id)
            reference = reference_notes(_relative(workspace, entry["annotation"]))
            _equal(row["reference"], reference, "Original source reference notes")
            prefix = _relative(workspace, source["prefix"])
            with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as arrays:
                target = {key: arrays[key].copy() for key in arrays.files}
            heads = aligned_heads(probability, target)
            _equal({key: row["head_metrics"][key] for key in heads}, heads,
                   "Independent aligned exact-state metrics")
            _equal(row["head_metrics"]["pitch"], aligned_pitch(probability, target),
                   "Independent fixed-threshold auxiliary pitch diagnostics")
            _equal(row["pitch_metrics"], evaluate(reference, row["notes"], run["scoring_config"], duration),
                   "Recomputed original-source note/pitch metrics")
            _equal(row["tab_metrics"], tablature_metrics(reference, row["notes"], run["scoring_config"],
                                                       run["scoring_config"]["score_seconds"], duration),
                   "Recomputed string metrics")
            _equal(row["latency"], measure(reference, row["notes"], run["scoring_config"], duration),
                   "Recomputed conditional matched-event latency")
            if name == "group8":
                previous_stem = workspace / "outputs/spectral11/validation/results/short128" / clip_id
                previous_path = previous_stem.with_name(previous_stem.name + "-metrics.json")
                previous = _read(previous_path)
                previous_probability = _relative(workspace / "outputs/spectral11", previous["probability_path"])
                reproduction = row["historical_control_reproduction"]
                _require(reproduction["historical_metrics_sha256"] == sha256(previous_path)
                         and reproduction["historical_probability_sha256"]
                         == previous["probability_sha256"] == sha256(previous_probability),
                         "Historical control reproduction source identities changed")
                exact, maximum = True, 0.
                with np.load(previous_probability, allow_pickle=False) as arrays:
                    _require(set(arrays.files) == set(probability), "Historical control probability heads changed")
                    for key in probability:
                        _require(arrays[key].shape == probability[key].shape, "Historical control probability shape changed")
                        exact = exact and np.array_equal(arrays[key], probability[key])
                        maximum = max(maximum, float(np.max(np.abs(arrays[key] - probability[key]))))
                _require(maximum <= 1e-6, "Current control no longer reproduces its historical probabilities")
                _equal(reproduction["max_probability_absolute_difference"], maximum,
                       "Recomputed historical probability difference")
                _require(reproduction["probabilities_bit_identical"] is bool(exact),
                         "Saved historical bit-identity claim is incorrect")
                previous_source_notes = [{key: value for key, value in note.items()
                                          if key not in ("on_delivered_at_seconds", "off_delivered_at_seconds")}
                                         for note in previous["notes"]]
                _equal(notes, previous_source_notes, "Recomputed historical source-note identity")
                _equal(without_delivery_clocks(row["pitch_metrics"]), without_delivery_clocks(previous["pitch_metrics"]),
                       "Historical control note scores")
                _equal(without_delivery_clocks(row["tab_metrics"]), without_delivery_clocks(previous["tab_metrics"]),
                       "Historical control string scores")
                for key in ("exact_state", "active_fret_accuracy", "pitch"):
                    _equal(row["head_metrics"][key], previous["head_metrics"][key], "Historical head decision " + key)
                _require(reproduction["source_notes_identical"] is True
                         and reproduction["note_and_tab_scores_identical"] is True
                         and reproduction["head_decisions_identical"] is True,
                         "Historical decision identity claim changed")
            else:
                _require(row["historical_control_reproduction"] is None,
                         "Treatment was misrepresented as a historical control replay")
    summary = aggregate(records)
    for name in ARMS:
        rows = [row for row in records if row["mode"] == name]
        summary["modes"][name]["exact_state_f1"] = float(np.mean([row["head_metrics"]["exact_state_f1"] for row in rows]))
        summary["modes"][name]["active_fret_accuracy"] = float(np.mean([row["head_metrics"]["active_fret_accuracy"] for row in rows]))
        summary["modes"][name]["auxiliary_pitch_f1"] = float(np.mean([row["head_metrics"]["pitch"]["overall"]["f1"] for row in rows]))
        summary["modes"][name]["low_register_pitch_f1"] = float(np.mean([row["head_metrics"]["pitch"]["low_register"]["f1"] for row in rows]))
    _equal(_read(output / "validation/summary.json"), summary, "Independent validation aggregation")
    return summary, {"validation_records": len(records), "notes": note_count, "events": event_count,
                     "recomputed_saved_probabilities": bool(recompute)}


def audit_artifacts(output, recompute=True):
    """Verify the frozen historical control, new run and validation ledgers."""
    import numpy as np
    import torch
    from acoustic10.audit import validate_sanity
    from baseline.download import ROOT, sha256
    from .model import CausalCNNGRU

    output, workspace = Path(output).resolve(), ROOT.resolve()
    config = _read(workspace / "configs/normalization12.json")
    protocol = _read(output / "protocol.json")
    _equal(protocol["config"], config, "Prespecified Stage12 configuration")
    _source_bindings(protocol["source_sha256"], workspace)
    _require(config["total_optimizer_update_cap"] == 500 and config["planned_new_optimizer_updates"] == 250,
             "Stage12 new-update cap or fixed treatment budget changed")
    _require(protocol["total_new_update_cap"] == 500 and protocol["new_sanity_updates"] == 0
             and protocol["test_inference_allowed"] is False,
             "Prespecified protocol update cap or no-test rule changed")
    _equal(protocol["planned_new_optimizer_updates"], {"group8": 0, "group1": 250}, "Prespecified new updates")
    _equal(protocol["historical_optimizer_updates"], {"group8": 250}, "Prespecified historical updates")
    _require(config["test_inference_allowed"] is False and config["production_promotion"] is False,
             "Stage12 cannot evaluate test clips or promote a product model")
    _equal(config["arms"], [{"name": "group8", "cnn_norm_groups": 8},
                           {"name": "group1", "cnn_norm_groups": 1}], "Normalization arms")
    old_config = _read(workspace / "configs/spectral11.json")
    for key in ("training", "seed", "common_target_delay_frames", "common_leading_mask_frames",
                "decoder_parameters", "validation_packet_ms", "pitch_diagnostics"):
        _equal(config[key], old_config[key], "Historical matched recipe " + key)
    control_reference = protocol["historical_control"]
    control, control_checkpoint, initial, descriptors = validate_historical_control(control_reference, workspace)
    _require(control_reference["historical_optimizer_updates"] == 250
             and control_reference["new_control_optimizer_updates"] == 0,
             "Historical control was counted as new training or retrained")
    verification_path = _relative(workspace, control_reference["verification_path"])
    _require(verification_path == workspace / "outputs/spectral11/verification.json"
             and sha256(verification_path) == control_reference["verification_sha256"],
             "Historical stage verification changed")
    verification = _read(verification_path)
    _require(verification["checkpoint_sha256"]["short128"] == control_reference["checkpoint_sha256"],
             "Historical binary differs from its preserved verification")
    _equal(control_reference["historical_receipt"], control, "Unmodified referenced historical receipt")
    _require(not (output / "models/group8/final.pt").exists(),
             "A new group8 control binary violates the frozen-control reuse protocol")
    _equal(_read(output / "models/group8/reference.json"), control_reference,
           "Reference-only historical-control artifact")
    local_schedule = _read(output / "crop-schedule.json")
    _equal(local_schedule["descriptors"], descriptors, "Stage12 schedule reproduces the original control")
    _require(local_schedule["sha256"] == _fingerprint(descriptors), "Local shared crop schedule changed")
    input_path = output / "input-audit.json"
    inputs = _read(input_path)
    manifest = _read(workspace / "data/cnn_gru/manifest.json")
    _require(len(manifest["train"]) == 48 and len(manifest["validation"]) == 12
             and not ({row["id"] for row in manifest["train"]}
                      & {row["id"] for row in manifest["validation"]}),
             "Expected disjoint forty-eight/twelve train-validation recordings")
    _check_inputs(inputs, manifest, workspace, include_validation=True)
    _require(inputs["config_sha256"] == _fingerprint(config)
             and inputs["new_optimizer_updates_for_preparation"] == 0
             and inputs["features_rewritten"] is False and inputs["targets_rewritten"] is False,
             "Input preparation rewrote frozen caches or used gradients")
    _equal(inputs["control_reference"], control_reference, "Input historical-control reference")
    frozen_cache = _read(workspace / "data/streaming/cache_audit.json")
    _equal(inputs["normalization"], frozen_cache["normalization"], "Frozen input normalization")
    _equal(inputs["positive_weights"], frozen_cache["positive_weights"], "Frozen input loss weights")
    train_records = [{"features": np.load(_relative(workspace, row["prefix"]).with_suffix(".npy"),
                                         mmap_mode="r", allow_pickle=False)} for row in inputs["inputs"]["train"]]
    normalization = validate_normalization(inputs["normalization"], train_records)
    expected_schedule = _verify_schedule(descriptors, config, inputs["inputs"]["train"])
    support = _fingerprint([{"id": row["id"], "frames": row["frames"],
                             "ignored_global_leading_frames": 26, "ignored_crop_edge_frames": 26,
                             "target_delay_frames": 13} for row in inputs["inputs"]["train"]])
    treatment_path = output / "models/group1/final.pt"
    treatment_checkpoint = torch.load(treatment_path, map_location="cpu", weights_only=True)
    treatment = _read(output / "models/group1/training-receipt.json")
    _equal(treatment_checkpoint["receipt"], treatment, "New treatment checkpoint receipt")
    _equal(treatment["config"], config, "New treatment configuration")
    _require(treatment["config_sha256"] == _fingerprint(config), "New treatment config fingerprint changed")
    _equal(treatment["source_sha256"], protocol["source_sha256"], "New treatment source identities")
    _equal(treatment["inputs"], inputs["inputs"], "New treatment original frozen inputs")
    _require(treatment["input_audit_sha256"] == sha256(input_path), "New treatment input-audit bytes changed")
    _equal(treatment["control_reference"], control_reference, "Treatment historical-control reference")
    _require(treatment["cnn_norm_groups"] == 1 and treatment["control_reused"] is False
             and treatment["new_optimizer_steps"] == treatment["optimizer_steps"] == 250,
             "Treatment group count or new update accounting changed")
    receipts = {"group8": control, "group1": treatment}
    budget = validate_budget(receipts)
    identities = matched_arms(control, treatment)
    _require(identities["common_support_sha256"] == support, "Common masked training support changed")
    for key, digest in expected_schedule.items():
        _require(identities[key] == digest, "Deterministic matched crop/gain exposure changed")
    _equal(treatment_checkpoint["config"], treatment["model_config"], "Treatment architecture")
    _equal(treatment_checkpoint["normalization"], inputs["normalization"], "Treatment feature statistics")
    _equal(treatment_checkpoint["positive_weights"], inputs["positive_weights"], "Treatment loss weights")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config["seed"])
        fresh = CausalCNNGRU(treatment["model_config"])
        _require(weight_digest(fresh.state_dict()) == initial["weights_sha256"],
                 "Treatment fresh initial tensors differ from the historical control")
        parameter_count, parameter_tensors = sum(value.numel() for value in fresh.parameters()), len(list(fresh.parameters()))
        _require(parameter_count == treatment["model_parameters"] == 931145, "Treatment parameter count changed")
    _optimizer_receipt(treatment_checkpoint, treatment, parameter_tensors)
    _require(weight_digest(treatment_checkpoint["model"]) != initial["weights_sha256"],
             "Treatment final weights did not change")
    _require(set(treatment_checkpoint["model"]) == set(initial["model"]), "Treatment model-state keys changed")
    for key, value in treatment_checkpoint["model"].items():
        _require(value.shape == initial["model"][key].shape and value.dtype == initial["model"][key].dtype
                 and bool(torch.isfinite(value).all()), "Treatment parameter tensor is invalid")
    training = _read(output / "training-summary.json")
    _equal(training["arms"], receipts, "Matched training-summary receipts")
    _equal(training["control_reference"], control_reference, "Training-summary control reference")
    _equal(training["source_sha256"], protocol["source_sha256"], "Training-summary producer identities")
    _require(training["new_optimizer_updates"] == 250 and training["reused_optimizer_updates"] == 250
             and training["new_sanity_updates"] == 0 and training["test_inference_calls"] == 0
             and training["validation_used_during_training"] is False,
             "Training-summary update accounting or validation scope changed")
    _require(training["input_audit_sha256"] == sha256(input_path), "Training input-audit identity changed")
    run = _read(output / "validation/run.json")
    _equal(run["config"], config, "Validation fixed configuration")
    _equal(run["source_sha256"], protocol["source_sha256"], "Validation producer identities")
    _equal(run["training_summary"], training, "Validation training accounting")
    _require(run["new_optimizer_updates"] == 250 and run["test_used"] is False and run["test_inference_calls"] == 0,
             "Validation scope violates the fixed development protocol")
    _require(run["new_sanity_updates"] == 0, "A new sanity training run entered Stage12")
    _require(run["reused_optimizer_updates"] == 250 and run["input_audit_sha256"] == sha256(input_path),
             "Validation historical accounting or input-audit identity changed")
    _equal(run["historical_control"], control_reference, "Validation historical control reference")
    _require(run["protocol_sha256"] == training["protocol_sha256"] == sha256(output / "protocol.json"),
             "Immutable Stage12 protocol bytes changed")
    _equal(run["input_provenance"]["inputs"], inputs["inputs"], "Validation original input identities")
    prior_sanity = _read(workspace / config["sanity_reference"])
    sanity_verdict = validate_sanity(prior_sanity["initial"], prior_sanity["final"], prior_sanity["coverage"], prior_sanity["passed"])
    _require(sanity_verdict["passed"], "Preserved training-only sanity evidence failed")
    _source_bindings(prior_sanity["source_sha256"], workspace)
    sanity = protocol["sanity_reference"]
    _require(sanity["sha256"] == sha256(workspace / config["sanity_reference"])
             and sanity["checkpoint_sha256"] == sha256(workspace / "outputs/acoustic10/sanity-checkpoint.pt")
             and sanity["new_optimizer_updates"] == 0 and sanity["passed"] is True,
             "Preserved prior sanity identity changed")
    _equal(training["sanity_reference"], sanity, "Training preserved prior-sanity reference")
    _equal(run["sanity_reference"], sanity, "Validation preserved prior-sanity reference")
    checkpoints = {"group8": control_checkpoint, "group1": treatment_checkpoint}
    checkpoint_hashes = {"group8": control_reference["checkpoint_sha256"], "group1": sha256(treatment_path)}
    _equal(run["checkpoint_sha256"], checkpoint_hashes, "Validation checkpoint identities")
    summary, counts = audit_validation_records(output, run, checkpoints, checkpoint_hashes, manifest, inputs["inputs"], recompute)
    selection = _read(output / "selection.json")
    _equal(run["selection"], selection, "Validation selection")
    _equal(selection["checkpoint_sha256"], checkpoint_hashes, "Selection frozen checkpoints")
    _require(selection["selected_from_test"] is False and selection["production_promoted"] is False
             and selection["rule"] == config["selection_rule"], "Selection scope or rule changed")
    scores = {name: {key: summary["modes"][name][key] for key in ("exact_state_f1", "onset_f1")} for name in ARMS}
    _equal(selection["scores"], scores, "Independent validation selection scores")
    _require(selection["selected_arm"] == choose_arm(scores), "Selected group contradicts the prespecified validation rule")
    return {"passed": True, **budget, **counts, "selected_arm": selection["selected_arm"],
            "checkpoint_sha256": checkpoint_hashes, "protocol_sha256": sha256(output / "protocol.json"),
            "matched_initial_weights_sha256": identities["initial_weights_sha256"],
            "normalization": normalization, "test_inference_calls": 0,
            "scope": "Read-only saved-tensor audit; zero audio/model inference or optimizer updates."}


def audit(output, recompute=True):
    """Public CLI entry point; the artifact adapter performs no training."""
    return audit_artifacts(output, recompute=recompute)
