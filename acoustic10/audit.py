"""Independent, read-only checks for the budgeted stage-10 acoustic trial.

The small helpers do not load audio, run model inference, train, or select a
checkpoint. They verify receipts supplied by the experiment producer. The
artifact adapter also checks the saved results against their source identities.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name, low=None, high=None):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool)
             and math.isfinite(value), name + " must be finite numeric data")
    _require(low is None or value >= low, name + " is below its allowed minimum")
    _require(high is None or value <= high, name + " exceeds its allowed maximum")
    return float(value)


def _integer(value, name, low=0):
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= low,
             name + " must be an integer at least " + str(low))
    return value


def weight_digest(state_dict):
    """Hash names, shapes, dtypes and exact tensor bytes in canonical order.

    This deliberately excludes torch serialization metadata and optimizer
    state, so independently serialized identical fresh models have one digest.
    """
    import torch

    _require(isinstance(state_dict, dict) and state_dict, "A nonempty tensor state dictionary is required")
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name]
        _require(isinstance(name, str) and isinstance(value, torch.Tensor), "Model state must map names to tensors")
        _require(value.layout == torch.strided, "Only dense model-state tensors are supported")
        tensor = value.detach().cpu().contiguous()
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode()
        payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        for item in (header, payload):
            digest.update(len(item).to_bytes(8, "big"))
            digest.update(item)
    return digest.hexdigest()


def validate_sanity(initial, final, coverage, passed=None):
    """Recompute the prespecified fixed-training-batch sanity verdict."""
    initial_loss = _number(initial["state_loss"], "Initial state loss", low=0)
    _require(initial_loss > 0, "The initial state loss must be positive")
    final_loss = _number(final["state_loss"], "Final state loss", low=0)
    accuracy = _number(final["active_fret_accuracy"], "Active-fret accuracy", 0, 1)
    f1 = _number(final["exact_state_f1"], "Exact state F1", 0, 1)
    frames = _integer(coverage["active_frames"], "Active labelled frame decisions")
    frets = _integer(coverage["distinct_frets"], "Distinct frets")
    result = (final_loss / initial_loss <= .5 and accuracy >= .8 and f1 >= .75
              and frames >= 100 and frets >= 4)
    if passed is not None:
        _require(type(passed) is bool and passed == result, "Saved sanity verdict contradicts its measurements")
    return {"passed": result, "state_loss_ratio": final_loss / initial_loss,
            "active_frames": frames, "distinct_frets": frets}


def validate_budget(receipts, require_complete=True):
    """Verify actual update/block ledgers, including the separate sanity run."""
    _require(isinstance(receipts, dict) and "sanity" in receipts, "The stage budget must include a sanity receipt")
    allowed = {"sanity": 100, "baseline": 200, "delay60": 200}
    _require(set(receipts).issubset(allowed), "An unbudgeted training run is present")
    if require_complete:
        _require(set(receipts) == set(allowed), "A complete trial requires both matched arms and sanity")
    total, blocks = 0, {}
    for name, receipt in receipts.items():
        steps = _integer(receipt["optimizer_steps"], name + " optimizer updates")
        _require(steps <= allowed[name], name + " exceeded its update budget")
        if require_complete:
            _require(steps == allowed[name], name + " did not finish its fixed update budget")
        history = receipt["history"]
        _require(isinstance(history, list) and 0 < len(history) <= 5, name + " must have one to five recorded training blocks")
        if require_complete:
            _require(len(history) == 5, name + " must finish all five prescribed reporting blocks")
        previous = 0
        for row in history:
            current = _integer(row["optimizer_steps"], name + " block update count", 1)
            _require(previous < current <= steps, name + " block ledger is not strictly increasing within budget")
            previous = current
        _require(previous == steps, name + " block ledger does not end at its actual update count")
        stride = allowed[name] // 5
        _require([row["optimizer_steps"] for row in history]
                 == list(range(stride, steps + 1, stride)),
                 name + " block ledger differs from its prespecified update boundaries")
        if "blocks_completed" in receipt:
            _require(_integer(receipt["blocks_completed"], name + " block count", 1) == len(history),
                     name + " block count differs from its history")
        _require(receipt.get("test_used", False) is False, "Test inference or selection occurred in a training run")
        total += steps
        blocks[name] = len(history)
    _require(total <= 500, "The total stage budget exceeded 500 optimizer updates")
    return {"optimizer_steps": total, "blocks": blocks,
            "complete": set(receipts) == set(allowed) and all(receipts[key]["optimizer_steps"] == limit
                                                            for key, limit in allowed.items())}


def matched_arms(baseline, delayed):
    """Verify fresh initialization, sample exposure, and fixed final models."""
    keys = ("initial_weights_sha256", "crop_schedule_sha256", "gain_schedule_sha256",
            "common_support_sha256", "manifest_sha256", "cache_audit_sha256")
    for key in keys:
        _require(isinstance(baseline.get(key), str) and len(baseline[key]) == 64,
                 "Missing matched-arm identity: " + key)
        _require(baseline[key] == delayed.get(key), "Matched arms differ in " + key)
    for name, receipt in (("baseline", baseline), ("delay60", delayed)):
        _require(receipt.get("warmstarted", False) is False, name + " must start from fresh weights")
        _require(receipt.get("selected_step") == receipt.get("optimizer_steps") == 200,
                 name + " must use the fixed final step-200 model")
        _require(receipt.get("common_leading_mask_frames") == 6,
                 name + " did not share the six-frame leading mask")
    _require(baseline.get("delay_frames") == 0 and delayed.get("delay_frames") == 6,
             "The only scheduled target-delay values are zero and six frames")
    return {key: baseline[key] for key in keys}


def choose_arm(summaries):
    """Apply the prespecified validation comparison; baseline wins exact ties."""
    _require(set(summaries) == {"baseline", "delay60"}, "Exactly two validation arms are required")
    scores = {}
    for name in ("baseline", "delay60"):
        row = summaries[name]
        scores[name] = (_number(row["exact_state_f1"], name + " mean exact-state F1", 0, 1),
                        _number(row["onset_f1"], name + " mean onset F1", 0, 1))
    return "delay60" if scores["delay60"] > scores["baseline"] else "baseline"


def audit_note_ledger(notes, events):
    """Check paired immutable event values and nonnegative source/delivery clocks."""
    groups = defaultdict(dict)
    last_emission = -math.inf
    for event in events:
        kind = event["type"]
        _require(kind in ("note_on", "note_off"), "Unknown event type")
        note_id = _integer(event["note_id"], "Event note ID")
        _require(kind not in groups[note_id], "A note event was published twice")
        if kind == "note_off":
            _require("note_on" in groups[note_id], "A release was published before its attack event")
        source = _number(event["source_time"], "Event source time", 0)
        emission = _number(event["emitted_at_audio_seconds"], "Event emission time", 0)
        _require(source <= emission + 1e-9, "Source time exceeds actual publication availability")
        _require(emission + 1e-9 >= last_emission, "Publication clock went backwards")
        last_emission = emission
        if "delivered_at_seconds" in event:
            _require(_number(event["delivered_at_seconds"], "Event delivery time", 0) + 1e-9 >= emission,
                     "Delivery precedes publication availability")
        _require(math.isclose(source, _number(event["time"], "Event time", 0), rel_tol=0, abs_tol=1e-10),
                 "Published event time differs from source time")
        boundary = "start" if kind == "note_on" else "end"
        _require(math.isclose(source, _number(event[boundary], "Event boundary", 0), rel_tol=0, abs_tol=1e-10),
                 "Published event boundary differs from source time")
        groups[note_id][kind] = event
    seen, strings = set(), defaultdict(list)
    for note in notes:
        note_id = _integer(note["note_id"], "Completed note ID")
        _require(note_id not in seen, "A completed note was duplicated")
        seen.add(note_id)
        _require(set(groups.get(note_id, {})) == {"note_on", "note_off"}, "A completed note lacks exactly two published events")
        start, end = _number(note["start"], "Note attack", 0), _number(note["end"], "Note release", 0)
        _require(end > start, "A completed note has nonpositive source duration")
        string = _integer(note["string"], "Note string")
        _require(string < 6, "A note string exceeds the supported guitar strings")
        fret = _integer(note["fret"], "Note fret")
        _require(fret <= 24, "A note fret exceeds the supported fretboard")
        _require(note["pitch"] == (40, 45, 50, 55, 59, 64)[string] + fret,
                 "A note pitch differs from its standard-tuning string and fret")
        strings[string].append((start, end))
        for kind, boundary in (("note_on", "start"), ("note_off", "end")):
            event = groups[note_id][kind]
            _require(math.isclose(note[boundary], event["source_time"], rel_tol=0, abs_tol=1e-10),
                     "A completed note revises a published source boundary")
            for field in ("string", "fret", "pitch"):
                _require(note[field] == event[field], "A completed note revises a published " + field)
            prefix = "on" if kind == "note_on" else "off"
            for field in ("emitted_at_audio_seconds", "delivered_at_seconds"):
                name = prefix + "_" + field
                if name in note:
                    _require(math.isclose(_number(note[name], "Completed note clock", 0), event[field], rel_tol=0, abs_tol=1e-10),
                             "A completed note revises its published clock")
        _require(groups[note_id]["note_off"]["emitted_at_audio_seconds"] + 1e-9
                 >= groups[note_id]["note_on"]["emitted_at_audio_seconds"],
                 "Release was emitted before its attack")
    _require(seen == set(groups), "The final ledger contains events without completed notes")
    for intervals in strings.values():
        ordered = sorted(intervals)
        _require(all(current[0] + 1e-10 >= previous[1]
                     for previous, current in zip(ordered, ordered[1:])),
                 "Public source notes overlap on one string")
    return {"notes": len(notes), "events": len(events)}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _equal(actual, expected, label, tolerance=1e-9):
    if isinstance(expected, dict):
        _require(isinstance(actual, dict) and set(actual) == set(expected), label + " object fields differ")
        for key in expected:
            _equal(actual[key], expected[key], label + "." + str(key), tolerance)
    elif isinstance(expected, list):
        _require(isinstance(actual, list) and len(actual) == len(expected), label + " list length differs")
        for index, item in enumerate(expected):
            _equal(actual[index], item, label + "[" + str(index) + "]", tolerance)
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        _require(type(actual) is type(expected) and actual == expected, label + " value differs")
    elif isinstance(expected, (int, float)):
        _number(actual, label)
        _number(expected, label)
        _require(math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance), label + " numeric value differs")
    else:
        _require(actual == expected, label + " value differs")


def _relative(root, value):
    path = (Path(root) / value).resolve()
    _require(path.is_relative_to(Path(root).resolve()), "An artifact path leaves its declared root")
    return path


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _source_bindings(value, workspace):
    from baseline.download import sha256

    _require(isinstance(value, dict) and value, "Source identities are missing")
    for name, digest in value.items():
        _require(sha256(_relative(workspace, name)) == digest, "Prespecified source changed: " + name)


def _independent_heads(probability, target, mask, inactive):
    """Recompute weighted CE and exact-state decisions without producer metrics."""
    import numpy as np

    target, mask = np.asarray(target), np.asarray(mask)
    valid = (mask > 0)[:, None] & (target != -100)
    estimate = np.asarray(probability).argmax(axis=-1)
    expected = valid & (target != inactive)
    predicted = valid & (estimate != inactive)
    tp = int((expected & predicted & (target == estimate)).sum())
    reference, observed = int(expected.sum()), int(predicted.sum())
    precision = tp / observed if observed else 0.
    recall = tp / reference if reference else 0.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    likelihood = np.take_along_axis(probability, np.maximum(target, 0)[..., None], axis=-1)[..., 0]
    weight = np.where(target == inactive, .35, 1.) * valid
    state_loss = float((-np.log(np.maximum(likelihood, 1e-12)) * weight).sum() / max(weight.sum(), 1))
    return {"exact_state": {"precision": precision, "recall": recall, "f1": f1,
                            "true_positives": tp, "reference_count": reference, "predicted_count": observed},
            "exact_state_f1": f1, "active_fret_accuracy": tp / reference if reference else 0.,
            "active_frames": reference, "state_loss": state_loss,
            "distinct_frets": int(len(np.unique(target[expected]))) if reference else 0}


def _check_inputs(provenance, manifest, workspace, include_validation):
    from baseline.download import sha256

    _require(provenance["manifest_sha256"] == sha256(workspace / "data/cnn_gru/manifest.json"), "Manifest changed")
    _require(provenance["cache_audit_sha256"] == sha256(workspace / "data/streaming/cache_audit.json"), "Frozen cache audit changed")
    _require(provenance["frozen_parent_checkpoint_sha256"] == sha256(workspace / "outputs/streaming/model/best-inference.pt"),
             "The frozen stage-8 parent changed")
    previous = _read(workspace / "outputs/diagnosis9/validation-head-diagnostics.json")
    _require(provenance["frozen_parent_checkpoint_sha256"] == previous["checkpoint_sha256"],
             "The preserved parent differs from its stage-9 frozen identity")
    _require(provenance["test_audio_loaded"] is False and provenance["test_inference_calls"] == 0,
             "Test audio or inference entered stage 10")
    wanted = {"train", "validation"} if include_validation else {"train"}
    _require(set(provenance["inputs"]) == wanted, "Unexpected input split in experiment provenance")
    for split in wanted:
        expected = {entry["id"]: entry for entry in manifest[split]}
        rows = provenance["inputs"][split]
        _require(len(rows) == len(expected) and {row["id"] for row in rows} == set(expected), "Input records differ from the source split")
        for row in rows:
            entry = expected[row["id"]]
            for kind in ("audio", "annotation"):
                _require(row[kind + "_sha256"] == entry[kind + "_sha256"]
                         == sha256(_relative(workspace, entry[kind])), "Original " + split + " " + kind + " changed")
            prefix = _relative(workspace, row["prefix"])
            for kind, extension in (("features", ".npy"), ("targets", ".npz")):
                _require(sha256(prefix.with_suffix(extension)) == row[kind + "_sha256"], "A frozen " + split + " cache changed")


def _verify_schedule(schedule, config, train_inputs):
    """Regenerate every deterministic crop and gain without opening audio."""
    import numpy as np
    import torch

    expected = []
    spec = config["training"]
    length = round(spec["crop_seconds"] * 100)
    for epoch in range(spec["epochs"]):
        generator = torch.Generator().manual_seed(config["seed"] + epoch)
        torch.empty((), dtype=torch.int64).random_(generator=generator)
        order = torch.randperm(spec["samples_per_epoch"], generator=generator).tolist()
        for index in order:
            rng = np.random.default_rng(config["seed"] + epoch * 100000 + index)
            record_index = int(rng.integers(len(train_inputs)))
            record = train_inputs[record_index]
            start = int(rng.integers(record["frames"] - length + 1))
            gain = float(rng.normal(0, spec["gain_augmentation_log_std"]))
            expected.append({"epoch": epoch + 1, "sample_index": index, "record_index": record_index,
                             "id": record["id"], "start_frame": start, "frames": length, "gain_log": gain})
    _equal(schedule, expected, "Prespecified matched crop/gain schedule", tolerance=0)
    return {"crop_schedule_sha256": _fingerprint(expected),
            "gain_schedule_sha256": _fingerprint([row["gain_log"] for row in expected])}


def audit_artifacts(output, recompute=False):
    """Verify stage receipts, frozen models and the validation-only result set.

    ``recompute`` additionally decodes saved exact packet probabilities on the
    CPU and recomputes aligned heads, note/string scores and matched-event
    latency. It performs no neural-network inference or optimization.
    """
    import numpy as np
    import torch

    from baseline.download import ROOT, sha256
    from baseline.evaluation import evaluate, reference_notes
    from cnn_gru.inference import tablature_metrics
    from streaming.evaluate import aggregate
    from streaming.latency import measure
    from streaming.model import CausalCNNGRU
    from .decoder import SourceTimeDecoder

    output, workspace = Path(output).resolve(), ROOT.resolve()
    protocol = _read(output / "protocol.json")
    config = _read(workspace / "configs/acoustic10.json")
    _equal(protocol["config"], config, "Prespecified configuration")
    _require(protocol["total_update_cap"] == 500 and protocol["test_inference_allowed"] is False,
             "Stage protocol lost its update cap or no-test rule")
    _equal(protocol["planned_optimizer_updates"], {"sanity": 100, "baseline": 200, "delay60": 200}, "Planned stage budget")
    _source_bindings(protocol["source_sha256"], workspace)
    _require(config["test_inference_allowed"] is False and config["production_promotion"] is False,
             "Stage 10 cannot evaluate test clips or promote a product model")
    manifest = _read(workspace / "data/cnn_gru/manifest.json")
    base = _read(workspace / config["model_config"])
    cache = _read(workspace / "data/streaming/cache_audit.json")
    _require(cache["config_sha256"] == _fingerprint(base), "Frozen feature/model configuration changed")
    sanity = _read(output / "sanity.json")
    _equal(sanity["config"], config, "Sanity configuration")
    _equal(sanity["criteria"], config["sanity"], "Sanity pass criterion")
    _equal(sanity["source_sha256"], protocol["source_sha256"], "Sanity producer identities")
    _check_inputs(sanity, manifest, workspace, include_validation=False)
    sanity_verdict = validate_sanity(sanity["initial"], sanity["final"], sanity["coverage"], sanity["passed"])
    _equal(sanity["state_loss_ratio"], sanity_verdict["state_loss_ratio"], "Sanity state-loss ratio")
    sanity_checkpoint = torch.load(output / "sanity-checkpoint.pt", map_location="cpu", weights_only=True)
    _require(sanity_checkpoint["step"] == sanity["optimizer_steps"] == 100, "Sanity checkpoint update count changed")
    _equal(sanity_checkpoint["source_sha256"], protocol["source_sha256"], "Sanity checkpoint sources")
    _equal(sanity_checkpoint["descriptors"], sanity["descriptors"], "Sanity fixed training crops")
    _require(sanity_checkpoint["config"]["model"]["dropout"] == 0., "Sanity dropout was not zero")
    _require(all(row["gain_log"] == 0. for row in sanity["descriptors"]), "Sanity training was augmented")
    _require(len(sanity["descriptors"]) == 4, "The tiny sanity run must repeat exactly four fixed crops")
    expected_descriptors = []
    train_inputs = sanity["inputs"]["train"]
    for index in config["sanity"]["train_entry_indices"]:
        for seconds in config["sanity"]["crop_starts_seconds"]:
            expected_descriptors.append({"epoch": 0, "sample_index": 0, "record_index": index,
                                         "id": train_inputs[index]["id"], "start_frame": round(seconds * 100),
                                         "frames": round(config["sanity"]["crop_seconds"] * 100), "gain_log": 0.})
    _equal(sanity["descriptors"], expected_descriptors, "Prespecified sanity fixed crop batch")
    active_values = []
    for descriptor in expected_descriptors:
        prefix = _relative(workspace, train_inputs[descriptor["record_index"]]["prefix"])
        with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as arrays:
            start, length = descriptor["start_frame"], descriptor["frames"]
            state = arrays["state"][start:start + length].copy()
        mask = np.ones(length, bool)
        mask[:max(0, 6 - start)] = False
        edge = config["training"]["edge_mask_frames"]
        if edge:
            mask[:edge] = mask[-edge:] = False
        active = mask[:, None] & (state >= 0) & (state < 25)
        active_values.extend(state[active].tolist())
    coverage = {"active_frames": len(active_values), "distinct_frets": len(set(active_values))}
    _equal(sanity["coverage"], coverage, "Independent sanity active-label coverage")
    _equal({key: sanity["final"][key] for key in coverage}, coverage, "Final sanity fixed-label coverage")
    summary = _read(output / "training-summary.json")
    _require(summary["test_inference_calls"] == 0, "Test inference entered the training summary")
    _equal(summary["source_sha256"], protocol["source_sha256"], "Training-summary sources")
    if not sanity["passed"]:
        budget = validate_budget({"sanity": sanity}, require_complete=False)
        _require(summary["new_optimizer_updates"] == 100 and summary["arms_trained"] == [], "Training continued after sanity failure")
        _require(not (output / "initial-weights.pt").exists() and not (output / "selection.json").exists(),
                 "Matched-arm artifacts exist despite failed sanity")
        return {"passed": True, "sanity_passed": False, **budget, "test_inference_calls": 0,
                "validation_records": 0, "scope": "Read-only verification; no new training or inference."}

    receipts = {name: _read(output / "models" / name / "training-receipt.json") for name in ("baseline", "delay60")}
    budget = validate_budget({"sanity": sanity, **receipts})
    _require(summary["new_optimizer_updates"] == budget["optimizer_steps"] == 500, "Total recorded stage budget differs")
    _equal(summary["arms"], receipts, "Saved matched training receipts")
    _require(summary["sanity_passed"] is True and summary["validation_used_during_training"] is False,
             "Training violated its sanity or no-validation-learning contract")
    identities = matched_arms(receipts["baseline"], receipts["delay60"])
    expected_model_config = copy.deepcopy(base)
    expected_model_config["training"].update(config["training"])
    expected_model_config["stage_budget"] = {"total_max_optimizer_updates": 500, "sanity_updates": 100,
                                             "arm_updates": 200, "purpose": "Stage 10 separate matched acoustic trial."}
    for name in ("baseline", "delay60"):
        _equal(receipts[name]["model_config"], expected_model_config, name + " matched model architecture/training settings")
    initial_path = output / "initial-weights.pt"
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    _require(initial["sanity_warmstart"] is False and initial["origin"] == "fresh random initialization",
             "The matched arms were warm-started from sanity")
    _require(weight_digest(initial["model"]) == initial["weights_sha256"] == identities["initial_weights_sha256"],
             "Initial model weights differ from matched receipts")
    # Construct a fresh seeded model within an isolated RNG context, without inference.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config["seed"])
        fresh = CausalCNNGRU(receipts["baseline"]["model_config"])
        _require(weight_digest(fresh.state_dict()) == initial["weights_sha256"], "Saved initial model is not fresh seed-42 initialization")
        del fresh
    schedule = _read(output / "crop-schedule.json")
    _require(schedule["sha256"] == _fingerprint(schedule["descriptors"]) == identities["crop_schedule_sha256"],
             "Matched training crop schedule changed")
    _check_inputs(receipts["baseline"], manifest, workspace, include_validation=True)
    expected_schedule = _verify_schedule(schedule["descriptors"], config, receipts["baseline"]["inputs"]["train"])
    for key, digest in expected_schedule.items():
        _require(identities[key] == digest, "Matched exposure differs from the prespecified schedule")
    support = _fingerprint([{"id": row["id"], "frames": row["frames"],
                             "ignored_global_leading_frames": 6,
                             "ignored_crop_edge_frames": config["training"]["edge_mask_frames"]}
                            for row in receipts["baseline"]["inputs"]["train"]])
    _require(support == identities["common_support_sha256"], "Shared masked training support changed")
    checkpoints, checkpoint_hashes = {}, {}
    for name, receipt in receipts.items():
        _equal(receipt["source_sha256"], protocol["source_sha256"], name + " producer sources")
        _equal(receipt["config"], config, name + " configuration")
        _require(receipt["config_sha256"] == _fingerprint(config), name + " configuration digest changed")
        _require(receipt["initial_checkpoint_sha256"] == sha256(initial_path), name + " initial checkpoint changed")
        _require(receipt["warm_started_from_sanity"] is False and receipt["early_stopping_used"] is False
                 and receipt["validation_used_during_training"] is False, name + " training protocol changed")
        _equal(receipt["inputs"], receipts["baseline"]["inputs"], name + " inputs")
        _equal(receipt["normalization"], cache["normalization"], name + " frozen training normalization")
        _equal(receipt["positive_weights"], cache["positive_weights"], name + " fixed class balancing")
        path = output / "models" / name / "final.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        _require(checkpoint["step"] == receipt["selected_step"] == 200, name + " final checkpoint is not step 200")
        _equal(checkpoint["receipt"], receipt, name + " final model receipt")
        _equal(checkpoint["config"], receipt["model_config"], name + " final model configuration")
        _require(weight_digest(checkpoint["model"]) != initial["weights_sha256"], name + " final weights did not change")
        _require(set(checkpoint["model"]) == set(initial["model"]), name + " model-state keys changed")
        for key, tensor in checkpoint["model"].items():
            _require(tensor.shape == initial["model"][key].shape and tensor.dtype == initial["model"][key].dtype,
                     name + " final model parameter shape or dtype changed")
            _require(bool(torch.isfinite(tensor).all()), name + " final model contains nonfinite parameters")
        checkpoints[name], checkpoint_hashes[name] = checkpoint, sha256(path)

    run = _read(output / "validation/run.json")
    _require(run["test_used"] is False and run["test_inference_calls"] == 0
             and run["new_optimizer_updates"] == 500, "Validation provenance violates the stage protocol")
    _equal(run["source_sha256"], protocol["source_sha256"], "Validation producer sources")
    _equal(run["config"], config, "Validation configuration")
    score_config = _read(workspace / "configs/stage1.json")
    score_config["selection_role"] = "Stage10 validation-only development comparison. No test inference or fresh generalization claim."
    _equal(run["scoring_config"], score_config, "Fixed original source-scoring settings")
    parent = torch.load(workspace / "outputs/streaming/model/best-inference.pt", map_location="cpu", weights_only=True)
    parent_scoring = {key: value for key, value in parent["scoring_config"].items() if key != "selection_role"}
    _equal({key: run["scoring_config"][key] for key in parent_scoring}, parent_scoring,
           "Preserved parent evaluation tolerances")
    del parent
    _require(run["protocol_sha256"] == summary["protocol_sha256"] == sha256(output / "protocol.json"), "Immutable protocol file changed")
    _equal(run["checkpoint_sha256"], checkpoint_hashes, "Validation checkpoints")
    _equal(run["training_summary"], summary, "Validation training accounting")
    _check_inputs(run["input_provenance"], manifest, workspace, include_validation=True)
    selection = _read(output / "selection.json")
    _equal(run["selection"], selection, "Saved validation selection")
    _equal(selection["checkpoint_sha256"], checkpoint_hashes, "Selection checkpoints")
    _require(selection["selected_from_test"] is False and selection["production_promoted"] is False,
             "The development comparison used test data or promoted a product model")
    _require(selection["rule"] == config["selection_rule"], "The prespecified selection rule changed")
    records = [_read(path) for path in sorted((output / "validation/results").glob("*/*-metrics.json"))]
    expected_ids = {entry["id"] for entry in manifest["validation"]}
    _require(len(records) == 24 and len(expected_ids) == 12
             and {(row["id"], row["mode"]) for row in records}
             == {(clip_id, mode) for clip_id in expected_ids for mode in ("baseline", "delay60")},
             "The result set must contain exactly twelve validation clips per arm")
    note_count = event_count = 0
    validation_inputs = {row["id"]: row for row in run["input_provenance"]["inputs"]["validation"]}
    for row in records:
        name, clip_id = row["mode"], row["id"]
        _equal(row["input_provenance"], validation_inputs[clip_id], "Per-record validation-only input identity")
        stem = output / "validation/results" / name / clip_id
        notes_file = _read(stem.with_suffix(".json"))
        _equal(notes_file["notes"], row["notes"], "Exported exact source notes")
        _require(notes_file["metadata"]["checkpoint_sha256"] == checkpoint_hashes[name], "Export checkpoint identity changed")
        _require(notes_file["metadata"]["mode"] == name, "Export arm identity changed")
        _equal(notes_file["metadata"]["decoder_parameters"], config["decoder_parameters"], "Fixed decoder parameters")
        delay = 0 if name == "baseline" else 6
        _equal(notes_file["metadata"]["source_delay_seconds"], delay / 100, "Source delay compensation")
        _equal(_read(stem.with_name(stem.name + "-events.json"))["events"], row["events"], "Published exact events")
        ledger = audit_note_ledger(row["notes"], row["events"])
        note_count += ledger["notes"]
        event_count += ledger["events"]
        import mido
        midi = mido.MidiFile(stem.with_suffix(".mid"))
        attacks = Counter(message.note for track in midi.tracks for message in track
                          if message.type == "note_on" and message.velocity > 0)
        _require(attacks == Counter(note["pitch"] for note in row["notes"]), "MIDI export changed or dropped published attacks")
        probability_path = _relative(output, row["probability_path"])
        _require(sha256(probability_path) == row["probability_sha256"], "Saved exact acoustic probabilities changed")
        _require(row["timing"]["lookahead_samples"] == 0 and row["timing"]["packet_ms"] == config["validation_packet_ms"],
                 "Validation lookahead or packet size changed")
        _require(row["timing"]["exact_probability_recording_included"] is True,
                 "Saved acoustic probabilities do not represent the measured exact packet decoder")
        if recompute:
            entry = next(entry for entry in manifest["validation"] if entry["id"] == clip_id)
            reference = reference_notes(_relative(workspace, entry["annotation"]))
            _equal(row["reference"], reference, "Original source references")
            duration = row["timing"]["input_seconds"]
            with np.load(probability_path, allow_pickle=False) as arrays:
                probabilities = {key: arrays[key].copy() for key in arrays.files}
            decoder = SourceTimeDecoder(checkpoints[name]["config"], config["decoder_parameters"], delay / 100)
            packet_frames = round(config["validation_packet_ms"] / 10)
            length = len(probabilities["state"])
            for start in range(0, length, packet_frames):
                end = min(length, start + packet_frames)
                decoder.push({key: value[start:end] for key, value in probabilities.items()}, end / 100)
            decoder.finish(duration, duration)
            # Measured delivery completion was attached after source decoding.
            saved_notes = [{key: value for key, value in note.items()
                            if key not in ("on_delivered_at_seconds", "off_delivered_at_seconds")}
                           for note in row["notes"]]
            saved_events = [{key: value for key, value in event.items() if key != "delivered_at_seconds"}
                            for event in row["events"]]
            _equal(saved_notes, decoder.notes, "Recomputed immutable source notes")
            _equal(saved_events, decoder.events, "Recomputed packet publication events")
            _equal(row["timing"]["source_clamps"], decoder.clamp_counts, "Source boundary clamps")
            prefix = _relative(workspace, row["input_provenance"]["prefix"])
            with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as targets:
                count = min(round(run["scoring_config"]["score_seconds"] * 100), len(targets["state"]), length - delay)
                mask = np.ones(count, np.float32)
                mask[:6] = 0
                heads = _independent_heads(probabilities["state"][delay:delay + count], targets["state"][:count], mask, 25)
            _equal({key: row["head_metrics"][key] for key in heads}, heads, "Independent aligned acoustic metrics")
            _require(row["head_metrics"]["source_frames_scored"] == count
                     and row["head_metrics"]["prediction_start_frame"] == delay, "Aligned evaluation frame span changed")
            _equal(row["pitch_metrics"], evaluate(reference, row["notes"], run["scoring_config"], duration), "Recomputed note/pitch metrics")
            _equal(row["tab_metrics"], tablature_metrics(reference, row["notes"], run["scoring_config"],
                                                       run["scoring_config"]["score_seconds"], duration), "Recomputed string metrics")
            _equal(row["latency"], measure(reference, row["notes"], run["scoring_config"], duration), "Recomputed matched-event latency")
    expected_summary = aggregate(records)
    for name in ("baseline", "delay60"):
        rows = [row for row in records if row["mode"] == name]
        expected_summary["modes"][name]["exact_state_f1"] = float(np.mean([row["head_metrics"]["exact_state_f1"] for row in rows]))
        expected_summary["modes"][name]["active_fret_accuracy"] = float(np.mean([row["head_metrics"]["active_fret_accuracy"] for row in rows]))
    _equal(_read(output / "validation/summary.json"), expected_summary, "Independent validation aggregation")
    scores = {name: {key: expected_summary["modes"][name][key] for key in ("exact_state_f1", "onset_f1")}
              for name in ("baseline", "delay60")}
    _equal(selection["scores"], scores, "Independent selection scores")
    _require(selection["selected_arm"] == choose_arm(scores), "Saved development model contradicts the fixed validation rule")
    return {"passed": True, "sanity_passed": True, **budget, "selected_arm": selection["selected_arm"],
            "validation_records": len(records), "notes": note_count, "events": event_count,
            "checkpoint_sha256": checkpoint_hashes, "protocol_sha256": sha256(output / "protocol.json"),
            "matched_initial_weights_sha256": identities["initial_weights_sha256"],
            "recomputed_saved_probabilities": bool(recompute), "test_inference_calls": 0,
            "scope": "Read-only integrity audit; no new training, acoustic inference or model promotion."}


def audit(output, recompute=True):
    """Short public entry point used by the experiment CLI."""
    return audit_artifacts(output, recompute=recompute)
