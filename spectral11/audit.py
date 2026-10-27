"""Read-only integrity checks for the matched causal-window experiment.

The checks use saved tensors and probability ledgers. They perform no neural
network inference, optimization, checkpoint selection, or audio preparation.
"""

from __future__ import annotations

from collections import Counter
import copy
import math
from pathlib import Path

from acoustic10.audit import (_equal, _fingerprint, _integer, _number, _read,
                              _relative, _require, _source_bindings,
                              _check_inputs, _verify_schedule, validate_sanity,
                              audit_note_ledger, weight_digest)


ARMS = ("short128", "long256")


def validate_budget(receipts, require_complete=True):
    """Count the two fixed 250-update receipts and reject unbudgeted runs."""
    _require(isinstance(receipts, dict) and bool(receipts), "Training receipts are missing")
    _require(set(receipts).issubset(ARMS), "An unbudgeted training run is present")
    if require_complete:
        _require(set(receipts) == set(ARMS), "Both matched window arms are required")
    total = 0
    blocks = {}
    for name, row in receipts.items():
        steps = _integer(row["optimizer_steps"], name + " optimizer updates", 1)
        _require(steps <= 250, name + " exceeded its update budget")
        if require_complete:
            _require(steps == 250, name + " did not finish its fixed update budget")
        history = row["history"]
        _require(isinstance(history, list) and 0 < len(history) <= 5,
                 name + " must record one to five training blocks")
        counts = [_integer(block["optimizer_steps"], name + " block count", 1) for block in history]
        _require(counts == list(range(50, steps + 1, 50)) and counts[-1] == steps,
                 name + " block ledger differs from its fixed update boundaries")
        if "blocks_completed" in row:
            _require(row["blocks_completed"] == len(history), name + " block count changed")
        _require(row.get("test_used", False) is False and row.get("test_inference_calls", 0) == 0,
                 "Test inference entered training")
        actual = row["actual_optimizer_step_counters"]
        _integer(actual["minimum"], name + " minimum parameter update count", 1)
        _integer(actual["maximum"], name + " maximum parameter update count", 1)
        _require(actual["minimum"] == actual["maximum"] == steps,
                 name + " loop count differs from actual optimizer parameter updates")
        for block in history:
            if "actual_optimizer_step_counters" in block:
                counters = block["actual_optimizer_step_counters"]
                _require(counters["minimum"] == counters["maximum"] == block["optimizer_steps"],
                         name + " block count differs from actual optimizer updates")
        total += steps
        blocks[name] = len(history)
    _require(total <= 500, "The total stage budget exceeded 500 optimizer updates")
    return {"optimizer_steps": total, "blocks": blocks,
            "complete": set(receipts) == set(ARMS) and all(row["optimizer_steps"] == 250
                                                            for row in receipts.values())}


def matched_arms(short, long):
    """Check matching independent of allowed per-arm frontend normalization."""
    identities = ("initial_weights_sha256", "crop_schedule_sha256", "gain_schedule_sha256",
                  "common_support_sha256", "manifest_sha256", "cache_audit_sha256")
    for key in identities:
        _require(isinstance(short.get(key), str) and len(short[key]) == 64,
                 "Missing matched-arm identity: " + key)
        _require(short[key] == long.get(key), "Matched arms differ in " + key)
    for name, row in zip(ARMS, (short, long)):
        _require(row.get("warmstarted", False) is False and row.get("warm_started_from_sanity", False) is False,
                 name + " must start from fresh weights")
        _require(row.get("selected_step") == row.get("optimizer_steps") == 250,
                 name + " must use its fixed final step-250 model")
        _require(row.get("target_delay_frames") == 13,
                 "Both arms must share the thirteen-frame target delay")
        _require(row.get("common_leading_mask_frames") == 26,
                 "Both arms must share the twenty-six-frame warmup mask")
        _require(row.get("early_stopping_used", False) is False
                 and row.get("validation_used_during_training", False) is False,
                 name + " used validation learning or early stopping")
    return {key: short[key] for key in identities}


def choose_arm(summaries):
    """Validation exact-state F1 first, onset F1 second, short window on ties."""
    _require(set(summaries) == set(ARMS), "Exactly two validation arms are required")
    values = {}
    for name in ARMS:
        row = summaries[name]
        values[name] = (_number(row["exact_state_f1"], name + " exact-state F1", 0, 1),
                        _number(row["onset_f1"], name + " onset F1", 0, 1))
    return ARMS[1] if values[ARMS[1]] > values[ARMS[0]] else ARMS[0]


def independent_heads(probability, target, mask, inactive=25):
    """Recompute exact string/fret and weighted CE without producer metrics."""
    import numpy as np

    probability, target, mask = np.asarray(probability), np.asarray(target), np.asarray(mask)
    _require(target.ndim == 2 and target.shape[1] == 6 and mask.shape == (len(target),),
             "Head target/mask shape changed")
    _require(probability.shape == target.shape + (inactive + 1,), "Head probability shape changed")
    _require(np.isfinite(probability).all() and np.all((probability >= 0) & (probability <= 1)),
             "Head probability must be finite and between zero and one")
    _require(np.allclose(probability.sum(-1), 1, rtol=0, atol=2e-6), "State probabilities are not normalized")
    _require(np.isfinite(mask).all() and np.all((mask >= 0) & (mask <= 1)), "Invalid head support mask")
    _require(np.issubdtype(target.dtype, np.integer)
             and np.all((target == -100) | ((target >= 0) & (target <= inactive))),
             "Unsupported head target class")
    valid = (mask > 0)[:, None] & (target != -100)
    estimate = probability.argmax(axis=-1)
    expected = valid & (target != inactive)
    predicted = valid & (estimate != inactive)
    tp = int((expected & predicted & (target == estimate)).sum())
    reference, observed = int(expected.sum()), int(predicted.sum())
    precision, recall = tp / observed if observed else 0., tp / reference if reference else 0.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    likelihood = np.take_along_axis(probability, np.maximum(target, 0)[..., None], axis=-1)[..., 0]
    weights = np.where(target == inactive, .35, 1.) * valid
    loss = float((-np.log(np.maximum(likelihood, 1e-12)) * weights).sum() / max(weights.sum(), 1))
    return {"exact_state": {"precision": precision, "recall": recall, "f1": f1,
                            "true_positives": tp, "reference_count": reference, "predicted_count": observed},
            "exact_state_f1": f1, "active_fret_accuracy": recall, "active_frames": reference,
            "state_loss": loss,
            "distinct_frets": int(len(np.unique(target[expected]))) if reference else 0}


def aligned_heads(probabilities, targets, score_frames=1000, delay_frames=13, leading_mask_frames=26):
    """Score identical source support; translate prediction indices once only."""
    import numpy as np

    _require(delay_frames == 13 and leading_mask_frames == 26,
             "The common source alignment must remain thirteen/twenty-six frames")
    count = min(score_frames, len(targets["state"]), len(probabilities["state"]) - delay_frames)
    _require(count >= 0, "Insufficient probability support for common source alignment")
    mask = np.ones(count, np.float32)
    mask[:leading_mask_frames] = 0
    result = independent_heads(probabilities["state"][delay_frames:delay_frames + count],
                               targets["state"][:count], mask)
    return {**result, "source_frames_scored": count, "prediction_start_frame": delay_frames}


def aligned_pitch(probabilities, targets, score_frames=1000, delay_frames=13, leading_mask_frames=26):
    """Fixed .5 auxiliary-pitch diagnostics on the same original source grid."""
    import numpy as np

    _require(delay_frames == 13 and leading_mask_frames == 26, "Pitch support alignment changed")
    probability, target = np.asarray(probabilities["pitch"]), np.asarray(targets["pitch"])
    _require(probability.ndim == 2 and probability.shape[1] == 49
             and target.ndim == 2 and target.shape[1] == 49, "Auxiliary pitch shape changed")
    _require(np.isfinite(probability).all() and np.all((probability >= 0) & (probability <= 1)),
             "Auxiliary pitch probabilities must be finite between zero and one")
    count = min(score_frames, len(target), len(probability) - delay_frames)
    _require(count >= 0, "Insufficient auxiliary pitch support")
    target = target[:count] > .5
    estimate = probability[delay_frames:delay_frames + count] >= .5
    target, estimate = target[leading_mask_frames:], estimate[leading_mask_frames:]

    def binary(expected, observed):
        tp, reference, predicted = int((expected & observed).sum()), int(expected.sum()), int(observed.sum())
        precision, recall = tp / predicted if predicted else 0., tp / reference if reference else 0.
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
        return {"precision": precision, "recall": recall, "f1": f1, "true_positives": tp,
                "reference_count": reference, "predicted_count": predicted}

    return {"overall": binary(target, estimate), "low_register": binary(target[:, :12], estimate[:, :12]),
            "threshold": .5, "low_register_midi": [40, 51]}


def validate_normalization(normalization, records):
    """Recompute the frozen stage-8 statistics rule using all train frames.

    The common loss mask does not alter normalization support. Squaring uses
    float32 cached values and accumulation uses float64, matching the producer
    convention; the minimum variance is .09 (minimum standard deviation .3).
    """
    import numpy as np

    total = 0
    sums = np.zeros(216, np.float64)
    squares = np.zeros(216, np.float64)
    _require(len(records) == 48, "Normalization requires exactly the forty-eight training clips")
    for record in records:
        values = np.asarray(record["features"], dtype=np.float32)
        _require(values.ndim == 2 and values.shape[1] == 216 and len(values) > 0
                 and np.isfinite(values).all(), "Invalid normalization training cache")
        sums += values.sum(0, dtype=np.float64)
        squares += np.square(values).sum(0, dtype=np.float64)
        total += len(values)
    mean = sums / total
    std = np.sqrt(np.maximum(squares / total - np.square(mean), .09))
    _require(np.asarray(normalization["mean"]).shape == (216,)
             and np.asarray(normalization["std"]).shape == (216,), "Normalization bin count changed")
    _require(np.allclose(normalization["mean"], mean, rtol=0, atol=2e-6)
             and np.allclose(normalization["std"], std, rtol=0, atol=2e-6),
             "Normalization differs from training-only cached features")
    return {"training_clips": len(records), "training_frames": total, "excluded_first_frames": 0,
            "variance_floor": .09, "square_precision": "float32", "accumulation_precision": "float64"}


def _cached_inputs(cache, provenance, manifest, workspace):
    """Bind caches to original sources and independently recompute statistics."""
    import numpy as np
    from baseline.download import sha256

    _require(cache["test_audio_loaded"] is False and cache["new_optimizer_updates"] == 0,
             "Cache preparation accessed test data or training")
    _equal(cache["provenance"]["spectral_receipts"], cache["arms"], "Cache arm receipts")
    _equal(provenance["spectral_receipts"], cache["arms"], "Experiment cache receipts")
    _require(provenance["original_targets_rewritten"] is False, "Original source labels were rewritten")
    _check_inputs(provenance, manifest, workspace, include_validation=True)
    _require(provenance["cache_audit_sha256_stage8"] == provenance["cache_audit_sha256"],
             "The frozen shared stage-8 audit changed")
    expected_configs = {}
    base = _read(workspace / "configs/streaming.json")
    frozen = _read(workspace / "data/streaming/cache_audit.json")
    for name, window in zip(ARMS, (2048, 4096)):
        expected_configs[name] = copy.deepcopy(base)
        expected_configs[name]["feature"].update(window_length=window, n_fft=window)
    _equal(cache["model_configs"], expected_configs, "Only the analysis window/FFT may differ")
    _equal(cache["positive_weights"], frozen["positive_weights"], "Common frozen positive loss weights")
    _equal(cache["normalization"]["short128"], frozen["normalization"], "Byte-identical short-arm normalization")
    datasets = {name: {} for name in ARMS}
    for name in ARMS:
        _require(set(cache["arms"][name]) == {"train", "validation"}, "Unexpected cache split")
        for split, expected_count in (("train", 48), ("validation", 12)):
            rows = cache["arms"][name][split]
            entries = {entry["id"]: entry for entry in manifest[split]}
            source_rows = {entry["id"]: entry for entry in provenance["inputs"][split]}
            _require(len(rows) == len(entries) == expected_count and {row["id"] for row in rows} == set(entries),
                     "Missing or duplicate per-arm " + split + " caches")
            datasets[name][split] = []
            for row in rows:
                entry, source = entries[row["id"]], source_rows[row["id"]]
                for kind in ("audio", "annotation"):
                    _require(row[kind + "_sha256"] == entry[kind + "_sha256"], "Source identity changed")
                feature_path = _relative(workspace, row["features_path"])
                target_path = _relative(workspace, row["targets_path"])
                _require(sha256(feature_path) == row["features_sha256"], "Spectral feature bytes changed")
                _require(sha256(target_path) == row["targets_sha256"] == source["targets_sha256"],
                         "Shared original targets changed")
                _require(target_path == _relative(workspace, source["prefix"] + ".npz"),
                         "An arm used alternate target labels")
                values = np.load(feature_path, mmap_mode="r", allow_pickle=False)
                _require(values.shape == (source["frames"], 216) and values.dtype == np.float32
                         and np.isfinite(values).all(), "Invalid spectral cache dimensions or values")
                _require(row["frames"] == source["frames"] and row["duration"] == source["duration"],
                         "Matched cache support changed")
                if name == "short128":
                    _require(row["byte_identical_frozen_features"] is True
                             and feature_path == _relative(workspace, source["prefix"] + ".npy")
                             and row["features_sha256"] == source["features_sha256"],
                             "The short arm did not reuse frozen feature bytes")
                else:
                    _require(row["source_features_sha256"] == source["features_sha256"],
                             "Long cache lost its frozen short-feature source identity")
                    receipt_path = _relative(workspace, row["receipt_path"])
                    _require(sha256(receipt_path) == row["receipt_sha256"], "Long cache receipt changed")
                    receipt = _read(receipt_path)
                    _require(receipt["features_sha256"] == row["features_sha256"]
                             and receipt["full_vs_20ms_packets_byte_equal"] is True
                             and receipt["lookahead_samples"] == 0, "Long cache packet contract changed")
                    _require(receipt["id"] == row["id"] and receipt["frames"] == row["frames"]
                             and receipt["duration"] == row["duration"], "Long cache receipt support changed")
                    identity = receipt["identity"]
                    _equal(identity["feature"], expected_configs[name]["feature"], "Long cache feature specification")
                    _require(identity["frozen_targets_sha256"] == source["targets_sha256"]
                             and identity["audio_sha256"] == entry["audio_sha256"]
                             and identity["annotation_sha256"] == entry["annotation_sha256"],
                             "Long cache does not bind the matched source")
                    _require(identity["frontend_sha256"] == sha256(workspace / "streaming/frontend.py")
                             and identity["spectral_data_sha256"] == sha256(workspace / "spectral11/data.py")
                             and identity["load_audio_sha256"] == sha256(workspace / "cnn_gru/data.py"),
                             "Long feature cache producer changed")
                datasets[name][split].append({"entry": entry, "info": row, "features": values})
        validate_normalization(cache["normalization"][name], datasets[name]["train"])
    return datasets, expected_configs


def _optimizer_receipt(checkpoint, receipt, parameter_tensors):
    """Check saved optimizer tensor counters rather than relying on loop counts."""
    import torch

    states = checkpoint["optimizer"]["state"]
    _require(len(states) == parameter_tensors, "Optimizer did not update every model parameter tensor")
    steps = []
    for state in states.values():
        value = state.get("step")
        _require(isinstance(value, torch.Tensor) and value.numel() == 1 and bool(torch.isfinite(value)),
                 "Invalid saved optimizer step counter")
        step = float(value.item())
        _require(step.is_integer(), "Saved optimizer step is not an integer")
        steps.append(int(step))
    counters = {"minimum": min(steps), "maximum": max(steps), "parameter_states": len(steps)}
    _equal(receipt["actual_optimizer_step_counters"], counters, "Actual optimizer step counters")
    _require(counters["minimum"] == counters["maximum"] == checkpoint["step"] == 250,
             "Final model does not have exactly 250 actual parameter updates")


def audit_artifacts(output, recompute=True):
    """Audit fixed-budget models and all twenty-four validation result ledgers.

    Optional recomputation decodes saved probabilities on the CPU. It does not
    run acoustic inference or decode audio, and it does not consume any updates.
    """
    import numpy as np
    import torch
    import mido

    from acoustic10.decoder import SourceTimeDecoder
    from baseline.download import ROOT, sha256
    from baseline.evaluation import evaluate, reference_notes
    from cnn_gru.inference import tablature_metrics
    from streaming.evaluate import aggregate
    from streaming.latency import measure
    from streaming.model import CausalCNNGRU

    output, workspace = Path(output).resolve(), ROOT.resolve()
    config = _read(workspace / "configs/spectral11.json")
    protocol = _read(output / "protocol.json")
    _equal(protocol["config"], config, "Prespecified stage-11 configuration")
    _source_bindings(protocol["source_sha256"], workspace)
    _require(protocol["total_update_cap"] == config["total_optimizer_update_cap"] == 500,
             "The stage update cap changed")
    _equal(protocol["planned_optimizer_updates"], {name: 250 for name in ARMS}, "Fixed planned budget")
    _require(config["test_inference_allowed"] is False and config["production_promotion"] is False
             and protocol["test_inference_allowed"] is False, "Unexpected test evaluation or product promotion")
    _require(config["common_target_delay_frames"] == 13 and config["common_leading_mask_frames"] == 26
             and config["training"]["edge_mask_frames"] == 26, "The matched supervision support changed")
    _equal(config["decoder_parameters"], {"onset_threshold": .55, "activity_threshold": .35,
                                         "onset_backdate_seconds": .06}, "Frozen decoder configuration")
    _equal(config["pitch_diagnostics"], {"threshold": .5, "low_register_midi": [40, 51]}, "Fixed auxiliary pitch diagnostics")
    manifest = _read(workspace / "data/cnn_gru/manifest.json")
    training = _read(output / "training-summary.json")
    _equal(training["source_sha256"], protocol["source_sha256"], "Training producer identities")
    _require(training["new_optimizer_updates"] == 500 and training["test_inference_calls"] == 0
             and training["validation_used_during_training"] is False, "Training scope or accounting changed")
    cache_path = output / "cache-audit.json"
    cache = _read(cache_path)
    _require(cache["config_sha256"] == _fingerprint(config), "Cache configuration changed")
    receipts = {name: _read(output / "models" / name / "training-receipt.json") for name in ARMS}
    budget = validate_budget(receipts)
    identities = matched_arms(receipts[ARMS[0]], receipts[ARMS[1]])
    _equal(training["arms"], receipts, "Training-summary receipts")
    provenance = receipts["short128"]
    datasets, base_configs = _cached_inputs(cache, provenance, manifest, workspace)
    _require(provenance["spectral_cache_receipts_sha256"] == sha256(cache_path)
             and _relative(workspace, provenance["spectral_cache_receipts_path"]) == cache_path,
             "Spectral cache provenance changed")
    schedule = _read(output / "crop-schedule.json")
    _require(schedule["sha256"] == _fingerprint(schedule["descriptors"]) == identities["crop_schedule_sha256"],
             "Saved common crop schedule changed")
    expected_schedule = _verify_schedule(schedule["descriptors"], config, provenance["inputs"]["train"])
    for key, digest in expected_schedule.items():
        _require(identities[key] == digest, "Deterministic matched exposure changed")
    support = _fingerprint([{"id": row["id"], "frames": row["frames"], "ignored_global_leading_frames": 26,
                             "ignored_crop_edge_frames": 26, "target_delay_frames": 13}
                            for row in provenance["inputs"]["train"]])
    _require(identities["common_support_sha256"] == support, "Common masked training support changed")
    initial_path = output / "initial-weights.pt"
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    _require(initial["origin"] == "fresh random initialization" and initial["sanity_warmstart"] is False,
             "The matched models were warm-started")
    _require(weight_digest(initial["model"]) == initial["weights_sha256"] == identities["initial_weights_sha256"],
             "Shared initial weights changed")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config["seed"])
        fresh = CausalCNNGRU(receipts["short128"]["model_config"])
        _require(weight_digest(fresh.state_dict()) == initial["weights_sha256"],
                 "Initial model differs from fresh seed-42 weights")
        parameter_tensors = len(list(fresh.parameters()))
        _require(sum(parameter.numel() for parameter in fresh.parameters()) == 931145,
                 "Matched model parameter count changed")
        del fresh
    checkpoints, checkpoint_hashes = {}, {}
    for name in ARMS:
        receipt = receipts[name]
        _equal(receipt["source_sha256"], protocol["source_sha256"], name + " source identities")
        _equal(receipt["config"], config, name + " configuration")
        _require(receipt["config_sha256"] == _fingerprint(config)
                 and receipt["initial_checkpoint_sha256"] == sha256(initial_path), "Training configuration/initial file changed")
        _equal(receipt["inputs"], provenance["inputs"], "Common original source inputs")
        _equal(receipt["spectral_receipts"], cache["arms"], "Common paired spectral receipts")
        _equal(receipt["normalization"], cache["normalization"][name], name + " training-only normalization")
        _equal(receipt["positive_weights"], cache["positive_weights"], name + " fixed loss weights")
        model_config = copy.deepcopy(base_configs[name])
        model_config["training"].update(config["training"])
        model_config["stage_budget"] = {"total_max_optimizer_updates": 500, "arm_updates": 250,
                                        "new_sanity_updates": 0, "purpose": "Stage11 window comparison."}
        _equal(receipt["model_config"], model_config, name + " matched architecture/training")
        path = output / "models" / name / "final.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        _equal(checkpoint["receipt"], receipt, name + " checkpoint receipt")
        _equal(checkpoint["config"], model_config, name + " final model configuration")
        _equal(checkpoint["normalization"], receipt["normalization"], name + " checkpoint normalization")
        _optimizer_receipt(checkpoint, receipt, parameter_tensors)
        _require(weight_digest(checkpoint["model"]) != initial["weights_sha256"], "Final weights did not change")
        _require(set(checkpoint["model"]) == set(initial["model"]), "Final model-state keys changed")
        for key, value in checkpoint["model"].items():
            _require(value.shape == initial["model"][key].shape and value.dtype == initial["model"][key].dtype
                     and bool(torch.isfinite(value).all()), "Final parameter tensor is invalid")
        checkpoints[name], checkpoint_hashes[name] = checkpoint, sha256(path)

    run = _read(output / "validation/run.json")
    _require(run["test_used"] is False and run["test_inference_calls"] == 0 and run["new_optimizer_updates"] == 500,
             "Validation scope violates its fixed-budget development protocol")
    _equal(run["source_sha256"], protocol["source_sha256"], "Validation source identities")
    _equal(run["config"], config, "Validation configuration")
    score = _read(workspace / "configs/stage1.json")
    score["selection_role"] = "Stage11 validation-only window-length comparison; no test inference or promotion."
    _equal(run["scoring_config"], score, "Prespecified original source-scoring settings")
    _equal(run["training_summary"], training, "Validation training accounting")
    _equal(run["checkpoint_sha256"], checkpoint_hashes, "Validation checkpoint identities")
    _require(run["protocol_sha256"] == training["protocol_sha256"] == sha256(output / "protocol.json"),
             "Immutable protocol file changed")
    prior = _read(workspace / config["sanity_reference"])
    validate_sanity(prior["initial"], prior["final"], prior["coverage"], prior["passed"])
    _equal(_read(workspace / "outputs/acoustic10/validation/run.json")["sanity"], prior,
           "Preserved stage-10 sanity evidence")
    _source_bindings(prior["source_sha256"], workspace)
    sanity = run["sanity_reference"]
    _require(sanity["sha256"] == sha256(workspace / config["sanity_reference"])
             and sanity["checkpoint_sha256"] == sha256(workspace / "outputs/acoustic10/sanity-checkpoint.pt")
             and sanity["new_optimizer_updates"] == 0 and sanity["passed"] is True,
             "The reused prior sanity evidence changed")
    _require(sanity["prior_verification_sha256"] == sha256(workspace / "outputs/acoustic10/verification.json")
             and protocol["new_sanity_updates"] == training["new_sanity_updates"] == run["new_sanity_updates"] == 0,
             "The prior sanity verification or zero-new-update accounting changed")
    _equal(protocol["sanity_reference"], sanity, "Frozen protocol prior-sanity reference")
    _equal(training["sanity_reference"], sanity, "Training prior-sanity reference")
    _equal(run["input_provenance"]["inputs"], provenance["inputs"], "Validation original inputs")
    selection = _read(output / "selection.json")
    _equal(run["selection"], selection, "Validation selection")
    _equal(selection["checkpoint_sha256"], checkpoint_hashes, "Selection checkpoints")
    _require(selection["selected_from_test"] is False and selection["production_promoted"] is False
             and selection["rule"] == config["selection_rule"], "Selection scope or rule changed")
    records = [_read(path) for path in sorted((output / "validation/results").glob("*/*-metrics.json"))]
    expected_ids = {entry["id"] for entry in manifest["validation"]}
    _require(len(records) == 24 and {(row["id"], row["mode"]) for row in records}
             == {(clip_id, name) for clip_id in expected_ids for name in ARMS}, "Expected twelve unique validation clips per arm")
    note_count = event_count = 0
    inputs = {row["id"]: row for row in provenance["inputs"]["validation"]}
    for row in records:
        name, clip_id = row["mode"], row["id"]
        cache_record = next(record for record in datasets[name]["validation"] if record["entry"]["id"] == clip_id)
        _equal(row["input_provenance"], cache_record["info"], "Per-record matched spectral source")
        stem = output / "validation/results" / name / clip_id
        exported = _read(stem.with_suffix(".json"))
        _equal(exported["notes"], row["notes"], "Exact source-note export")
        metadata = exported["metadata"]
        _require(metadata["checkpoint_sha256"] == checkpoint_hashes[name] and metadata["mode"] == name,
                 "Export checkpoint/arm identity changed")
        _equal(metadata["decoder_parameters"], config["decoder_parameters"], "Export fixed decoder parameters")
        _equal(metadata["source_delay_seconds"], .13, "Public source-clock compensation")
        _equal(_read(stem.with_name(stem.name + "-events.json"))["events"], row["events"], "Published event export")
        ledger = audit_note_ledger(row["notes"], row["events"])
        note_count += ledger["notes"]
        event_count += ledger["events"]
        midi = mido.MidiFile(stem.with_suffix(".mid"))
        attacks = Counter(message.note for track in midi.tracks for message in track
                          if message.type == "note_on" and message.velocity > 0)
        _require(attacks == Counter(note["pitch"] for note in row["notes"]), "MIDI dropped or changed published attacks")
        probability_path = _relative(output, row["probability_path"])
        _require(probability_path == output / "validation/probabilities" / name / (clip_id + ".npz"),
                 "Probability file was rebound to another clip or arm")
        _require(sha256(probability_path) == row["probability_sha256"], "Exact packet probabilities changed")
        timing = row["timing"]
        _require(timing["lookahead_samples"] == 0 and timing["packet_ms"] == config["validation_packet_ms"] == 20
                 and timing["exact_probability_recording_included"] is True, "Measured packet contract changed")
        _equal(timing["source_delay_compensation_seconds"], .13, "Measured public source compensation")
        if recompute:
            with np.load(probability_path, allow_pickle=False) as arrays:
                probability = {key: arrays[key].copy() for key in arrays.files}
            _require(set(probability) == {"state", "onset", "offset", "pitch"}, "Acoustic probability heads changed")
            length = len(probability["state"])
            duration = timing["input_seconds"]
            _require(length == math.floor(duration * 100 + 1e-8), "Packet probability frame count changed")
            _require(probability["state"].shape == (length, 6, 26) and np.isfinite(probability["state"]).all()
                     and np.all((probability["state"] >= 0) & (probability["state"] <= 1))
                     and np.allclose(probability["state"].sum(-1), 1, rtol=0, atol=2e-6),
                     "Saved full state probability distribution is invalid")
            for key, width in (("onset", 6), ("offset", 6), ("pitch", 49)):
                _require(probability[key].shape == (length, width) and np.isfinite(probability[key]).all()
                         and np.all((probability[key] >= 0) & (probability[key] <= 1)), "Invalid acoustic probability head")
            decoder = SourceTimeDecoder(checkpoints[name]["config"], config["decoder_parameters"], .13)
            for start in range(0, length, 2):
                end = min(start + 2, length)
                decoder.push({key: value[start:end] for key, value in probability.items()}, end / 100)
            decoder.finish(duration, duration)
            notes = [{key: value for key, value in note.items()
                      if key not in ("on_delivered_at_seconds", "off_delivered_at_seconds")} for note in row["notes"]]
            events = [{key: value for key, value in event.items() if key != "delivered_at_seconds"} for event in row["events"]]
            _equal(notes, decoder.notes, "Recomputed immutable source notes")
            _equal(events, decoder.events, "Recomputed packet publication events")
            _equal(timing["source_clamps"], decoder.clamp_counts, "Source-clock clamp counts")
            entry = next(entry for entry in manifest["validation"] if entry["id"] == clip_id)
            reference = reference_notes(_relative(workspace, entry["annotation"]))
            _equal(row["reference"], reference, "Original source reference notes")
            prefix = _relative(workspace, inputs[clip_id]["prefix"])
            with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as arrays:
                target = {key: arrays[key].copy() for key in arrays.files}
            heads = aligned_heads(probability, target)
            _equal({key: row["head_metrics"][key] for key in heads}, heads, "Independent aligned exact-state metrics")
            _equal(row["head_metrics"]["pitch"], aligned_pitch(probability, target), "Independent fixed-threshold auxiliary pitch diagnostics")
            _equal(row["pitch_metrics"], evaluate(reference, row["notes"], run["scoring_config"], duration), "Recomputed original-source note/pitch metrics")
            _equal(row["tab_metrics"], tablature_metrics(reference, row["notes"], run["scoring_config"],
                                                       run["scoring_config"]["score_seconds"], duration), "Recomputed string metrics")
            _equal(row["latency"], measure(reference, row["notes"], run["scoring_config"], duration), "Recomputed conditional matched-event latency")
    summary = aggregate(records)
    for name in ARMS:
        rows = [row for row in records if row["mode"] == name]
        summary["modes"][name]["exact_state_f1"] = float(np.mean([row["head_metrics"]["exact_state_f1"] for row in rows]))
        summary["modes"][name]["active_fret_accuracy"] = float(np.mean([row["head_metrics"]["active_fret_accuracy"] for row in rows]))
        summary["modes"][name]["auxiliary_pitch_f1"] = float(np.mean([row["head_metrics"]["pitch"]["overall"]["f1"] for row in rows]))
        summary["modes"][name]["low_register_pitch_f1"] = float(np.mean([row["head_metrics"]["pitch"]["low_register"]["f1"] for row in rows]))
    _equal(_read(output / "validation/summary.json"), summary, "Independent validation aggregation")
    scores = {name: {key: summary["modes"][name][key] for key in ("exact_state_f1", "onset_f1")} for name in ARMS}
    _equal(selection["scores"], scores, "Independent selection scores")
    _require(selection["selected_arm"] == choose_arm(scores), "Selected arm contradicts prespecified validation rule")
    return {"passed": True, **budget, "selected_arm": selection["selected_arm"], "validation_records": len(records),
            "notes": note_count, "events": event_count, "checkpoint_sha256": checkpoint_hashes,
            "protocol_sha256": sha256(output / "protocol.json"), "matched_initial_weights_sha256": identities["initial_weights_sha256"],
            "recomputed_saved_probabilities": bool(recompute), "test_inference_calls": 0,
            "new_sanity_updates": 0, "scope": "Read-only audit with no acoustic inference, training or product promotion."}


def audit(output, recompute=True):
    """Short public entry point for the experiment CLI."""
    return audit_artifacts(output, recompute=recompute)
