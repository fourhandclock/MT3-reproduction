"""Explicit ground-truth head experiments for the frozen streaming decoder.

These are decoder diagnostics, not an acoustic inference path. Reference notes
are converted to the same right-edge labels used during stage 8; Gaussian
boundaries therefore have privileged annotation information. They must never be
presented as causal model predictions or used to estimate deployment accuracy.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np

from baseline.download import ROOT, sha256
from baseline.evaluation import STANDARD_TUNING, evaluate, match_notes, reference_notes
from cnn_gru.inference import tablature_metrics
from streaming.decoder import OnlineDecoder
from streaming.targets import make_targets


def oracle_probabilities(reference, frame_count, config, duration):
    """Return ideal label heads plus masked-conflict counts, without any audio.

    State probabilities are exactly one-hot. Ambiguous same-string/different-
    fret state labels become inactive, while pitch/onset/offset labels are kept
    as written by ``make_targets``. This deliberate masking is disclosed.
    """
    for note in reference:
        string, fret, pitch = note["string"], note["fret"], note["pitch"]
        if any(not isinstance(value, (int, np.integer)) for value in (string, fret, pitch)):
            raise ValueError("Oracle string, fret and MIDI pitch must be integer labels.")
        if not 0 <= string < len(STANDARD_TUNING):
            raise ValueError("Oracle string is outside standard tuning.")
        if STANDARD_TUNING[string] + fret != pitch:
            raise ValueError("Oracle pitch must equal its string tuning plus fret.")
    labels, conflicts = make_targets(reference, frame_count, config, duration)
    classes = int(config["model"]["max_fret"]) + 2
    states = np.asarray(labels["state"])
    indices = np.where(states == -100, classes - 1, states)
    probabilities = {
        "state": np.eye(classes, dtype=np.float32)[indices],
        "onset": labels["onset"].astype(np.float32),
        "offset": labels["offset"].astype(np.float32),
        "pitch": labels["pitch"].astype(np.float32),
    }
    return probabilities, {"conflict_string_frames": conflicts,
                           "conflict_frame_count": int(np.any(states == -100, axis=1).sum()),
                           "conflict_policy": "Reference -100 string states become one-hot inactive; other heads remain reference labels."}


def replace_heads(predicted, oracle, oracle_heads):
    """Build an explicitly named head intervention without mutating either input."""
    expected = {"state", "onset", "offset", "pitch"}
    if set(predicted) != expected or set(oracle) != expected:
        raise ValueError("Both experiments must contain exactly four probability heads.")
    heads = set(oracle_heads)
    if not heads <= expected:
        raise ValueError("Unknown oracle head.")
    if any(np.asarray(predicted[key]).shape != np.asarray(oracle[key]).shape for key in expected):
        raise ValueError("Predicted and oracle heads must share the right-edge frame grid and shapes.")
    return {key: np.array(oracle[key] if key in heads else predicted[key], copy=True)
            for key in ("state", "onset", "offset", "pitch")}


def decode_packets(probabilities, config, parameters, duration, packet_ms=20):
    """Replay complete frames with packet-rounded audio availability; no compute clock.

    Source notes are packet-size invariant. Delivery represents ideal zero-
    compute file arrival, so it is a structural latency lower bound only.
    Incomplete-hop EOF audio never becomes a future probability frame.
    """
    period = float(config["feature"]["hop_length"]) / config["feature"]["sample_rate"]
    packet_seconds = float(packet_ms) / 1000
    frames = packet_seconds / period
    if not math.isfinite(frames) or frames < 1 or abs(frames - round(frames)) > 1e-9:
        raise ValueError("Packet size must be a positive integer number of feature hops.")
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("Observed audio duration must be finite and nonnegative.")
    count = len(probabilities["state"])
    if count * period > duration + 1e-9:
        raise ValueError("Probability frames extend beyond the observed audio.")
    stream = OnlineDecoder(config, parameters)
    frames = int(round(frames))
    for start in range(0, count, frames):
        end = min(count, start + frames)
        availability = min(duration, (start + frames) * period)
        stream.push({key: value[start:end] for key, value in probabilities.items()},
                    available_at_audio_seconds=availability)
    stream.finish(duration, available_at_audio_seconds=duration)
    return stream.notes, stream.events


def _quantiles(values):
    if not values:
        return {"count": 0, "p50": None, "p95": None}
    return {"count": len(values), "p50": float(np.quantile(values, .5)),
            "p95": float(np.quantile(values, .95))}


def structural_counts(reference, predicted, config, score_config, duration):
    """Describe short/repeated missed notes using unchanged reference intervals."""
    limit = min(duration, float(score_config["score_seconds"]))
    ref = [note for note in reference if 0 <= note["start"] < limit]
    est = [note for note in predicted if 0 <= note["start"] < limit]
    matched = set()
    paired = []
    for string in range(6):
        ri = [index for index, note in enumerate(ref) if note["string"] == string]
        ei = [index for index, note in enumerate(est) if note["string"] == string]
        for a, b in match_notes([ref[index] for index in ri], [est[index] for index in ei], score_config):
            matched.add(ri[a])
            paired.append((ri[a], ei[b]))
    fps = config["feature"]["sample_rate"] / config["feature"]["hop_length"]
    count = int(math.floor(duration * fps + 1e-9))
    # Match make_targets' right-edge, half-open grid exactly. A note whose
    # original release is beyond EOF occupies the last observed complete frame;
    # clipping its release to duration here would incorrectly drop that frame.
    def grid_index(seconds):
        return min(count, max(0, int(math.ceil(seconds * fps - 1.0 - 1e-9))))

    observable_frames = [max(0, grid_index(note["end"]) - grid_index(note["start"])) for note in ref]
    confirm = int(config["streaming"]["confirm_frames"])
    short = [index for index, frames in enumerate(observable_frames) if frames < confirm]
    repeated = []
    for string in range(6):
        ordered = sorted((index for index, note in enumerate(ref) if note["string"] == string),
                         key=lambda index: (ref[index]["start"], ref[index]["end"]))
        for previous, current in zip(ordered, ordered[1:]):
            if (ref[current]["pitch"] == ref[previous]["pitch"] and
                    ref[current]["start"] - ref[previous]["end"] <= config["decoder"]["minimum_retrigger_seconds"] + 1e-9):
                repeated.append(current)
    latency = [1000 * (est[j]["on_emitted_at_audio_seconds"] - ref[i]["start"]) for i, j in paired]
    return {"reference_note_count": len(ref), "same_string_matched_note_count": len(matched),
            "short_note_reference_count": len(short),
            "short_note_missed_count": sum(index not in matched for index in short),
            "short_note_reference_indices": short,
            "short_note_definition": f"Fewer than {confirm} complete right-edge occupied frames for the original note; potential confirmation loss, not proof of sole cause.",
            "repeated_note_reference_count": len(repeated),
            "repeated_note_missed_count": sum(index not in matched for index in repeated),
            "repeated_note_reference_indices": repeated,
            "repeated_note_definition": "Consecutive same-string, same-pitch reference notes separated by at most the retrigger guard (overlap included).",
            "initial_subhop_reference_count": sum(0 <= note["start"] < 1 / fps for note in ref),
            "unobserved_terminal_release_count": sum(note["end"] >= duration for note in ref),
            "eof_censored_prediction_count": sum(note.get("censored_at_end", False) for note in est),
            "same_string_onset_arrival_ms": _quantiles(latency),
            "onset_arrival_samples_ms": latency,
            "arrival_definition": "20-ms packet-rounded audio availability minus original onset, same-string one-to-one onset matches only; no compute or microphone delay."}


def diagnose_case(reference, probabilities, config, parameters, score_config, duration, packet_ms=20):
    notes, events = decode_packets(probabilities, config, parameters, duration, packet_ms)
    result = {"decoder_parameters": copy.deepcopy(parameters), "packet_ms": packet_ms,
              "notes": notes, "events": events,
              "pitch_metrics": evaluate(reference, notes, score_config, duration),
              "tab_metrics": tablature_metrics(reference, notes, score_config,
                                                score_config["score_seconds"], duration),
              "structural": structural_counts(reference, notes, config, score_config, duration)}
    result["structural"]["arrival_definition"] = result["structural"]["arrival_definition"].replace("20-ms", f"{packet_ms}-ms")
    return result


def aggregate(records):
    result = {}
    for variant in sorted({row["variant"] for row in records}):
        rows = [row for row in records if row["variant"] == variant]
        arrival = [v for row in rows for v in row["structural"]["onset_arrival_samples_ms"]]
        result[variant] = {"count": len(rows),
                           "note_onset_f1": float(np.mean([row["pitch_metrics"]["onset"]["f1"] for row in rows])),
                           "note_onset_offset_f1": float(np.mean([row["pitch_metrics"]["onset_offset"]["f1"] for row in rows])),
                           "string_onset_f1": float(np.mean([row["tab_metrics"]["onset"]["f1"] for row in rows])),
                           "string_onset_offset_f1": float(np.mean([row["tab_metrics"]["onset_offset"]["f1"] for row in rows])),
                           "frame_f1": float(np.mean([row["pitch_metrics"]["frame"]["f1"] for row in rows])),
                           "onset_arrival_ms": _quantiles(arrival)}
        for key in ("reference_note_count", "short_note_reference_count", "short_note_missed_count",
                    "repeated_note_reference_count", "repeated_note_missed_count",
                    "initial_subhop_reference_count", "unobserved_terminal_release_count",
                    "eof_censored_prediction_count"):
            result[variant][key] = sum(row["structural"][key] for row in rows)
    return result


def run_validation(output=ROOT / "outputs/diagnosis9/oracle-diagnostics.json"):
    """Run only progression-2 validation; this function never reads test labels."""
    import soundfile as sf

    previous = json.loads((ROOT / "outputs/streaming/evaluation/run.json").read_text(encoding="utf-8"))
    config, parameters, score_config = previous["config"], previous["decoder_parameters"], previous["scoring_config"]
    source = {"decoder.py": sha256(ROOT / "streaming/decoder.py"),
              "targets.py": sha256(ROOT / "streaming/targets.py")}
    for filename, digest in source.items():
        if previous["source_sha256"][filename] != digest:
            raise ValueError("Frozen streaming source changed: " + filename)
    manifest_path = ROOT / "data/cnn_gru/manifest.json"
    if sha256(manifest_path) != previous["manifest_sha256"]:
        raise ValueError("Frozen split manifest changed.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    variants = {"oracle_selected_backdate": dict(parameters),
                "oracle_zero_backdate": dict(parameters, onset_backdate_seconds=0.0)}
    records = []
    for entry in manifest["validation"]:
        for field in ("audio", "annotation"):
            if sha256(ROOT / entry[field]) != entry[field + "_sha256"]:
                raise ValueError("Original validation source changed: " + entry["id"])
        info = sf.info(ROOT / entry["audio"])
        duration = min(float(config["evaluation_audio_seconds"]), info.frames / info.samplerate)
        fps = config["feature"]["sample_rate"] / config["feature"]["hop_length"]
        frame_count = int(math.floor(duration * fps + 1e-9))
        reference = reference_notes(ROOT / entry["annotation"])
        probabilities, conflicts = oracle_probabilities(reference, frame_count, config, duration)
        for variant, decoder_parameters in variants.items():
            row = diagnose_case(reference, probabilities, config, decoder_parameters, score_config, duration)
            row.update({"id": entry["id"], "variant": variant, "input_seconds": duration,
                        "frame_count": frame_count, "conflicts": conflicts,
                        "full_original_reference": copy.deepcopy(reference),
                        "source_audio_sha256": entry["audio_sha256"],
                        "source_annotation_sha256": entry["annotation_sha256"]})
            records.append(row)
    result = {"experiment": "Reference labels to frozen causal decoder: explicit privileged oracle diagnostic",
              "split": "validation", "test_used_for_selection": False, "new_optimizer_updates": 0,
              "summary": aggregate(records), "records": records,
              "protocol": {"input_seconds": config["evaluation_audio_seconds"],
                           "score_seconds": score_config["score_seconds"],
                           "packet_ms": 20, "config": config, "scoring_config": score_config,
                           "source_sha256": source, "manifest_sha256": sha256(manifest_path),
                           "oracles_source_sha256": sha256(Path(__file__)),
                           "data_path": "JAMS -> explicit oracle targets -> decoder; audio is metadata/hash only; no feature extraction, model, fitting or normalization.",
                           "limitations": ["Gaussian reference boundaries contain privileged information and are not deployable causal predictions.",
                                           "Offset and auxiliary pitch heads are accepted but ignored by the frozen OnlineDecoder.",
                                           "Masked same-string state conflicts can create inactive gaps; their counts are reported.",
                                           "Zero compute packet replay is a structural latency lower bound, not measured model or microphone latency.",
                                           "Backdate zero avoids applying acoustic lag correction to exact reference occupancy; the selected 60-ms value is a separate stress check."]}}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def validate_cached_target_identity(reference, cached_targets, frame_count, config, duration):
    """Verify cached label arrays exactly reproduce the unmodified source JAMS."""
    expected, conflicts = make_targets(reference, frame_count, config, duration)
    if set(cached_targets) != set(expected):
        raise ValueError("Cached targets do not contain the frozen four label heads.")
    for head in expected:
        if not np.array_equal(np.asarray(cached_targets[head]), expected[head]):
            raise ValueError("Cached target identity differs from original annotation: " + head)
    return conflicts


def hybrid_variants(predicted, oracle, parameters):
    """Cross all state/onset provenance combinations with both source backdates."""
    zero = dict(parameters, onset_backdate_seconds=0.0)
    selected = dict(parameters)
    state_only = replace_heads(predicted, oracle, ["state"])
    onset_only = replace_heads(predicted, oracle, ["onset"])
    both = replace_heads(predicted, oracle, ["state", "onset"])
    return {
        "predicted_state_predicted_onset": (predicted, selected, []),
        "predicted_state_predicted_onset_zero_backdate": (predicted, zero, []),
        "oracle_state_predicted_onset": (state_only, zero, ["state"]),
        "oracle_state_predicted_onset_selected_backdate": (state_only, selected, ["state"]),
        "predicted_state_oracle_onset": (onset_only, selected, ["onset"]),
        "predicted_state_oracle_onset_zero_backdate": (onset_only, zero, ["onset"]),
        "oracle_state_oracle_onset": (both, zero, ["state", "onset"]),
        "oracle_state_oracle_onset_selected_backdate": (both, selected, ["state", "onset"]),
    }


def run_hybrid_validation(output=ROOT / "outputs/diagnosis9/hybrid-head-diagnostics.json"):
    """Intervene on cached validation heads; never load or run a model.

    Head and input hashes bind each cache to the frozen checkpoint and source
    split. All four state/onset provenance combinations are decoded at both
    zero and selected source backdate. Fixed-backdate comparisons isolate head
    interventions; the reference-state/zero-backdate and acoustic-state/selected-
    backdate rows additionally expose the appropriate timing conventions.
    """
    from streaming.data import cache_path

    prior_path = ROOT / "outputs/streaming/evaluation/run.json"
    previous = json.loads(prior_path.read_text(encoding="utf-8"))
    audit_path = ROOT / "outputs/diagnosis9/validation-head-diagnostics.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    manifest_path = ROOT / "data/cnn_gru/manifest.json"
    manifest_digest = sha256(manifest_path)
    if manifest_digest != previous["manifest_sha256"] or manifest_digest != audit["manifest_sha256"]:
        raise ValueError("Frozen validation split changed.")
    checkpoint_path = ROOT / audit["checkpoint_path"]
    checkpoint_digest = sha256(checkpoint_path)
    if checkpoint_digest != previous["checkpoint_sha256"] or checkpoint_digest != audit["checkpoint_sha256"]:
        raise ValueError("Cached validation heads do not belong to the selected frozen checkpoint.")
    if audit["decoder_parameters"] != previous["decoder_parameters"]:
        raise ValueError("Head-audit decoder parameters differ from the frozen selected decoder.")
    if sha256(ROOT / "diagnosis9/head_audit.py") != audit["head_audit_sha256"]:
        raise ValueError("Cached head-audit source changed after probability provenance was recorded.")
    frozen_sources = ("frontend.py", "model.py", "decoder.py", "targets.py",
                      "data.py", "inference.py", "training.py")
    for filename in frozen_sources:
        digest = previous["source_sha256"][filename]
        if sha256(ROOT / "streaming" / filename) != digest:
            raise ValueError("Frozen streaming source changed: " + filename)
    config, parameters = previous["config"], previous["decoder_parameters"]
    score_config = previous["scoring_config"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit_rows = {row["id"]: row for row in audit["clips"]}
    expected_ids = {entry["id"] for entry in manifest["validation"]}
    if set(audit_rows) != expected_ids or len(audit["clips"]) != len(expected_ids):
        raise ValueError("Head cache does not contain exactly the frozen validation identities.")
    records, verified_inputs = [], []
    for entry in manifest["validation"]:
        for field in ("audio", "annotation"):
            if sha256(ROOT / entry[field]) != entry[field + "_sha256"]:
                raise ValueError("Original validation source changed: " + entry["id"])
        audited = audit_rows[entry["id"]]
        prefix = cache_path(entry, config, config["evaluation_audio_seconds"])
        info = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
        if info["id"] != entry["id"] or info["duration"] != audited["duration"]:
            raise ValueError("Cached input identity or observed duration changed.")
        for field, suffix in (("features", ".npy"), ("targets", ".npz")):
            digest = sha256(prefix.with_suffix(suffix))
            if digest != info[field + "_sha256"] or digest != audited[field + "_sha256"]:
                raise ValueError("Frozen validation " + field + " cache changed.")
        probability_path = ROOT / audited["probability_path"]
        if sha256(probability_path) != audited["probability_sha256"]:
            raise ValueError("Frozen validation probability cache changed.")
        with np.load(probability_path, allow_pickle=False) as arrays:
            predicted = {key: arrays[key].copy() for key in arrays.files}
        if len(predicted["state"]) != info["frames"]:
            raise ValueError("Cached probability frame count differs from its audio feature input.")
        reference = reference_notes(ROOT / entry["annotation"])
        with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as arrays:
            targets = {key: arrays[key].copy() for key in arrays.files}
        conflict_count = validate_cached_target_identity(reference, targets, info["frames"], config, info["duration"])
        if conflict_count != info["conflict_string_frames"]:
            raise ValueError("Cached conflict-mask count changed.")
        oracle, conflicts = oracle_probabilities(reference, info["frames"], config, info["duration"])
        variants = hybrid_variants(predicted, oracle, parameters)
        verified = {"id": entry["id"], "audio_sha256": entry["audio_sha256"],
                    "annotation_sha256": entry["annotation_sha256"],
                    "features_sha256": info["features_sha256"], "targets_sha256": info["targets_sha256"],
                    "probability_sha256": audited["probability_sha256"],
                    "probability_path": audited["probability_path"],
                    "target_identity": "Exact array equality with targets regenerated from original verified annotation."}
        verified_inputs.append(verified)
        for variant, (probabilities, decoder_parameters, replaced) in variants.items():
            row = diagnose_case(reference, probabilities, config, decoder_parameters, score_config, info["duration"])
            if variant == "predicted_state_predicted_onset":
                recorded = audited["note_modes"]["strict_saved"][str(score_config["onset_tolerance_seconds"])]
                for metric in ("onset", "onset_offset", "frame"):
                    for field in ("f1", "true_positives", "reference_count", "predicted_count"):
                        if not np.isclose(row["pitch_metrics"][metric][field], recorded[metric][field], rtol=0, atol=1e-12):
                            raise ValueError("Frozen strict decoder no longer reproduces cached head-audit scoring.")
            row.update({"id": entry["id"], "variant": variant, "oracle_heads": replaced,
                        "fixed_backdate_group_ms": 1000 * decoder_parameters["onset_backdate_seconds"],
                        "input_seconds": info["duration"], "frame_count": info["frames"],
                        "conflicts": conflicts, "full_original_reference": copy.deepcopy(reference)})
            records.append(row)
    result = {"experiment": "Frozen cached prediction heads with explicit privileged reference-head interventions",
              "split": "validation", "test_used": False, "new_optimizer_updates": 0,
              "new_model_inference_calls": 0, "checkpoint_sha256": checkpoint_digest,
              "manifest_sha256": manifest_digest, "summary": aggregate(records), "records": records,
              "verified_inputs": verified_inputs,
              "protocol": {"input_seconds": config["evaluation_audio_seconds"],
                           "score_seconds": score_config["score_seconds"], "packet_ms": 20,
                           "confirmation_frames": config["streaming"]["confirm_frames"],
                           "release_confirmation_frames": config["streaming"]["release_confirm_frames"],
                           "frozen_parameters": parameters,
                           "oracle_state_backdate_seconds": 0.0,
                           "predicted_state_backdate_seconds": parameters["onset_backdate_seconds"],
                           "factorial_backdates_seconds": [0.0, parameters["onset_backdate_seconds"]],
                           "factorial_rule": "All four state/onset provenance pairs run with both backdates; comparisons within a fixed-backdate group change only the named head(s). Offset/pitch probabilities stay frozen acoustic predictions throughout and are ignored by the decoder.",
                           "head_audit_sha256": sha256(audit_path),
                           "frozen_streaming_source_sha256": {filename: sha256(ROOT / "streaming" / filename) for filename in frozen_sources},
                           "common_scoring_source_sha256": {"baseline/evaluation.py": sha256(ROOT / "baseline/evaluation.py"),
                                                            "cnn_gru/inference.py": sha256(ROOT / "cnn_gru/inference.py")},
                           "strict_baseline_reproduction": "All 12 strict rows reproduce recorded onset/offset/frame counts and F1 at 1e-12 tolerance before interpreting interventions.",
                           "oracles_source_sha256": sha256(Path(__file__)),
                           "data_path": "Verified frozen NPZ prediction heads; verified JAMS used only for explicit oracle replacements and scoring. No feature extraction or model load.",
                           "limitations": ["Oracle heads contain annotation information; these interventions are not deployable predictions.",
                                           "Appropriate-lag cross-group rows change both head provenance and backdate; use fixed-backdate rows to isolate head effects.",
                                           "Offset and pitch heads are ignored by the frozen decoder.",
                                           "Arrival is zero compute file replay, not measured microphone/model latency."]}}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid", action="store_true", help="Use verified cached validation prediction heads.")
    arguments = parser.parse_args()
    result = run_hybrid_validation() if arguments.hybrid else run_validation()
    print(json.dumps(result["summary"], indent=2, allow_nan=False))
