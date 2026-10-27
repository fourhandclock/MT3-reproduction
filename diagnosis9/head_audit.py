"""Validation-only audit of frozen stage-8 acoustics and strict decoding.

No training, source changes, test clips, or decoder parameter selection occurs.
"""

import json
from pathlib import Path

import numpy as np
import torch

from baseline.download import ROOT, sha256
from baseline.evaluation import evaluate, reference_notes
from streaming.data import cache_path
from streaming.evaluate import load_checkpoint, save_json
from streaming.inference import decode_probabilities, predict_features


def metric(tp, expected, estimated):
    precision = tp / estimated if estimated else 0.0
    recall = tp / expected if expected else 0.0
    return {"precision": float(precision), "recall": float(recall),
            "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
            "true_positives": int(tp), "reference_count": int(expected), "predicted_count": int(estimated)}


def exact_state_metric(target, estimated, inactive):
    valid = target != -100
    expected = valid & (target != inactive)
    observed = valid & (estimated != inactive)
    return metric(int((expected & observed & (target == estimated)).sum()),
                  int(expected.sum()), int(observed.sum()))


def binary_metric(target, estimated):
    return metric(int((target & estimated).sum()), int(target.sum()), int(estimated.sum()))


def describe(values):
    values = np.asarray(values).reshape(-1)
    if not len(values):
        return {"count": 0, "mean": None, "p10": None, "p50": None, "p90": None}
    return {"count": int(len(values)), "mean": float(values.mean()),
            **{f"p{p}": float(np.percentile(values, p)) for p in (10, 50, 90)}}


def runs(states, inactive):
    lengths = []
    if not len(states):
        return lengths
    for string in range(states.shape[1]):
        values = states[:, string]
        edges = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
        lengths.extend(int(end - start) for start, end in zip(edges[:-1], edges[1:])
                       if values[start] >= 0 and values[start] != inactive)
    return lengths


def run_summary(lengths):
    value = describe(lengths)
    value["fraction_under_4_frames"] = float(np.mean(np.asarray(lengths) < 4)) if lengths else None
    value["fraction_under_10_frames"] = float(np.mean(np.asarray(lengths) < 10)) if lengths else None
    return value


def raw_run_notes(states, inactive, period, duration, backdate=0.0):
    """Each direct active-class run, used diagnostically rather than as a decoder."""
    notes = []
    if not len(states):
        return notes
    for string, tuning in enumerate((40, 45, 50, 55, 59, 64)):
        values = states[:, string]
        edges = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
        for start, end in zip(edges[:-1], edges[1:]):
            fret = int(values[start])
            if fret < 0 or fret == inactive:
                continue
            first = max(0.0, (start + 1) * period - backdate)
            last = min(duration, (end + 1) * period - backdate)
            if last > first:
                notes.append({"start": first, "end": last, "pitch": tuning + fret,
                              "string": string, "fret": fret})
    return sorted(notes, key=lambda note: (note["start"], note["pitch"]))


def pooled(rows):
    return metric(sum(row["true_positives"] for row in rows),
                  sum(row["reference_count"] for row in rows),
                  sum(row["predicted_count"] for row in rows))


def main():
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = ROOT / "outputs/diagnosis9"
    checkpoint_path, checkpoint, model = load_checkpoint(ROOT / "outputs/streaming", device)
    config = checkpoint["config"]
    manifest_path = ROOT / "data/cnn_gru/manifest.json"
    entries = json.loads(manifest_path.read_text())["validation"]
    score = checkpoint["scoring_config"]
    inactive = config["model"]["max_fret"] + 1
    period = config["feature"]["hop_length"] / config["feature"]["sample_rate"]
    records, distributions, all_runs = [], [], {"target": [], "argmax": [], "activity_035": []}
    cache_directory = output / "validation-head-cache"
    cache_directory.mkdir(parents=True, exist_ok=True)
    prior_path = output / "validation-head-diagnostics.json"
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else {}
    prior_rows = {row["id"]: row for row in prior.get("clips", [])}
    reusable_checkpoint = prior.get("checkpoint_sha256") == sha256(checkpoint_path)
    for entry in entries:
        prefix = cache_path(entry, config, config["evaluation_audio_seconds"])
        info = json.loads(prefix.with_suffix(".json").read_text())
        for kind, extension in (("features", ".npy"), ("targets", ".npz")):
            if sha256(prefix.with_suffix(extension)) != info[kind + "_sha256"]:
                raise ValueError("Frozen validation cache changed")
        cache_file = cache_directory / (entry["id"] + ".npz")
        prior_row = prior_rows.get(entry["id"], {})
        reusable = (reusable_checkpoint and cache_file.exists()
                    and prior_row.get("features_sha256") == info["features_sha256"]
                    and prior_row.get("targets_sha256") == info["targets_sha256"]
                    and prior_row.get("probability_sha256") == sha256(cache_file))
        if reusable:
            with np.load(cache_file) as arrays:
                probability = {key: arrays[key].copy() for key in arrays.files}
        else:
            probability = predict_features(model, np.load(prefix.with_suffix(".npy"), mmap_mode="r"),
                                           checkpoint["normalization"], config, device)
            np.savez_compressed(cache_file, **probability)
        count = min(len(probability["state"]), int(score["score_seconds"] / period))
        with np.load(prefix.with_suffix(".npz")) as arrays:
            target = arrays["state"][:count].copy()
            pitch_target = arrays["pitch"][:count].astype(bool)
            onset_target = arrays["onset"][:count].astype(np.float32)
        state = probability["state"][:count]
        frets = state[:, :, :-1].argmax(axis=-1)
        best = state[:, :, :-1].max(axis=-1)
        odds = best / (best + state[:, :, -1] + 1e-8)
        argmax = state.argmax(axis=-1)
        activity_state = np.where(odds >= .35, frets, inactive)
        valid = target != -100
        active = valid & (target != inactive)
        correct = active & (frets == target)
        other = active & (frets != target)
        sorted_active = np.sort(state[:, :, :-1], axis=-1)
        margin = sorted_active[:, :, -1] - sorted_active[:, :, -2]
        row = {"id": entry["id"], "frames": count, "duration": info["duration"],
               "state_argmax": exact_state_metric(target, argmax, inactive),
               "state_activity_035": exact_state_metric(target, activity_state, inactive),
               "state_total_accuracy": float(np.mean(argmax[valid] == target[valid])),
               "activity_argmax": binary_metric(active, valid & (argmax != inactive)),
               "activity_035": binary_metric(active, valid & (activity_state != inactive)),
               "pitch": {str(threshold): binary_metric(pitch_target, probability["pitch"][:count] >= threshold)
                         for threshold in (.25, .5, .75)}, "note_modes": {},
               "onset_probability_true_boundaries": describe(probability["onset"][:count][onset_target >= .5]),
               "onset_probability_elsewhere": describe(probability["onset"][:count][onset_target < .5]),
               "probability_path": str(cache_file.relative_to(ROOT)).replace("\\", "/"),
               "probability_sha256": sha256(cache_file),
               "features_sha256": info["features_sha256"], "targets_sha256": info["targets_sha256"]}
        reference = reference_notes(ROOT / entry["annotation"])
        strict, _ = decode_probabilities(probability, config, checkpoint["decoder_parameters"], info["duration"])
        full_state = probability["state"]
        full_best = full_state[:, :, :-1].max(axis=-1)
        full_odds = full_best / (full_best + full_state[:, :, -1] + 1e-8)
        full_activity = np.where(full_odds >= .35, full_state[:, :, :-1].argmax(axis=-1), inactive)
        raw_notes = raw_run_notes(full_activity, inactive, period, info["duration"], .06)
        for name, notes in (("strict_saved", strict), ("direct_activity_runs_backdate_006", raw_notes)):
            row["note_modes"][name] = {}
            for tolerance in (.05, .1, .2):
                evaluation_config = dict(score, onset_tolerance_seconds=tolerance)
                result = evaluate(reference, notes, evaluation_config, info["duration"])
                row["note_modes"][name][str(tolerance)] = {"onset": result["onset"], "onset_offset": result["onset_offset"],
                                                         "frame": result["frame"], "errors": result["errors"]}
        for name, values in (("target", target), ("argmax", argmax), ("activity_035", activity_state)):
            lengths = runs(values, inactive)
            all_runs[name].extend(lengths)
            row.setdefault("run_lengths", {})[name] = run_summary(lengths)
        row["per_string"] = []
        for string in range(6):
            item = {"string": string,
                    "argmax": exact_state_metric(target[:, string], argmax[:, string], inactive),
                    "activity_035": exact_state_metric(target[:, string], activity_state[:, string], inactive)}
            for group, mask in (("correct_fret", correct[:, string]), ("wrong_fret", other[:, string]),
                                ("reference_inactive", valid[:, string] & ~active[:, string])):
                item[group] = {"activity_odds": describe(odds[:, string][mask]),
                               "best_fret_probability": describe(best[:, string][mask]),
                               "best_second_margin": describe(margin[:, string][mask])}
            row["per_string"].append(item)
        shifts = {}
        for shift in range(-20, 21, 2):
            if shift > 0:
                a, b = target[:-shift], activity_state[shift:]
            elif shift < 0:
                a, b = target[-shift:], activity_state[:shift]
            else:
                a, b = target, activity_state
            shifts[str(shift * period * 1000)] = exact_state_metric(a, b, inactive)
        row["diagnostic_frame_lag_ms"] = shifts
        distributions.append({"correct_odds": odds[correct], "wrong_odds": odds[other],
                              "correct_margin": margin[correct], "wrong_margin": margin[other]})
        records.append(row)
        print(entry["id"], "raw state F1", round(row["state_activity_035"]["f1"], 3),
              "strict note F1", round(row["note_modes"]["strict_saved"]["0.05"]["onset"]["f1"], 3), flush=True)
    summary = {key: {"pooled": pooled([row[key] for row in records]),
                     "mean_clip_f1": float(np.mean([row[key]["f1"] for row in records]))}
               for key in ("state_argmax", "state_activity_035", "activity_argmax", "activity_035")}
    summary["pitch"] = {key: {"pooled": pooled([row["pitch"][key] for row in records]),
                                    "mean_clip_f1": float(np.mean([row["pitch"][key]["f1"] for row in records]))}
                        for key in ("0.25", "0.5", "0.75")}
    summary["runs"] = {key: run_summary(value) for key, value in all_runs.items()}
    summary["distributions"] = {key: describe(np.concatenate([row[key] for row in distributions]))
                                for key in distributions[0]}
    summary["note_modes"] = {name: {tolerance: {"onset_pooled": pooled([row["note_modes"][name][tolerance]["onset"] for row in records]),
                    "mean_clip_onset_f1": float(np.mean([row["note_modes"][name][tolerance]["onset"]["f1"] for row in records])),
                    "mean_clip_frame_f1": float(np.mean([row["note_modes"][name][tolerance]["frame"]["f1"] for row in records]))}
                    for tolerance in ("0.05", "0.1", "0.2")}
                    for name in ("strict_saved", "direct_activity_runs_backdate_006")}
    summary["diagnostic_frame_lag_ms"] = {key: {"pooled": pooled([row["diagnostic_frame_lag_ms"][key] for row in records]),
                    "mean_clip_f1": float(np.mean([row["diagnostic_frame_lag_ms"][key]["f1"] for row in records]))}
                    for key in records[0]["diagnostic_frame_lag_ms"]}
    result = {"scope": "Validation progression-2 only; first 10 seconds scored with 12 seconds input.",
              "new_optimizer_updates": 0, "test_used": False, "checkpoint_sha256": sha256(checkpoint_path),
              "checkpoint_path": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
              "manifest_sha256": sha256(manifest_path), "head_audit_sha256": sha256(Path(__file__)),
              "decoder_parameters": checkpoint["decoder_parameters"],
              "target_grid": "Frame i at (i+1)*hop/sample_rate; conflicting string frames ignored in state metrics.",
              "raw_run_limit": "Diagnostic direct runs oversegment; no onset evidence or duration filtering. Not a production decoder.",
              "lag_limit": "Signed lag is diagnostic: positive means compare target to later model frames; no production correction selected.",
              "summary": summary, "clips": records}
    save_json(output / "validation-head-diagnostics.json", result)
    text = ["# Stage 9 validation head diagnosis", "", result["scope"], "", "Zero optimizer updates; no test clips used.", "",
            "| Mode | Mean clip F1 | Pooled precision | Pooled recall |", "|---|---:|---:|---:|"]
    for key in ("state_argmax", "state_activity_035", "activity_argmax", "activity_035"):
        row = summary[key]
        text.append(f"| {key} | {row['mean_clip_f1']:.4f} | {row['pooled']['precision']:.4f} | {row['pooled']['recall']:.4f} |")
    text.extend(["", "| Pitch threshold | Mean F1 | Precision | Recall |", "|---|---:|---:|---:|"])
    for key, row in summary["pitch"].items():
        text.append(f"| {key} | {row['mean_clip_f1']:.4f} | {row['pooled']['precision']:.4f} | {row['pooled']['recall']:.4f} |")
    text.extend(["", "| Note mode | Tolerance ms | Mean onset F1 | Pooled precision | Pooled recall |", "|---|---:|---:|---:|---:|"])
    for name, modes in summary["note_modes"].items():
        for key, row in modes.items():
            text.append(f"| {name} | {1000*float(key):.0f} | {row['mean_clip_onset_f1']:.4f} | {row['onset_pooled']['precision']:.4f} | {row['onset_pooled']['recall']:.4f} |")
    text.extend(["", "| Active class runs | Count | Median frames | Under 4 frames | Under 10 frames |", "|---|---:|---:|---:|---:|"])
    for key, row in summary["runs"].items():
        text.append(f"| {key} | {row['count']} | {row['p50']} | {row['fraction_under_4_frames']:.3f} | {row['fraction_under_10_frames']:.3f} |")
    best_lag = max(summary["diagnostic_frame_lag_ms"], key=lambda k: summary["diagnostic_frame_lag_ms"][k]["mean_clip_f1"])
    text.extend(["", f"Best diagnostic frame lag: {best_lag} ms, mean exact string/fret F1 "
                 f"{summary['diagnostic_frame_lag_ms'][best_lag]['mean_clip_f1']:.4f}. "
                 "Positive lag compares labels to later predictions; this is descriptive and is not selected for production.",
                 "", result["raw_run_limit"], "", "Per-string probability, class margin and per-clip diagnostics are saved in the JSON."])
    (output / "validation-head-diagnostics.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
