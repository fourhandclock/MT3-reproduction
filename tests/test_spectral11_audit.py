import copy

import numpy as np
import pytest
import torch

from spectral11.audit import (aligned_heads, aligned_pitch, choose_arm, independent_heads,
                              matched_arms, validate_budget,
                              validate_normalization, _optimizer_receipt)


def receipt(name):
    value = {"name": name, "optimizer_steps": 250, "selected_step": 250,
             "history": [{"optimizer_steps": step} for step in range(50, 251, 50)],
             "actual_optimizer_step_counters": {"minimum": 250, "maximum": 250},
             "target_delay_frames": 13, "delay_frames": 13,
             "common_leading_mask_frames": 26, "warmstarted": False,
             "validation_used_during_training": False}
    value.update({key: "a" * 64 for key in ("initial_weights_sha256", "crop_schedule_sha256",
                                           "gain_schedule_sha256", "common_support_sha256",
                                           "manifest_sha256", "cache_audit_sha256")})
    return value


def test_fixed_500_budget_counts_actual_optimizer_updates():
    receipts = {name: receipt(name) for name in ("short128", "long256")}
    assert validate_budget(receipts)["optimizer_steps"] == 500
    receipts["long256"]["actual_optimizer_step_counters"]["minimum"] = 249
    with pytest.raises(ValueError, match="actual optimizer"):
        validate_budget(receipts)


@pytest.mark.parametrize("change", ["extra", "repeat", "missing", "test", "too_many", "steps_bool"])
def test_budget_rejects_retry_or_incomplete_ledgers(change):
    receipts = {name: receipt(name) for name in ("short128", "long256")}
    if change == "extra":
        receipts["retry"] = receipt("retry")
    elif change == "repeat":
        receipts["long256"]["history"][2] = {"optimizer_steps": 100}
    elif change == "missing":
        del receipts["long256"]
    elif change == "test":
        receipts["long256"]["test_inference_calls"] = 1
    elif change == "too_many":
        receipts["short128"]["optimizer_steps"] = 251
    else:
        receipts["short128"]["optimizer_steps"] = True
    with pytest.raises(ValueError):
        validate_budget(receipts)


def test_arm_match_allows_frontend_statistics_to_differ_but_not_exposure():
    short, long = receipt("short128"), receipt("long256")
    short["normalization"] = {"mean": [0] * 216}
    long["normalization"] = {"mean": [1] * 216}
    assert matched_arms(short, long)["initial_weights_sha256"] == "a" * 64
    long["gain_schedule_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="gain_schedule"):
        matched_arms(short, long)


@pytest.mark.parametrize("key,value", [("target_delay_frames", 6), ("target_delay_frames", 0),
                                       ("common_leading_mask_frames", 13),
                                       ("selected_step", 200), ("warmstarted", True),
                                       ("validation_used_during_training", True)])
def test_arm_match_rejects_supervision_or_selection_confound(key, value):
    short, long = receipt("short128"), receipt("long256")
    long[key] = value
    with pytest.raises(ValueError):
        matched_arms(short, long)


def test_selection_uses_acoustic_primary_score_and_stable_short_tie():
    values = {"short128": {"exact_state_f1": .1, "onset_f1": .2},
              "long256": {"exact_state_f1": .1, "onset_f1": .2}}
    assert choose_arm(values) == "short128"
    values["long256"]["onset_f1"] = .3
    assert choose_arm(values) == "long256"
    values["short128"]["exact_state_f1"] = .100001
    assert choose_arm(values) == "short128"
    values["long256"]["exact_state_f1"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        choose_arm(values)


def targets_and_probabilities():
    states = np.full((1200, 6), 25, np.int16)
    states[10:20, 0] = 7  # This active interval is outside the shared source mask.
    states[40:60, 0] = 3
    states[50:60, 1] = -100  # Conflicts must not count as missed active frets.
    probabilities = np.zeros((1200, 6, 26), np.float32)
    probabilities[:, :, 25] = 1
    for i in range(1187):
        for string in range(6):
            target = states[i, string]
            if target >= 0:
                probabilities[i + 13, string] = 0
                probabilities[i + 13, string, target] = 1
    return {"state": states}, {"state": probabilities}


def test_aligned_head_score_shifts_prediction_once_and_masks_source_warmup():
    targets, probability = targets_and_probabilities()
    result = aligned_heads(probability, targets)
    assert result["source_frames_scored"] == 1000
    assert result["prediction_start_frame"] == 13
    assert result["active_frames"] == 20
    assert result["exact_state_f1"] == result["active_fret_accuracy"] == 1
    assert result["distinct_frets"] == 1
    # Delaying targets again would wrongly make a perfect prediction lose F1.
    shifted = np.full_like(targets["state"], 25)
    shifted[13:] = targets["state"][:-13]
    assert aligned_heads(probability, {"state": shifted})["exact_state_f1"] < 1


def test_head_score_does_not_reward_inactive_accuracy_as_fret_recognition():
    targets, probability = targets_and_probabilities()
    probability["state"][:] = 0
    probability["state"][:, :, 25] = 1
    result = aligned_heads(probability, targets)
    assert result["exact_state_f1"] == result["active_fret_accuracy"] == 0
    assert result["active_frames"] == 20


@pytest.mark.parametrize("change", ["nan", "negative", "unnormalized", "bad_shape"])
def test_saved_probabilities_must_be_valid_normalized_state_distributions(change):
    target = np.full((2, 6), 25, np.int16)
    probability = np.zeros((2, 6, 26), np.float32)
    probability[:, :, 25] = 1
    if change == "nan":
        probability[0, 0, 0] = np.nan
    elif change == "negative":
        probability[0, 0, 0] = -.1
    elif change == "unnormalized":
        probability[0, 0, 25] = .9
    else:
        probability = probability[:, :, :-1]
    with pytest.raises(ValueError):
        independent_heads(probability, target, np.ones(2))


def test_training_normalization_includes_startup_despite_loss_warmup_mask():
    records = [{"features": np.full((30, 216), float(i), np.float32)} for i in range(48)]
    for record in records:
        record["features"][:26] = 1000
    population = np.concatenate([record["features"][:, 0] for record in records])
    mean = float(population.mean(dtype=np.float64))
    variance = float(np.square(population).mean(dtype=np.float64)) - mean ** 2
    normalization = {"mean": [mean] * 216, "std": [np.sqrt(variance)] * 216}
    assert validate_normalization(normalization, records)["training_frames"] == 48 * 30
    altered = copy.deepcopy(normalization)
    altered["mean"][0] += .01
    with pytest.raises(ValueError, match="training-only"):
        validate_normalization(altered, records)


def test_training_normalization_preserves_frozen_point_three_std_floor():
    records = [{"features": np.zeros((30, 216), np.float32)} for _ in range(48)]
    normalization = {"mean": [0.] * 216, "std": [.3] * 216}
    assert validate_normalization(normalization, records)["variance_floor"] == .09
    normalization["std"] = [.001] * 216
    with pytest.raises(ValueError, match="training-only"):
        validate_normalization(normalization, records)


def test_pitch_diagnostic_uses_same_source_mask_and_prespecified_low_register():
    target = np.zeros((1100, 49), np.uint8)
    target[30:40, 0] = 1  # MIDI 40: low-register coverage.
    target[50:55, 12] = 1  # MIDI 52: overall, outside low-register range.
    target[10:20, 0] = 1  # Excluded source warmup.
    probability = np.zeros((1200, 49), np.float32)
    probability[13:1113] = target
    values = aligned_pitch({"pitch": probability}, {"pitch": target})
    assert values["overall"]["f1"] == values["low_register"]["f1"] == 1
    assert values["overall"]["reference_count"] == 15
    assert values["low_register"]["reference_count"] == 10
    assert values["low_register_midi"] == [40, 51]
    assert values["threshold"] == .5


def test_final_checkpoint_optimizer_state_corroborates_receipt_not_just_loop_count():
    checkpoint = {"step": 250, "optimizer": {"state": {0: {"step": torch.tensor(250.)},
                                                       1: {"step": torch.tensor(250.)}}}}
    receipt = {"actual_optimizer_step_counters": {"minimum": 250, "maximum": 250, "parameter_states": 2}}
    _optimizer_receipt(checkpoint, receipt, 2)
    checkpoint["optimizer"]["state"][1]["step"] = torch.tensor(249.)
    with pytest.raises(ValueError, match="counter"):
        _optimizer_receipt(checkpoint, receipt, 2)


@pytest.mark.parametrize("change", ["missing_parameter", "fractional", "nan"])
def test_actual_optimizer_audit_rejects_missing_or_invalid_updates(change):
    checkpoint = {"step": 250, "optimizer": {"state": {0: {"step": torch.tensor(250.)},
                                                       1: {"step": torch.tensor(250.)}}}}
    receipt = {"actual_optimizer_step_counters": {"minimum": 250, "maximum": 250, "parameter_states": 2}}
    if change == "missing_parameter":
        del checkpoint["optimizer"]["state"][1]
    else:
        checkpoint["optimizer"]["state"][1]["step"] = torch.tensor(249.5 if change == "fractional" else np.nan)
    with pytest.raises(ValueError):
        _optimizer_receipt(checkpoint, receipt, 2)
