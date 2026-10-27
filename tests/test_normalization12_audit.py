import copy

import pytest

from normalization12.audit import choose_arm, matched_arms, validate_budget, without_delivery_clocks


def receipt(groups):
    value = {"name": "short128" if groups == 8 else "group1",
             "optimizer_steps": 250, "selected_step": 250,
             "history": [{"optimizer_steps": step} for step in range(50, 251, 50)],
             "actual_optimizer_step_counters": {"minimum": 250, "maximum": 250},
             "target_delay_frames": 13, "delay_frames": 13,
             "common_leading_mask_frames": 26, "crop_edge_mask_frames": 26,
             "warmstarted": False, "validation_used_during_training": False,
             "normalization": {"mean": [0.], "std": [.3]}, "positive_weights": {"onset": 2.},
             "model_config": {"model": {"dropout": .1}, "feature": {"window_length": 2048},
                              "training": {"max_steps": 250}}}
    if groups == 1:
        value["model_config"]["model"]["cnn_norm_groups"] = 1
    value.update({key: "a" * 64 for key in ("initial_weights_sha256", "initial_checkpoint_sha256",
                                           "crop_schedule_sha256", "gain_schedule_sha256",
                                           "common_support_sha256", "manifest_sha256", "cache_audit_sha256")})
    return value


def test_historical_control_does_not_consume_new_stage_budget():
    result = validate_budget({"group8": receipt(8), "group1": receipt(1)})
    assert result["new_optimizer_updates"] == result["historical_optimizer_updates"] == 250
    assert result["compared_optimizer_updates"] == 500
    assert result["new_sanity_updates"] == 0
    assert result["complete"]


@pytest.mark.parametrize("change", ["extra", "repeat", "missing", "test", "too_many", "boolean", "amp_skip"])
def test_budget_rejects_unbudgeted_or_partial_update_ledgers(change):
    values = {"group8": receipt(8), "group1": receipt(1)}
    if change == "extra":
        values["retry"] = receipt(1)
    elif change == "repeat":
        values["group1"]["history"][2]["optimizer_steps"] = 100
    elif change == "missing":
        del values["group8"]
    elif change == "test":
        values["group1"]["test_inference_calls"] = 1
    elif change == "too_many":
        values["group1"]["optimizer_steps"] = 251
    elif change == "boolean":
        values["group1"]["optimizer_steps"] = True
    else:
        values["group1"]["actual_optimizer_step_counters"]["minimum"] = 249
    with pytest.raises(ValueError):
        validate_budget(values)


def test_match_uses_original_control_recipe_and_allows_only_group_count():
    control, variant = receipt(8), receipt(1)
    control["model_config"]["stage_budget"] = {"purpose": "Stage11", "total": 500}
    variant["model_config"]["stage_budget"] = {"purpose": "Stage12", "new": 250}
    assert matched_arms(control, variant)["initial_weights_sha256"] == "a" * 64
    assert control["model_config"]["stage_budget"]["purpose"] == "Stage11"


@pytest.mark.parametrize("key", ["initial_weights_sha256", "initial_checkpoint_sha256",
                                "crop_schedule_sha256", "gain_schedule_sha256",
                                "common_support_sha256", "manifest_sha256", "cache_audit_sha256"])
def test_match_rejects_changed_weight_exposure_or_source_binding(key):
    control, variant = receipt(8), receipt(1)
    variant[key] = "b" * 64
    with pytest.raises(ValueError, match=key):
        matched_arms(control, variant)


@pytest.mark.parametrize("change", ["feature", "training", "dropout", "normalization", "weights", "groups"])
def test_match_rejects_another_experimental_variable(change):
    control, variant = receipt(8), receipt(1)
    if change == "feature":
        variant["model_config"]["feature"]["window_length"] = 4096
    elif change == "training":
        variant["model_config"]["training"]["max_steps"] = 300
    elif change == "dropout":
        variant["model_config"]["model"]["dropout"] = 0
    elif change == "normalization":
        variant["normalization"]["mean"] = [1.]
    elif change == "weights":
        variant["positive_weights"]["onset"] = 3.
    else:
        variant["model_config"]["model"]["cnn_norm_groups"] = 4
    with pytest.raises(ValueError):
        matched_arms(control, variant)


@pytest.mark.parametrize("key,value", [("target_delay_frames", 6), ("delay_frames", 0),
                                      ("common_leading_mask_frames", 13), ("crop_edge_mask_frames", 13),
                                      ("selected_step", 200), ("warmstarted", True),
                                      ("early_stopping_used", True), ("validation_used_during_training", True)])
def test_match_rejects_target_or_selection_confound(key, value):
    control, variant = receipt(8), receipt(1)
    variant[key] = value
    with pytest.raises(ValueError):
        matched_arms(control, variant)


def test_selection_primary_score_then_onset_with_stable_control_tie():
    values = {"group8": {"exact_state_f1": .1, "onset_f1": .2},
              "group1": {"exact_state_f1": .1, "onset_f1": .2}}
    assert choose_arm(values) == "group8"
    values["group1"]["onset_f1"] = .3
    assert choose_arm(values) == "group1"
    values["group8"]["exact_state_f1"] = .100001
    assert choose_arm(values) == "group8"
    original = copy.deepcopy(values)
    values["group1"]["exact_state_f1"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        choose_arm(values)
    assert choose_arm(original) == "group8"


def test_historical_score_comparison_allows_only_remeasured_delivery_clocks():
    old = {"onset": {"f1": .07}, "predicted_notes": [{"start": .3, "end": .5, "pitch": 40,
           "on_emitted_at_audio_seconds": .4, "on_delivered_at_seconds": .41,
           "off_emitted_at_audio_seconds": .6, "off_delivered_at_seconds": .61}]}
    current = copy.deepcopy(old)
    current["predicted_notes"][0]["on_delivered_at_seconds"] = .43
    current["predicted_notes"][0]["off_delivered_at_seconds"] = .64
    assert without_delivery_clocks(old) == without_delivery_clocks(current)
    assert "on_delivered_at_seconds" in old["predicted_notes"][0]  # Preserve artifact data.
    current["predicted_notes"][0]["start"] = .31
    assert without_delivery_clocks(old) != without_delivery_clocks(current)


def test_historical_score_comparison_keeps_publication_times_and_f1():
    old = {"f1": .07, "predicted_notes": [{"start": .3, "on_emitted_at_audio_seconds": .4}]}
    current = copy.deepcopy(old)
    current["predicted_notes"][0]["on_emitted_at_audio_seconds"] = .42
    assert without_delivery_clocks(old) != without_delivery_clocks(current)
    current = copy.deepcopy(old)
    current["f1"] = .08
    assert without_delivery_clocks(old) != without_delivery_clocks(current)
