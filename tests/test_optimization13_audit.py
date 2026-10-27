import copy

import numpy as np
import pytest
import torch

from optimization13.audit import (choose_arm, optimizer_counters, optimizer_digest,
                                  paired_deltas, validate_budget, verify_restoration, verify_schedule)


def receipt():
    return {"parent_optimizer_steps": 250, "new_optimizer_steps": 250,
            "optimizer_steps": 500, "selected_step": 500,
            "history": [{"optimizer_steps": 250 + step, "new_optimizer_steps": step}
                        for step in range(50, 251, 50)],
            "actual_optimizer_step_counters": {"minimum": 500, "maximum": 500, "parameter_states": 34},
            "validation_used_during_training": False, "test_inference_calls": 0}


def checkpoint(step):
    return {"step": step, "model": {"weight" + str(index): torch.tensor([float(step)]) for index in range(34)},
            "optimizer": {"state": {index: {"step": torch.tensor(float(step)),
                                            "exp_avg": torch.tensor([index + step / 1000.]),
                                            "exp_avg_sq": torch.tensor([index + step / 10000.])}
                                     for index in range(34)},
                          "param_groups": [{"lr": .0001, "initial_lr": .001,
                                            "params": list(range(34)), "betas": (.9, .999),
                                            "weight_decay": .0001, "eps": 1e-8}]}}


def test_budget_counts_frozen_parents_separately_from_new_training():
    result = validate_budget({"group8": receipt(), "group1": receipt()})
    assert result["new_optimizer_updates"] == result["historical_optimizer_updates"] == 500
    assert result["lifetime_optimizer_updates"] == 1000
    assert result["new_sanity_updates"] == 0
    assert result["complete"]


@pytest.mark.parametrize("change", ["extra", "missing", "parent", "new", "lifetime", "boolean", "counter",
                                   "skip", "block", "selected", "validation", "test", "early"])
def test_budget_rejects_resume_and_optimizer_accounting_errors(change):
    values = {"group8": receipt(), "group1": receipt()}
    row = values["group1"]
    if change == "extra":
        values["retry"] = receipt()
    elif change == "missing":
        del values["group8"]
    elif change == "parent":
        row["parent_optimizer_steps"] = 0
    elif change == "new":
        row["new_optimizer_steps"] = 251
    elif change == "lifetime":
        row["optimizer_steps"] = 250
    elif change == "boolean":
        row["new_optimizer_steps"] = True
    elif change == "counter":
        row["actual_optimizer_step_counters"]["minimum"] = 499
    elif change == "skip":
        row["actual_optimizer_step_counters"]["parameter_states"] = 33
    elif change == "block":
        row["history"][1]["new_optimizer_steps"] = 50
    elif change == "selected":
        row["selected_step"] = 450
    elif change == "validation":
        row["validation_used_during_training"] = True
    elif change == "test":
        row["test_inference_calls"] = 1
    else:
        row["early_stopping_used"] = True
    with pytest.raises(ValueError):
        validate_budget(values)


def test_partial_budget_is_explicitly_incomplete():
    row = receipt()
    row.update(new_optimizer_steps=50, optimizer_steps=300, history=row["history"][:1],
               actual_optimizer_step_counters={"minimum": 300, "maximum": 300, "parameter_states": 34})
    result = validate_budget({"group8": row}, require_complete=False)
    assert not result["complete"]
    assert result["new_optimizer_updates"] == 50


def test_optimizer_digest_is_order_independent_but_binds_all_tensor_bytes():
    state = checkpoint(250)["optimizer"]
    reordered = {"param_groups": copy.deepcopy(state["param_groups"]),
                 "state": dict(reversed(list(state["state"].items())))}
    assert optimizer_digest(state) == optimizer_digest(reordered)
    changed = copy.deepcopy(state)
    changed["state"][0]["exp_avg"][0] += .001
    assert optimizer_digest(state) != optimizer_digest(changed)
    changed = copy.deepcopy(state)
    changed["state"][0]["step"] = torch.tensor(250, dtype=torch.int64)
    assert optimizer_digest(state) != optimizer_digest(changed)
    changed = copy.deepcopy(state)
    changed["param_groups"][0]["lr"] = .001
    assert optimizer_digest(state) != optimizer_digest(changed)


def test_restoration_checks_actual_moments_model_and_step_delta():
    parent, restored, final = checkpoint(250), checkpoint(250), checkpoint(500)
    result = verify_restoration(parent, restored, final, receipt())
    assert result["parent_weights_sha256"] != result["final_weights_sha256"]
    assert result["parent_optimizer_sha256"] != result["final_optimizer_sha256"]


@pytest.mark.parametrize("change", ["model", "moments", "restored_counter", "final_counter", "lr",
                                   "negative_square", "nan", "missing_parameter", "param_order", "unchanged"])
def test_restoration_rejects_reset_optimizer_and_bad_resume_tensors(change):
    parent, restored, final = checkpoint(250), checkpoint(250), checkpoint(500)
    if change == "model":
        restored["model"]["weight0"] += 1
    elif change == "moments":
        restored["optimizer"]["state"][0]["exp_avg"].zero_()
    elif change == "restored_counter":
        restored["optimizer"]["state"][0]["step"].zero_()
    elif change == "final_counter":
        final["optimizer"]["state"][0]["step"] -= 1
    elif change == "lr":
        final["optimizer"]["param_groups"][0]["lr"] = .001
    elif change == "negative_square":
        final["optimizer"]["state"][0]["exp_avg_sq"] *= -1
    elif change == "nan":
        final["model"]["weight0"][0] = float("nan")
    elif change == "missing_parameter":
        del final["optimizer"]["state"][0]
    elif change == "param_order":
        final["optimizer"]["param_groups"][0]["params"].reverse()
    else:
        final["model"] = copy.deepcopy(parent["model"])
    with pytest.raises(ValueError):
        verify_restoration(parent, restored, final, receipt())


def test_noninteger_optimizer_counter_is_rejected():
    value = checkpoint(250)
    value["optimizer"]["state"][0]["step"] += .5
    with pytest.raises(ValueError):
        optimizer_counters(value)


def test_selection_requires_final_arms_and_uses_state_then_onset_with_stable_tie():
    scores = {"group8": {"exact_state_f1": .1, "onset_f1": .2},
              "group1": {"exact_state_f1": .1, "onset_f1": .2}}
    assert choose_arm(scores) == "group8"
    scores["group1"]["onset_f1"] = .3
    assert choose_arm(scores) == "group1"
    scores["group8"]["exact_state_f1"] = .11
    assert choose_arm(scores) == "group8"
    with pytest.raises(ValueError):
        choose_arm({"group8_250": scores["group8"], "group1_500": scores["group1"]})
    scores["group1"]["exact_state_f1"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        choose_arm(scores)


def test_paired_deltas_keep_each_recording_and_dose_aligned():
    records = [{"id": clip, "mode": group + "_" + str(step),
                "head_metrics": {"exact_state_f1": .1 if step == 250 else after},
                "pitch_metrics": {"onset": {"f1": .2 if step == 250 else after}}}
               for clip, after in (("a", .3), ("b", .1))
               for group in ("group8", "group1") for step in (250, 500)]
    result = paired_deltas(records)
    assert result["group8"]["exact_state_f1_delta"] == pytest.approx(.1)
    assert result["group8"]["onset_f1_delta"] == pytest.approx(0.)
    assert [row["id"] for row in result["group1"]["records"]] == ["a", "b"]
    with pytest.raises(ValueError):
        paired_deltas(records[:-1])
    with pytest.raises(ValueError):
        paired_deltas(records + records[:1])


def schedule_and_config():
    config = {"seed": 42, "data_epoch_start": 5,
              "training": {"epochs": 5, "samples_per_epoch": 200, "batch_size": 4,
                           "crop_seconds": 2., "gain_augmentation_log_std": .15}}
    records = [{"id": "one", "frames": 550}, {"id": "two", "frames": 750}]
    schedule = []
    for epoch in range(5, 10):
        generator = torch.Generator().manual_seed(42 + epoch)
        torch.empty((), dtype=torch.int64).random_(generator=generator)
        for index in torch.randperm(200, generator=generator).tolist():
            rng = np.random.default_rng(42 + epoch * 100000 + index)
            record_index = int(rng.integers(len(records)))
            start = int(rng.integers(records[record_index]["frames"] - 200 + 1))
            gain = float(rng.normal(0, .15))
            schedule.append({"epoch": epoch + 1, "sample_index": index, "record_index": record_index,
                             "id": records[record_index]["id"], "start_frame": start,
                             "frames": 200, "gain_log": gain})
    return schedule, config, records


def test_schedule_uses_next_data_epochs_and_all_gain_draws():
    schedule, config, records = schedule_and_config()
    result = verify_schedule(schedule, config, records)
    assert len(schedule) == 1000 and {row["epoch"] for row in schedule} == {6, 7, 8, 9, 10}
    assert len(result["crop_schedule_sha256"]) == 64
    assert result["crop_schedule_sha256"] != result["gain_schedule_sha256"]


@pytest.mark.parametrize("change", ["old_epoch", "gain", "crop", "order", "count"])
def test_schedule_rejects_old_exposure_or_modified_crop_gain(change):
    schedule, config, records = schedule_and_config()
    if change == "old_epoch":
        config["data_epoch_start"] = 0
    elif change == "gain":
        schedule[0]["gain_log"] += .001
    elif change == "crop":
        schedule[0]["start_frame"] += 1
    elif change == "order":
        schedule[0], schedule[1] = schedule[1], schedule[0]
    else:
        schedule.pop()
    with pytest.raises(ValueError):
        verify_schedule(schedule, config, records)
