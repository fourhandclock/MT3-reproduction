import copy

import pytest
import torch

from acoustic10.audit import (audit_note_ledger, choose_arm, matched_arms,
                             validate_budget, validate_sanity, weight_digest)


def receipt(steps):
    return {"optimizer_steps": steps, "history": [{"optimizer_steps": steps // 5 * i} for i in range(1, 6)],
            "blocks_completed": 5, "test_used": False}


def arm(delay):
    value = receipt(200)
    value.update({key: "a" * 64 for key in ("initial_weights_sha256", "crop_schedule_sha256",
                  "gain_schedule_sha256", "common_support_sha256", "manifest_sha256", "cache_audit_sha256")})
    value.update(warmstarted=False, selected_step=200, common_leading_mask_frames=6, delay_frames=delay)
    return value


def ledger():
    note = {"note_id": 0, "string": 0, "fret": 2, "pitch": 42, "start": .12, "end": .4}
    events = []
    for kind, boundary, emission in (("note_on", "start", .22), ("note_off", "end", .48)):
        events.append(dict(note_id=0, string=0, fret=2, pitch=42, type=kind,
                           time=note[boundary], source_time=note[boundary],
                           emitted_at_audio_seconds=emission, delivered_at_seconds=emission + .01,
                           **{boundary: note[boundary]}))
    return [note], events


def test_weight_identity_ignores_dictionary_order_and_serialization():
    a = {"x": torch.tensor([1., 2.]), "y": torch.tensor(3, dtype=torch.int64)}
    b = {"y": a["y"].clone(), "x": a["x"].clone()}
    assert weight_digest(a) == weight_digest(b)
    b["x"][0] = 1.0001
    assert weight_digest(a) != weight_digest(b)


def test_weight_identity_binds_dtype_shape_and_name():
    base = {"x": torch.arange(4, dtype=torch.float32)}
    assert weight_digest(base) != weight_digest({"x": base["x"].reshape(2, 2)})
    assert weight_digest(base) != weight_digest({"x": base["x"].double()})
    assert weight_digest(base) != weight_digest({"other": base["x"]})


def test_budget_accounts_for_sanity_and_all_blocks():
    values = {"sanity": receipt(100), "baseline": receipt(200), "delay60": receipt(200)}
    assert validate_budget(values)["optimizer_steps"] == 500
    values["sanity"]["optimizer_steps"] += 1
    with pytest.raises(ValueError, match="update budget"):
        validate_budget(values)


@pytest.mark.parametrize("change", ["test", "repeat", "missing_last", "extra_block", "unbudgeted"])
def test_budget_rejects_receipt_changes(change):
    values = {"sanity": receipt(100), "baseline": receipt(200), "delay60": receipt(200)}
    if change == "test":
        values["baseline"]["test_used"] = True
    elif change == "repeat":
        values["delay60"]["history"][2] = copy.deepcopy(values["delay60"]["history"][1])
    elif change == "missing_last":
        values["delay60"]["history"].pop()
    elif change == "extra_block":
        values["sanity"]["history"].append({"optimizer_steps": 100})
    else:
        values["retry"] = receipt(100)
    with pytest.raises(ValueError):
        validate_budget(values)


def test_failed_sanity_can_end_the_stage_without_unbudgeted_arms():
    assert validate_budget({"sanity": receipt(100)}, require_complete=False)["optimizer_steps"] == 100
    with pytest.raises(ValueError, match="both matched arms"):
        validate_budget({"sanity": receipt(100)})


def test_sanity_verdict_uses_active_quality_and_coverage():
    initial = {"state_loss": 3.}
    final = {"state_loss": 1.5, "active_fret_accuracy": .8, "exact_state_f1": .75}
    coverage = {"active_frames": 100, "distinct_frets": 4}
    assert validate_sanity(initial, final, coverage, True)["passed"]
    poor = dict(final, exact_state_f1=.74)
    assert not validate_sanity(initial, poor, coverage)["passed"]
    with pytest.raises(ValueError, match="contradicts"):
        validate_sanity(initial, poor, coverage, True)
    assert not validate_sanity(initial, final, dict(coverage, distinct_frets=3))["passed"]


def test_sanity_rejects_nan_and_inactive_only_claims():
    with pytest.raises(ValueError, match="finite"):
        validate_sanity({"state_loss": 3}, {"state_loss": float("nan"), "active_fret_accuracy": 1., "exact_state_f1": 1.},
                        {"active_frames": 100, "distinct_frets": 4})
    assert not validate_sanity({"state_loss": 3}, {"state_loss": .5, "active_fret_accuracy": 1., "exact_state_f1": 1.},
                               {"active_frames": 0, "distinct_frets": 0})["passed"]


def test_matching_requires_identical_exposure_and_fresh_final_models():
    baseline, delayed = arm(0), arm(6)
    assert matched_arms(baseline, delayed)["initial_weights_sha256"] == "a" * 64
    delayed["gain_schedule_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="gain_schedule"):
        matched_arms(baseline, delayed)


@pytest.mark.parametrize("field,value", [("warmstarted", True), ("selected_step", 160),
                                          ("common_leading_mask_frames", 0), ("delay_frames", 5)])
def test_matching_rejects_changes_in_fixed_protocol(field, value):
    baseline, delayed = arm(0), arm(6)
    delayed[field] = value
    with pytest.raises(ValueError):
        matched_arms(baseline, delayed)


def test_selection_applies_primary_secondary_and_stable_baseline_ties():
    rows = {"baseline": {"exact_state_f1": .2, "onset_f1": .1},
            "delay60": {"exact_state_f1": .2, "onset_f1": .1}}
    assert choose_arm(rows) == "baseline"
    rows["delay60"]["onset_f1"] = .11
    assert choose_arm(rows) == "delay60"
    rows["baseline"]["exact_state_f1"] = .201
    assert choose_arm(rows) == "baseline"
    rows["baseline"]["exact_state_f1"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        choose_arm(rows)


def test_note_audit_checks_source_and_publication_clocks_separately():
    notes, events = ledger()
    assert audit_note_ledger(notes, events) == {"notes": 1, "events": 2}
    notes[0]["start"] = .13
    with pytest.raises(ValueError, match="revises"):
        audit_note_ledger(notes, events)


@pytest.mark.parametrize("change", ["duplicate", "missing", "delivery", "source", "pitch"])
def test_note_audit_rejects_corrupted_event_receipts(change):
    notes, events = ledger()
    if change == "duplicate":
        events.append(events[-1].copy())
    elif change == "missing":
        events.pop()
    elif change == "delivery":
        events[0]["delivered_at_seconds"] = .1
    elif change == "source":
        events[0]["source_time"] = .25
    else:
        events[0]["pitch"] += 1
    with pytest.raises(ValueError):
        audit_note_ledger(notes, events)
