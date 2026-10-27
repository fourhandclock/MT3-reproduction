import copy
import numpy as np
import pytest

from diagnosis9.oracles import (decode_packets, diagnose_case, oracle_probabilities,
                               replace_heads, structural_counts, validate_cached_target_identity,
                               hybrid_variants)


CONFIG = {"feature": {"sample_rate": 16000, "hop_length": 160},
          "model": {"max_fret": 24, "pitch_min": 40, "pitch_max": 88},
          "decoder": {"minimum_note_seconds": .04, "minimum_retrigger_seconds": .06},
          "streaming": {"confirm_frames": 4, "release_confirm_frames": 2}}
PARAMETERS = {"onset_threshold": .55, "activity_threshold": .35,
              "onset_backdate_seconds": 0.0}
SCORE = {"score_seconds": .3, "frame_rate": 100, "onset_tolerance_seconds": .05,
         "offset_tolerance_seconds": .05, "offset_ratio": .2}


def note(start, end, fret=0, string=0):
    return {"start": start, "end": end, "fret": fret, "string": string,
            "pitch": [40, 45, 50, 55, 59, 64][string] + fret}


def test_oracle_probabilities_are_exact_right_edge_one_hot_and_do_not_mutate_labels():
    reference = [note(.1, .2, fret=3, string=4)]
    original = copy.deepcopy(reference)
    values, conflicts = oracle_probabilities(reference, 30, CONFIG, .3)
    assert reference == original and conflicts["conflict_string_frames"] == 0
    assert values["state"].shape == (30, 6, 26)
    assert np.all(values["state"].sum(-1) == 1)
    assert set(np.unique(values["state"])) == {0., 1.}
    np.testing.assert_array_equal(np.flatnonzero(values["state"][:, 4, 3]), np.arange(9, 19))
    assert np.argmax(values["onset"][:, 4]) == 9
    assert np.argmax(values["offset"][:, 4]) == 19
    np.testing.assert_array_equal(np.flatnonzero(values["pitch"][:, 62 - 40]), np.arange(9, 19))
    assert np.all(values["state"][:, [0, 1, 2, 3, 5], -1] == 1)


def test_oracle_rejects_inconsistent_per_string_midi_relationship():
    invalid = note(.01, .1, fret=3, string=4)
    invalid["pitch"] += 1
    with pytest.raises(ValueError, match="tuning plus fret"):
        oracle_probabilities([invalid], 30, CONFIG, .3)
    invalid = note(.01, .1)
    invalid["fret"] = 0.5
    with pytest.raises(ValueError, match="integer"):
        oracle_probabilities([invalid], 30, CONFIG, .3)


def test_conflicting_string_states_become_inactive_without_forged_single_fret():
    values, conflicts = oracle_probabilities([note(.01, .08), note(.03, .06, fret=3)], 10, CONFIG, .1)
    assert conflicts["conflict_string_frames"] == 3
    assert conflicts["conflict_frame_count"] == 3
    assert np.all(values["state"][2:5, 0, -1] == 1)
    assert not np.any(values["state"][2:5, 0, :-1])
    assert np.all(values["pitch"][2:5, [0, 3]] == 1)


@pytest.mark.parametrize("attack", [0., .004, .009999])
def test_subhop_attacks_start_at_earliest_observable_frame(attack):
    values, _ = oracle_probabilities([note(attack, .08)], 10, CONFIG, .1)
    notes, events = decode_packets(values, CONFIG, PARAMETERS, .1)
    assert notes[0]["start"] == pytest.approx(.01)
    assert notes[0]["on_emitted_at_audio_seconds"] == pytest.approx(.04)
    assert events[0]["emitted_at_audio_seconds"] >= events[0]["time"]


def test_unheard_attack_and_terminal_release_never_leak_future_frames_or_offset():
    values, _ = oracle_probabilities([note(.1, .5)], 10, CONFIG, .1)
    assert np.all(values["state"][:, :, -1] == 1)
    assert not np.any(values["onset"]) and not np.any(values["offset"])
    values, _ = oracle_probabilities([note(.02, .5)], 10, CONFIG, .105)
    assert len(values["state"]) == 10 and not np.any(values["offset"])
    notes, _ = decode_packets(values, CONFIG, PARAMETERS, .105)
    assert notes[0]["end"] == .105 and notes[0]["censored_at_end"]
    with pytest.raises(ValueError, match="beyond"):
        oracle_probabilities([], 11, CONFIG, .105)


def test_packet_size_changes_arrival_but_not_estimated_source_intervals():
    values, _ = oracle_probabilities([note(.03, .15, fret=2)], 20, CONFIG, .2)
    small, _ = decode_packets(values, CONFIG, PARAMETERS, .2, packet_ms=20)
    large, _ = decode_packets(values, CONFIG, PARAMETERS, .2, packet_ms=80)
    assert [(n["start"], n["end"], n["pitch"]) for n in small] == [(n["start"], n["end"], n["pitch"]) for n in large]
    assert small[0]["on_emitted_at_audio_seconds"] == pytest.approx(.06)
    assert large[0]["on_emitted_at_audio_seconds"] == pytest.approx(.08)
    with pytest.raises(ValueError, match="integer"):
        decode_packets(values, CONFIG, PARAMETERS, .2, packet_ms=15)


def test_confirmation_can_remove_short_note_and_repeated_notes_are_counted_separately():
    reference = [note(.01, .03), note(.05, .15, fret=3), note(.15, .25, fret=3)]
    values, _ = oracle_probabilities(reference, 30, CONFIG, .3)
    result = diagnose_case(reference, values, CONFIG, PARAMETERS, SCORE, .3)
    counts = result["structural"]
    assert counts["short_note_reference_count"] == counts["short_note_missed_count"] == 1
    assert counts["repeated_note_reference_count"] == 1
    assert counts["repeated_note_missed_count"] == 0
    assert result["pitch_metrics"]["reference"] == reference
    assert result["pitch_metrics"]["onset"]["reference_count"] == 3


def test_terminal_active_fourth_frame_is_counted_before_censored_eof_release():
    reference = [note(0., .2)]
    values, _ = oracle_probabilities(reference, 4, CONFIG, .04)
    notes, _ = decode_packets(values, CONFIG, PARAMETERS, .04)
    counts = structural_counts(reference, notes, CONFIG, SCORE, .04)
    assert counts["short_note_reference_count"] == 0
    assert counts["same_string_matched_note_count"] == 1
    assert counts["unobserved_terminal_release_count"] == 1
    assert notes[0]["censored_at_end"]


def test_head_interventions_are_explicit_and_do_not_modify_real_predictions():
    oracle, _ = oracle_probabilities([note(.01, .08)], 10, CONFIG, .1)
    predicted = {key: np.zeros_like(value) for key, value in oracle.items()}
    hybrid = replace_heads(predicted, oracle, ["state"])
    np.testing.assert_array_equal(hybrid["state"], oracle["state"])
    assert not np.any(hybrid["onset"])
    hybrid["state"][0, 0, 0] = .25
    assert not np.any(predicted["state"]) and oracle["state"][0, 0, 0] == 1
    with pytest.raises(ValueError, match="Unknown"):
        replace_heads(predicted, oracle, ["reference"])
    bad = dict(predicted, state=predicted["state"][:-1])
    with pytest.raises(ValueError, match="grid"):
        replace_heads(bad, oracle, ["state"])


def test_target_identity_requires_original_annotation_times_and_exact_heads():
    from streaming.targets import make_targets

    reference = [note(.01, .08)]
    labels, _ = make_targets(reference, 10, CONFIG, .1)
    assert validate_cached_target_identity(reference, labels, 10, CONFIG, .1) == 0
    changed = {key: value.copy() for key, value in labels.items()}
    changed["onset"][0, 0] = 0
    with pytest.raises(ValueError, match="annotation: onset"):
        validate_cached_target_identity(reference, changed, 10, CONFIG, .1)
    with pytest.raises(ValueError, match="four label heads"):
        validate_cached_target_identity(reference, {"state": labels["state"]}, 10, CONFIG, .1)


def test_hybrid_factorial_controls_keep_nonintervened_heads_and_thresholds_fixed():
    oracle, _ = oracle_probabilities([note(.01, .08)], 10, CONFIG, .1)
    predicted = {key: np.full_like(value, .2) for key, value in oracle.items()}
    parameters = dict(PARAMETERS, onset_backdate_seconds=.06)
    variants = hybrid_variants(predicted, oracle, parameters)
    for backdate in (0., .06):
        group = [value for value in variants.values() if value[1]["onset_backdate_seconds"] == backdate]
        assert len(group) == 4
        assert {tuple(sorted(heads)) for _, _, heads in group} == {(), ("state",), ("onset",), ("onset", "state")}
        for probabilities, settings, heads in group:
            assert settings["onset_threshold"] == parameters["onset_threshold"]
            assert settings["activity_threshold"] == parameters["activity_threshold"]
            for key in ("state", "onset", "offset", "pitch"):
                np.testing.assert_array_equal(probabilities[key], oracle[key] if key in heads else predicted[key])
    assert parameters["onset_backdate_seconds"] == .06


def test_oracle_path_is_independent_of_audio_features_and_model_load(monkeypatch):
    # The low-level experiment takes already named probability heads or labels;
    # patching either legacy feature extraction or model construction cannot be
    # invoked by it. Labels are privileged only in oracle_probabilities.
    import cnn_gru.data
    import cnn_gru.model

    def forbidden(*args, **kwargs):
        raise AssertionError("Oracle diagnostic must not call audio features or model construction")

    monkeypatch.setattr(cnn_gru.data, "features", forbidden)
    monkeypatch.setattr(cnn_gru.model, "GuitarCNNGRU", forbidden)
    values, _ = oracle_probabilities([note(.01, .08)], 10, CONFIG, .1)
    result = diagnose_case([note(.01, .08)], values, CONFIG, PARAMETERS, SCORE, .1)
    assert result["pitch_metrics"]["onset"]["f1"] == 1
