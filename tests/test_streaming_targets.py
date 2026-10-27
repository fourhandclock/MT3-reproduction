import numpy as np
import pytest

from streaming.targets import make_targets


CONFIG = {"feature": {"sample_rate": 16000, "hop_length": 160},
          "model": {"max_fret": 24, "pitch_min": 40, "pitch_max": 88}}


def note(start, end, fret=0, string=0):
    tuning = [40, 45, 50, 55, 59, 64]
    return {"start": start, "end": end, "fret": fret, "string": string,
            "pitch": tuning[string] + fret}


def test_exact_grid_occupancy_is_right_edge_and_half_open():
    targets, conflicts = make_targets([note(.1, .2)], 30, CONFIG, .3)
    np.testing.assert_array_equal(np.flatnonzero(targets["state"][:, 0] == 0), np.arange(9, 19))
    np.testing.assert_array_equal(np.flatnonzero(targets["pitch"][:, 0]), np.arange(9, 19))
    assert np.argmax(targets["onset"][:, 0]) == 9
    assert np.argmax(targets["offset"][:, 0]) == 19
    assert targets["state"].dtype == np.int16
    assert targets["onset"].dtype == targets["offset"].dtype == np.float16
    assert targets["pitch"].dtype == np.uint8 and conflicts == 0


@pytest.mark.parametrize("attack", [0., .004, .009999, .01])
def test_initial_sub_hop_attacks_have_first_frame_targets(attack):
    targets, _ = make_targets([note(attack, .05)], 10, CONFIG, .1)
    assert targets["state"][0, 0] == 0
    assert targets["onset"][0, 0] == 1
    assert np.argmax(targets["onset"][:, 0]) == 0


@pytest.mark.parametrize("attack", [.1, .10000001, .101, .109])
def test_unheard_attacks_never_leak_gaussian_labels(attack):
    targets, conflicts = make_targets([note(attack, .2)], 10, CONFIG, .1)
    assert np.all(targets["state"] == 25)
    assert not np.any(targets["onset"])
    assert not np.any(targets["offset"])
    assert not np.any(targets["pitch"]) and conflicts == 0


@pytest.mark.parametrize("release", [.1, .10000001, .5])
def test_terminal_release_is_censored_but_occupancy_is_preserved(release):
    targets, _ = make_targets([note(.02, release)], 10, CONFIG, .1)
    assert not np.any(targets["offset"])
    expected_end = 9 if release == .1 else 10
    np.testing.assert_array_equal(np.flatnonzero(targets["state"][:, 0] == 0), np.arange(1, expected_end))


def test_fractional_boundary_has_continuous_gaussian_and_exact_occupancy():
    targets, _ = make_targets([note(.105, .205)], 30, CONFIG, .3)
    np.testing.assert_array_equal(np.flatnonzero(targets["state"][:, 0] == 0), np.arange(10, 20))
    assert targets["onset"][9, 0] == targets["onset"][10, 0]
    assert targets["onset"][9, 0] == pytest.approx(np.exp(-.5 * .5 ** 2), abs=.0003)
    assert targets["onset"][6, 0] == 0  # Distance 3.5 frames exceeds support.


def test_same_string_conflicts_are_masked_and_pitch_labels_remain_present():
    targets, conflicts = make_targets([note(.01, .08), note(.03, .06, fret=3)], 10, CONFIG, .1)
    np.testing.assert_array_equal(np.flatnonzero(targets["state"][:, 0] == -100), np.arange(2, 5))
    assert conflicts == 3
    assert np.all(targets["pitch"][2:5, [0, 3]])


def test_notes_already_active_at_recording_start_have_no_new_attack_label():
    targets, _ = make_targets([note(-.5, .05)], 10, CONFIG, .1)
    assert not np.any(targets["onset"])
    np.testing.assert_array_equal(np.flatnonzero(targets["state"][:, 0] == 0), np.arange(4))


def test_partial_tail_does_not_create_future_feature_frames():
    targets, _ = make_targets([note(.095, .2)], 10, CONFIG, .105)
    assert len(targets["state"]) == 10 and targets["state"][-1, 0] == 0
    assert not np.any(targets["offset"])
    with pytest.raises(ValueError, match="beyond"):
        make_targets([], 11, CONFIG, .105)


def test_out_of_range_and_invalid_labels_are_rejected():
    with pytest.raises(ValueError, match="fretboard"):
        make_targets([note(.02, .05, fret=25)], 10, CONFIG, .1)
    with pytest.raises(ValueError, match="positive"):
        make_targets([note(.05, .02)], 10, CONFIG, .1)
    with pytest.raises(ValueError, match="finite"):
        make_targets([note(np.nan, .1)], 10, CONFIG, .1)
