import numpy as np
import pytest

from acoustic10.targets import delay_targets
from streaming.targets import make_targets


CONFIG = {"feature": {"sample_rate": 16000, "hop_length": 160},
          "model": {"max_fret": 24, "pitch_min": 40, "pitch_max": 88}}


def note(start, end, fret=0, string=0):
    tuning = [40, 45, 50, 55, 59, 64]
    return {"start": start, "end": end, "fret": fret, "string": string,
            "pitch": tuning[string] + fret}


def labels(notes=(), length=30, duration=.3):
    return make_targets(notes, length, CONFIG, duration)[0]


def test_six_frame_delay_keeps_exact_right_edge_grid_and_all_dtypes():
    original = labels([note(.1, .2)])
    shifted = delay_targets(original, 6, CONFIG)
    np.testing.assert_array_equal(np.flatnonzero(shifted["state"][:, 0] == 0),
                                  np.arange(15, 25))
    np.testing.assert_array_equal(np.flatnonzero(shifted["pitch"][:, 0]),
                                  np.arange(15, 25))
    assert np.argmax(shifted["onset"][:, 0]) == 15
    assert np.argmax(shifted["offset"][:, 0]) == 25
    assert np.all(shifted["state"][:6] == 25)
    for key in original:
        assert original[key].dtype == shifted[key].dtype
        np.testing.assert_array_equal(shifted[key][6:], original[key][:-6])


def test_zero_delay_is_exact_but_returns_independent_arrays():
    original = labels([note(.02, .2)])
    original["mask"] = np.linspace(0, 1, 30, dtype=np.float32)
    shifted = delay_targets(original, 0, CONFIG)
    for key in original:
        np.testing.assert_array_equal(shifted[key], original[key])
        assert not np.shares_memory(shifted[key], original[key])
    shifted["state"][0, 0] = -100
    assert original["state"][0, 0] != -100


def test_conflicts_and_mask_validity_move_with_the_labels():
    original = labels([note(.02, .2), note(.04, .1, fret=3)])
    original["mask"] = np.ones(30, dtype=np.float32)
    original["mask"][10] = .25
    original["mask"][12] = 0
    shifted = delay_targets(original, 6, CONFIG)
    np.testing.assert_array_equal(np.flatnonzero(shifted["state"][:, 0] == -100),
                                  np.arange(9, 15))
    assert np.all(shifted["mask"][:6] == 0)
    assert shifted["mask"][16] == .25 and shifted["mask"][18] == 0
    assert shifted["mask"].dtype == np.float32
    np.testing.assert_array_equal(shifted["pitch"][9:15, [0, 3]], np.ones((6, 2)))


def test_shift_full_record_before_crop_preserves_pre_crop_attack_history():
    original = labels([note(.04, .2)])
    whole = delay_targets(original, 6, CONFIG)
    crop = {key: value[8:18] for key, value in whole.items()}
    assert np.argmax(crop["onset"][:, 0]) == 1
    np.testing.assert_array_equal(np.flatnonzero(crop["state"][:, 0] == 0),
                                  np.arange(1, 10))
    incorrectly_shifted_crop = delay_targets(
        {key: value[8:18] for key, value in original.items()}, 6, CONFIG)
    assert not np.array_equal(crop["state"], incorrectly_shifted_crop["state"])
    assert not np.array_equal(crop["onset"], incorrectly_shifted_crop["onset"])


@pytest.mark.parametrize("release", [.3, .30000001, 1.])
def test_source_terminal_censoring_survives_delay_without_fake_release(release):
    original = labels([note(.2, release)])
    shifted = delay_targets(original, 6, CONFIG)
    assert not np.any(original["offset"]) and not np.any(shifted["offset"])
    assert shifted["state"][-1, 0] == 0
    assert shifted["pitch"][-1, 0] == 1


def test_partial_tail_is_clipped_without_wrapping_or_synthetic_offset():
    original = labels([note(.095, .2)], length=10, duration=.105)
    shifted = delay_targets(original, 6, CONFIG)
    assert np.all(shifted["state"] == 25)
    assert not np.any(shifted["onset"]) and not np.any(shifted["pitch"])
    assert not np.any(shifted["offset"])


def test_gaussian_support_translates_exactly_even_when_center_leaves_tail():
    original = labels([note(.05, .27)])
    shifted = delay_targets(original, 6, CONFIG)
    # The source release center is frame 26. Its delayed center is beyond
    # frame 29, but pre-peak support at 23 is valid array data and is retained.
    assert shifted["offset"][29, 0] == original["offset"][23, 0] > 0
    np.testing.assert_array_equal(shifted["offset"][6:], original["offset"][:-6])
    assert shifted["offset"][-1, 0] < 1


@pytest.mark.parametrize("delay", [30, 31, 10000])
def test_delay_at_least_record_length_has_only_leading_fill(delay):
    shifted = delay_targets(labels([note(.02, .2)]), delay, CONFIG)
    assert np.all(shifted["state"] == 25)
    assert not np.any(shifted["onset"]) and not np.any(shifted["offset"])
    assert not np.any(shifted["pitch"])


def test_empty_record_is_supported_and_keeps_dtypes():
    original = labels(length=0, duration=0.)
    shifted = delay_targets(original, 6, CONFIG)
    assert shifted["state"].shape == (0, 6)
    for key in original:
        assert original[key].dtype == shifted[key].dtype


@pytest.mark.parametrize("delay", [-1, 1.5, True, np.bool_(False), np.nan])
def test_invalid_delay_cannot_silently_change_timing(delay):
    with pytest.raises(ValueError, match="nonnegative integer"):
        delay_targets(labels(), delay, CONFIG)


@pytest.mark.parametrize("bad", ["mismatched_frames", "invalid_class", "nan_boundary",
                                  "invalid_probability", "mismatched_mask", "unknown"])
def test_invalid_targets_are_rejected_before_any_input_changes(bad):
    original = labels([note(.02, .1)])
    if bad == "mismatched_frames":
        original["pitch"] = original["pitch"][:-1]
    elif bad == "invalid_class":
        original["state"][2, 0] = 26
    elif bad == "nan_boundary":
        original["onset"][2, 0] = np.nan
    elif bad == "invalid_probability":
        original["offset"][2, 0] = 2
    elif bad == "mismatched_mask":
        original["mask"] = np.ones(29, dtype=np.float32)
    else:
        original["source_time"] = np.arange(30)
    snapshots = {key: value.copy() for key, value in original.items()}
    with pytest.raises(ValueError):
        delay_targets(original, 6, CONFIG)
    for key in original:
        np.testing.assert_array_equal(original[key], snapshots[key])
