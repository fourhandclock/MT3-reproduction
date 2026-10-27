import numpy as np
import pytest

from mixtures.audio import make_accompaniment, mix


def _backing(seed=42):
    beats = [{"time": i * .5, "value": {"position": i % 4 + 1}} for i in range(8)]
    chords = [{"time": 0, "duration": 2, "value": "C:maj"},
              {"time": 2, "duration": 2, "value": "A:min7/3"}]
    return make_accompaniment(4, 16000, beats, chords, seed)


def test_renderer_is_reproducible_finite_and_exact_length():
    audio, metadata = _backing()
    repeat, repeat_metadata = _backing()
    alternate, _ = _backing(43)
    assert audio.dtype == np.float32 and audio.shape == (64000,)
    assert np.isfinite(audio).all() and metadata["rms"] > 0
    np.testing.assert_array_equal(audio, repeat)
    assert metadata == repeat_metadata
    assert not np.array_equal(audio, alternate)
    assert {event["instrument"] for event in metadata["events"]} == {"bass", "pad", "keyboard", "kick", "snare", "hihat"}
    assert all(0 <= event["start"] < event["end"] <= 4 for event in metadata["events"])


def test_supplied_chords_control_pitches_and_silence_is_respected():
    audio, metadata = make_accompaniment(2, 16000, [0, .5, 1, 1.5],
        [{"start": 0, "end": 1, "label": "D:min"}, {"start": 1, "end": 2, "label": "N"}], 1)
    assert np.any(audio)
    keyboard = [event for event in metadata["events"] if event["instrument"] == "keyboard"]
    assert {event["pitch"] for event in keyboard} == {62, 65, 69}
    assert all(event["end"] <= 1 for event in metadata["events"] if "pitch" in event)


@pytest.mark.parametrize("ratio_db", [6, 0, -6])
def test_mixture_ratio_and_target_alignment_are_preserved(ratio_db):
    accompaniment, _ = _backing()
    time = np.arange(len(accompaniment)) / 16000
    guitar = (.85 * np.sin(2*np.pi*220*time)).astype(np.float32)
    guitar[1234] = 1.3
    result = mix(guitar, accompaniment, ratio_db)
    metadata = result["metadata"]
    assert abs(metadata["achieved_guitar_to_accompaniment_db"] - ratio_db) < 1e-5
    assert metadata["mixture_peak"] <= .980001
    assert metadata["common_peak_gain"] < 1
    np.testing.assert_array_equal(result["target"], (guitar.astype(np.float64)*metadata["guitar_gain"]).astype(np.float32))
    np.testing.assert_array_equal(result["mixture"], result["target"] + result["accompaniment"])
    assert all(result[key].dtype == np.float32 and result[key].shape == guitar.shape
               for key in ["mixture", "target", "accompaniment"])


def test_mixing_validates_sources_and_undefined_ratios():
    with pytest.raises(ValueError, match="equal-length"):
        mix(np.ones(4), np.ones(3), 0)
    with pytest.raises(ValueError, match="nonzero RMS"):
        mix(np.ones(4), np.zeros(4), 0)
    with pytest.raises(ValueError, match="finite"):
        mix(np.array([1, np.nan]), np.ones(2), 0)


def test_schedule_validation_and_unsupported_labels_are_explicit():
    with pytest.raises(ValueError, match="distinct"):
        make_accompaniment(1, 16000, [0, 0], [], 1)
    with pytest.raises(ValueError, match="non-overlapping"):
        make_accompaniment(1, 16000, [], [{"start": 0, "end": 1, "label": "C:maj"},
                                        {"start": .5, "end": 1, "label": "D:maj"}], 1)
    _, metadata = make_accompaniment(1, 16000, [0, .5], [{"start": 0, "end": 1, "label": "C:unknown"}], 1)
    assert metadata["unsupported_chord_labels"] == ["C:unknown"]
    assert all("pitch" not in event for event in metadata["events"])
