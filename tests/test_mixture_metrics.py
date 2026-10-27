"""Independent analytical and alignment cases for diagnostic waveform scores."""
import numpy as np
import pytest

from mixtures.metrics import separation_metrics


def _orthogonal_sources():
    # Integer numbers of cycles make these mean-zero tones mutually orthogonal.
    time = np.arange(16000, dtype=np.float64) / 16000
    return np.sin(2*np.pi*220*time), np.sin(2*np.pi*733*time)


def test_si_sdr_ignores_gain_but_direct_sdr_penalizes_it():
    target, interference = _orthogonal_sources()
    estimate = target + .2*interference
    score = separation_metrics(target, estimate, 16000)
    scaled_score = separation_metrics(target, 2*estimate, 16000)
    # Known error/source power is .2**2, independent of the overall SI gain.
    expected_si = 20*np.log10(1/.2)
    assert score['si_sdr_db'] == pytest.approx(expected_si, abs=1e-9)
    assert scaled_score['si_sdr_db'] == pytest.approx(expected_si, abs=1e-9)
    assert score['sdr_db'] == pytest.approx(expected_si, abs=1e-9)
    # After doubling, direct error is target + .4*interference: power 1+.4**2.
    assert scaled_score['sdr_db'] == pytest.approx(-10*np.log10(1+.4**2), abs=1e-9)
    assert scaled_score['sdr_db'] < score['sdr_db'] - 10


def test_known_orthogonal_noise_level_has_expected_scores_and_correlation():
    target, interference = _orthogonal_sources()
    score = separation_metrics(target, target + .1*interference, 16000)
    assert score['si_sdr_db'] == pytest.approx(20, abs=1e-9)
    assert score['sdr_db'] == pytest.approx(20, abs=1e-9)
    assert score['normalized_correlation'] == pytest.approx(1/np.sqrt(1+.1**2), abs=1e-12)
    assert score['gain_projection'] == pytest.approx(1, abs=1e-12)
    assert score['alignment_correction_applied'] is False


@pytest.mark.parametrize('shift', [-80, 80])
def test_known_lag_is_diagnostic_and_never_changes_inputs_or_scores(shift):
    target = np.zeros(4096)
    target[500:3000] = np.random.default_rng(281).standard_normal(2500)
    estimate = np.zeros_like(target)
    if shift > 0:
        estimate[shift:] = target[:-shift]
    else:
        estimate[:shift] = target[-shift:]
    original_target, original_estimate = target.copy(), estimate.copy()
    score = separation_metrics(target, estimate, 16000)
    assert score['diagnostic_lag_ms'] == pytest.approx(1000*shift/16000)
    # Delay remains in the scoring inputs, so direct SDR is poor despite the
    # identical time-shifted signal that the diagnostic identifies.
    assert score['sdr_db'] < 0
    assert abs(score['normalized_correlation']) < .1
    assert score['alignment_correction_applied'] is False
    np.testing.assert_array_equal(target, original_target)
    np.testing.assert_array_equal(estimate, original_estimate)


@pytest.mark.parametrize('amplitude', [0, 1e-12])
def test_silent_or_near_zero_reference_has_no_defined_scores(amplitude):
    target = amplitude*np.linspace(-1, 1, 1000)
    score = separation_metrics(target, np.ones_like(target), 16000)
    assert score['si_sdr_db'] is None
    assert score['sdr_db'] is None
    assert score['normalized_correlation'] is None
    assert score['diagnostic_lag_ms'] is None


@pytest.mark.parametrize('amplitude', [0, 1e-12])
def test_silent_or_near_zero_estimate_does_not_get_a_defined_si_score(amplitude):
    target, _ = _orthogonal_sources()
    score = separation_metrics(target, amplitude*target, 16000)
    # A zero-energy estimate has an undefined 0/0 SI projection ratio and
    # correlation. Direct error against an audible target remains meaningful.
    assert score['si_sdr_db'] is None
    assert score['normalized_correlation'] is None
    assert score['diagnostic_lag_ms'] is None
    assert score['sdr_db'] == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize('target,estimate', [
    (np.ones(10), np.ones(9)),
    (np.ones((2, 10)), np.ones((2, 10))),
    (np.array([0, np.nan]), np.ones(2)),
    (np.ones(2), np.array([0, np.inf])),
])
def test_invalid_waveform_shapes_and_nonfinite_values_are_rejected(target, estimate):
    with pytest.raises(ValueError):
        separation_metrics(target, estimate, 16000)
