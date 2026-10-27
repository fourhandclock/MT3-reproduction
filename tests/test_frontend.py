from pathlib import Path

import numpy as np
import pytest

from baseline.frontend import compute_logmel, tensorflow_mel_weights

FIXTURE = Path(__file__).parent / 'fixtures/mt3_frontend.npz'


@pytest.mark.parametrize('name', ['silence', 'noise', 'impulse', 'guitar'])
def test_frontend_matches_outputs_from_official_tensorflow_code(name):
    with np.load(FIXTURE) as fixture:
        actual = compute_logmel(fixture[name + '_audio'])
        expected = fixture[name + '_logmel']
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-5)


def test_gain_changes_log_magnitude_without_squaring_power():
    with np.load(FIXTURE) as fixture:
        audio = fixture['noise_audio']
    first, doubled = compute_logmel(audio), compute_logmel(audio * 2)
    nonzero_bands = tensorflow_mel_weights().sum(axis=0) > 0
    np.testing.assert_allclose((doubled - first)[:, nonzero_bands], np.log(2), atol=1e-4)
