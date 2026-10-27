import copy
import numpy as np
import pytest

from prior21.analysis import aligned, confusion, describe, expected_confusion, priors
from prior21.data import ROOT, read, validate_config
from prior21.workflow import run


def example():
    reference = np.zeros((6, 25), np.int64)
    reference[0, :2] = [90, 10]
    training = np.zeros((6, 25), np.int64)
    training[0, :2] = [80, 20]
    return reference, training


def test_training_prior_ties_use_lowest_fret_and_empty_strings_are_explicit():
    counts = np.zeros((6, 25), np.int64)
    counts[0, [2, 4]] = [30, 30]
    result = priors(counts)
    assert result['modal_frets'] == [2, None, None, None, None, None]
    assert result['distribution'][0][2] == .5
    assert result['top1'][0][2] == 1
    assert sum(result['distribution'][1]) == 0


@pytest.mark.parametrize('change', ['float', 'negative', 'shape', 'nan'])
def test_malformed_training_counts_fail(change):
    counts = np.zeros((6, 25), np.int64)
    if change == 'float': counts = counts.astype(float)
    if change == 'negative': counts[0, 0] = -1
    if change == 'shape': counts = counts[:, :24]
    if change == 'nan': counts = counts.astype(float); counts[0, 0] = np.nan
    with pytest.raises(ValueError):
        priors(counts)


def test_distribution_is_analytic_expected_mass_and_keeps_integer_gt_support():
    ref, counts = example()
    prior = priors(counts)
    matrix = expected_confusion(ref, prior['distribution'])
    assert matrix[0, 0, 0] == 72 and matrix[0, 1, 1] == 2
    metrics = describe(matrix, ref, prior['modal_frets'])
    assert metrics['reference_active_frames'] == 100
    assert metrics['accuracy_frame_weighted'] == .74
    assert metrics['accuracy_equal_supported_position'] == .5
    assert metrics['supported_positions'] == 2 and metrics['supported_strings'] == 1
    assert metrics['classes'][2]['accuracy'] is None
    assert metrics['accuracy_minus_independent_marginals'] == pytest.approx(0)


def test_majority_accuracy_does_not_imply_information_beyond_frequency():
    ref, counts = example()
    prior = priors(counts)
    matrix = expected_confusion(ref, prior['top1'])
    metrics = describe(matrix, ref, prior['modal_frets'])
    assert metrics['accuracy_frame_weighted'] == .9
    assert metrics['accuracy_equal_supported_position'] == .5
    assert metrics['pooled_per_string_top_prediction_share'] == 1
    assert metrics['accuracy_minus_independent_marginals'] == 0


def test_perfect_alignment_exceeds_its_own_independence_expectation():
    ref, counts = example()
    matrix = np.zeros((6, 25, 25), np.int64)
    matrix[0, 0, 0], matrix[0, 1, 1] = 90, 10
    metrics = describe(matrix, ref, priors(counts)['modal_frets'])
    assert metrics['accuracy_frame_weighted'] == 1
    assert metrics['accuracy_equal_supported_position'] == 1
    assert metrics['independent_marginal_expected_accuracy'] == pytest.approx(.82)
    assert metrics['accuracy_minus_independent_marginals'] == pytest.approx(.18)
    assert metrics['fraction_on_training_modal'] == .9


def test_uniform_prior_is_four_percent_for_both_weightings():
    ref, counts = example()
    metrics = describe(expected_confusion(ref, np.full((6, 25), .04)), ref, priors(counts)['modal_frets'])
    assert metrics['accuracy_frame_weighted'] == pytest.approx(.04)
    assert metrics['accuracy_equal_supported_position'] == pytest.approx(.04)


def test_equal_strings_does_not_reweight_by_duration():
    ref = np.zeros((6, 25), np.int64)
    ref[0, 0], ref[1, 0] = 90, 10
    matrix = np.zeros((6, 25, 25), np.int64)
    matrix[0, 0, 0], matrix[1, 0, 1] = 90, 10
    metrics = describe(matrix, ref, [0] * 6)
    assert metrics['accuracy_frame_weighted'] == .9
    assert metrics['accuracy_equal_supported_string'] == .5


def test_gt_active_string_without_training_prior_fails_instead_of_dropping_frames():
    ref, _ = example()
    with pytest.raises(ValueError, match='GT-supported'):
        expected_confusion(ref, np.zeros((6, 25)))


def test_source_alignment_conflicts_and_inactive_model_decision_are_separate():
    state = np.full((6, 6), 25, np.int16)
    state[:, 0] = 3
    state[2, 0] = -100
    probability = np.zeros((8, 6, 26), np.float32)
    probability[:, :, 25] = 1
    probability[2:, 0, :25] = .15 / 24
    probability[2:, 0, 3] = .05
    probability[2:, 0, 25] = .8
    strings, truth, best = aligned(probability, state, count=6, delay=2, leading=1)
    assert len(truth) == 4
    assert np.all(strings == 0) and np.all(truth == 3) and np.all(best == 3)
    assert np.all(probability[2:, 0].argmax(-1) == 25)


@pytest.mark.parametrize('change', ['short', 'nan', 'negative', 'unnormalized'])
def test_malformed_saved_probability_fails(change):
    state = np.full((6, 6), 25, np.int16)
    p = np.eye(26, dtype=np.float32)[np.full((8, 6), 25, int)]
    if change == 'short': p = p[:3]
    if change == 'nan': p[3, 0, 0] = np.nan
    if change == 'negative': p[3, 0, 0] = -.1
    if change == 'unnormalized': p[3, 0] = .1
    with pytest.raises(ValueError):
        aligned(p, state, count=6, delay=2, leading=1)


@pytest.mark.parametrize('change', ['string', 'fret', 'length', 'float'])
def test_invalid_observation_vectors_fail(change):
    s, t, p = np.array([0]), np.array([1]), np.array([2])
    if change == 'string': s[0] = 6
    if change == 'fret': t[0] = 25
    if change == 'length': p = np.array([], dtype=int)
    if change == 'float': t = t.astype(float)
    with pytest.raises(ValueError):
        confusion(s, t, p)


def test_confusion_preserves_string_identity_and_exact_counts():
    matrix = confusion(np.array([0, 0, 1]), np.array([2, 2, 2]), np.array([2, 3, 2]))
    assert matrix.sum() == 3
    assert matrix[0, 2, 2] == 1 and matrix[0, 2, 3] == 1 and matrix[1, 2, 2] == 1


def test_empty_support_has_null_metrics():
    metrics = describe(np.zeros((6, 25, 25)), np.zeros((6, 25), int), [None] * 6)
    assert metrics['accuracy_frame_weighted'] is None
    assert metrics['accuracy_equal_supported_position'] is None
    assert metrics['accuracy_equal_supported_string'] is None


@pytest.mark.parametrize('change', ['source_frames', 'prior_count_sources', 'primary_prior', 'unknown'])
def test_fixed_protocol_rejects_alignment_or_baseline_search(change):
    config = copy.deepcopy(read(ROOT / 'configs/prior21.json'))
    if change == 'source_frames': config[change] = 900
    if change == 'prior_count_sources': config[change].reverse()
    if change == 'primary_prior': config[change] = 'unique_top1'
    if change == 'unknown': config['calibrate'] = True
    with pytest.raises(ValueError):
        validate_config(config)


def test_verified_output_cannot_be_overwritten(tmp_path):
    receipt = tmp_path / 'verification.json'
    receipt.write_bytes(b'preserved')
    with pytest.raises(ValueError, match='already verified'):
        run({}, tmp_path)
    assert receipt.read_bytes() == b'preserved'
