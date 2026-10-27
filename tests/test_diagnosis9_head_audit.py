import numpy as np
import pytest

from diagnosis9.context_audit import state_to_pitch
from diagnosis9.head_audit import binary_metric, exact_state_metric, pooled, raw_run_notes, runs


def test_exact_state_counts_wrong_fret_and_ignores_conflict():
    target = np.array([[0, 25, -100], [1, 25, 2]])
    estimate = np.array([[0, 3, 5], [2, 25, 2]])
    result = exact_state_metric(target, estimate, 25)
    assert result['true_positives'] == 2
    assert result['reference_count'] == 3
    assert result['predicted_count'] == 4
    assert result['precision'] == .5
    assert result['recall'] == pytest.approx(2 / 3)


def test_pooling_uses_counts_instead_of_mean_f1():
    first = binary_metric(np.array([True]), np.array([True]))
    second = binary_metric(np.ones(9, dtype=bool), np.zeros(9, dtype=bool))
    result = pooled([first, second])
    assert result['precision'] == 1
    assert result['recall'] == .1
    assert result['f1'] == pytest.approx(2 / 11)


def test_runs_separate_changes_in_fret_and_ignore_inactivity():
    states = np.array([[2, 25], [2, 25], [3, 25], [25, 25], [3, 25]])
    assert runs(states, 25) == [2, 1, 1]
    assert runs(np.empty((0, 6), dtype=int), 25) == []


def test_direct_runs_use_right_edge_grid_and_backdate():
    states = np.full((5, 6), 25)
    states[1:3, 0] = 2
    notes = raw_run_notes(states, 25, .01, .05, backdate=.01)
    assert len(notes) == 1
    assert notes[0]['start'] == pytest.approx(.01)
    assert notes[0]['end'] == pytest.approx(.03)
    assert notes[0]['pitch'] == 42
    assert raw_run_notes(np.empty((0, 6), dtype=int), 25, .01, 0) == []


def test_pitch_union_counts_unisons_once_and_filters_inactive():
    config = {'model': {'max_fret': 24, 'pitch_min': 40, 'pitch_max': 88}}
    states = np.full((2, 6), 25)
    states[0, 0] = 5
    states[0, 1] = 0
    result = state_to_pitch(states, config)
    assert result.sum() == 1
    assert result[0, 5]
    assert not result[1].any()
