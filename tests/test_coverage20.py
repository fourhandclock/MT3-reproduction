import copy

import numpy as np
import pytest

from coverage20.analysis import bin_index, class_rows, exposure, frame_counts, grouped_counts, rates
from coverage20.audit import independent_exposure
from coverage20.data import BINS
from coverage20.workflow import note_rows, verify_saved_frame_totals


def record(length=120, name='train'):
    return {'id': name, 'targets': {'state': np.full((length, 6), 25, np.int16),
            'onset': np.zeros((length, 6), np.float16), 'offset': np.zeros((length, 6), np.float16),
            'pitch': np.zeros((length, 49), np.uint8)}}


def crop(index=0, name='train', start=0, length=100, epoch=1):
    return {'record_index': index, 'id': name, 'start_frame': start, 'frames': length, 'epoch': epoch}


@pytest.mark.parametrize('value,expected', [(0, 0), (1, 1), (99, 1), (100, 2), (999, 2), (1000, 3)])
def test_fixed_bin_boundaries(value, expected):
    assert bin_index(value) == expected


def test_negative_exposure_is_rejected():
    with pytest.raises(ValueError):
        bin_index(-1)


def test_delay_full_recording_masks_repeats_and_source_identity():
    item = record(12)
    item['targets']['state'][:, 0] = 2
    item['targets']['state'][2, 0] = -100
    item['targets']['state'][10, 1] = 4
    rows = [crop(start=0, length=8), crop(start=2, length=8)]
    result = exposure([item], rows, delay=2, leading=3, edge=1)
    assert result['available'][0][2] == 11
    assert result['available'][1][4] == 1
    assert result['presentations'][0][2] == 8
    assert result['unique'][0][2] == 5
    assert result['presentations'][1][4] == 0
    again = exposure([item], rows * 2, delay=2, leading=3, edge=1)
    assert again['presentations'][0][2] == 16
    assert again['unique'] == result['unique']
    second = copy.deepcopy(item)
    second['id'] = 'other'
    other_rows = [{**row, 'record_index': 1, 'id': 'other'} for row in rows]
    both = exposure([item, second], rows + other_rows, delay=2, leading=3, edge=1)
    assert both['unique'][0][2] == 10


def test_combined_history_unions_unique_frames_instead_of_adding_them():
    item = record()
    item['targets']['state'][:, 2] = 7
    first = [crop(start=0)]
    second = [crop(start=10, epoch=6)]
    left, right = exposure([item], first), exposure([item], second)
    full = exposure([item], first + second)
    assert full['presentations'][2][7] == left['presentations'][2][7] + right['presentations'][2][7]
    assert full['unique'][2][7] < left['unique'][2][7] + right['unique'][2][7]
    repeated, unique = independent_exposure([item], first + second)
    assert repeated == full['presentations']
    assert unique == full['unique']


@pytest.mark.parametrize('mutation', ['id', 'index', 'negative', 'past_end', 'zero'])
def test_bad_crop_descriptor_fails(mutation):
    row = crop()
    if mutation == 'id': row['id'] = 'wrong'
    if mutation == 'index': row['record_index'] = -1
    if mutation == 'negative': row['start_frame'] = -1
    if mutation == 'past_end': row['start_frame'] = 21
    if mutation == 'zero': row['frames'] = 0
    with pytest.raises(ValueError):
        exposure([record()], [row])


def test_fully_masked_crop_has_no_supervision():
    item = record()
    item['targets']['state'][:, 0] = 0
    result = exposure([item], [crop(length=40)])
    assert np.asarray(result['presentations']).sum() == 0
    assert result['per_epoch_active_presentations'] == {'1': 0}


def test_pitch_on_other_string_does_not_make_exact_position_seen():
    item = record()
    item['targets']['state'][:, 1] = 0  # A2, also string0 fret5.
    rows = class_rows(exposure([item], [crop()]))
    low_a = rows[5]
    assert low_a['pitch'] == 45 and low_a['presentations'] == 0
    assert low_a['pitch_coverage'] == 'pitch_seen_elsewhere'
    assert low_a['other_string_presentations'] == rows[25]['presentations'] > 0
    assert rows[25]['pitch_coverage'] == 'position_seen'
    assert rows[-1]['pitch_coverage'] == 'pitch_unseen'


def example_frames():
    item = record(6)
    item['targets']['state'][:, 0] = 3
    item['targets']['state'][2, 2] = -100
    predicted = np.full((8, 6), 25, int)
    predicted[2:, 0] = [3, 3, 25, 15, 4, 3]
    predicted[2, 1] = 5  # Masked first source frame.
    predicted[4, 1] = 4  # One false activity on a truly inactive string.
    predicted[4, 2] = 2  # Conflict is excluded.
    probabilities = np.eye(26, dtype=np.float32)[predicted]
    return item, probabilities


def test_frame_alignment_masks_and_reference_error_partition():
    item, probabilities = example_frames()
    value = frame_counts(probabilities, item['targets'], count=6, delay=2, leading=1)['argmax']
    counts = np.asarray(value['class_counts'])
    assert counts[3, :5].tolist() == [5, 2, 1, 1, 1]
    assert counts[:, 5].sum() == 1 and counts[25 + 4, 5] == 1
    assert value['valid_string_frames'] == 29
    assert value['reference_inactive_frames'] == 24


def test_decoder_activity_is_separate_from_argmax():
    item = record(1)
    item['targets']['state'][0, 0] = 3
    probability = np.zeros((1, 6, 26), np.float32)
    probability[:, :, 25] = 1
    probability[0, 0, :25] = .55 / 24
    probability[0, 0, 3] = .2
    probability[0, 0, 25] = .25
    policies = frame_counts(probability, item['targets'], count=1, delay=0, leading=0)
    assert policies['argmax']['class_counts'][3][2] == 1
    assert policies['decoder_activity']['class_counts'][3][1] == 1


@pytest.mark.parametrize('change', ['short', 'nan', 'negative', 'unnormalized'])
def test_malformed_probability_recordings_fail(change):
    item, probabilities = example_frames()
    if change == 'short': probabilities = probabilities[:4]
    if change == 'nan': probabilities[3, 0, 0] = np.nan
    if change == 'negative': probabilities[3, 0, 0] = -.1
    if change == 'unnormalized': probabilities[3, 0] = .1
    with pytest.raises(ValueError):
        frame_counts(probabilities, item['targets'], count=6, delay=2, leading=1)


def test_false_activity_group_does_not_invent_gt_active_denominator():
    item, probabilities = example_frames()
    counts = frame_counts(probabilities, item['targets'], count=6, delay=2, leading=1)['argmax']['class_counts']
    labels = [{'bin': '1000+' if i == 3 else 'unseen'} for i in range(150)]
    bins = grouped_counts(counts, labels, 'bin', BINS)
    assert bins['1000+']['reference_active'] == 5
    assert bins['1000+']['error_rate_on_reference_active'] == .6
    assert bins['unseen']['reference_active'] == 0
    assert bins['unseen']['error_rate_on_reference_active'] is None
    assert bins['unseen']['false_active_on_reference_inactive'] == 1
    assert bins['1-99']['note_miss_rate'] is None


def test_note_matching_uses_gt_position_and_keeps_early_onsets(monkeypatch):
    ref = [{'start': .02, 'end': .5, 'pitch': 45, 'string': 0, 'fret': 5},
           {'start': 1.2, 'end': 1.5, 'pitch': 45, 'string': 0, 'fret': 5}]
    pred = [{'start': .03, 'end': .5, 'pitch': 45, 'string': 1, 'fret': 0}]
    score = {'score_seconds': 1., 'onset_tolerance_seconds': .05, 'offset_tolerance_seconds': .05, 'offset_ratio': .2}
    monkeypatch.setattr('coverage20.workflow.reference_notes', lambda path: ref)
    row = {'id': 'dev', 'mode': 'parent250', 'reference': ref, 'notes': pred,
           'pitch_metrics': {'reference': ref[:1], 'predicted': pred, 'onset_matches': [[0, 0]], 'onset': {'true_positives': 1}}}
    classes = class_rows(exposure([record()], []))
    result = note_rows(row, classes, score, 'unused')
    assert len(result) == 1 and result[0]['start'] == .02
    assert result[0]['official_matched'] is True
    assert result[0]['official_matched_same_string'] is False
    assert result[0]['bin'] == 'unseen'
    row['pitch_metrics']['onset_matches'] = []
    with pytest.raises(ValueError, match='one-to-one'):
        note_rows(row, classes, score, 'unused')


def test_cached_frame_totals_are_a_required_external_cross_check():
    item, probabilities = example_frames()
    value = frame_counts(probabilities, item['targets'], count=6, delay=2, leading=1)['argmax']
    with pytest.raises(ValueError):
        verify_saved_frame_totals(value, {'counts': {key: 0 for key in [
            'reference_active_frames', 'correct_fret', 'predicted_inactive', 'wrong_octave_same_string',
            'other_wrong_fret', 'active_on_reference_inactive', 'reference_inactive_frames', 'valid_string_frames']}})


def test_zero_support_rates_are_null_instead_of_zero_error():
    result = rates([0] * 9)
    assert result['error_rate_on_reference_active'] is None
    assert result['conditional_fret_accuracy'] is None
    assert result['note_miss_rate'] is None
