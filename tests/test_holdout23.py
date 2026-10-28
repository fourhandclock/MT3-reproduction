import copy
import json
import pytest

from holdout23.data import ROOT, freeze, load_test_inputs, read, validate_config, validate_partition
from holdout23.workflow import contrasts, locked_selection, run


def manifest():
    return {'train': [{'id': '00_Rock1-x'}, {'id': '01_Funk1-y'}],
            'validation': [{'id': '00_Rock2-x'}, {'id': '02_Funk2-y'}],
            'test': [{'id': f'{i // 2:02d}_Rock3-{i}-C_comp'} for i in range(12)]}


def test_progression3_identity_partition_reports_player_overlap():
    value = validate_partition(manifest())
    assert value['train_test_shared_players'] == ['00', '01']
    assert value['validation_test_shared_players'] == ['00', '02']


@pytest.mark.parametrize('kind', ['count', 'duplicate', 'progression', 'overlap'])
def test_bad_test_partition_fails(kind):
    value = manifest()
    if kind == 'count': value['test'].pop()
    if kind == 'duplicate': value['test'][1] = value['test'][0].copy()
    if kind == 'progression': value['test'][0]['id'] = '00_Rock2-x'
    if kind == 'overlap': value['train'].append(value['test'][0].copy())
    with pytest.raises(ValueError): validate_partition(value)


@pytest.mark.parametrize('kind', ['selection', 'checkpoint', 'delay', 'mask', 'decoder', 'budget', 'calibrate', 'promotion'])
def test_locked_protocol_rejects_model_or_test_tuning(kind):
    config = copy.deepcopy(read(ROOT / 'configs/holdout23.json'))
    if kind == 'selection': config['locked_selected_arm'] = 'balanced_ce650'
    if kind == 'checkpoint': config['models']['uniform_ce650']['sha256'] = '0' * 64
    if kind == 'delay': config['common_target_delay_frames'] = 12
    if kind == 'mask': config['common_leading_mask_frames'] = 0
    if kind == 'decoder': config['decoder_parameters']['activity_threshold'] = .3
    if kind == 'budget': config['new_optimizer_update_cap'] = 1
    if kind == 'calibrate': config['calibration_search_allowed'] = True
    if kind == 'promotion': config['production_promotion'] = True
    with pytest.raises(ValueError): validate_config(config)


def test_source_lock_is_written_before_audio_or_annotation_loader_calls(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Test audio/annotation content read before lock')
    monkeypatch.setattr('holdout23.data.load_audio', forbidden)
    monkeypatch.setattr('holdout23.data.reference_notes', forbidden)
    config = read(ROOT / 'configs/holdout23.json')
    protocol = freeze(config, tmp_path)
    assert protocol['locked_selected_arm'] == 'uniform_ce650'
    assert protocol['lock_written_before_test_input_reads'] is True
    assert protocol['test_audio_or_annotations_read_before_lock'] is False
    assert protocol['new_optimizer_updates'] == protocol['normalization_updates'] == 0
    assert len(protocol['test_ids']) == 12
    assert (tmp_path / 'selection.json').read_bytes() == (ROOT / 'outputs/balance22/selection.json').read_bytes()
    assert protocol['test_scored_replays_planned'] == 48 and protocol['test_audio_warmups_planned'] == 4


def test_audio_cannot_be_loaded_without_a_protocol_lock(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr('holdout23.data.load_audio', lambda *args: calls.append(args))
    with pytest.raises(FileNotFoundError):
        load_test_inputs(read(ROOT / 'configs/holdout23.json'), tmp_path)
    assert calls == []


def test_existing_protocol_rejects_repeat_evaluation_before_test_loading(tmp_path):
    (tmp_path / 'protocol.json').write_text('{}')
    with pytest.raises(ValueError, match='already locked'):
        freeze(read(ROOT / 'configs/holdout23.json'), tmp_path)


def test_another_test_model_outperforming_uniform_does_not_change_selection():
    keys = ('exact_state_f1', 'onset_f1', 'onset_offset_f1', 'tab_onset_f1', 'frame_f1',
            'best_active_fret_accuracy', 'best_active_equal_position_accuracy')
    scores = {'modes': {mode: {key: .2 for key in keys} for mode in
              ['parent500', 'original650', 'uniform_ce650', 'balanced_ce650']}}
    scores['modes']['uniform_ce650'] = {key: .1 for key in keys}
    scores['modes']['balanced_ce650'] = {key: .9 for key in keys}
    result = contrasts(scores)
    assert result['locked_selected_arm'] == 'uniform_ce650'
    assert result['locked_selected_vs_original']['exact_state_f1'] == pytest.approx(-.1)
    choice = locked_selection(read(ROOT / 'configs/holdout23.json'))
    assert choice['selected_arm'] == 'uniform_ce650' and not choice['model_selection_performed']
    assert not choice['selected_from_test'] and not choice['production_promoted']


def test_run_locks_before_calling_the_test_input_loader(tmp_path, monkeypatch):
    order = []
    def fake_freeze(config, output):
        order.append('lock')
        (output / 'protocol.json').write_text('{}')
        return {}
    def fake_load(config, output):
        order.append('test_inputs')
        assert (output / 'protocol.json').exists()
        raise RuntimeError('stop before neural inference')
    monkeypatch.setattr('holdout23.workflow.freeze', fake_freeze)
    monkeypatch.setattr('holdout23.workflow.load_test_inputs', fake_load)
    with pytest.raises(RuntimeError, match='stop before neural'):
        run({}, tmp_path, None)
    assert order == ['lock', 'test_inputs']
