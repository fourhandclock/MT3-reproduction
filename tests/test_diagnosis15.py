import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from diagnosis15.analysis import (CATEGORIES, add_note_evidence, associate, frame_diagnosis,
                                  pitch_union, source_arrays, taxonomy)
from diagnosis15.loss import loss_with_state_pitch, state_pitch_loss, state_pitch_probabilities
from diagnosis15.workflow import recommend, validate_config


CONFIG = json.loads(Path('configs/diagnosis15.json').read_text())
SCORE = {'score_seconds': 1., 'onset_tolerance_seconds': .05, 'offset_tolerance_seconds': .05,
         'offset_ratio': .2, 'frame_rate': 100}


def note(start, pitch=40, string=0, end=None):
    return {'start': start, 'end': start + .1 if end is None else end, 'pitch': pitch,
            'string': string, 'fret': pitch - [40,45,50,55,59,64][string]}


def test_taxonomy_exclusive_one_to_one_partition_covers_all_notes():
    reference = [note(.05), note(.2, 45, 1), note(.4, 50, 2), note(.6, 52, 2), note(.8, 54, 2), note(.95, 60, 3)]
    predicted = [note(.05), note(.2, 45, 0), note(.52, 50, 2), note(.6, 40), note(.8, 57, 3), note(.05, 70, 5)]
    result = taxonomy(reference, predicted, SCORE, CONFIG)
    assert [r['category'] for r in result['reference_rows']] == CATEGORIES
    assert result['reference_count'] == sum(result['reference_counts'].values()) == 6
    paired = [r['prediction_index'] for r in result['reference_rows'] if r['prediction_index'] is not None]
    assert len(paired) == len(set(paired)) == 5
    assert result['extra_predictions'] == 1
    assert len(paired) + result['extra_predictions'] == result['prediction_count']


def test_maximum_cardinality_pairing_avoids_greedy_trap():
    result = associate([note(0.), note(.15)], [note(.1), note(.3)], [0,1], [0,1], 'nearby_same_pitch', .2)
    assert result == [(0,0),(1,1)]


def test_category_order_prioritizes_same_pitch_timing_before_octave_association():
    result = taxonomy([note(.2), note(.35,52)], [note(.35)], SCORE, CONFIG)
    assert result['reference_rows'][0]['category'] == 'nearby_same_pitch'
    assert result['reference_rows'][1]['category'] == 'missed'


@pytest.mark.parametrize('reference,predicted', [([],[]),([note(.2)],[]),([],[note(.2)])])
def test_empty_reference_or_prediction_partition(reference, predicted):
    result = taxonomy(reference, predicted, SCORE, CONFIG)
    assert result['reference_count'] == len(reference)
    assert result['prediction_count'] == len(predicted)
    assert sum(result['reference_counts'].values()) == len(reference)


def test_score_prefix_excludes_context_onsets_without_mutating_notes():
    reference, predicted = [note(.95), note(1.01)], [note(.95), note(1.01)]
    before = copy.deepcopy((reference,predicted))
    result = taxonomy(reference,predicted,SCORE,CONFIG)
    assert result['matched_onsets'] == 1 and result['reference_count'] == result['prediction_count'] == 1
    assert before == (reference,predicted)


def test_repetition_and_fragment_overlap_tags_are_separate_from_note_partition():
    reference = [note(.1,end=.4),note(.4,end=.7)]
    predicted = [note(.11,end=.22),note(.24,end=.39),note(.41,end=.69),note(0.,end=1e-9)]
    result = taxonomy(reference,predicted,SCORE,CONFIG)
    assert result['repeated_reference_count'] == result['repeated_reference_matched'] == 1
    assert result['reference_notes_with_multiple_overlaps'] == 1
    assert result['reference_rows'][0]['overlapping_same_string_pitch_predictions'] == [0,1]


def test_state_pitch_union_collapses_unisons_and_ignores_inactive():
    states = np.full((2,6),25,np.int16)
    states[0,0],states[0,1] = 5,0  # both MIDI45
    union = pitch_union(states)
    assert union.shape == (2,49) and union.sum() == 1 and union[0,5]


def test_aligned_source_grid_applies_known_delay_and_shared_leading_mask_once():
    cfg = {**CONFIG,'source_frames':4,'source_delay_frames':1,'leading_mask_frames':1}
    targets = {'state':np.full((4,6),25,np.int16),'pitch':np.zeros((4,49),np.uint8)}
    state = np.zeros((5,6,26),np.float32);state[:,:,-1]=1
    state[2,0,-1],state[2,0,3] = 0,1
    pitch = np.zeros((5,49),np.float32);pitch[2,3]=1
    arrays = source_arrays({'state':state,'pitch':pitch},targets,cfg,.25)
    assert arrays['mask'].tolist() == [False,True,True,True]
    assert arrays['observed'][1,0] == 3 and arrays['auxiliary_pitch'][1,3]
    with pytest.raises(ValueError,match='context'):
        source_arrays({'state':state[:-1],'pitch':pitch[:-1]},targets,cfg,.25)


def test_active_frame_error_categories_conserve_support_and_exclude_conflicts():
    target = np.full((4,6),25,np.int16);target[:,0]=[0,1,2,3];target[0,2]=-100
    observed = np.full((4,6),25,np.int16);observed[:,0]=[0,25,14,4];observed[1,1]=0
    arrays = {'target':target,'observed':observed,'best_fret':np.minimum(observed,24),'mask':np.ones(4,bool),
              'target_pitch':pitch_union(np.where(target<0,25,target)),
              'state_pitch':pitch_union(observed),'auxiliary_pitch':np.zeros((4,49),bool)}
    result = frame_diagnosis(arrays)
    counts=result['counts']
    assert [counts[k] for k in ['correct_fret','predicted_inactive','wrong_octave_same_string','other_wrong_fret']] == [1,1,1,1]
    assert counts['reference_active_frames'] == 4 and counts['active_on_reference_inactive'] == 1
    assert counts['reference_inactive_frames'] == 19


def test_short_initial_note_with_no_valid_frame_evidence_is_unavailable_not_zero_quality():
    target=np.full((30,6),25,np.int16)
    arrays={'target':target,'observed':target,'mask':np.arange(30)>=26,
            'state_pitch':np.zeros((30,49),bool),'auxiliary_pitch':np.zeros((30,49),bool)}
    result=taxonomy([note(0.,end=.1)],[],SCORE,CONFIG)
    evidence=add_note_evidence(result,arrays)['reference_rows'][0]['evidence']
    assert evidence['valid_source_frames']==0 and evidence['auxiliary_pitch_fraction'] is None


@pytest.mark.parametrize('key,value',[('new_optimizer_updates',1),('neural_inference_calls',1),
    ('test_inference_allowed',True),('source_delay_frames',0),('proposed_state_pitch_loss_weight',1.)])
def test_diagnostic_protocol_mutations_rejected(key,value):
    cfg=copy.deepcopy(CONFIG);cfg[key]=value
    with pytest.raises(ValueError):validate_config(cfg)


def test_recommendation_requires_both_model_checks_and_never_claims_training():
    row={'interventions':{'predicted':{'onset_f1':.1},'oracle_state':{'onset_f1':.8},'oracle_onset':{'onset_f1':.2}},
         'auxiliary_pitch_f1':.2,'state_derived_pitch_f1':.1}
    summary={m:copy.deepcopy(row) for m in CONFIG['models']}
    result=recommend(summary,CONFIG)
    assert result['supported_by_declared_rule'] and result['training_executed'] is False
    summary[CONFIG['models'][1]]['auxiliary_pitch_f1']=.05
    assert recommend(summary,CONFIG)['supported_by_declared_rule'] is False


def test_deterministic_state_projection_follows_standard_tuning_for_every_fret():
    for string in range(6):
        for fret in [0,7,24]:
            logits=torch.full((1,1,6,26),-50.)
            logits[...,25]=50.;logits[...,string,25]=-50.;logits[...,string,fret]=50.
            values=state_pitch_probabilities(logits)
            pitch=[0,5,10,15,19,24][string]+fret
            assert values.shape==(1,1,49) and values[0,0,pitch]>.99999
            assert values.sum().item()==pytest.approx(1.)


def test_probabilistic_unison_uses_union_not_double_counting():
    logits=torch.full((1,1,6,26),-100.)
    logits[...,25]=0.
    logits[...,0,5]=0.;logits[...,1,0]=0.
    probability=state_pitch_probabilities(logits)
    assert probability[0,0,5].item()==pytest.approx(.75)


def test_state_pitch_loss_masks_crop_edges_and_conflict_frames_and_has_finite_gradient():
    logits=torch.zeros((1,3,6,26),requires_grad=True)
    target={'state':torch.full((1,3,6),25),'pitch':torch.zeros((1,3,49)),'mask':torch.tensor([[1.,0.,1.]])}
    target['pitch'][0,0,7]=1.;target['state'][0,2,0]=-100
    loss=state_pitch_loss(logits,target);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad[:,0].abs().sum()>0 and logits.grad[:,1:].abs().sum()==0


def test_all_masked_state_pitch_loss_is_zero_and_preserves_gradient_connection():
    logits=torch.zeros((1,2,6,26),requires_grad=True)
    target={'state':torch.full((1,2,6),25),'pitch':torch.zeros((1,2,49)),'mask':torch.zeros((1,2))}
    loss=state_pitch_loss(logits,target);loss.backward()
    assert loss.item()==0 and logits.grad.abs().sum()==0


def test_zero_loss_coefficient_reproduces_legacy_total_and_positive_weight_adds_only_new_term():
    from cnn_gru.model import transcription_loss
    torch.manual_seed(17)
    outputs={'state':torch.randn((1,2,6,26),requires_grad=True),'onset':torch.randn((1,2,6)),
             'offset':torch.randn((1,2,6)),'pitch':torch.randn((1,2,49))}
    target={'state':torch.full((1,2,6),25),'pitch':torch.zeros((1,2,49)),
            'mask':torch.ones((1,2)),'onset':torch.zeros((1,2,6)),'offset':torch.zeros((1,2,6))}
    config={'model':{'max_fret':24},'training':{'inactive_class_weight':.35}}
    weights={'onset':[1.]*6,'offset':[1.]*6,'pitch':[1.]*49}
    baseline=transcription_loss(outputs,target,config,weights)
    zero=loss_with_state_pitch(outputs,target,config,weights,0.)
    positive=loss_with_state_pitch(outputs,target,config,weights,.5)
    assert torch.equal(zero['total'],baseline['total'])
    assert torch.allclose(positive['total'],baseline['total']+.5*positive['state_pitch'])
    assert all(torch.equal(positive[k],baseline[k]) for k in ['state','onset','offset','pitch'])


@pytest.mark.parametrize('shape',[(1,2,6,25),(1,2,5,26),(2,6,26)])
def test_projection_rejects_incompatible_state_interface(shape):
    with pytest.raises(ValueError):state_pitch_probabilities(torch.zeros(shape))
