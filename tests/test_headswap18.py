"""Fixed cached-head crossing, distinct frame policies and model-free replay."""

import copy
import numpy as np
import pytest
import torch
from baseline.download import ROOT
from decoder14.core import replay,score_case
from decoder14.workflow import read
from diagnosis15.analysis import pitch_union
from headswap18.data import MODELS,COMBINATIONS,cross_heads,validate_config
from headswap18.analysis import state_diagnosis,contrasts,note_changes,conditional
from headswap18.workflow import make_case,run


@pytest.fixture
def config():return read(ROOT/'configs/headswap18.json')


@pytest.mark.parametrize('state_source,onset_source',list(COMBINATIONS.values()))
def test_all_head_combinations_use_exact_independent_source_arrays(state_source,onset_source):
    probabilities={m:{k:np.full((3,2),v+i,np.float32) for k,v in [('state',0),('onset',2),('offset',4),('pitch',6)]}
                   for i,m in enumerate(MODELS)}
    before=copy.deepcopy(probabilities);out=cross_heads(probabilities,state_source,onset_source)
    for key,value in out.items():
        expected=probabilities[onset_source if key=='onset' else state_source][key]
        assert np.array_equal(value,expected) and not np.shares_memory(value,expected)
        value[:]=0
    for model in MODELS:
        for key in probabilities[model]:assert np.array_equal(before[model][key],probabilities[model][key])
    with pytest.raises(ValueError):cross_heads(probabilities,'unknown',onset_source)


def test_argmax_and_activity_policies_have_separate_support_and_confusion(config):
    config={**config,'source_frames':5,'source_delay_frames':1,'leading_mask_frames':0}
    target=np.full((5,6),25,np.int16);target[:,0]=[0,1,2,3,4];target[0,1]=-100
    state=np.zeros((6,6,26),np.float32);state[:,:,-1]=1
    for frame,fret,p in [(1,0,.6),(2,7,.4),(3,2,.2),(4,15,.4),(5,4,.3)]:
        state[frame,0,-1]=1-p;state[frame,0,fret]=p
    state[1,1,25]=0;state[1,1,7]=1 # ignored same-string conflict
    probabilities={'state':state,'pitch':np.zeros((6,49),np.float32)}
    targets={'state':target,'pitch':pitch_union(np.where(target<0,25,target))}
    result,arrays=state_diagnosis(probabilities,targets,config)
    argmax=result['policies']['argmax'];activity=result['policies']['decoder_activity']
    assert argmax['exact_state']['true_positives']==argmax['exact_state']['predicted_count']==1
    assert activity['exact_state']['true_positives']==2 and activity['exact_state']['predicted_count']==4
    counts=activity['counts']
    assert [counts[k] for k in ['correct_fret','predicted_inactive','wrong_octave_same_string','other_wrong_fret']]==[2,1,1,1]
    assert counts['reference_active_frames']==5 and counts['valid_string_frames']==29
    assert activity['activity']['true_positives']==4 and activity['activity']['reference_count']==5
    assert activity['fret_given_reference_and_prediction_active']=={'correct':2,'count':4,'accuracy':.5}
    assert result['best_active_fret_on_reference_active']=={'correct':3,'count':5,'accuracy':.6}
    for policy in [argmax,activity]:
        matrix=np.array(policy['confusion']);assert matrix.sum()==29 and matrix.shape==(26,26)
        assert matrix[:25,:25].trace()==policy['exact_state']['true_positives']
    assert arrays['target'][0,1]==-100


def test_delay_and_leading_mask_apply_once_and_empty_conditional_is_unavailable(config):
    cfg={**config,'source_frames':4,'source_delay_frames':2,'leading_mask_frames':2}
    state=np.zeros((6,6,26),np.float32);state[:,:,-1]=1;state[4,0,25]=0;state[4,0,3]=1
    target=np.full((4,6),25,np.int16);target[2,0]=3
    result,arrays=state_diagnosis({'state':state,'pitch':np.zeros((6,49),np.float32)},
                {'state':target,'pitch':pitch_union(target)},cfg)
    assert arrays['mask'].tolist()==[False,False,True,True]
    assert result['policies']['argmax']['exact_state']['f1']==1
    assert result['policies']['argmax']['counts']['valid_string_frames']==12
    assert conditional(0,0)['accuracy'] is None


def test_factorial_contrasts_retain_both_paths_and_interaction():
    rows={k:{'onset_f1':v} for k,v in zip(COMBINATIONS,[.1,.2,.15,.12])}
    value=contrasts(rows)
    assert value['onset_swap_at_control_state']==pytest.approx(.05)
    assert value['state_swap_at_control_onset']==pytest.approx(.02)
    assert value['interaction']==pytest.approx(.03)
    assert value['onset_swap_at_control_state']+value['state_swap_at_fused_onset']==pytest.approx(.1)
    assert value['state_swap_at_control_onset']+value['onset_swap_at_fused_state']==pytest.approx(.1)


def test_note_change_rows_use_identity_and_official_matches_only():
    def row(categories):return {'taxonomy':{'reference_rows':[{'reference_index':i,'note':{'start':i},'category':c}
                            for i,c in enumerate(categories)]}}
    a=row(['matched_same_string','missed','matched_other_string','nearby_same_pitch'])
    b=row(['matched_other_string','matched_same_string','nearby_other_pitch','missed'])
    changes=note_changes(a,b)
    assert [r['change'] for r in changes]==['matched_both','fused_only','control_only','missed_both']
    b['taxonomy']['reference_rows'][0]['note']['start']=.5
    with pytest.raises(ValueError):note_changes(a,b)


@pytest.mark.parametrize('key,value',[('new_optimizer_updates',1),('neural_inference_calls',1),
    ('test_inference_allowed',True),('production_promotion',True),('calibration_search_allowed',True),
    ('source_delay_frames',0),('leading_mask_frames',0),('packet_ms',10),
    ('decoder_parameters',{'onset_threshold':.55,'activity_threshold':.25,'onset_backdate_seconds':0})])
def test_fixed_zero_budget_protocol_rejects_mutations(config,key,value):
    validate_config(config);config[key]=value
    with pytest.raises(ValueError):validate_config(config)


def test_completed_output_root_cannot_be_run_again(config,tmp_path):
    (tmp_path/'protocol.json').write_text('{}')
    with pytest.raises(ValueError,match='already exists'):run(config,tmp_path)


def test_packet_replay_ignores_auxiliary_heads_and_has_no_measured_completion(config,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Neural inference or optimizer step was called')
    monkeypatch.setattr(torch.nn.Module,'_call_impl',forbidden)
    monkeypatch.setattr(torch.optim.AdamW,'step',forbidden)
    base=read(ROOT/'configs/streaming.json')
    def probabilities(fret):
        state=np.zeros((1200,6,26),np.float32);state[:,:,-1]=1
        state[30:90,0,25]=0;state[30:90,0,fret]=1
        onset=np.zeros((1200,6),np.float32);onset[30,0]=1
        return {'state':state,'onset':onset,'offset':np.zeros((1200,6),np.float32),'pitch':np.zeros((1200,49),np.float32)}
    left=probabilities(3);right=probabilities(5);right['onset'][:]=0;right['onset'][40,0]=1
    decoded={}
    for mode,(ss,os) in COMBINATIONS.items():
        arrays=cross_heads(dict(zip(MODELS,[left,right])),ss,os)
        notes,events,clamps=replay(arrays,base,config['decoder_parameters'],12.)
        assert notes and len(events)==2*len(notes)
        decoded[mode]=notes
        assert all('delivered_at_seconds' not in e for e in events)
        assert all('on_delivered_at_seconds' not in n and 'off_delivered_at_seconds' not in n for n in notes)
        arrays['offset'][:]=1;arrays['pitch'][:]=1
        assert replay(arrays,base,config['decoder_parameters'],12.)==(notes,events,clamps)
    # Activity opens the first note without an onset peak; a later onset only splits a same-fret run.
    assert len(decoded['control'])==len(decoded['fused_state_control_onset'])==1
    assert len(decoded['fused'])==len(decoded['control_state_fused_onset'])==2
    assert decoded['control'][0]['start']==decoded['control_state_fused_onset'][0]['start']
    assert decoded['control'][0]['end']==decoded['control_state_fused_onset'][-1]['end']
