"""Fret-only gradients, complete continuation state and fixed budget contracts."""

import copy
import math
import numpy as np
import pytest
import torch
from acoustic10.training import optimize as legacy_optimize
from baseline.download import ROOT
from decoder14.workflow import read
from fret19.data import validate_config
from fret19.loss import conditional_fret_loss,active_support
from fret19.restore import restore_parent
from fret19.training import optimize
from fret19.workflow import choose,run
from multires17.model import CausalCNNGRU
from optimization13.data import build_schedule
from optimization13.optimizer import recursive_digest


def labels(batch=1,time=3):
    return {'state':torch.full((batch,time,6),25,dtype=torch.long),'mask':torch.ones(batch,time)}


def test_uniform_active_conditional_ce_ignores_inactive_conflicts_and_masks():
    logits=torch.zeros(1,3,6,26,requires_grad=True);target=labels()
    target['state'][0,0,:3]=torch.tensor([2,-100,5]);target['state'][0,1,0]=4;target['mask'][0,1]=0
    loss=conditional_fret_loss(logits,target)
    assert loss.item()==pytest.approx(math.log(25)) and active_support(logits,target).sum()==2
    loss.backward();grad=logits.grad
    assert torch.count_nonzero(grad[...,-1])==0
    assert torch.count_nonzero(grad[0,0,1])==0 and torch.count_nonzero(grad[0,1:])==0
    assert torch.count_nonzero(grad[0,0,2])>0 # a conflict on another string does not mask this valid target


def test_loss_mean_uses_active_string_frames_and_is_invariant_to_inactive_logits():
    torch.manual_seed(91);logits=torch.randn(1,3,6,26,requires_grad=True);target=labels()
    target['state'][0,0,0]=2;target['state'][0,2,5]=24
    expected=torch.nn.functional.cross_entropy(torch.stack([logits[0,0,0,:25],logits[0,2,5,:25]]),torch.tensor([2,24]))
    actual=conditional_fret_loss(logits,target);assert torch.equal(actual,expected)
    changed=logits.detach().clone();changed[...,-1]=10000
    torch.testing.assert_close(conditional_fret_loss(changed,target),actual)
    changed[...,:25]+=4;torch.testing.assert_close(conditional_fret_loss(changed,target),actual)


@pytest.mark.parametrize('kind',['inactive','conflicts','masked'])
def test_no_valid_active_targets_return_differentiable_zero(kind):
    logits=torch.randn(1,3,6,26,requires_grad=True);target=labels()
    if kind=='conflicts':target['state'][:]=-100
    if kind=='masked':target['state'][:]=3;target['mask'][:]=0
    loss=conditional_fret_loss(logits,target);assert loss.requires_grad and loss.item()==0
    loss.backward();assert torch.count_nonzero(logits.grad)==0


@pytest.mark.parametrize('bad',['shape','label','mask_nan','mask_range','dtype'])
def test_malformed_supervision_is_rejected(bad):
    logits=torch.zeros(1,3,6,26);target=labels()
    if bad=='shape':logits=logits[...,:25]
    if bad=='label':target['state'][0,0,0]=-1
    if bad=='mask_nan':target['mask'][0,0]=float('nan')
    if bad=='mask_range':target['mask'][0,0]=2
    if bad=='dtype':target['state']=target['state'].float()
    with pytest.raises(ValueError):conditional_fret_loss(logits,target)


@pytest.fixture(scope='module')
def parent():
    torch.manual_seed(711)
    config={'feature':{'n_bins':8},'feature_planes':[2048,2048],
        'model':{'channels':[8,8,8],'projection_size':8,'gru_hidden_size':8,'gru_layers':2,
                 'bidirectional':False,'dropout':.1,'output_size':8,'max_fret':24,'pitch_min':40,'pitch_max':88},
        'training':{'inactive_class_weight':.35}}
    model=CausalCNNGRU(config);optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001)
    for i,p in enumerate(model.parameters()):p.grad=torch.full_like(p,(i+1)/100.)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    for state in optimizer.state.values():state['step'].fill_(250)
    optimizer.param_groups[0]['initial_lr']=.001
    return {'model':copy.deepcopy(model.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),'step':250,
        'config':config,'normalization':{'mean':[[0.]*8]*2,'std':[[1.]*8]*2},
        'positive_weights':{'onset':1.,'offset':1.,'pitch':1.},'scaler':{},
        'rng_cpu':torch.get_rng_state().clone(),'rng_cuda':[],'scheduler':{'last_epoch':250}}


def test_full_restore_retains_parent_and_paired_dropout_state(parent):
    before=recursive_digest(parent);out=[]
    for _ in range(2):
        model,optimizer,scaler,proof=restore_parent(parent,'cpu')
        assert proof['parent_state_sha256']==proof['restored_state_sha256'] and proof['parent_step']==250
        assert recursive_digest(optimizer.state_dict())==recursive_digest(parent['optimizer'])
        assert torch.equal(torch.get_rng_state(),parent['rng_cpu']) and scaler.state_dict()=={}
        out.append(model(torch.zeros(2,20,2,8)))
    for key in out[0]:assert torch.equal(out[0][key],out[1][key])
    assert recursive_digest(parent)==before


@pytest.mark.parametrize('change',['step','counter','moment','plane','normalization','rng','scaler'])
def test_restore_rejects_incomplete_or_mismatched_parent(parent,change):
    p=copy.deepcopy(parent)
    if change=='step':p['step']=500
    if change=='counter':p['optimizer']['state'][0]['step'].fill_(249)
    if change=='moment':del p['optimizer']['state'][0]['exp_avg']
    if change=='plane':p['config']['feature_planes']=[2048,4096]
    if change=='normalization':p['normalization']['std'][0][0]=0
    if change=='rng':p['rng_cpu']=torch.zeros(2,dtype=torch.float32)
    if change=='scaler':p['scaler']={'scale':1024.}
    with pytest.raises((ValueError,RuntimeError)):restore_parent(p,'cpu')


def batch_target():
    target=labels(1,10);target.update(pitch=torch.zeros(1,10,49),onset=torch.zeros(1,10,6),offset=torch.zeros(1,10,6))
    target['state'][:,:,0]=2
    return target


def test_zero_coefficient_control_is_bit_exact_to_legacy_step(parent):
    x=torch.zeros(1,10,2,8);target=batch_target();spec={'mixed_precision':False,'gradient_clip':1.}
    a,oa,sa,_=restore_parent(parent,'cpu')
    old,gn=legacy_optimize(a,x,target,oa,sa,parent['config'],parent['positive_weights'],spec,torch.device('cpu'))
    b,ob,sb,_=restore_parent(parent,'cpu')
    new,ngn,count=optimize(b,x,target,ob,sb,parent['config'],parent['positive_weights'],spec,torch.device('cpu'),0.)
    assert new==old and ngn==gn and count==10
    assert recursive_digest(a.state_dict())==recursive_digest(b.state_dict())
    assert recursive_digest(oa.state_dict())==recursive_digest(ob.state_dict())


def test_treatment_adds_expected_term_and_advances_only_one_toy_step(parent):
    model,optimizer,scaler,_=restore_parent(parent,'cpu')
    values,gradient,count=optimize(model,torch.zeros(1,10,2,8),batch_target(),optimizer,scaler,
        parent['config'],parent['positive_weights'],{'mixed_precision':False,'gradient_clip':1.},torch.device('cpu'),.5)
    expected=values['state']+values['onset']+.5*values['offset']+.5*values['pitch']+.5*values['conditional_fret']
    assert values['total']==pytest.approx(expected,abs=1e-6) and count==10 and np.isfinite(gradient)
    assert all(s['step'].item()==251 for s in optimizer.state.values())


@pytest.mark.parametrize('key,value',[('data_epoch_start',10),('total_optimizer_update_cap',1000),
    ('test_inference_allowed',True),('production_promotion',True),('decoder_parameters',{}),('arms',[])])
def test_fixed_protocol_rejects_mutations(key,value):
    config=read(ROOT/'configs/fret19.json');validate_config(config);config[key]=value
    with pytest.raises(ValueError):validate_config(config)


def test_schedule_epochs6_to10_and_final_only_selection():
    config=read(ROOT/'configs/fret19.json')
    records=[{'entry':{'id':str(i)},'features':np.zeros((2400,2))} for i in range(3)]
    schedule=build_schedule(config,records)
    assert len(schedule)==1000 and {r['epoch'] for r in schedule}=={6,7,8,9,10}
    assert schedule==build_schedule(config,records)
    rows={m:{'exact_state_f1':.1,'onset_f1':.2} for m in ['control500','conditional_fret500']}
    assert choose(rows)=='control500';rows['conditional_fret500']['onset_f1']=.21
    assert choose(rows)=='conditional_fret500';rows['control500']['exact_state_f1']=.11
    assert choose(rows)=='control500'
    rows['parent250']=rows['control500']
    with pytest.raises(ValueError):choose(rows)


def test_existing_run_rejects_repeat_training(tmp_path):
    (tmp_path/'protocol.json').write_text('{}')
    with pytest.raises(ValueError,match='already started'):run(read(ROOT/'configs/fret19.json'),tmp_path,torch.device('cpu'))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_real_parent_amp_restoration_and_auxiliary_gradients_without_optimizer_update():
    parent=torch.load(ROOT/'outputs/multires17/models/short_short/final.pt',map_location='cpu',weights_only=True)
    model,optimizer,scaler,proof=restore_parent(parent,'cuda')
    assert proof['model_parameters']==931361 and scaler.get_scale()==1024
    x=torch.linspace(-1,1,40*2*216,device='cuda').reshape(1,40,2,216)
    with torch.autocast('cuda',dtype=torch.float16):logits=model(x)['state']
    target={'state':torch.full((1,40,6),3,device='cuda',dtype=torch.long),'mask':torch.ones(1,40,device='cuda')}
    loss=conditional_fret_loss(logits,target);assert loss.dtype==torch.float32
    scaler.scale(loss).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(s['step'].item()==250 for s in optimizer.state.values())
