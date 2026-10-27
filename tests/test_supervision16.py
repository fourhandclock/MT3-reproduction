"""Continuation contracts; synthetic steps initialize toy moments, never train audio."""

import copy

import numpy as np
import pytest
import torch

from baseline.download import ROOT
from decoder14.workflow import read
from normalization12.model import CausalCNNGRU
from optimization13.data import build_schedule
from optimization13.optimizer import recursive_digest
from supervision16.data import validate_config
from supervision16.restore import restore_parent
from supervision16.training import optimize
from supervision16.workflow import choose


@pytest.fixture(scope='module')
def parent():
    torch.manual_seed(811)
    config={'feature':{'n_bins':8},'model':{'channels':[8,8,8],'projection_size':8,'gru_hidden_size':8,
        'gru_layers':2,'bidirectional':False,'dropout':.1,'output_size':8,'max_fret':24,'pitch_min':40,'pitch_max':88},
        'training':{'inactive_class_weight':.35}}
    model=CausalCNNGRU(config)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001)
    for index,parameter in enumerate(model.parameters()):parameter.grad=torch.full_like(parameter,(index+1)/100.)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    for state in optimizer.state.values():state['step'].fill_(500)
    optimizer.param_groups[0]['initial_lr']=.001
    return {'model':copy.deepcopy(model.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),'step':500,
        'config':config,'normalization':{'mean':[0.]*8,'std':[1.]*8},'positive_weights':{'onset':1.,'offset':1.,'pitch':1.},
        'scaler':{},'rng_cpu':torch.get_rng_state().clone(),'rng_cuda':[]}


def test_complete_restore_preserves_all_parent_state_and_cpu_random_stream(parent):
    before=recursive_digest(parent)
    model,optimizer,scaler,proof=restore_parent(parent,'cpu')
    assert proof['parent_state_sha256']==proof['restored_state_sha256']
    assert proof['historical_rng_restored'] and proof['historical_scaler_restored']
    assert torch.equal(torch.get_rng_state(),parent['rng_cpu'])
    assert recursive_digest(model.state_dict())==recursive_digest(parent['model'])
    assert recursive_digest(optimizer.state_dict())==recursive_digest(parent['optimizer'])
    assert scaler.state_dict()=={} and recursive_digest(parent)==before


def test_repeated_restores_produce_identical_training_dropout_outputs(parent):
    x=torch.zeros((2,20,8)); outputs=[]
    for _ in range(2):
        model,_,_,_=restore_parent(parent,'cpu')
        model.train()
        with torch.no_grad():outputs.append(model(x))
    assert all(torch.equal(outputs[0][key],outputs[1][key]) for key in outputs[0])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA not available')
def test_actual_shared_parent_restores_amp_rng_and_matches_stochastic_forward_without_updates():
    path=ROOT/'outputs/optimization13/models/group8/final.pt'
    if not path.exists():pytest.skip('Cached research parent not present')
    parent=torch.load(path,map_location='cpu',weights_only=True)
    before=recursive_digest({key:parent[key] for key in ['model','optimizer','scaler','rng_cpu','rng_cuda']})
    values=[]
    for _ in range(2):
        model,optimizer,scaler,proof=restore_parent(parent,'cuda')
        assert scaler.state_dict()==parent['scaler']
        assert torch.equal(torch.get_rng_state(),parent['rng_cpu'])
        assert all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),parent['rng_cuda']))
        model.train()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            values.append({key:value.cpu() for key,value in model(torch.zeros((2,30,216),device='cuda')).items()})
        assert all(state['step'].item()==500 for state in optimizer.state.values())
        del model,optimizer,scaler
    assert all(torch.equal(values[0][key],values[1][key]) for key in values[0])
    assert recursive_digest({key:parent[key] for key in ['model','optimizer','scaler','rng_cpu','rng_cuda']})==before


@pytest.mark.parametrize('change',['step','moment_step','negative_square','wrong_order','missing_rng','wrong_rng_dtype',
                                  'wrong_rng_shape','missing_scaler','scaler_on_cpu','missing_metadata','normalization',
                                  'weights','weight_dtype','moment_dtype'])
def test_ambiguous_or_corrupt_parent_rejected(parent,change):
    p=copy.deepcopy(parent)
    if change=='step':p['step']=250
    elif change=='moment_step':p['optimizer']['state'][0]['step'].fill_(250)
    elif change=='negative_square':p['optimizer']['state'][0]['exp_avg_sq'].flatten()[0]=-1.
    elif change=='wrong_order':p['optimizer']['param_groups'][0]['params'].reverse()
    elif change=='missing_rng':del p['rng_cpu']
    elif change=='wrong_rng_dtype':p['rng_cpu']=p['rng_cpu'].float()
    elif change=='wrong_rng_shape':p['rng_cpu']=torch.zeros(1,dtype=torch.uint8)
    elif change=='missing_scaler':del p['scaler']
    elif change=='scaler_on_cpu':p['scaler']={'scale':1024.}
    elif change=='missing_metadata':del p['positive_weights']
    elif change=='normalization':p['normalization']['std'][0]=0
    elif change=='weights':p['positive_weights']['pitch']=float('nan')
    elif change=='weight_dtype':name=next(iter(p['model']));p['model'][name]=p['model'][name].double()
    elif change=='moment_dtype':p['optimizer']['state'][0]['exp_avg']=p['optimizer']['state'][0]['exp_avg'].double()
    with pytest.raises((ValueError,RuntimeError)):restore_parent(p,'cpu')


def test_toy_optimizer_next_counter_is501_and_parent_is_unmutated(parent):
    before=recursive_digest(parent)
    model,optimizer,_,_=restore_parent(parent,'cpu')
    for p in model.parameters():p.grad=torch.ones_like(p)
    optimizer.step()
    assert all(state['step'].item()==501 for state in optimizer.state.values())
    assert recursive_digest(parent)==before


@pytest.mark.parametrize('coefficient',[0.,.5])
def test_optimization_uses_expected_term_and_advances_one_toy_step(parent,coefficient):
    model,optimizer,scaler,_=restore_parent(parent,'cpu')
    x=torch.zeros((1,10,8))
    target={'state':torch.full((1,10,6),25),'pitch':torch.zeros((1,10,49)),
            'onset':torch.zeros((1,10,6)),'offset':torch.zeros((1,10,6)),'mask':torch.ones((1,10))}
    values,gradient=optimize(model,x,target,optimizer,scaler,parent['config'],parent['positive_weights'],
                            {'mixed_precision':True,'gradient_clip':1.},torch.device('cpu'),coefficient)
    assert ('state_pitch' in values)==bool(coefficient)
    assert all(state['step'].item()==501 for state in optimizer.state.values())
    assert np.isfinite(gradient) and all(np.isfinite(value) for value in values.values())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA not available')
def test_probability_bce_is_safe_outside_cuda_autocast(parent):
    p=copy.deepcopy(parent)
    p['rng_cuda']=torch.cuda.get_rng_state_all()
    # This tiny 8-bin synthetic model has near-degenerate GroupNorm groups.
    # Use a small archived toy scale; the actual parent test above verifies its
    # original1024 scale without changing the real continuation recipe.
    p['scaler']=torch.amp.GradScaler('cuda',init_scale=1.).state_dict()
    model,optimizer,scaler,_=restore_parent(p,'cuda')
    target={'state':torch.full((1,10,6),25,device='cuda'),'pitch':torch.zeros((1,10,49),device='cuda'),
            'onset':torch.zeros((1,10,6),device='cuda'),'offset':torch.zeros((1,10,6),device='cuda'),
            'mask':torch.ones((1,10),device='cuda')}
    values,_=optimize(model,torch.zeros((1,10,8),device='cuda'),target,optimizer,scaler,p['config'],p['positive_weights'],
                       {'mixed_precision':True,'gradient_clip':1.},torch.device('cuda'),.5)
    assert np.isfinite(values['state_pitch'])
    assert all(state['step'].item()==501 for state in optimizer.state.values())


@pytest.mark.parametrize('key,value',[('data_epoch_start',5),('total_optimizer_update_cap',1000),('test_inference_allowed',True),
                                     ('production_promotion',True),('decoder_parameters',{}),('arms',[])])
def test_fixed_loss_comparison_protocol_rejects_mutations(key,value):
    config=read(ROOT/'configs/supervision16.json');config[key]=value
    with pytest.raises(ValueError):validate_config(config)


def test_next_crop_schedule_uses_epochs11_to15_and_is_identical_on_repeat():
    config=read(ROOT/'configs/supervision16.json')
    records=[{'entry':{'id':str(i)},'features':np.zeros((2400,2))} for i in range(3)]
    schedule=build_schedule(config,records)
    assert len(schedule)==1000 and {r['epoch'] for r in schedule}=={11,12,13,14,15}
    assert schedule==build_schedule(config,records)


def test_selection_only_fixed_finals_primary_state_secondary_onset_control_tie():
    scores={'control750':{'exact_state_f1':.1,'onset_f1':.2},'state_pitch750':{'exact_state_f1':.1,'onset_f1':.2}}
    assert choose(scores)=='control750'
    scores['state_pitch750']['onset_f1']=.21;assert choose(scores)=='state_pitch750'
    scores['control750']['exact_state_f1']=.101;assert choose(scores)=='control750'
    scores['parent500']=scores['control750']
    with pytest.raises(ValueError):choose(scores)
