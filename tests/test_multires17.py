"""Causality, capacity and paired-experiment contracts; no acoustic optimization."""

import copy
import numpy as np
import pytest
import torch
from baseline.download import ROOT
from decoder14.workflow import read
from acoustic10.training import tensor_digest
from multires17.data import ARMS,validate_config,model_configuration,batch_from_schedule
from multires17.frontend import MultiResolutionFrontend
from multires17.model import CausalCNNGRU
from multires17.training import fresh_arm
from multires17.workflow import choose
from streaming.frontend import CausalFrontend


@pytest.fixture
def config():
    return read(ROOT/'configs/multires17.json')


def model_config(config,arm=0):
    return model_configuration(read(ROOT/'configs/streaming.json'),config,ARMS[arm])


@pytest.mark.parametrize('arm',[0,1])
def test_frontend_packet_full_single_plane_equivalence_and_bounded_cache(config,arm):
    spec=model_config(config,arm);pcm=np.random.default_rng(1).normal(0,.1,16137).astype(np.float32)
    full=MultiResolutionFrontend(spec).push(pcm)
    frontend=MultiResolutionFrontend(spec);parts=[]
    for start in range(0,len(pcm),137):
        parts.append(frontend.push(pcm[start:start+137]))
        assert frontend.cached_samples < sum(set(ARMS[arm]['windows']))
    assert np.array_equal(np.concatenate(parts),full)
    assert full.shape==(100,2,216) and frontend.last_frame_sample==16000
    for plane,window in enumerate(ARMS[arm]['windows']):
        single=copy.deepcopy(spec);single['feature'].update(window_length=window,n_fft=window)
        assert np.array_equal(full[:,plane],CausalFrontend(single).push(pcm))
    assert frontend.push(np.empty(0,np.float32)).shape==(0,2,216)
    frontend.reset();assert frontend.received_samples==frontend.cached_samples==0
    assert np.array_equal(frontend.push(pcm),full)


@pytest.mark.parametrize('arm',[0,1])
def test_frontend_prefix_first_frame_and_invalid_input_do_not_advance(config,arm):
    spec=model_config(config,arm);pcm=np.random.default_rng(2).normal(0,.1,3000).astype(np.float32)
    frontend=MultiResolutionFrontend(spec)
    assert frontend.push(pcm[:159]).shape==(0,2,216)
    assert frontend.push(pcm[159:160]).shape==(1,2,216)
    for bad in [np.array([np.nan]),np.zeros((2,2)),np.array([np.inf])]:
        with pytest.raises(ValueError):frontend.push(bad)
        assert all(f.received_samples==160 for f in frontend.frontends.values())
    prefix=MultiResolutionFrontend(spec).push(pcm[:1440])
    assert np.array_equal(prefix,MultiResolutionFrontend(spec).push(pcm)[:len(prefix)])


def test_model_chunk_equivalence_prefix_detached_bounded_state_and_empty(config):
    torch.manual_seed(17);model=CausalCNNGRU(model_config(config)).eval()
    x=torch.randn(2,19,2,216)
    with torch.no_grad():
        full=model(x);state=None;parts={k:[] for k in full}
        for a,b in [(0,1),(1,3),(3,10),(10,19)]:
            block,state=model.forward_chunk(x[:,a:b],state)
            for k in full:parts[k].append(block[k])
        for k in full:
            torch.testing.assert_close(torch.cat(parts[k],1),full[k],rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(model(x[:,:9])[k],full[k][:,:9],rtol=1e-5,atol=1e-6)
        assert [tuple(c.shape) for c in state['conv_contexts']]==[(2,2,2,216),(2,24,2,108),(2,48,2,54)]
        assert all(not c.requires_grad for c in state['conv_contexts'])
        empty,unchanged=model.forward_chunk(x[:,:0],state)
        assert unchanged is state and empty['state'].shape==(2,0,6,26)
        with pytest.raises(ValueError):model(torch.zeros(2,19,216))
        with pytest.raises(ValueError):model.forward_chunk(x,{'conv_contexts':state['conv_contexts'][:1]})
        bad={**state,'conv_contexts':(torch.zeros(2,1,2,216),)+state['conv_contexts'][1:]}
        with pytest.raises(ValueError):model.forward_chunk(x,bad)


def test_capacity_shared_initialization_and_dropout(config):
    torch.manual_seed(42);model=CausalCNNGRU(model_config(config))
    initial={'model':copy.deepcopy(model.state_dict()),'rng_cpu':torch.get_rng_state(),
             'rng_cuda':[]}
    assert sum(p.numel() for p in model.parameters())==931361
    outputs=[];digests=[]
    x=torch.linspace(-1,1,17*2*216).reshape(1,17,2,216)
    for arm in range(2):
        net,opt,scheduler,scaler=fresh_arm(initial,model_config(config,arm),config['training'],torch.device('cpu'))
        assert not opt.state and scheduler.last_epoch==0 and not scaler.is_enabled()
        digests.append(tensor_digest(net.state_dict()));outputs.append(net(x))
    assert digests[0]==digests[1]
    for key in outputs[0]:assert torch.equal(outputs[0][key],outputs[1][key])


@pytest.mark.parametrize('arm',[0,1])
def test_both_planes_get_same_gain_normalization_masks_and_no_cache_mutation(arm):
    short=np.arange(60,dtype=np.float32).reshape(10,6)/10
    long=short+2;before=short.copy();targets=[{'state':np.zeros((10,6),np.int64),'mask':np.ones(10,np.float32)}]
    row={'record_index':0,'start_frame':2,'frames':6,'gain_log':.2}
    stats={'mean':[[0.]*6,[1.]*6],'std':[[1.]*6,[2.]*6]}
    x,y=batch_from_schedule([row],[{'features':short,'long_features':long}],targets,stats,1,ARMS[arm])
    second=short if arm==0 else long
    expected=np.stack([short[2:8]+.2,(second[2:8]+.2-1)/2],axis=1)
    np.testing.assert_allclose(x[0],expected)
    assert y['mask'].tolist()==[[0,1,1,1,1,0]]
    assert y['state'].dtype==torch.int64 and np.array_equal(short,before)
    assert targets[0]['mask'].sum()==10


@pytest.mark.parametrize('key,value',[('total_optimizer_update_cap',501),('seed',43),('data_epoch_start',10),
    ('test_inference_allowed',True),('production_promotion',True),('common_target_delay_frames',26),
    ('common_leading_mask_frames',13),('arms',[{'name':'short_long','windows':[2048,4096]}]),
    ('decoder_parameters',{'onset_threshold':.55,'activity_threshold':.25,'onset_backdate_seconds':0})])
def test_protocol_rejects_unmatched_changes(config,key,value):
    validate_config(config);config[key]=value
    with pytest.raises(ValueError):validate_config(config)


def test_selection_fixed_final_primary_secondary_and_control_tie():
    rows={m:{'exact_state_f1':.1,'onset_f1':.2} for m in ['short_short','short_long']}
    assert choose(rows)=='short_short'
    rows['short_long']['onset_f1']=.21;assert choose(rows)=='short_long'
    rows['short_short']['exact_state_f1']=.11;assert choose(rows)=='short_short'
    rows['short_long']['exact_state_f1']=float('nan')
    with pytest.raises(ValueError):choose(rows)
    with pytest.raises(ValueError):choose({'short_short':rows['short_short']})


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_two_plane_cuda_amp_forward_backward_has_finite_scaled_gradients(config):
    torch.manual_seed(42);model=CausalCNNGRU(model_config(config,1)).cuda().train()
    x=torch.linspace(-2,2,2*40*2*216,device='cuda').reshape(2,40,2,216)
    with torch.autocast('cuda',dtype=torch.float16):
        outputs=model(x)
        loss=sum(v.float().square().mean() for v in outputs.values())
    (loss*1024).backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
