"""Restore actual step500 model, AdamW, AMP and RNG state without relabeling it."""

import copy
import math

import torch

from normalization12.model import CausalCNNGRU
from optimization13.optimizer import _validate_group, recursive_digest


def restore_parent(parent, device):
    device = torch.device(device)
    required = {'model','optimizer','step','config','normalization','positive_weights','scaler','rng_cpu','rng_cuda'}
    if not required.issubset(parent) or type(parent['step']) is not int or parent['step'] != 500:
        raise ValueError('A complete actual step500 checkpoint is required.')
    normalization = parent['normalization']
    bins = parent['config']['feature']['n_bins']
    if not isinstance(normalization,dict) or set(normalization) != {'mean','std'}:
        raise ValueError('Frozen normalization is incomplete.')
    for name in ['mean','std']:
        values = normalization[name]
        if (not isinstance(values,list) or len(values)!=bins or
                any(type(v) not in (int,float) or not math.isfinite(v) for v in values)
                or (name=='std' and any(v<=0 for v in values))):
            raise ValueError('Frozen normalization is invalid.')
    weights = parent['positive_weights']
    if (not isinstance(weights,dict) or set(weights)!={'onset','offset','pitch'}
            or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in weights.values())):
        raise ValueError('Frozen positive loss weights are invalid.')
    before = {key: recursive_digest(parent[key]) for key in ['model','optimizer','scaler','rng_cpu','rng_cuda']}
    model = CausalCNNGRU(copy.deepcopy(parent['config'])).to(device)
    parameters = list(model.named_parameters())
    if list(parent['model']) != [name for name,_ in parameters]:
        raise ValueError('Saved model parameter ordering changed.')
    _validate_group(parent['optimizer'],len(parameters),.0001)
    for index,(name,parameter) in enumerate(parameters):
        value = parent['model'][name]
        if not isinstance(value,torch.Tensor) or value.shape!=parameter.shape or value.dtype!=parameter.dtype or not torch.isfinite(value).all():
            raise ValueError('Invalid saved model tensor: '+name)
        state = parent['optimizer']['state'][index]
        if not isinstance(state,dict) or set(state)!={'step','exp_avg','exp_avg_sq'}:
            raise ValueError('Missing or unknown AdamW moment fields.')
        step = state['step']
        if not isinstance(step,torch.Tensor) or step.shape!=torch.Size([]) or step.dtype!=torch.float32 or float(step.item())!=500.:
            raise ValueError('Every real parent AdamW counter must equal500.')
        for key in ['exp_avg','exp_avg_sq']:
            value = state[key]
            if not isinstance(value,torch.Tensor) or value.shape!=parameter.shape or value.dtype!=parameter.dtype or not torch.isfinite(value).all():
                raise ValueError('Invalid saved AdamW moment: '+name)
            if key=='exp_avg_sq' and (value<0).any():
                raise ValueError('Squared moments must be nonnegative.')
    rng_cpu,rng_cuda = parent['rng_cpu'],parent['rng_cuda']
    if not isinstance(rng_cpu,torch.Tensor) or rng_cpu.dtype!=torch.uint8 or rng_cpu.ndim!=1:
        raise ValueError('Invalid archived CPU RNG state.')
    torch.Generator(device='cpu').set_state(rng_cpu.cpu())
    expected_devices = torch.cuda.device_count() if device.type=='cuda' else 0
    if not isinstance(rng_cuda,list) or len(rng_cuda)!=expected_devices:
        raise ValueError('Archived CUDA RNG states require the original CUDA device count.')
    for index,state in enumerate(rng_cuda):
        if not isinstance(state,torch.Tensor) or state.dtype!=torch.uint8 or state.ndim!=1:
            raise ValueError('Invalid archived CUDA RNG state.')
        torch.Generator(device='cuda:'+str(index)).set_state(state.cpu())
    scaler = torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    if device.type=='cuda':
        state = parent['scaler']
        if (not isinstance(state,dict) or set(state)!={'scale','growth_factor','backoff_factor','growth_interval','_growth_tracker'}
                or any(type(state[k]) not in (int,float) or not math.isfinite(state[k]) for k in ['scale','growth_factor','backoff_factor'])
                or state['scale']<=0 or state['growth_factor']<=1 or not 0<state['backoff_factor']<1
                or type(state['growth_interval']) is not int or state['growth_interval']<1
                or type(state['_growth_tracker']) is not int or not 0<=state['_growth_tracker']<state['growth_interval']):
            raise ValueError('Invalid archived enabled AMP scaler state.')
    elif parent['scaler']!={}:
        raise ValueError('An enabled CUDA scaler cannot be silently discarded on CPU.')
    model.load_state_dict(parent['model'],strict=True)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001)
    optimizer.load_state_dict(copy.deepcopy(parent['optimizer']))
    scaler.load_state_dict(copy.deepcopy(parent['scaler']))
    # Reset after module construction/loading so initialization cannot consume
    # any of the restored parent random stream used by the actual continuation.
    torch.set_rng_state(rng_cpu.cpu().clone())
    if device.type=='cuda':
        torch.cuda.set_rng_state_all([state.cpu().clone() for state in rng_cuda])
    restored = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
    after = {key:recursive_digest(value) for key,value in restored.items()}
    if before!=after or any(recursive_digest(parent[key])!=digest for key,digest in before.items()):
        raise ValueError('Full restoration changed numerical state or mutated its parent.')
    proof = {'parent_step':500,'new_optimizer_updates':0,'parent_state_sha256':before,'restored_state_sha256':after,
             'parameter_names':[name for name,_ in parameters],'parameter_states':len(parameters),
             'model_parameters':sum(parameter.numel() for _,parameter in parameters),
             'held_learning_rate':.0001,'historical_rng_restored':True,'historical_scaler_restored':True,
             'parent_unmutated':True,'device':str(device)}
    return model,optimizer,scaler,proof
