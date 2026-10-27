"""One fresh initialization, matched dropout/crops, exactly 250 updates per arm."""

import time
import numpy as np
import torch
from acoustic10.data import prepare_targets
from acoustic10.training import optimize, tensor_digest
from baseline.download import ROOT,sha256
from cnn_gru.training import save_checkpoint
from optimization13.optimizer import recursive_digest
from spectral11.training import actual_optimizer_steps, verified_sanity
from spectral11.training import producer_hashes as spectral_producers
from streaming.data import fingerprint
from streaming.evaluate import save_json
from .data import model_configuration,batch_from_schedule
from .model import CausalCNNGRU


def producer_hashes():
    paths = [ROOT/name for name in spectral_producers()]
    paths += [ROOT/'multires17'/name for name in
              ['__init__.py','__main__.py','model.py','frontend.py','data.py','training.py','workflow.py']]
    paths += [ROOT/name for name in ['configs/multires17.json','optimization13/data.py',
              'optimization13/optimizer.py','spectral11/workflow.py','diagnosis15/analysis.py',
              'decoder14/core.py','decoder14/workflow.py']]
    return {p.relative_to(ROOT).as_posix():sha256(p) for p in sorted(set(paths))}


def initialize(config,base,path):
    torch.manual_seed(config['seed'])
    model = CausalCNNGRU(model_configuration(base,config,config['arms'][0]))
    state = {'model':model.state_dict(),'rng_cpu':torch.get_rng_state(),
             'rng_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
             'model_parameters':sum(p.numel() for p in model.parameters()), 'step':0}
    save_checkpoint(path,state)


def fresh_arm(initial,model_config,spec,device):
    model = CausalCNNGRU(model_config).to(device)
    model.load_state_dict(initial['model'])
    optimizer = torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=250,eta_min=spec['learning_rate']*.1)
    scaler = torch.amp.GradScaler('cuda',init_scale=1024,enabled=device.type=='cuda' and spec['mixed_precision'])
    torch.set_rng_state(initial['rng_cpu'])
    if device.type=='cuda':
        torch.cuda.set_rng_state_all(initial['rng_cuda'])
    return model,optimizer,scheduler,scaler


def train_arm(arm,config,base,audit,records,provenance,schedule,output,device):
    directory = output/'models'/arm['name']
    directory.mkdir(parents=True,exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError('Existing arm cannot be trained again')
    initial_path = output/'initial-weights.pt'
    initial = torch.load(initial_path,map_location='cpu',weights_only=True)
    model_config = model_configuration(base,config,arm)
    spec = config['training']
    model,optimizer,scheduler,scaler = fresh_arm(initial,model_config,spec,device)
    pre = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
           'scaler':scaler.state_dict(),'rng_cpu':torch.get_rng_state(),
           'rng_cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],'step':0}
    save_checkpoint(directory/'before-training.pt',pre)
    initial_state_digest = recursive_digest(pre)
    targets = prepare_targets(records,13,config,base)
    history,losses_in_epoch = [],[]
    if device.type=='cuda':
        torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for step in range(1,251):
        x,target = batch_from_schedule(schedule[(step-1)*4:step*4],records,targets,
                    audit['normalization'][arm['name']],26,arm)
        losses,gradient = optimize(model,x.to(device),{k:v.to(device) for k,v in target.items()},
                    optimizer,scaler,model_config,audit['positive_weights'],spec,device)
        counters = actual_optimizer_steps(optimizer)
        if counters != {'minimum':step,'maximum':step,'parameter_states':34}:
            raise ValueError('Actual optimizer steps differ; no skipped-update replacements permitted')
        scheduler.step();losses_in_epoch.append(losses)
        if step%50==0:
            history.append({'epoch':step//50,'optimizer_steps':step,
                'actual_optimizer_step_counters':counters,
                'training_losses':{k:float(np.mean([row[k] for row in losses_in_epoch])) for k in losses},
                'gradient_norm':gradient,'learning_rate_for_next_update':scheduler.get_last_lr()[0]})
            losses_in_epoch=[]
            print('Stage17',arm['name'],'epoch',step//50,'update',step,'state loss',
                  round(history[-1]['training_losses']['state'],4),flush=True)
    if device.type=='cuda':torch.cuda.synchronize(device)
    receipt = {'name':arm['name'],'new_optimizer_steps':250,'optimizer_steps':250,'lifetime_updates':250,
        'history':history,'actual_optimizer_step_counters':counters,'warmstarted':False,
        'initial_weights_sha256':tensor_digest(initial['model']),'initial_checkpoint_sha256':sha256(initial_path),
        'initial_training_state_sha256':initial_state_digest,'crop_schedule_sha256':fingerprint(schedule),
        'gain_schedule_sha256':fingerprint([row['gain_log'] for row in schedule]),
        'source_sha256':producer_hashes(),'config':config,'model_config':model_config,
        'normalization':audit['normalization'][arm['name']],'positive_weights':audit['positive_weights'],
        'model_parameters':sum(p.numel() for p in model.parameters()),'training_seconds':time.perf_counter()-started,
        'peak_gpu_mib':torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
        'early_stopping_used':False,'validation_used_during_training':False,'device':str(device),**provenance}
    checkpoint = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
        'scaler':scaler.state_dict(),'rng_cpu':torch.get_rng_state(),
        'rng_cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],'step':250,
        'receipt':receipt,'config':model_config,'normalization':receipt['normalization'],
        'positive_weights':audit['positive_weights']}
    save_checkpoint(directory/'final.pt',checkpoint)
    save_json(directory/'training-receipt.json',receipt)
    return receipt
