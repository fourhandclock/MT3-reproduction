"""Same restored state and crop schedule; only the auxiliary fret term differs."""

import copy
import time
import numpy as np
import torch
from acoustic10.data import prepare_targets
from baseline.download import ROOT,sha256
from cnn_gru.model import transcription_loss
from cnn_gru.training import save_checkpoint
from multires17.data import batch_from_schedule
from optimization13.optimizer import recursive_digest
from spectral11.training import actual_optimizer_steps,verified_sanity
from streaming.data import fingerprint
from streaming.evaluate import save_json
from .loss import conditional_fret_loss,active_support
from .restore import restore_parent


def producer_hashes():
    paths=[ROOT/'fret19'/name for name in ['__init__.py','__main__.py','data.py','loss.py','restore.py','training.py','workflow.py']]
    paths+=[ROOT/name for name in ['configs/fret19.json','multires17/model.py','multires17/frontend.py',
        'multires17/data.py','multires17/workflow.py','optimization13/data.py','optimization13/optimizer.py',
        'acoustic10/data.py','acoustic10/targets.py','acoustic10/training.py','acoustic10/decoder.py','acoustic10/workflow.py',
        'cnn_gru/data.py','cnn_gru/model.py','cnn_gru/training.py','cnn_gru/inference.py','baseline/download.py',
        'baseline/evaluation.py','fingering/export.py','spectral11/workflow.py','spectral11/training.py',
        'headswap18/analysis.py','diagnosis15/analysis.py','diagnosis9/head_audit.py','decoder14/core.py','decoder14/workflow.py']]
    paths+=list((ROOT/'streaming').glob('*.py'))
    return {p.relative_to(ROOT).as_posix():sha256(p) for p in sorted(set(paths))}


def optimize(model,x,target,optimizer,scaler,model_config,weights,spec,device,coefficient):
    model.train();optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda' and spec['mixed_precision']):
        outputs=model(x);losses=transcription_loss(outputs,target,model_config,weights)
    valid=int(active_support(outputs['state'],target).sum())
    if coefficient:
        extra=conditional_fret_loss(outputs['state'],target)
        losses={**losses,'conditional_fret':extra,'total':losses['total']+coefficient*extra}
    if not torch.isfinite(losses['total']):raise FloatingPointError('Nonfinite loss; no replacement updates')
    scaler.scale(losses['total']).backward();scaler.unscale_(optimizer)
    gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),spec['gradient_clip'])
    if not torch.isfinite(gradient):raise FloatingPointError('Nonfinite gradient; no replacement updates')
    scaler.step(optimizer);scaler.update()
    return {k:float(v.detach()) for k,v in losses.items()},float(gradient),valid


def train_arm(arm,config,base,audit,records,provenance,parent,schedule,output,device):
    directory=output/'models'/arm['name'];directory.mkdir(parents=True,exist_ok=True)
    if any(directory.iterdir()):raise ValueError('This arm already started; its budget cannot be repeated')
    model,optimizer,scaler,proof=restore_parent(parent,device)
    snapshot={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
        'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
        'step':250,'config':parent['config'],'normalization':parent['normalization'],'positive_weights':parent['positive_weights'],
        'source_scheduler':copy.deepcopy(parent['scheduler']),'restoration_proof':proof}
    save_checkpoint(directory/'resume-state.pt',snapshot)
    spec=config['training'];model_config=copy.deepcopy(parent['config']);model_config['training'].update(spec)
    model_config['stage_budget']={'total_max_new_optimizer_updates':500,'new_arm_updates':250,'shared_parent_updates':250,
                                 'final_lifetime_updates':500,'new_sanity_updates':0,'purpose':'Stage19 conditional-fret loss'}
    targets=prepare_targets(records,13,config,base);history=[];loss_rows=[];valid_counts=[]
    if device.type=='cuda':torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
    started=time.perf_counter()
    for new_step in range(1,251):
        x,target=batch_from_schedule(schedule[(new_step-1)*4:new_step*4],records,targets,audit['normalization'],26,{'name':'short_short'})
        losses,gradient,valid=optimize(model,x.to(device),{k:v.to(device) for k,v in target.items()},optimizer,scaler,
                model_config,audit['positive_weights'],spec,device,arm['conditional_fret_coefficient'])
        step=250+new_step;counters=actual_optimizer_steps(optimizer)
        if counters!={'minimum':step,'maximum':step,'parameter_states':34}:raise ValueError('Actual optimizer counters differ; stop without replacements')
        loss_rows.append(losses);valid_counts.append(valid)
        if new_step%50==0:
            history.append({'data_epoch':5+new_step//50,'new_optimizer_steps':new_step,'optimizer_steps':step,
                'actual_optimizer_step_counters':counters,'learning_rate':optimizer.param_groups[0]['lr'],
                'training_losses':{k:float(np.mean([r[k] for r in loss_rows])) for k in losses},
                'auxiliary_valid_active_string_frames':sum(valid_counts),'gradient_norm':gradient})
            loss_rows=[];valid_counts=[]
            print('Stage19',arm['name'],'epoch',history[-1]['data_epoch'],'new update',new_step,
                  'state loss',round(history[-1]['training_losses']['state'],4),flush=True)
    if device.type=='cuda':torch.cuda.synchronize(device)
    receipt={'name':arm['name'],'new_optimizer_steps':250,'parent_optimizer_steps':250,'optimizer_steps':500,
        'selected_step':500,'history':history,'actual_optimizer_step_counters':counters,
        'conditional_fret_coefficient':arm['conditional_fret_coefficient'],'restoration_proof':proof,
        'resume_state_sha256':sha256(directory/'resume-state.pt'),'source_sha256':producer_hashes(),
        'crop_schedule_sha256':fingerprint(schedule),'gain_schedule_sha256':fingerprint([r['gain_log'] for r in schedule]),
        'config_sha256':fingerprint(config),'config':config,'model_config':model_config,
        'normalization':audit['normalization'],'positive_weights':audit['positive_weights'],
        'model_parameters':sum(p.numel() for p in model.parameters()),'scheduler_used':False,
        'early_stopping_used':False,'validation_used_during_training':False,'test_inference_calls':0,
        'device':str(device),'training_seconds':time.perf_counter()-started,
        'peak_gpu_mib':torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,**provenance}
    checkpoint={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
        'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
        'step':500,'config':model_config,'normalization':audit['normalization'],'positive_weights':audit['positive_weights'],
        'source_scheduler':copy.deepcopy(parent['scheduler']),'scheduler_used':False,'receipt':receipt}
    save_checkpoint(directory/'final.pt',checkpoint);save_json(directory/'training-receipt.json',receipt)
    return receipt
