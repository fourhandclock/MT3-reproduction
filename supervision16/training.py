"""Two capped continuations; the only treatment is a direct state-pitch term."""

import copy
import time

import numpy as np
import torch

from acoustic10.data import batch_from_schedule,prepare_targets
from baseline.download import ROOT,sha256
from cnn_gru.model import transcription_loss
from cnn_gru.training import save_checkpoint
from diagnosis15.loss import state_pitch_loss
from optimization13.optimizer import recursive_digest
from spectral11.training import actual_optimizer_steps,verified_sanity
from streaming.data import fingerprint
from streaming.evaluate import save_json
from .restore import restore_parent


def producer_hashes():
    paths=[ROOT/'configs/supervision16.json']+[ROOT/'supervision16'/name for name in
           ['__init__.py','data.py','restore.py','training.py','workflow.py']]
    paths += [ROOT/name for name in ['diagnosis15/loss.py','optimization13/optimizer.py','optimization13/data.py',
               'normalization12/model.py','spectral11/workflow.py','spectral11/training.py',
               'acoustic10/data.py','acoustic10/targets.py','acoustic10/training.py','acoustic10/decoder.py','acoustic10/workflow.py',
               'cnn_gru/data.py','cnn_gru/model.py','cnn_gru/training.py','cnn_gru/inference.py',
               'baseline/evaluation.py','diagnosis9/head_audit.py','diagnosis15/analysis.py']]
    paths+=list((ROOT/'streaming').glob('*.py'))
    return {path.relative_to(ROOT).as_posix():sha256(path) for path in sorted(set(paths))}


def optimize(model,x,target,optimizer,scaler,model_config,weights,spec,device,coefficient):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda' and spec['mixed_precision']):
        outputs=model(x)
        losses=transcription_loss(outputs,target,model_config,weights)
    if coefficient:
        # Probability-space BCE is explicitly outside CUDA autocast; the frozen
        # Stage15 helper softmaxes in float32 and preserves gradients to logits.
        with torch.autocast(device_type=device.type,enabled=False):
            extra=state_pitch_loss(outputs['state'],target)
        losses={**losses,'state_pitch':extra,'total':losses['total']+coefficient*extra}
    if not torch.isfinite(losses['total']):
        raise FloatingPointError('Nonfinite training loss; no further updates allowed.')
    scaler.scale(losses['total']).backward()
    scaler.unscale_(optimizer)
    gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),spec['gradient_clip'])
    if not torch.isfinite(gradient):
        raise FloatingPointError('Nonfinite gradient; no replacement updates allowed.')
    scaler.step(optimizer)
    scaler.update()
    result={key:float(value.detach()) for key,value in losses.items()}
    result['new_term_valid_frames']=int(((target['mask']>0)&torch.all(target['state']!=-100,dim=-1)).sum())
    return result,float(gradient)


def train_arm(arm,config,base,audit,records,provenance,parent,schedule,output,device):
    name=arm['name']
    directory=output/'models'/name
    directory.mkdir(parents=True,exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError('This arm already started; its update budget cannot be repeated.')
    model,optimizer,scaler,proof=restore_parent(parent,device)
    snapshot={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':500,
              'config':parent['config'],'normalization':parent['normalization'],'positive_weights':parent['positive_weights'],
              'scaler':scaler.state_dict(),'rng_cpu':torch.get_rng_state(),
              'rng_cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],'restoration_proof':proof}
    resume_path=directory/'resume-state.pt'
    save_checkpoint(resume_path,snapshot)
    del snapshot
    support=fingerprint([{'id':row['entry']['id'],'frames':len(row['features']),
                         'ignored_global_leading_frames':26,'ignored_crop_edge_frames':26,'target_delay_frames':13}
                        for row in records])
    if support!=parent['receipt']['common_support_sha256']:
        raise ValueError('Training source/target support differs from the shared parent.')
    spec=config['training']
    model_config=copy.deepcopy(parent['config'])
    model_config['training'].update(spec)
    model_config['stage_budget']={'new_total_cap':500,'new_per_arm':250,'shared_parent_updates':500,
                                  'final_lifetime_updates':750,'new_sanity_updates':0,'purpose':'Stage16 loss-only comparison.'}
    targets=prepare_targets(records,13,config,base)
    history,block=[],[]
    started=time.perf_counter()
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for new_step in range(1,251):
        x,target=batch_from_schedule(schedule[(new_step-1)*4:new_step*4],records,targets,audit['normalization'],26)
        x=x.to(device)
        target={key:value.to(device) for key,value in target.items()}
        losses,gradient=optimize(model,x,target,optimizer,scaler,model_config,audit['positive_weights'],spec,device,
                                 arm['state_pitch_coefficient'])
        lifetime=500+new_step
        counters=actual_optimizer_steps(optimizer)
        if counters['minimum']!=lifetime or counters['maximum']!=lifetime or counters['parameter_states']!=34:
            raise ValueError('Actual AdamW counters differ from the fixed budget; no replacement updates allowed.')
        if any(group['lr']!=.0001 for group in optimizer.param_groups):
            raise ValueError('Stored-end learning rate changed.')
        block.append(losses)
        if new_step%50==0:
            history.append({'additional_epoch':new_step//50,'data_epoch':10+new_step//50,
                'new_optimizer_steps':new_step,'optimizer_steps':lifetime,'actual_optimizer_step_counters':counters,
                'training_losses':{key:float(np.mean([row[key] for row in block])) for key in losses},
                'gradient_norm':gradient,'learning_rate':.0001,'scaler_scale':scaler.get_scale()})
            block=[]
            receipt={'name':name,'state_pitch_coefficient':arm['state_pitch_coefficient'],
                'state_pitch_loss_computed':bool(arm['state_pitch_coefficient']),
                'parent_optimizer_steps':500,'new_optimizer_steps':new_step,'optimizer_steps':lifetime,'selected_step':750,
                'actual_optimizer_step_counters':counters,'additional_epochs_completed':len(history),'history':history,
                'parent_reference':provenance['parent'],'restoration_proof':proof,
                'resume_state_path':resume_path.relative_to(output).as_posix(),'resume_state_sha256':sha256(resume_path),
                'historical_rng_restored':True,'historical_scaler_restored':True,'scheduler_used':False,'learning_rate':.0001,
                'crop_schedule_sha256':fingerprint(schedule),'gain_schedule_sha256':fingerprint([r['gain_log'] for r in schedule]),
                'common_support_sha256':support,'delay_frames':13,'common_leading_mask_frames':26,'crop_edge_mask_frames':26,
                'config_sha256':fingerprint(config),'source_sha256':producer_hashes(),'model_config':model_config,
                'normalization':audit['normalization'],'positive_weights':audit['positive_weights'],
                'model_parameters':sum(p.numel() for p in model.parameters()),'early_stopping_used':False,
                'validation_used_during_training':False,'test_inference_calls':0,'device':str(device),
                'training_seconds':time.perf_counter()-started,
                'peak_gpu_mib':torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,**provenance}
            checkpoint={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':lifetime,'receipt':receipt,
                'config':model_config,'normalization':audit['normalization'],'positive_weights':audit['positive_weights'],
                'scaler':scaler.state_dict(),'rng_cpu':torch.get_rng_state(),
                'rng_cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
            save_checkpoint(directory/'last.pt',checkpoint)
            save_json(directory/'training-receipt.json',receipt)
            print(name,'new updates',new_step,'lifetime',lifetime,'state loss',round(history[-1]['training_losses']['state'],3),flush=True)
    save_checkpoint(directory/'final.pt',checkpoint)
    for key,digest in proof['parent_state_sha256'].items():
        if recursive_digest(parent[key])!=digest:
            raise ValueError('Continuation mutated the in-memory shared parent: '+key)
    return receipt
