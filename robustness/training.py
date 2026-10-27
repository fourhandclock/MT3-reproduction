"""One from-scratch augmentation trial, retaining the original optimizer and step cap."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from baseline.download import ROOT,sha256
from baseline.evaluation import evaluate,reference_notes
from cnn_gru.data import cache_path,fingerprint,validate_splits
from .data import AugmentedCropDataset
from .control import validate_prepared,source_hashes as robustness_source_hashes
from .validation import validate
from cnn_gru.inference import decode,predict_probabilities,tablature_metrics
from cnn_gru.model import GuitarCNNGRU,transcription_loss


def save_checkpoint(path,state):
    temporary=path.with_suffix('.tmp')
    torch.save(state,temporary)
    temporary.replace(path)


def train(config,output,device,resume=False):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    train=config['training']
    if train['num_workers']!=0:
        raise ValueError('This limited-RAM configuration requires num_workers=0.')
    manifest_path=ROOT/'data/cnn_gru/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    audit=json.loads((ROOT/'data/cnn_gru/cache_audit.json').read_text())
    manifest,audit=validate_prepared(config)
    validate_splits(manifest)
    evaluation_config=json.loads((ROOT/'configs/stage1.json').read_text())
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    torch.set_num_threads(train['cpu_threads'])
    model=GuitarCNNGRU(config).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=train['learning_rate'],weight_decay=train['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=train['max_steps'],eta_min=train['learning_rate']*0.1)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024,enabled=device.type=='cuda' and train['mixed_precision'])
    dataset=AugmentedCropDataset(manifest['train'],config,audit['normalization'])
    step,epoch_start,batch_start,best_score,bad_epochs=0,0,0,-1.0,0
    history=[];prior_seconds=0.0;epoch_sums={}
    source_hashes=robustness_source_hashes()
    last_path=output/'last.pt'
    if last_path.exists():
        if not resume:
            raise ValueError('Training checkpoint already exists. Use --resume or choose another output directory.')
        state=torch.load(last_path,map_location=device,weights_only=True)
        if state['config']!=config or state['source_sha256']!=source_hashes or state['manifest_sha256']!=sha256(manifest_path):
            raise ValueError('Resume requires the same configuration, source and dataset manifest.')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
        step,epoch_start,batch_start=state['step'],state['next_epoch'],state['next_batch']
        best_score,bad_epochs=state['best_score'],state['bad_epochs']
        history,prior_seconds,epoch_sums=state['history'],state['elapsed_seconds'],state['epoch_loss_sums']
        torch.set_rng_state(state['rng_cpu'].cpu())
        if device.type=='cuda': torch.cuda.set_rng_state_all([value.cpu() for value in state['rng_cuda']])
    elif resume:
        raise ValueError('No checkpoint to resume.')
    parameter_count=sum(parameter.numel() for parameter in model.parameters())
    print(f'Model: {parameter_count:,} parameters on {device}; budget {train["epochs"]} epochs / {train["max_steps"]} steps',flush=True)
    started=time.perf_counter()
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats()

    def checkpoint(next_epoch,next_batch):
        return {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'scaler':scaler.state_dict(),'config':config,'normalization':audit['normalization'],
                'positive_weights':audit['positive_weights'],'step':step,'next_epoch':next_epoch,'next_batch':next_batch,
                'best_score':best_score,'bad_epochs':bad_epochs,'history':history,'epoch_loss_sums':epoch_sums,
                'elapsed_seconds':prior_seconds+time.perf_counter()-started,
                'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                'manifest_sha256':sha256(manifest_path),'source_sha256':source_hashes,'parameter_count':parameter_count,
                'decoder_parameters':history[-1]['validation']['parameters'] if history else None}

    stop_reason='epoch_limit'
    for epoch in range(epoch_start,train['epochs']):
        if step>=train['max_steps'] or bad_epochs>=train['early_stopping_patience']:
            stop_reason='step_limit' if step>=train['max_steps'] else 'early_stopping'
            break
        dataset.epoch=epoch
        generator=torch.Generator().manual_seed(config['seed']+epoch)
        loader=torch.utils.data.DataLoader(dataset,batch_size=train['batch_size'],shuffle=True,num_workers=0,
                                            generator=generator,pin_memory=device.type=='cuda')
        if epoch!=epoch_start or batch_start==0: epoch_sums={}
        model.train()
        processed=batch_start if epoch==epoch_start else 0
        for batch_index,(features,targets) in enumerate(loader):
            if epoch==epoch_start and batch_index<batch_start: continue
            if step>=train['max_steps']: break
            features=features.to(device,non_blocking=True)
            targets={key:value.to(device,non_blocking=True) for key,value in targets.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,
                                enabled=device.type=='cuda' and train['mixed_precision']):
                logits=model(features)
                losses=transcription_loss(logits,targets,config,audit['positive_weights'])
            if not torch.isfinite(losses['total']): raise FloatingPointError('Non-finite training loss.')
            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),train['gradient_clip'])
            if not torch.isfinite(gradient): raise FloatingPointError('Non-finite gradients.')
            scaler.step(optimizer);scaler.update();scheduler.step()
            step+=1;processed+=1
            for key,value in losses.items(): epoch_sums[key]=epoch_sums.get(key,0.0)+float(value.detach())
            if step%25==0:
                print(f'Epoch {epoch+1}, step {step}/{train["max_steps"]}, loss {float(losses["total"]):.3f}',flush=True)
                save_checkpoint(last_path,checkpoint(epoch,batch_index+1))
        validation=validate(model,manifest,audit,config,evaluation_config,device)
        record={'epoch':epoch+1,'optimizer_steps':step,'training_losses':{key:value/max(processed,1) for key,value in epoch_sums.items()},
                'validation':validation,'elapsed_seconds':prior_seconds+time.perf_counter()-started}
        history.append(record)
        if validation['onset_f1']>best_score+1e-4:
            best_score=validation['onset_f1'];bad_epochs=0
            save_checkpoint(output/'best.pt',checkpoint(epoch+1,0))
        else: bad_epochs+=1
        (output/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print(f'Epoch {epoch+1}: train loss {record["training_losses"]["total"]:.3f}; '
              f'validation onset F1 {validation["onset_f1"]:.3f}; tab onset F1 {validation["tab_onset_f1"]:.3f}',flush=True)
        epoch_sums={};batch_start=0
        save_checkpoint(last_path,checkpoint(epoch+1,0))
        if step>=train['max_steps']: stop_reason='step_limit';break
        if bad_epochs>=train['early_stopping_patience']: stop_reason='early_stopping';break
    if not (output/'best.pt').exists(): raise RuntimeError('No validated checkpoint was produced.')
    summary={'parameter_count':parameter_count,'optimizer_steps':step,'epochs_completed':len(history),
             'best_validation_onset_f1':best_score,'stop_reason':stop_reason,'device':str(device),
             'training_and_validation_seconds':prior_seconds+time.perf_counter()-started,
             'gpu_peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None,
             'train_count':len(manifest['train']),'validation_count':len(manifest['validation']),
             'test_count':len(manifest['test']),'normalization_source':'train only',
             'test_used_for_model_selection':False,'config':config,'source_sha256':source_hashes,
             'manifest_sha256':sha256(manifest_path)}
    (output/'training_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('Saved best checkpoint and training summary:',output,flush=True)
    return summary
