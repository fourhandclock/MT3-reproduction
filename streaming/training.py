import json
import time
from pathlib import Path

import numpy as np
import torch

from baseline.download import ROOT,sha256
from baseline.evaluation import evaluate,reference_notes
from cnn_gru.model import transcription_loss
from cnn_gru.training import save_checkpoint
from .data import CropDataset,cache_path,fingerprint
from .inference import decode_probabilities,predict_features
from .model import CausalCNNGRU


def validate(model,manifest,audit,config,score_config,device):
    dataset=[]
    for entry in manifest['validation']:
        prefix=cache_path(entry,config,config['evaluation_audio_seconds']);x=np.load(prefix.with_suffix('.npy'),mmap_mode='r')
        probability=predict_features(model,x,audit['normalization'],config,device)
        dataset.append((probability,reference_notes(ROOT/entry['annotation']),json.loads(prefix.with_suffix('.json').read_text())['duration']))
    best=None
    for onset in config['decoder']['onset_thresholds']:
        for activity in config['decoder']['activity_thresholds']:
            for backdate in config['decoder']['onset_backdates_seconds']:
                parameters={'onset_threshold':onset,'activity_threshold':activity,'onset_backdate_seconds':backdate}
                rows=[]
                for probability,reference,duration in dataset:
                    notes,_=decode_probabilities(probability,config,parameters,duration)
                    rows.append(evaluate(reference,notes,score_config,duration))
                candidate={'onset_f1':float(np.mean([r['onset']['f1'] for r in rows])),
                           'onset_offset_f1':float(np.mean([r['onset_offset']['f1'] for r in rows])),
                           'parameters':parameters}
                if best is None or (candidate['onset_f1'],candidate['onset_offset_f1'])>(best['onset_f1'],best['onset_offset_f1']):best=candidate
    return best


def train(config,output,device,resume=False):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);spec=config['training']
    if spec['epochs']>5 or spec['max_steps']>500 or spec['num_workers']!=0:
        raise ValueError('This stage is capped at5 epochs/500 updates and zero worker processes.')
    manifest_path=ROOT/'data/cnn_gru/manifest.json';manifest=json.loads(manifest_path.read_text())
    audit_path=ROOT/'data/streaming/cache_audit.json';audit=json.loads(audit_path.read_text())
    if audit['config_sha256']!=fingerprint(config) or audit['manifest_sha256']!=sha256(manifest_path):raise ValueError('Prepared configuration changed.')
    for split in ['train','validation','test']:
        for entry,info in zip(manifest[split],audit['splits'][split]):
            for key in ['audio','annotation']:
                if sha256(ROOT/entry[key])!=entry[key+'_sha256']:raise ValueError('Dataset source changed.')
            seconds=config['train_audio_seconds'] if split=='train' else config['evaluation_audio_seconds']
            prefix=cache_path(entry,config,seconds)
            if sha256(prefix.with_suffix('.npy'))!=info['features_sha256'] or sha256(prefix.with_suffix('.npz'))!=info['targets_sha256']:
                raise ValueError('Prepared causal features changed.')
    torch.manual_seed(config['seed']);np.random.seed(config['seed']);torch.set_num_threads(spec['cpu_threads'])
    model=CausalCNNGRU(config).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=spec['max_steps'],eta_min=spec['learning_rate']*.1)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024,enabled=device.type=='cuda' and spec['mixed_precision'])
    dataset=CropDataset(manifest['train'],config,audit['normalization']);score_config=json.loads((ROOT/'configs/stage1.json').read_text())
    hashes={name:sha256(ROOT/'streaming'/name) for name in ['frontend.py','model.py','decoder.py','targets.py','data.py','inference.py','training.py']}
    step=epoch_start=batch_start=bad_epochs=0;best_score=-1.;history=[];prior_seconds=0.;epoch_sums={}
    last=output/'last.pt'
    if last.exists():
        if not resume:raise ValueError('Training already exists; use --resume within the remaining budget or a fresh output directory.')
        state=torch.load(last,map_location=device,weights_only=True)
        if state['config']!=config or state['source_sha256']!=hashes or state['cache_audit_sha256']!=sha256(audit_path):raise ValueError('Resume identities changed.')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
        step,epoch_start,batch_start=state['step'],state['next_epoch'],state['next_batch'];bad_epochs=state['bad_epochs'];best_score=state['best_score']
        history=state['history'];prior_seconds=state['elapsed_seconds'];epoch_sums=state['epoch_loss_sums']
        torch.set_rng_state(state['rng_cpu'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([r.cpu() for r in state['rng_cuda']])
    elif resume:raise ValueError('No checkpoint to resume.')
    parameters=sum(p.numel() for p in model.parameters());started=time.perf_counter()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    print('Causal model:',parameters,'parameters;',device,'budget',spec['epochs'],'epochs/',spec['max_steps'],'updates',flush=True)
    def checkpoint(next_epoch,next_batch):
        return {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
                'config':config,'normalization':audit['normalization'],'positive_weights':audit['positive_weights'],'step':step,
                'next_epoch':next_epoch,'next_batch':next_batch,'bad_epochs':bad_epochs,'best_score':best_score,'history':history,
                'elapsed_seconds':prior_seconds+time.perf_counter()-started,'epoch_loss_sums':epoch_sums,
                'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                'manifest_sha256':sha256(manifest_path),'cache_audit_sha256':sha256(audit_path),'source_sha256':hashes,
                'model_parameters':parameters,'scoring_config':score_config,
                'legacy_loss_sha256':sha256(ROOT/'cnn_gru/model.py'),'decoder_parameters':history[-1]['validation']['parameters'] if history else None}
    reason='epoch_limit'
    for epoch in range(epoch_start,spec['epochs']):
        if step>=spec['max_steps'] or bad_epochs>=spec['early_stopping_patience']:break
        dataset.epoch=epoch;generator=torch.Generator().manual_seed(config['seed']+epoch)
        loader=torch.utils.data.DataLoader(dataset,batch_size=spec['batch_size'],shuffle=True,num_workers=0,generator=generator,pin_memory=device.type=='cuda')
        if epoch!=epoch_start or batch_start==0:epoch_sums={}
        model.train();processed=batch_start if epoch==epoch_start else 0
        for batch_index,(x,targets) in enumerate(loader):
            if epoch==epoch_start and batch_index<batch_start:continue
            if step>=spec['max_steps']:break
            x=x.to(device);targets={k:v.to(device) for k,v in targets.items()};optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda' and spec['mixed_precision']):
                loss=transcription_loss(model(x),targets,config,audit['positive_weights'])
            if not torch.isfinite(loss['total']):raise FloatingPointError('Nonfinite causal training loss.')
            scaler.scale(loss['total']).backward();scaler.unscale_(optimizer)
            gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),spec['gradient_clip'])
            if not torch.isfinite(gradient):raise FloatingPointError('Nonfinite causal gradients.')
            scaler.step(optimizer);scaler.update();scheduler.step();step+=1;processed+=1
            for key,value in loss.items():epoch_sums[key]=epoch_sums.get(key,0.)+float(value.detach())
            if step%25==0:
                save_checkpoint(last,checkpoint(epoch,batch_index+1));print('Causal epoch',epoch+1,'step',step,'loss',round(float(loss['total']),3),flush=True)
        validation=validate(model,manifest,audit,config,score_config,device)
        history.append({'epoch':epoch+1,'optimizer_steps':step,'training_losses':{k:v/max(1,processed) for k,v in epoch_sums.items()},
                        'validation':validation,'elapsed_seconds':prior_seconds+time.perf_counter()-started})
        if validation['onset_f1']>best_score+1e-4:
            best_score=validation['onset_f1'];bad_epochs=0;save_checkpoint(output/'best.pt',checkpoint(epoch+1,0))
        else:bad_epochs+=1
        epoch_sums={};batch_start=0;save_checkpoint(last,checkpoint(epoch+1,0))
        (output/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print('Causal epoch',epoch+1,'validation onset F1',round(validation['onset_f1'],3),validation['parameters'],flush=True)
        if step>=spec['max_steps']:reason='step_limit';break
        if bad_epochs>=spec['early_stopping_patience']:reason='early_stopping';break
    if not (output/'best.pt').exists():raise ValueError('No validated causal checkpoint.')
    summary={'model_parameters':parameters,'optimizer_steps':step,'epochs_completed':len(history),'stop_reason':reason,
             'best_validation_onset_f1':best_score,'training_and_validation_seconds':prior_seconds+time.perf_counter()-started,
             'peak_gpu_mib':torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
             'normalization_source':'train only','test_used_for_selection':False,'source_sha256':hashes,'config':config}
    (output/'training_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');return summary
