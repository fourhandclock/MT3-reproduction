import time

import numpy as np

from .decoder import OnlineDecoder
from .frontend import CausalFrontend


def predict_features(model,x,normalization,config,device):
    import torch
    mean=np.asarray(normalization['mean'],np.float32);std=np.asarray(normalization['std'],np.float32)
    size=config['streaming']['model_feature_chunk_frames'];state=None;outputs={};model.eval()
    with torch.inference_mode():
        for start in range(0,len(x),size):
            values=np.clip((np.asarray(x[start:start+size],np.float32)-mean)/std,-5,5)
            tensor=torch.from_numpy(values[None]).to(device)
            logits,state=model.forward_chunk(tensor,state)
            for key,value in logits.items():
                probability=torch.softmax(value,dim=-1) if key=='state' else torch.sigmoid(value)
                outputs.setdefault(key,[]).append(probability[0].cpu().numpy())
    return {key:np.concatenate(value,axis=0) for key,value in outputs.items()}


def decode_probabilities(probabilities,config,parameters,duration,packet_ms=None):
    decoder=OnlineDecoder(config,parameters);length=len(probabilities['state'])
    size=max(1,round((packet_ms or config['streaming']['validation_packet_ms'])/1000*
                     config['feature']['sample_rate']/config['feature']['hop_length']))
    for start in range(0,length,size):
        end=min(length,start+size);availability=end*config['feature']['hop_length']/config['feature']['sample_rate']
        decoder.push({key:value[start:end] for key,value in probabilities.items()},availability)
    decoder.finish(duration,duration)
    return decoder.notes,decoder.events


class PCMStream:
    """A mono 16k PCM consumer. Source time and delivery time remain separate."""
    def __init__(self,model,normalization,config,parameters,device):
        import torch
        self.model=model.eval();self.config=config;self.device=torch.device(device);self.torch=torch
        self.frontend=CausalFrontend(config);self.decoder=OnlineDecoder(config,parameters)
        self.mean=np.asarray(normalization['mean'],np.float32);self.std=np.asarray(normalization['std'],np.float32)
        self.state=None;self._events=[];self.packet_times=[];self.packet_backlogs=[];self.virtual_finish=0.;self.finished=False
        self.peak_frontend_cache_samples=0

    @property
    def events(self):return [dict(event) for event in self._events]

    def push(self,samples):
        if self.finished:raise ValueError('Stream is finished; create a fresh PCMStream to reset.')
        torch=self.torch
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        started=time.perf_counter()
        x=self.frontend.push(samples)
        arrival=self.frontend.received_samples/self.config['feature']['sample_rate']
        new=[]
        if len(x):
            x=np.clip((x-self.mean)/self.std,-5,5)
            tensor=torch.from_numpy(x[None]).to(self.device)
            with torch.inference_mode():
                logits,self.state=self.model.forward_chunk(tensor,self.state)
                probabilities={key:(torch.softmax(value,dim=-1) if key=='state' else torch.sigmoid(value))[0].cpu().numpy()
                               for key,value in logits.items()}
            new=self.decoder.push(probabilities,arrival)
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        elapsed=time.perf_counter()-started
        self.virtual_finish=max(self.virtual_finish,arrival)+elapsed
        for event in new:event['delivered_at_seconds']=self.virtual_finish
        self._events.extend(dict(event) for event in new);self.packet_times.append(elapsed)
        self.packet_backlogs.append(max(0,self.virtual_finish-arrival))
        self.peak_frontend_cache_samples=max(self.peak_frontend_cache_samples,self.frontend.cached_samples)
        return [dict(event) for event in new]

    def finish(self):
        if self.finished:return []
        duration=self.frontend.received_samples/self.config['feature']['sample_rate'];started=time.perf_counter()
        new=self.decoder.finish(duration,duration)
        elapsed=time.perf_counter()-started;self.virtual_finish=max(self.virtual_finish,duration)+elapsed
        for event in new:event['delivered_at_seconds']=self.virtual_finish
        self._events.extend(dict(event) for event in new);self.finished=True
        deliveries={(event['note_id'],event['type']):event['delivered_at_seconds'] for event in self._events}
        self.notes=[]
        for note in self.decoder.notes:
            note=dict(note);note['on_delivered_at_seconds']=deliveries[(note['note_id'],'note_on')]
            note['off_delivered_at_seconds']=deliveries[(note['note_id'],'note_off')]
            self.notes.append(note)
        return [dict(event) for event in new]


def replay(model,audio,normalization,config,parameters,device,packet_ms):
    sample_rate=config['feature']['sample_rate'];packet=round(packet_ms/1000*sample_rate)
    if packet<=0:raise ValueError('Packet size must contain samples.')
    stream=PCMStream(model,normalization,config,parameters,device)
    for start in range(0,len(audio),packet):stream.push(audio[start:start+packet])
    stream.finish();times=np.asarray(stream.packet_times)
    timing={'processing_seconds':float(times.sum()),'input_seconds':len(audio)/sample_rate,
            'real_time_factor':float(times.sum())/(len(audio)/sample_rate) if len(audio) else None,
            'chunk_processing_p50_ms':float(np.quantile(times,.5)*1000) if len(times) else None,
            'chunk_processing_p95_ms':float(np.quantile(times,.95)*1000) if len(times) else None,
            'chunk_processing_max_ms':float(times.max()*1000) if len(times) else None,'packet_ms':packet_ms,
            'max_virtual_backlog_ms':max(stream.packet_backlogs or [0])*1000,
            'peak_frontend_cached_samples':stream.peak_frontend_cache_samples,'lookahead_samples':0,
            'timing_scope':'Warm PCM push: causal STFT, fixed normalization, GPU transfer/model, CPU probabilities and incremental note decoder. File read/live resampling/hardware capture excluded.',
            'delivery_clock':'Virtualfinish=max(previous_finish,packet_arrival_audio_time)+measured_push_time; unpaced replay, no hardware scheduling.'}
    return stream.notes,stream.events,timing
