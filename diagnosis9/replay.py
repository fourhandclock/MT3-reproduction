import numpy as np

from streaming.inference import PCMStream
from .decoder import SmoothedDecoder


def replay(model,audio,normalization,model_config,parameters,device,packet_ms,mode):
    sample_rate=model_config['feature']['sample_rate'];packet=round(packet_ms*sample_rate/1000)
    if packet<=0:raise ValueError('Packet size must contain samples.')
    stream=PCMStream(model,normalization,model_config,parameters,device)
    if mode=='repair':stream.decoder=SmoothedDecoder(model_config,parameters)
    elif mode!='strict':raise ValueError('Expected strict or repair decoder.')
    for start in range(0,len(audio),packet):stream.push(audio[start:start+packet])
    stream.finish();times=np.asarray(stream.packet_times)
    timing={'processing_seconds':float(times.sum()),'input_seconds':len(audio)/sample_rate,
            'real_time_factor':float(times.sum())/(len(audio)/sample_rate) if len(audio) else None,
            'chunk_processing_p50_ms':float(np.quantile(times,.5)*1000) if len(times) else None,
            'chunk_processing_p95_ms':float(np.quantile(times,.95)*1000) if len(times) else None,
            'max_virtual_backlog_ms':max(stream.packet_backlogs or [0])*1000,
            'packet_ms':packet_ms,'lookahead_samples':0,'mode':mode,
            'timing_scope':'Warm mono16k PCM through unchanged causal STFT/normalization/network and selected incremental decoder; no file read/resampling or native capture.',
            'delivery_clock':'virtual finish=max(previous finish,packet arrival)+measured processing; source backdates do not affect delivery.'}
    return stream.notes,stream.events,timing
