import numpy as np

from baseline.evaluation import match_notes


def quantiles(values):
    return {'p50':float(np.quantile(values,.5)),'p95':float(np.quantile(values,.95))} if values else {'p50':None,'p95':None}


def measure(reference,predicted,score_config,duration):
    score_end=min(duration,score_config['score_seconds'])
    ref=[n for n in reference if 0<=n['start']<score_end]
    est=[n for n in predicted if 0<=n['start']<score_end]
    pairs=match_notes(ref,est,score_config);onsets=[];source_audio=[]
    for i,j in pairs:
        if 'on_delivered_at_seconds' not in est[j]:continue
        onsets.append(1000*(est[j]['on_delivered_at_seconds']-ref[i]['start']))
        source_audio.append(1000*(est[j]['on_emitted_at_audio_seconds']-ref[i]['start']))
    censored_ref={i for i,n in enumerate(ref) if n['end']>=duration}
    removed_est={j for i,j in pairs if i in censored_ref}
    ref_offsets=[n for i,n in enumerate(ref) if i not in censored_ref]
    est_offsets=[n for j,n in enumerate(est) if j not in removed_est and not n.get('censored_at_end',False)]
    offsets=[]
    for i,j in match_notes(ref_offsets,est_offsets,score_config,offsets=True):
        if 'off_delivered_at_seconds' in est_offsets[j]:offsets.append(1000*(est_offsets[j]['off_delivered_at_seconds']-ref_offsets[i]['end']))
    on=quantiles(onsets);off=quantiles(offsets)
    return {'matched_onset_count':len(onsets),'matched_offset_count':len(offsets),
            'matched_onset_delivery_p50_ms':on['p50'],'matched_onset_delivery_p95_ms':on['p95'],
            'matched_offset_delivery_p50_ms':off['p50'],'matched_offset_delivery_p95_ms':off['p95'],
            'matched_onset_packet_delivery_p50_ms':quantiles(source_audio)['p50'],
            'onset_delivery_samples_ms':onsets,'offset_delivery_samples_ms':offsets,
            'negative_onset_delivery_count':sum(v<0 for v in onsets),
            'censored_terminal_release_count':sum(n.get('censored_at_end',False) for n in est),
            'definition':'Signed delivery minus performed reference event, only one-to-one correctly matched notes. Offsets require onset+offset match and observed release; EOF flush excluded.'}
