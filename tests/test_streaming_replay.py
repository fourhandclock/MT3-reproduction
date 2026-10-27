import json

import numpy as np
import pytest
import torch

from streaming.inference import PCMStream,replay
from streaming.latency import measure


class ScriptedModel:
    def eval(self):return self

    def forward_chunk(self,x,state=None):
        start=state['frames'] if state else 0;t=x.shape[1]
        states=torch.full((1,t,6,26),-10.,device=x.device);states[:,:,:,25]=10.;states[:,:,0,25]=-10.;states[:,:,0,0]=10.
        onset=torch.full((1,t,6),-10.,device=x.device)
        for i in range(t):
            if start+i==19:onset[:,i,0]=10.
        return {'state':states,'onset':onset,'offset':torch.zeros_like(onset),'pitch':torch.zeros((1,t,49),device=x.device)}, {'frames':start+t}


@pytest.fixture
def setup():
    config=json.load(open('configs/streaming.json'))
    norm={'mean':[0.]*216,'std':[1.]*216}
    params={'activity_threshold':.6,'onset_threshold':.4,'onset_backdate_seconds':0}
    return config,norm,params


def test_end_to_end_pcm_causal_events_and_eof_retrigger(setup):
    config,norm,params=setup
    audio=np.sin(np.arange(3200)*.05).astype(np.float32)
    notes,events,timing=replay(ScriptedModel(),audio,norm,config,params,torch.device('cpu'),20)
    assert len(notes)==1 and len(events)==2
    assert notes[0]['start']==pytest.approx(.01) and notes[0]['end']==pytest.approx(.2)
    assert notes[0]['censored_at_end']
    assert events[0]['emitted_at_audio_seconds']==pytest.approx(.04)
    assert all(e['delivered_at_seconds']>=e['emitted_at_audio_seconds'] for e in events)
    assert timing['lookahead_samples']==0 and timing['peak_frontend_cached_samples']<2048


def test_pcm_prefix_published_events_do_not_use_future_and_partial_hop(setup):
    config,norm,params=setup
    stream=PCMStream(ScriptedModel(),norm,config,params,torch.device('cpu'))
    for _ in range(4):stream.push(np.zeros(272,np.float32))
    before=[dict(e) for e in stream.events]
    assert before and before[0]['emitted_at_audio_seconds']>=.04
    stream.push(np.ones(979,np.float32));stream.finish()
    assert stream.events[:len(before)]==before
    assert stream.frontend.frames_emitted==2067//160
    assert stream.notes[0]['end']==pytest.approx(2067/16000)
    with pytest.raises(ValueError):stream.push(np.zeros(160,np.float32))


def test_latency_matching_is_conditional_and_excludes_terminal_flush():
    scoring=json.load(open('configs/stage1.json'))
    reference=[{'pitch':40,'start':.12,'end':.4}]
    note={'pitch':40,'start':.12,'end':.4,'on_delivered_at_seconds':.2,'off_delivered_at_seconds':.48,
          'on_emitted_at_audio_seconds':.18,'censored_at_end':False}
    metrics=measure(reference,[note],scoring,1)
    assert metrics['matched_onset_count']==metrics['matched_offset_count']==1
    assert metrics['matched_onset_delivery_p50_ms']==pytest.approx(80)
    assert metrics['matched_offset_delivery_p95_ms']==pytest.approx(80)
    note['censored_at_end']=True
    assert measure(reference,[note],scoring,1)['matched_offset_count']==0
    note['start']=.8
    missing=measure(reference,[note],scoring,1)
    assert missing['matched_onset_count']==0 and missing['matched_onset_delivery_p50_ms'] is None


def test_pcm_event_copies_and_empty_replay(setup):
    config,norm,params=setup
    stream=PCMStream(ScriptedModel(),norm,config,params,torch.device('cpu'))
    published=stream.push(np.zeros(640,np.float32));published[0]['note_id']=999
    copy=stream.events;copy[0]['note_id']=888
    stream.finish();assert stream.notes[0]['note_id']==0
    notes,events,timing=replay(ScriptedModel(),np.empty(0,np.float32),norm,config,params,torch.device('cpu'),20)
    assert not notes and not events and timing['real_time_factor'] is None


def test_latency_does_not_match_unheard_future_annotations():
    scoring=json.load(open('configs/stage1.json'))
    reference=[{'pitch':40,'start':.22,'end':.4}]
    prediction={'pitch':40,'start':.18,'end':.2,'on_delivered_at_seconds':.2,'on_emitted_at_audio_seconds':.2,
                'off_delivered_at_seconds':.2,'censored_at_end':True}
    assert measure(reference,[prediction],scoring,.2)['matched_onset_count']==0
