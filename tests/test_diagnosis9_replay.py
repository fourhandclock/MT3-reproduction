"""PCM integration checks for the optional causal smoothing recipe."""

import json

import numpy as np
import torch

from diagnosis9.replay import replay


class FrameScript:
    def eval(self):
        return self

    def forward_chunk(self, x, state=None):
        start = state or 0
        count = x.shape[1]
        values = torch.full((1, count, 6, 26), -10., device=x.device)
        values[:, :, :, 25] = 10.
        for i in range(count):
            frame = start + i
            if frame < 20:
                values[:, i, 0, 25] = -10.
                values[:, i, 0, 5 if frame < 8 or frame >= 10 else 7] = 10.
        onset = torch.full((1, count, 6), -10., device=x.device)
        return {'state': values, 'onset': onset, 'offset': onset.clone(),
                'pitch': torch.zeros((1, count, 49), device=x.device)}, start + count


def source_only(notes):
    return [{k: v for k, v in note.items() if 'emitted_at' not in k and 'delivered_at' not in k}
            for note in notes]


def test_pcm_identity_and_smoothed_intervals_survive_packet_boundaries():
    config = json.load(open('configs/streaming.json'))
    norm = {'mean': [0.] * 216, 'std': [1.] * 216}
    parameters = {'onset_threshold': .55, 'activity_threshold': .35,
                  'onset_backdate_seconds': .06, 'confirm_frames': 4, 'state_alpha': 1.}
    audio = np.zeros(4067, np.float32)
    strict, strict_events, _ = replay(FrameScript(), audio, norm, config, parameters, 'cpu', 20, 'strict')
    identity, identity_events, _ = replay(FrameScript(), audio, norm, config, parameters, 'cpu', 20, 'repair')
    assert source_only(identity) == source_only(strict)
    assert [{k: v for k, v in e.items() if k != 'delivered_at_seconds'} for e in identity_events] == [
        {k: v for k, v in e.items() if k != 'delivered_at_seconds'} for e in strict_events]
    parameters['state_alpha'] = .5
    results = [replay(FrameScript(), audio, norm, config, parameters, 'cpu', packet, 'repair')
               for packet in [20, 40, 80]]
    assert results[0][0]
    assert source_only(results[0][0]) == source_only(results[1][0]) == source_only(results[2][0])
    for notes, events, timing in results:
        assert len(events) == 2 * len(notes)
        assert all(e['delivered_at_seconds'] >= e['emitted_at_audio_seconds'] >= e['source_time']
                   for e in events)
        assert timing['lookahead_samples'] == 0


def test_empty_smoothed_pcm_replay_keeps_latency_unavailable():
    config = json.load(open('configs/streaming.json'))
    norm = {'mean': [0.] * 216, 'std': [1.] * 216}
    parameters = {'onset_threshold': .55, 'activity_threshold': .35,
                  'onset_backdate_seconds': .06, 'confirm_frames': 4, 'state_alpha': .5}
    notes, events, timing = replay(FrameScript(), np.empty(0, np.float32), norm, config,
                                   parameters, 'cpu', 20, 'repair')
    assert notes == events == []
    assert timing['real_time_factor'] is None
    assert timing['chunk_processing_p50_ms'] is None
