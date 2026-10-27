"""Two-input-channel causal CNN-GRU; both arms use this exact network."""

import torch
from torch import nn
from torch.nn import functional as F
from streaming.model import CausalCNNGRU as SinglePlaneModel


class CausalCNNGRU(SinglePlaneModel):
    def __init__(self, config):
        super().__init__(config)
        first = self.convolutions[0]
        self.convolutions[0] = nn.Conv2d(2, first.out_channels, 3, padding=(0, 1))

    def _forward_chunk(self, features, state, detach_state):
        if features.ndim != 4 or features.shape[2:] != (2, self.n_bins):
            raise ValueError('Expected [batch, time, 2, n_bins] features')
        batch, time = features.shape[:2]
        if time == 0:
            shapes = {'state': (batch, 0, 6, self.fret_classes),
                      'onset': (batch, 0, 6), 'offset': (batch, 0, 6),
                      'pitch': (batch, 0, self.pitch_count)}
            return {key: features.new_empty(shape) for key, shape in shapes.items()}, state
        contexts = state['conv_contexts'] if state is not None else None
        if contexts is not None and len(contexts) != len(self.convolutions):
            raise ValueError('Wrong number of convolution caches')
        hidden = features.permute(0, 2, 1, 3)
        next_contexts = []
        for index, (convolution, normalization) in enumerate(zip(self.convolutions, self.normalizations)):
            if contexts is None:
                context = hidden.new_zeros((batch, hidden.shape[1], 2, hidden.shape[-1]))
            else:
                context = contexts[index]
                if tuple(context.shape) != (batch, hidden.shape[1], 2, hidden.shape[-1]) or context.device != hidden.device:
                    raise ValueError('State shape/device differs; reset for a new stream')
                context = context.to(dtype=hidden.dtype)
            padded = torch.cat((context, hidden), dim=2)
            tail = padded[:, :, -2:, :]
            next_contexts.append(tail.detach().clone() if detach_state else tail)
            hidden = F.max_pool2d(F.gelu(normalization(convolution(padded))), (1, 2))
        hidden = self.projection(hidden.permute(0, 2, 1, 3).flatten(start_dim=2))
        recurrent = state['gru_hidden'] if state is not None else None
        with torch.autocast(device_type=features.device.type, enabled=False):
            hidden, recurrent = self.gru(hidden.float(), recurrent)
        hidden = self.output_projection(hidden)
        outputs = {'state': self.state_head(hidden).reshape(batch, time, 6, self.fret_classes),
                   'onset': self.onset_head(hidden), 'offset': self.offset_head(hidden),
                   'pitch': self.pitch_head(hidden)}
        return outputs, {'conv_contexts': tuple(next_contexts),
                         'gru_hidden': recurrent.detach() if detach_state else recurrent}
