import math

import torch
from torch import nn

from cnn_gru.model import GuitarCNNGRU


class GuitarCNNTransformer(GuitarCNNGRU):
    def __init__(self,config):
        # Construct the shared modules in the same RNG order as the GRU baseline,
        # then replace only its temporal module. No trained GRU weights are used.
        super().__init__(config)
        del self.gru
        specification=config['transformer']
        width=config['model']['projection_size']
        layer=nn.TransformerEncoderLayer(d_model=width,nhead=specification['heads'],
                    dim_feedforward=specification['feedforward_size'],dropout=config['model']['dropout'],
                    activation='gelu',batch_first=True,norm_first=True)
        self.temporal=nn.TransformerEncoder(layer,num_layers=specification['layers'],
                                            norm=nn.LayerNorm(width),enable_nested_tensor=False)
        # TransformerEncoder clones layers; initialize each clone independently.
        def reset(module):
            if isinstance(module,(nn.Linear,nn.LayerNorm)):
                module.reset_parameters()
            elif isinstance(module,nn.MultiheadAttention):
                module._reset_parameters()
        for block in self.temporal.layers:
            block.apply(reset)
        head_width=config['model']['gru_hidden_size']*(2 if config['model']['bidirectional'] else 1)
        self.temporal_output=nn.Linear(width,head_width)
        positions=torch.arange(specification['max_frames'],dtype=torch.float32)[:,None]
        rates=torch.exp(torch.arange(0,width,2,dtype=torch.float32)*(-math.log(10000.0)/width))
        encoding=torch.zeros(specification['max_frames'],width,dtype=torch.float32)
        encoding[:,0::2]=torch.sin(positions*rates)
        encoding[:,1::2]=torch.cos(positions*rates)
        self.register_buffer('position_encoding',encoding[None])

    def forward(self,features):
        if features.shape[1]>self.position_encoding.shape[1]:
            raise ValueError('Input exceeds configured positional-encoding length.')
        hidden=self.cnn(features[:,None]).permute(0,2,1,3).flatten(start_dim=2)
        hidden=self.projection(hidden)
        # Match the GRU's float32 temporal arithmetic; shared CNN/heads may use AMP.
        with torch.autocast(device_type=features.device.type,enabled=False):
            hidden=hidden.float()+self.position_encoding[:,:features.shape[1]]
            hidden=self.temporal_output(self.temporal(hidden))
        return {'state':self.state_head(hidden).reshape(features.shape[0],features.shape[1],6,self.fret_classes),
                'onset':self.onset_head(hidden),'offset':self.offset_head(hidden),'pitch':self.pitch_head(hidden)}
