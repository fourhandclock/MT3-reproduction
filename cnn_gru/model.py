import torch
from torch import nn
from torch.nn import functional as F


class GuitarCNNGRU(nn.Module):
    def __init__(self, config):
        super().__init__()
        specification = config['model']
        channels, input_channels = specification['channels'], 1
        layers = []
        frequency = config['feature']['n_bins']
        for output_channels in channels:
            layers.extend([nn.Conv2d(input_channels, output_channels, 3, padding=1),
                           nn.GroupNorm(8, output_channels), nn.GELU(), nn.MaxPool2d((1,2))])
            input_channels, frequency = output_channels, frequency // 2
        self.cnn = nn.Sequential(*layers)
        self.projection = nn.Sequential(nn.Linear(channels[-1]*frequency, specification['projection_size']),
                                        nn.LayerNorm(specification['projection_size']), nn.GELU(),
                                        nn.Dropout(specification['dropout']))
        self.gru = nn.GRU(specification['projection_size'], specification['gru_hidden_size'],
                          num_layers=specification['gru_layers'], batch_first=True,
                          bidirectional=specification['bidirectional'],
                          dropout=specification['dropout'] if specification['gru_layers'] > 1 else 0)
        output_size = specification['gru_hidden_size'] * (2 if specification['bidirectional'] else 1)
        self.fret_classes = specification['max_fret'] + 2
        self.state_head = nn.Linear(output_size, 6*self.fret_classes)
        self.onset_head, self.offset_head = nn.Linear(output_size,6), nn.Linear(output_size,6)
        self.pitch_head = nn.Linear(output_size,specification['pitch_max']-specification['pitch_min']+1)

    def forward(self, features):
        hidden = self.cnn(features[:,None])
        hidden = hidden.permute(0,2,1,3).flatten(start_dim=2)
        hidden = self.projection(hidden)
        # Keep recurrent arithmetic in float32; CNN/projection can use CUDA AMP.
        with torch.autocast(device_type=features.device.type, enabled=False):
            hidden, _ = self.gru(hidden.float())
        return {'state': self.state_head(hidden).reshape(features.shape[0],features.shape[1],6,self.fret_classes),
                'onset': self.onset_head(hidden), 'offset': self.offset_head(hidden), 'pitch': self.pitch_head(hidden)}


def transcription_loss(outputs, targets, config, positive_weights):
    mask = targets['mask'].float()
    state = targets['state']
    classes = config['model']['max_fret'] + 2
    weights = torch.ones(classes, device=mask.device)
    weights[-1] = config['training']['inactive_class_weight']
    state_loss = F.cross_entropy(outputs['state'].float().reshape(-1,classes), state.reshape(-1),
                                 weight=weights, ignore_index=-100,reduction='none').reshape(state.shape)
    valid = mask[:,:,None]*(state != -100)
    # Normalize by actual class weights, including the downweighted inactive class.
    sample_weights = weights[state.clamp_min(0)]*valid
    losses = {'state': (state_loss*valid).sum()/sample_weights.sum().clamp_min(1)}
    for key in ['onset','offset','pitch']:
        weight = torch.tensor(positive_weights[key],device=mask.device)
        element = F.binary_cross_entropy_with_logits(outputs[key].float(),targets[key].float(),pos_weight=weight,reduction='none')
        valid = mask[:,:,None].expand_as(element)
        losses[key] = (element*valid).sum()/valid.sum().clamp_min(1)
    losses['total'] = losses['state']+losses['onset']+0.5*losses['offset']+0.5*losses['pitch']
    return losses
