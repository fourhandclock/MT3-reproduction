"""Proposed direct pitch supervision for the existing state logits.

This module changes no model parameters or legacy producer code. Stage15 only
tests its algebra/gradients; an acoustic comparison needs a separate budget.
"""

import math

import torch
from torch.nn import functional as F

from cnn_gru.model import transcription_loss


def state_pitch_probabilities(logits):
    if logits.ndim != 4 or logits.shape[-2:] != (6, 26):
        raise ValueError('Expected state logits [batch,time,6,26].')
    if not torch.isfinite(logits).all():
        raise ValueError('State logits must be finite.')
    probability = torch.softmax(logits.float(), dim=-1)[..., :25]
    # Each categorical string supplies at most one pitch; collapse unisons by
    # the independent-string union probability, not by summing duplicate notes.
    strings = [F.pad(probability[..., string, :], (offset, 24 - offset))
               for string, offset in enumerate([0, 5, 10, 15, 19, 24])]
    return 1. - torch.prod(1. - torch.stack(strings, dim=-2), dim=-2)


def state_pitch_loss(logits, targets):
    probability = state_pitch_probabilities(logits)
    pitch, states, mask = targets['pitch'], targets['state'], targets['mask']
    if pitch.shape != probability.shape or states.shape != logits.shape[:-1] or mask.shape != logits.shape[:2]:
        raise ValueError('State-pitch targets must share the exact batch/time grid.')
    if not torch.isfinite(pitch).all() or torch.any(pitch < 0) or torch.any(pitch > 1):
        raise ValueError('Pitch targets must lie in [0,1].')
    if not torch.isfinite(mask).all() or torch.any(mask < 0) or torch.any(mask > 1):
        raise ValueError('The crop/leading validity mask must lie in [0,1].')
    # Same-string polyphonic conflicts cannot be represented by a categorical
    # fret distribution. Exclude those complete frames from this new term only.
    valid = mask.float() * torch.all(states != -100, dim=-1)
    element = F.binary_cross_entropy(probability.clamp(1e-6, 1 - 1e-6), pitch.float(), reduction='none')
    return (element * valid[..., None]).sum() / (valid.sum() * 49).clamp_min(1.)


def loss_with_state_pitch(outputs, targets, config, positive_weights, coefficient=.5):
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError('The new loss coefficient must be finite and nonnegative.')
    losses = transcription_loss(outputs, targets, config, positive_weights)
    extra = state_pitch_loss(outputs['state'], targets)
    return {**losses, 'state_pitch': extra, 'total': losses['total'] + coefficient * extra}
