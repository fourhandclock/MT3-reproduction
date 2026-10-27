"""Conditional-fret CE excludes the inactive logit and inactive/conflict targets."""

import torch
from torch.nn import functional as F


def active_support(logits,target):
    state,mask=target['state'],target['mask']
    if logits.ndim!=4 or logits.shape[-2:]!=(6,26) or state.shape!=logits.shape[:-1] or mask.shape!=state.shape[:2]:
        raise ValueError('Expected logits[B,T,6,26], states[B,T,6] and mask[B,T]')
    if not logits.is_floating_point() or state.dtype!=torch.long or logits.device!=state.device or mask.device!=state.device:
        raise ValueError('Fret logits/labels/masks have incompatible dtype or device')
    if not torch.isfinite(mask).all() or torch.any((mask<0)|(mask>1)) or torch.any((state!=-100)&((state<0)|(state>25))):
        raise ValueError('Invalid supervision mask or fret label')
    return (mask>0)[...,None]&(state>=0)&(state<25)


def conditional_fret_loss(logits,target):
    valid=active_support(logits,target)
    # Keep CE numerically stable under CUDA mixed precision, with a differentiable
    # zero for batches containing no supervised active string-frame targets.
    with torch.autocast(device_type=logits.device.type,enabled=False):
        active=logits[...,:25].float()
        if not bool(valid.any()):return active.sum()*0.
        return F.cross_entropy(active[valid],target['state'][valid],reduction='mean')
