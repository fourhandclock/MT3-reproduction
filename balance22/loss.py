"""Training-only fixed weights and float32 active-logit cross entropy."""

import numpy as np
import torch
from torch.nn import functional as F
from fret19.loss import active_support, conditional_fret_loss


def class_weights(counts, available=None):
    values = np.asarray(counts)
    if values.shape != (6, 25) or not np.issubdtype(values.dtype, np.integer) or np.any(values < 0):
        raise ValueError('Unique training counts must be nonnegative integer [6,25]')
    result = np.zeros((6, 25), np.float64)
    for string, row in enumerate(values):
        seen = row > 0
        if seen.any():
            result[string, seen] = np.minimum(4., np.sqrt(float(row.max()) / row[seen]))
    if available is not None:
        available = np.asarray(available)
        if (available.shape != (6, 25) or not np.issubdtype(available.dtype, np.integer)
                or np.any(available < 0) or np.any(values > available)):
            raise ValueError('Available training support must contain unique-supervised support')
        result[(values == 0) & (available > 0)] = 4.
    return result.astype(np.float32)


def weighted_fret_loss(logits, target, weights):
    valid = active_support(logits, target)
    if (weights.shape != (6, 25) or not weights.is_floating_point() or weights.requires_grad
            or weights.device != logits.device or not torch.isfinite(weights).all()
            or torch.any(weights < 0) or torch.any(weights > 4)):
        raise ValueError('Fixed class weights must be finite [6,25], within [0,4], on the logits device')
    with torch.autocast(device_type=logits.device.type, enabled=False):
        active = logits[..., :25].float()
        if not bool(valid.any()):
            return active.sum() * 0.
        labels = target['state'][valid]
        strings = torch.arange(6, device=logits.device).expand_as(target['state'])[valid]
        selected = weights[strings, labels].float()
        if torch.any(selected <= 0):
            raise ValueError('An active target has no permitted training support')
        losses = F.cross_entropy(active[valid], labels, reduction='none')
        return (losses * selected).sum() / selected.sum()
