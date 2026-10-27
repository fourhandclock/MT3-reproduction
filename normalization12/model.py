"""Frame-local CNN normalization ablation with otherwise identical weights.

Eight groups reproduces the frozen streaming model. One group normalizes all
CNN channels and frequency positions in each frame together, retaining relative
levels across the previous eight groups. Both remove the frame's overall mean
and scale; neither normalizes across time or across examples in a batch.
"""

from streaming.model import CausalCNNGRU as StreamingCausalCNNGRU


class CausalCNNGRU(StreamingCausalCNNGRU):
    """Select one or eight CNN groups without replacing any module or tensor."""

    def __init__(self, config):
        groups = config["model"].get("cnn_norm_groups", 8)
        if isinstance(groups, bool) or not isinstance(groups, int) or groups not in (1, 8):
            raise ValueError("CNN normalization groups must be the integer 1 or 8")
        # Parent construction consumes exactly the legacy initialization draws.
        # GroupNorm's group count is metadata; its affine tensors stay per-channel.
        super().__init__(config)
        for normalization in self.normalizations:
            normalization.normalization.num_groups = groups

