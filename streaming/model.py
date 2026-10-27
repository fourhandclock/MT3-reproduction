"""A causal CNN–GRU with bounded convolution caches and carried GRU state."""

import torch
from torch import nn
from torch.nn import functional as F


class FrameGroupNorm(nn.Module):
    """Normalize channel/frequency groups independently for every frame."""

    def __init__(self, channels, groups=8):
        super().__init__()
        self.normalization = nn.GroupNorm(groups, channels)

    def forward(self, features):
        batch, channels, time, frequency = features.shape
        frames = features.permute(0, 2, 1, 3).reshape(batch * time, channels, frequency)
        frames = self.normalization(frames)
        return frames.reshape(batch, time, channels, frequency).permute(0, 2, 1, 3)


class CausalCNNGRU(nn.Module):
    def __init__(self, config):
        super().__init__()
        specification = config["model"]
        if specification.get("bidirectional", False):
            raise ValueError("A streaming GRU must be unidirectional")
        self.n_bins = int(config["feature"]["n_bins"])
        channels = specification["channels"]
        input_channels, frequency = 1, self.n_bins
        self.convolutions = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        for output_channels in channels:
            self.convolutions.append(nn.Conv2d(input_channels, output_channels, 3, padding=(0, 1)))
            self.normalizations.append(FrameGroupNorm(output_channels))
            input_channels, frequency = output_channels, frequency // 2
        projection_size = int(specification["projection_size"])
        self.projection = nn.Sequential(
            nn.Linear(channels[-1] * frequency, projection_size),
            nn.LayerNorm(projection_size), nn.GELU(), nn.Dropout(specification["dropout"]),
        )
        hidden_size = int(specification["gru_hidden_size"])
        layers = int(specification["gru_layers"])
        self.gru = nn.GRU(
            projection_size, hidden_size, num_layers=layers, batch_first=True,
            bidirectional=False, dropout=specification["dropout"] if layers > 1 else 0,
        )
        self.output_size = int(specification.get("output_size", 256))
        self.output_projection = nn.Linear(hidden_size, self.output_size)
        self.fret_classes = int(specification["max_fret"]) + 2
        self.pitch_count = int(specification["pitch_max"]) - int(specification["pitch_min"]) + 1
        self.state_head = nn.Linear(self.output_size, 6 * self.fret_classes)
        self.onset_head = nn.Linear(self.output_size, 6)
        self.offset_head = nn.Linear(self.output_size, 6)
        self.pitch_head = nn.Linear(self.output_size, self.pitch_count)

    def _forward_chunk(self, features, state, detach_state):
        if features.ndim != 3 or features.shape[-1] != self.n_bins:
            raise ValueError("Expected features with shape [batch, time, n_bins]")
        batch, time, _ = features.shape
        if time == 0:
            shapes = {"state": (batch, 0, 6, self.fret_classes),
                      "onset": (batch, 0, 6), "offset": (batch, 0, 6),
                      "pitch": (batch, 0, self.pitch_count)}
            return {key: features.new_empty(shape) for key, shape in shapes.items()}, state
        contexts = state["conv_contexts"] if state is not None else None
        if contexts is not None and len(contexts) != len(self.convolutions):
            raise ValueError("Streaming state has the wrong number of convolution caches")
        hidden = features[:, None]
        next_contexts = []
        for index, (convolution, normalization) in enumerate(zip(self.convolutions, self.normalizations)):
            if contexts is None:
                context = hidden.new_zeros((batch, hidden.shape[1], 2, hidden.shape[-1]))
            else:
                context = contexts[index]
                expected = (batch, hidden.shape[1], 2, hidden.shape[-1])
                if tuple(context.shape) != expected or context.device != hidden.device:
                    raise ValueError("Streaming state shape/device differs; reset before a new stream")
                context = context.to(dtype=hidden.dtype)
            padded = torch.cat((context, hidden), dim=2)
            next_context = padded[:, :, -2:, :]
            next_contexts.append(next_context.detach().clone() if detach_state else next_context)
            hidden = F.max_pool2d(F.gelu(normalization(convolution(padded))), (1, 2))
        hidden = hidden.permute(0, 2, 1, 3).flatten(start_dim=2)
        hidden = self.projection(hidden)
        recurrent_state = state["gru_hidden"] if state is not None else None
        with torch.autocast(device_type=features.device.type, enabled=False):
            hidden, recurrent_state = self.gru(hidden.float(), recurrent_state)
        hidden = self.output_projection(hidden)
        outputs = {
            "state": self.state_head(hidden).reshape(batch, time, 6, self.fret_classes),
            "onset": self.onset_head(hidden), "offset": self.offset_head(hidden),
            "pitch": self.pitch_head(hidden),
        }
        next_state = {"conv_contexts": tuple(next_contexts),
                      "gru_hidden": recurrent_state.detach() if detach_state else recurrent_state}
        return outputs, next_state

    def forward(self, features):
        outputs, _ = self._forward_chunk(features, None, detach_state=False)
        return outputs

    def forward_chunk(self, features, state=None):
        """Process new frames and return bounded, detached inference state.

        Call with state=None to start/reset a stream. Full and chunked outputs
        agree in eval mode; training dropout has its usual stochastic behavior.
        """
        return self._forward_chunk(features, state, detach_state=True)
