import numpy as np
import pytest
import torch

from streaming.frontend import CausalFrontend, features
from streaming.model import CausalCNNGRU, FrameGroupNorm


def config():
    return {"feature": {"kind": "causal_log_stft", "sample_rate": 16000,
                        "hop_length": 160, "window_length": 2048, "n_fft": 2048,
                        "fmin_midi": 40, "n_bins": 216, "bins_per_octave": 36,
                        "log_floor": 1e-5},
            "model": {"channels": [24, 48, 96], "projection_size": 128,
                      "gru_hidden_size": 200, "gru_layers": 2, "bidirectional": False,
                      "output_size": 256, "dropout": 0.1, "max_fret": 24,
                      "pitch_min": 40, "pitch_max": 88}}


def test_frontend_exact_chunk_equivalence_and_bounded_memory():
    audio = np.random.default_rng(11).standard_normal(7913).astype(np.float32)
    expected = features(audio, config())
    frontend, pieces, offset = CausalFrontend(config()), [], 0
    for length in [1, 159, 1, 997, 3, 160, 2001, 17, 4574]:
        pieces.append(frontend.push(audio[offset:offset + length]))
        offset += length
        assert frontend.cached_samples < frontend.window_length
        assert frontend.frames_emitted == min(offset, len(audio)) // 160
    actual = np.concatenate(pieces)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32 and actual.shape == (len(audio) // 160, 216)
    assert frontend.last_frame_sample == (len(audio) // 160) * 160
    assert frontend.received_samples == len(audio)


def test_frontend_impulse_never_appears_before_available_samples():
    quiet = np.zeros(3200, dtype=np.float32)
    impulse = quiet.copy()
    impulse[512] = 1.0
    expected = features(quiet, config())
    actual = features(impulse, config())
    np.testing.assert_array_equal(actual[:3], expected[:3])
    assert np.any(actual[3] != expected[3])
    # No incomplete final frame or artificial flushing of future audio.
    assert features(quiet[:159], config()).shape == (0, 216)
    frontend = CausalFrontend(config())
    assert frontend.push(quiet[:160]).shape == (1, 216)
    assert frontend.push([]).shape == (0, 216)


def test_frontend_reset_and_input_contract():
    audio = np.random.default_rng(1).standard_normal(1000).astype(np.float32)
    frontend = CausalFrontend(config())
    expected = frontend.push(audio)
    frontend.reset()
    np.testing.assert_array_equal(frontend.push(audio), expected)
    with pytest.raises(ValueError, match="mono"):
        frontend.push(np.zeros((10, 2)))
    with pytest.raises(ValueError, match="non-finite"):
        frontend.push([np.nan])


def test_per_frame_normalization_is_future_independent():
    torch.manual_seed(2)
    normalization = FrameGroupNorm(24)
    inputs = torch.randn(2, 24, 9, 31)
    changed = inputs.clone()
    changed[:, :, 4:] *= 1000
    torch.testing.assert_close(normalization(inputs)[:, :, :4], normalization(changed)[:, :, :4],
                               rtol=0, atol=0)


@pytest.fixture
def model():
    torch.set_num_threads(4)
    torch.manual_seed(3)
    return CausalCNNGRU(config()).eval()


def test_model_future_and_prefix_independence(model):
    inputs = torch.randn(2, 31, 216)
    changed = inputs.clone()
    changed[:, 13:] = 50 * torch.randn_like(changed[:, 13:])
    with torch.no_grad():
        whole = model(inputs)
        modified = model(changed)
        prefix = model(inputs[:, :13])
    for name in whole:
        torch.testing.assert_close(whole[name][:, :13], modified[name][:, :13], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(whole[name][:, :13], prefix[name], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("chunk_sizes", [[1] * 23, [3, 1, 8, 2, 9]])
def test_model_full_chunk_equivalence_and_detached_bounded_state(model, chunk_sizes):
    inputs = torch.randn(2, 23, 216)
    with torch.no_grad():
        whole = model(inputs)
    state, collected, position = None, {}, 0
    # Deliberately leave autograd enabled: returned state is inference-only.
    for count in chunk_sizes:
        outputs, state = model.forward_chunk(inputs[:, position:position + count], state)
        for name, output in outputs.items():
            collected.setdefault(name, []).append(output)
        position += count
        assert all(context.shape[2] == 2 and not context.requires_grad
                   for context in state["conv_contexts"])
        assert state["gru_hidden"].shape == (2, 2, 200)
        assert not state["gru_hidden"].requires_grad
    for name in whole:
        torch.testing.assert_close(torch.cat(collected[name], dim=1), whole[name], rtol=1e-5, atol=1e-6)


def test_model_reset_empty_input_and_gradient_flow(model):
    inputs = torch.randn(1, 12, 216, requires_grad=True)
    first, state = model.forward_chunk(inputs)
    empty, same_state = model.forward_chunk(inputs[:, :0], state)
    assert same_state is state and all(value.shape[1] == 0 for value in empty.values())
    model.forward_chunk(torch.randn_like(inputs), state)
    reset, _ = model.forward_chunk(inputs, None)
    for name in first:
        torch.testing.assert_close(reset[name], first[name], rtol=0, atol=0)
    model(inputs)["state"].square().mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert model.convolutions[0].weight.grad.abs().sum() > 0
    assert model.gru.weight_hh_l0.grad.abs().sum() > 0


def test_model_rejects_bidirectionality_and_wrong_stream_batch(model):
    invalid = config()
    invalid["model"]["bidirectional"] = True
    with pytest.raises(ValueError, match="unidirectional"):
        CausalCNNGRU(invalid)
    _, state = model.forward_chunk(torch.zeros(1, 1, 216))
    with pytest.raises(ValueError, match="reset"):
        model.forward_chunk(torch.zeros(2, 1, 216), state)
