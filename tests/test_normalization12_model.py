"""Protect the single-variable normalization trial and its causal contract."""

from copy import deepcopy

import pytest
import torch

from normalization12.model import CausalCNNGRU
from streaming.model import CausalCNNGRU as LegacyCausalCNNGRU


def config(groups=8):
    return {"feature": {"n_bins": 216},
            "model": {"channels": [24, 48, 96], "projection_size": 128,
                      "gru_hidden_size": 200, "gru_layers": 2, "bidirectional": False,
                      "output_size": 256, "dropout": .1, "max_fret": 24,
                      "pitch_min": 40, "pitch_max": 88, "cnn_norm_groups": groups}}


@pytest.fixture
def model():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    return CausalCNNGRU(config(1)).eval()


def assert_outputs_equal(actual, expected, exact=False):
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key],
                                   rtol=0 if exact else 1e-5,
                                   atol=0 if exact else 1e-6)


@pytest.mark.parametrize("groups", [1, 8])
def test_initialization_rng_keys_and_tensor_shapes_match_legacy(groups):
    torch.manual_seed(42)
    legacy = LegacyCausalCNNGRU(config())
    legacy_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(42)
    candidate = CausalCNNGRU(config(groups))
    assert torch.equal(torch.random.get_rng_state(), legacy_rng)
    assert sum(value.numel() for value in candidate.parameters()) == 931145
    assert candidate.state_dict().keys() == legacy.state_dict().keys()
    for key, value in candidate.state_dict().items():
        assert torch.equal(value, legacy.state_dict()[key]), key
    for normalization, channels in zip(candidate.normalizations, [24, 48, 96]):
        assert normalization.normalization.num_groups == groups
        assert normalization.normalization.weight.shape == (channels,)
        assert normalization.normalization.bias.shape == (channels,)


@pytest.mark.parametrize("explicit_groups", [False, True])
def test_group8_reproduces_legacy_full_and_chunk_outputs_exactly(explicit_groups):
    specification = config()
    if not explicit_groups:
        del specification["model"]["cnn_norm_groups"]
    torch.set_num_threads(4)
    torch.manual_seed(42)
    legacy = LegacyCausalCNNGRU(specification).eval()
    candidate = CausalCNNGRU(deepcopy(specification)).eval()
    candidate.load_state_dict(legacy.state_dict())
    inputs = torch.randn(2, 19, 216)
    with torch.no_grad():
        assert_outputs_equal(candidate(inputs), legacy(inputs), exact=True)
        legacy_state, candidate_state, position = None, None, 0
        for count in [1, 4, 2, 12]:
            legacy_outputs, legacy_state = legacy.forward_chunk(inputs[:, position:position + count], legacy_state)
            candidate_outputs, candidate_state = candidate.forward_chunk(inputs[:, position:position + count], candidate_state)
            assert_outputs_equal(candidate_outputs, legacy_outputs, exact=True)
            for left, right in zip(candidate_state["conv_contexts"], legacy_state["conv_contexts"]):
                assert torch.equal(left, right)
            assert torch.equal(candidate_state["gru_hidden"], legacy_state["gru_hidden"])
            position += count


def test_group1_normalization_is_frame_and_batch_local(model):
    normalization = model.normalizations[0]
    inputs = torch.randn(2, 24, 9, 31)
    changed = inputs.clone()
    changed[0, :, 4:] = 1000 * torch.randn_like(changed[0, :, 4:])
    changed[1] = -1000 * torch.randn_like(changed[1])
    with torch.no_grad():
        expected = normalization(inputs)
        actual = normalization(changed)
        isolated = normalization(inputs[:1, :, :4])
    torch.testing.assert_close(actual[0, :, :4], expected[0, :, :4], rtol=0, atol=0)
    torch.testing.assert_close(isolated[0], expected[0, :, :4], rtol=0, atol=0)
    # Verify the intended statistics explicitly: each frame has one group.
    manual = inputs.permute(0, 2, 1, 3).reshape(18, 24 * 31)
    manual = (manual - manual.mean(-1, keepdim=True)) / (manual.var(-1, unbiased=False, keepdim=True) + 1e-5).sqrt()
    manual = manual.reshape(2, 9, 24, 31).permute(0, 2, 1, 3)
    torch.testing.assert_close(expected, manual, rtol=1e-5, atol=1e-6)


def test_group1_preserves_relative_group_level_information():
    one = CausalCNNGRU(config(1)).normalizations[0]
    eight = CausalCNNGRU(config(8)).normalizations[0]
    # Constant levels within the former three-channel groups disappear under
    # eight groups, but remain distinct after one global frame normalization.
    inputs = torch.arange(8, dtype=torch.float32).repeat_interleave(3)[None, :, None, None].expand(1, 24, 2, 7)
    with torch.no_grad():
        grouped = eight(inputs)
        shared = one(inputs)
    assert grouped.abs().max().item() < 1e-4
    assert shared[0, :3].mean().item() < -.5
    assert shared[0, -3:].mean().item() > .5


def test_group1_model_prefix_future_and_batch_independence(model):
    inputs = torch.randn(2, 23, 216)
    changed = inputs.clone()
    changed[0, 11:] = 100 * torch.randn_like(changed[0, 11:])
    changed[1] = -100 * torch.randn_like(changed[1])
    with torch.no_grad():
        expected = model(inputs)
        modified = model(changed)
        isolated = model(inputs[:1, :11])
    assert_outputs_equal({key: value[:1, :11] for key, value in modified.items()},
                         {key: value[:1, :11] for key, value in expected.items()})
    assert_outputs_equal(isolated, {key: value[:1, :11] for key, value in expected.items()})


@pytest.mark.parametrize("chunks", [[1] * 31, [3, 1, 8, 2, 17]])
def test_group1_chunk_equivalence_and_bounded_detached_state(model, chunks):
    inputs = torch.randn(2, 31, 216)
    with torch.no_grad():
        expected = model(inputs)
    state, position, collected = None, 0, {}
    for count in chunks:
        outputs, state = model.forward_chunk(inputs[:, position:position + count], state)
        position += count
        for key, value in outputs.items():
            collected.setdefault(key, []).append(value)
        assert [tuple(value.shape) for value in state["conv_contexts"]] == [
            (2, 1, 2, 216), (2, 24, 2, 108), (2, 48, 2, 54)]
        assert all(not value.requires_grad for value in state["conv_contexts"])
        assert state["gru_hidden"].shape == (2, 2, 200)
        assert not state["gru_hidden"].requires_grad
    assert_outputs_equal({key: torch.cat(values, 1) for key, values in collected.items()}, expected)


def test_group1_reset_empty_and_gradient_contract(model):
    inputs = torch.randn(1, 12, 216, requires_grad=True)
    first, state = model.forward_chunk(inputs)
    empty, untouched = model.forward_chunk(inputs[:, :0], state)
    assert untouched is state and all(value.shape[1] == 0 for value in empty.values())
    model.forward_chunk(torch.randn_like(inputs), state)
    reset, _ = model.forward_chunk(inputs, None)
    assert_outputs_equal(reset, first, exact=True)
    model(inputs)["state"].square().mean().backward()
    assert torch.isfinite(inputs.grad).all()
    assert model.normalizations[0].normalization.weight.grad.abs().sum().item() > 0
    assert model.convolutions[0].weight.grad.abs().sum().item() > 0
    assert model.gru.weight_hh_l0.grad.abs().sum().item() > 0


@pytest.mark.parametrize("groups", [0, 2, 4, 16, True, 1.0, "1", None])
def test_invalid_group_selection_rejected_before_rng_consumption(groups):
    torch.manual_seed(42)
    before = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="integer 1 or 8"):
        CausalCNNGRU(config(groups))
    assert torch.equal(torch.random.get_rng_state(), before)


def test_legacy_bidirectionality_and_wrong_batch_guards_retained(model):
    invalid = config(1)
    invalid["model"]["bidirectional"] = True
    with pytest.raises(ValueError, match="unidirectional"):
        CausalCNNGRU(invalid)
    _, state = model.forward_chunk(torch.zeros(1, 1, 216))
    with pytest.raises(ValueError, match="reset"):
        model.forward_chunk(torch.zeros(2, 1, 216), state)
