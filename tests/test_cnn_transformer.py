import copy
import json

import pytest
import torch

from cnn_gru.model import GuitarCNNGRU,transcription_loss
from cnn_transformer.control import common_config
from cnn_transformer.model import GuitarCNNTransformer


@pytest.fixture
def config():
    with open('configs/cnn_transformer.json') as stream: return json.load(stream)


def test_shared_initial_weights_and_parameter_budget_match(config):
    torch.set_num_threads(2)
    torch.manual_seed(config['seed']);gru=GuitarCNNGRU(common_config(config))
    torch.manual_seed(config['seed']);transformer=GuitarCNNTransformer(config)
    gru_state=gru.state_dict();transformer_state=transformer.state_dict()
    prefixes=('cnn.','projection.','state_head.','onset_head.','offset_head.','pitch_head.')
    for key in gru_state:
        if key.startswith(prefixes): assert torch.equal(gru_state[key],transformer_state[key]),key
    original=sum(p.numel() for p in gru.parameters());candidate=sum(p.numel() for p in transformer.parameters())
    assert abs(candidate-original)/original<0.01
    assert not hasattr(transformer,'gru')
    assert not torch.equal(transformer.temporal.layers[0].linear1.weight,transformer.temporal.layers[1].linear1.weight)


def test_transformer_heads_keep_time_resolution_and_gradients_are_finite(config):
    torch.set_num_threads(2)
    model=GuitarCNNTransformer(config)
    output=model(torch.randn(2,20,216))
    assert output['state'].shape==(2,20,6,26)
    target={'state':torch.full((2,20,6),25),'onset':torch.zeros(2,20,6),
            'offset':torch.zeros(2,20,6),'pitch':torch.zeros(2,20,49),'mask':torch.ones(2,20)}
    loss=transcription_loss(output,target,config,{'onset':20,'offset':20,'pitch':8})['total']
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_common_recipe_is_exactly_the_existing_gru_configuration(config):
    with open('configs/cnn_gru.json') as stream: reference=json.load(stream)
    assert common_config(config)==reference
