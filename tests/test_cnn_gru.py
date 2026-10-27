import json

import numpy as np
import pytest
import torch

from cnn_gru.data import make_targets,validate_splits
from cnn_gru.inference import decode,predict_probabilities,read_string_midi,save_midi
from cnn_gru.model import GuitarCNNGRU,transcription_loss


@pytest.fixture
def config():
    with open('configs/cnn_gru.json') as source:
        return json.load(source)


def test_labels_preserve_time_and_mask_conflicting_frets(config):
    notes=[{'start':0.1,'end':0.5,'string':0,'fret':0,'pitch':40},
           {'start':0.4,'end':0.8,'string':0,'fret':2,'pitch':42}]
    labels,conflicts=make_targets(notes,100,config)
    assert labels['state'][9,0]==25
    assert labels['state'][10,0]==0
    assert labels['state'][40,0]==-100
    assert labels['state'][50,0]==2
    assert labels['state'][80,0]==25
    assert conflicts==10
    assert labels['onset'][10,0]==1
    assert labels['offset'][80,0]==1


def test_split_check_rejects_progression_leakage():
    manifest={'train':[{'id':'00_Rock1-100-C_comp'}],
              'validation':[{'id':'00_Rock2-100-C_comp'}],
              'test':[{'id':'00_Rock1-100-C_comp'}]}
    with pytest.raises(ValueError): validate_splits(manifest)


def test_network_keeps_time_resolution_and_has_finite_gradients(config):
    torch.set_num_threads(2)
    model=GuitarCNNGRU(config)
    assert sum(p.numel() for p in model.parameters())<1_000_000
    output=model(torch.randn(2,20,216))
    assert output['state'].shape==(2,20,6,26)
    target={'state':torch.full((2,20,6),25),'onset':torch.zeros(2,20,6),
            'offset':torch.zeros(2,20,6),'pitch':torch.zeros(2,20,49),'mask':torch.ones(2,20)}
    target['state'][0,8,0]=-100
    loss=transcription_loss(output,target,config,{'onset':10,'offset':10,'pitch':4})['total']
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_decoder_splits_repeated_plucks_and_preserves_string_pitch(config):
    state=np.zeros((100,6,26),dtype=np.float32);state[:,:,-1]=1
    state[10:90,0,-1]=0;state[10:90,0,0]=1
    onset=np.zeros((100,6),dtype=np.float32);onset[10,0]=1;onset[50,0]=1
    offset=np.zeros_like(onset);offset[90,0]=1
    notes=decode({'state':state,'onset':onset,'offset':offset},config,
                 {'activity_threshold':0.45,'onset_threshold':0.4})
    assert len(notes)==2
    assert notes[0]['start']==0.1 and notes[0]['end']==0.5
    assert notes[1]['start']==0.5 and notes[1]['end']==0.9
    assert all(note['pitch']==40 and note['string']==0 and note['fret']==0 for note in notes)


def test_overlap_inference_covers_tail_without_changing_constant_predictions(config):
    class Constant(torch.nn.Module):
        def forward(self,x):
            return {'state':torch.zeros(len(x),len(x[0]),6,26),
                    'onset':torch.zeros(len(x),len(x[0]),6),
                    'offset':torch.zeros(len(x),len(x[0]),6),
                    'pitch':torch.zeros(len(x),len(x[0]),49)}
    probability=predict_probabilities(Constant(),np.zeros((431,216),dtype=np.float32),
                                       {'mean':[0]*216,'std':[1]*216},config,torch.device('cpu'))
    assert probability['state'].shape==(431,6,26)
    np.testing.assert_allclose(probability['state'],1/26,atol=1e-6)
    np.testing.assert_allclose(probability['onset'],0.5,atol=1e-6)


def test_string_and_fret_survive_midi_round_trip(tmp_path):
    notes=[{'start':0.1,'end':0.5,'pitch':52,'string':2,'fret':2,'program':24},
           {'start':0.1,'end':0.6,'pitch':64,'string':5,'fret':0,'program':24}]
    path=tmp_path/'guitar.mid'
    save_midi(notes,path)
    restored=read_string_midi(path)
    assert [(n['pitch'],n['string'],n['fret']) for n in restored]==[(52,2,2),(64,5,0)]
    json.dumps(restored)


def test_release_refinement_cannot_overlap_the_next_fret(config):
    state=np.zeros((100,6,26),dtype=np.float32);state[:,:,-1]=1
    state[10:90,0,-1]=0;state[10:50,0,0]=1;state[50:90,0,2]=1
    onset=np.zeros((100,6),dtype=np.float32);onset[10,0]=1;onset[50,0]=1
    offset=np.zeros_like(onset);offset[52,0]=1;offset[90,0]=1
    notes=decode({'state':state,'onset':onset,'offset':offset},config,
                 {'activity_threshold':0.45,'onset_threshold':0.4})
    assert len(notes)==2
    assert notes[0]['end']<=notes[1]['start']
