import json

import pytest

from baseline.download import sha256
from mixtures import workflow


def test_condition_coverage_rejects_partial_duplicate_and_unexpected():
    source={'test':[{'id':'a'},{'id':'b'}]};conditions=['clean','mix','stem']
    entries=[{'id':clip,'condition':condition} for clip in ['a','b'] for condition in conditions]
    workflow.validate_coverage(entries,source,conditions)
    for wrong in [entries[:-1],entries+[entries[0]],entries+[{'id':'new','condition':'clean'}]]:
        with pytest.raises(ValueError): workflow.validate_coverage(wrong,source,conditions)


def test_cached_render_rechecks_original_labels_and_renderer(tmp_path,monkeypatch):
    monkeypatch.setattr(workflow,'ROOT',tmp_path)
    (tmp_path/'data/cnn_gru').mkdir(parents=True);(tmp_path/'configs').mkdir();(tmp_path/'mixtures').mkdir()
    raw=tmp_path/'original.wav';raw.write_bytes(b'original audio')
    annotation=tmp_path/'original.jams';annotation.write_text('original annotation')
    renderer=tmp_path/'mixtures/audio.py';renderer.write_text('fixed renderer')
    entry={'id':'00_Rock3-100-C_comp','audio':'original.wav','audio_sha256':sha256(raw),
           'annotation':'original.jams','annotation_sha256':sha256(annotation)}
    source={'train':[],'validation':[],'test':[entry]};manifest=tmp_path/'data/cnn_gru/manifest.json'
    manifest.write_text(json.dumps(source));(tmp_path/'configs/cnn_gru.json').write_text(json.dumps({'feature':{'sample_rate':16000}}))
    output=tmp_path/'output';output.mkdir();rendered=output/'saved.wav';rendered.write_bytes(b'rendered waveform')
    config={'sample_rate':16000,'conditions':['clean']}
    cached={'config':config,'source_manifest_sha256':sha256(manifest),'renderer_sha256':sha256(renderer),
            'inputs':[{'id':entry['id'],'condition':'clean','audio':'saved.wav','audio_sha256':sha256(rendered)}]}
    (output/'data-manifest.json').write_text(json.dumps(cached))
    assert workflow.prepare_data(config,output)==cached
    annotation.write_text('changed annotation')
    with pytest.raises(ValueError,match='source'): workflow.prepare_data(config,output)
    annotation.write_text('original annotation');renderer.write_text('changed renderer')
    with pytest.raises(ValueError,match='renderer'): workflow.prepare_data(config,output)
