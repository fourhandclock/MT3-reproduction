"""Validate the complete saved experiment and add cache contracts to its initial run."""
import json
from pathlib import Path

from baseline.download import ROOT,sha256
from .separator import verify
from .workflow import fingerprint,validate_coverage,validate_sources,write_json


def audit(output,upgrade_initial=False):
    output=Path(output).resolve();run_path=output/'run.json';run=json.loads(run_path.read_text())
    config=run['config'];source=json.loads((ROOT/'data/cnn_gru/manifest.json').read_text())
    validate_sources(source)
    _,provenance=verify(config)
    if run['model_provenance']!=provenance: raise ValueError('Saved model provenance changed.')
    data_path=output/'data-manifest.json';data=json.loads(data_path.read_text())
    if run['data_manifest_sha256']!=sha256(data_path): raise ValueError('Data manifest bytes changed.')
    separated_path=output/'separation-run.json';separated=json.loads(separated_path.read_text())
    if data['config']!=config or data['source_manifest_sha256']!=sha256(ROOT/'data/cnn_gru/manifest.json'):
        raise ValueError('Saved data recipe changed.')
    if upgrade_initial:
        if run.get('cache_contracts_added_after_initial_run'): raise ValueError('Initial contract upgrade already completed; use ordinary audit.')
        # Only cache validation changed after the first measured run. DSP sources must match it exactly.
        for name in ['audio.py','separator.py','prepare.py','metrics.py']:
            if run['source_hashes'][name]!=sha256(ROOT/'mixtures'/name):
                raise ValueError('Cannot adopt cached outputs from a changed renderer/separator.')
        if run['data_manifest_sha256']!=sha256(data_path): raise ValueError('Initial data manifest changed.')
        data['renderer_sha256']=sha256(ROOT/'mixtures/audio.py')
        separated['config_fingerprint']=fingerprint(config)
        separated['separator_wrapper_sha256']=sha256(ROOT/'mixtures/separator.py')
        for entry in separated['entries']:
            entry['separation']['separator_wrapper_sha256']=separated['separator_wrapper_sha256']
            log=output/'separation'/entry['condition']/(entry['id']+'.json')
            write_json(log,entry['separation'])
        run['data_manifest_sha256_at_inference']=run['data_manifest_sha256']
        run['source_hashes_at_inference']=run['source_hashes']
        run['cache_contracts_added_after_initial_run']=True
        write_json(data_path,data);write_json(separated_path,separated)
        run['data_manifest_sha256']=sha256(data_path)
    if data.get('renderer_sha256')!=sha256(ROOT/'mixtures/audio.py'): raise ValueError('Renderer cache identity mismatch.')
    if separated.get('config_fingerprint')!=fingerprint(config) or separated['model_provenance']!=provenance:
        raise ValueError('Separation settings/provenance mismatch.')
    if separated.get('separator_wrapper_sha256')!=sha256(ROOT/'mixtures/separator.py'):
        raise ValueError('Separator wrapper identity mismatch.')
    entries=data['inputs']+separated['entries'];validate_coverage(entries,source,config['conditions'])
    for entry in entries:
        if sha256(output/entry['audio'])!=entry['audio_sha256']: raise ValueError('Saved waveform changed.')
    records=[json.loads(p.read_text()) for p in output.glob('results/*/*-metrics.json')]
    validate_coverage(records,source,config['conditions'])
    by_key={(e['id'],e['condition']):e for e in entries}
    for record in records:
        entry=by_key[(record['id'],record['condition'])]
        if record['audio_sha256']!=entry['audio_sha256']: raise ValueError('Metric waveform identity mismatch.')
        stem=output/'results'/record['condition']/record['id']
        notes=json.loads(stem.with_suffix('.json').read_text())['notes']
        if notes!=record['notes']: raise ValueError('Note export differs from scored events.')
        for suffix in ['.json','.csv','.mid']:
            if not stem.with_suffix(suffix).exists(): raise ValueError('Missing note export.')
    score=json.loads((ROOT/'configs/stage1.json').read_text())
    run['scoring_config']=score;run['scoring_config_sha256']=sha256(ROOT/'configs/stage1.json')
    run['current_source_hashes']={p.name:sha256(p) for p in sorted((ROOT/'mixtures').glob('*.py'))}
    run['artifact_audit']={'paired_records':len(records),'waveforms':len(entries),'source_labels_checked':True,
                           'waveform_checksums_verified':True,'unique_complete_coverage':True,
                           'note_json_equals_scored_events':True,'no_model_updates':True}
    write_json(run_path,run)
    print('Audit passed:',len(records),'paired records and exact source/waveform/export identities.',flush=True)
    return run['artifact_audit']


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(ROOT/'outputs/mixtures'))
    parser.add_argument('--upgrade-initial-contracts',action='store_true');args=parser.parse_args()
    audit(args.output,args.upgrade_initial_contracts)
