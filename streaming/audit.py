"""Check saved causal-event, MIDI and frozen-weight identities without training."""
import json
from collections import Counter
from pathlib import Path

import torch

from baseline.download import ROOT,sha256
from baseline.evaluation import midi_notes
from .evaluate import aggregate,load_checkpoint,save_json


def audit(output):
    output=Path(output).resolve();directory=output/'evaluation'
    run=json.loads((directory/'run.json').read_text());summary=json.loads((directory/'summary.json').read_text())
    path,checkpoint,model=load_checkpoint(output,torch.device('cpu'));del model
    parent_path=output/'model/best.pt';parent=torch.load(parent_path,map_location='cpu',weights_only=True)
    if sha256(parent_path)!=checkpoint['inference_calibration']['parent_checkpoint_sha256']:
        raise ValueError('Frozen parent checkpoint changed.')
    if any(not torch.equal(value,parent['model'][key]) for key,value in checkpoint['model'].items()):
        raise ValueError('Calibration changed learned weights.')
    manifest=json.loads((ROOT/'data/cnn_gru/manifest.json').read_text())
    if sha256(ROOT/'data/cnn_gru/manifest.json')!=run['manifest_sha256']:raise ValueError('Evaluation split changed.')
    for entry in manifest['test']:
        for field in ['audio','annotation']:
            if sha256(ROOT/entry[field])!=entry[field+'_sha256']:raise ValueError('Original evaluation input changed.')
    records=[json.loads(p.read_text()) for p in directory.glob('results/*/*-metrics.json')]
    expected={(e['id'],mode) for e in manifest['test'] for mode in summary['modes']}
    actual=[(r['id'],r['mode']) for r in records]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError('Missing or duplicate evaluation records.')
    if aggregate(records)!=summary:raise ValueError('Saved summary does not reproduce from per-example metrics.')
    minimum_ticks=0
    for record in records:
        stem=directory/'results'/record['mode']/record['id'];document=json.loads(stem.with_suffix('.json').read_text())
        if document['notes']!=record['notes']:raise ValueError('Exact note JSON differs from scored source events.')
        written,_=midi_notes(stem.with_suffix('.mid'))
        if Counter(n['pitch'] for n in written)!=Counter(n['pitch'] for n in record['notes']):raise ValueError('MIDI lost or duplicated note attacks.')
        minimum_ticks+=document['metadata'].get('midi_minimum_tick_adjustments',0)
        if record['mode']=='offline_bidir':continue
        events=record['events'];on=[e for e in events if e['type']=='note_on'];off=[e for e in events if e['type']=='note_off']
        if len(on)!=len(record['notes']) or len(off)!=len(record['notes']):raise ValueError('Incremental ledger does not have exact off/on pairs.')
        for event in events:
            if event['delivered_at_seconds']+1e-9<event['emitted_at_audio_seconds'] or event['emitted_at_audio_seconds']+1e-9<event['source_time']:
                raise ValueError('Event predates the audio evidence/source time.')
        if len({e['note_id'] for e in on})!=len(on):raise ValueError('Duplicate note identity.')
    by_key={(r['id'],r['mode']):r for r in records};same=True
    fields=['start','end','pitch','string','fret']
    for entry in manifest['test']:
        bases=[[tuple(n[k] for k in fields) for n in by_key[(entry['id'],f'causal_{packet}ms')]['notes']]
               for packet in run['config']['streaming']['evaluation_packet_ms']]
        same&=all(value==bases[0] for value in bases)
    if checkpoint['step']+run['discarded_optimizer_updates']>500:raise ValueError('Stage training budget exceeded.')
    result={'records':len(records),'unique_complete_coverage':True,'summary_reproduces':True,'original_source_checksums_verified':True,
            'calibration_weights_equal_parent':True,'note_json_equals_scored_intervals':True,'midi_attack_pitch_counts_preserved':True,
            'midi_minimum_tick_adjustments_all_modes':minimum_ticks,'incremental_events_paired':True,
            'source_intervals_identical_across_packet_sizes':same,'total_stage_optimizer_updates':checkpoint['step']+run['discarded_optimizer_updates']}
    save_json(directory/'audit.json',result)
    run['current_source_sha256']={p.name:sha256(p) for p in sorted((ROOT/'streaming').glob('*.py'))}
    run['artifact_audit']=result;save_json(directory/'run.json',run)
    print('Causal artifact audit passed:',result,flush=True)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(ROOT/'outputs/streaming'));args=parser.parse_args()
    audit(args.output)
