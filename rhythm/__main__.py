import argparse
import gc
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description='Offline guitar rhythm and editable MusicXML; no training.')
    parser.add_argument('--config', default=str(ROOT/'configs/rhythm.json'))
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prepare')
    for name in ['evaluate', 'report', 'export']:
        sub = commands.add_parser(name); sub.add_argument('--output', default=str(ROOT/'outputs/rhythm'))
        if name == 'evaluate': sub.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    score = commands.add_parser('score'); score.add_argument('notes'); score.add_argument('--corrections', required=True)
    score.add_argument('--output', default=str(ROOT/'outputs/rhythm/corrected'))
    transcribe = commands.add_parser('transcribe'); transcribe.add_argument('audio')
    transcribe.add_argument('--output', default=str(ROOT/'outputs/rhythm/prediction'))
    transcribe.add_argument('--seconds', type=float, default=30)
    transcribe.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    transcribe.add_argument('--corrections')
    args = parser.parse_args(); config = json.loads(Path(args.config).read_text())
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ[key] = '4'
    os.environ['NUMBA_CACHE_DIR'] = str(ROOT/'.cache/numba')
    if args.command == 'prepare':
        from .prepare import prepare
        prepare(); return
    if args.command == 'export':
        from .workflow import regenerate_exports
        regenerate_exports(args.output); return
    if args.command == 'report':
        from .report import write_report
        folder = Path(args.output)
        records = [json.loads(p.read_text()) for p in folder.glob('scores/*/*/*-metrics.json')]
        write_report(folder, records, json.loads((folder/'summary.json').read_text()), json.loads((folder/'run.json').read_text()))
        return
    if args.command == 'score':
        source = json.loads(Path(args.notes).read_text())
        notes = source['notes'] if isinstance(source, dict) else source
        if notes and 'raw' in notes[0]: notes = [n['raw'] for n in notes]
        corrected_score(notes, args.corrections, config, args.output); return
    import torch
    torch.set_num_threads(4)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device)
    if args.command == 'evaluate':
        from .workflow import evaluate_rhythm
        evaluate_rhythm(config, args.output, device); return
    if args.seconds <= 0: parser.error('--seconds must be positive.')
    from fingering.predict_audio import transcribe as transcribe_notes
    output = Path(args.output).resolve(); raw = output.parent/(output.name+'-raw')
    fingering_config = json.loads((ROOT/'configs/fingering.json').read_text())
    transcribe_notes(args.audio, raw, fingering_config, ROOT/'outputs/fingering/selection.json', args.seconds, device)
    notes = json.loads(raw.with_suffix('.json').read_text())['notes']
    gc.collect()
    if device.type == 'cuda': torch.cuda.empty_cache()
    if args.corrections:
        corrected_score(notes, args.corrections, config, output); return
    from .tracker import Tracker
    from .grid import make_grid
    from .workflow import score_document
    from .notation import export_score
    tracker = Tracker(config, device); result = tracker.predict(args.audio, args.seconds)
    result.pop('beat_probability'); result.pop('downbeat_probability')
    output.with_name(output.name+'-beats.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    grid = make_grid(result['beats'], result['downbeats'], result['duration_seconds'])
    document = score_document(notes, grid, config)
    document['metadata'] = {'audio': str(Path(args.audio).resolve()), 'mode': 'automatic',
                            'raw_note_file': str(raw.with_suffix('.json')), 'beat_checkpoint_sha256': tracker.provenance['sha256']}
    export_score(document, output, Path(args.audio).stem)
    print('Score exported:', output.with_suffix('.musicxml'), flush=True)


def corrected_score(notes, corrections_path, config, output):
    import math
    from .grid import make_grid
    from .workflow import score_document
    from .notation import export_score
    from .chords import ROOTS, INTERVALS
    corrections = json.loads(Path(corrections_path).read_text())
    duration = float(corrections.get('duration_seconds', max(n['end'] for n in notes) if notes else 0))
    if any(n['start'] >= duration for n in notes): raise ValueError('Corrected duration excludes source attacks.')
    chords = corrections.get('chords')
    if chords is not None:
        previous = -1
        allowed = {'N'} | {f'{r}:{q}' for r in ROOTS for q in INTERVALS}
        for chord in chords:
            if chord['label'] not in allowed or not all(math.isfinite(chord[k]) for k in ['start','end']):
                raise ValueError('Invalid corrected chord label/interval.')
            if not 0 <= chord['start'] < chord['end'] <= duration or chord['start'] < previous:
                raise ValueError('Corrected chords must be ordered, nonoverlapping intervals within the input.')
            previous = chord['end']
    grid = make_grid(corrections['beats'], corrections.get('downbeats', []), duration, corrections.get('meter'))
    document = score_document(notes, grid, config, chords, corrections.get('subdivision'))
    document['metadata'] = {'mode': 'corrected', 'corrections_file': str(Path(corrections_path).resolve())}
    export_score(document, output, 'Corrected guitar score')
    print('Corrected score exported:', Path(output).with_suffix('.musicxml'), flush=True)


if __name__ == '__main__': main()
