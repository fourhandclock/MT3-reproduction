import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description='Pitch-preserving guitar fingering; no model training.')
    parser.add_argument('--config',default=str(ROOT/'configs/fingering.json'))
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ['select','evaluate','report']:
        command=commands.add_parser(name);command.add_argument('--output',default=str(ROOT/'outputs/fingering'))
        if name=='select': command.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    retab=commands.add_parser('retab');retab.add_argument('input')
    retab.add_argument('--output',default=str(ROOT/'outputs/fingering/retab'))
    transcribe=commands.add_parser('transcribe');transcribe.add_argument('audio')
    transcribe.add_argument('--output',default=str(ROOT/'outputs/fingering/prediction'))
    transcribe.add_argument('--selection',default=str(ROOT/'outputs/fingering/selection.json'))
    transcribe.add_argument('--seconds',type=float,default=30)
    transcribe.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    arguments=parser.parse_args();config=json.loads(Path(arguments.config).read_text())
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='4'
    if arguments.command=='select':
        import torch
        torch.set_num_threads(4)
        device=torch.device('cuda' if arguments.device=='auto' and torch.cuda.is_available() else
                            'cpu' if arguments.device=='auto' else arguments.device)
        from .workflow import select
        select(config,arguments.output,device)
    elif arguments.command=='evaluate':
        from .workflow import evaluate_fingering
        evaluate_fingering(config,arguments.output)
    elif arguments.command=='report':
        from .report import write_report
        folder=Path(arguments.output).resolve();evaluation=folder/'evaluation'
        records=[json.loads(path.read_text()) for path in evaluation.glob('*/*/*-metrics.json')]
        write_report(folder,records,json.loads((evaluation/'summary.json').read_text()),json.loads((evaluation/'run.json').read_text()))
    elif arguments.command=='transcribe':
        if arguments.seconds<=0: parser.error('--seconds must be positive.')
        import torch
        torch.set_num_threads(4)
        device=torch.device('cuda' if arguments.device=='auto' and torch.cuda.is_available() else
                            'cpu' if arguments.device=='auto' else arguments.device)
        from .predict_audio import transcribe
        transcribe(arguments.audio,arguments.output,config,arguments.selection,arguments.seconds,device)
    else:
        from .decoder import solve
        from .export import export
        from .metrics import preserved
        path=Path(arguments.input)
        if path.suffix.lower() in ['.mid','.midi']:
            from baseline.evaluation import midi_notes
            notes,_=midi_notes(path)
        else:
            document=json.loads(path.read_text());notes=document['notes'] if isinstance(document,dict) else document
        decoded,diagnostic=solve(notes,config)
        if not preserved(notes,decoded): raise RuntimeError('Pitch or timing changed.')
        export(decoded,Path(arguments.output),{'fretboard':config,'diagnostic':diagnostic,'mode':'pitch-only constrained retab'})
        print('Exported retabulated notes:',arguments.output,'; unresolved:',diagnostic['unassigned'],flush=True)


if __name__=='__main__': main()
