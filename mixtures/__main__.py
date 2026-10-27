import argparse
import gc
import json
import os
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description='Frozen guitar separation and controlled mixture diagnostic; no training.')
    parser.add_argument('--config',default=str(ROOT/'configs/mixtures.json'))
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('prepare')
    for name in ['render','separate','evaluate','report']:
        command=commands.add_parser(name);command.add_argument('--output',default=str(ROOT/'outputs/mixtures'))
        if name in ['separate','evaluate']: command.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    predict=commands.add_parser('transcribe');predict.add_argument('audio')
    predict.add_argument('--seconds',type=float,default=30)
    predict.add_argument('--output',default=str(ROOT/'outputs/mixtures/prediction'))
    predict.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    predict.add_argument('--score',action='store_true',help='Also estimate beats on original audio and export MusicXML.')
    args=parser.parse_args();config=json.loads(Path(args.config).read_text())
    if config['training_epochs'] or config['optimizer_updates']: parser.error('This stage is inference-only.')
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[name]=str(config['cpu_threads'])
    os.environ['NUMBA_CACHE_DIR']=str(ROOT/'.cache/numba')
    if args.command=='prepare':
        from .prepare import prepare
        prepare(config);return
    if args.command=='render':
        from .workflow import prepare_data
        prepare_data(config,args.output);return
    if args.command=='report':
        from .report import write_report
        output=Path(args.output).resolve()
        records=[json.loads(p.read_text()) for p in output.glob('results/*/*-metrics.json')]
        write_report(output,records,json.loads((output/'summary.json').read_text()),json.loads((output/'run.json').read_text()));return
    import torch
    torch.set_num_threads(config['cpu_threads'])
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else 'cpu' if args.device=='auto' else args.device)
    if args.command in ['separate','evaluate']:
        from .workflow import separate_data,transcribe_data
        separate_data(config,args.output,device)
        if args.command=='evaluate': transcribe_data(config,args.output,device)
        return
    if args.seconds<=0: parser.error('--seconds must be positive.')
    from .predict_audio import transcribe
    transcribe(args.audio,args.output,args.seconds,args.score,config,device)


if __name__=='__main__': main()
