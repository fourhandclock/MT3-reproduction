import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description='Causal isolated-guitar PCM replay; native microphone capture is not included.')
    parser.add_argument('--config',default=str(ROOT/'configs/streaming.json'))
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('prepare')
    for name in ['train','calibrate','evaluate','report']:
        command=commands.add_parser(name);command.add_argument('--output',default=str(ROOT/'outputs/streaming'))
        if name in ['train','calibrate','evaluate']:command.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
        if name=='train':command.add_argument('--resume',action='store_true')
    play=commands.add_parser('replay');play.add_argument('audio')
    play.add_argument('--output',default=str(ROOT/'outputs/streaming/prediction'))
    play.add_argument('--checkpoint-root',default=str(ROOT/'outputs/streaming'))
    play.add_argument('--seconds',type=float,default=30);play.add_argument('--packet-ms',type=float,default=20)
    play.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=parser.parse_args();config=json.loads(Path(args.config).read_text())
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]=str(config['training']['cpu_threads'])
    os.environ['NUMBA_CACHE_DIR']=str(ROOT/'.cache/numba')
    if args.command=='prepare':
        from .data import prepare
        prepare(config);return
    if args.command=='report':
        from .report import write_report
        output=Path(args.output).resolve();directory=output/'evaluation'
        records=[json.loads(p.read_text()) for p in directory.glob('results/*/*-metrics.json')]
        write_report(output,records,json.loads((directory/'summary.json').read_text()),json.loads((directory/'run.json').read_text()));return
    import torch
    torch.set_num_threads(config['training']['cpu_threads'])
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else 'cpu' if args.device=='auto' else args.device)
    if args.command=='train':
        budget=config.get('stage_budget',{})
        if budget.get('discarded_initial_updates',0)+config['training']['max_steps']>budget.get('total_max_optimizer_updates',500):
            parser.error('Requested run exceeds the remaining stage update budget.')
        from .training import train
        train(config,Path(args.output)/'model',device,args.resume);return
    if args.command=='calibrate':
        from .calibration import calibrate
        calibrate(args.output,device);return
    if args.command=='evaluate':
        from .evaluate import evaluate_streaming
        evaluate_streaming(args.output,device);return
    if args.seconds<=0 or args.packet_ms<=0:parser.error('Audio duration and packet size must be positive.')
    from .predict_file import predict_file
    predict_file(args.audio,args.output,args.checkpoint_root,args.seconds,args.packet_ms,device)


if __name__=='__main__':main()
