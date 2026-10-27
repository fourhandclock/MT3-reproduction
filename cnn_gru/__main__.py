import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description='Small guitar CNN-GRU with a capped training budget.')
    parser.add_argument('--config',default=str(ROOT/'configs/cnn_gru.json'))
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare');sub.add_parser('cache')
    train=sub.add_parser('train')
    train.add_argument('--output',default=str(ROOT/'outputs/cnn_gru'))
    train.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    train.add_argument('--resume',action='store_true')
    evaluate=sub.add_parser('evaluate')
    evaluate.add_argument('--output',default=str(ROOT/'outputs/cnn_gru'))
    evaluate.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    calibrate=sub.add_parser('calibrate')
    calibrate.add_argument('--output',default=str(ROOT/'outputs/cnn_gru'))
    calibrate.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    transcribe=sub.add_parser('transcribe')
    transcribe.add_argument('audio')
    transcribe.add_argument('--checkpoint',default=str(ROOT/'outputs/cnn_gru/best-inference.pt'))
    transcribe.add_argument('--output',default=str(ROOT/'outputs/cnn_gru/prediction.mid'))
    transcribe.add_argument('--seconds',type=float,default=30)
    transcribe.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    arguments=parser.parse_args()
    config=json.loads(Path(arguments.config).read_text(encoding='utf-8-sig'))
    for key,relative in [('NUMBA_CACHE_DIR','.cache/numba'),('TEMP','tmp/runtime'),('TMP','tmp/runtime')]:
        target=ROOT/relative;target.mkdir(parents=True,exist_ok=True);os.environ[key]=str(target)
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
        os.environ[key]=str(config['training']['cpu_threads'])
    if arguments.command=='prepare':
        from .data import prepare
        prepare(config)
    elif arguments.command=='cache':
        from .data import cache_data
        cache_data(config)
    else:
        import torch
        device=torch.device('cuda' if arguments.device=='auto' and torch.cuda.is_available()
                            else 'cpu' if arguments.device=='auto' else arguments.device)
        if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable.')
        torch.set_num_threads(config['training']['cpu_threads'])
        if arguments.command=='train':
            from .training import train
            train(config,arguments.output,device,arguments.resume)
        elif arguments.command=='calibrate':
            from .calibrate import calibrate_decoder
            calibrate_decoder(config,arguments.output,device)
        elif arguments.command=='transcribe':
            if arguments.seconds<=0: parser.error('--seconds must be positive.')
            from .predict_file import transcribe_file
            transcribe_file(arguments.audio,arguments.checkpoint,arguments.output,device,arguments.seconds)
        else:
            from .evaluate import evaluate_checkpoint
            evaluate_checkpoint(config,arguments.output,device)


if __name__=='__main__': main()
