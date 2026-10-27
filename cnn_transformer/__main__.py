import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description='Matched-budget CNN-transformer experiment; reuses stage-2 data.')
    parser.add_argument('--config',default=str(ROOT/'configs/cnn_transformer.json'))
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ['train','compare']:
        command=commands.add_parser(name)
        command.add_argument('--output',default=str(ROOT/'outputs/cnn_transformer'))
        command.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
        if name=='train': command.add_argument('--resume',action='store_true')
    arguments=parser.parse_args()
    config=json.loads(Path(arguments.config).read_text(encoding='utf-8-sig'))
    for key,relative in [('NUMBA_CACHE_DIR','.cache/numba'),('TEMP','tmp/runtime'),('TMP','tmp/runtime')]:
        path=ROOT/relative;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
        os.environ[key]=str(config['training']['cpu_threads'])
    import torch
    device=torch.device('cuda' if arguments.device=='auto' and torch.cuda.is_available()
                        else 'cpu' if arguments.device=='auto' else arguments.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable.')
    torch.set_num_threads(config['training']['cpu_threads'])
    if arguments.command=='train':
        from .training import train
        train(config,arguments.output,device,arguments.resume)
    else:
        from .compare import compare
        compare(config,arguments.output,device)


if __name__=='__main__': main()
