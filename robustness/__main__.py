import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description='Guitar effect stress tests and one budget-capped augmentation experiment.')
    parser.add_argument('--config',default=str(ROOT/'configs/robustness.json'))
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('prepare');commands.add_parser('external')
    for name in ['baseline','train','evaluate','report']:
        command=commands.add_parser(name)
        command.add_argument('--output',default=str(ROOT/'outputs/robustness/augmented' if name=='train' else ROOT/'outputs/robustness'))
        command.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
        if name=='train': command.add_argument('--resume',action='store_true')
    arguments=parser.parse_args();config=json.loads(Path(arguments.config).read_text())
    for key,relative in [('NUMBA_CACHE_DIR','.cache/numba'),('TEMP','tmp/runtime'),('TMP','tmp/runtime')]:
        location=ROOT/relative;location.mkdir(parents=True,exist_ok=True);os.environ[key]=str(location)
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]=str(config['training']['cpu_threads'])
    if arguments.command=='prepare':
        from .data import prepare_cache
        prepare_cache(config)
    elif arguments.command=='external':
        from .external import prepare_external
        prepare_external()
    elif arguments.command=='report':
        from .report import write_report
        output=Path(arguments.output).resolve();folder=output/'evaluation'
        run=json.loads((folder/'run.json').read_text());summary=json.loads((folder/'summary.json').read_text())
        records={name:[json.loads(path.read_text()) for path in (folder/name).glob('*.json')] for name in run['models']}
        write_report(output,records,summary,run)
    else:
        import torch
        device=torch.device('cuda' if arguments.device=='auto' and torch.cuda.is_available() else
                            'cpu' if arguments.device=='auto' else arguments.device)
        if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable.')
        torch.set_num_threads(config['training']['cpu_threads'])
        if arguments.command=='train':
            from .training import train
            train(config,arguments.output,device,arguments.resume)
        else:
            from .evaluate import evaluate_models
            evaluate_models(config,arguments.output,device,only_baseline=arguments.command=='baseline')


if __name__=='__main__': main()
