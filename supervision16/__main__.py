import argparse
from pathlib import Path

import torch

from baseline.download import ROOT
from decoder14.workflow import read
from streaming.evaluate import save_json


def main():
    parser=argparse.ArgumentParser(description='Stage16 matched loss supervision;500 total new updates.')
    parser.add_argument('action',choices=['run','evaluate','report','audit'])
    parser.add_argument('--config',type=Path,default=ROOT/'configs/supervision16.json')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/supervision16')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args();torch.set_num_threads(4)
    if args.action in ['run','evaluate']:
        from .workflow import run,evaluate_validation
        (run if args.action=='run' else evaluate_validation)(read(args.config),args.output,torch.device(args.device))
    elif args.action=='audit':
        from .audit import audit
        result=audit(args.output);save_json(args.output/'audit.json',result);print(result)
    else:
        from .report import write_report
        write_report(args.output)


if __name__=='__main__':main()
