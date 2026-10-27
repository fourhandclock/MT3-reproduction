import argparse
from pathlib import Path
from baseline.download import ROOT
from decoder14.workflow import read


def main():
    parser=argparse.ArgumentParser(description='Stage18 fixed head-crossing diagnosis; zero new training/inference')
    parser.add_argument('action',choices=['run','audit','report'])
    parser.add_argument('--config',type=Path,default=ROOT/'configs/headswap18.json')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/headswap18')
    args=parser.parse_args()
    if args.action=='run':
        from .workflow import run
        run(read(args.config),args.output)
    elif args.action=='audit':
        from .audit import audit
        from .workflow import save
        result=audit(args.output);save(args.output/'audit.json',result);print(result)
    else:
        from .report import write_report
        write_report(args.output)


if __name__=='__main__':main()
