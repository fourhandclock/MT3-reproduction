import argparse
from pathlib import Path

from baseline.download import ROOT
from streaming.evaluate import save_json
from .workflow import read


def main():
    parser = argparse.ArgumentParser(description='Stage14: frozen cached decoder calibration, zero training.')
    parser.add_argument('action', choices=['run', 'report', 'audit'])
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/decoder14.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/decoder14')
    args = parser.parse_args()
    if args.action == 'run':
        from .workflow import run
        run(read(args.config), args.output)
        from .report import write_report
        write_report(args.output)
    elif args.action == 'report':
        from .report import write_report
        write_report(args.output)
    else:
        from .audit import audit
        result = audit(args.output)
        save_json(args.output / 'audit.json', result)
        print(result)


if __name__ == '__main__':
    main()
