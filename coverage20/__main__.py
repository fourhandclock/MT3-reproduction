import argparse

from .data import ROOT, read


def main():
    parser = argparse.ArgumentParser(description='Zero-training label-coverage audit of saved Stage19 errors.')
    parser.add_argument('command', choices=('run', 'audit', 'report'))
    args = parser.parse_args()
    config = read(ROOT / 'configs/coverage20.json')
    output = ROOT / 'outputs/coverage20'
    if args.command == 'run':
        from .workflow import run
        run(config, output)
    elif args.command == 'audit':
        from .audit import audit
        audit(config, output)
    else:
        from .report import report
        report(output)


if __name__ == '__main__':
    main()
