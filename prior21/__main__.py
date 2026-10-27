import argparse
from .data import ROOT, no_neural_runtime, read


def main():
    parser = argparse.ArgumentParser(description='Zero-training comparison of saved fret decisions and training priors.')
    parser.add_argument('command', choices=('run', 'audit', 'report'))
    args = parser.parse_args()
    no_neural_runtime()
    config = read(ROOT / 'configs/prior21.json')
    output = ROOT / 'outputs/prior21'
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
