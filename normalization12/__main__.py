import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description='Stage12 CNN frame-local group-count comparison;250 newupdates.')
    parser.add_argument('--config', default=str(ROOT / 'configs/normalization12.json'))
    parser.add_argument('command', choices=['run', 'evaluate', 'report', 'audit'])
    parser.add_argument('--output', default=str(ROOT / 'outputs/normalization12'))
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    output = Path(args.output).resolve()
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[name] = str(config['training']['cpu_threads'])
    os.environ['NUMBA_CACHE_DIR'] = str(ROOT / '.cache/numba')
    import torch
    torch.set_num_threads(config['training']['cpu_threads'])
    if args.command == 'report':
        from .report import write_report
        directory = output / 'validation'
        records = [json.loads(path.read_text(encoding='utf-8')) for path in sorted(directory.glob('results/*/*-metrics.json'))]
        write_report(output, records, json.loads((directory / 'summary.json').read_text(encoding='utf-8')),
                     json.loads((directory / 'run.json').read_text(encoding='utf-8')))
        return
    if args.command == 'audit':
        from .audit import audit
        from streaming.evaluate import save_json
        result = audit(output)
        save_json(output / 'audit.json', result)
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    from .workflow import evaluate_validation, run
    result = run(config, output, device) if args.command == 'run' else evaluate_validation(config, output, device)
    print(json.dumps({'completed': True, 'command': args.command}, indent=2))


if __name__ == '__main__':
    main()
