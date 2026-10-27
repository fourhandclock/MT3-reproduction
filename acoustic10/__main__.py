import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description='Stage10: 100 sanity + two200-update matched acoustic runs; validation only.')
    parser.add_argument('--config', default=str(ROOT / 'configs/acoustic10.json'))
    parser.add_argument('command', choices=['run', 'evaluate', 'report', 'audit'])
    parser.add_argument('--output', default=str(ROOT / 'outputs/acoustic10'))
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    output = Path(args.output).resolve()
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[name] = str(config['training']['cpu_threads'])
    os.environ['NUMBA_CACHE_DIR'] = str(ROOT / '.cache/numba')
    import torch
    torch.set_num_threads(config['training']['cpu_threads'])
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    if args.command == 'report':
        from .report import write_report
        directory = output / 'validation'
        records = [json.loads(path.read_text(encoding='utf-8'))
                   for path in sorted(directory.glob('results/*/*-metrics.json'))]
        if (directory / 'run.json').exists():
            write_report(output, records, json.loads((directory / 'summary.json').read_text(encoding='utf-8')),
                         json.loads((directory / 'run.json').read_text(encoding='utf-8')))
        else:
            write_report(output, [], {'modes': {}}, {'config': config,
                         'sanity': json.loads((output / 'sanity.json').read_text(encoding='utf-8')),
                         'training_summary': json.loads((output / 'training-summary.json').read_text(encoding='utf-8'))})
        return
    if args.command == 'audit':
        from .audit import audit
        from streaming.evaluate import save_json
        result = audit(output)
        save_json(output / 'audit.json', result)
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    from .workflow import evaluate_validation, run
    result = run(config, output, device) if args.command == 'run' else evaluate_validation(config, output, device)
    print(json.dumps(result if args.command == 'evaluate' else {'completed': True}, indent=2))


if __name__ == '__main__':
    main()
