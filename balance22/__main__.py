import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description='Stage22 three-arm matched 450-update class-weighting trial.')
    parser.add_argument('command', choices=('run', 'evaluate', 'audit', 'report'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '4'
    for key, name in [('NUMBA_CACHE_DIR', '.cache/numba'), ('TEMP', 'tmp/runtime'), ('TMP', 'tmp/runtime')]:
        path = ROOT / name
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    import torch
    from decoder14.workflow import read
    torch.set_num_threads(4)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; full saved AMP/RNG state requires the original GPU setup')
    config, output = read(ROOT / 'configs/balance22.json'), ROOT / 'outputs/balance22'
    if args.command == 'run':
        from .workflow import run
        run(config, output, torch.device(args.device))
    elif args.command == 'evaluate':
        from .evaluation import evaluate_validation
        evaluate_validation(config, output, torch.device(args.device))
    elif args.command == 'audit':
        from .audit import audit
        audit(config, output)
    else:
        from .report import write_report
        write_report(output)


if __name__ == '__main__':
    main()
