"""Run the frozen-model stage-9 decoder experiment without training."""

import argparse
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/diagnosis9.json'))
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ['select', 'evaluate', 'report', 'audit']:
        command = commands.add_parser(name)
        command.add_argument('--output', default=str(ROOT / 'outputs/diagnosis9'))
        if name == 'evaluate':
            command.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
        if name == 'audit':
            command.add_argument('--recompute', action='store_true',
                                 help='Rescore selected and identity decoders on cached validation heads.')
    play = commands.add_parser('replay', help='Export notes and events using a frozen decoder recipe.')
    play.add_argument('audio')
    play.add_argument('--mode', choices=['strict', 'repair'], default='strict')
    play.add_argument('--output', default=str(ROOT / 'outputs/diagnosis9/example'))
    play.add_argument('--diagnostics-root', default=str(ROOT / 'outputs/diagnosis9'))
    play.add_argument('--seconds', type=float, default=30)
    play.add_argument('--packet-ms', type=float, default=20)
    play.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if config['optimizer_updates'] != 0 or config['selection_split'] != 'validation':
        parser.error('Stage 9 requires zero training updates and validation-only selection.')
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[name] = '4'
    os.environ['NUMBA_CACHE_DIR'] = str(ROOT / '.cache/numba')

    if args.command == 'report':
        from .report import write_report
        output = Path(args.output).resolve()
        directory = output / 'evaluation'
        records = [json.loads(path.read_text(encoding='utf-8'))
                   for path in sorted(directory.glob('results/*/*-metrics.json'))]
        write_report(output, records, json.loads((directory / 'summary.json').read_text()),
                     json.loads((directory / 'run.json').read_text()))
        return

    import torch
    torch.set_num_threads(4)
    if args.command == 'select':
        from .workflow import select
        select(config, args.output)
        return
    if args.command == 'audit':
        from .audit import audit
        result = audit(config, args.output, recompute=args.recompute)
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    from .audit import validate_selection
    validate_selection(config, args.diagnostics_root if args.command == 'replay' else args.output)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    if args.command == 'evaluate':
        from .workflow import evaluate_selected
        evaluate_selected(config, args.output, device)
        return
    if (not math.isfinite(args.seconds) or not math.isfinite(args.packet_ms)
            or args.seconds <= 0 or args.packet_ms <= 0):
        parser.error('Audio duration and packet size must be finite and positive.')
    from .predict_file import predict_file
    predict_file(args.audio, args.output, args.diagnostics_root, config,
                 args.seconds, args.packet_ms, device, args.mode)


if __name__ == '__main__':
    main()
