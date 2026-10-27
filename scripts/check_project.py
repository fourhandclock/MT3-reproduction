"""Check the saved workspace without regenerating historical experiment evidence."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def local_path(root, name):
    """Only accept paths within the declared root, including through symlinks."""
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError(f'Expected a nonempty relative path: {name!r}')
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes its root: {name}')
    return path


def integrity(root):
    """Return checked file hashes and coverage; perform no writes or imports of models."""
    root = Path(root).resolve()
    checked = {}

    def bind(path, expected):
        if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
            raise ValueError(f'Invalid SHA256 for {path}')
        if not path.is_file():
            raise ValueError(f'Missing saved asset: {path}. Restore ignored assets from backup.')
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f'SHA256 mismatch: {path}')
        checked[path.relative_to(root).as_posix()] = actual

    index_path = root / 'experiments/index.json'
    index = json.loads(index_path.read_bytes())
    checked['experiments/index.json'] = sha256(index_path)
    if index['schema_version'] != 1:
        raise ValueError('Unsupported experiment index schema')
    stages = index['stages']
    if [row['stage'] for row in stages] != list(range(1, index['completed_through'] + 1)):
        raise ValueError('Completed stage index must be contiguous and ordered')
    for row in stages:
        if row['status'] != 'completed':
            raise ValueError('Unexpected incomplete stage in completed index')
        for key in ('module', 'config', 'output_root', 'review', 'report', 'verification'):
            if row[key] is None and key == 'verification':
                continue
            path = local_path(root, row[key])
            directory = key in ('module', 'output_root')
            if not (path.is_dir() if directory else path.is_file()):
                raise ValueError(f'Missing indexed {key}: {path}. Restore ignored assets from backup.')
    for row in index.get('planned', []):
        if not local_path(root, row['proposal']).is_file():
            raise ValueError('Missing indexed proposal')

    latest = index['latest_snapshot']
    if latest['stage'] != index['completed_through']:
        raise ValueError('Snapshot is not for the latest completed stage')
    snapshot_path = local_path(root, latest['path'])
    bind(snapshot_path, latest['sha256'])
    snapshot = json.loads(snapshot_path.read_bytes())
    if snapshot['stage'] != latest['stage']:
        raise ValueError('Snapshot stage differs from index')
    # Exact bytes matter: JSON equivalence alone does not preserve the receipt.
    bind(local_path(root, stages[-1]['verification']), latest['sha256'])
    count = 0
    for field, base in (('artifact_sha256', 'artifact_root'), ('code_sha256', 'code_root'),
                        ('input_sha256', 'code_root')):
        base_path = local_path(root, latest[base])
        for name, expected in snapshot.get(field, {}).items():
            bind(local_path(base_path, name), expected)
            count += 1
    if count != latest['binding_count']:
        raise ValueError('Snapshot binding count differs from index')
    default = index['default']
    bind(local_path(root, default['checkpoint']), default['checkpoint_sha256'])

    # The hashed run record supplies paths for checkpoints outside the latest output root.
    run_path = local_path(root, latest.get('model_record', latest['artifact_root'] + '/validation/run.json'))
    if run_path.relative_to(root).as_posix() not in checked:
        raise ValueError('Checkpoint run record must be hash-bound by the snapshot')
    run = json.loads(run_path.read_bytes())
    if run['checkpoint_sha256'] != snapshot['checkpoint_sha256']:
        raise ValueError('Snapshot/run checkpoint identities differ')
    if run['checkpoint_paths'].keys() != run['checkpoint_sha256'].keys():
        raise ValueError('Missing checkpoint path or hash')
    for mode, name in run['checkpoint_paths'].items():
        bind(local_path(root, name), run['checkpoint_sha256'][mode])
    research = index['latest_research']
    selection = snapshot.get('research_selection', {'stage': latest['stage'], 'selected_arm': snapshot.get('selected_arm')})
    if (research['stage'] != selection['stage'] or research['stage'] > latest['stage']
            or research['selected_arm'] != selection['selected_arm']
            or research['checkpoint'] != run['checkpoint_paths'][research['selected_arm']]):
        raise ValueError('Latest research selection differs from saved evidence')
    return {'indexed_stages': len(stages), 'snapshot_bindings': count,
            'unique_hashed_files': len(checked), 'checked_sha256': checked}


def runtime_env(root):
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[key] = '4'
    env['NUMBA_CACHE_DIR'] = str(root / '.cache/numba')
    env['UV_CACHE_DIR'] = str(root / '.cache/uv')
    env['PYTHONNOUSERSITE'] = '1'
    env.pop('PYTHONPATH', None)
    return env


def run_check(name, command, directory, root, env):
    print(f'Running {name}...', flush=True)
    started = time.perf_counter()
    result = subprocess.run(command, cwd=root, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    (directory / f'{name}.log').write_text(result.stdout, encoding='utf-8')
    print(result.stdout.encode('ascii', errors='backslashreplace').decode('ascii').rstrip(), flush=True)
    return {'passed': result.returncode == 0, 'returncode': result.returncode,
            'seconds': time.perf_counter() - started, 'log': f'{name}.log', 'command': command}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--integrity-only', action='store_true',
                        help='Check indexed paths and hashes; skip dependencies and tests (stdlib only).')
    args = parser.parse_args(argv)
    directory_root = ROOT / 'tmp/project-check'
    directory_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='verify-', dir=directory_root))
    report = {'schema_version': 1, 'started_utc': datetime.now(timezone.utc).isoformat(),
              'python': sys.executable, 'integrity_only': args.integrity_only,
              'historical_evidence_regenerated': False, 'checks': {}}
    try:
        before = integrity(ROOT)
        report['integrity'] = before
        print(f"Integrity passed: {before['indexed_stages']} indexed stages, "
              f"{before['snapshot_bindings']} snapshot bindings.", flush=True)
        if not args.integrity_only:
            env = runtime_env(ROOT)
            uv = shutil.which('uv')
            if not uv:
                raise ValueError('uv is required for the dependency check; no packages were installed.')
            report['checks']['dependencies'] = run_check(
                'dependencies', [uv, 'pip', 'check', '--python', sys.executable], directory, ROOT, env)
            report['checks']['pytest'] = run_check(
                'pytest', [sys.executable, '-m', 'pytest', '-q',
                           '--basetemp', str(directory / 'pytest'),
                           '--junitxml', str(directory / 'pytest.xml')], directory, ROOT, env)
            after = integrity(ROOT)
            if before != after:
                raise ValueError('Saved evidence changed while checks ran')
            report['historical_hashes_unchanged_after_checks'] = True
        report['passed'] = all(row['passed'] for row in report['checks'].values())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report['passed'] = False
        report['error'] = str(exc)
        print(f'Check failed: {exc}'.encode('ascii', errors='backslashreplace').decode('ascii'), flush=True)
    report_path = directory / 'report.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {report_path}", flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
