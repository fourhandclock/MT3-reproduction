"""Failure checks for the new read-only evidence checker; no research assets needed."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts.check_project import integrity, local_path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def workspace(tmp_path):
    def write(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
        return path

    source = write('stage/model.py', b'preserved producer\r\n')
    config = write('configs/stage.json', {})
    write('outputs/stage/review.html', b'review')
    write('outputs/stage/report.md', b'report')
    child = write('outputs/stage/child.pt', b'child')
    parent = write('outputs/prior/parent.pt', b'parent')
    default = write('outputs/default.pt', b'default')
    paths = {'parent': 'outputs/prior/parent.pt', 'child': 'outputs/stage/child.pt'}
    hashes = {'parent': sha(parent), 'child': sha(child)}
    run = write('outputs/stage/validation/run.json', {'checkpoint_paths': paths, 'checkpoint_sha256': hashes})
    snapshot = {'stage': 1, 'selected_arm': 'child', 'checkpoint_sha256': hashes,
                'artifact_sha256': {'child.pt': sha(child), 'validation/run.json': sha(run)},
                'code_sha256': {'stage/model.py': sha(source), 'configs/stage.json': sha(config)}}
    saved = write('experiments/snapshot.json', snapshot)
    write('outputs/stage/verification.json', saved.read_bytes())
    index = {'schema_version': 1, 'completed_through': 1,
             'stages': [{'stage': 1, 'status': 'completed', 'module': 'stage', 'config': 'configs/stage.json',
                         'output_root': 'outputs/stage', 'review': 'outputs/stage/review.html',
                         'report': 'outputs/stage/report.md', 'verification': 'outputs/stage/verification.json'}],
             'latest_snapshot': {'stage': 1, 'path': 'experiments/snapshot.json', 'sha256': sha(saved),
                                 'artifact_root': 'outputs/stage', 'code_root': '.', 'binding_count': 4},
             'default': {'checkpoint': 'outputs/default.pt', 'checkpoint_sha256': sha(default)},
             'latest_research': {'stage': 1, 'selected_arm': 'child', 'checkpoint': paths['child']}}
    write('experiments/index.json', index)
    return tmp_path


def test_integrity_checks_bytes_and_external_parent_without_writing(workspace):
    before = {p: p.read_bytes() for p in workspace.rglob('*') if p.is_file()}
    result = integrity(workspace)
    assert result['snapshot_bindings'] == 4
    assert 'outputs/prior/parent.pt' in result['checked_sha256']
    assert before == {p: p.read_bytes() for p in workspace.rglob('*') if p.is_file()}


@pytest.mark.parametrize('name', ['stage/model.py', 'outputs/stage/child.pt', 'outputs/prior/parent.pt',
                                 'outputs/default.pt', 'outputs/stage/verification.json', 'experiments/snapshot.json'])
def test_changed_producer_checkpoint_or_receipt_fails(workspace, name):
    path = workspace / name
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        integrity(workspace)


def test_missing_ignored_assets_fail_instead_of_skipping(workspace):
    (workspace / 'outputs/stage/child.pt').unlink()
    with pytest.raises(ValueError, match='Missing saved asset'):
        integrity(workspace)


@pytest.mark.parametrize('change', ['count', 'stage_gap', 'selection', 'path_escape'])
def test_inconsistent_registry_fails(workspace, change):
    path = workspace / 'experiments/index.json'
    index = json.loads(path.read_bytes())
    if change == 'count':
        index['latest_snapshot']['binding_count'] = 3
    elif change == 'stage_gap':
        index['completed_through'] = 2
    elif change == 'selection':
        index['latest_research']['checkpoint'] = 'outputs/default.pt'
    else:
        index['stages'][0]['module'] = '../outside'
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError):
        integrity(workspace)


def test_absolute_and_escaping_paths_are_rejected(tmp_path):
    for name in ('../outside', str(tmp_path.resolve())):
        with pytest.raises(ValueError):
            local_path(tmp_path, name)
