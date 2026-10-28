"""Freeze the zero-training held-out evaluation without changing model selection."""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.download import sha256
from streaming.evaluate import save_json
from holdout23.data import read, source_hashes
from holdout23.report import findings


def main():
    output = ROOT / 'outputs/holdout23'
    audit, viewer, tests, presentation = (read(output / name) for name in
        ('audit.json', 'viewer-validation.json', 'test-results.json', 'presentation-provenance.json'))
    if not all(row['passed'] for row in (audit, viewer, tests)) or not presentation['visual_inspection_passed']:
        raise ValueError('A required Stage23 gate failed')
    for name, digest in audit['artifact_sha256'].items():
        if sha256(output / name) != digest:
            raise ValueError('Audited held-out evidence changed: ' + name)
    if (audit['new_optimizer_updates'] or audit['optimizer_instantiations'] or audit['normalization_updates']
            or audit['model_selection_performed'] or audit['locked_selected_arm'] != 'uniform_ce650'
            or audit['selection_stage'] != 22 or audit['test_scored_replays'] != 48
            or audit['test_audio_warmups'] != 4 or audit['target_cache_reproductions'] != 12
            or audit['checkpoint_files_unchanged'] != 4 or not audit['development_selection_byte_identical']
            or audit['production_promoted']):
        raise ValueError('Locked zero-training/test-selection scope differs')
    if (viewer['views'] != 48 or viewer['export_links'] != 240 or viewer['exact_json_downloads'] != 48
            or viewer['paired_switches'] != 48 or viewer['completion_gated_logs'] != 48
            or viewer['three_clock_event_checks'] != audit['exact_events']
            or viewer['active_playback_switches'] != 1 or viewer['javascript_errors']):
        raise ValueError('Browser coverage differs')
    if (viewer['review_html_sha256'] != sha256(output / 'review.html')
            or viewer['browser_check_source_sha256'] != sha256(ROOT / 'scripts/check_holdout23_viewer.cjs')
            or viewer['screenshot_sha256'] != sha256(output / 'review.png')
            or viewer['mobile_screenshot_sha256'] != sha256(output / 'review-mobile.png')
            or presentation['review_html_sha256'] != viewer['review_html_sha256']
            or presentation['screenshot_sha256'] != viewer['screenshot_sha256']
            or presentation['mobile_screenshot_sha256'] != viewer['mobile_screenshot_sha256']):
        raise ValueError('Browser/visual evidence is stale')
    for name, digest in presentation['numeric_artifact_sha256'].items():
        if sha256(output / name) != digest:
            raise ValueError('Presentation numbers changed')
    if any((output / name).read_text(encoding='utf-8') != findings(output) for name in ('report.md', 'findings.md')):
        raise ValueError('Held-out findings do not reproduce')
    if tests['failed'] or tests['skipped'] or tests['new_stage_tests'] != 18 or tests['passed_count'] < 688:
        raise ValueError('Full test gate differs')
    inputs = read(output / 'input-bindings.json')
    for name, digest in inputs.items():
        if sha256(ROOT / name) != digest:
            raise ValueError('Immutable input changed: ' + name)
    if (output / 'selection.json').read_bytes() != (ROOT / 'outputs/balance22/selection.json').read_bytes():
        raise ValueError('Copied development selection changed')
    run = read(output / 'test/run.json')
    for mode, name in run['checkpoint_paths'].items():
        if sha256(ROOT / name) != run['checkpoint_sha256'][mode]:
            raise ValueError('Locked model changed')
    extras = ['holdout23/audit.py', 'holdout23/report.py', 'tests/test_holdout23.py',
              'scripts/check_holdout23_viewer.cjs', 'scripts/verify_holdout23.py']
    paths = [p for p in output.rglob('*') if p.is_file() and p.name != 'verification.json']
    verification = {'stage': 23, 'client_date': '2026-10-28', 'new_optimizer_updates': 0,
        'optimizer_instantiations': 0, 'normalization_updates': 0, 'model_selection_performed': False,
        'production_promoted': False, 'selected_arm': None,
        'locked_selected_arm': 'uniform_ce650', 'research_selection': {'stage': 22, 'selected_arm': 'uniform_ce650'},
        'checkpoint_sha256': run['checkpoint_sha256'], 'test_scored_replays': 48, 'test_audio_warmups': 4,
        'target_cache_reproductions': 12, 'checkpoint_files_unchanged': 4,
        'development_selection_byte_identical': True, 'exact_notes': audit['exact_notes'], 'exact_events': audit['exact_events'],
        'prior_stage22_verification_sha256': sha256(ROOT / 'outputs/balance22/verification.json'),
        'pytest': tests, 'viewer': viewer, 'visual_inspection_passed': True,
        'artifact_sha256': {p.relative_to(output).as_posix(): sha256(p) for p in sorted(paths)},
        'code_sha256': {**source_hashes(), **{name: sha256(ROOT / name) for name in extras}}, 'input_sha256': inputs,
        'scope': 'Locked evaluation of existing historically evaluated progression3 data; shared players. No training, calibration, reselection, promotion or fresh-independent/electric-guitar generalization claim.'}
    path = output / 'verification.json'
    if path.exists():
        if read(path) != verification:
            raise ValueError('Completed Stage23 verification differs; no file rewritten')
    else:
        save_json(path, verification)
    print('Stage23 verification passed; bindings:', sum(len(verification[k]) for k in ('artifact_sha256', 'code_sha256', 'input_sha256')))


if __name__ == '__main__':
    main()
