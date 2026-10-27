"""Freeze a completed coverage audit after numeric, export, test and browser gates."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coverage20.data import read, save, sha
from coverage20.workflow import source_hashes


def main():
    output = ROOT / 'outputs/coverage20'
    audit, viewer, tests, presentation = (read(output / name) for name in
        ('audit.json', 'viewer-validation.json', 'test-results.json', 'presentation-provenance.json'))
    if not all(row['passed'] for row in (audit, viewer, tests)) or not presentation['visual_inspection_passed']:
        raise ValueError('A required audit, browser, test or visual gate failed')
    if (audit['source_sha256'] != source_hashes() or audit['validation_cases'] != 36
            or audit['reference_note_rows'] != 2259 or audit['independently_reconstructed_histories'] != 2):
        raise ValueError('Audit source or coverage differs')
    for name, digest in audit['artifact_sha256'].items():
        if sha(output / name) != digest:
            raise ValueError('Audited output changed: ' + name)
    if viewer['filter_states'] != 420 or viewer['model_policy_views'] != 6 or viewer['javascript_errors']:
        raise ValueError('Browser coverage differs')
    if (viewer['review_html_sha256'] != sha(output / 'review.html')
            or viewer['browser_source_sha256'] != sha(ROOT / 'scripts/check_coverage20_viewer.cjs')
            or viewer['screenshot_sha256'] != sha(output / 'review.png')
            or viewer['mobile_screenshot_sha256'] != sha(output / 'review-mobile.png')
            or presentation['review_html_sha256'] != viewer['review_html_sha256']
            or presentation['screenshot_sha256'] != viewer['screenshot_sha256']
            or presentation['mobile_screenshot_sha256'] != viewer['mobile_screenshot_sha256']):
        raise ValueError('Browser/visual evidence is stale')
    from coverage20.report import findings
    expected_text = findings(read(output / 'coverage.json'), read(output / 'summary.json'))
    if any((output / name).read_text(encoding='utf-8') != expected_text for name in ('report.md', 'findings.md')):
        raise ValueError('Numeric findings no longer reproduce')
    if tests['failed'] or tests['skipped'] or tests['new_stage_tests'] != 26 or tests['passed_count'] < 606:
        raise ValueError('Full test gate differs')
    inputs = read(output / 'input-bindings.json')
    for name, digest in inputs.items():
        if sha(ROOT / name) != digest:
            raise ValueError('Archived input changed: ' + name)
    run = read(ROOT / 'outputs/fret19/validation/run.json')
    source = read(ROOT / 'outputs/fret19/verification.json')
    if run['checkpoint_sha256'] != source['checkpoint_sha256'] or run['selection']['selected_arm'] != 'control500':
        raise ValueError('Original model identities/selection changed')
    paths = [p for p in output.rglob('*') if p.is_file() and p.name != 'verification.json']
    extras = ['tests/test_coverage20.py', 'tests/test_project_check.py', 'scripts/check_project.py',
              'scripts/check_coverage20_viewer.cjs', 'scripts/verify_coverage20.py']
    code = {**source_hashes(), **{name: sha(ROOT / name) for name in extras}}
    verification = {'stage': 20, 'client_date': '2026-10-28', 'new_optimizer_updates': 0,
        'neural_inference_calls': 0, 'test_inference_calls': 0, 'new_clock_measurements': 0,
        'model_selection_performed': False, 'production_promoted': False, 'selected_arm': None,
        'research_selection': {'stage': 19, 'selected_arm': 'control500'},
        'checkpoint_sha256': run['checkpoint_sha256'],
        'prior_stage19_verification_sha256': sha(ROOT / 'outputs/fret19/verification.json'),
        'validation_cases': 36, 'reference_note_rows': 2259, 'pytest': tests,
        'independently_reconstructed_histories': 2, 'viewer': viewer, 'visual_inspection_passed': True,
        'artifact_sha256': {p.relative_to(output).as_posix(): sha(p) for p in sorted(paths)},
        'code_sha256': code, 'input_sha256': inputs,
        'scope': 'Counts and associations on frozen training targets/schedules and saved reused-development predictions. No training/inference, independent generalization, selection or causal claim.'}
    path = output / 'verification.json'
    if path.exists():
        if read(path) != verification:
            raise ValueError('Completed verification changed; no file was rewritten')
    else:
        save(path, verification)
    print('Stage20 verification passed; bindings:', sum(len(verification[k]) for k in ('artifact_sha256', 'code_sha256', 'input_sha256')))


if __name__ == '__main__':
    main()
