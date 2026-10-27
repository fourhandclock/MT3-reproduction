"""Freeze Stage21 after numeric, export, complete-suite and inspected browser gates."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prior21.data import read, save, sha
from prior21.workflow import source_hashes
from prior21.report import findings


def main():
    output = ROOT / 'outputs/prior21'
    audit, viewer, tests, presentation = (read(output / name) for name in
        ('audit.json', 'viewer-validation.json', 'test-results.json', 'presentation-provenance.json'))
    if not all(value['passed'] for value in (audit, viewer, tests)) or not presentation['visual_inspection_passed']:
        raise ValueError('A required Stage21 gate failed')
    if (audit['source_sha256'] != source_hashes() or audit['validation_cases'] != 36
            or audit['active_frames_per_model'] != 22219 or audit['prior_confusion_reconstructions'] != 21
            or audit['scalar_confusion_reconstructions'] != 3):
        raise ValueError('Numeric audit coverage/source changed')
    for name, digest in audit['artifact_sha256'].items():
        if sha(output / name) != digest:
            raise ValueError('Audited output changed: ' + name)
    if (viewer['filter_states'] != 576 or viewer['model_predictor_views'] != 24
            or viewer['exact_json_downloads'] != 3 or viewer['source_case_links'] != 36 or viewer['javascript_errors']):
        raise ValueError('Browser coverage differs')
    for key, name in [('review_html_sha256', 'review.html'), ('screenshot_sha256', 'review.png'),
                      ('mobile_screenshot_sha256', 'review-mobile.png')]:
        if viewer[key] != sha(output / name) or presentation[key] != viewer[key]:
            raise ValueError('Browser/visual evidence is stale')
    if viewer['browser_source_sha256'] != sha(ROOT / 'scripts/check_prior21_viewer.cjs'):
        raise ValueError('Browser producer changed')
    if tests['failed'] or tests['skipped'] or tests['new_stage_tests'] != 27 or tests['passed_count'] < 633:
        raise ValueError('Full test gate differs')
    expected = findings(read(output / 'summary.json'))
    if any((output / name).read_text(encoding='utf-8') != expected for name in ('report.md', 'findings.md')):
        raise ValueError('Findings do not reproduce')
    inputs = read(output / 'input-bindings.json')
    for name, digest in inputs.items():
        if sha(ROOT / name) != digest:
            raise ValueError('Archived predecessor changed: ' + name)
    run = read(ROOT / 'outputs/fret19/validation/run.json')
    if run['selection']['selected_arm'] != 'control500':
        raise ValueError('Original research selection changed')
    paths = [p for p in output.rglob('*') if p.is_file() and p.name != 'verification.json']
    extras = ['tests/test_prior21.py', 'scripts/check_prior21_viewer.cjs', 'scripts/verify_prior21.py']
    verification = {'stage': 21, 'client_date': '2026-10-28', 'new_optimizer_updates': 0,
        'neural_inference_calls': 0, 'test_inference_calls': 0, 'new_clock_measurements': 0,
        'model_selection_performed': False, 'production_promoted': False, 'selected_arm': None,
        'research_selection': {'stage': 19, 'selected_arm': 'control500'},
        'checkpoint_sha256': run['checkpoint_sha256'],
        'prior_stage20_verification_sha256': sha(ROOT / 'outputs/coverage20/verification.json'),
        'validation_cases': 36, 'active_frames_per_model': 22219, 'supported_positions': 64,
        'pytest': tests, 'viewer': viewer, 'visual_inspection_passed': True,
        'artifact_sha256': {p.relative_to(output).as_posix(): sha(p) for p in sorted(paths)},
        'code_sha256': {**source_hashes(), **{name: sha(ROOT / name) for name in extras}},
        'input_sha256': inputs,
        'scope': 'GT-active oracle-support prior/confusion diagnosis on saved reused-development predictions; no new training, inference, selection, generalization or causal claim.'}
    path = output / 'verification.json'
    if path.exists():
        if read(path) != verification:
            raise ValueError('Completed Stage21 verification changed; no file rewritten')
    else:
        save(path, verification)
    print('Stage21 verification passed; bindings:', sum(len(verification[k]) for k in ('artifact_sha256', 'code_sha256', 'input_sha256')))


if __name__ == '__main__':
    main()
