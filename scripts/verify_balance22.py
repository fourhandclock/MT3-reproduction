"""Freeze the actual 450-update trial and its inspected four-model review."""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.download import sha256
from decoder14.workflow import read
from streaming.evaluate import save_json
from balance22.training import producer_hashes
from balance22.report import findings


def main():
    output = ROOT / 'outputs/balance22'
    audit, viewer, tests, presentation = (read(output / name) for name in
        ('audit.json', 'viewer-validation.json', 'test-results.json', 'presentation-provenance.json'))
    if not all(row['passed'] for row in (audit, viewer, tests)) or not presentation['visual_inspection_passed']:
        raise ValueError('A required Stage22 gate failed')
    for name, digest in audit['artifact_sha256'].items():
        if sha256(output / name) != digest:
            raise ValueError('Audited numeric evidence changed: ' + name)
    if (audit['new_optimizer_updates'] != 450 or audit['final_lifetime_updates_per_arm'] != 650
            or audit['historical_unique_parent_updates'] != 500 or audit['new_sanity_updates']
            or audit['validation_cases'] != 48 or audit['parent_probability_sets_bit_identical'] != 12
            or not audit['paired_final_rng_identical'] or audit['test_inference_calls'] or audit['production_promoted']):
        raise ValueError('Actual budget, reproduction or scope differs')
    if (viewer['views'] != 48 or viewer['export_links'] != 240 or viewer['exact_json_downloads'] != 48
            or viewer['paired_switches'] != 48 or viewer['completion_gated_logs'] != 48
            or viewer['three_clock_event_checks'] != audit['exact_events'] or viewer['active_playback_switches'] != 1
            or viewer['javascript_errors']):
        raise ValueError('Browser coverage differs')
    if (viewer['review_html_sha256'] != sha256(output / 'review.html')
            or viewer['browser_check_source_sha256'] != sha256(ROOT / 'scripts/check_balance22_viewer.cjs')
            or viewer['screenshot_sha256'] != sha256(output / 'review.png')
            or viewer['mobile_screenshot_sha256'] != sha256(output / 'review-mobile.png')
            or presentation['review_html_sha256'] != viewer['review_html_sha256']
            or presentation['screenshot_sha256'] != viewer['screenshot_sha256']
            or presentation['mobile_screenshot_sha256'] != viewer['mobile_screenshot_sha256']):
        raise ValueError('Browser/visual evidence is stale')
    for name, digest in presentation['numeric_artifact_sha256'].items():
        if sha256(output / name) != digest:
            raise ValueError('Presentation numeric source changed')
    expected = findings(output)
    if any((output / name).read_text(encoding='utf-8') != expected for name in ('report.md', 'findings.md')):
        raise ValueError('Findings no longer reproduce')
    if tests['failed'] or tests['skipped'] or tests['new_stage_tests'] != 37 or tests['passed_count'] < 670:
        raise ValueError('Full test gate differs')
    inputs = read(output / 'input-bindings.json')
    for name, digest in inputs.items():
        if sha256(ROOT / name) != digest:
            raise ValueError('Archived predecessor changed: ' + name)
    run = read(output / 'validation/run.json')
    code = {**producer_hashes(), **{name: sha256(ROOT / name) for name in
        ['balance22/evaluation.py', 'balance22/audit.py', 'balance22/report.py', 'tests/test_balance22.py',
         'scripts/check_balance22_viewer.cjs', 'scripts/verify_balance22.py']}}
    paths = [p for p in output.rglob('*') if p.is_file() and p.name != 'verification.json']
    verification = {'stage': 22, 'client_date': '2026-10-28', 'new_optimizer_updates': 450,
        'new_updates_per_arm': 150, 'historical_unique_parent_updates': 500, 'final_lifetime_updates_per_arm': 650,
        'new_sanity_updates': 0, 'additional_sampled_epochs_per_arm': 3, 'test_inference_calls': 0,
        'production_promoted': False, 'selected_arm': audit['selected_arm'],
        'research_selection': {'stage': 22, 'selected_arm': audit['selected_arm']},
        'selection_scope': run['selection']['rule'], 'checkpoint_sha256': run['checkpoint_sha256'],
        'validation_cases': 48, 'parent_probability_sets_bit_identical': 12,
        'exact_notes': audit['exact_notes'], 'exact_events': audit['exact_events'],
        'paired_final_rng_identical': True, 'pre_training_amendment': audit['pre_training_amendment'],
        'active_supervised_support_per_epoch': audit['active_supervised_support_per_epoch'],
        'prior_stage21_verification_sha256': sha256(ROOT / 'outputs/prior21/verification.json'),
        'pytest': tests, 'viewer': viewer, 'visual_inspection_passed': True,
        'artifact_sha256': {p.relative_to(output).as_posix(): sha256(p) for p in sorted(paths)},
        'code_sha256': code, 'input_sha256': inputs,
        'scope': 'Matched capped continuations and fixed development selection, with one recorded training-only pre-training zero-support amendment. No new independent test or default promotion.'}
    path = output / 'verification.json'
    if path.exists():
        if read(path) != verification:
            raise ValueError('Completed Stage22 verification changed; no file rewritten')
    else:
        save_json(path, verification)
    print('Stage22 verification passed; bindings:', sum(len(verification[k]) for k in ('artifact_sha256', 'code_sha256', 'input_sha256')))


if __name__ == '__main__':
    main()
