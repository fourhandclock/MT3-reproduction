"""Bind final Stage17 evidence and verify that prior frozen results stay intact."""

import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'outputs/multires17'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def check_bindings(root,bindings):
    for name,digest in bindings.items():
        if sha(root/name)!=digest:raise ValueError('Final evidence changed: '+name)


def main():
    protocol=read(OUTPUT/'protocol.json');audit=read(OUTPUT/'audit.json')
    run=read(OUTPUT/'validation/run.json');viewer=read(OUTPUT/'viewer-validation.json')
    presentation=read(OUTPUT/'presentation-provenance.json');tests=read(OUTPUT/'test-results.json')
    if not audit['passed'] or not viewer['passed']:raise ValueError('Audit/browser gate failed')
    check_bindings(ROOT,protocol['source_sha256']);check_bindings(OUTPUT,audit['artifact_sha256'])
    check_bindings(OUTPUT,presentation['numeric_artifact_sha256']);check_bindings(ROOT,presentation['borrowed_source_sha256'])
    if presentation['report_source_sha256']!=sha(ROOT/'multires17/report.py'):raise ValueError('Report source changed')
    html=sha(OUTPUT/'review.html')
    if html!=viewer['review_html_sha256'] or html!=presentation['review_html_sha256']:raise ValueError('HTML QA is stale')
    if viewer['browser_check_source_sha256']!=sha(ROOT/'scripts/check_multires17_viewer.cjs'):raise ValueError('Browser script changed')
    if viewer['screenshot_sha256']!=sha(OUTPUT/'review.png') or presentation['report_md_sha256']!=sha(OUTPUT/'report.md'):
        raise ValueError('Screenshot/report binding changed')
    if (viewer['views']!=24 or viewer['export_links']!=120 or viewer['exact_json_downloads']!=24
            or viewer['paired_switches']!=24 or viewer['completion_gated_logs']!=24
            or viewer['three_clock_event_checks']!=audit['exact_events'] or viewer['javascript_errors']
            or viewer.get('active_playback_switches')!=1):raise ValueError('Incomplete browser coverage')
    for mode,path in run['checkpoint_paths'].items():
        if sha(ROOT/path)!=run['checkpoint_sha256'][mode]:raise ValueError('Final checkpoint changed')
    prior_root=ROOT/'outputs/supervision16';prior=read(prior_root/'verification.json')
    check_bindings(prior_root,prior['artifact_sha256']);check_bindings(ROOT,prior['code_sha256'])
    prior_run=read(prior_root/'validation/run.json')
    for mode,path in prior_run['checkpoint_paths'].items():
        if sha(ROOT/path)!=prior['checkpoint_sha256'][mode]:raise ValueError('Prior checkpoint changed')
    if tests['passed']!=517 or tests['failed']!=0:raise ValueError('Test evidence differs')
    if audit['new_optimizer_updates']!=500 or run['test_inference_calls']!=0 or run['production_promoted']:
        raise ValueError('Budget/test/promotion scope differs')
    paths=[p for p in OUTPUT.rglob('*') if p.is_file() and p.name!='verification.json']
    code={**protocol['source_sha256'],**{p.relative_to(ROOT).as_posix():sha(p) for p in
        [ROOT/'multires17/audit.py',ROOT/'multires17/report.py',ROOT/'tests/test_multires17.py',
         ROOT/'scripts/check_multires17_viewer.cjs',Path(__file__).resolve()]}}
    modes=read(OUTPUT/'validation/summary.json')['modes'];control=modes['short_short'];treatment=modes['short_long']
    verification={'stage':17,'client_date':'2026-10-20','pytest':tests,
        'new_optimizer_updates':500,'new_updates_per_arm':250,'historical_warmstart_updates':0,
        'final_lifetime_updates_per_arm':250,'new_sanity_updates':0,'sampled_crop_epochs_per_arm':5,
        'test_inference_calls':0,'production_promoted':False,'selected_arm':audit['selected_arm'],
        'selection_scope':'State-first rule retains short_short; long plane helps note scores but lowers state score. Reused development clips only.',
        'state_f1_treatment_minus_control':treatment['exact_state_f1']-control['exact_state_f1'],
        'onset_f1_treatment_minus_control':treatment['onset_f1']-control['onset_f1'],
        'validation_cases':24,'frontend_parity_cases':24,'exact_notes':audit['exact_notes'],'exact_events':audit['exact_events'],
        'shared_initial_training_state_identical':True,'paired_final_rng_identical':True,
        'viewer':{k:viewer[k] for k in ['views','export_links','exact_json_downloads','paired_switches',
            'active_playback_switches','three_clock_event_checks']},
        'visual_inspection_passed':True,'checkpoint_sha256':run['checkpoint_sha256'],
        'prior_stage16_bindings_verified':True,'prior_stage16_verification_sha256':sha(prior_root/'verification.json'),
        'artifact_sha256':{p.relative_to(OUTPUT).as_posix():sha(p) for p in sorted(paths)},'code_sha256':code,
        'scope':'Fixed fresh paired experiment. No new acoustic training was used for tests, reports, audits or browser verification.'}
    (OUTPUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:verification[k] for k in ['stage','new_optimizer_updates','selected_arm','validation_cases','exact_notes','exact_events','prior_stage16_bindings_verified']}))


if __name__=='__main__':main()
