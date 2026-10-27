"""Bind final fret-loss evidence to the actual update budget and prior immutable stages."""

import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'outputs/fret19'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def check(root,bindings):
    for name,digest in bindings.items():
        if sha(root/name)!=digest:raise ValueError('Final evidence changed: '+name)


def main():
    protocol=read(OUTPUT/'protocol.json');run=read(OUTPUT/'validation/run.json');audit=read(OUTPUT/'audit.json')
    viewer=read(OUTPUT/'viewer-validation.json');presentation=read(OUTPUT/'presentation-provenance.json')
    tests=read(OUTPUT/'test-results.json');summary=read(OUTPUT/'validation/summary.json')
    if not audit['passed'] or not viewer['passed']:raise ValueError('Audit/browser gate failed')
    check(ROOT,protocol['source_sha256']);check(OUTPUT,audit['artifact_sha256'])
    check(OUTPUT,presentation['numeric_artifact_sha256']);check(ROOT,presentation['borrowed_source_sha256'])
    if presentation['report_source_sha256']!=sha(ROOT/'fret19/report.py'):raise ValueError('Report source changed')
    html=sha(OUTPUT/'review.html')
    if html!=viewer['review_html_sha256'] or html!=presentation['review_html_sha256']:raise ValueError('HTML QA is stale')
    if viewer['browser_check_source_sha256']!=sha(ROOT/'scripts/check_fret19_viewer.cjs'):raise ValueError('Browser source changed')
    if viewer['screenshot_sha256']!=sha(OUTPUT/'review.png') or presentation['report_md_sha256']!=sha(OUTPUT/'report.md'):
        raise ValueError('Screenshot/report binding changed')
    coverage={'views':36,'export_links':180,'exact_json_downloads':36,'paired_switches':36,
        'completion_gated_logs':36,'three_clock_event_checks':audit['exact_events'],'active_playback_switches':1}
    if any(viewer[k]!=v for k,v in coverage.items()) or viewer['javascript_errors']:raise ValueError('Browser coverage differs')
    if tests['passed']!=565 or tests['failed']!=0 or tests['new_stage_tests']!=29:raise ValueError('Test evidence differs')
    if (audit['new_optimizer_updates']!=500 or audit['historical_unique_parent_updates']!=250
            or audit['final_lifetime_updates_per_arm']!=500 or audit['new_sanity_updates']!=0
            or audit['validation_cases']!=36 or audit['parent_reproductions']!=12
            or audit['test_inference_calls']!=0 or audit['production_promoted']):raise ValueError('Budget/test/promotion scope changed')
    if audit['parent_probability_sets_bit_identical']!=12 or not audit['paired_final_rng_identical']:
        raise ValueError('Parent or paired-randomness reproduction differs')
    for mode,path in run['checkpoint_paths'].items():
        if sha(ROOT/path)!=run['checkpoint_sha256'][mode]:raise ValueError('Final model changed')
    prior_root=ROOT/'outputs/headswap18';prior=read(prior_root/'verification.json')
    check(prior_root,prior['artifact_sha256']);check(ROOT,prior['code_sha256'])
    if sha(ROOT/'outputs/multires17/verification.json')!=prior['prior_stage17_verification_sha256']:raise ValueError('Stage17 verification changed')
    check(ROOT,read(OUTPUT/'input-audit.json')['proposal_bindings'])
    paths=[p for p in OUTPUT.rglob('*') if p.is_file() and p.name!='verification.json']
    extra=[ROOT/'fret19/audit.py',ROOT/'fret19/report.py',ROOT/'tests/test_fret19.py',ROOT/'scripts/check_fret19_viewer.cjs',Path(__file__).resolve()]
    modes=summary['modes'];control=modes['control500'];treatment=modes['conditional_fret500']
    verification={'stage':19,'client_date':'2026-10-22','pytest':tests,'new_optimizer_updates':500,
        'new_updates_per_arm':250,'historical_unique_parent_updates':250,'final_lifetime_updates_per_arm':500,
        'new_sanity_updates':0,'additional_sampled_epochs_per_arm':5,'test_inference_calls':0,'production_promoted':False,
        'selected_arm':audit['selected_arm'],'selection_scope':'Fixed final argmax state F1 then onset F1; reused development data only.',
        'state_f1_treatment_minus_control':treatment['exact_state_f1']-control['exact_state_f1'],
        'onset_f1_treatment_minus_control':treatment['onset_f1']-control['onset_f1'],
        'validation_cases':36,'parent_probability_sets_bit_identical':12,'exact_notes':audit['exact_notes'],'exact_events':audit['exact_events'],
        'historical_model_optimizer_scaler_rng_restored':True,'paired_final_rng_identical':True,
        'auxiliary_active_support_per_epoch':audit['auxiliary_active_support_per_epoch'],
        'viewer':coverage,'visual_inspection_passed':True,'checkpoint_sha256':run['checkpoint_sha256'],
        'prior_stage18_bindings_verified':True,'prior_stage18_verification_sha256':sha(prior_root/'verification.json'),
        'artifact_sha256':{p.relative_to(OUTPUT).as_posix():sha(p) for p in sorted(paths)},
        'code_sha256':{**protocol['source_sha256'],**{p.relative_to(ROOT).as_posix():sha(p) for p in extra}},
        'scope':'Same parent and new exposure, one extra loss term. Tests/reports/audits perform no additional acoustic optimizer updates; no independent generalization claim.'}
    (OUTPUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:verification[k] for k in ['stage','new_optimizer_updates','selected_arm','validation_cases','exact_notes','exact_events','prior_stage18_bindings_verified']}))


if __name__=='__main__':main()
