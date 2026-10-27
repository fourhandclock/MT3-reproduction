"""Bind the final model-free diagnosis, presentation, tests and future-only proposal."""

import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'outputs/headswap18'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def check(root,bindings):
    for name,digest in bindings.items():
        if sha(root/name)!=digest:raise ValueError('Evidence changed: '+name)


def main():
    protocol=read(OUTPUT/'protocol.json');run=read(OUTPUT/'run.json');audit=read(OUTPUT/'audit.json')
    summary=read(OUTPUT/'summary.json');viewer=read(OUTPUT/'viewer-validation.json')
    presentation=read(OUTPUT/'presentation-provenance.json');tests=read(OUTPUT/'test-results.json')
    proposal=read(OUTPUT/'next-experiment.json')
    check(ROOT,protocol['source_sha256']);check(ROOT,protocol['input_sha256']);check(OUTPUT,audit['artifact_sha256'])
    check(OUTPUT,presentation['numeric_artifact_sha256']);check(ROOT,presentation['borrowed_source_sha256'])
    if not audit['passed'] or not viewer['passed']:raise ValueError('Audit/browser gate failed')
    if presentation['report_source_sha256']!=sha(ROOT/'headswap18/report.py'):raise ValueError('Report source changed')
    html=sha(OUTPUT/'review.html')
    if html!=viewer['review_html_sha256'] or html!=presentation['review_html_sha256']:raise ValueError('HTML QA is stale')
    if viewer['browser_check_source_sha256']!=sha(ROOT/'scripts/check_headswap18_viewer.cjs'):raise ValueError('Browser source changed')
    if viewer['screenshot_sha256']!=sha(OUTPUT/'review.png') or presentation['report_md_sha256']!=sha(OUTPUT/'report.md'):
        raise ValueError('Screenshot/report binding changed')
    coverage={'views':48,'export_links':288,'exact_json_downloads':48,'two_clock_event_checks':12144,
        'head_switches':96,'packet_gated_logs':48,'category_filters':288,'reference_rows_checked':3012,
        'note_seek_checks':48,'active_playback_switches':2}
    if any(viewer[k]!=v for k,v in coverage.items()) or viewer['javascript_errors']:raise ValueError('Browser coverage differs')
    for record in [protocol,run,audit,summary]:
        if (record['new_optimizer_updates']!=0 or record['neural_inference_calls']!=0 or record['test_inference_calls']!=0
                or record['production_promoted'] or record['model_selection_performed']):raise ValueError('Zero-budget/selection scope changed')
    if (audit['original_reproductions']!=24 or audit['crossed_cases']!=24 or audit['frame_cases']!=24
            or not audit['no_measured_completion_clock']):raise ValueError('Replay/clock coverage changed')
    if tests['passed']!=536 or tests['failed']!=0 or tests['new_stage_tests']!=19:raise ValueError('Test evidence differs')
    prior_root=ROOT/'outputs/multires17';prior=read(prior_root/'verification.json')
    check(prior_root,prior['artifact_sha256']);check(ROOT,prior['code_sha256'])
    prior_run=read(prior_root/'validation/run.json')
    for mode,path in prior_run['checkpoint_paths'].items():
        if sha(ROOT/path)!=prior['checkpoint_sha256'][mode]:raise ValueError('Prior model changed')
    if (proposal['status']!='proposed_not_trained' or proposal['training_executed'] or proposal['stage']!=19
            or proposal['future_total_new_optimizer_update_cap']!=500 or proposal['new_updates_per_arm']!=250
            or proposal['future_new_sampled_epochs_per_arm']!=5 or proposal['parent_lifetime_updates']!=250
            or proposal['future_lifetime_updates_per_arm']!=500 or proposal['new_sanity_updates']!=0
            or proposal['production_promotion'] or proposal['test_inference_allowed']):raise ValueError('Future proposal scope changed')
    if sha(ROOT/proposal['parent_checkpoint'])!=proposal['parent_checkpoint_sha256']:raise ValueError('Future parent changed')
    paths=[p for p in OUTPUT.rglob('*') if p.is_file() and p.name!='verification.json']
    extra=[ROOT/'headswap18/audit.py',ROOT/'headswap18/report.py',ROOT/'tests/test_headswap18.py',
           ROOT/'scripts/check_headswap18_viewer.cjs',Path(__file__).resolve()]
    verification={'stage':18,'client_date':'2026-10-20','pytest':tests,'new_optimizer_updates':0,
        'neural_inference_calls':0,'test_inference_calls':0,'production_promoted':False,'model_selection_performed':False,
        'validation_cases':48,'original_reproductions':24,'crossed_cases':24,'frame_cases':24,
        'exact_notes':audit['exact_notes'],'exact_events':audit['exact_events'],'reference_evidence_rows':3012,
        'viewer':coverage,'visual_inspection_passed':True,'no_measured_completion_clock':True,
        'prior_stage17_bindings_verified':True,'prior_stage17_verification_sha256':sha(prior_root/'verification.json'),
        'contrasts':summary['contrasts'],'original_note_changes':summary['original_note_changes'],
        'artifact_sha256':{p.relative_to(OUTPUT).as_posix():sha(p) for p in sorted(paths)},
        'code_sha256':{**protocol['source_sha256'],**{p.relative_to(ROOT).as_posix():sha(p) for p in extra}},
        'scope':'Reused development diagnosis with no new acoustic training/inference. Synthetic crossed heads have no measured completion clock. Future loss proposal has not trained.'}
    (OUTPUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:verification[k] for k in ['stage','new_optimizer_updates','neural_inference_calls',
        'validation_cases','exact_notes','exact_events','prior_stage17_bindings_verified']}))


if __name__=='__main__':main()
