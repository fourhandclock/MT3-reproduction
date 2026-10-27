"""Local prior comparison, full supported-class confusion inspection and exports."""

import base64
import json
from pathlib import Path
from urllib.parse import quote

from .data import MODES, PREDICTORS, ROOT, read, save


def pct(value):
    return 'n/a' if value is None else f'{100 * (0 if abs(value) < 5e-7 else value):.2f}%'


def findings(summary):
    control = summary['modes']['control500']
    model = control['predictors']['model_best_active']
    prior = control['predictors']['presentations_top1']
    secondary = control['predictors']['unique_top1']
    lines = ['# Stage 21: saved fret predictions versus training priors', '',
        'Zero new training updates, zero neural inference calls and no model/baseline selection.', '',
        f"On GT-active support, control500 best-active fret accuracy is {pct(model['accuracy_frame_weighted'])}, "
        f"versus {pct(prior['accuracy_frame_weighted'])} for the prespecified presentation-count majority prior "
        f"({100 * control['primary_comparison']['model_minus_prior_frame_accuracy']:.2f} percentage points higher). "
        f"Equal-supported-position accuracy is {pct(model['accuracy_equal_supported_position'])} versus "
        f"{pct(prior['accuracy_equal_supported_position'])}.", '',
        f"The model chooses its training-modal fret on {pct(model['fraction_on_training_modal'])} of active GT string-frames. "
        f"Its own output/GT marginal independence expectation is {pct(model['independent_marginal_expected_accuracy'])}; "
        f"observed alignment exceeds that expectation by {100 * model['accuracy_minus_independent_marginals']:.2f} percentage points. "
        'This shows some alignment beyond frequency marginals, alongside concentrated predictions; it does not identify a causal mechanism.', '',
        f"The declared unique-count and available-count majority priors each reach {pct(secondary['accuracy_frame_weighted'])} "
        'for the control history. They are reported as fixed secondary views; the presentation-count prior remains primary. '
        'The parent and children have different recorded histories, so their majority frets can differ.', '',
        '## All declared comparisons', '',
        '| Saved model | Predictor | Frame-weighted accuracy | Equal supported-position accuracy | Equal-string accuracy |',
        '| --- | --- | ---: | ---: | ---: |']
    for mode in MODES:
        for name, row in summary['modes'][mode]['predictors'].items():
            lines.append(f"| {mode} | {name} | {pct(row['accuracy_frame_weighted'])} | "
                         f"{pct(row['accuracy_equal_supported_position'])} | {pct(row['accuracy_equal_supported_string'])} |")
    lines += ['', 'Each comparison uses 22,219 valid active string-frames and 64 supported string/fret positions across the same twelve development recordings. The model takes argmax over active frets0-24, even when inactive25 has the greatest full-state probability. GT-active conditioning removes the activity decision and is an oracle diagnostic, not end-to-end transcription accuracy.', '',
        'Training priors come only from each model\'s frozen Stage20 history. Top1 is the most frequent active fret per string; ties use the lowest fret. The three count sources are repeated presentations, unique supervised source-frame/string identities and all available active labels. Distribution rows are analytic expected confusions, with fractional expected mass, not sampled predictions. Uniform25 has expected accuracy4%. No prior source or setting was fitted to validation.', '',
        'Equal-position accuracy averages class recalls over GT-supported positions only. Unsupported positions retain null accuracy. Equal-string accuracy weights supported strings equally. Frame-weighted accuracy weights every eligible active frame equally. These denominators must not be interchanged.', '',
        'The independence expectation keeps each string\'s GT and predicted fret marginals but assumes their pairing is independent. It is an analytic association diagnostic, with no shuffled model run, significance test or independence claim about neighbouring frames.', '',
        '## Paired development clips against the primary prior', '',
        '| Model | Wins / ties / losses | Mean clip difference, percentage points | Pooled frame difference, percentage points |',
        '| --- | --- | ---: | ---: |']
    for mode in MODES:
        row = summary['modes'][mode]['primary_comparison']
        lines.append(f"| {mode} | {row['model_wins']} / {row['ties']} / {row['model_losses']} | "
                     f"{100 * row['mean_paired_clip_difference']:.2f} | {100 * row['model_minus_prior_frame_accuracy']:.2f} |")
    lines += ['', 'Clip means and pooled frame differences use different weights. Full within-string25x25 confusion matrices retain all predicted frets. CSVs contain every prediction column for every GT-supported position, including zero-valued cells.', '',
        '## Limits and next proposal', '', summary['limitations'], '',
        'Supported positions and correlated frames do not measure independent tonal, instrument or musical-context diversity. Concentration and prior gaps do not prove that class imbalance caused the errors, or that balancing will improve note scores. The model retains some alignment beyond frequency marginals. Stage19\'s control500 remains the research selection; this conditional diagnostic does not select another model.', '',
        'Stage22 is proposed as a matched three-arm, 450-update experiment: original loss, uniform active-only fret CE, and the same auxiliary term with clipped inverse-square-root class weights. All arms start from the same saved control500 state and receive the same three sampled-crop epochs. The unique-supervised training counts supply fixed weights. This is a hypothesis test, not a demonstrated improvement, and has not run.', '',
        'No new notes, processing clocks, latency measurements, calibration search or independent test inference were produced.', '']
    return '\n'.join(lines)


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Stage21 - Fret priors</title>
<style>:root{font:15px Segoe UI,Arial,sans-serif;color:#e4edf4;background:#111c27}body{max-width:1180px;margin:auto;padding:30px}h1{font-size:30px}h2{font-size:21px}.muted,footer{color:#adc1d0;line-height:1.6}.badge{display:inline-block;background:#254052;padding:7px 12px;border-radius:18px;margin:0 6px 8px 0}section{background:#192b3a;border:1px solid #345164;border-radius:10px;padding:20px;margin-top:18px}label{display:inline-block;margin:0 16px 12px 0}select,button{background:#233d4e;color:#f0f7fa;border:1px solid #56788d;padding:8px;border-radius:5px;max-width:100%}a{color:#8dd9e7}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.card{background:#213c4b;border-left:3px solid #6bd1bd;padding:12px}.big{display:block;font-size:25px;margin:6px 0}.small{font-size:12px;color:#b3cbd8}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;text-align:right;white-space:nowrap;border-bottom:1px solid #355264}th:first-child,td:first-child{text-align:left}th{color:#b1dfeb;background:#233e50}tr.selected{background:#294757}#heatmap th,#heatmap td{min-width:36px;padding:7px 4px;font-size:11px}.exports{display:flex;gap:13px;flex-wrap:wrap}footer{margin-top:20px}@media(max-width:700px){body{padding:16px}.cards{grid-template-columns:1fr}section{padding:14px}h1{font-size:25px}}</style>
<h1>Stage 21: how much does the fret model beat frequency?</h1><p class="muted">Saved predictions versus fixed training priors. Scores condition on ground-truth active strings; they do not include activity detection or note decoding.</p>
<span class="badge">0 training updates</span><span class="badge">0 neural inference calls</span><span class="badge">12 reused development clips</span><span class="badge">No new selection</span>
<section><label>Saved model <select id="model"><option>parent250</option><option selected>control500</option><option>conditional_fret500</option></select></label><p id="history" class="muted"></p><div class="cards"><div class="card">Model / primary majority prior<strong id="primary" class="big"></strong><span class="small">Frame-weighted, GT-active accuracy</span></div><div class="card">Model / prior, equal positions<strong id="macro" class="big"></strong><span class="small">64 supported string/fret positions</span></div><div class="card">Model chooses its training-modal fret<strong id="modal" class="big"></strong><span class="small">All active GT frames, own-history presentation prior</span></div></div></section>
<section><h2>Every fixed predictor</h2><p class="muted">Top1 priors always choose a training-majority fret. Distribution priors show analytic expected accuracy. The primary comparison uses repeated presentations; no baseline is selected here.</p><div class="scroll"><table id="scores"><thead><tr><th>Predictor</th><th>Frame weighted</th><th>Equal positions</th><th>Equal strings</th><th>Kind</th></tr></thead><tbody></tbody></table></div></section>
<section><h2>Inspect within-string confusion</h2><label>Predictor <select id="predictor">__OPTIONS__</select></label><label>String <select id="string"><option value="0">0 - low E</option><option value="1">1 - A</option><option value="2">2 - D</option><option value="3">3 - G</option><option value="4">4 - B</option><option value="5">5 - high E</option></select></label><label><input type="checkbox" id="normalize" checked> Row percentages</label><label><input type="checkbox" id="all-rows"> Include unsupported GT frets</label><p id="concentration" class="muted"></p><p class="small">Rows = ground-truth frets. Columns = predicted frets0-24. Green cells are correct; orange cells are errors. Blank cells have zero mass. Hover for exact mass and denominator; expected prior values may be fractional.</p><div class="scroll"><table id="heatmap"><thead></thead><tbody></tbody></table></div></section>
<section><h2>Ground-truth positions for the selected string</h2><div class="scroll"><table id="classes"><thead><tr><th>GT fret</th><th>Active GT frames</th><th>Correct mass</th><th>Class recall</th><th>Predicted mass</th><th>Unique training frames</th></tr></thead><tbody></tbody></table></div></section>
<section><h2>Paired clip differences: model minus primary majority prior</h2><p id="paired" class="muted"></p><div class="scroll"><table id="clips"><thead><tr><th>Clip</th><th>Accuracy difference, percentage points</th></tr></thead><tbody></tbody></table></div></section>
<section><h2>Evidence and exports</h2><div class="exports">__LINKS__<button id="download-confusions">Download exact confusion JSON</button></div><p class="muted"><a href="report.md">Findings and methods</a> · <a href="../coverage20/review.html">Stage20 coverage</a> · <a href="../fret19/review.html">Original audio and notes</a></p></section>
<footer>GT-active conditioning is an oracle diagnostic. A high majority-prior score is possible without using audio. Prediction concentration and marginal comparisons do not establish causes or independent generalization. The Stage19 model selection is preserved; Stage22 remains a proposal.</footer>
<script>const data=__DATA__,exactBytes='__BYTES__';const $=id=>document.getElementById(id),pct=x=>x===null?'—':(100*(Math.abs(x)<5e-7?0:x)).toFixed(2)+'%',mass=x=>Number.isInteger(x)?String(x):x.toFixed(2);window.current=null;
function render(){const mode=$('model').value,predictor=$('predictor').value,string=Number($('string').value),model=data.summary.modes[mode],detail=model.predictors[predictor],network=model.predictors.model_best_active,prior=model.predictors.presentations_top1,normalize=$('normalize').checked,all=$('all-rows').checked;const rows=detail.classes.filter(r=>r.string===string&&(all||r.reference_active_frames));const matrix=data.confusions[mode][predictor][string];window.current={mode,predictor,string,detail,rows,matrix};
 $('history').textContent='Training history: '+model.history.replaceAll('_','–')+'. All predictors use the same 22,219 GT-active string-frames. Lowest fret resolves ties.';$('primary').textContent=pct(network.accuracy_frame_weighted)+' / '+pct(prior.accuracy_frame_weighted);$('macro').textContent=pct(network.accuracy_equal_supported_position)+' / '+pct(prior.accuracy_equal_supported_position);$('modal').textContent=pct(network.fraction_on_training_modal);
 $('scores').tBodies[0].innerHTML=Object.entries(model.predictors).map(([p,r])=>`<tr class="${p===predictor?'selected':''}"><td>${p}</td><td>${pct(r.accuracy_frame_weighted)}</td><td>${pct(r.accuracy_equal_supported_position)}</td><td>${pct(r.accuracy_equal_supported_string)}</td><td>${p==='model_best_active'?'saved choices':p.endsWith('_distribution')?'analytic expectation':'deterministic prior'}</td></tr>`).join('');
 const s=detail.strings[string];$('concentration').textContent='String '+string+': '+s.reference_active_frames+' active GT frames; training-modal fret '+s.training_modal_fret+'; top predicted fret '+s.top_predicted_fret+' ('+pct(s.top_prediction_share)+'). Accuracy '+pct(s.accuracy)+', within-string independence expectation '+pct(s.independent_marginal_expected_accuracy)+'.';
 $('heatmap').tHead.innerHTML='<tr><th>GT / pred</th>'+Array.from({length:25},(_,f)=>'<th>'+f+'</th>').join('')+'</tr>';$('heatmap').tBodies[0].innerHTML=rows.map(r=>'<tr><th>'+r.fret+'</th>'+matrix[r.fret].map((value,p)=>{const fraction=r.reference_active_frames?value/r.reference_active_frames:0;return `<td data-gt="${r.fret}" data-pred="${p}" data-value="${value}" title="GT${r.fret} to ${p}: ${value} / ${r.reference_active_frames}" style="background:${p===r.fret?'rgba(54,171,145,':'rgba(216,139,75,'}${value?(.12+.75*Math.sqrt(fraction)):0})">${value?(normalize?pct(fraction):mass(value)):''}</td>`;}).join('')+'</tr>').join('');
 const training=data.training[model.history].unique.counts[string];$('classes').tBodies[0].innerHTML=rows.map(r=>`<tr><td>${r.fret}</td><td>${r.reference_active_frames}</td><td>${mass(r.correct_mass)}</td><td>${pct(r.accuracy)}</td><td>${mass(r.predicted_mass)}</td><td>${training[r.fret]}</td></tr>`).join('');
 const comparison=model.primary_comparison;$('paired').textContent=comparison.model_wins+' wins / '+comparison.ties+' ties / '+comparison.model_losses+' losses. Mean clip difference '+(100*comparison.mean_paired_clip_difference).toFixed(2)+' pp; pooled frame difference '+(100*comparison.model_minus_prior_frame_accuracy).toFixed(2)+' pp.';$('clips').tBodies[0].innerHTML=comparison.paired_clips.map(r=>`<tr><td><a href="../fret19/validation/results/${encodeURIComponent(mode)}/${encodeURIComponent(r.id)}-metrics.json">${r.id}</a></td><td>${r.difference===null?'—':(100*r.difference).toFixed(2)}</td></tr>`).join('');}
for(const id of ['model','predictor','string','normalize','all-rows'])$(id).addEventListener('change',render);$('download-confusions').addEventListener('click',()=>{const bytes=Uint8Array.from(atob(exactBytes),c=>c.charCodeAt(0)),url=URL.createObjectURL(new Blob([bytes],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='confusions.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});render();</script></html>'''


def report(output):
    output = Path(output)
    training, summary, matrices = (read(output / name) for name in ('priors.json', 'summary.json', 'confusions.json'))
    text = findings(summary)
    for name in ('report.md', 'findings.md'):
        (output / name).write_text(text, encoding='utf-8')
    names = ['summary.json', 'priors.json', 'cases.json', 'confusions.json', 'priors.csv', 'classes.csv',
             'strings.csv', 'cases.csv', 'confusions.csv', 'protocol.json', 'input-bindings.json']
    links = ''.join(f'<a href="{quote(name, safe="/")}">{name}</a>' for name in names)
    payload = json.dumps({'training': training, 'summary': summary, 'confusions': matrices}, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    encoded = base64.b64encode((output / 'confusions.json').read_bytes()).decode('ascii')
    options = ''.join(f'<option value="{name}">{name}</option>' for name in PREDICTORS)
    html = HTML.replace('__DATA__', payload).replace('__BYTES__', encoded).replace('__OPTIONS__', options).replace('__LINKS__', links)
    (output / 'review.html').write_text(html, encoding='utf-8')
    parent = read(ROOT / 'outputs/fret19/validation/run.json')
    save(output / 'next-experiment.json', {
        'stage': 22, 'status': 'proposed_not_trained', 'training_executed': False,
        'kind': 'matched_active_fret_class_weighting',
        'question': 'Does bounded training-only class weighting improve supported-fret discrimination without sacrificing original state/note scores?',
        'parent_checkpoint': parent['checkpoint_paths']['control500'],
        'parent_checkpoint_sha256': parent['checkpoint_sha256']['control500'],
        'source_root': 'outputs/prior21', 'total_new_optimizer_update_cap': 450,
        'new_optimizer_updates_per_arm': 150, 'new_sampled_crop_epochs_per_arm': 3,
        'data_epoch_start': 10, 'samples_per_epoch': 200, 'batch_size': 4, 'crop_seconds': 2.,
        'arms': [{'name': 'original', 'active_fret_ce_coefficient': 0., 'weighting': 'none'},
                 {'name': 'uniform_ce', 'active_fret_ce_coefficient': .5, 'weighting': 'uniform'},
                 {'name': 'balanced_ce', 'active_fret_ce_coefficient': .5, 'weighting': 'clipped_inverse_sqrt'}],
        'weight_source': 'Stage20 epochs1_10 unique supervised source-frame/string counts, training only.',
        'weight_rule': 'For seen frets within each string: min(4,sqrt(max_seen_count_in_string/class_count)); unseen classes weight0 and any active unseen target fails. Weighted CE=sum(weight*CE)/sum(weight) on valid active targets. Uniform CE uses unit weights. No inactive-logit auxiliary gradient.',
        'restoration': 'Same complete control500 model, AdamW options/moments/counters, AMP scaler and CPU/CUDA RNG per arm. Archived ended scheduler is not restarted. Shared epochs11-13 crop/gain order and fixed final step650.',
        'learning_rate': .0001, 'common_target_delay_frames': 13, 'common_leading_mask_frames': 26,
        'crop_edge_mask_frames': 26, 'new_sanity_optimizer_updates': 0,
        'decoder_parameters': {'onset_threshold': .7, 'activity_threshold': .25, 'onset_backdate_seconds': 0.},
        'selection_rule': 'Fixed step650 mean development argmax exact-state F1, then official note-onset F1; exact ties prefer original, then uniform_ce, then balanced_ce. Report conditional/macro fret diagnostics and all tradeoffs. No independent-generalization claim.',
        'test_inference_allowed': False, 'production_promotion': False, 'calibration_search_allowed': False,
        'limitations': summary['limitations']})
    print('Stage21 review saved:', output / 'review.html', flush=True)
