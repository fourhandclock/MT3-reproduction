"""Keep argmax evaluation separate from decoder-policy occupancy and fret choice."""

from collections import Counter
import numpy as np
from diagnosis15.analysis import activity_states,pitch_union,source_arrays,frame_diagnosis
from diagnosis9.head_audit import binary_metric,exact_state_metric,metric,pooled


def conditional(numerator,denominator):
    return {'correct':int(numerator),'count':int(denominator),
            'accuracy':float(numerator/denominator) if denominator else None}


def state_diagnosis(probabilities,targets,config):
    arrays=source_arrays(probabilities,targets,config,config['decoder_parameters']['activity_threshold'])
    truth=arrays['target'];valid=arrays['mask'][:,None]&(truth!=-100);active=valid&(truth!=25)
    state=probabilities['state'][config['source_delay_frames']:config['source_delay_frames']+config['source_frames']]
    result={'frame_diagnosis':frame_diagnosis(arrays),'policies':{}}
    for policy,observed in [('argmax',state.argmax(-1)),('decoder_activity',arrays['observed'])]:
        pred_active=observed!=25;correct=active&pred_active&(observed==truth)
        wrong=active&pred_active&(observed!=truth);octave=wrong&(np.abs(observed-truth)==12)
        counts={'correct_fret':int(correct.sum()),'predicted_inactive':int((active&~pred_active).sum()),
            'wrong_octave_same_string':int(octave.sum()),'other_wrong_fret':int((wrong&~octave).sum()),
            'reference_active_frames':int(active.sum()),'reference_inactive_frames':int((valid&~active).sum()),
            'active_on_reference_inactive':int((valid&~active&pred_active).sum()),
            'valid_string_frames':int(valid.sum())}
        confusion=np.bincount((truth[valid]*26+observed[valid]).astype(np.int64),minlength=26*26).reshape(26,26)
        result['policies'][policy]={
            'exact_state':exact_state_metric(np.where(valid,truth,-100),observed,25),
            'activity':binary_metric((truth[valid]!=25),pred_active[valid]),'counts':counts,
            'fret_given_reference_and_prediction_active':conditional(correct.sum(),(active&pred_active).sum()),
            'confusion':confusion.tolist(),
            'wrong_fret_delta_histogram':{str(k):v for k,v in sorted(Counter(int(x) for x in (observed-truth)[wrong]).items())}}
    result['best_active_fret_on_reference_active']=conditional((active&(truth==arrays['best_fret'])).sum(),active.sum())
    result['auxiliary_pitch']=binary_metric(arrays['target_pitch'][arrays['mask']],arrays['auxiliary_pitch'][arrays['mask']])
    result['state_derived_pitch']=binary_metric(arrays['target_pitch'][arrays['mask']],arrays['state_pitch'][arrays['mask']])
    return result,arrays


def summarize_frames(records):
    result={}
    for model in ['short_short','short_long']:
        rows=[r['diagnosis'] for r in records if r['model']==model];value={'policies':{}}
        for policy in ['argmax','decoder_activity']:
            values=[r['policies'][policy] for r in rows]
            counts={k:sum(r['counts'][k] for r in values) for k in values[0]['counts']}
            value['policies'][policy]={'counts':counts,
                'exact_state_mean_clip_f1':float(np.mean([r['exact_state']['f1'] for r in values])),
                'exact_state_pooled':pooled([r['exact_state'] for r in values]),
                'activity_mean_clip_f1':float(np.mean([r['activity']['f1'] for r in values])),
                'activity_pooled':pooled([r['activity'] for r in values]),
                'fret_given_reference_and_prediction_active':conditional(
                    sum(r['fret_given_reference_and_prediction_active']['correct'] for r in values),
                    sum(r['fret_given_reference_and_prediction_active']['count'] for r in values)),
                'confusion':np.sum([r['confusion'] for r in values],axis=0).tolist()}
        for field in ['auxiliary_pitch','state_derived_pitch']:
            value[field]={'mean_clip_f1':float(np.mean([r[field]['f1'] for r in rows])),
                          'pooled':pooled([r[field] for r in rows])}
        value['best_active_fret_on_reference_active']=conditional(
            sum(r['best_active_fret_on_reference_active']['correct'] for r in rows),
            sum(r['best_active_fret_on_reference_active']['count'] for r in rows))
        result[model]=value
    return result


def contrasts(modes):
    scores={key:modes[key]['onset_f1'] for key in
            ['control','fused','control_state_fused_onset','fused_state_control_onset']}
    a,b,c,d=(scores[k] for k in ['control','fused','control_state_fused_onset','fused_state_control_onset'])
    return {'total_fused_minus_control':b-a,'onset_swap_at_control_state':c-a,
        'onset_swap_at_fused_state':b-d,'state_swap_at_control_onset':d-a,
        'state_swap_at_fused_onset':b-c,'interaction':b-d-c+a,
        'scope':'Differences of macro note-onset F1 under the fixed decoder. Cross-head calibration mismatch prevents pure causal attribution. No model selection.'}


def note_changes(control,fused):
    left=control['taxonomy']['reference_rows'];right=fused['taxonomy']['reference_rows']
    if [r['note'] for r in left]!=[r['note'] for r in right]:raise ValueError('Reference note identities differ')
    result=[]
    for a,b in zip(left,right):
        ca=a['category'].startswith('matched_');cb=b['category'].startswith('matched_')
        category=('matched_both' if ca and cb else 'fused_only' if cb else 'control_only' if ca else 'missed_both')
        result.append({'reference_index':a['reference_index'],'note':a['note'],'change':category,
                       'control_category':a['category'],'fused_category':b['category']})
    return result
