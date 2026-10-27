from collections import Counter

from .decoder import posture


def preserved(before,after):
    return len(before)==len(after) and all((a['pitch'],a['start'],a['end'])==(b['pitch'],b['start'],b['end'])
                                         for a,b in zip(before,after))


def playability(notes,config):
    assigned=[note for note in notes if note.get('string') is not None]
    times=sorted({note['start'] for note in assigned})
    collisions=span_violations=finger_violations=0
    max_span=max_fingers=movement=0
    prior_position=0
    for time in times:
        active=[note for note in assigned if note['start']<=time+1e-9 and note['end']>time+config['release_tolerance_seconds']]
        counts=Counter(note['string'] for note in active)
        collisions+=sum(value-1 for value in counts.values() if value>1)
        positions={note['string']:note['fret'] for note in active}
        _,span,fingers=posture(positions,config)
        span_violations+=span>config['max_fret_span'];finger_violations+=fingers>config['max_finger_groups']
        max_span=max(max_span,span);max_fingers=max(max_fingers,fingers)
        fretted=[value for value in positions.values() if value>0]
        if fretted:
            lower=max(1,max(fretted)-config['max_fret_span']);upper=min(fretted)
            position=min(fretted) if prior_position==0 else max(lower,min(upper,prior_position))
            if prior_position: movement+=abs(position-prior_position)
            prior_position=position
    return {'note_count':len(notes),'assigned_count':len(assigned),'unassigned_count':len(notes)-len(assigned),
            'assignment_fraction':len(assigned)/len(notes) if notes else 1.0,
            'same_string_overlap_events':int(collisions),'span_violation_events':int(span_violations),
            'finger_proxy_violation_events':int(finger_violations),'maximum_fret_span':int(max_span),
            'maximum_finger_groups':int(max_fingers),'hand_position_travel_frets':int(movement)}
