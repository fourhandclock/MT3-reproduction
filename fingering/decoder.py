import math
from dataclasses import dataclass

import numpy as np


def candidates(pitch,config):
    effective=[value+config['capo'] for value in config['tuning_midi']]
    maximum=config['max_physical_fret']-config['capo']
    return [(string,pitch-open_pitch) for string,open_pitch in enumerate(effective)
            if 0<=pitch-open_pitch<=maximum]


def finger_groups(positions):
    """Barre-aware lower-bound proxy, not a full human-hand model.

    A barre can pass below higher fretted notes or inactive strings, but cannot
    pass through an active open/lower-fretted note.
    """
    count=0
    for fret in sorted({value for value in positions.values() if value>0}):
        strings=sorted(string for string,value in positions.items() if value==fret)
        previous=None
        for string in strings:
            if previous is None or any(positions.get(intermediate,fret)>=0 and
                                      positions.get(intermediate,fret)<fret
                                      for intermediate in range(previous+1,string)):
                count+=1
            previous=string
    return count


def posture(positions,config):
    fretted=[fret for fret in positions.values() if fret>0]
    span=max(fretted)-min(fretted) if fretted else 0
    fingers=finger_groups(positions)
    valid=span<=config['max_fret_span'] and fingers<=config['max_finger_groups']
    return valid,span,fingers


def lowest_fret(notes,config):
    output=[]
    for index,note in enumerate(notes):
        available=candidates(int(note['pitch']),config)
        choice=min(available,key=lambda pair:(pair[1],-pair[0])) if available else None
        output.append(dict(note,note_id=index,string=choice[0] if choice else None,
                           fret=choice[1] if choice else None,
                           assignment_status='assigned' if choice else 'outside supported fretboard'))
    return output


@dataclass
class State:
    active: tuple
    position: int
    cost: float
    missing: int
    path: tuple


def prune(states,limit):
    best={}
    for state in states:
        key=(state.active,state.position)
        if key not in best or (state.missing,state.cost)<(best[key].missing,best[key].cost): best[key]=state
    return sorted(best.values(),key=lambda state:(state.missing,state.cost))[:limit]


def acoustic_costs(notes,probability,config):
    costs={}
    if probability is None: return costs
    if config['capo']!=0 or config['tuning_midi']!=[40,45,50,55,59,64]:
        raise ValueError('Learned fret probabilities apply only to the model\'s standard tuning and capo 0.')
    state=probability['state'];fps=config['acoustic_frame_rate']
    for index,note in enumerate(notes):
        start=max(0,min(len(state)-1,int(np.ceil(note['start']*fps))))
        end=max(start+1,min(len(state),int(np.ceil(note['end']*fps))))
        # Penalize low acoustic support over the note, with a bounded log floor.
        for string,fret in candidates(int(note['pitch']),config):
            costs[index,string,fret]=float(-np.log(np.clip(state[start:end,string,fret],1e-6,1)).mean())
    return costs


def solve(notes,config,probability=None,acoustic_weight=0.0):
    if len(config['tuning_midi'])!=6: raise ValueError('Six-string guitar tuning required.')
    if not 0<=config['capo']<=config['max_physical_fret']: raise ValueError('Invalid capo.')
    if config['beam_width']<1 or config['partial_beam_width']<1: raise ValueError('Positive beam widths required.')
    for note in notes:
        if not math.isfinite(note['start']) or not math.isfinite(note['end']) or note['end']<=note['start']:
            raise ValueError('Positive, finite note intervals required.')
        if int(note['pitch'])!=note['pitch']: raise ValueError('Integer MIDI pitches required.')
    available={index:candidates(int(note['pitch']),config) for index,note in enumerate(notes)}
    acoustic=acoustic_costs(notes,probability,config) if acoustic_weight else {}
    order=sorted(range(len(notes)),key=lambda index:(notes[index]['start'],len(available[index]),notes[index]['pitch'],index))
    groups=[]
    for index in order:
        if not groups or abs(notes[index]['start']-notes[groups[-1][0]]['start'])>1e-9: groups.append([])
        groups[-1].append(index)
    states=[State((-1,)*6,0,0.0,0,())]
    processed=[];previous_time=None;expanded=0
    for group in groups:
        time=notes[group[0]]['start']
        states=[State(tuple(index if index>=0 and notes[index]['end']>time+config['release_tolerance_seconds'] else -1
                            for index in state.active),state.position,state.cost,state.missing,state.path) for state in states]
        for index in group:
            proposals=[]
            for state in states:
                for string,fret in available[index]:
                    if state.active[string]>=0: continue
                    active=list(state.active);active[string]=index
                    positions={s:notes[n]['pitch']-config['tuning_midi'][s]-config['capo'] for s,n in enumerate(active) if n>=0}
                    if not posture(positions,config)[0]: continue
                    cost=config['fret_weight']*fret-(config['open_string_bonus'] if fret==0 else 0)
                    cost+=acoustic_weight*acoustic.get((index,string,fret),0.0)
                    proposals.append(State(tuple(active),state.position,state.cost+cost,state.missing,state.path+((string,fret),)))
                    expanded+=1
                # Keep the note but explicitly leave its tab position unresolved.
                # Assignment count has priority over all soft preference costs.
                proposals.append(State(state.active,state.position,state.cost,state.missing+1,state.path+(None,)))
            states=prune(proposals,config['partial_beam_width'])
            processed.append(index)
        finalized=[]
        for state in states:
            positions={s:notes[n]['pitch']-config['tuning_midi'][s]-config['capo'] for s,n in enumerate(state.active) if n>=0}
            _,span,fingers=posture(positions,config)
            fretted=[fret for fret in positions.values() if fret>0]
            anchor=state.position
            if fretted:
                lower=max(1,max(fretted)-config['max_fret_span']);upper=min(fretted)
                anchor=min(fretted) if state.position==0 else max(lower,min(upper,state.position))
            distance=abs(anchor-state.position) if state.position else 0
            elapsed=max(0.04,time-previous_time) if previous_time is not None else 1.0
            cost=state.cost+config['span_weight']*span+config['finger_weight']*fingers
            cost+=config['movement_weight']*distance+config['fast_shift_weight']*distance/elapsed
            finalized.append(State(state.active,anchor,cost,state.missing,state.path))
        states=prune(finalized,config['beam_width']);previous_time=time
    winner=states[0]
    placements=dict(zip(processed,winner.path))
    output=[]
    for index,note in enumerate(notes):
        choice=placements.get(index)
        reason='assigned' if choice else ('outside supported fretboard' if not available[index] else 'unresolved under search/feasibility constraints')
        output.append(dict(note,note_id=index,string=choice[0] if choice else None,fret=choice[1] if choice else None,
                           assignment_status=reason))
    return output,{'cost':winner.cost,'unassigned':winner.missing,'expanded_candidates':expanded,
                   'search':'bounded beam; globally optimal assignment not guaranteed'}
