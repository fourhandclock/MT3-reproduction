"""MusicXML 4.0, two staves (concert pitch and tab), with independent sustained voices."""
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from .chords import ROOTS
from .grid import to_quarters


def element(parent, name, value=None, **attributes):
    node = ET.SubElement(parent, name, {k: str(v) for k, v in attributes.items()})
    if value is not None: node.text = str(value)
    return node


def pitch_parts(midi):
    name = ROOTS[int(midi) % 12]
    return name[0], int('#' in name), int(midi)//12-1


def rhythm_atoms(ticks, subdivision):
    straight = [(72, 'whole', True, False), (48, 'whole', False, False),
                (36, 'half', True, False), (24, 'half', False, False),
                (18, 'quarter', True, False), (12, 'quarter', False, False),
                (9, 'eighth', True, False), (6, 'eighth', False, False), (3, '16th', False, False)]
    triplets = [(32, 'whole', False, True), (16, 'half', False, True),
                (8, 'quarter', False, True), (4, 'eighth', False, True),
                (2, '16th', False, True), (1, '32nd', False, True)]
    # A 3:2 eighth is four divisions when there are twelve divisions per quarter.
    choices = sorted(straight+triplets, key=lambda item: (-item[0], item[3]))
    if subdivision == 3:
        choices = sorted(triplets+[(48, 'whole', False, False), (24, 'half', False, False),
                                  (12, 'quarter', False, False)], key=lambda item: -item[0])
    result = []
    while ticks:
        value = next(item for item in choices if item[0] <= ticks)
        result.append(value); ticks -= value[0]
    return result


def allocate_voices(notes):
    groups = {}
    for note in sorted(notes, key=lambda n: (n['start_tick'], n['id'])):
        group = groups.setdefault(note['raw'].get('string'), [])
        lane = next((lane for lane in group if lane[-1]['end_tick'] <= note['start_tick']), None)
        if lane is None: lane = []; group.append(lane)
        lane.append(note)
    return [(string, lane) for string, lanes in sorted(groups.items(), key=lambda item: -1 if item[0] is None else item[0])
            for lane in lanes]


def write_xml(document, path, title='Guitar rhythm prototype'):
    if document['divisions'] != 12: raise ValueError('MusicXML renderer expects 12 divisions per quarter.')
    root = ET.Element('score-partwise', version='4.0')
    element(element(root, 'work'), 'work-title', title)
    identification = element(root, 'identification')
    encoding = element(identification, 'encoding'); element(encoding, 'software', 'MT3 guitar rhythm prototype')
    element(identification, 'miscellaneous')
    part_list = element(root, 'part-list'); score_part = element(part_list, 'score-part', id='P1')
    element(score_part, 'part-name', 'Guitar (concert pitch and tablature)')
    part = element(root, 'part', id='P1')
    lanes = allocate_voices(document['notes']) or [(None, [])]
    streams = [(i+1, 1, string, lane) for i, (string, lane) in enumerate(lanes)]
    streams += [(len(lanes)+i+1, 2, string, lane) for i, (string, lane) in enumerate(lanes) if string is not None]
    previous_meter = None
    previous_chord = None
    for bar in document['grid']['bars']:
        start = round(bar['start']*12); end = round(bar['end']*12); length = end-start
        measure = element(part, 'measure', number=bar['number'], implicit='yes' if bar['pickup'] else 'no')
        meter = (bar['numerator'], bar['denominator'])
        if meter != previous_meter or bar['number'] == 1:
            attributes = element(measure, 'attributes')
            element(attributes, 'divisions', 12)
            time = element(attributes, 'time'); element(time, 'beats', meter[0]); element(time, 'beat-type', meter[1])
            if bar['number'] == 1:
                element(attributes, 'staves', 2)
                for staff, sign, line in [(1, 'G', 2), (2, 'TAB', 5)]:
                    clef = element(attributes, 'clef', number=staff); element(clef, 'sign', sign); element(clef, 'line', line)
                details = element(attributes, 'staff-details', number=2)
                element(details, 'staff-lines', 6)
                for string, midi in enumerate(document['tuning_midi']):
                    step, alter, octave = pitch_parts(midi)
                    tuning = element(details, 'staff-tuning', line=string+1)
                    element(tuning, 'tuning-step', step)
                    if alter: element(tuning, 'tuning-alter', alter)
                    element(tuning, 'tuning-octave', octave)
            previous_meter = meter
        # Local tempo follows the detected variable beat grid, rather than one global BPM.
        beat_span = document['grid']['pulse_quarters']
        from .grid import to_seconds
        for q in [q for q in document['grid']['quarters'] if bar['start'] <= q < bar['end']]:
            seconds = to_seconds(q+beat_span, document['grid'])-to_seconds(q, document['grid'])
            direction = element(measure, 'direction', placement='above')
            kind = element(direction, 'direction-type'); metronome = element(kind, 'metronome')
            element(metronome, 'beat-unit', 'quarter'); element(metronome, 'per-minute', round(60*beat_span/seconds, 2))
            element(direction, 'offset', round((q-bar['start'])*12)); element(direction, 'staff', 1)
            element(direction, 'sound', tempo=round(60*beat_span/seconds, 2))
        for chord in document.get('chords', []):
            tick = round(to_quarters(chord['start'], document['grid'])*12)
            if start <= tick < end and chord['label'] != previous_chord:
                harmony = element(measure, 'harmony')
                if chord['label'] == 'N':
                    # MusicXML requires a root even for kind=none; hide the placeholder root.
                    chord_root = element(harmony, 'root')
                    element(chord_root, 'root-step', 'C', text='')
                    element(harmony, 'kind', 'none', text='N')
                else:
                    root_name, quality = chord['label'].split(':')
                    chord_root = element(harmony, 'root'); element(chord_root, 'root-step', root_name[0])
                    if '#' in root_name: element(chord_root, 'root-alter', 1)
                    element(harmony, 'kind', {'maj': 'major', 'min': 'minor', '7': 'dominant',
                                             'maj7': 'major-seventh', 'min7': 'minor-seventh'}[quality])
                element(harmony, 'offset', tick-start); element(harmony, 'staff', 1)
                previous_chord = chord['label']
        for stream_index, (voice, staff, string, lane) in enumerate(streams):
            if stream_index: element(element(measure, 'backup'), 'duration', length)
            cursor = start
            for source in lane:
                a, b = max(start, source['start_tick']), min(end, source['end_tick'])
                if b <= a: continue
                if a > cursor: append_duration(measure, a-cursor, voice, staff, None, None, None, bar['subdivision'])
                atoms = rhythm_atoms(b-a, bar['subdivision'])
                for index, atom in enumerate(atoms):
                    tie_stop = source['start_tick'] < a or index > 0
                    tie_start = source['end_tick'] > b or index < len(atoms)-1
                    append_note(measure, atom, voice, staff, source, tie_stop, tie_start)
                cursor = b
            if cursor < end: append_duration(measure, end-cursor, voice, staff, None, None, None, bar['subdivision'])
    ET.indent(root)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
    validate_xml(path)


def append_duration(parent, ticks, voice, staff, source, stop, start, subdivision):
    for atom in rhythm_atoms(ticks, subdivision): append_note(parent, atom, voice, staff, source, stop, start)


def append_note(parent, atom, voice, staff, source, tie_stop, tie_start):
    duration, kind, dotted, triplet = atom
    note = element(parent, 'note')
    if source is None: element(note, 'rest')
    else:
        pitch = element(note, 'pitch'); step, alter, octave = pitch_parts(source['raw']['pitch'])
        element(pitch, 'step', step)
        if alter: element(pitch, 'alter', alter)
        element(pitch, 'octave', octave)
    element(note, 'duration', duration)
    if source is not None:
        if tie_stop: element(note, 'tie', type='stop')
        if tie_start: element(note, 'tie', type='start')
    element(note, 'voice', voice); element(note, 'type', kind)
    if dotted: element(note, 'dot')
    if triplet:
        modification = element(note, 'time-modification')
        element(modification, 'actual-notes', 3); element(modification, 'normal-notes', 2)
    element(note, 'staff', staff)
    if source is not None:
        notations = element(note, 'notations')
        if tie_stop: element(notations, 'tied', type='stop')
        if tie_start: element(notations, 'tied', type='start')
        if source['raw'].get('string') is not None:
            technical = element(notations, 'technical')
            element(technical, 'fret', source['raw']['fret'])
            element(technical, 'string', 6-int(source['raw']['string']))
        elif staff == 1:
            element(notations, 'other-notation', 'Unresolved guitar position', type='single')


def validate_xml(path):
    """Structural bar/voice conservation, independent of the writer's lane assignment."""
    tree = ET.parse(path); divisions = 12
    for measure in tree.findall('./part/measure'):
        cursor = 0; voice_totals = {}
        for child in measure:
            if child.tag == 'backup': cursor -= int(child.findtext('duration'))
            if child.tag == 'note':
                duration = int(child.findtext('duration'))
                key = (child.findtext('voice'), child.findtext('staff'))
                voice_totals[key] = voice_totals.get(key, 0)+duration; cursor += duration
                if duration <= 0: raise ValueError('Non-positive notation duration.')
            if cursor < 0: raise ValueError('MusicXML backup went before bar start.')
        if len(set(voice_totals.values())) != 1: raise ValueError('MusicXML voices have different bar lengths.')
    return True


def export_score(document, stem, title):
    stem = Path(stem); stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix('.json').write_text(json.dumps(document, indent=2), encoding='utf-8')
    write_xml(document, stem.with_suffix('.musicxml'), title)
    with stem.with_suffix('.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream); writer.writerow(['id', 'raw_start_seconds', 'raw_end_seconds', 'pitch', 'string', 'fret',
                                                      'score_start_quarters', 'score_end_quarters', 'flags'])
        for note in document['notes']:
            raw = note['raw']
            writer.writerow([note['id'], raw['start'], raw['end'], raw['pitch'], raw.get('string'), raw.get('fret'),
                             note['start_tick']/12, note['end_tick']/12, '; '.join(note['flags'])])
