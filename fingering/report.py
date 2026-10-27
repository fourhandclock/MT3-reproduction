import csv
import html
import json
from urllib.parse import quote


def tab_svg(notes,duration,title):
    width,height,left=1100,220,75
    labels=['low E','A','D','G','B','high E']
    parts=[f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)} timed tablature">',
           f'<rect width="{width}" height="{height}" fill="#101c2d"/>']
    for string in range(6):
        y=30+(5-string)*27
        parts.append(f'<path d="M{left},{y} H{width-10}" stroke="#61728b"/>')
        parts.append(f'<text x="5" y="{y+5}" fill="#c9d7eb" font-size="13">{labels[string]}</text>')
    for second in range(int(duration)+1):
        x=left+second/duration*(width-left-10)
        parts.append(f'<text x="{x:.1f}" y="208" fill="#9baec7" font-size="11">{second}s</text>')
    for note in notes:
        if note.get('string') is None or note['start']>=duration: continue
        x=left+max(0,note['start'])/duration*(width-left-10);y=30+(5-note['string'])*27
        end=left+min(duration,note['end'])/duration*(width-left-10)
        parts.append(f'<path d="M{x:.1f},{y} H{end:.1f}" stroke="#5fd4ad" stroke-width="3"/>')
        label=f'MIDI {note["pitch"]}, string {note["string"]}, fret {note["fret"]}, {note["start"]:.3f}-{note["end"]:.3f}s'
        parts.append(f'<g><title>{html.escape(label)}</title><rect x="{x-8:.1f}" y="{y-10}" width="18" height="20" rx="3" fill="#243950"/>'
                     f'<text x="{x+1:.1f}" y="{y+5}" text-anchor="middle" fill="#f1f6fc" font-size="12">{note["fret"]}</text></g>')
    unresolved=[note for note in notes if note.get('string') is None and note['start']<duration]
    parts.append(f'<text x="{left}" y="187" fill="#ffad88" font-size="12">Unresolved positions in view: {len(unresolved)}; all pitches/times remain in JSON and MIDI.</text>')
    return ''.join(parts)+'</svg>'


def write_report(output,records,summary,run):
    directory=output/'evaluation';config=run['config'];selection=run['selection']
    lines=['# Stage 5: guitar fingering decoder','',
           'A bounded beam search maps note events to guitar strings/frets. Reference-note inputs isolate '
           'fingering decisions; CNN-GRU note inputs expose the interaction with transcription errors. '
           'No additional training was performed.','',
           f'Selected acoustic weight: {selection["acoustic_weight"]}, using progression-2 validation only. '
           f'Fret-span limit: {config["max_fret_span"]}; finger-group proxy limit: {config["max_finger_groups"]}.',
           'All input pitches, onsets, offsets and note identities are retained. Infeasible or pruned-away '
           'assignments are explicitly unresolved, not deleted or octave-shifted.','']
    tables={}
    for scenario,group in summary.items():
        lines.extend([f'## {scenario.capitalize()} note inputs','',
                      '| Mode | Pitch onset F1 | String-aware onset F1 | Assigned | Unresolved | String collision events | Span violations | Finger-proxy violations | Hand travel |',
                      '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
        rows=[]
        for mode,value in group.items():
            fields=[mode,f'{value["mean_pitch_onset_f1"]:.3f}',f'{value["mean_string_onset_f1"]:.3f}',
                    f'{100*value["assignment_fraction"]:.1f}%',str(value['unassigned_notes']),
                    str(value['same_string_overlap_events']),str(value['span_violation_events']),
                    str(value['finger_proxy_violation_events']),str(value['hand_position_travel_frets'])]
            lines.append('| '+' | '.join(fields)+' |')
            rows.append('<tr>'+''.join('<td>'+html.escape(field)+'</td>' for field in fields)+'</tr>')
        tables[scenario]=''.join(rows)
        lines.append('')
    lines+=['## How to read these results','',
            'Oracle inputs use reference pitch, onset and offset events; ground-truth string positions are removed '
            'before search and used only to score recovery. Reference_positions shows the originally performed '
            'fingerings. Lowest_fret independently chooses the smallest fret for each pitch and can conflict '
            'with other active notes. Constrained_pitch_only uses physical proxies and movement costs; '
            'constrained_acoustic additionally uses the frozen CNN-GRU fret-state probabilities.',
            'String-aware onset F1 measures agreement with the recorded performance. A lower value does not '
            'automatically mean an alternative fingering is worse for a player. Conversely, passing our proxies '
            'does not prove full human playability. The original performance can violate a conservative proxy '
            'without being a labeling error.',
            'Pitch onset F1 must be identical across all modes in each scenario: this decoder cannot repair '
            'wrong pitches, missed notes or extra notes. Unresolved notes remain in the pitch evaluation and '
            'count as unmatched string assignments. Assignment coverage is reported so removing difficult '
            'positions cannot masquerade as a quality gain.',
            'Violations and hand-travel totals cover the full 12-second input; note F1 is the mean over twelve '
            'recordings scored in the first 10 seconds. Collision counts are event-based, not unique note-pair counts. '
            'Hand travel is a fret-position proxy rather than measured hand motion.','',
            '## Search and constraints','',
            'Valid candidate positions satisfy sounding MIDI pitch = tuned open-string pitch + capo + relative fret. '
            'Sustained assigned notes keep their string until release. New notes cannot use an occupied string. '
            'The search limits the span among fretted notes and a barre-aware finger-group lower bound.',
            'A barre may pass beneath higher-fretted notes or inactive strings, but cannot obstruct an active '
            'open/lower-fretted intermediate string. This proxy omits detailed finger geometry, thumb fretting, '
            'picking/muting, stretches at different neck positions and advanced techniques.',
            'Soft costs penalize fret height, span, finger groups, hand movement and fast shifts, with a small '
            'open-string preference. Learned acoustic evidence is averaged over candidate string/fret states '
            'during each note. The priority is assignment coverage before soft cost.',
            f'Beam widths are {config["beam_width"]} at completed onset groups and {config["partial_beam_width"]} '
            'during expansion. Beam pruning can miss a globally feasible or optimal path; unresolved positions '
            'are flagged for review. Search uses sequence context and is offline.','',
            '## Validation and provenance','',
            '| Acoustic weight | Validation string-aware onset F1 | Assigned fraction | Mean hand travel |',
            '| --- | ---: | ---: | ---: |']
    for row in selection['validation_grid']:
        lines.append(f'| {row["acoustic_weight"]} | {row["mean_string_onset_f1"]:.3f} | '
                     f'{row["mean_assignment_fraction"]:.3f} | {row["mean_hand_travel"]:.2f} |')
    lines+=['','Weights were selected on twelve progression-2 reference-note performances. Final results use '
            'the unchanged progression-3 recordings. Decoder constraints and preference weights were declared '
            'before these results. The acoustic model remains the original clean-trained CNN-GRU; no augmentation '
            'or retraining is added here.',
            'JSON retains exact intervals. MIDI is quantized to ticks, and extra melodic lanes preserve overlapping '
            'duplicate-pitch voices. Unresolved notes use an unassigned MIDI lane. CSV exports leave their position '
            'blank with an explicit status. More than 15 required MIDI lanes is flagged instead of silently merging voices.',
            '[Inspect timed tabs and audio](evaluation/review.html). Human review is required before calling a '
            'result a preferred playable transcription.',
            'Graph/constraint and learned symbolic fingering methods are established research directions. '
            '[Related work and alternative-fingering evaluation](https://arxiv.org/abs/2506.14223).','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    selected=['00_Rock3-148-C_comp','00_Funk3-98-A_solo','02_BN3-119-G_comp']
    sections=[]
    for clip in selected:
        body=f'<section><h2>{html.escape(clip)}</h2><audio controls preload="none" src="clips/{quote(clip,safe="")}.wav"></audio>'
        for scenario in ['oracle','predicted']:
            body+=f'<h3>{scenario.capitalize()} inputs</h3>'
            for mode in [ 'reference_positions' if scenario=='oracle' else 'raw_cnn','constrained_pitch_only','constrained_acoustic']:
                row=next(value for value in records if value['id']==clip and value['scenario']==scenario and value['mode']==mode)
                position=row['playability'];encoded=quote(clip,safe='')
                body+=f'<h4>{mode} · string-aware F1 {row["tab_metrics"]["onset"]["f1"]:.3f} · unresolved {position["unassigned_count"]}</h4>'
                body+=tab_svg(row['notes'],config['score_seconds'],mode)
                body+=f'<p><a href="{scenario}/{mode}/{encoded}.mid">MIDI</a> · <a href="{scenario}/{mode}/{encoded}.csv">Timed tab CSV</a> · '
                body+=f'<a href="{scenario}/{mode}/{encoded}.json">Note positions and status</a></p>'
        sections.append(body+'</section>')
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Guitar fingering review</title>'
    page+='<style>body{background:#0b1421;color:#e8eef8;font:16px system-ui;max-width:1150px;margin:40px auto;padding:0 20px;line-height:1.5}section{margin:32px 0;padding:20px;background:#142135;border-radius:12px}svg{width:100%;height:auto}a{color:#94bdff}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:8px;border-bottom:1px solid #33435a}audio{width:100%;max-width:600px}h2{font-size:21px}h3{font-size:18px}h4{font-size:15px}.table{overflow-x:auto}</style>'
    page+='<h1>Pitch-preserving guitar fingering</h1><p>Positions are constrained by simple feasibility proxies. '
    page+='Alternative tabs can differ from the original performance. Unresolved notes remain explicitly visible; no model training.</p>'
    for scenario,rows in tables.items():
        page+=f'<h2>{scenario.capitalize()} note inputs</h2><div class="table"><table><tr>'
        page+=''.join('<th>'+value+'</th>' for value in ['Mode','Pitch F1','String F1','Assigned','Unresolved','Collisions','Span','Finger proxy','Travel'])
        page+='</tr>'+rows+'</table></div>'
    page+='<p>Tabs show the first 10 seconds. Top line is high E, bottom line is low E. Numbers are frets; green segments show note duration. '
    page+='Hover for pitch and timing. Dense passages are easier to inspect in the CSV/JSON exports.</p>'+''.join(sections)
    page+='<p><a href="../report.md">Full protocol, constraints and limitations</a></p></html>'
    (directory/'review.html').write_text(page,encoding='utf-8')
    with (directory/'metrics.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['id','scenario','mode','pitch_onset_f1','string_onset_f1','unassigned',
                                                  'collision_events','span_events','finger_proxy_events','hand_travel','solver_seconds'])
        for row in records:
            p=row['playability'];writer.writerow([row['id'],row['scenario'],row['mode'],row['pitch_metrics']['onset']['f1'],
                                                  row['tab_metrics']['onset']['f1'],p['unassigned_count'],p['same_string_overlap_events'],
                                                  p['span_violation_events'],p['finger_proxy_violation_events'],p['hand_position_travel_frets'],row['solver_seconds']])
    print('Saved fingering report:',output/'report.md',flush=True)
