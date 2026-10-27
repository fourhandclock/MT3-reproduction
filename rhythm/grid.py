"""Map seconds to score quarters. Detected pulse grouping is a meter proposal."""
import copy
import math

import numpy as np


def interpolate(value, x, y):
    if value < x[0]:
        return float(y[0] + (value-x[0]) * (y[1]-y[0]) / (x[1]-x[0]))
    if value > x[-1]:
        return float(y[-1] + (value-x[-1]) * (y[-1]-y[-2]) / (x[-1]-x[-2]))
    return float(np.interp(value, x, y))


def to_quarters(time, grid):
    return interpolate(time, grid['beats'], grid['quarters'])


def to_seconds(quarter, grid):
    return interpolate(quarter, grid['quarters'], grid['beats'])


def make_grid(beats, downbeats, duration, override=None):
    beats = np.asarray(sorted(set(float(t) for t in beats)), dtype=float)
    if len(beats) < 2 or not np.isfinite(beats).all() or np.any(np.diff(beats) <= 0):
        raise ValueError('At least two finite, distinct beats required. Supply corrected beats JSON.')
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Duration must be positive.')
    if not all(math.isfinite(float(t)) for t in downbeats):
        raise ValueError('Downbeat times must be finite.')
    indices = sorted(set(int(np.argmin(abs(beats-float(t)))) for t in downbeats))
    warnings = ['Pulse count does not resolve denominator, compound meter or half/double tempo.']
    counts = np.diff(indices).tolist()
    valid = [c for c in counts if 2 <= c <= 12]
    default_count = max(set(valid), key=lambda c: (valid.count(c), -c)) if valid else 4
    if not valid:
        warnings.append('Insufficient downbeat groups: provisional 4/4; correct the grid before relying on bars.')
    anchor = indices[0] if indices else 0
    pulse_quarters = 1.0
    if override:
        numerator = int(override['numerator']); denominator = int(override['denominator'])
        pulse_quarters = float(override.get('pulse_quarters', 4/denominator))
        if numerator <= 0 or denominator not in [1, 2, 4, 8, 16] or pulse_quarters <= 0:
            raise ValueError('Invalid meter override.')
        count = numerator*4/denominator/pulse_quarters
        if abs(count-round(count)) > 1e-6:
            raise ValueError('Meter length must contain an integer number of tracker pulses.')
        default_count = int(round(count))
    quarters = (np.arange(len(beats))-anchor)*pulse_quarters
    grid = {'beats': beats.tolist(), 'downbeats': [float(beats[i]) for i in indices],
            'quarters': quarters.tolist(), 'pulse_quarters': pulse_quarters,
            'duration_seconds': float(duration), 'warnings': warnings,
            'meter_source': 'manual' if override else 'provisional pulse grouping',
            'observed_pulses_per_bar': counts, 'bars': []}
    # A partial first bar is retained as a pickup, including extrapolated time before the first beat.
    start = math.floor(to_quarters(0, grid)*12+1e-8)/12
    finish = to_quarters(duration, grid)
    if start < 0:
        grid['bars'].append({'start': start, 'end': 0.0, 'numerator': override['numerator'] if override else default_count,
                             'denominator': override['denominator'] if override else 4, 'pickup': True})
    position = 0.0
    boundaries = {round(float(quarters[i]), 8): c for i, c in zip(indices, counts)}
    while position < finish-1e-8:
        count = default_count if override else boundaries.get(round(position, 8), default_count)
        if count < 2 or count > 12:
            count = default_count
            grid['warnings'].append('Implausible downbeat interval replaced by default pulse grouping.')
        bar = {'start': position, 'end': position+count*pulse_quarters,
               'numerator': int(override['numerator']) if override else int(count),
               'denominator': int(override['denominator']) if override else 4, 'pickup': False}
        grid['bars'].append(bar); position = bar['end']
    for number, bar in enumerate(grid['bars'], 1):
        bar['number'] = number
        bar['start_seconds'] = to_seconds(bar['start'], grid)
        bar['end_seconds'] = to_seconds(bar['end'], grid)
    return grid


def quantize(notes, grid, config, subdivision=None):
    """Retain every source note; score changes are derived and explicitly flagged."""
    grid = copy.deepcopy(grid); divisions = config['divisions']
    if subdivision is not None and subdivision not in config['subdivisions']:
        raise ValueError('Unsupported subdivision override.')
    for bar in grid['bars']:
        starts = [to_quarters(n['start'], grid)-bar['start'] for n in notes
                  if bar['start'] <= to_quarters(n['start'], grid) < bar['end']]
        costs = {s: float(np.mean([(q-round(q*s)/s)**2 for q in starts])) if starts else 0.0
                 for s in config['subdivisions']}
        for s in costs:
            if s == 3: costs[s] += config['triplet_penalty']
        bar['subdivision'] = subdivision or min(costs, key=costs.get)
        bar['grid_costs'] = costs
    def snap(q):
        bar = next((b for b in grid['bars'] if b['start'] <= q < b['end']), grid['bars'][-1])
        s = bar['subdivision']
        return round((bar['start']+round((q-bar['start'])*s)/s)*divisions)
    output = []; first_tick = round(grid['bars'][0]['start']*divisions)
    for index, note in enumerate(notes):
        raw = copy.deepcopy(note)
        if not all(math.isfinite(note[k]) for k in ['start', 'end', 'pitch']) or not (0 <= note['start'] < note['end']) or not 0 <= int(note['pitch']) <= 127:
            raise ValueError('Notes require valid MIDI pitch and positive, nonnegative intervals.')
        a = snap(to_quarters(note['start'], grid))
        b = snap(to_quarters(min(note['end'], grid['duration_seconds']), grid))
        flags = []
        if note['end'] > grid['duration_seconds']: flags.append('release clipped in score at input boundary')
        a = max(first_tick, a)
        if b <= a:
            # A small positive written duration prevents deleted short/repeated attacks.
            bar = next((v for v in grid['bars'] if v['start'] <= a/divisions < v['end']), grid['bars'][-1])
            b = a + divisions//bar['subdivision']; flags.append('minimum written duration')
        start_shift = to_seconds(a/divisions, grid)-note['start']
        end_shift = to_seconds(b/divisions, grid)-min(note['end'], grid['duration_seconds'])
        if max(abs(start_shift), abs(end_shift)) > config['large_shift_seconds']:
            flags.append('large quantization shift')
        output.append({'id': index, 'raw': raw, 'start_tick': a, 'end_tick': b,
                       'onset_shift_seconds': start_shift, 'offset_shift_seconds': end_shift, 'flags': flags})
    # Quantization can create overlaps on a string even when raw intervals were feasible.
    for i, note in enumerate(output):
        string = note['raw'].get('string')
        if string is None: continue
        for other in output[:i]:
            if other['raw'].get('string') == string and max(note['start_tick'], other['start_tick']) < min(note['end_tick'], other['end_tick']):
                for n in [note, other]:
                    if 'quantized string overlap' not in n['flags']: n['flags'].append('quantized string overlap')
    # Retain any minimum-duration note extending beyond the original last bar.
    while output and max(n['end_tick'] for n in output) > round(grid['bars'][-1]['end']*divisions):
        previous = grid['bars'][-1]; bar = dict(previous)
        bar.update(number=previous['number']+1, start=previous['end'], end=previous['end']+previous['end']-previous['start'], pickup=False)
        bar.update(start_seconds=to_seconds(bar['start'], grid), end_seconds=to_seconds(bar['end'], grid))
        grid['bars'].append(bar)
    return {'grid': grid, 'divisions': divisions, 'notes': output,
            'source_timing_preserved': True, 'duration_policy': 'Snapped audible release; written articulation is not inferred.'}
