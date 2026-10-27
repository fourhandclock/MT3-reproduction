"""Append-only note events from causal, per-string frame probabilities.

Frame ``n`` is evidence available at audio time ``(n + 1) * hop / sr``.
Confirmation can backdate a source timestamp, but never a delivery timestamp.
The offset and pitch heads are accepted for the common model interface; releases
are determined by sustained inactivity, rather than future offset peaks.
Onset retriggers require one subsequent same-fret frame before publication, so a
trigger at EOF cannot create a zero-duration published note. The configured
minimum note duration is a retrigger guard, not a minimum for every emitted note.
"""

from __future__ import annotations

import math

import numpy as np


STANDARD_TUNING = (40, 45, 50, 55, 59, 64)
_TOLERANCE = 1e-9


class OnlineDecoder:
    """Decode blocks without revising an event that has already been published.

    ``push`` returns events produced by that block. With an explicit block
    availability, every event in the block has that same delivery time. Omitting
    it models frame-by-frame arrival. ``notes`` contains completed notes only;
    calling ``finish`` closes the remaining notes with ``censored_at_end=True``.
    Returned dictionaries are copies so later decoder work cannot alter them.
    """

    def __init__(self, config, parameters):
        feature = config['feature']
        self.frame_period = float(feature['hop_length']) / float(feature['sample_rate'])
        if not math.isfinite(self.frame_period) or self.frame_period <= 0:
            raise ValueError('The feature frame period must be positive and finite.')
        model = config['model']
        self.fret_count = int(model['max_fret']) + 1
        self.pitch_count = int(model['pitch_max']) - int(model['pitch_min']) + 1
        decoder = config['decoder']
        stream = config['streaming']
        self.confirm_frames = int(stream['confirm_frames'])
        self.release_confirm_frames = int(stream['release_confirm_frames'])
        if self.confirm_frames < 1 or self.release_confirm_frames < 1:
            raise ValueError('Confirmation lengths must be positive.')
        self.minimum_note_seconds = float(decoder['minimum_note_seconds'])
        self.minimum_retrigger_seconds = float(decoder['minimum_retrigger_seconds'])
        self.onset_threshold = float(parameters['onset_threshold'])
        self.activity_threshold = float(parameters['activity_threshold'])
        self.onset_backdate_seconds = float(parameters.get('onset_backdate_seconds', 0.0))
        finite = (self.minimum_note_seconds, self.minimum_retrigger_seconds,
                  self.onset_threshold, self.activity_threshold, self.onset_backdate_seconds)
        if not all(math.isfinite(value) for value in finite):
            raise ValueError('Decoder parameters must be finite.')
        if min(self.minimum_note_seconds, self.minimum_retrigger_seconds,
               self.onset_backdate_seconds) < 0:
            raise ValueError('Note durations and backdating must not be negative.')
        if not 0 <= self.onset_threshold <= 1 or not 0 <= self.activity_threshold <= 1:
            raise ValueError('Probability thresholds must lie in [0, 1].')
        self._framecount = 0
        self._next_note_id = 0
        self._last_available = 0.0
        self._finished = False
        self._events = []
        self._notes = []
        self._strings = [
            {'active': None, 'candidate': None, 'candidate_count': 0,
             'candidate_start': None, 'inactive_count': 0, 'inactive_start': None,
             'previous_onset_above': False, 'pending_retrigger': None, 'last_end': 0.0}
            for _ in STANDARD_TUNING
        ]

    @property
    def framecount(self):
        return self._framecount

    @property
    def frame_count(self):
        """Readable alias for the number of consumed frames."""
        return self._framecount

    @property
    def events(self):
        return [dict(event) for event in self._events]

    @property
    def notes(self):
        return [dict(note) for note in sorted(
            self._notes, key=lambda note: (note['start'], note['pitch'], note['string'], note['note_id']))]

    def _publish(self, kind, note, source_time, available, flush=False):
        event = {
            'type': kind, 'note_id': note['note_id'], 'string': note['string'],
            'fret': note['fret'], 'pitch': note['pitch'],
            'time': float(source_time), 'source_time': float(source_time),
            'emitted_at_audio_seconds': float(available), 'flush': bool(flush),
        }
        event['start' if kind == 'note_on' else 'end'] = float(source_time)
        self._events.append(event)

    def _open(self, string, fret, start, available):
        slot = self._strings[string]
        start = max(float(start), 0.0, slot['last_end'])
        note = {'note_id': self._next_note_id, 'string': string, 'fret': int(fret),
                'pitch': STANDARD_TUNING[string] + int(fret), 'program': 24,
                'start': start, 'on_emitted_at_audio_seconds': float(available)}
        self._next_note_id += 1
        slot['active'] = note
        self._publish('note_on', note, start, available)

    def _close(self, string, end, available, flush=False):
        slot = self._strings[string]
        note = slot['active']
        if note is None:
            return
        if end <= note['start']:
            raise ValueError('An emitted note must finish after its published attack.')
        completed = dict(note, end=float(end), off_emitted_at_audio_seconds=float(available),
                         censored_at_end=bool(flush))
        self._notes.append(completed)
        self._publish('note_off', note, end, available, flush=flush)
        slot['last_end'] = float(end)
        slot['active'] = None

    @staticmethod
    def _clear_candidate(slot):
        slot['candidate'] = None
        slot['candidate_count'] = 0
        slot['candidate_start'] = None

    def _validate_probabilities(self, probabilities):
        expected = {'state': (6, self.fret_count + 1), 'onset': (6,),
                    'offset': (6,), 'pitch': (self.pitch_count,)}
        arrays = {}
        length = None
        for key, trailing_shape in expected.items():
            if key not in probabilities:
                raise ValueError(f'Missing probability head: {key}.')
            value = np.asarray(probabilities[key])
            if value.ndim != 1 + len(trailing_shape) or value.shape[1:] != trailing_shape:
                raise ValueError(f'Unexpected {key} probability shape: {value.shape}.')
            if length is None:
                length = len(value)
            elif length != len(value):
                raise ValueError('Probability heads must have the same frame count.')
            if not np.all(np.isfinite(value)) or np.any(value < 0) or np.any(value > 1):
                raise ValueError('Probabilities must be finite and lie in [0, 1].')
            arrays[key] = value
        return arrays, length

    def push(self, probabilities, available_at_audio_seconds=None):
        """Consume a contiguous block, preserving previously emitted events.

        Explicit availability must be monotonic and at least the block's last
        frame time. It represents audio availability, not elapsed model compute.
        """
        if self._finished:
            raise RuntimeError('Cannot push probabilities after finish.')
        arrays, length = self._validate_probabilities(probabilities)
        if not length:
            return []
        block_end = (self._framecount + length) * self.frame_period
        available = None
        if available_at_audio_seconds is not None:
            available = float(available_at_audio_seconds)
            if not math.isfinite(available) or available + _TOLERANCE < max(block_end, self._last_available):
                raise ValueError('Block availability must follow all its frames and prior availability.')
        elif (self._framecount + 1) * self.frame_period + _TOLERANCE < self._last_available:
            raise ValueError('Frame-by-frame arrival cannot precede prior block availability.')
        first_event = len(self._events)
        state = arrays['state']
        best = state[:, :, :-1].max(axis=-1)
        activity = best / (best + state[:, :, -1] + 1e-8)
        frets = state[:, :, :-1].argmax(axis=-1)
        for frame in range(length):
            timestamp = (self._framecount + 1) * self.frame_period
            delivery = timestamp if available is None else available
            for string, slot in enumerate(self._strings):
                above = bool(arrays['onset'][frame, string] >= self.onset_threshold)
                rising = above and not slot['previous_onset_above']
                slot['previous_onset_above'] = above
                fret = int(frets[frame, string]) if activity[frame, string] >= self.activity_threshold else None
                active = slot['active']
                if fret is None:
                    self._clear_candidate(slot)
                    slot['pending_retrigger'] = None
                    if active is not None:
                        if slot['inactive_count'] == 0:
                            slot['inactive_start'] = timestamp
                        slot['inactive_count'] += 1
                        if slot['inactive_count'] >= self.release_confirm_frames:
                            end = max(slot['inactive_start'], active['start'] + _TOLERANCE)
                            self._close(string, end, delivery)
                    continue
                slot['inactive_count'] = 0
                slot['inactive_start'] = None
                if active is not None and fret == active['fret']:
                    self._clear_candidate(slot)
                    if slot['pending_retrigger'] is not None:
                        # Wait for one more same-fret frame before publishing.
                        # The attack keeps the trigger's source time, while
                        # delivery reflects this later confirmation frame.
                        trigger = slot['pending_retrigger']
                        self._close(string, trigger, delivery)
                        self._open(string, fret, trigger, delivery)
                        slot['pending_retrigger'] = None
                        active = slot['active']
                    elapsed = timestamp - active['start']
                    if (rising and elapsed + _TOLERANCE >= self.minimum_retrigger_seconds
                            and elapsed + _TOLERANCE >= self.minimum_note_seconds):
                        slot['pending_retrigger'] = timestamp
                    continue
                slot['pending_retrigger'] = None
                if slot['candidate'] == fret:
                    slot['candidate_count'] += 1
                else:
                    slot['candidate'] = fret
                    slot['candidate_count'] = 1
                    slot['candidate_start'] = timestamp
                if slot['candidate_count'] >= self.confirm_frames:
                    start = max(0.0, slot['candidate_start'] - self.onset_backdate_seconds,
                                slot['last_end'])
                    if active is not None:
                        # A backdated fret change cannot cross a published attack.
                        start = max(start, active['start'] + _TOLERANCE)
                        self._close(string, start, delivery)
                    self._open(string, fret, start, delivery)
                    self._clear_candidate(slot)
            self._framecount += 1
            self._last_available = float(delivery)
        return [dict(event) for event in self._events[first_event:]]

    def finish(self, duration, available_at_audio_seconds=None):
        """Close active notes at EOF; these releases are explicitly censored.

        Unconfirmed activity and retrigger candidates produce no events. Duration
        must include every consumed frame; callers must not push a zero-padded
        future frame at EOF.
        Repeated calls return no additional events.
        """
        if self._finished:
            return []
        duration = float(duration)
        if not math.isfinite(duration) or duration < 0 or duration + _TOLERANCE < self._framecount * self.frame_period:
            raise ValueError('Duration must include every consumed frame.')
        available = (max(duration, self._last_available) if available_at_audio_seconds is None
                     else float(available_at_audio_seconds))
        if not math.isfinite(available) or available + _TOLERANCE < max(duration, self._last_available):
            raise ValueError('Finish availability must follow the duration and prior availability.')
        if any(slot['active'] is not None and duration <= slot['active']['start'] for slot in self._strings):
            raise ValueError('Finish duration must follow every published attack.')
        first_event = len(self._events)
        for string, slot in enumerate(self._strings):
            self._close(string, duration, available, flush=True)
            self._clear_candidate(slot)
            slot['pending_retrigger'] = None
        self._last_available = available
        self._finished = True
        return [dict(event) for event in self._events[first_event:]]
