"""Publication-time source-clock compensation for delayed supervision.

Acoustic confirmation and internal decoder clocks remain the frozen Stage 8
behavior. Only public source timestamps are translated; availability and
delivery timestamps never move earlier.
"""

import math

from streaming.decoder import OnlineDecoder


_MINIMUM_PUBLIC_INTERVAL = 1e-9


class SourceTimeDecoder(OnlineDecoder):
    """Subtract a declared training-label delay from every published boundary.

    The decoder's existing onset backdate is independent and stays unchanged.
    It applies to new activity/fret changes inside the parent decoder. This
    source-clock translation applies once to all attacks, retriggers, releases
    and EOF closes after the parent makes its acoustic decision.
    """

    def __init__(self, config, parameters, source_delay_seconds=0.0):
        delay = float(source_delay_seconds)
        if not math.isfinite(delay) or delay < 0:
            raise ValueError("Source delay must be nonnegative and finite")
        self.source_delay_seconds = delay
        self._mapped_notes = {}
        self._public_last_end = [0.0] * 6
        self._clamp_counts = {"onset_zero": 0, "onset_previous_release": 0,
                              "offset_positive_interval": 0}
        super().__init__(config, parameters)

    @property
    def clamp_counts(self):
        counts = dict(self._clamp_counts)
        counts["total"] = sum(counts.values())
        return counts

    def _publish(self, kind, note, source_time, available, flush=False):
        note_id, string = note["note_id"], note["string"]
        # Keep exact identity arithmetic and ledger values when delay is zero.
        translated = (float(source_time) if self.source_delay_seconds == 0.0
                      else float(source_time) - self.source_delay_seconds)
        if kind == "note_on":
            if self.source_delay_seconds:
                if translated < 0.0:
                    translated = 0.0
                    self._clamp_counts["onset_zero"] += 1
                if translated < self._public_last_end[string]:
                    translated = self._public_last_end[string]
                    self._clamp_counts["onset_previous_release"] += 1
            self._mapped_notes[note_id] = {"start": translated}
        elif kind == "note_off":
            start = self._mapped_notes[note_id]["start"]
            if self.source_delay_seconds and translated <= start:
                translated = start + _MINIMUM_PUBLIC_INTERVAL
                self._clamp_counts["offset_positive_interval"] += 1
            self._mapped_notes[note_id]["end"] = translated
            self._public_last_end[string] = translated
        else:
            raise ValueError("Unknown public note event")
        super()._publish(kind, note, translated, available, flush)

    @property
    def notes(self):
        result = []
        for raw in super().notes:
            public = dict(raw)
            public.update(self._mapped_notes[raw["note_id"]])
            result.append(public)
        return sorted(result, key=lambda note: (
            note["start"], note["pitch"], note["string"], note["note_id"]))
