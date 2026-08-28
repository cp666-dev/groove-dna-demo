"""Beat / downbeat tracking and grid definition.

Establishes tempo, the location of beat 1, and a subdivision grid so every onset
(and every target MIDI note) can be assigned to an integer slot within a bar plus
a signed offset expressed as a % of the subdivision interval.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Grid:
    bpm: float
    beats_per_bar: int = 4
    subdivision: int = 4          # slices per beat: 4 => 16th grid
    origin: float = 0.0           # time (s) of the first downbeat

    @property
    def slots_per_bar(self) -> int:
        return self.beats_per_bar * self.subdivision

    @property
    def beat_dur(self) -> float:
        return 60.0 / self.bpm

    @property
    def slot_dur(self) -> float:
        """Duration of one subdivision slot, in seconds."""
        return self.beat_dur / self.subdivision

    def quantize(self, t: float) -> tuple[int, float]:
        """Map an absolute time to (slot_in_bar, offset_pct).

        offset_pct is the signed deviation from the nearest grid line as a
        percentage of one slot: negative = early (ahead), positive = late (behind).
        Range is nominally (-50, +50].
        """
        rel = (t - self.origin) / self.slot_dur          # position in slot-units
        nearest = round(rel)
        offset_pct = (rel - nearest) * 100.0
        slot_in_bar = int(nearest) % self.slots_per_bar
        return slot_in_bar, float(offset_pct)

    def slot_time(self, bar: int, slot_in_bar: int) -> float:
        """Absolute grid time (s) for a given bar + slot index."""
        return self.origin + (bar * self.slots_per_bar + slot_in_bar) * self.slot_dur

    def nearest_grid_time(self, t: float) -> float:
        """Absolute time (s) of the grid line closest to t."""
        rel = (t - self.origin) / self.slot_dur
        return self.origin + round(rel) * self.slot_dur


def track_grid(y: np.ndarray, sr: int, beats_per_bar: int = 4,
               subdivision: int = 4) -> Grid:
    """Detect tempo + downbeat from an audio buffer with librosa.

    Falls back gracefully if beat tracking is weak. madmom gives better downbeats
    but adds a heavy dependency; librosa is the MVP default per the brief.
    """
    import librosa

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    tempo = float(np.atleast_1d(tempo)[0])
    origin = float(beats[0]) if len(beats) else 0.0
    if not np.isfinite(tempo) or tempo <= 0:
        tempo = 120.0
    return Grid(bpm=round(tempo, 2), beats_per_bar=beats_per_bar,
                subdivision=subdivision, origin=origin)
