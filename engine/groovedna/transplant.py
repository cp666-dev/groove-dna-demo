"""Apply a `.groovedna` template to target MIDI: resolve each note's voice + grid
slot, sample the stored distribution, nudge timing (tempo-scaled) and scale
velocity, then blend against the original by `amount` (0 = original, 1 = full).
"""
from __future__ import annotations

import numpy as np

from .grid import Grid
from .midi_io import MidiNote
from .template import Template


def apply_template(notes: list[MidiNote], target_bpm: float, template: Template,
                   amount: float = 1.0, seed: int = 0,
                   humanize: float = 1.0) -> list[MidiNote]:
    """Return a new list of notes with the groove transplanted.

    Timing is tempo-portable: the stored offset is a % of the reference's
    subdivision, re-expressed in seconds against the *target* tempo's subdivision.
    `humanize` scales how much of each cell's std is sampled (0 = use the mean
    exactly, 1 = full stochastic spread).
    """
    rng = np.random.default_rng(seed)
    grid = Grid(bpm=target_bpm, beats_per_bar=template.grid.beats_per_bar,
                subdivision=template.grid.subdivision, origin=0.0)
    slot_dur = grid.slot_dur
    amount = float(max(0.0, min(1.0, amount)))

    out: list[MidiNote] = []
    for n in notes:
        if n.voice is None:
            out.append(n)
            continue
        slot, _orig_off = grid.quantize(n.start)
        cell = template.lookup(n.voice, slot)
        if cell is None:
            out.append(n)  # nothing to say about this voice/slot -> leave as-is
            continue

        # --- timing ---
        off_pct = cell.timing["mean"] + humanize * cell.timing["std"] * rng.standard_normal()
        delta_s = (off_pct / 100.0) * slot_dur         # tempo-scaled seconds
        new_start_full = grid.nearest_grid_time(n.start) + delta_s
        new_start = n.start + amount * (new_start_full - n.start)
        dur = n.end - n.start

        # --- velocity ---
        vscale = cell.velocity["mean"] + humanize * cell.velocity["std"] * rng.standard_normal()
        vscale = max(0.0, vscale)
        # Preserve the note's own accent: scale a normalised copy, not the raw value.
        target_v = float(np.clip(n.velocity * (0.5 + vscale), 1, 127))
        new_v = n.velocity + amount * (target_v - n.velocity)

        out.append(MidiNote(start=max(0.0, new_start), end=max(0.0, new_start) + dur,
                            pitch=n.pitch, velocity=int(round(new_v)), voice=n.voice))
    out.sort(key=lambda x: x.start)
    return out
