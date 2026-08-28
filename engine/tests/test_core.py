"""Core pipeline tests — no audio deps. Prove the two claims that make this a
*transplant* rather than a fancy quantize: (1) tempo-relative timing, and
(2) interpolation of never-played grid positions.
"""
import numpy as np

from groovedna.grid import Grid
from groovedna.separation import Hit
from groovedna.template import Template
from groovedna.transplant import apply_template
from groovedna.midi_io import MidiNote


def _shuffle_template(bpm=96.0):
    grid = Grid(bpm=bpm, beats_per_bar=4, subdivision=4, origin=0.0)
    slot = grid.slot_dur

    def pct(p):
        return (p / 100.0) * slot

    hits = []
    for bar in range(4):
        base = bar * grid.slots_per_bar * slot
        for s in (0, 8):
            hits.append(Hit(base + s * slot + pct(-3.0), "kick", 0.98, 0.9))
        for s in (4, 12):
            hits.append(Hit(base + s * slot + pct(6.5), "snare", 0.82, 0.9))
        for s in range(0, 16, 2):  # hats on 8ths only
            swing = pct(9.0) if (s // 2) % 2 == 1 else pct(-1.0)
            hits.append(Hit(base + s * slot + swing, "hihat", 0.6, 0.8))
    return Template.from_hits(hits, grid, name="test").interpolate(decay=1.5)


def test_observed_slots_measured():
    t = _shuffle_template()
    # kick played slots 0 and 8, pushed ~3% early
    assert t.lookup("kick", 0).observed is True
    assert t.lookup("kick", 0).timing["mean"] < 0
    # snare backbeat drags late
    assert t.lookup("snare", 4).timing["mean"] > 0


def test_interpolation_fills_never_played_slots():
    t = _shuffle_template()
    # hats never played odd 16ths; those slots must now exist, be interpolated,
    # and carry a non-flat feel derived from neighbours.
    odd = t.lookup("hihat", 1)
    assert odd is not None
    assert odd.observed is False
    lo = min(t.lookup("hihat", 0).timing["mean"], t.lookup("hihat", 2).timing["mean"])
    hi = max(t.lookup("hihat", 0).timing["mean"], t.lookup("hihat", 2).timing["mean"])
    assert lo - 1e-6 <= odd.timing["mean"] <= hi + 1e-6


def test_transplant_moves_notes_and_respects_blend():
    t = _shuffle_template()
    grid = Grid(bpm=96.0, subdivision=4)
    target = [MidiNote(s * grid.slot_dur, s * grid.slot_dur + 0.05, 42, 90, "hihat")
              for s in range(16)]

    at_zero = apply_template(target, 96.0, t, amount=0.0, seed=1)
    assert all(abs(a.start - b.start) < 1e-9 for a, b in zip(target, at_zero)), \
        "amount=0 must be a no-op"

    at_full = apply_template(target, 96.0, t, amount=1.0, seed=1)
    moved = sum(1 for a, b in zip(target, at_full) if abs(a.start - b.start) > 1e-6)
    assert moved >= 12, "full transplant should move most notes off the grid"


def test_timing_is_tempo_relative():
    """Same template, two tempos: the offset in *beats* must be identical, so the
    offset in seconds scales with tempo (the whole point of storing %)."""
    t = _shuffle_template(bpm=96.0)
    note_slow = [MidiNote(0.0, 0.05, 38, 100, "snare")]  # slot 0

    def offset_beats(bpm):
        g = Grid(bpm=bpm, subdivision=4)
        # put a snare on slot 4 (a backbeat the ref actually played)
        s4 = 4 * g.slot_dur
        n = [MidiNote(s4, s4 + 0.05, 38, 100, "snare")]
        out = apply_template(n, bpm, t, amount=1.0, humanize=0.0, seed=0)
        delta_s = out[0].start - s4
        return delta_s / g.beat_dur  # convert to beats

    slow = offset_beats(96.0)
    fast = offset_beats(150.0)
    assert abs(slow - fast) < 1e-6, "timing offset should be constant in beats"
    assert abs(slow) > 0
