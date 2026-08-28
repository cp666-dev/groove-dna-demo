"""Generate demo artifacts + smoke-test the core (no audio deps needed).

Simulates a drummer who plays a laid-back shuffle: kick on beats, snare on the
backbeat dragging slightly late, hats on 8ths swung. The *target* MIDI plays
straight 16th hats — so the interpolation step has to invent feel for the 16th
positions the reference never hit. Writes:

  examples/demo.groovedna        the template
  examples/demo.fingerprint.json the payload the 3D frontend renders
  examples/demo.before.json      target notes, quantized
  examples/demo.after.json       target notes, transplanted (amount=1.0)
"""
from __future__ import annotations

import json
import os

from groovedna.grid import Grid
from groovedna.separation import Hit
from groovedna.template import Template
from groovedna.transplant import apply_template
from groovedna.midi_io import MidiNote

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "examples")
os.makedirs(OUT, exist_ok=True)

BPM = 96.0
grid = Grid(bpm=BPM, beats_per_bar=4, subdivision=4, origin=0.0)  # 16th grid
slot = grid.slot_dur
bars = 4


def pct(p: float) -> float:
    """percent-of-slot -> seconds offset."""
    return (p / 100.0) * slot


hits: list[Hit] = []
for bar in range(bars):
    base = bar * grid.slots_per_bar * slot
    # kick: slots 0 and 8 (beats 1 and 3), pushed a hair early, strong
    for s in (0, 8):
        hits.append(Hit(base + s * slot + pct(-3.0), "kick", 0.98, 0.9))
    # snare: backbeat slots 4 and 12, dragging late, medium-strong
    for s in (4, 12):
        hits.append(Hit(base + s * slot + pct(6.5), "snare", 0.82, 0.9))
    # hats: only EIGHTHS (even slots 0,2,4,...,14) with swing on the off-8ths
    for s in range(0, 16, 2):
        swing = pct(9.0) if (s // 2) % 2 == 1 else pct(-1.0)
        vel = 0.7 if s % 4 == 0 else 0.5
        hits.append(Hit(base + s * slot + swing, "hihat", vel, 0.8))

tmpl = Template.from_hits(hits, grid, name="Laid-back shuffle (demo)")
observed = {v: sorted(s for s, c in cs.items() if c.observed)
            for v, cs in tmpl.voices.items() if cs}
tmpl.interpolate(decay=1.5)

tmpl.save(os.path.join(OUT, "demo.groovedna"))
from groovedna.viz import save_fingerprint_json
save_fingerprint_json(tmpl, os.path.join(OUT, "demo.fingerprint.json"))

# --- target: STRAIGHT 16th hats + backbeat snare + four-on-floor kick --------
target: list[MidiNote] = []
for bar in range(2):
    base = bar * grid.slots_per_bar * slot
    for s in range(0, 16, 4):                       # kick 4-on-the-floor
        target.append(MidiNote(base + s * slot, base + s * slot + 0.1, 36, 100, "kick"))
    for s in (4, 12):                               # snare backbeat
        target.append(MidiNote(base + s * slot, base + s * slot + 0.1, 38, 100, "snare"))
    for s in range(16):                             # hats on ALL 16ths
        target.append(MidiNote(base + s * slot, base + s * slot + 0.05, 42, 90, "hihat"))

after = apply_template(target, BPM, tmpl, amount=1.0, seed=7)


def dump(notes, path):
    with open(path, "w") as f:
        json.dump([{"start": round(n.start, 5), "voice": n.voice,
                    "velocity": n.velocity} for n in notes], f, indent=2)


dump(target, os.path.join(OUT, "demo.before.json"))
dump(after, os.path.join(OUT, "demo.after.json"))

# --- report -----------------------------------------------------------------
print(f"reference observed slots per voice: {observed}")
n = grid.slots_per_bar
filled = {v: sum(1 for c in cs.values() if not c.observed)
          for v, cs in tmpl.voices.items() if cs}
print(f"interpolated (never-played) slots filled per voice: {filled}")

# Prove the odd (never-played) 16th hats got a non-zero, interpolated feel.
odd_hat = tmpl.lookup("hihat", 1)
even_hat = tmpl.lookup("hihat", 2)
print(f"hihat slot 1 (never played): observed={odd_hat.observed} "
      f"timing={odd_hat.timing['mean']}%  <- interpolated from neighbours")
print(f"hihat slot 2 (played 8th):   observed={even_hat.observed} "
      f"timing={even_hat.timing['mean']}%")

moved = sum(1 for a, b in zip(target, after) if abs(a.start - b.start) > 1e-6)
print(f"transplant moved {moved}/{len(target)} target notes off the grid")
assert odd_hat is not None and not odd_hat.observed
assert moved > 0, "transplant did nothing"
print("OK — core pipeline verified")
