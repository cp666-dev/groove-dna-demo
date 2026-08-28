"""Build a `.groovedna` template from separated hits, including the novel
interpolation step that fills grid positions the reference never played.

See SCHEMA.md for the on-disk format.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

import numpy as np

from . import VOICES
from .grid import Grid
from .separation import Hit

# On-disk schema version (see SCHEMA.md). Independent of the package version.
FORMAT_VERSION = "groovedna/v1"


@dataclass
class Cell:
    timing: dict            # {"mean": %, "std": %}
    velocity: dict          # {"mean": 0..1, "std": 0..1}
    hits: int
    observed: bool


@dataclass
class Template:
    grid: Grid
    voices: dict[str, dict[int, Cell]] = field(default_factory=dict)
    name: str = "untitled"
    bars: int = 0

    # ---- construction -----------------------------------------------------
    @classmethod
    def from_hits(cls, hits: list[Hit], grid: Grid, name: str = "untitled",
                  min_confidence: float = 0.0) -> "Template":
        """Aggregate observed hits into per-(voice, slot) distributions."""
        # slot -> lists of (offset_pct, velocity)
        buckets: dict[str, dict[int, list[tuple[float, float]]]] = {
            v: {} for v in VOICES
        }
        max_bar = 0
        for h in hits:
            if h.confidence < min_confidence:
                continue
            slot, off = grid.quantize(h.time)
            bar = int((h.time - grid.origin) / (grid.slot_dur * grid.slots_per_bar))
            max_bar = max(max_bar, bar)
            buckets.setdefault(h.voice, {}).setdefault(slot, []).append(
                (off, h.velocity))

        voices: dict[str, dict[int, Cell]] = {}
        for v in VOICES:
            slots: dict[int, Cell] = {}
            for slot, samples in buckets.get(v, {}).items():
                offs = np.array([s[0] for s in samples], dtype=float)
                vels = np.array([s[1] for s in samples], dtype=float)
                slots[slot] = Cell(
                    timing={"mean": round(float(offs.mean()), 3),
                            "std": round(float(offs.std()), 3)},
                    velocity={"mean": round(float(vels.mean()), 3),
                              "std": round(float(vels.std()), 3)},
                    hits=len(samples),
                    observed=True,
                )
            voices[v] = slots
        return cls(grid=grid, voices=voices, name=name, bars=max_bar + 1)

    # ---- the novel step: interpolation -----------------------------------
    def interpolate(self, decay: float = 1.5) -> "Template":
        """Fill every grid slot the reference never played, per voice, by
        proximity-weighted interpolation from that voice's *observed* slots.

        This is what turns "groove extraction" into "groove transplant": your
        16th-note hi-hat gets a plausible feel even where the drummer only
        played 8ths, instead of snapping robotically back to the grid.

        `decay` controls how fast influence falls with slot distance (higher =
        more local). Distance is circular within the bar.
        """
        n = self.grid.slots_per_bar
        for v in VOICES:
            observed = {s: c for s, c in self.voices[v].items() if c.observed}
            if not observed:
                continue  # voice never played at all -> leave empty (no basis)
            for slot in range(n):
                if slot in observed:
                    continue
                num_t = num_v = num_vs = num_ts = denom = 0.0
                for os, cell in observed.items():
                    d = min(abs(slot - os), n - abs(slot - os))  # circular
                    w = 1.0 / (1.0 + d) ** decay
                    num_t += w * cell.timing["mean"]
                    num_ts += w * cell.timing["std"]
                    num_v += w * cell.velocity["mean"]
                    num_vs += w * cell.velocity["std"]
                    denom += w
                self.voices[v][slot] = Cell(
                    timing={"mean": round(num_t / denom, 3),
                            "std": round(num_ts / denom, 3)},
                    velocity={"mean": round(num_v / denom, 3),
                              "std": round(num_vs / denom, 3)},
                    hits=0,
                    observed=False,
                )
        return self

    def lookup(self, voice: str, slot: int) -> Cell | None:
        return self.voices.get(voice, {}).get(slot)

    # ---- serialisation ----------------------------------------------------
    def to_dict(self) -> dict:
        from .voices import gm_notes
        total_hits = sum(c.hits for slots in self.voices.values()
                         for c in slots.values())
        return {
            "format": FORMAT_VERSION,
            "source": {"name": self.name, "bpm": self.grid.bpm,
                       "bars": self.bars, "hits": total_hits},
            "grid": {
                "beats_per_bar": self.grid.beats_per_bar,
                "subdivision": self.grid.subdivision,
                "slots_per_bar": self.grid.slots_per_bar,
            },
            "voices": {
                v: {"gm_notes": list(gm_notes(v)),
                    "slots": {str(s): asdict(c) for s, c in sorted(slots.items())}}
                for v, slots in self.voices.items()
            },
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Template":
        with open(path) as f:
            d = json.load(f)
        g = Grid(bpm=d["source"]["bpm"],
                 beats_per_bar=d["grid"]["beats_per_bar"],
                 subdivision=d["grid"]["subdivision"])
        voices: dict[str, dict[int, Cell]] = {}
        for v, vd in d["voices"].items():
            voices[v] = {int(s): Cell(**c) for s, c in vd["slots"].items()}
        return cls(grid=g, voices=voices, name=d["source"]["name"],
                   bars=d["source"].get("bars", 0))
