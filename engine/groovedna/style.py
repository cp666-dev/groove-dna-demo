"""Drummer-blend v1 (heuristic feel embedding + morph).

`blend(a, b, w)` keeps groove **a**'s pattern (which drums hit where) and morphs its
*feel* — micro-timing and velocity — toward groove **b** by weight `w` (0 = pure a,
1 = a's notes played with b's feel). This is the "play it like ___" primitive; a learned
embedding space (docs/AI_ROADMAP.md #3 v2) can replace the interpolation later without
changing the interface.

`embed(t)` returns a small feel feature vector (swing, push/drag per voice, dynamics)
useful for library search and, later, training a learned space.
"""
from __future__ import annotations

from . import VOICES
from .template import Template, Cell


def _lerp(x: float, y: float, w: float) -> float:
    return x + (y - x) * w


def blend(a: Template, b: Template, w: float) -> Template:
    """Morph a's feel toward b by w in [0,1]. Keeps a's pattern, tempo, and grid."""
    w = max(0.0, min(1.0, float(w)))
    voices: dict[str, dict[int, Cell]] = {}
    for v in VOICES:
        av = a.voices.get(v, {})
        bv = b.voices.get(v, {})
        slots: dict[int, Cell] = {}
        for slot, ca in av.items():
            cb = bv.get(slot)
            if cb is not None:
                slots[slot] = Cell(
                    timing={"mean": round(_lerp(ca.timing["mean"], cb.timing["mean"], w), 3),
                            "std": round(_lerp(ca.timing["std"], cb.timing["std"], w), 3)},
                    velocity={"mean": round(_lerp(ca.velocity["mean"], cb.velocity["mean"], w), 3),
                              "std": round(_lerp(ca.velocity["std"], cb.velocity["std"], w), 3)},
                    hits=ca.hits, observed=ca.observed)
            else:
                slots[slot] = ca  # b never played here — keep a's own feel
        voices[v] = slots
    name = f"{a.name} × {b.name} ({round(w * 100)}%)"
    return Template(grid=a.grid, voices=voices, name=name, bars=a.bars)


def embed(t: Template) -> dict:
    """Heuristic feel embedding: per-voice mean push/drag + dynamics, and overall swing."""
    feat: dict[str, float] = {}
    for v in VOICES:
        cells = [c for c in t.voices.get(v, {}).values() if c.observed]
        if not cells:
            feat[f"{v}_timing"] = 0.0
            feat[f"{v}_dyn"] = 0.0
            continue
        tim = [c.timing["mean"] for c in cells]
        vel = [c.velocity["mean"] for c in cells]
        feat[f"{v}_timing"] = round(sum(tim) / len(tim), 3)          # + = drags, - = pushes
        vmean = sum(vel) / len(vel)
        feat[f"{v}_dyn"] = round((sum((x - vmean) ** 2 for x in vel) / len(vel)) ** 0.5, 3)
    # swing: how much off-8th hats lag vs on-8th hats (positive = swung)
    hats = {s: c for s, c in t.voices.get("hihat", {}).items() if c.observed}
    on = [c.timing["mean"] for s, c in hats.items() if (s // (t.grid.subdivision // 2)) % 2 == 0]
    off = [c.timing["mean"] for s, c in hats.items() if (s // (t.grid.subdivision // 2)) % 2 == 1]
    feat["swing"] = round((sum(off) / len(off) if off else 0) - (sum(on) / len(on) if on else 0), 3)
    return feat
