"""Groove-fingerprint visualisation: timing-deviation-per-grid-slot, per voice.

Two outputs:
  - a matplotlib PNG (quick, ships with the CLI)
  - a compact JSON payload the 3D web app consumes to render the animated
    fingerprint (observed vs interpolated cells kept distinct).
"""
from __future__ import annotations

import json

from . import VOICES
from .template import Template


def from_fingerprint(d: dict) -> Template:
    """Rebuild a Template from a fingerprint payload (the inverse of to_fingerprint_json).

    Used by the server so the app can send its current fingerprint to the AI feel
    tools (coach/edit/variations) without round-tripping a full .groovedna file.
    """
    from .grid import Grid
    from .template import Cell

    grid = Grid(bpm=float(d.get("bpm", 120)),
                beats_per_bar=int(d.get("beatsPerBar", 4)),
                subdivision=int(d.get("subdivision", 4)))
    voices: dict[str, dict[int, Cell]] = {}
    for vd in d.get("voices", []):
        slots: dict[int, Cell] = {}
        for c in vd.get("cells", []):
            slots[int(c["slot"])] = Cell(
                timing={"mean": float(c.get("timing", 0)),
                        "std": float(c.get("timingStd", 0))},
                velocity={"mean": float(c.get("velocity", 0.8)), "std": 0.0},
                hits=int(c.get("hits", 0)),
                observed=bool(c.get("observed", True)))
        voices[vd["voice"]] = slots
    return Template(grid=grid, voices=voices,
                    name=d.get("name", "groove"), bars=int(d.get("bars", 1)))


def to_fingerprint_json(t: Template) -> dict:
    """Flatten a template into a render-friendly payload for the frontend."""
    n = t.grid.slots_per_bar
    voices = []
    for v in VOICES:
        cells = []
        for slot in range(n):
            c = t.lookup(v, slot)
            if c is None:
                continue
            cells.append({
                "slot": slot,
                "timing": c.timing["mean"],
                "timingStd": c.timing["std"],
                "velocity": c.velocity["mean"],
                "observed": c.observed,
                "hits": c.hits,
            })
        voices.append({"voice": v, "cells": cells})
    return {
        "name": t.name, "bpm": t.grid.bpm, "slotsPerBar": n,
        "beatsPerBar": t.grid.beats_per_bar, "subdivision": t.grid.subdivision,
        "voices": voices,
    }


def save_fingerprint_json(t: Template, path: str) -> None:
    with open(path, "w") as f:
        json.dump(to_fingerprint_json(t), f, indent=2)


def save_png(t: Template, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = t.grid.slots_per_bar
    fig, axes = plt.subplots(len(VOICES), 1, figsize=(10, 8), sharex=True)
    for ax, v in zip(axes, VOICES):
        xs_obs, ys_obs, xs_int, ys_int = [], [], [], []
        for slot in range(n):
            c = t.lookup(v, slot)
            if c is None:
                continue
            if c.observed:
                xs_obs.append(slot); ys_obs.append(c.timing["mean"])
            else:
                xs_int.append(slot); ys_int.append(c.timing["mean"])
        ax.axhline(0, color="#888", lw=0.6)
        ax.bar(xs_int, ys_int, color="#c9b7ff", width=0.6, label="interpolated")
        ax.bar(xs_obs, ys_obs, color="#7b52ff", width=0.6, label="observed")
        ax.set_ylabel(v, rotation=0, ha="right", va="center")
        ax.set_ylim(-25, 25)
    axes[0].set_title(f"Groove fingerprint — {t.name} @ {t.grid.bpm} BPM "
                      f"(timing % of subdivision, +late / -early)")
    axes[-1].set_xlabel("grid slot within bar")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
