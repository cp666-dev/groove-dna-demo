"""AI feel tools on top of Claude — the parts that make the tool *understand* and
*transform* feel, not just measure it. All reuse the structured-output pattern from
compose.py.

  - coach(t)        -> critique of why a groove sounds stiff + a humanized "fixed" groove
  - edit(t, instr)  -> the groove edited per a natural-language instruction
  - variation(t)    -> a tasteful in-style variation or fill

Each returns plain dicts; groove dicts use the same schema as compose (turn them into
templates with compose.groove_to_template).
"""
from __future__ import annotations

import json

from . import VOICES
from .template import Template
from .compose import MODEL, _schema


def _client(api_key: str | None = None):
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            'Anthropic SDK not installed. Run: pip install "groovedna[compose]"') from e
    return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()


def _summary(t: Template) -> dict:
    """Compact, model-friendly view of the current groove's pattern + feel."""
    g = {"name": t.name, "bpm": t.grid.bpm, "beats_per_bar": t.grid.beats_per_bar,
         "subdivision": t.grid.subdivision, "bars": t.bars, "hits": []}
    for v in VOICES:
        for slot, c in sorted(t.voices.get(v, {}).items()):
            if not c.observed:
                continue  # only real hits describe the pattern
            g["hits"].append({
                "voice": v, "slot": slot,
                "velocity": round(c.velocity["mean"], 3),
                "timing": round(c.timing["mean"], 3),
                "timing_std": round(c.timing["std"], 3),
                "velocity_std": round(c.velocity["std"], 3),
            })
    return g


def _text(resp) -> str:
    return next(b.text for b in resp.content if b.type == "text")


# ---------------------------------------------------------------- Feel Coach ----
_COACH_SYSTEM = """\
You are a world-class drum feel coach. You are given a programmed drum groove as JSON:
each hit has a voice, grid slot, velocity (0..1), and micro-timing as a PERCENT of one
grid slot (negative = ahead/rushing, positive = behind/dragging), plus take-to-take
std deviations.

Diagnose what makes it sound stiff, robotic, or wrong for its apparent style, in
concrete musical terms — dead-even velocities, machine-perfect timing, a backbeat that
should drag, hats that should push or swing, missing ghost notes, etc. Be specific and
actionable; reference the beat (1-based, e.g. "2" or the "e/&/a" of a beat).

Then produce a "fixed" groove: the SAME pattern, humanized — apply the timing/velocity
feel and looseness you'd coach, so the user can hear your advice with one click.
"""

_COACH_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string"},
        "humanness": {"type": "integer"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "voice": {"type": "string"},
                    "where": {"type": "string"},
                    "issue": {"type": "string"},
                    "fix": {"type": "string"},
                },
                "required": ["voice", "where", "issue", "fix"],
            },
        },
        "fixed": _schema(),
    },
    "required": ["verdict", "humanness", "findings", "fixed"],
}


def coach(t: Template, api_key: str | None = None) -> dict:
    client = _client(api_key)
    resp = client.messages.create(
        model=MODEL, max_tokens=8000, thinking={"type": "adaptive"},
        system=_COACH_SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": _COACH_SCHEMA}},
        messages=[{"role": "user", "content": json.dumps(_summary(t))}],
    )
    return json.loads(_text(resp))


# ------------------------------------------------------- Natural-language edit ----
_EDIT_SYSTEM = """\
You edit an existing drum groove per the user's instruction and return the FULL edited
groove in the given schema. Keep everything the instruction doesn't touch. Slots are
global 0-based across the phrase (slots_per_bar = beats_per_bar * subdivision; a
multi-bar part runs 0 .. slots_per_bar*bars-1). Timing is a percent of one slot
(negative = ahead, positive = behind); velocity is 0..1. Be musical — if the user says
"busier hats" add idiomatic hat subdivisions; "push the snare" means make its timing
more negative; "more Dilla" means loosen and drag with wonky, uneven feel.

FEEL-ONLY edits: if the instruction asks to apply a named drummer's or style's FEEL
without rewriting the part (e.g. "play this exact part with Bernard Purdie's feel"),
KEEP the identical set of hits (same voices + slots) and only reshape their timing,
velocity, and take-to-take std to embody that drummer. Do not add or remove notes.

Always set `reasoning` to 2-4 plain-English sentences naming the specific traits you
applied, and `kit` to a suggested sample per voice used.
"""


def edit(t: Template, instruction: str, api_key: str | None = None) -> dict:
    client = _client(api_key)
    user = (f"Current groove:\n{json.dumps(_summary(t))}\n\n"
            f"Instruction: {instruction.strip()}")
    resp = client.messages.create(
        model=MODEL, max_tokens=8000, thinking={"type": "adaptive"},
        system=_EDIT_SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": _schema()}},
        messages=[{"role": "user", "content": user}],
    )
    return json.loads(_text(resp))


# ------------------------------------------------------------- Variations/fills ----
_VAR_SYSTEM = """\
You are a drummer creating a tasteful, in-style variation or fill of a base groove.
Return the FULL groove in the given schema, same tempo/subdivision/style. A "fill"
should build tension and resolve (often busier toms/snare into beat 1); a "variation"
keeps the feel but changes the pattern enough to relieve repetition.
"""


def variation(t: Template, kind: str = "variation", api_key: str | None = None) -> dict:
    client = _client(api_key)
    user = (f"Base groove:\n{json.dumps(_summary(t))}\n\n"
            f"Create a {kind} that fits the same style and feel.")
    resp = client.messages.create(
        model=MODEL, max_tokens=8000, thinking={"type": "adaptive"},
        system=_VAR_SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": _schema()}},
        messages=[{"role": "user", "content": user}],
    )
    return json.loads(_text(resp))
