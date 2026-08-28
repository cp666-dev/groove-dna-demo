"""AI groove composer — generate a drum *pattern* and its micro-timing *feel*
from a natural-language description, using Claude.

This is the AI-necessary core: a rule engine can quantize notes to a grid, but it
cannot invent a genre-appropriate pattern *and* the relational feel (swing, ghost
notes, backbeat drag, kick pocket) from "a laid-back 70s soul groove at 88 bpm".
The model emits a validated JSON groove which we turn into the same `.groovedna`
template + MIDI the rest of the engine already speaks — so `viz`, `apply`, and the
Logic Scripter all work on generated grooves unchanged.

Requires the Anthropic SDK (`pip install "groovedna[compose]"`) and credentials:
ANTHROPIC_API_KEY, or an `ant auth login` profile.
"""
from __future__ import annotations

from . import VOICES
from .grid import Grid
from .template import Template, Cell
from .voices import default_note
from .midi_io import MidiNote

MODEL = "claude-opus-4-8"

_SYSTEM = """\
You are an expert drum programmer and session drummer. You translate a plain-English
brief into a precise, musical drum groove: the pattern (which drums hit where) AND
the human micro-timing/velocity feel that makes it groove.

The grid: a bar has `beats_per_bar` beats, each split into `subdivision` slots, so
slots_per_bar = beats_per_bar * subdivision (16 for 4/4 sixteenths). Slot 0 is beat 1.

For every hit you place, give:
- voice: one of kick, snare, tom, hihat, cymbal
- slot: integer 0..slots_per_bar-1
- velocity: 0..1 (1 = hardest). Use dynamics — accents, ghost notes (~0.15-0.3).
- timing: micro-timing as a PERCENT of one slot. NEGATIVE = ahead/rushing (on top of
  the beat), POSITIVE = behind/dragging (laid back). Range about -25..25. This is the
  feel: e.g. a laid-back backbeat snare drags +6..+12; a pushed hat sits -3..-6;
  swung offbeat hats land around +15..+20.
- timing_std / velocity_std: how much this hit varies take-to-take (human looseness).
  Tight machine-like parts ~0-2 timing_std; loose human parts 3-8.

Rules:
- Make it genuinely idiomatic for the requested style; don't just fill the grid.
- Encode the *relationships* that define the feel (hat vs kick, backbeat placement).
- Default to 1 bar unless the brief implies a longer phrase; keep bars <= 4.
- Pick a sensible bpm and subdivision if the brief doesn't specify.
"""


def _schema() -> dict:
    voices = list(VOICES)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "bpm": {"type": "number"},
            "beats_per_bar": {"type": "integer"},
            "subdivision": {"type": "integer"},
            "bars": {"type": "integer"},
            "hits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "voice": {"type": "string", "enum": voices},
                        "slot": {"type": "integer"},
                        "velocity": {"type": "number"},
                        "timing": {"type": "number"},
                        "timing_std": {"type": "number"},
                        "velocity_std": {"type": "number"},
                    },
                    "required": ["voice", "slot", "velocity", "timing",
                                 "timing_std", "velocity_std"],
                },
            },
        },
        "required": ["name", "bpm", "beats_per_bar", "subdivision", "bars", "hits"],
    }


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def groove_to_template(g: dict) -> Template:
    """Turn a generated-groove dict into a .groovedna Template (observed cells)."""
    grid = Grid(bpm=float(g["bpm"]),
                beats_per_bar=int(g.get("beats_per_bar", 4)),
                subdivision=int(g.get("subdivision", 4)))
    voices: dict[str, dict[int, Cell]] = {v: {} for v in VOICES}
    for h in g["hits"]:
        v = h["voice"]
        if v not in voices:
            continue
        slot = int(h["slot"]) % grid.slots_per_bar
        voices[v][slot] = Cell(
            timing={"mean": round(float(h["timing"]), 3),
                    "std": round(max(0.0, float(h.get("timing_std", 0))), 3)},
            velocity={"mean": round(_clamp01(h["velocity"]), 3),
                      "std": round(max(0.0, float(h.get("velocity_std", 0))), 3)},
            hits=1, observed=True)
    return Template(grid=grid, voices=voices,
                    name=g.get("name", "generated"), bars=int(g.get("bars", 1)))


def groove_to_notes(g: dict) -> list[MidiNote]:
    """Turn a generated groove into quantized MIDI notes (feel lives in the template)."""
    grid = Grid(bpm=float(g["bpm"]),
                beats_per_bar=int(g.get("beats_per_bar", 4)),
                subdivision=int(g.get("subdivision", 4)))
    slot = grid.slot_dur
    bars = max(1, int(g.get("bars", 1)))
    notes: list[MidiNote] = []
    for bar in range(bars):
        base = bar * grid.slots_per_bar * slot
        for h in g["hits"]:
            s = int(h["slot"]) % grid.slots_per_bar
            t = base + s * slot
            notes.append(MidiNote(
                start=t, end=t + 0.05, pitch=default_note(h["voice"]),
                velocity=int(round(_clamp01(h["velocity"]) * 126)) + 1,
                voice=h["voice"]))
    notes.sort(key=lambda n: n.start)
    return notes


def generate(prompt: str, bpm: float | None = None, bars: int | None = None,
             model: str = MODEL, api_key: str | None = None) -> dict:
    """Call Claude and return the validated groove dict.

    Raises RuntimeError with a clear message if the SDK isn't installed or no
    credentials are available.
    """
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            'Anthropic SDK not installed. Run: pip install "groovedna[compose]"') from e

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    brief = prompt.strip()
    if bpm:
        brief += f"\n(Target tempo: {bpm:.0f} BPM.)"
    if bars:
        brief += f"\n(Write {bars} bar(s).)"

    resp = client.messages.create(
        model=model,
        max_tokens=8000,
        thinking={"type": "adaptive"},
        system=_SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": _schema()}},
        messages=[{"role": "user", "content": brief}],
    )
    import json
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text)
