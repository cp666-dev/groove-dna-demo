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
For a multi-bar phrase, slots run CONTINUOUSLY across the whole phrase: slot index
ranges 0 .. slots_per_bar*bars - 1 (bar 2 beat 1 is slot slots_per_bar, etc.). Use
this to write real phrases — vary bars, drop a fill into the last bar, add a crash on
the downbeat of a new section.

Voices (a full kit — use whatever the style needs, not just the basics):
  kick, snare, clap, rim (side-stick/cross-stick), tom_hi, tom_mid, tom_lo,
  hihat (closed), open_hat, ride, crash, perc (tambourine/cowbell/shaker/aux).

For every hit you place, give:
- voice: one of the voices above (exact string)
- slot: integer 0..slots_per_bar*bars-1
- velocity: 0..1 (1 = hardest). Use dynamics — accents, ghost notes (~0.15-0.3).
- timing: micro-timing as a PERCENT of one slot. NEGATIVE = ahead/rushing (on top of
  the beat), POSITIVE = behind/dragging (laid back). Range about -25..25. This is the
  feel: e.g. a laid-back backbeat snare drags +6..+12; a pushed hat sits -3..-6;
  swung offbeat hats land around +15..+20.
- timing_std / velocity_std: how much this hit varies take-to-take (human looseness).
  Tight machine-like parts ~0-2 timing_std; loose human parts 3-8.

Also return:
- reasoning: 2-4 sentences, plain English, naming the specific traits of the requested
  drummer/style you implemented (e.g. "Purdie's ghost-note triplet snare, hats pushed
  ~4% ahead, backbeat dragged +8%"). This is shown to the user as the 'why'.
- kit: a suggested drum-sample palette — one entry per voice you actually used, each a
  short concrete sample description the user could load (e.g. voice "kick",
  sample "deep round 70s soul kick, felt beater, minimal click").

Rules:
- Make it genuinely idiomatic for the requested style; don't just fill the grid.
- Encode the *relationships* that define the feel (hat vs kick, backbeat placement).
- Reach for the fuller kit when the genre calls for it (ride-driven jazz, open-hat
  house, tom-heavy tribal/rock fills, clap-backbeat pop, rim cross-sticks for bossa).
- Default to 1 bar unless the brief implies a longer phrase; honour a requested bar
  count and keep bars <= 8.
- `subdivision` is slots PER BEAT: use 4 for normal sixteenth-note grids (the
  default for most styles), 3 or 6 for triplet/shuffle feels, 2 for straight
  eighths. Do NOT use large values — subdivision 4 in 4/4 already gives 16 slots/bar.
- TEMPO: choose a bpm that is genuinely typical for the style, and prefer the MIDDLE
  of its usual range — do NOT reach for the extreme fast end. Reference ranges (pick
  near the centre unless the brief says "fast"/"slow" or gives a bpm):
    ballad/soul 65-85 · hip-hop/boom-bap 82-96 · lo-fi 70-85 · reggae/dub 70-90 ·
    pop 100-118 · funk/disco 108-120 · house/techno 120-128 · rock 108-140 ·
    metal 140-180 · punk / pop-punk 150-190 (typical ~165, NOT ~198) ·
    drum'n'bass 168-176 · trap/half-time 130-150 (felt slow). A number the brief
    states always wins; otherwise err toward comfortable, not breakneck.
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
            "reasoning": {"type": "string"},
            "kit": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "voice": {"type": "string", "enum": voices},
                        "sample": {"type": "string"},
                    },
                    "required": ["voice", "sample"],
                },
            },
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
        "required": ["name", "bpm", "beats_per_bar", "subdivision", "bars",
                     "reasoning", "kit", "hits"],
    }


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def groove_to_template(g: dict) -> Template:
    """Turn a generated-groove dict into a .groovedna Template (observed cells).

    Slots are kept GLOBAL (0 .. slots_per_bar*bars - 1) so multi-bar phrases keep
    their per-bar variation and fills instead of being folded onto one bar.
    """
    from .voices import canonical_voice
    grid = Grid(bpm=float(g["bpm"]),
                beats_per_bar=int(g.get("beats_per_bar", 4)),
                subdivision=int(g.get("subdivision", 4)))
    bars = max(1, int(g.get("bars", 1)))
    total = grid.slots_per_bar * bars
    voices: dict[str, dict[int, Cell]] = {v: {} for v in VOICES}
    for h in g["hits"]:
        v = canonical_voice(h["voice"])
        if v not in voices:
            continue
        slot = int(h["slot"]) % total
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
