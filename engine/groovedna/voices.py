"""Drum voice taxonomy: General MIDI note maps + spectral band profiles.

Two jobs live here:
  * MIDI side  -> map a GM drum note number to a voice (kick/snare/hihat/tom/cymbal).
  * Audio side -> a coarse frequency-band profile per voice, used by the cheap
    (no-ML) onset clustering in ``separation.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

# Canonical voice order (also the render order in the fingerprint UI, top -> bottom).
# A full kit so grooves can be idiomatic across genres (multiple toms, ride vs
# crash vs open hat, claps, rims, aux percussion) rather than five coarse buckets.
VOICE_ORDER = [
    "kick", "snare", "clap", "rim",
    "tom_hi", "tom_mid", "tom_lo",
    "hihat", "open_hat", "ride", "crash", "perc",
]

# Legacy 5-voice names -> nearest full-kit voice, so old fingerprints still load.
VOICE_ALIASES = {"tom": "tom_mid", "cymbal": "ride"}


@dataclass(frozen=True)
class Voice:
    name: str
    gm_notes: tuple[int, ...]
    # Coarse band the voice's energy concentrates in, in Hz (lo, hi).
    band_hz: tuple[float, float]
    # Typical transient length: cymbals ring, kicks are short. Purely a hint.
    ring: str = "short"


# General MIDI percussion map, grouped into our full-kit working voices.
VOICES: dict[str, Voice] = {
    "kick":     Voice("kick",     (35, 36),                    (20, 120),     "short"),
    "snare":    Voice("snare",    (38, 40),                    (120, 400),    "medium"),
    "clap":     Voice("clap",     (39,),                       (800, 3000),   "short"),
    "rim":      Voice("rim",      (37,),                       (400, 1500),   "short"),
    "tom_hi":   Voice("tom_hi",   (48, 50),                    (200, 500),    "medium"),
    "tom_mid":  Voice("tom_mid",  (45, 47),                    (120, 350),    "medium"),
    "tom_lo":   Voice("tom_lo",   (41, 43),                    (80, 250),     "medium"),
    "hihat":    Voice("hihat",    (42, 44),                    (4000, 12000), "short"),
    "open_hat": Voice("open_hat", (46,),                       (3000, 11000), "long"),
    "ride":     Voice("ride",     (51, 53, 59),                (3000, 10000), "long"),
    "crash":    Voice("crash",    (49, 52, 55, 57),            (2000, 16000), "long"),
    "perc":     Voice("perc",     (54, 56, 58, 60, 61, 62, 63, 64, 65, 66,
                                   67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
                                   77, 78, 79, 80, 81),        (300, 8000),   "short"),
}

# Reverse lookup GM note -> voice name.
_NOTE_TO_VOICE: dict[int, str] = {}
for _v in VOICES.values():
    for _n in _v.gm_notes:
        _NOTE_TO_VOICE[_n] = _v.name


def canonical_voice(name: str) -> str:
    """Map a possibly-legacy voice name to a current one (identity if already current)."""
    return VOICE_ALIASES.get(name, name)


def voice_for_note(note: int) -> str | None:
    """Return the voice name for a GM drum note, or None if unmapped."""
    return _NOTE_TO_VOICE.get(int(note))


def gm_notes(voice: str) -> tuple[int, ...]:
    return VOICES[voice].gm_notes


def default_note(voice: str) -> int:
    """A representative GM note to emit for a voice (for synthesized/preview MIDI)."""
    return VOICES[voice].gm_notes[0]
