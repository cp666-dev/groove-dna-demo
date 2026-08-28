"""Drum voice taxonomy: General MIDI note maps + spectral band profiles.

Two jobs live here:
  * MIDI side  -> map a GM drum note number to a voice (kick/snare/hihat/tom/cymbal).
  * Audio side -> a coarse frequency-band profile per voice, used by the cheap
    (no-ML) onset clustering in ``separation.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

# Canonical voice order (also the render order in the fingerprint UI, low -> high).
VOICE_ORDER = ["kick", "snare", "tom", "hihat", "cymbal"]


@dataclass(frozen=True)
class Voice:
    name: str
    gm_notes: tuple[int, ...]
    # Coarse band the voice's energy concentrates in, in Hz (lo, hi).
    band_hz: tuple[float, float]
    # Typical transient length: cymbals ring, kicks are short. Purely a hint.
    ring: str = "short"


# General MIDI percussion map, grouped into our 5 working voices.
VOICES: dict[str, Voice] = {
    "kick":   Voice("kick",   (35, 36),                     (20, 120),     "short"),
    "snare":  Voice("snare",  (37, 38, 39, 40),             (120, 400),    "medium"),
    "tom":    Voice("tom",    (41, 43, 45, 47, 48, 50),     (80, 350),     "medium"),
    "hihat":  Voice("hihat",  (42, 44, 46),                 (3000, 12000), "short"),
    "cymbal": Voice("cymbal", (49, 51, 52, 53, 55, 57, 59), (2000, 16000), "long"),
}

# Reverse lookup GM note -> voice name.
_NOTE_TO_VOICE: dict[int, str] = {}
for _v in VOICES.values():
    for _n in _v.gm_notes:
        _NOTE_TO_VOICE[_n] = _v.name


def voice_for_note(note: int) -> str | None:
    """Return the voice name for a GM drum note, or None if unmapped."""
    return _NOTE_TO_VOICE.get(int(note))


def gm_notes(voice: str) -> tuple[int, ...]:
    return VOICES[voice].gm_notes


def default_note(voice: str) -> int:
    """A representative GM note to emit for a voice (for synthesized/preview MIDI)."""
    return VOICES[voice].gm_notes[0]
