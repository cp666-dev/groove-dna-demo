"""Groove DNA — extract and transplant the micro-timing + velocity feel of a
real drum performance onto programmed MIDI, per voice, tempo-relative, with
interpolation to positions the reference never played.
"""

from .voices import VOICE_ORDER as VOICES, voice_for_note, default_note  # noqa: F401

__version__ = "0.1.0"
