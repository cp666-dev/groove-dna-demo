"""MIDI read/write via pretty_midi, reduced to the fields the transplant needs."""
from __future__ import annotations

from dataclasses import dataclass

from .voices import voice_for_note, default_note


@dataclass
class MidiNote:
    start: float
    end: float
    pitch: int
    velocity: int      # 1..127
    voice: str | None


def load_drum_notes(path: str):
    """Return (notes, bpm). Reads drum tracks (is_drum) and tags each note's voice."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(path)
    tempi_times, tempi = pm.get_tempo_changes()
    bpm = float(tempi[0]) if len(tempi) else 120.0

    notes: list[MidiNote] = []
    for inst in pm.instruments:
        if not inst.is_drum:
            continue
        for n in inst.notes:
            notes.append(MidiNote(start=n.start, end=n.end, pitch=n.pitch,
                                  velocity=n.velocity, voice=voice_for_note(n.pitch)))
    notes.sort(key=lambda n: n.start)
    return notes, bpm


def save_drum_notes(notes: list[MidiNote], bpm: float, path: str) -> None:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    drum = pretty_midi.Instrument(program=0, is_drum=True, name="Groove DNA")
    for n in notes:
        pitch = n.pitch if n.pitch else (default_note(n.voice) if n.voice else 38)
        drum.notes.append(pretty_midi.Note(
            velocity=int(max(1, min(127, n.velocity))),
            pitch=int(pitch), start=max(0.0, n.start), end=max(n.start + 1e-3, n.end)))
    pm.instruments.append(drum)
    pm.write(path)
