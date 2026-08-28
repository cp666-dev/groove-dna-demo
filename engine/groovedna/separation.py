"""Cheap, no-ML voice separation: cluster onsets by frequency band + spectral
shape into kick / snare / hihat / tom / cymbal.

This is deliberately the "weekend" version from the brief — good enough to prove
the interpolation logic. The interface (returns Hit objects tagged with a voice)
is identical to what a trained ADT model would produce, so it can be swapped later
without touching the template or transplant code.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Hit:
    time: float          # onset time in seconds
    voice: str
    velocity: float      # 0..1, peak of the transient envelope
    confidence: float    # 0..1, how clean the band assignment was


# Band edges (Hz). Kick low, snare body mid + noise, toms low-mid, hats/cymbals high.
_BANDS = {
    "kick":   (20, 120),
    "tom":    (90, 300),
    "snare":  (150, 2500),
    "hihat":  (5000, 12000),
    "cymbal": (8000, 20000),
}


def _band_energy(S: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any():
        return np.zeros(S.shape[1])
    return S[mask].mean(axis=0)


def separate(y: np.ndarray, sr: int) -> list[Hit]:
    """Detect onsets and assign each to a voice by comparing band energies and
    spectral flatness (noisiness) at the onset frame.
    """
    import librosa

    hop = 512
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    flatness = librosa.feature.spectral_flatness(S=S)[0]  # 0 tonal .. 1 noisy
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]

    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop, backtrack=True, units="frames"
    )
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    env_max = float(env.max()) if env.size else 1.0

    band_energy = {v: _band_energy(S, freqs, lo, hi) for v, (lo, hi) in _BANDS.items()}
    # Normalise each band to itself so a quiet cymbal still registers.
    for v in band_energy:
        m = float(band_energy[v].max()) or 1.0
        band_energy[v] = band_energy[v] / m

    hits: list[Hit] = []
    for f in onset_frames:
        t = librosa.frames_to_time(f, sr=sr, hop_length=hop)
        scores = {v: float(band_energy[v][f]) for v in _BANDS}
        flat = float(flatness[f])
        cen = float(centroid[f])

        # Spectral-shape nudges: snare is noisy+mid, hats are noisy+very-high,
        # kick is tonal+low, toms are tonal+low-mid.
        scores["snare"] *= 1.0 + 1.5 * flat
        scores["hihat"] *= 1.0 + 1.2 * flat
        scores["cymbal"] *= 1.0 + 1.0 * flat
        scores["kick"] *= 1.0 + 1.5 * (1.0 - flat)
        scores["tom"] *= 1.0 + 1.0 * (1.0 - flat)
        if cen < 200:
            scores["kick"] *= 1.4
        if cen > 6000:
            scores["hihat"] *= 1.2
            scores["cymbal"] *= 1.2

        voice = max(scores, key=scores.get)
        ordered = sorted(scores.values(), reverse=True)
        top = ordered[0] or 1e-9
        runner = ordered[1] if len(ordered) > 1 else 0.0
        confidence = float(max(0.0, min(1.0, (top - runner) / top)))
        velocity = float(min(1.0, env[f] / env_max)) if f < len(env) else 0.5
        hits.append(Hit(time=float(t), voice=voice, velocity=velocity,
                        confidence=confidence))
    return hits
