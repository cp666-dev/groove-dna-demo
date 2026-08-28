# Groove DNA — engine

The analysis + transplant core. Extracts a `.groovedna` feel template from reference
drum audio and applies it to your MIDI, per voice, tempo-relative, with interpolation.

## Install
```bash
cd engine
python -m venv .venv && source .venv/bin/activate
pip install -e ".[viz]"          # add ",dev" for tests, ",demucs" for local stem-sep
```

## CLI
```bash
# 1) Extract feel from a reference (full mix -> stem-sep -> template)
groovedna extract reference.wav -o feel.groovedna --name "Purdie shuffle"
#    already have an isolated drum stem? skip separation:
groovedna extract drums_stem.wav -o feel.groovedna --isolated
#    force the local free backend:
groovedna extract reference.wav -o feel.groovedna --backend demucs

# 2) Transplant onto your MIDI (blend 0..1; tempo taken from the MIDI unless --bpm)
groovedna apply my_drums.mid feel.groovedna -o out.mid --amount 0.8

# 3) See the fingerprint (PNG + the JSON the 3D app consumes)
groovedna viz feel.groovedna -o fingerprint.png --json fingerprint.json

# 4) Bake it into a Logic Scripter MIDI-FX you can paste into Logic
groovedna scripter feel.groovedna -o GrooveDNA.js

# 5) Run the HTTP server the web app calls when you drop an mp3/wav
pip install -e ".[serve]"
groovedna serve                      # POST /extract  (audio -> fingerprint JSON)
```

### Web app audio ingestion
The browser can't run librosa, so the 3D app (`../app`) sends dropped audio to this
server's `POST /extract` and renders the returned fingerprint. Start `groovedna serve`
before dropping an mp3/wav in the app. `/extract` treats the upload as an already-isolated
drum loop by default (no API key needed); pass `isolated=false` to run stem separation first.

## Stem separation
`extract` resolves a drum stem via, in order: **ElevenLabs** (`ELEVENLABS_API_KEY`,
`POST /v1/music/stem-separation`) → **Demucs v4** (local, `pip install demucs`) →
error telling you to pass `--isolated`. Choose with `--backend`.

Set your key (see `.env.example`):
```bash
export ELEVENLABS_API_KEY=sk-...
```

## Pipeline (what `extract` does)
1. `stemsep.get_drum_stem` — isolate drums (or pass through an isolated stem).
2. `separation.separate` — onset detection + frequency-band/spectral-shape clustering
   into kick/snare/tom/hihat/cymbal (the cheap, no-ML "weekend" version).
3. `grid.track_grid` — librosa beat tracking → tempo, downbeat, subdivision grid.
4. `template.Template.from_hits` — aggregate each (voice, slot) into a distribution
   of timing-offset% and velocity.
5. `template.interpolate` — the novel step: fill never-played slots by
   proximity-weighted blend of observed neighbours (circular within the bar).

`apply` (`transplant.apply_template`) resolves each MIDI note's voice+slot, samples
the distribution, nudges timing (scaled to the *target* tempo) and velocity, and
blends against the original by `--amount`.

## Smoke test / demo
No audio needed — simulates a laid-back shuffle and proves interpolation + transplant:
```bash
PYTHONPATH=. python scripts/make_demo.py       # writes examples/*.json + .groovedna
pytest                                          # same, as an assertion suite
```

See [`../SCHEMA.md`](../SCHEMA.md) for the on-disk format.
