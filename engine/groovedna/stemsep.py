"""Pull an isolated drum stem out of a full mix.

Priority per the project decision:
  1. ElevenLabs stem-separation API (POST /v1/music/stem-separation) -- default,
     uses the ELEVENLABS_API_KEY the user already has.
  2. Demucs v4 (htdemucs) local -- free/offline fallback if torch+demucs installed.
  3. Bring-your-own-stem -- if the input already is an isolated drum stem, skip.

All paths return a path to a mono/stereo wav containing (mostly) drums.
"""
from __future__ import annotations

import io
import os
import zipfile
import tempfile


class StemSepError(RuntimeError):
    pass


def elevenlabs_drums(mix_path: str, out_dir: str | None = None,
                     api_key: str | None = None) -> str:
    """Separate `mix_path` via ElevenLabs and return the path to the drums stem."""
    import requests

    api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise StemSepError("ELEVENLABS_API_KEY not set")
    out_dir = out_dir or tempfile.mkdtemp(prefix="groovedna_")

    url = "https://api.elevenlabs.io/v1/music/stem-separation"
    with open(mix_path, "rb") as fh:
        resp = requests.post(
            url,
            headers={"xi-api-key": api_key},
            files={"audio": (os.path.basename(mix_path), fh, "audio/wav")},
            timeout=600,
        )
    if resp.status_code != 200:
        raise StemSepError(f"ElevenLabs {resp.status_code}: {resp.text[:300]}")

    # Response is a ZIP of stems; find the one that looks like drums.
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    names = zf.namelist()
    drum = next((n for n in names if "drum" in n.lower()), None)
    if drum is None:
        raise StemSepError(f"no drums stem in response (got {names})")
    out = os.path.join(out_dir, "drums.wav")
    with open(out, "wb") as f:
        f.write(zf.read(drum))
    return out


def demucs_drums(mix_path: str, out_dir: str | None = None) -> str:
    """Separate via a local Demucs v4 install and return the drums stem path."""
    import subprocess
    import glob

    out_dir = out_dir or tempfile.mkdtemp(prefix="groovedna_")
    cmd = ["python", "-m", "demucs", "-n", "htdemucs", "--two-stems", "drums",
           "-o", out_dir, mix_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise StemSepError(f"demucs failed: {e}") from e
    hits = glob.glob(os.path.join(out_dir, "**", "drums.*"), recursive=True)
    if not hits:
        raise StemSepError("demucs produced no drums stem")
    return hits[0]


def get_drum_stem(path: str, prefer: str = "elevenlabs",
                  already_isolated: bool = False) -> str:
    """Resolve `path` to an isolated drum stem, honouring the preferred backend
    and falling back gracefully.
    """
    if already_isolated:
        return path
    order = [prefer] + [b for b in ("elevenlabs", "demucs") if b != prefer]
    last: Exception | None = None
    for backend in order:
        try:
            if backend == "elevenlabs":
                return elevenlabs_drums(path)
            if backend == "demucs":
                return demucs_drums(path)
        except StemSepError as e:
            last = e
    raise StemSepError(
        f"all stem-separation backends failed; last error: {last}. "
        "Pass --isolated if the file is already a drum stem.")
