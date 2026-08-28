"""HTTP server bridging the web app to the engine: POST an audio file, get back
a fingerprint JSON (the same payload `viz.to_fingerprint_json` produces).

The browser can't run librosa, so audio ingestion lives here. Run it with:
    groovedna serve
and the app's file-drop / "Load audio" button will POST to it.
"""
from __future__ import annotations

import os
import tempfile
from typing import Optional

from pydantic import BaseModel


# Request models live at module scope so FastAPI can resolve the endpoint
# annotations (deferred by `from __future__ import annotations`).
class ComposeReq(BaseModel):
    prompt: str
    bpm: Optional[float] = None
    bars: Optional[int] = None


class GrooveReq(BaseModel):
    fingerprint: dict


class EditReq(BaseModel):
    fingerprint: dict
    instruction: str


class VaryReq(BaseModel):
    fingerprint: dict
    kind: str = "variation"


def create_app():
    from fastapi import FastAPI, UploadFile, File, Form, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="Groove DNA engine", version="0.1.0")
    # Dev-friendly CORS so the Vite app (any localhost port) can call us.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _allow_private_network(request, call_next):
        # Let an https page (e.g. the GitHub Pages demo) reach this local server
        # under Chrome's Private Network Access rules — the browser sends a
        # preflight and requires this header back to permit public -> localhost.
        resp = await call_next(request)
        resp.headers["Access-Control-Allow-Private-Network"] = "true"
        return resp

    @app.get("/health")
    def health():
        return {"ok": True}

    def _with_ai_notes(g: dict, fp: dict) -> dict:
        """Attach Claude's plain-English reasoning + suggested drum-sample kit
        (both live on the raw groove dict, not the flattened fingerprint)."""
        if g.get("reasoning"):
            fp["notes"] = g["reasoning"]
        if g.get("kit"):
            fp["kit"] = g["kit"]
        return fp

    @app.post("/compose")
    def compose(req: ComposeReq):
        """Generate a drum pattern + feel from a text prompt (Claude) -> fingerprint."""
        from .compose import generate, groove_to_template
        from .viz import to_fingerprint_json
        try:
            g = generate(req.prompt, bpm=req.bpm, bars=req.bars)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        return _with_ai_notes(g, to_fingerprint_json(groove_to_template(g)))

    @app.post("/coach")
    def coach_ep(req: GrooveReq):
        """Critique the current groove's feel + return a humanized 'fixed' fingerprint."""
        from .ai import coach
        from .viz import from_fingerprint, to_fingerprint_json
        from .compose import groove_to_template
        try:
            c = coach(from_fingerprint(req.fingerprint))
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        c["fixed"] = to_fingerprint_json(groove_to_template(c["fixed"]))
        return c

    @app.post("/edit")
    def edit_ep(req: EditReq):
        """Edit the current groove per a natural-language instruction -> fingerprint."""
        from .ai import edit
        from .viz import from_fingerprint, to_fingerprint_json
        from .compose import groove_to_template
        try:
            g = edit(from_fingerprint(req.fingerprint), req.instruction)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        return _with_ai_notes(g, to_fingerprint_json(groove_to_template(g)))

    @app.post("/variations")
    def variations_ep(req: VaryReq):
        """Generate an in-style variation/fill of the current groove -> fingerprint."""
        from .ai import variation
        from .viz import from_fingerprint, to_fingerprint_json
        from .compose import groove_to_template
        try:
            g = variation(from_fingerprint(req.fingerprint), kind=req.kind)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        return _with_ai_notes(g, to_fingerprint_json(groove_to_template(g)))

    @app.post("/extract")
    async def extract(
        audio: UploadFile = File(...),
        # Default: treat the upload as an already-isolated drum stem/loop, so the
        # common case (drop a drum loop) needs no stem-sep and no API key.
        isolated: bool = Form(True),
        backend: str = Form("elevenlabs"),
        subdivision: int = Form(4),
        bpm: Optional[float] = Form(None),
    ):
        import librosa
        from .stemsep import get_drum_stem, StemSepError
        from .separation import separate
        from .grid import track_grid, Grid
        from .template import Template
        from .viz import to_fingerprint_json

        name = os.path.splitext(audio.filename or "extracted")[0]
        suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await audio.read())
            path = tmp.name

        try:
            try:
                stem = get_drum_stem(path, prefer=backend, already_isolated=isolated)
            except StemSepError as e:
                raise HTTPException(status_code=422, detail=f"stem separation failed: {e}")

            y, sr = librosa.load(stem, sr=None, mono=True)
            grid = track_grid(y, sr, subdivision=subdivision)
            if bpm:
                grid = Grid(bpm=bpm, subdivision=subdivision, origin=grid.origin)
            hits = separate(y, sr)
            if not hits:
                raise HTTPException(status_code=422, detail="no drum onsets detected")
            tmpl = Template.from_hits(hits, grid, name=name)
            tmpl.interpolate()
            return to_fingerprint_json(tmpl)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    return app


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port)
