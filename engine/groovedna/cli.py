"""Groove DNA command-line interface.

    groovedna extract  REF_AUDIO  -o feel.groovedna [--isolated] [--bpm N]
    groovedna apply    TARGET.mid feel.groovedna -o out.mid [--amount 0.8] [--bpm N]
    groovedna viz      feel.groovedna -o fingerprint.png [--json fingerprint.json]

`extract` runs stem-sep (if needed) -> onset separation -> grid tracking ->
template build -> interpolation. `apply` transplants onto your MIDI with a blend.
"""
from __future__ import annotations

import argparse
import sys


def _extract(args) -> int:
    import librosa
    from .stemsep import get_drum_stem, StemSepError
    from .separation import separate
    from .grid import track_grid, Grid
    from .template import Template

    try:
        stem = get_drum_stem(args.ref, prefer=args.backend,
                             already_isolated=args.isolated)
    except StemSepError as e:
        print(f"stem separation failed: {e}", file=sys.stderr)
        return 2

    y, sr = librosa.load(stem, sr=None, mono=True)
    grid = track_grid(y, sr, subdivision=args.subdivision)
    if args.bpm:
        grid = Grid(bpm=args.bpm, subdivision=args.subdivision, origin=grid.origin)
    hits = separate(y, sr)
    tmpl = Template.from_hits(hits, grid, name=args.name or "extracted",
                              min_confidence=args.min_confidence)
    tmpl.interpolate(decay=args.decay)
    tmpl.save(args.out)
    obs = sum(1 for v in tmpl.voices.values() for c in v.values() if c.observed)
    print(f"extracted {len(hits)} hits @ {grid.bpm} BPM -> {obs} observed cells; "
          f"wrote {args.out}")
    return 0


def _apply(args) -> int:
    from .midi_io import load_drum_notes, save_drum_notes
    from .template import Template

    notes, src_bpm = load_drum_notes(args.target)
    bpm = args.bpm or src_bpm
    tmpl = Template.load(args.template)
    from .transplant import apply_template
    out = apply_template(notes, bpm, tmpl, amount=args.amount, seed=args.seed,
                         humanize=args.humanize)
    save_drum_notes(out, bpm, args.out)
    print(f"transplanted {len(notes)} notes @ {bpm} BPM "
          f"(amount={args.amount}) -> {args.out}")
    return 0


def _viz(args) -> int:
    from .template import Template
    from .viz import save_png, save_fingerprint_json

    tmpl = Template.load(args.template)
    save_png(tmpl, args.out)
    print(f"wrote {args.out}")
    if args.json:
        save_fingerprint_json(tmpl, args.json)
        print(f"wrote {args.json}")
    return 0


def _compose(args) -> int:
    from .compose import generate, groove_to_template, groove_to_notes
    from .viz import save_fingerprint_json
    from .midi_io import save_drum_notes

    try:
        g = generate(args.prompt, bpm=args.bpm, bars=args.bars)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2

    tmpl = groove_to_template(g)
    notes = groove_to_notes(g)
    base = args.out
    tmpl.save(base + ".groovedna")
    save_fingerprint_json(tmpl, base + ".fingerprint.json")
    save_drum_notes(notes, tmpl.grid.bpm, base + ".mid")
    n_hits = len(g.get("hits", []))
    print(f'generated "{g.get("name", "groove")}" @ {tmpl.grid.bpm} BPM — '
          f"{n_hits} hits -> {base}.groovedna / .mid / .fingerprint.json")
    return 0


def _write_groove(g: dict, base: str) -> int:
    from .compose import groove_to_template, groove_to_notes
    from .viz import save_fingerprint_json
    from .midi_io import save_drum_notes
    tmpl = groove_to_template(g)
    tmpl.save(base + ".groovedna")
    save_fingerprint_json(tmpl, base + ".fingerprint.json")
    save_drum_notes(groove_to_notes(g), tmpl.grid.bpm, base + ".mid")
    print(f'wrote {base}.groovedna / .mid / .fingerprint.json')
    return 0


def _coach(args) -> int:
    from .template import Template
    from .ai import coach
    from .viz import save_fingerprint_json
    from .compose import groove_to_template
    try:
        c = coach(Template.load(args.template))
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2
    print(f"humanness {c['humanness']}/100 — {c['verdict']}\n")
    for f in c["findings"]:
        print(f"  • [{f['voice']} {f['where']}] {f['issue']}\n    fix: {f['fix']}")
    if args.out:
        tmpl = groove_to_template(c["fixed"])
        tmpl.save(args.out + ".groovedna")
        save_fingerprint_json(tmpl, args.out + ".fingerprint.json")
        print(f"\nhumanized groove -> {args.out}.groovedna / .fingerprint.json")
    return 0


def _edit(args) -> int:
    from .template import Template
    from .ai import edit
    try:
        g = edit(Template.load(args.template), args.instruction)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2
    return _write_groove(g, args.out)


def _variations(args) -> int:
    from .template import Template
    from .ai import variation
    try:
        g = variation(Template.load(args.template), kind=args.kind)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2
    return _write_groove(g, args.out)


def _blend(args) -> int:
    from .template import Template
    from .style import blend
    from .viz import save_fingerprint_json
    a = Template.load(args.a)
    b = Template.load(args.b)
    out = blend(a, b, args.weight)
    out.save(args.out + ".groovedna")
    save_fingerprint_json(out, args.out + ".fingerprint.json")
    print(f'blended "{a.name}" x "{b.name}" @ {args.weight:.0%} -> '
          f"{args.out}.groovedna / .fingerprint.json")
    return 0


def _serve(args) -> int:
    from .server import serve

    print(f"Groove DNA engine on http://{args.host}:{args.port}  (POST /extract)")
    serve(host=args.host, port=args.port)
    return 0


def _scripter(args) -> int:
    from .scripter import export_scripter

    out = export_scripter(args.template, args.out)
    print(f"baked {args.template} into Logic Scripter -> {out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="groovedna", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="reference audio -> .groovedna template")
    e.add_argument("ref")
    e.add_argument("-o", "--out", required=True)
    e.add_argument("--isolated", action="store_true",
                   help="input is already an isolated drum stem")
    e.add_argument("--backend", default="elevenlabs",
                   choices=["elevenlabs", "demucs"])
    e.add_argument("--bpm", type=float, default=None)
    e.add_argument("--subdivision", type=int, default=4)
    e.add_argument("--decay", type=float, default=1.5)
    e.add_argument("--min-confidence", type=float, default=0.0)
    e.add_argument("--name", default=None)
    e.set_defaults(func=_extract)

    a = sub.add_parser("apply", help="transplant a template onto MIDI")
    a.add_argument("target")
    a.add_argument("template")
    a.add_argument("-o", "--out", required=True)
    a.add_argument("--amount", type=float, default=1.0, help="blend 0..1")
    a.add_argument("--humanize", type=float, default=1.0, help="std spread 0..1")
    a.add_argument("--bpm", type=float, default=None)
    a.add_argument("--seed", type=int, default=0)
    a.set_defaults(func=_apply)

    v = sub.add_parser("viz", help="render the groove fingerprint")
    v.add_argument("template")
    v.add_argument("-o", "--out", required=True)
    v.add_argument("--json", default=None, help="also write frontend JSON")
    v.set_defaults(func=_viz)

    s = sub.add_parser("scripter", help="bake a template into a Logic Scripter JS")
    s.add_argument("template")
    s.add_argument("-o", "--out", required=True)
    s.set_defaults(func=_scripter)

    c = sub.add_parser("compose", help="generate a drum pattern + feel from a text prompt (Claude)")
    c.add_argument("prompt")
    c.add_argument("-o", "--out", required=True, help="output basename (writes .groovedna/.mid/.fingerprint.json)")
    c.add_argument("--bpm", type=float, default=None)
    c.add_argument("--bars", type=int, default=None)
    c.set_defaults(func=_compose)

    co = sub.add_parser("coach", help="critique a groove's feel + write a humanized fix (Claude)")
    co.add_argument("template")
    co.add_argument("-o", "--out", default=None, help="basename for the humanized fix")
    co.set_defaults(func=_coach)

    ed = sub.add_parser("edit", help="edit a groove by a natural-language instruction (Claude)")
    ed.add_argument("template")
    ed.add_argument("instruction")
    ed.add_argument("-o", "--out", required=True)
    ed.set_defaults(func=_edit)

    vr = sub.add_parser("variations", help="generate an in-style variation or fill (Claude)")
    vr.add_argument("template")
    vr.add_argument("-o", "--out", required=True)
    vr.add_argument("--kind", default="variation", help="e.g. variation, fill")
    vr.set_defaults(func=_variations)

    bl = sub.add_parser("blend", help="morph groove A's feel toward B (drummer blend v1)")
    bl.add_argument("a")
    bl.add_argument("b")
    bl.add_argument("-o", "--out", required=True)
    bl.add_argument("-w", "--weight", type=float, default=0.5, help="0=A, 1=A played like B")
    bl.set_defaults(func=_blend)

    sv = sub.add_parser("serve", help="run the audio->fingerprint server for the web app")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=_serve)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
