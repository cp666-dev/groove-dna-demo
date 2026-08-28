"""Bake a `.groovedna` template into a ready-to-paste Logic Scripter MIDI-FX.

`logic/GrooveDNA.js` ships with the demo template inline and a clearly-marked
`var TEMPLATE = {...};` block. This rewrites that block with a real template so a
producer can drop the result straight into Logic's Scripter and hit Run.
"""
from __future__ import annotations

import json
import os
import re

# Repo-relative path to the canonical Scripter (engine/groovedna/ -> repo root).
_DEFAULT_JS = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "logic", "GrooveDNA.js")
)

_TEMPLATE_RE = re.compile(r"var TEMPLATE = \{[\s\S]*?\n\};")


def export_scripter(groovedna_path: str, out_path: str,
                    base_js: str | None = None) -> str:
    """Read a .groovedna file, inline it into GrooveDNA.js, write to `out_path`."""
    base_js = base_js or _DEFAULT_JS
    with open(base_js) as f:
        src = f.read()
    with open(groovedna_path) as f:
        template = json.load(f)

    literal = "var TEMPLATE = " + json.dumps(template, indent=2) + ";"
    if not _TEMPLATE_RE.search(src):
        raise ValueError(
            "could not find the 'var TEMPLATE = {...};' block in the base Scripter")
    src = _TEMPLATE_RE.sub(lambda _m: literal, src, count=1)

    with open(out_path, "w") as f:
        f.write(src)
    return out_path
