"""Shared Studio library: theme, scorers, data loaders. Imported by app.py pages."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CSS = """
<style>
.stApp { background:
  radial-gradient(1000px 500px at 15% -10%, #1b2a4a 0%, transparent 60%),
  radial-gradient(900px 500px at 90% 0%, #3b1d5e 0%, transparent 55%),
  radial-gradient(700px 600px at 50% 110%, #0e3a3a 0%, transparent 60%),
  #0a0f1e fixed; }
h1, h2, h3 { letter-spacing: -0.02em; }
.hero-title { font-size: 40px; font-weight: 800; line-height: 1.05; }
.grad { background: linear-gradient(90deg, #7dd3fc, #c084fc, #f0abfc);
  -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero-sub { color: #94a3b8; font-size: 15px; margin-bottom: 10px; }
.card { border: 1px solid #263248; border-radius: 14px; padding: 14px 16px;
  margin-bottom: 12px; background: linear-gradient(180deg, #111a2e, #0d1526); }
.badge { display: inline-block; padding: 2px 10px; border-radius: 20px;
  font-size: 12px; font-weight: 700; }
.badge-ok { background: #052e22; color: #34d399; border: 1px solid #065f46; }
.badge-warn { background: #3b2305; color: #fbbf24; border: 1px solid #92400e; }
.badge-bad { background: #3b0a0a; color: #f87171; border: 1px solid #7f1d1d; }
.small { color: #94a3b8; font-size: 13px; }
.risk-bar { height: 10px; border-radius: 6px; background: #1f2b44;
  overflow: hidden; margin: 8px 0 6px; }
.risk-fill { height: 100%; border-radius: 6px; }
table.dataframe { font-size: 13px; }
</style>
"""


def verdict_of(risk: float) -> tuple[str, str]:
    if risk != risk:
        return ("N/A", "badge-warn")
    if risk < 0.4:
        return ("GROUNDED", "badge-ok")
    if risk < 0.6:
        return ("UNCERTAIN", "badge-warn")
    return ("HALLUCINATED?", "badge-bad")


def bar_of(risk: float) -> str:
    if risk != risk:
        return "linear-gradient(90deg,#475569,#64748b)"
    if risk < 0.4:
        return "linear-gradient(90deg,#10b981,#34d399)"
    if risk < 0.6:
        return "linear-gradient(90deg,#f59e0b,#fbbf24)"
    return "linear-gradient(90deg,#ef4444,#f87171)"


def get_clip(checkpoint: str, cos_min: float, cos_max: float, ensemble: bool = True):
    import streamlit as st

    from models.clip_wrapper import CLIPWrapper

    @st.cache_resource(show_spinner="Loading verifier (once, then cached)…")
    def _load(ckpt: str, lo: float, hi: float, ens: bool):
        return CLIPWrapper({"device": "cpu", "dtype": "float32", "seed": 42,
                            "clip": {"backend": "transformers", "checkpoint": ckpt,
                                     "region_grid": 3, "region_overlap": 0.2,
                                     "cos_min": lo, "cos_max": hi,
                                     "templates": (["{}", "a photo of {}",
                                                    "a picture showing {}"]
                                                   if ens else ["{}"])}})

    return _load(checkpoint, cos_min, cos_max, ensemble)


def load_gallery() -> list[dict]:
    p = REPO_ROOT / "assets" / "gallery_results.json"
    return json.loads(p.read_text(encoding="utf-8"))


def load_pope_tables() -> dict:
    out = {}
    for name in ("pope15.json", "pope24.json", "pope30.json"):
        p = REPO_ROOT.parent / "adaptive-vtrace" / "results" / name
        if p.exists():
            out[name] = json.loads(p.read_text(encoding="utf-8"))
    return out
