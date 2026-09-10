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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
.stApp { background:
  radial-gradient(1100px 520px at 12% -10%, #1e3a5f 0%, transparent 60%),
  radial-gradient(900px 520px at 88% -5%, #4c1d95 0%, transparent 55%),
  radial-gradient(760px 620px at 50% 112%, #0f766e 0%, transparent 55%),
  #070d1a fixed;
  font-family: 'Inter', system-ui, sans-serif; }
h1, h2, h3 { letter-spacing: -0.02em; }
.hero-title { font-size: 44px; font-weight: 800; line-height: 1.05;
  text-shadow: 0 0 32px rgba(125, 211, 252, .25); }
.grad { background: linear-gradient(90deg, #7dd3fc, #c084fc, #f0abfc, #7dd3fc);
  background-size: 220% auto; -webkit-background-clip: text;
  background-clip: text; color: transparent;
  animation: shine 6s linear infinite; }
@keyframes shine { to { background-position: 220% center; } }
.hero-sub { color: #a9b8d0; font-size: 15px; margin-bottom: 10px; }
.step-pills { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 18px; }
.step-pill { padding: 6px 14px; border-radius: 999px; font-size: 12px; font-weight: 700;
  letter-spacing: .04em; color: #e2e8f0; background: rgba(30, 41, 59, .75);
  border: 1px solid #334155; backdrop-filter: blur(6px); }
.step-pill.hot { background: linear-gradient(135deg, #0ea5e9, #8b5cf6);
  border-color: transparent; box-shadow: 0 0 18px rgba(139, 92, 246, .45); }
.card { border: 1px solid rgba(148, 163, 184, .16); border-radius: 16px;
  padding: 16px 18px; margin-bottom: 12px;
  background: linear-gradient(180deg, rgba(17, 26, 46, .92), rgba(13, 21, 38, .92));
  box-shadow: 0 8px 28px rgba(0, 0, 0, .35); backdrop-filter: blur(8px);
  transition: transform .15s ease, box-shadow .15s ease; }
.card:hover { transform: translateY(-2px); box-shadow: 0 12px 34px rgba(0, 0, 0, .5); }
.badge { display: inline-block; padding: 3px 12px; border-radius: 999px;
  font-size: 12px; font-weight: 700; letter-spacing: .02em; }
.badge-ok { background: rgba(5, 46, 34, .85); color: #34d399; border: 1px solid #065f46;
  box-shadow: 0 0 12px rgba(52, 211, 153, .25); }
.badge-warn { background: rgba(59, 35, 5, .85); color: #fbbf24; border: 1px solid #92400e;
  box-shadow: 0 0 12px rgba(251, 191, 36, .22); }
.badge-bad { background: rgba(59, 10, 10, .85); color: #f87171; border: 1px solid #7f1d1d;
  box-shadow: 0 0 12px rgba(248, 113, 113, .28); }
.small { color: #94a3b8; font-size: 13px; }
.risk-bar { height: 12px; border-radius: 8px; background: #1a2540;
  overflow: hidden; margin: 8px 0 6px; box-shadow: inset 0 1px 3px rgba(0,0,0,.5); }
.risk-fill { height: 100%; border-radius: 8px; transition: width .6s ease; }
table.dataframe { font-size: 13px; }
/* Sidebar glass */
section[data-testid="stSidebar"] { background: rgba(10, 17, 34, .82);
  backdrop-filter: blur(12px); border-right: 1px solid rgba(148, 163, 184, .12); }
/* Buttons */
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #0ea5e9, #8b5cf6); border: none;
  font-weight: 700; border-radius: 12px; padding: .6rem 1rem;
  box-shadow: 0 4px 22px rgba(139, 92, 246, .45); transition: all .15s ease; }
.stButton > button[kind="primary"]:hover { transform: translateY(-1px);
  box-shadow: 0 8px 30px rgba(139, 92, 246, .6); }
.stButton > button[kind="secondary"] { border-radius: 10px; }
.stDownloadButton > button { border-radius: 10px; }
/* Inputs */
.stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
  border-radius: 10px !important; }
.stTextArea textarea:focus, .stTextInput input:focus {
  border-color: #8b5cf6 !important; box-shadow: 0 0 0 1px #8b5cf6 !important; }
/* Metric tiles */
div[data-testid="stMetric"] { background: linear-gradient(180deg, #111a2e, #0c1426);
  border: 1px solid rgba(148, 163, 184, .16); border-radius: 14px; padding: 12px;
  box-shadow: 0 6px 20px rgba(0, 0, 0, .3); }
div[data-testid="stMetricValue"] { font-weight: 800; }
/* Alerts */
.stAlert { border-radius: 12px; backdrop-filter: blur(6px); }
/* Expanders + tabs */
.streamlit-expanderHeader { border-radius: 10px; font-weight: 600; }
button[data-baseweb="tab"] { font-weight: 600; }
/* Dataframes */
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
/* Images */
img { border-radius: 12px; }
/* Scrollbar */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: #0a1120; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 6px; }
::-webkit-scrollbar-thumb:hover { background: #475569; }
/* Spinners */
.stSpinner > div { border-top-color: #8b5cf6 !important; }
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
