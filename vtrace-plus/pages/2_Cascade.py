"""Cascade — how much expensive verification we avoided. Measured only."""
from __future__ import annotations

import json

import streamlit as st

import studio_lib as L

st.set_page_config(page_title="Cascade — V-TRACE+ Studio", layout="wide")
st.markdown(L.CSS, unsafe_allow_html=True)
st.markdown("<div class='hero-title'>Verification <span class='grad'>Cascade</span></div>",
            unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>Free CLIP signals decide easy claims; expensive VLM "
            "calls fire only on ambiguity. Every number below is measured on real "
            "POPE items — never claimed without data.</div>", unsafe_allow_html=True)

pool: dict = {}
import pathlib as _p

for name in ("pope24.json", "pope30.json", "pope31.json"):
    p = L.REPO_ROOT.parent / "adaptive-vtrace" / "results" / name
    if _p.Path(p).exists():
        for c in json.loads(_p.Path(p).read_text(encoding="utf-8")).get("_claims", []):
            pool.setdefault((c["image"], c["claim"]), c)
rows = [v for v in pool.values() if "siglip-whole" in v]
st.caption(f"Measured on {len(rows)} unique POPE claims (deduped across splits).")

import pandas as _pd

out = []
for tau in (0.3, 0.5, 0.6, 0.8):
    dec = [abs(r["siglip-whole"] - 0.5) * 2 >= tau for r in rows]
    out.append({"decisiveness threshold": tau,
                "resolved by CLIP": f"{sum(dec)}/{len(rows)}",
                "VLM calls avoided": f"{100 * sum(dec) / len(rows):.0f}%"})
st.dataframe(_pd.DataFrame(out), use_container_width=True)
st.info("Target (unclaimed until the GPU run): ≥95% of full-system detection quality "
        "while using ≤15% of expensive VLM calls. The Studio already reports live "
        "per-run savings under every verdict banner.")
