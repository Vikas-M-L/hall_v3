"""Benchmarks — real POPE numbers, methods compared, limits stated."""
from __future__ import annotations

import streamlit as st

import studio_lib as L

st.set_page_config(page_title="Benchmarks — V-TRACE+ Studio", layout="wide")
st.markdown(L.CSS, unsafe_allow_html=True)
st.markdown("<div class='hero-title'>Real <span class='grad'>Benchmarks</span></div>",
            unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>COCO images + POPE labels, three splits, "
            "AUROC / AUPRC / F1 vs baselines. Small-n and honestly labeled."
            "</div>", unsafe_allow_html=True)

tables = L.load_pope_tables()
if not tables:
    st.warning("No benchmark results found yet.")
    st.stop()

import pandas as pd

for name, t in tables.items():
    methods = [m for m in t if not m.startswith("_")]
    meta = t.get("_meta", {})
    st.subheader(f"{name} — n={meta.get('n_per_split', '?')}/split, "
                 f"{meta.get('seconds', '?')}s on CPU")
    df = pd.DataFrame([{"method": m, **{s: round(t[m][s]['auroc'], 3)
                                        for s in ("random", "popular", "adversarial", "all")}}
                       for m in methods])
    st.dataframe(df, use_container_width=True)
    df2 = pd.DataFrame([{"method": m, "F1": round(t[m]["all"]['f1'], 3),
                         "acc": round(t[m]["all"]['accuracy'], 3),
                         "auroc_ci95": str(t[m]["all"].get("auroc_ci95", ""))}
                        for m in methods])
    st.dataframe(df2, use_container_width=True)
    st.caption(meta.get("note", ""))

st.info("Read honestly: pooled n=67 unique claims — siglip-whole AUROC 0.933/F1 0.905, "
        "clipB32-grid 0.954/0.928 (n=50). Claims are parsed from POPE questions, "
        "not VLM responses — a Gemini key closes that gap. Full 9000-item run needs a GPU.")
