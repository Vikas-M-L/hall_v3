"""Benchmarks — real POPE numbers, methods compared, limits stated."""
from __future__ import annotations

import streamlit as st

import studio_lib as L

st.set_page_config(page_title="Benchmarks — V-TRACE+ Studio", layout="wide")
st.markdown(L.CSS, unsafe_allow_html=True)
st.warning("Research audit: legacy results below are development-exposed object-phrase "
           "compatibility scores, not a validated end-to-end repair benchmark. "
           "Repeated claims/images inflate the old row counts. See the audited comparison first.")
import json
audit_path = L.REPO_ROOT.parent / "results" / "audited" / "report.json"
if audit_path.exists():
    import pandas as pd
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    st.subheader("Audited matched-cohort comparison")
    st.dataframe(pd.DataFrame([
        {"method": name, **{k: m[k] for k in ("n", "images", "accuracy", "f1", "auroc", "ece", "brier")}}
        for name, m in audit["matched_cohort"]["methods"].items()]))
    st.info("After deduplication, each v3 run correctly decides 10 of 17 claims. "
            "Confidence-only SigLIP selection also gets 10/10 at that coverage. "
            "These records do not establish a pairwise-v3 advantage.")
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

st.subheader("Headline: v3 abstention vs v1 forced decisions (same items, same model)")
import json as _json
import pathlib as _pl

rows_v3 = []
for name in ("pope_v3_s3.json", "pope_v3_s4.json"):
    p = L.REPO_ROOT.parent / "adaptive-vtrace" / "results" / name
    if _pl.Path(p).exists():
        d = _json.loads(_pl.Path(p).read_text(encoding="utf-8"))
        a, b = d["v3-siglip"]["all"], d["siglip-whole"]["all"]
        rows_v3.append({"run": name.replace(".json", ""), "n": a["n"],
                        "v1 forced accuracy": round(b["accuracy"], 3),
                        "v3 coverage": round(a["coverage"], 2),
                        "v3 accuracy on decided": round(a["decisive_acc"], 3),
                        "v3 unresolved": a["n_unresolved"]})
if rows_v3:
    st.dataframe(pd.DataFrame(rows_v3), use_container_width=True)
    st.caption("Historical, non-deduplicated summaries. Comparing selective accuracy "
               "to forced-answer accuracy alone does not establish superiority; "
               "the matched-coverage baseline ties v3 on these development cases.")

st.info("Read honestly: pooled n=67 unique claims — siglip-whole AUROC 0.933/F1 0.905, "
        "clipB32-grid 0.954/0.928 (n=50). Claims are parsed from POPE questions, "
         "not VLM responses. Generated answers, independent gold and complete baseline "
         "runs are still required. Larger CPU evaluation is possible but slower.")
