"""Scenario Gallery — every hallucination type, precomputed, with corrections."""
from __future__ import annotations

import streamlit as st

import studio_lib as L

st.set_page_config(page_title="Gallery — V-TRACE+ Studio", layout="wide")
st.markdown(L.CSS, unsafe_allow_html=True)
st.markdown("<div class='hero-title'>Scenario <span class='grad'>Gallery</span></div>",
            unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>Seven hallucination types beyond the benchmark. "
            "Each card: true claim vs false claim, system ranking, and the correction. "
            "Precomputed with SigLIP (no API spend) — re-run any of them live in the Checker."
            "</div>", unsafe_allow_html=True)

gallery = L.load_gallery()
n_pair = sum(1 for sc in gallery if sc.get("pairwise_ok"))
st.success(f"Pairwise ranking: **{n_pair}/{len(gallery)}** correct "
           "(the claimed capability — read the ranking, not the absolutes)")

for sc in gallery:
    with st.expander(f"{'CORRECT ORDER' if sc.get('pairwise_ok') else 'WRONG ORDER'} — "
                     f"{sc['title']}: {sc['blurb']}", expanded=False):
        c1, c2 = st.columns([5, 7])
        with c1:
            st.image(str(L.REPO_ROOT / "assets" / sc["image"]), use_container_width=True)
        with c2:
            for r in sc["results"]:
                vtext, vcls = L.verdict_of(r["risk"])
                pct = 0 if r["risk"] != r["risk"] else int(r["risk"] * 100)
                ok = "CORRECT" if r["correct"] else "OFF"
                okcls = "badge-ok" if r["correct"] else "badge-bad"
                st.markdown(
                    f"<div class='card'><b>{r['text']}</b><br>"
                    f"<span class='badge {vcls}'>{vtext} {r['risk']:.3f}</span> "
                    f"<span class='badge {okcls}'>{ok} (expected {r['expected']})</span>"
                    f"<div class='risk-bar'><div class='risk-fill' "
                    f"style='width:{pct}%;background:{L.bar_of(r['risk'])}'></div></div>"
                    + (f"<span class='small'>Correction: {r['correction']}</span><br>"
                       if r.get("correction") and not r["correct"] else "")
                    + (f"<span class='small'>Rules: {', '.join(r['rules'])}</span>"
                       if r.get("rules") else "")
                    + "</div>",
                    unsafe_allow_html=True,
                )
st.caption("Threshold labels are provisional (n=14) — the ranking is the result. "
           "Count cards additionally need the live OWL-ViT check in the Checker.")
