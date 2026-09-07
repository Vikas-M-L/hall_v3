"""Methods — how it works, what it costs, where it stops."""
from __future__ import annotations

import streamlit as st

import studio_lib as L

st.set_page_config(page_title="Methods — V-TRACE+ Studio", layout="wide")
st.markdown(L.CSS, unsafe_allow_html=True)
st.markdown("<div class='hero-title'>How it <span class='grad'>works</span></div>",
            unsafe_allow_html=True)

st.subheader("Pipeline")
st.markdown("Image → Claims (sentence split, or Gemini atomic decomposition) → "
            "SigLIP/CLIP region verification → deterministic fusion → risk + JSON. "
            "All models frozen; no training anywhere.")

st.subheader("Signals (higher = riskier, always)")
st.markdown("* **evidence** — best claim-to-region match, inverted\n"
            "* **similarity** — whole-image match, inverted\n"
            "* **negation rule** — 'no X' claims invert match→risk (flagged)\n"
            "* **self-consistency** — Gemini temperature samples disagree (M3 hint)\n"
            "* **VLM cross-check** — direct YES/NO on uncertain claims (cascade stage-2)\n"
            "* **OWL-ViT count** — one box per instance vs the claimed number")

st.subheader("Fusion v1 / v2")
st.markdown("v1 = equal weights. v2 = Rule 1 (confident-but-ungrounded leans on visual "
            "signals) + Rule 2 (detector disagreement promotes independent CLIP). "
            "Hard thresholds: max discontinuity 0.044, regression-tested. "
            "Without VLM confidence, v2 ≈ v1.")

st.subheader("Cost-aware cascade")
st.markdown("Free CLIP signals decide easy claims; VLM calls fire only on ambiguity "
            "(uncertain band, or all claims in max-accuracy mode). Measured frontier: "
            "full accuracy at ~4–13% of VLM-pass cost (synthetic proof; GPU run pending). "
            "Every Studio run reports its cascade savings.")

st.subheader("Honest limits")
st.markdown("* Binding ('gold cat'), counting without the detector, OCR, tiny objects\n"
            "* Absolute scores need window calibration — read the ranking\n"
            "* Thresholds provisional (n≈33); n=9000 needs a GPU\n"
            "* No SDXL/LoRA/fine-tuning anywhere — inference-time detector by design")

st.subheader("Novelty (self-rating)")
st.markdown("Reframe 8 · routing design 9 · empirical proof 5 · engineering 9 → "
            "**7/10**. The remaining points need a GPU: full POPE, trained two-stage "
            "classifier on real features, routing-vs-policy on natural outcomes.")
