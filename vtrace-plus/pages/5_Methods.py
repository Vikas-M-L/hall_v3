"""Methods — how it works, what it costs, where it stops."""
from __future__ import annotations

import streamlit as st

import studio_lib as L

st.set_page_config(page_title="Methods — V-TRACE+ Studio", layout="wide")
st.markdown(L.CSS, unsafe_allow_html=True)
st.warning("Historical methods overview. The research audit supersedes accuracy, "
           "novelty and cost claims below. Diagnosis labels are hypotheses; "
           "independent repair validation is still required.")
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
st.markdown("The prototype can select uncertain claims for API cross-checking. "
            "The earlier 4–13% cost figure used assumed synthetic costs and full-stack "
            "outputs; it is not evidence of real savings. Research telemetry now records "
            "actual API stages separately from replay and local scoring.")

st.subheader("Honest limits")
st.markdown("* Binding ('gold cat'), counting without the detector, OCR, tiny objects\n"
            "* Absolute scores need window calibration — read the ranking\n"
            "* Thresholds provisional (n≈33); n=9000 needs a GPU\n"
            "* No SDXL/LoRA/fine-tuning anywhere — inference-time detector by design")

st.subheader("Research status")
st.markdown("The novelty claim is unestablished. FaithScore, Woodpecker and VISOR "
            "overlap the proposed contributions. The research question is whether "
            "error-specific routing improves answer preservation at matched cost. "
            "See literature_review.md, research_audit.md and FINAL_RESEARCH_ROADMAP.md.")
