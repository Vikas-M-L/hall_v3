"""V-TRACE+ Studio: persistent, evidence-aware claim analysis and verified candidates."""
from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

import studio_lib as L
from fusion.decision import PRESENTATION, json_safe

st.set_page_config(page_title="V-TRACE+ Studio", page_icon="V", layout="wide")
st.markdown(L.CSS, unsafe_allow_html=True)
st.title("V-TRACE+ Studio")
st.caption("Extract facts · gather evidence · identify conflicts · propose and recheck corrections")


@st.cache_resource(show_spinner="Loading image verifier…", max_entries=2)
def get_clip(model, lo, hi, ensemble):
    from models.clip_wrapper import CLIPWrapper
    return CLIPWrapper({"device": "cpu", "dtype": "float32", "seed": 42,
                        "clip": {"backend": "transformers", "checkpoint": model,
                                 "cos_min": lo, "cos_max": hi, "region_grid": 3,
                                 "region_overlap": .2, "templates": ["{}", "a photo of {}", "a picture showing {}"] if ensemble else ["{}"]}})


@st.cache_resource(show_spinner="Loading object detector…")
def get_owl():
    from models.owl_wrapper import OWLDetector
    return OWLDetector()


def gemini_client():
    from models.gemini_wrapper import GeminiWrapper
    return GeminiWrapper()


try:
    gemini_client()
    key_available = True
except Exception:
    key_available = False

with st.sidebar:
    st.header("Studio settings")
    backend = st.selectbox("VLM backend", ["Verifier only (fast, CPU)", "Gemini description + verify"])
    api_decomposition = st.checkbox("Gemini atomic decomposition", value=False, disabled=not key_available,
                                    help="Optional API call for more complex text. Extracted facts still need verification.")
    crosscheck = st.checkbox("VLM fact cross-check", value=key_available, disabled=not key_available)
    verify_all = st.checkbox("Verify all facts (more API calls)", value=False, disabled=not key_available)
    budget = st.slider("Fact cross-check budget", 1, 20, 3, disabled=verify_all or not crosscheck)
    consistency = st.checkbox("Sampling consistency (3 extra calls)", value=False, disabled=not key_available)
    count_on = st.checkbox("Object boxes and count check (OWL-ViT)", value=False,
                           help="Deduplicated detector proposals. Missed/duplicate instances remain possible.")
    fusion_mode = st.selectbox("Fusion mode", ["v1", "v2", "v3"])
    presets = {"google/siglip-base-patch16-224": (-.10, .12),
               "google/siglip-so400m-patch14-384": (-.06, .17),
               "openai/clip-vit-base-patch32": (.15, .35),
               "openai/clip-vit-large-patch14": (.10, .40)}
    clip_model = st.selectbox("CLIP checkpoint", list(presets))
    agg_method = st.selectbox("Aggregation", ["max", "topk_mean", "mean"])
    grid = st.selectbox("Region grid", [3, 2, 1])
    ensemble = st.checkbox("Prompt ensembling", value=True)
    with st.expander("Advanced score scaling"):
        lo, hi = st.slider("CLIP window", -.2, .6, presets[clip_model], key=f"window_{clip_model}")
        st.caption("Experimental score scale, not calibrated probability. Do not tune it on test images.")

left, right = st.columns([1, 1])
with left:
    up = st.file_uploader("Drop an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
with right:
    question = st.text_input("Original question / verification goal", value="Describe what is visible in the image.")
    st.session_state.setdefault("claim_text", "")
    text = st.text_area("Response text — compound claims will be split where possible", key="claim_text", height=150)
    st.caption("Example: '5 cats are eating' becomes presence, count and activity checks. Unhandled wording remains explicitly unverified.")

cfg = {"question": question, "generate": backend.startswith("Gemini"), "api_decomposition": api_decomposition or backend.startswith("Gemini"),
       "crosscheck": crosscheck, "verify_all": verify_all, "crosscheck_limit": budget,
       "consistency": consistency, "count_on": count_on, "fusion_mode": fusion_mode,
       "clip_model": clip_model, "window": [lo, hi], "grid": grid,
       "ensemble": ensemble, "aggregation": agg_method, "max_claims": 20}
image_bytes = up.getvalue() if up else None
fingerprint = hashlib.sha256((image_bytes or b"") + json.dumps({"text": text, "cfg": cfg}, sort_keys=True).encode()).hexdigest()

if st.button("Run verification", type="primary", use_container_width=True):
    if not image_bytes:
        st.error("Upload an image first.")
    elif hi <= lo:
        st.error("The upper score-window bound must exceed the lower bound.")
    elif not question.strip():
        st.error("Enter the original question or verification goal.")
    else:
        try:
            started = time.perf_counter()
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
            needs_api = cfg["generate"] or api_decomposition or crosscheck or consistency
            gem = gemini_client() if needs_api else None
            from studio_service import analyze
            with st.spinner("Extracting facts and collecting evidence…"):
                clip = get_clip(clip_model, lo, hi, ensemble)
                detector = get_owl() if count_on else None
                result = analyze(image, text, cfg, clip, gem, detector)
            result.update(image=up.name, backend=backend, total_with_model_loading_seconds=time.perf_counter()-started)
            st.session_state["last_run"] = {"image_name": up.name, "image_bytes": image_bytes,
                                            "out": result, "fingerprint": fingerprint}
            st.session_state.pop("repair_result", None)
        except Exception as exc:
            st.error(f"Analysis could not finish ({type(exc).__name__}): {exc}")

saved = st.session_state.get("last_run")
if saved:
    result = saved["out"]
    if saved.get("fingerprint") != fingerprint:
        st.info("Showing the last completed analysis. Inputs/settings changed; run again to analyze them.")
    label, _, _ = PRESENTATION[result["final_verdict"]]
    st.subheader("2 — Final evidence decision")
    {"supported": st.success, "contradicted": st.error, "unresolved": st.warning}[result["final_verdict"]](label)
    risk = result.get("image_risk")
    st.caption(f"Raw fusion risk: {risk:.3f}" if risk is not None else "Raw fusion risk unavailable")
    if result.get("total_with_model_loading_seconds") is not None:
        st.caption(f"Total wall time including model loading and enabled API stages: {result['total_with_model_loading_seconds']:.1f}s")
    cols = st.columns(3)
    for col, name in zip(cols, ("supported", "contradicted", "unresolved")):
        col.metric(name.upper(), result["decision_counts"][name])
    for warning in result.get("warnings", []):
        st.warning(warning)
    with st.expander("Source response and execution stages"):
        st.write(result.get("response", ""))
        st.dataframe(result.get("events", []))
        st.caption("Stages count backend operations, not HTTP retries. No cost-saving guarantee is implied.")

    claims = result["claims"]
    st.subheader("3 — Facts and visual evidence")
    selected = st.selectbox("Inspect fact", range(len(claims)),
                            format_func=lambda i: f"{i+1}. {claims[i]['claim_text']}", key=f"inspect_{saved.get('fingerprint','legacy')}")
    fact = claims[selected]
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(saved["image_bytes"]))).convert("RGB")
    modes = ["Original image", "Detector candidates", "Similarity crop (not object box)"]
    mode = st.radio("Evidence overlay", modes, horizontal=True, key="overlay_mode")
    det = fact.get("detector_evidence") or {}
    visual = fact.get("visual_evidence") or {}
    from models.boxes import overlay
    caption = "Original image; no inferred object boxes drawn."
    if mode == "Detector candidates":
        if det.get("boxes"):
            image = overlay(image, det["boxes"], det.get("label", "object"))
            caption = "Detector candidates after duplicate suppression. These are fallible detections."
        else:
            caption = "No detector boxes available for this fact. A missing box does not prove absence."
    elif mode.startswith("Similarity"):
        if visual.get("box"):
            image = overlay(image, [visual["box"]], "matching crop")
            caption = "Best matching grid crop — NOT an object bounding box or proof of the claim."
        else:
            caption = "No local matching crop stored (whole-image match or unavailable evidence)."
    st.image(image, caption=caption, width=500)

    for fact in claims:
        with st.container(border=True):
            st.markdown(f"**{fact['claim_text']}**")
            verdict = fact["final_decision"]["verdict"]
            st.write(PRESENTATION[verdict][0])
            st.caption(f"Type: {fact['claim_type']} · extraction: {fact.get('decomposition', 'legacy')}")
            st.write(" ".join(fact["final_decision"]["reasons"]))
            if fact.get("vlm_verdict"):
                st.caption("VLM observation (not ground truth): " + fact["vlm_verdict"].get("raw", ""))
            if fact.get("count_check"):
                st.caption("Detector: " + fact["count_check"].get("reason", ""))
            st.caption("Diagnosis hypothesis: " + fact["diagnosis"]["name"] + "; next action: " + fact["diagnosis"]["repair"])

    st.subheader("4 — Candidate correction and re-verification")
    st.caption("Explicit API action: recheck original facts, propose a revision, verify its facts and question relevance, then audit whether supported original facts were retained. Missing or inconclusive preservation checks reject the candidate. This uses multiple API calls.")
    if st.button("Propose correction and verify", disabled=not key_available):
        try:
            from studio_service import repair_and_verify
            original_cfg = result["config"]
            with st.spinner("Generating a candidate and independently rechecking its claims…"):
                clip = get_clip(original_cfg["clip_model"], *original_cfg["window"], original_cfg["ensemble"])
                repair = repair_and_verify(ImageOps.exif_transpose(Image.open(io.BytesIO(saved["image_bytes"]))).convert("RGB"),
                                           result["response"], result["question"], gemini_client(), clip, original_cfg)
                st.session_state["repair_result"] = {"fingerprint": saved.get("fingerprint"), "result": json_safe(repair)}
        except Exception as exc:
            st.error(f"Correction unavailable ({type(exc).__name__}); the original has not been replaced.")
    repair_saved = st.session_state.get("repair_result")
    if repair_saved and repair_saved["fingerprint"] == saved.get("fingerprint"):
        repair = repair_saved["result"]
        st.write("Model-verified candidate" if repair["accepted"] else "Candidate rejected — review required")
        st.text(repair["candidate"])
        st.caption(repair["note"])
        st.json(repair["relevance_check"])
        if repair.get("preservation"):
            st.subheader("Supported-fact preservation audit")
            gate = repair["preservation"]
            st.write(" ".join(gate["reasons"]))
            st.metric("Originally supported facts", gate["protected_count"])
            st.metric("Confirmed retained by model audit", gate["preserved_count"] if gate["preserved_count"] is not None else "Unknown")
            alignment = gate.get("alignment") or {}
            st.dataframe(alignment.get("alignments", []))
            st.caption(gate["note"])

    st.subheader("5 — Data + export")
    import pandas as pd
    table = pd.DataFrame([{"claim": r["claim_text"], "type": r["claim_type"], "risk": r["risk"],
                           "verdict": r["final_decision"]["verdict"], "diagnosis": r["diagnosis"]["name"],
                           "repair": r["diagnosis"]["repair"]} for r in claims])
    st.dataframe(table, use_container_width=True)
    export = {**result, "correction": repair_saved["result"] if repair_saved and repair_saved["fingerprint"] == saved.get("fingerprint") else None}
    st.download_button("Download JSON", json.dumps(json_safe(export), indent=2, allow_nan=False), "results.json", "application/json")
    st.download_button("Download CSV", table.to_csv(index=False), "claims.csv", "text/csv")
    with st.expander("JSON"):
        st.json(export)
else:
    st.info("Upload an image and run verification to inspect individual facts.")

st.caption("Research prototype: model evidence is fallible. See the research audit for measured results and uncompleted validation.")
