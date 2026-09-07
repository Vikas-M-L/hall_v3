"""Hallucination Autopsy — one claim, full chain: risk → evidence → diagnosis → proof → repair."""
from __future__ import annotations

import io

import streamlit as st

import studio_lib as L

st.set_page_config(page_title="Autopsy — V-TRACE+ Studio", layout="wide")
st.markdown(L.CSS, unsafe_allow_html=True)
st.markdown("<div class='hero-title'>Hallucination <span class='grad'>Autopsy</span></div>",
            unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>Detect → Explain → Prove → Route. "
            "Pick a claim from your last Checker run (or the gallery) and dissect it."
            "</div>", unsafe_allow_html=True)

run = st.session_state.get("last_run")
gallery = L.load_gallery()

if run is None:
    st.info("No Checker run yet this session — autopsying a gallery case. "
            "Run the Checker first to autopsy your own image.")
    labels = [f"{sc['title']}: {r['text'][:60]}"
              for sc in gallery for r in sc["results"] if r["verdict"] != "grounded"]
    options = [(sc, r) for sc in gallery for r in sc["results"] if r["verdict"] != "grounded"]
    if not options:
        st.stop()
    sel = st.selectbox("Hallucinated gallery claim:", labels, index=0)
    sc, r = options[labels.index(sel)]
    from PIL import Image as _I

    image = _I.open(L.REPO_ROOT / "assets" / sc["image"]).convert("RGB")
    claim_text, risk = r["text"], r["risk"]
    ev = {"evidence": 1 - r["risk"], "similarity": 1 - r["risk"]}
    extra = {"expected": r["expected"], "correction": r.get("correction", "")}
else:
    from PIL import Image as _I

    image = _I.open(io.BytesIO(run["image_bytes"])).convert("RGB")
    claims = run["out"]["claims"]
    sel = st.selectbox("Claim to autopsy:",
                       [f"#{i+1} {c['claim_text'][:60]} ({c['risk']:.2f})"
                        for i, c in enumerate(claims)], index=0)
    c = claims[int(sel.split()[0][1:]) - 1]
    claim_text, risk = c["claim_text"], c["risk"]
    ev = {"evidence": 1 - c["risks"].get("evidence", float("nan")),
          "similarity": 1 - c["risks"].get("clip_similarity", float("nan")),
          "uncertainty": c.get("consistency_uncertainty"),
          "vlm": (c.get("vlm_verdict") or {}).get("supported"),
          "count": c.get("count_check"),
          "rules": c.get("rules_fired", []),
          "diagnosis": c.get("diagnosis")}
    extra = {}

# 1 — claim + risk
vtext, vcls = L.verdict_of(risk)
pct = 0 if risk != risk else int(risk * 100)
st.markdown(f"<div class='card'><b style='font-size:18px'>{claim_text}</b><br>"
            f"<span class='badge {vcls}'>RISK {risk:.3f} — {vtext}</span>"
            f"<div class='risk-bar'><div class='risk-fill' style='width:{pct}%;"
            f"background:{L.bar_of(risk)}'></div></div></div>", unsafe_allow_html=True)

# 2 — evidence table
st.subheader("Evidence")
rows = [(k, ("—" if v is None or v != v else f"{v:.3f}"))
        for k, v in ev.items() if k not in ("rules", "diagnosis")]
st.table([{"signal": k, "value": v} for k, v in rows
          if not isinstance(v, (dict, list))])
if ev.get("rules"):
    st.caption("Rules fired: " + ", ".join(ev["rules"]))
if ev.get("count"):
    st.caption(f"Counter: {ev['count'].get('reason', '')}")

# 3 — diagnosis (rule engine over the same numbers)
st.subheader("Diagnosis")
import sys as _sys

_sys.path.insert(0, str(L.REPO_ROOT))
from fusion.diagnose import diagnose as _diag

if isinstance(ev.get("diagnosis"), dict):
    dg = ev["diagnosis"]
else:
    cc = ev.get("count")
    dg = _diag(risk, 1 - ev.get("evidence", float("nan")),
               1 - ev.get("similarity", float("nan")),
               count_match=(cc["match"] if cc else None),
               uncertainty=ev.get("uncertainty"),
               vlm_supported=ev.get("vlm"))
st.markdown(f"<div class='card'><b>{dg['mechanism']} — {dg['name']}</b><ul>"
            + "".join(f"<li>{x}</li>" for x in dg["reasons"]) + "</ul></div>",
            unsafe_allow_html=True)

# 4 — visual proof (recomputed box, cached model)
st.subheader("Visual proof")
try:
    clip = L.get_clip("google/siglip-base-patch16-224", -0.10, 0.12, True)
    regions = clip.encode_regions(image)
    _s, box, whole = clip.evidence(claim_text, regions)
    if box is not None and not whole:
        from PIL import ImageDraw as _D

        ov = image.copy()
        d = _D.Draw(ov)
        x0, y0, x1, y1 = box
        for w in range(4):
            d.rectangle([x0 - w, y0 - w, x1 + w, y1 + w], outline=(52, 211, 153))
        st.image(ov, caption="Best-match region", use_container_width=True)
    else:
        st.image(image, caption="Matched globally — no local box.", use_container_width=True)
except Exception as exc:
    st.image(image, caption=f"Proof unavailable: {exc}", use_container_width=True)

# 5 — repair
st.subheader("Repair")
st.markdown(f"<div class='card'><b>{dg['repair']}</b> <span class='small'>({dg['cost']})</span><br>"
            f"{dg['action']}</div>", unsafe_allow_html=True)
if extra.get("correction"):
    st.success("Correction: " + extra["correction"])
