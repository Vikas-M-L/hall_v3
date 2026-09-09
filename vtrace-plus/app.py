"""V-TRACE+ Studio — claim-level hallucination check.

Image -> Claims (local split, no VLM) -> CLIP verification -> fusion -> JSON
Run:  streamlit run app.py   (from vtrace-plus/)
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from fusion.negation import apply_negation
from fusion.vtrace_fusion import SignalBundle, aggregate, fuse

st.set_page_config(page_title="V-TRACE+ Studio", page_icon="V", layout="wide")

CSS = """
<style>
.stApp { background:
  radial-gradient(1000px 500px at 15% -10%, #1b2a4a 0%, transparent 60%),
  radial-gradient(900px 500px at 90% 0%, #3b1d5e 0%, transparent 55%),
  radial-gradient(700px 600px at 50% 110%, #0e3a3a 0%, transparent 60%),
  #0a0f1e fixed; }
h1, h2, h3 { letter-spacing: -0.02em; }
.hero-title { font-size: 44px; font-weight: 800; line-height: 1.05; margin-bottom: 2px; }
.grad { background: linear-gradient(90deg, #7dd3fc, #c084fc, #f0abfc);
  -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero-sub { color: #94a3b8; font-size: 15px; margin-bottom: 14px; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 4px; }
.chip { padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: 700;
  letter-spacing: 0.04em; background: #131d33; border: 1px solid #2b3a5c; color: #cbd5e1; }
.chip-hot { background: linear-gradient(90deg, #334155, #475569); color: #fff; }
.step-arrow { color: #475569; font-weight: 800; align-self: center; }
.banner { border-radius: 16px; padding: 18px 22px; margin: 14px 0;
  border: 1px solid #2b3a5c; background: linear-gradient(180deg, #101a30, #0c1426); }
.banner-safe { border-left: 6px solid #10b981; }
.banner-review { border-left: 6px solid #f59e0b; }
.banner-risky { border-left: 6px solid #ef4444; }
.banner-num { font-size: 40px; font-weight: 800; line-height: 1; }
.banner-label { font-size: 13px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; }
.tile { border: 1px solid #263248; border-radius: 12px; padding: 10px 14px;
  background: linear-gradient(180deg, #111a2e, #0c1426); text-align: center; }
.tile-num { font-size: 26px; font-weight: 800; }
.tile-label { font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; }
.gauge { position: relative; height: 14px; border-radius: 8px; margin: 10px 0 4px;
  background: linear-gradient(90deg, #10b981 0%, #10b981 40%, #f59e0b 40%, #f59e0b 60%, #ef4444 60%, #ef4444 100%); }
.gauge-marker { position: absolute; top: -5px; width: 4px; height: 24px; background: #fff;
  border-radius: 2px; box-shadow: 0 0 8px #fff; }
.risk-card { border: 1px solid #263248; border-radius: 14px; padding: 13px 16px; margin-bottom: 10px;
  background: linear-gradient(180deg, #111a2e 0%, #0d1526 100%); }
.medal { display: inline-flex; align-items: center; justify-content: center;
  width: 30px; height: 30px; border-radius: 50%; font-weight: 800; font-size: 14px; margin-right: 8px; }
.medal-1 { background: linear-gradient(135deg, #f59e0b, #ef4444); color: #fff; }
.medal-2 { background: #334155; color: #e2e8f0; border: 1px solid #475569; }
.medal-3 { background: #1e293b; color: #94a3b8; border: 1px solid #334155; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 12px; font-weight: 700; }
.badge-ok { background: #052e22; color: #34d399; border: 1px solid #065f46; }
.badge-warn { background: #3b2305; color: #fbbf24; border: 1px solid #92400e; }
.badge-bad { background: #3b0a0a; color: #f87171; border: 1px solid #7f1d1d; }
.risk-bar { height: 10px; border-radius: 6px; background: #1f2b44; overflow: hidden; margin: 8px 0 6px; }
.risk-fill { height: 100%; border-radius: 6px; }
.sig-row { display: flex; align-items: center; gap: 8px; font-size: 12px; color: #94a3b8; margin-top: 3px; }
.sig-name { width: 88px; }
.sig-track { flex: 1; height: 6px; border-radius: 4px; background: #1f2b44; overflow: hidden; }
.sig-fill-e { height: 100%; background: linear-gradient(90deg, #38bdf8, #818cf8); }
.sig-fill-s { height: 100%; background: linear-gradient(90deg, #c084fc, #f0abfc); }
.sig-val { width: 44px; text-align: right; color: #e2e8f0; font-weight: 700; }
.small { color: #94a3b8; font-size: 13px; }
.footer { text-align: center; color: #475569; font-size: 12px; margin-top: 26px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------ hero
st.markdown("<div class='hero-title'>V-TRACE<span class='grad'>+ Studio</span></div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>Inference-time hallucination diagnosis and "
            "repair-routing for vision-language output. "
            "Text-only mode, CLIP verification, CPU friendly.</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='chips'><span class='chip chip-hot'>1 IMAGE</span><span class='step-arrow'>&gt;</span>"
    "<span class='chip chip-hot'>2 CLAIMS</span><span class='step-arrow'>&gt;</span>"
    "<span class='chip chip-hot'>3 VERIFY</span><span class='step-arrow'>&gt;</span>"
    "<span class='chip chip-hot'>4 FUSE</span><span class='step-arrow'>&gt;</span>"
    "<span class='chip chip-hot'>5 JSON</span></div>",
    unsafe_allow_html=True,
)

with st.expander("How it scores (30 seconds)"):
    st.markdown(
        "Your text is split into sentences = **claims**. Each claim is embedded with CLIP and "
        "compared against a 3x3 grid of image regions. **evidence** = best-region match, "
        "**similarity** = whole-image match. Both map to risk (`1 - match`) and fuse into one "
        "score per claim. **Read the ranking, not the absolutes** — the top claim is the one "
        "to double-check."
    )

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Studio settings")
    with st.expander("Model", expanded=True):
        backend = st.selectbox("VLM backend",
                               ["Verifier only (fast, CPU)",
                                "Gemini VLM + verify (real, ~1-2 min)"],
                               index=0)
        try:
            from models.gemini_wrapper import GeminiWrapper as _G

            _G()
            _key_ok = True
        except Exception:
            _key_ok = False
        crosscheck_on = st.checkbox(
            "VLM cross-check on uncertain claims" + ("" if _key_ok else " (needs GEMINI_API_KEY)"),
            value=_key_ok,
            help="Asks Gemini YES/NO on claims with risk in [0.35, 0.65), max 3 per run. "
                 "One API call each — the cascade's stage-2.")
        verify_all = st.checkbox(
            "Maximum accuracy: verify ALL claims" + ("" if _key_ok else " (needs GEMINI_API_KEY)"),
            value=False,
            help="Cross-checks every claim (~3s each). With So400M + consistency this is "
                 "the most accurate configuration on this machine.")
        consistency_on = st.checkbox("Self-consistency samples (n=3)", value=True,
                                     help="Needs Gemini backend. Extra API calls.")
        count_on = st.checkbox("Count check (OWL-ViT detector)", value=False,
                               help="Draws one box per animal/object and counts. Fixes CLIP "
                                    "count-blindness ('2 dogs' vs 2 cats). Slow first load (~3 min), "
                                    "seconds after.")
        fusion_mode = st.selectbox("Fusion mode", ["v1", "v2", "v3"], index=0,
                                     help="v3 = pairwise contradiction + abstention: "
                                          "supported / contradicted / unresolved + reasons.")
        clip_model = st.selectbox(
            "CLIP checkpoint",
            ["google/siglip-so400m-patch14-384",
             "google/siglip-base-patch16-224",
             "openai/clip-vit-base-patch32",
             "openai/clip-vit-large-patch14"],
            index=1,
            help="Base (default): fast + accurate. So400M: best margin (0.89), ~5-10x slower on CPU — for single important images.",
        )
        agg_method = st.selectbox("Aggregation", ["max", "topk_mean", "mean"], index=0)
    with st.expander("Speed vs precision", expanded=True):
        grid = st.selectbox("Region grid", [3, 2, 1], index=0,
                            help="3 = 10 regions, best grounding. 1 = full image only, ~5x faster.")
        ensemble = st.checkbox("Prompt ensembling", value=True,
                               help="Averages 3 caption templates per claim. 2 extra text encodes, better accuracy.")
    with st.expander("Calibration", expanded=True):
        # Per-model window presets (CLIP and SigLIP live in different bands).
        presets = {"google/siglip-so400m-patch14-384": (-0.06, 0.17),
                   "google/siglip-base-patch16-224": (-0.10, 0.12),
                   "openai/clip-vit-base-patch32": (0.15, 0.35),
                   "openai/clip-vit-large-patch14": (0.10, 0.40)}
        plo, phi = presets[clip_model]
        cos_min, cos_max = st.slider("CLIP window", -0.2, 0.6, (plo, phi),
                                     key=f"coswin_{clip_model}")
        st.caption("Scores compress when the window dwarfs the observed cosine span. "
                   "The app suggests a window after each run.")


# ------------------------------------------------------------------ inputs
st.subheader("1 — Input")
col_img, col_txt = st.columns([5, 7])
with col_img:
    up = st.file_uploader("Drop an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
with col_txt:
    st.session_state.setdefault("claim_text", "A red square. A blue elephant.")
    p1, p2, p3 = st.columns(3)
    if p1.button("Red-square test"):
        st.session_state.claim_text = "A red square. A blue elephant."
    if p2.button("Puppy test"):
        st.session_state.claim_text = "A dog. A black dog. A golden puppy."
    if p3.button("Clear"):
        st.session_state.claim_text = ""
    text = st.text_area("Response text — one claim per sentence", key="claim_text",
                        height=130, placeholder="e.g. A dog. A black dog. A golden puppy.")

run = st.button("Run verification", type="primary", use_container_width=True)


@st.cache_resource(show_spinner="Loading CLIP on CPU (once, then cached)…")
def get_clip(checkpoint: str, cos_min: float, cos_max: float, ensemble: bool):
    from models.clip_wrapper import CLIPWrapper

    cfg = {
        "device": "cpu", "dtype": "float32", "seed": 42,
        "clip": {"backend": "transformers", "checkpoint": checkpoint,
                 "region_grid": 3, "region_overlap": 0.2,
                 "cos_min": cos_min, "cos_max": cos_max,
                 "templates": (["{}", "a photo of {}", "a picture showing {}"]
                               if ensemble else ["{}"])},
    }
    return CLIPWrapper(cfg)


@st.cache_resource(show_spinner="Loading OWL-ViT detector (once, ~3 min first time)…")
def get_owl():
    from models.owl_wrapper import OWLDetector

    return OWLDetector()


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


if run:
    if up is None:
        st.error("Upload an image first.")
        st.stop()
    if not text or not text.strip():
        st.error("Enter response text first.")
        st.stop()

    from PIL import Image

    from models.claim_extractor import Claim, ClaimExtractor

    image = Image.open(io.BytesIO(up.getvalue())).convert("RGB")

    def _f1(a: str, b: str) -> float:
        import re as _re
        from collections import Counter as _C
        ta, tb = _re.findall(r"[a-z0-9]+", a.lower()), _re.findall(r"[a-z0-9]+", b.lower())
        if not ta or not tb:
            return 0.0
        ca, cb = _C(ta), _C(tb)
        inter = sum((ca & cb).values())
        if not inter:
            return 0.0
        return 2 * inter / (sum(ca.values()) + sum(cb.values()))

    extractor = None
    from models.claim_extractor import Claim, ClaimExtractor

    def _mk_extractor():
        return ClaimExtractor(
            {"claim_extractor": {"max_claims": 20,
                                 "type_taxonomy": ["object", "attribute", "count",
                                                   "spatial", "relation", "action",
                                                   "ocr", "scene"]}},
            vlm=None)

    use_gemini = backend.startswith("Gemini")
    samples: list[str] = []
    if use_gemini:
        from models.gemini_wrapper import GeminiWrapper

        gem = GeminiWrapper()
        with st.spinner("Gemini describing image…"):
            gemini_response = gem.describe(image)
        st.markdown(f"**Gemini says:** {gemini_response}")
        if consistency_on:
            with st.spinner("Gemini sampling 3x for self-consistency…"):
                samples = gem.sample(image, "Describe this image in detail.",
                                     n=3, temperature=0.7)
        entries = gem.decompose(gemini_response)
        extractor = _mk_extractor()
        if entries:
            claims = [Claim(text=str(e.get("claim", "")).strip(),
                            source=str(e.get("source", e.get("claim", ""))).strip(),
                            span=ClaimExtractor._locate(
                                str(e.get("source", e.get("claim", ""))).strip(),
                                gemini_response),
                            claim_type=extractor.classify_type(str(e.get("claim", ""))),
                            claim_id=f"ui_c{i}")
                      for i, e in enumerate(entries[:20]) if str(e.get("claim", "")).strip()]
            st.caption(f"Gemini decomposed {len(claims)} atomic claims.")
        else:
            claims = _mk_extractor().extract(gemini_response, response_id="ui")
            st.caption("Decomposition fallback: sentence split.")
    else:
        if not text or not text.strip():
            st.error("Enter response text first.")
            st.stop()
        claims = _mk_extractor().extract(text.strip(), response_id="ui")

    with st.spinner("Encoding regions + scoring claims…"):
        t0 = time.time()
        clip = get_clip(clip_model, cos_min, cos_max, ensemble)
        regions = clip.encode_regions(image, grid=grid)
        active = {"confidence": False, "evidence": True, "clip_similarity": True,
                  "uniprobe": False, "counterfactual": False}
        rows = []
        v3_on = (fusion_mode == "v3")
        if v3_on:
            from fusion.negation import make_counterclaim
        for c in claims:
            score, box, whole_won = clip.evidence(c.text, regions)
            sim = clip.similarity(c.text, regions)
            b = SignalBundle(claim_id=c.claim_id, claim_text=c.text,
                             claim_type=c.claim_type,
                             evidence=score, clip_similarity=sim,
                             stubbed=("uniprobe", "counterfactual"))
            pw = {}
            if v3_on:
                cc = make_counterclaim(c.text)
                if cc["counterclaim_text"]:
                    pw = clip.pairwise_scores(c.text, cc["counterclaim_text"], regions)
            r = fuse(b, mode=fusion_mode, active_signals=active, drop_stubbed=True,
                     pairwise=pw if v3_on else None)
            if not v3_on:
                # v3 verifies negated claims through the positive form itself.
                corr, neg = apply_negation(c.text, r.risks.get("evidence", float("nan")),
                                           r.risks.get("clip_similarity", float("nan")))
                if neg and corr == corr:
                    r.risk = corr
                    r.rules_fired = [*r.rules_fired,
                                     "negation-inverted (negated claim: match = risk)"]
            r.consistency_uncertainty = (
                float(1.0 - sum(_f1(c.text, s) for s in samples) / len(samples))
                if samples else float("nan"))
            rows.append((c, r, None if whole_won else box))
        img_risk = aggregate([rr.risk for _, rr, _ in rows], method=agg_method, topk=3)
        dt = time.time() - t0

    if not rows:
        st.warning("No claims found — write at least one sentence.")
        st.stop()

    # Mechanism diagnosis + repair routing for every claim (transparent rules).
    from fusion.diagnose import diagnose
    from fusion.negation import has_negation

    def _nan2none(v):
        return None if v != v else float(v)

    for c, r, _ in rows:
        unc = _nan2none(getattr(r, "consistency_uncertainty", float("nan")))
        cc = getattr(r, "count_check", None)
        vv = getattr(r, "vlm_verdict", None) or {}
        r.diagnosis = diagnose(
            r.risk, r.risks.get("evidence", float("nan")),
            r.risks.get("clip_similarity", float("nan")),
            claim_type=c.claim_type,
            negated=(any("negation-inverted" in f for f in r.rules_fired)
                     or (v3_on and has_negation(c.text))),
            uncertainty=unc,
            count_match=(cc["match"] if cc else None),
            vlm_supported=vv.get("supported"))

    # Count check: one batched OWL-ViT call for every countable claim.
    if count_on:
        from models.owl_wrapper import count_verdict, parse_count

        needs = {}
        for c, _, _ in rows:
            num, lab = parse_count(c.text)
            if num is not None and lab:
                needs[c.claim_id] = (num, lab)
        if needs:
            with st.spinner(f"OWL-ViT counting {len(set(v[1] for v in needs.values()))} object type(s)…"):
                det = get_owl().detect(image, sorted(set(v[1] for v in needs.values())))
            for c, r, _ in rows:
                if c.claim_id in needs:
                    num, lab = needs[c.claim_id]
                    r.count_check = {"expected_num": num, "expected_label": lab,
                                     **count_verdict(num, lab, det.get(lab, {"count": 0}))}
        for _, r, _ in rows:
            if not hasattr(r, "count_check"):
                r.count_check = None
    else:
        for _, r, _ in rows:
            r.count_check = None
    n_xcheck = 0
    if (crosscheck_on or verify_all) and _key_ok:
        from models.gemini_wrapper import GeminiWrapper

        if verify_all:
            amb = [(c, r) for c, r, _ in rows if r.risk == r.risk]
        else:
            amb = [(c, r) for c, r, _ in rows
                   if r.risk == r.risk and 0.35 <= r.risk < 0.65][:3]
        if amb:
            try:
                gem_x = GeminiWrapper()
                with st.spinner(f"VLM cross-checking {len(amb)} uncertain claim(s)…"):
                    for c, r in amb:
                        try:
                            r.vlm_verdict = gem_x.verify(image, c.text)
                            n_xcheck += 1
                        except Exception as exc:
                            r.vlm_verdict = {"supported": None, "raw": str(exc)[:100]}
            except Exception as exc:
                st.warning(f"Cross-check unavailable: {exc}")
        for _, r, _ in rows:
            if not hasattr(r, "vlm_verdict"):
                r.vlm_verdict = {"supported": None, "raw": ""}

    # ---------------------------------------------------------- verdict banner
    st.subheader("2 — Verdict")
    if img_risk != img_risk:
        bcls, big = "banner-review", "N/A"
    elif img_risk < 0.4:
        bcls, big = "banner-safe", "LOOKS GROUNDED"
    elif img_risk < 0.6:
        bcls, big = "banner-review", "NEEDS REVIEW"
    else:
        bcls, big = "banner-risky", "LIKELY HALLUCINATED"
    pos = 0 if img_risk != img_risk else int(max(0.0, min(1.0, img_risk)) * 100)
    n_g = sum(1 for _, r, _ in rows if r.risk == r.risk and r.risk < 0.4)
    n_u = sum(1 for _, r, _ in rows if r.risk == r.risk and 0.4 <= r.risk < 0.6)
    n_b = sum(1 for _, r, _ in rows if r.risk == r.risk and r.risk >= 0.6)
    st.markdown(
        f"<div class='banner {bcls}'><div style='display:flex;justify-content:space-between;align-items:center'>"
        f"<div><div class='banner-label'>Image risk ({agg_method}) · {dt:.1f}s on CPU</div>"
        f"<div class='banner-num'>{img_risk:.3f} — {big}</div>"
        f"<div class='gauge'><div class='gauge-marker' style='left:{pos}%'></div></div></div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )
    t1, t2, t3, t4 = st.columns(4)
    t1.markdown(f"<div class='tile'><div class='tile-num'>{len(rows)}</div><div class='tile-label'>claims</div></div>", unsafe_allow_html=True)
    t2.markdown(f"<div class='tile'><div class='tile-num' style='color:#34d399'>{n_g}</div><div class='tile-label'>grounded</div></div>", unsafe_allow_html=True)
    t3.markdown(f"<div class='tile'><div class='tile-num' style='color:#fbbf24'>{n_u}</div><div class='tile-label'>uncertain</div></div>", unsafe_allow_html=True)
    t4.markdown(f"<div class='tile'><div class='tile-num' style='color:#f87171'>{n_b}</div><div class='tile-label'>flagged</div></div>", unsafe_allow_html=True)
    if fusion_mode == "v3":
        from fusion.vtrace_fusion import evidence_coverage

        cov = evidence_coverage([r for _, r, _ in rows])
        vs = sum(1 for _, r, _ in rows if r.verdict == "supported")
        vc = sum(1 for _, r, _ in rows if r.verdict == "contradicted")
        vu = sum(1 for _, r, _ in rows if r.verdict == "unresolved")
        st.info(f"v3 verdicts: {vs} supported · {vc} contradicted · {vu} unresolved — "
                f"evidence coverage {cov['coverage']:.0%} decisive "
                f"({cov['decisive']}/{cov['total']}).")
    if n_xcheck:
        st.caption(f"Cascade: {n_xcheck}/{len(rows)} claims needed the VLM "
                   f"({100 * (1 - n_xcheck / len(rows)):.0f}% decided by free CLIP signals).")
    else:
        st.caption("Cascade: all claims decided by free CLIP signals — 0 VLM calls.")

    sug = clip.suggested_window()
    if sug is not None and (abs(sug[0] - cos_min) > 1e-9 or abs(sug[1] - cos_max) > 1e-9):
        st.info(f"Suggested window from observed cosines: [{sug[0]}, {sug[1]}] "
                f"(current [{cos_min}, {cos_max}]). Set the sidebar slider there and re-run for wider spread.")

    # ---------------------------------------------------------- visual proof
    st.subheader("3 — Visual proof + ranked claims")
    ranked = sorted(rows, key=lambda cr: (cr[1].risk != cr[1].risk, -cr[1].risk))
    left, right = st.columns([5, 7])
    with left:
        labels = [f"#{i+1} {c.text[:55]} ({r.risk:.2f})" for i, (c, r, _) in enumerate(ranked)]
        sel = st.selectbox("Grounding box for:", labels, index=0)
        pick = ranked[labels.index(sel)]
        if pick[2] is not None:
            from PIL import ImageDraw

            overlay = image.copy()
            d = ImageDraw.Draw(overlay)
            x0, y0, x1, y1 = pick[2]
            for w in range(4):
                d.rectangle([x0 - w, y0 - w, x1 + w, y1 + w], outline=(52, 211, 153))
            st.image(overlay, caption=f"Best-match region — {pick[0].text}", use_container_width=True)
        else:
            st.image(image, caption="Best match is the whole image — no local box.", use_container_width=True)
        st.caption(f"{up.name} ({image.size[0]}x{image.size[1]}) · grid {grid} · {clip_model.split('/')[-1]}")
    with right:
        for i, (c, r, _box) in enumerate(ranked):
            vtext, vcls = verdict_of(r.risk)
            pct = 0 if r.risk != r.risk else int(r.risk * 100)
            er = r.risks.get("evidence", float("nan"))
            sr = r.risks.get("clip_similarity", float("nan"))
            tot = (0 if er != er else 1 - er) + (0 if sr != sr else 1 - sr)
            ew = 0 if tot <= 0 or er != er else int(round((1 - er) / tot * 100))
            sw = 0 if tot <= 0 or sr != sr else int(round((1 - sr) / tot * 100))
            mcls = "medal-1" if i == 0 else ("medal-2" if i == 1 else "medal-3")
            unc = getattr(r, "consistency_uncertainty", float("nan"))
            unc_txt = (f" · self-consistency uncertainty: {unc:.2f}"
                       + (" (M3 hint: confident-but-unstable)" if unc == unc and unc > 0.5 else "")
                       if unc == unc else "")
            dg = r.diagnosis
            mech_txt = (f"<div class='sig-row'><span class='sig-name'>diagnosis</span>"
                        f"<span class='badge badge-warn'>{dg['mechanism']} — {dg['name']}</span></div>"
                        f"<div class='sig-row'><span class='sig-name'>repair</span>"
                        f"<span class='small'>{dg['repair']} <i>({dg['cost']})</i></span></div>")
            vv = getattr(r, "vlm_verdict", None) or {"supported": None}
            vs = vv.get("supported")
            vlm_txt = "" if vs is None else (
                " · <b style='color:#34d399'>VLM: SUPPORTED</b>" if vs
                else " · <b style='color:#f87171'>VLM: REFUTED</b>")
            cc = getattr(r, "count_check", None)
            if cc is None:
                count_txt = ""
            elif cc["match"]:
                count_txt = f" · <b style='color:#34d399'>COUNT OK: {cc['reason']}</b>"
            else:
                count_txt = f" · <b style='color:#f87171'>COUNT MISMATCH: {cc['reason']}</b>"
            v3_txt = ""
            if getattr(r, "verdict", None):
                vcls3 = {"supported": "badge-ok", "contradicted": "badge-bad"}.get(
                    r.verdict, "badge-warn")
                pw = r.pairwise or {}
                pm = pw.get("claim_match", float("nan"))
                nm = pw.get("counter_match", float("nan"))
                mg = pw.get("margin", float("nan"))
                side = ("claim %.2f vs counter %.2f (margin %+.2f)" % (pm, nm, mg)
                        if mg == mg else "pairwise unavailable")
                v3_txt = (f"<div class='sig-row'><span class='sig-name'>v3 verdict</span>"
                          f"<span class='badge {vcls3}'>{r.verdict.upper()}</span></div>"
                          f"<div class='small'>{side}"
                          + (f"<br>reasons: {'; '.join(r.reasons)}" if r.reasons else "")
                          + "</div>")
            st.markdown(
                f"<div class='risk-card'><div style='display:flex;justify-content:space-between;align-items:center'>"
                f"<div><span class='medal {mcls}'>#{i+1}</span><b>{c.text}</b></div>"
                f"<span class='badge {vcls}'>{vtext}</span></div>"
                f"<div class='risk-bar'><div class='risk-fill' style='width:{pct}%;background:{bar_of(r.risk)}'></div></div>"
                f"<div class='sig-row'><span class='sig-name'>evidence</span>"
                f"<div class='sig-track'><div class='sig-fill-e' style='width:{ew}%'></div></div>"
                f"<span class='sig-val'>{er:.2f}</span></div>"
                f"<div class='sig-row'><span class='sig-name'>similarity</span>"
                f"<div class='sig-track'><div class='sig-fill-s' style='width:{sw}%'></div></div>"
                f"<span class='sig-val'>{sr:.2f}</span></div>"
                f"<div class='small' style='margin-top:6px'>risk {r.risk:.3f} · {c.claim_type} · "
                f"top signal: {r.top_signal()}{unc_txt}{vlm_txt}{count_txt}</div>{mech_txt}{v3_txt}</div>",
                unsafe_allow_html=True,
            )

    # ---------------------------------------------------------- data + export
    st.subheader("4 — Data + export")
    out = {"image": up.name, "fusion_mode": fusion_mode, "aggregation": agg_method,
           "image_risk": img_risk, "backend": backend,
           "claims": [{**c.to_dict(), **r.to_dict(),
                       "consistency_uncertainty": getattr(r, "consistency_uncertainty", None),
                       "vlm_verdict": getattr(r, "vlm_verdict", None),
                       "count_check": getattr(r, "count_check", None),
                       "diagnosis": r.diagnosis}
                      for c, r, _ in rows]}
    st.session_state.last_run = {"image_name": up.name,
                                 "image_bytes": up.getvalue(),
                                 "out": json.loads(json.dumps(out, default=str))}
    import pandas as pd

    table = pd.DataFrame([{
        "rank": i + 1, "claim": c.text, "type": c.claim_type,
        "risk": round(r.risk, 3),
        "evidence": round(r.risks.get("evidence", float("nan")), 3),
        "similarity": round(r.risks.get("clip_similarity", float("nan")), 3),
        "top_signal": r.top_signal(), "verdict": verdict_of(r.risk)[0]}
        for i, (c, r, _) in enumerate(ranked)])
    tab1, tab2, tab3 = st.tabs(["Table", "JSON", "Calibration"])
    with tab1:
        st.dataframe(table, use_container_width=True)
        st.download_button("Download CSV", table.to_csv(index=False),
                           file_name="claims.csv", mime="text/csv")
    with tab2:
        st.json(out)
        st.download_button("Download JSON", json.dumps(out, indent=2),
                           file_name="results.json", mime="application/json")
    with tab3:
        rng = clip.observed_cosine_range()
        if rng:
            st.write(f"Raw cosines — min **{rng[0]:.4f}**, median **{rng[1]:.4f}**, max **{rng[2]:.4f}** "
                     f"over {len(clip._cos_seen)} comparisons, window [{cos_min}, {cos_max}].")
            if sug is not None:
                st.write(f"Suggested window: **[{sug[0]}, {sug[1]}]**.")
            else:
                st.write("Window looks fine — no change suggested.")
        st.caption("confidence / counterfactual need the VLM (GPU); uniprobe is an unwired stub. "
                   "CPU text-only mode fuses CLIP evidence + similarity.")

st.markdown("<div class='footer'>V-TRACE+ Studio · deterministic fusion, no training · "
            "read the ranking, not the absolutes</div>", unsafe_allow_html=True)
