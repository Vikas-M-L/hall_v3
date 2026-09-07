"""CPU-only offline GBM M0-M4 diagnosis classifier (STANDALONE — not wired live).

Pipeline position (when connected): claim signals -> 6-feature vector ->
GradientBoosting -> {mechanism, probabilities} -> ROUTING table -> action.

Deliberately NOT connected to app.py / run_pipeline.py: the live product uses
the transparent rule-based diagnose() (fusion/diagnose.py). This GBM is the
benchmarkable counterpart — present it as the "Gradient Boosting M0-M4
Diagnosis Classifier", never as a fine-tuned VLM.

Features (fixed order, do not reorder without retraining):
    clip_similarity, region_evidence, owl_count, claimed_count,
    gemini_consistency, gemini_confidence
where *_consistency/_confidence are agreements/probabilities in [0,1]
(higher = more supported), and counts are integer box/claimed numbers.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURES = ["clip_similarity", "region_evidence", "owl_count", "claimed_count",
            "gemini_consistency", "gemini_confidence"]

MECHANISMS = ("M0", "M1", "M2", "M3", "M4")

ROUTING = {
    "M0": {"diagnosis": "Correct / Grounded", "action": "ACCEPT"},
    "M1": {"diagnosis": "Object Hallucination", "action": "RE-PROMPT"},
    "M2": {"diagnosis": "Counting Error", "action": "RECOUNT"},
    "M3": {"diagnosis": "Uncertain / Conflicting Evidence", "action": "ABSTAIN_OR_RECHECK"},
    "M4": {"diagnosis": "Spatial / Relationship Error", "action": "RELATION_REASK"},
}


def generate_synthetic(n_per_class: int = 120, seed: int = 0) -> pd.DataFrame:
    """Taxonomy-shaped training rows. M0: everything agrees. M1: weak visual +
    weak consistency (prior talking). M2: strong visual but count mismatch.
    M3: mid signals + low consistency. M4: decent visual, low consistency."""
    rng = np.random.default_rng(seed)
    rows = []

    def jit(v, s=0.06):
        return float(np.clip(rng.normal(v, s), 0.0, 1.0))

    for _ in range(n_per_class):
        rows.append({"clip_similarity": jit(0.90), "region_evidence": jit(0.88),
                     "owl_count": 2, "claimed_count": 2,
                     "gemini_consistency": jit(0.93), "gemini_confidence": jit(0.92),
                     "diagnosis": "M0"})
        rows.append({"clip_similarity": jit(0.22), "region_evidence": jit(0.18),
                     "owl_count": 0, "claimed_count": 1,
                     "gemini_consistency": jit(0.15), "gemini_confidence": jit(0.20),
                     "diagnosis": "M1"})
        rows.append({"clip_similarity": jit(0.82), "region_evidence": jit(0.78),
                     "owl_count": 2, "claimed_count": 3,
                     "gemini_consistency": jit(0.70), "gemini_confidence": jit(0.68),
                     "diagnosis": "M2"})
        rows.append({"clip_similarity": jit(0.45), "region_evidence": jit(0.40),
                     "owl_count": 2, "claimed_count": 2,
                     "gemini_consistency": jit(0.32), "gemini_confidence": jit(0.35),
                     "diagnosis": "M3"})
        rows.append({"clip_similarity": jit(0.74), "region_evidence": jit(0.60),
                     "owl_count": 2, "claimed_count": 2,
                     "gemini_consistency": jit(0.42), "gemini_confidence": jit(0.44),
                     "diagnosis": "M4"})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def train(df: pd.DataFrame, n_estimators: int = 150, learning_rate: float = 0.05,
          max_depth: int = 3, test_size: float = 0.2, seed: int = 42) -> dict:
    """Stratified train/test split, GBM fit, report + confusion matrix."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split

    missing = [c for c in FEATURES + ["diagnosis"] if c not in df.columns]
    if missing:
        raise KeyError(f"training data missing columns: {missing}")
    X, y = df[FEATURES], df["diagnosis"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                          random_state=seed, stratify=y)
    model = GradientBoostingClassifier(n_estimators=n_estimators,
                                       learning_rate=learning_rate,
                                       max_depth=max_depth, random_state=seed)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return {"model": model, "features": list(FEATURES),
            "report": classification_report(yte, pred, output_dict=True),
            "report_text": classification_report(yte, pred),
            "confusion": confusion_matrix(yte, pred, labels=list(MECHANISMS)),
            "labels": list(MECHANISMS), "n_train": len(Xtr), "n_test": len(Xte)}


def save_bundle(bundle: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": bundle["model"], "features": bundle["features"]}, p)
    return p


def load_bundle(path: str | Path) -> dict:
    b = joblib.load(path)
    if list(b.get("features", [])) != FEATURES:
        raise ValueError("artifact feature order mismatch — retrain, do not patch")
    return b


def diagnose_claim(bundle: dict, claim: dict) -> dict:
    """claim: the 6 signals. Returns {mechanism, action, probabilities}."""
    model, features = bundle["model"], bundle["features"]
    X = pd.DataFrame([{k: claim[k] for k in features}])
    mech = str(model.predict(X)[0])
    probs = {str(l): float(p) for l, p in zip(model.classes_, model.predict_proba(X)[0])}
    route = ROUTING.get(mech, {"diagnosis": "Unknown", "action": "RE-VERIFY"})
    return {"mechanism": mech, "diagnosis": route["diagnosis"], "action": route["action"],
            "confidence": probs.get(mech, 0.0), "probabilities": probs}
