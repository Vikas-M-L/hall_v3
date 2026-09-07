"""Two-stage mechanism classifier (plan section 5).

Factorization (a correct claim has no failure mechanism, so a flat
risk = sum_m p(m)*risk_m is incoherent):

    p(hallucinated | features)       <- detection head, all claims
    p(m | features, hallucinated)    <- mechanism head, hallucinated claims only
    router: argmax over {M0..M3}, M0 = decline to repair

Model class: gradient boosting first (every taxonomy row is a conjunction of
thresholds, which multinomial LR cannot express without interaction terms);
logistic regression kept as an interpretable reference. "Gating network" /
"mixture of experts" for a boosted tree over ~10 features would be inflation:
this is a two-stage classifier.

Probe leakage: pass oof_column=probe-feature index to fit() and the mechanism
head trains on out-of-fold probe predictions (StratifiedKFold), so a leaked
probe cannot be over-trusted.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

MECHANISMS = ("M0", "M1", "M2", "M3", "M4")

# Canonical feature order. Every row of the taxonomy table is a conjunction
# over these; keep this order fixed across extraction, training and serving.
FEATURES = ("confidence", "prior", "vig", "vig_per_token", "grounding",
            "uncertainty_self", "uniprobe", "clip_similarity",
            "claim_length", "claim_type_id")


def _make_model(kind: str):
    if kind == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric="logloss", n_jobs=1)
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=2000)
    raise ValueError(f"unknown model kind: {kind}")


class TwoStageClassifier:
    """Detection head + mechanism head. predict() returns (is_hall, mechanism)."""

    def __init__(self, detection_model: str = "xgboost",
                 mechanism_model: str = "xgboost"):
        self.detection_model = _make_model(detection_model)
        self.mechanism_model = _make_model(mechanism_model)
        self.classes_: tuple[str, ...] = MECHANISMS

    def fit(self, X: np.ndarray, y_hall: np.ndarray, y_mech: np.ndarray,
            oof_column: int | None = None, n_folds: int = 5):
        """X: (n, d). y_hall: {0,1}. y_mech: mechanism label per claim
        (only entries with y_hall==1 are used for the mechanism head).
        String labels are encoded internally; predict() returns strings."""
        X = np.asarray(X, dtype=float)
        y_hall = np.asarray(y_hall, dtype=int)
        self.detection_model.fit(X, y_hall)
        mask = y_hall == 1
        if int(mask.sum()) == 0:
            raise ValueError("no hallucinated claims: mechanism head has no data")
        Xm, ym_raw = X[mask], np.asarray(y_mech)[mask]
        self.mech_labels_ = tuple(sorted(set(ym_raw.tolist())))
        enc = {lab: i for i, lab in enumerate(self.mech_labels_)}
        ym = np.array([enc[v] for v in ym_raw], dtype=int)
        if oof_column is not None:
            # Replace the probe column with out-of-fold predictions of a
            # probe-only model: values the probe would have on unseen data.
            # Trains K single-feature classifiers, each predicting its held-out
            # fold — so the mechanism head can never trust a leaked probe.
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import StratifiedKFold

            oof_col = np.zeros(len(Xm))
            skf = StratifiedKFold(n_folds, shuffle=True, random_state=0)
            for tr, te in skf.split(Xm, ym):
                single = LogisticRegression(max_iter=1000)
                single.fit(Xm[tr][:, [oof_column]], ym[tr])
                oof_col[te] = single.predict(Xm[te][:, [oof_column]])
            # Back to the column's scale via per-class means of the true probe.
            Xm = Xm.copy()
            col = Xm[:, oof_column].astype(float)
            vals = col.copy()
            for cls in np.unique(ym):
                sel = oof_col == cls
                if sel.any() and (ym == cls).any():
                    vals[sel] = float(np.mean(col[ym == cls]))
            Xm[:, oof_column] = vals
            logger.info("OOF probe column built (%d hallucinated, %d folds)",
                        len(ym), n_folds)
        self.mechanism_model.fit(Xm, ym)
        self.classes_ = tuple(getattr(self.mechanism_model, "classes_", MECHANISMS))
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        is_hall = np.asarray(self.detection_model.predict(X), dtype=int)
        mechs = np.full(len(X), "M0", dtype=object)
        if is_hall.any():
            enc_pred = self.mechanism_model.predict(X[is_hall == 1])
            labs = getattr(self, "mech_labels_", self.classes_)
            mechs[is_hall == 1] = [labs[int(i)] for i in enc_pred]
        return is_hall, mechs

    def predict_proba_detection(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.detection_model.predict_proba(np.asarray(X, dtype=float)))[:, 1]
