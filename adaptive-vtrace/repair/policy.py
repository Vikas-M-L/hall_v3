"""Headline-experiment machinery (plan section 7).

Per-claim repair outcomes are the primitive: for every claim and every
operator, did it fix a hallucination / break a correct claim? Abstention
counts as removal on BOTH sides (else abstain-everything scores 100% fixed).

Strategies compared (rows of the §7 table):
  no_repair, uniform_<op>, all_ops, detect_then_best_uniform,
  outcome_policy (taxonomy-free 4-way policy trained on observed outcomes —
  the control that kills the paper if it ties routing), claimtype_route,
  random_route, mechanism_routed (ours), routed_no_abstain, oracle.

`compare()` takes recorded outcomes + per-claim predictions and returns the
net-gain table. GPU work is only in *recording* outcomes; everything here is
numpy/sklearn and tested synthetically.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

OPERATORS = ("contrastive", "zoom", "abstain", "relverify")
MECH_TO_OP = {"M0": None, "M1": "contrastive", "M2": "zoom", "M3": "abstain",
              "M4": "relverify"}


def net_gain_of(assign: dict[str, str | None], outcomes: dict,
                y_hall: dict[str, int]) -> dict:
    """assign: claim -> operator or None. outcomes: claim -> op -> (fixed, broke).
    Returns {fixed, broken, net}."""
    fixed = broken = 0
    for cid, op in assign.items():
        if op is None:
            continue
        f, b = outcomes[cid][op]
        hall = y_hall[cid]
        if op == "abstain":
            # Abstention removes the claim: fix iff it was hallucinated,
            # damage iff it was correct. Counted exactly once either way.
            if hall:
                fixed += 1
            else:
                broken += 1
            continue
        if hall and f:
            fixed += 1
        if hall and b:
            broken += 1  # repair corrupted an already-wrong claim: still damage
        if not hall and (f or b):
            broken += 1  # any change to a correct claim = damage
    return {"fixed": fixed, "broken": broken, "net": fixed - broken}


OP_LABELS = ("none", "contrastive", "zoom", "abstain")


def train_outcome_policy(X: np.ndarray, best_op: np.ndarray, kind: str = "xgboost"):
    """Taxonomy-free control: direct 4-way (3 ops + none) policy from features
    to observed best repair. No mechanisms, no induction, no taxonomy.
    String labels are encoded internally; model.predict returns ints, decoded
    via model.policy_labels_."""
    from fusion.two_stage import _make_model

    X = np.asarray(X, dtype=float)
    labs = tuple(sorted(set(np.asarray(best_op).tolist())))
    enc = {lab: i for i, lab in enumerate(labs)}
    model = _make_model(kind)
    model.fit(X, np.array([enc[v] for v in best_op], dtype=int))
    model.policy_labels_ = labs
    return model


def compare(y_hall: dict[str, int], outcomes: dict, X: dict[str, np.ndarray] | None = None,
            pred_mech: dict[str, str] | None = None,
            detect_flag: dict[str, int] | None = None,
            claim_type: dict[str, str] | None = None,
            seed: int = 0) -> dict[str, dict]:
    """Full §7 table from recorded outcomes. X/features only needed for the
    outcome-policy row (cross-validated); everything else is arithmetic."""
    cids = list(y_hall)
    table: dict[str, dict] = {}
    table["no_repair"] = {"fixed": 0, "broken": 0, "net": 0}

    for op in OPERATORS:
        table[f"uniform_{op}"] = net_gain_of({c: op for c in cids}, outcomes, y_hall)

    # all three applied to all claims: per-claim best fix, worst breakage.
    fixed = sum(1 for c in cids if y_hall[c] and any(outcomes[c][o][0] for o in OPERATORS))
    broken = sum(1 for c in cids if not y_hall[c] and any(outcomes[c][o][0] or outcomes[c][o][1] for o in OPERATORS))
    table["all_ops"] = {"fixed": fixed, "broken": broken, "net": fixed - broken}

    # oracle: best repair per claim, post hoc (bounds headroom, no taxonomy).
    of = sum(1 for c in cids if y_hall[c] and any(outcomes[c][o][0] for o in OPERATORS))
    table["oracle"] = {"fixed": of, "broken": 0, "net": of}

    rng = np.random.default_rng(seed)
    table["random_route"] = net_gain_of(
        {c: OPERATORS[int(rng.integers(0, len(OPERATORS)))] for c in cids},
        outcomes, y_hall)

    if detect_flag is not None:
        # fixed best-uniform op on flagged claims: find best uniform row first.
        best = max([f"uniform_{o}" for o in OPERATORS],
                   key=lambda k: table[k]["net"])
        op = best.split("_", 1)[1]
        table["detect_then_best_uniform"] = net_gain_of(
            {c: (op if detect_flag.get(c, 0) else None) for c in cids}, outcomes, y_hall)

    if claim_type is not None:
        # null taxonomy: majority-best op per claim type (observed, post hoc).
        from collections import Counter

        type_op: dict[str, str] = {}
        for t in set(claim_type.values()):
            votes = Counter()
            for c in cids:
                if claim_type[c] == t and y_hall[c]:
                    for o in OPERATORS:
                        if outcomes[c][o][0]:
                            votes[o] += 1
            type_op[t] = votes.most_common(1)[0][0] if votes else "abstain"
        table["claimtype_route"] = net_gain_of(
            {c: type_op[claim_type[c]] for c in cids}, outcomes, y_hall)

    if pred_mech is not None:
        table["mechanism_routed"] = net_gain_of(
            {c: MECH_TO_OP.get(pred_mech.get(c, "M0")) for c in cids}, outcomes, y_hall)
        table["routed_no_abstain"] = net_gain_of(
            {c: ({"M3": "zoom"}.get(pred_mech.get(c, "M0"),
                                    MECH_TO_OP.get(pred_mech.get(c, "M0"))))
             for c in cids}, outcomes, y_hall)

    if X is not None:
        # outcome policy, 5-fold cross-validated (honest: never evaluated in-sample).
        from sklearn.model_selection import StratifiedKFold

        Xs = np.array([X[c] for c in cids])
        best = []
        for c in cids:
            if not y_hall[c]:
                best.append("none")
                continue
            hit = [o for o in OPERATORS if outcomes[c][o][0]]
            best.append(sorted(hit)[0] if hit else "none")
        best = np.array(best)
        pred = np.full(len(cids), "none", dtype=object)
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        try:
            for tr, te in skf.split(Xs, best):
                m = train_outcome_policy(Xs[tr], best[tr])
                pred[te] = np.array(m.policy_labels_)[m.predict(Xs[te])]
        except Exception as exc:
            logger.warning("outcome-policy CV failed: %s", exc)
            pred = np.full(len(cids), "abstain", dtype=object)
        table["outcome_policy"] = net_gain_of(
            {c: (None if p == "none" else p) for c, p in zip(cids, pred)}, outcomes, y_hall)

    return table
