#!/usr/bin/env python3
"""Tests for repair/policy.py — synthetic outcomes where the taxonomy is TRUE
by construction: M1 fixed by contrastive only, M2 by zoom only, M3 hidden by
abstain only. Checks: routed wins, oracle bounds, abstention counted both ways.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from repair.policy import compare, net_gain_of  # noqa: E402

rng = np.random.default_rng(0)


def synth(k: int = 120):
    y, out, mech, X = {}, {}, {}, {}
    mechs = ["M1", "M2", "M3", "M4", "M0"]
    for i in range(k):
        c, m = f"c{i}", mechs[i % 5]
        hall = 0 if m == "M0" else 1
        y[c] = hall
        mech[c] = m
        oc = {"contrastive": (False, False), "zoom": (False, False),
              "abstain": (False, False), "relverify": (False, False)}
        if m == "M1":
            oc["contrastive"] = (True, False)
            oc["zoom"] = (False, True)  # wrong repair damages
        elif m == "M2":
            oc["zoom"] = (True, False)
            oc["contrastive"] = (False, True)
        elif m == "M3":
            oc["abstain"] = (True, False)  # abstain hides it
        elif m == "M4":
            oc["relverify"] = (True, False)  # only isolated re-ask fixes binding
            oc["contrastive"] = (False, True)
        out[c] = oc
        # Features that reveal the mechanism (else nothing can route).
        f = rng.random(4)
        f[0] = 0.9 if m == "M1" else 0.1
        f[1] = 0.9 if m == "M2" else 0.1
        f[2] = 0.9 if m == "M3" else 0.1
        f[3] = 0.9 if m == "M4" else 0.1
        X[c] = f
    return y, out, mech, X


def test_routed_beats_uniform_when_taxonomy_true():
    y, out, mech, X = synth()
    flags = {c: (0 if mech[c] == "M0" else 1) for c in y}
    t = compare(y, out, X=X, pred_mech=mech, detect_flag=flags,
                claim_type={c: "object" for c in y})
    # Routed reaches oracle: every hallucinated claim gets its matched repair.
    assert t["mechanism_routed"]["net"] == 96, t["mechanism_routed"]
    assert t["oracle"]["net"] == 96
    for k in ("uniform_contrastive", "uniform_zoom", "uniform_relverify",
              "claimtype_route"):
        assert t[k]["net"] < t["mechanism_routed"]["net"], (k, t[k])
    # Honest tie, documented in plan section 7: detect-then-abstain matches
    # routing here because abstention is available from risk alone. The
    # mechanism story proves itself in the no-abstain row: matched repairs
    # beat every non-abstain uniform without touching abstention.
    assert t["detect_then_best_uniform"]["net"] == t["mechanism_routed"]["net"]
    assert t["routed_no_abstain"]["net"] == 72, t["routed_no_abstain"]
    assert t["routed_no_abstain"]["net"] > t["uniform_contrastive"]["net"]
    assert t["outcome_policy"]["net"] >= 90, t["outcome_policy"]


def test_abstain_everything_scores_zero_net():
    y = {f"c{i}": i % 2 for i in range(10)}
    out = {c: {o: (False, False) for o in ("contrastive", "zoom", "abstain")} for c in y}
    t = net_gain_of({c: "abstain" for c in y}, out, y)
    assert t["fixed"] == 5 and t["broken"] == 5 and t["net"] == 0, t


def test_outcome_policy_row_runs():
    y, out, mech, X = synth(60)
    t = compare(y, out, X=X)
    assert "outcome_policy" in t and isinstance(t["outcome_policy"]["net"], int)


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
