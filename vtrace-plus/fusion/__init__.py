"""Deterministic, rule-based V-TRACE+ fusion."""

from .vtrace_fusion import (  # noqa: F401
    ALL_SIGNALS,
    FusionResult,
    SignalBundle,
    aggregate,
    compute_ced,
    compute_disagreement,
    fuse,
    to_risk,
)
