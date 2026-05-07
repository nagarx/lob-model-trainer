"""Post-experiment analysis utilities for trained models.

Per D-γ Pure-Function Analysis Layer pattern (preserved in MEMORY
POST-COMPACT-3 from prior session's Agent 4 attack 4 architectural
recommendation): post-experiment analysis lives in
``lobtrainer.analysis.*`` — NOT in ``scripts/`` (per hft-rules §4
"every experiment is a manifest" applies to research experiments;
post-experiment statistical analysis is utility code, lives in library).

This package is intentionally torch-free where possible — analysis
runs on stored signal arrays (NPY) without re-loading model
checkpoints. Per Round 2 Agent G #PY-67 mitigation: avoiding
checkpoint reload sidesteps stale ``compatibility_fingerprint`` issues
on R9-R14 (pre-Phase-C.1 stale horizons).

Subpackages:
  - ``stat_rigor``: Phase 2 STAT RIGOR FLOOR primitives (Cyclelet B):
    * ``ci``: P2.A — bootstrap CI on test metrics from stored signals
    * (future) ``importance``: P2.B — permutation importance
    * (future) ``pairwise``: P2.C — pairwise statistical comparison
"""
