"""Mechanical enforcement of the §3.4 ownership rule for ``configs/bases/``.

Each top-level dotted-key (e.g., ``model.dropout``, ``train.batch_size``)
is owned by **exactly one axis**. If a field appears in multiple axes'
bases, merges become order-dependent and the ownership invariant breaks.

These tests scan every YAML under ``configs/bases/{models,datasets,labels,train}/``
and assert no cross-axis duplication. They also validate the ``_partial: true``
sentinel convention.

Scope notes:
- The pre-existing ``configs/bases/e5_tlob_regression.yaml`` (monolith) is
  EXCLUDED from this check because it deliberately straddles all axes;
  it is retired at end of Phase 3b.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import pytest
import yaml


_REPO_ROOT = Path(__file__).resolve().parents[1]   # lob-model-trainer/
_BASES_ROOT = _REPO_ROOT / "configs" / "bases"


# Axes recognized by the ownership rule (must match bases/ subdirectory names)
_AXES = ("models", "datasets", "labels", "train")


# Monolith `e5_tlob_regression.yaml` was RETIRED on 2026-04-15 after all 5 E5
# consumers migrated to multi-base (Batch 1a). The `_MONOLITH_BASE` constant
# and `test_monolith_excluded_from_this_check` test were removed as obsolete.


def _flatten_dotted(d: Dict[str, Any], prefix: str = "") -> Iterable[str]:
    """Yield every leaf dotted-key in ``d``.

    Example: ``{"model": {"dropout": 0.1, "tlob_hidden_dim": 32}}`` →
        ``["model.dropout", "model.tlob_hidden_dim"]``.

    Keys starting with ``_`` (e.g., ``_partial``, ``_base``) are skipped —
    those are directives, not config fields.
    """
    for k, v in d.items():
        if isinstance(k, str) and k.startswith("_"):
            continue
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            yield from _flatten_dotted(v, key)
        else:
            yield key


def _load_axis_files(axis: str) -> list[tuple[str, dict]]:
    """Load every base YAML under ``bases/<axis>/``. Returns [(file_name, dict)]."""
    out: list[tuple[str, dict]] = []
    axis_dir = _BASES_ROOT / axis
    if not axis_dir.is_dir():
        return out
    for yp in sorted(axis_dir.glob("*.yaml")):
        with open(yp) as f:
            data = yaml.safe_load(f) or {}
        out.append((yp.name, data))
    return out


# -----------------------------------------------------------------------------
# Axis-ownership invariant
# -----------------------------------------------------------------------------


class TestAxisOwnership:
    def test_no_field_straddles_axes(self):
        """Every dotted-key appears in at most one axis across bases/*/."""
        # key → set of axes where it appears
        ownership: Dict[str, set[str]] = {}

        for axis in _AXES:
            for fname, data in _load_axis_files(axis):
                for key in _flatten_dotted(data):
                    ownership.setdefault(key, set()).add(axis)

        conflicts = {
            k: sorted(axes) for k, axes in ownership.items() if len(axes) > 1
        }

        assert not conflicts, (
            f"§3.4 ownership violation — the following fields are set in "
            f"multiple axes' base files: {conflicts}. Each dotted-key must "
            f"be owned by exactly one axis. See bases/README.md."
        )

    def test_monolith_retired(self):
        """Post-2026-04-15: the monolith `e5_tlob_regression.yaml` is retired.

        Asserts it no longer exists as a top-level base file. If this test
        fails, someone re-introduced the monolith — either delete it again or
        decompose its contents into the 4 axis-partitioned bases.
        """
        monolith = _BASES_ROOT / "e5_tlob_regression.yaml"
        assert not monolith.exists(), (
            f"Monolith {monolith} was retired 2026-04-15 after all 5 E5 "
            f"consumers migrated to multi-base. Do not re-introduce it — "
            f"decompose additions into the 4 axis-partitioned bases."
        )


# -----------------------------------------------------------------------------
# Partial-base sentinel convention
# -----------------------------------------------------------------------------


class TestPartialBaseSentinel:
    @pytest.mark.parametrize(
        "axis,fname,data",
        [
            (axis, fname, data)
            for axis in _AXES
            for fname, data in _load_axis_files(axis)
        ],
        ids=lambda v: v if isinstance(v, str) else "<dict>",
    )
    def test_every_axis_base_declares_partial_true(
        self, axis: str, fname: str, data: Dict[str, Any]
    ):
        """Every base under bases/{models,datasets,labels,train}/ must declare
        ``_partial: true`` so accidental direct loading raises a clear error."""
        assert data.get("_partial") is True, (
            f"bases/{axis}/{fname} must declare `_partial: true` at the top "
            f"level — it's standalone-invalid (only valid when composed with "
            f"peer bases). See bases/README.md."
        )


# -----------------------------------------------------------------------------
# Chained-inheritance purity: `tlob_compact_bare` MUST NOT set cvml fields
# -----------------------------------------------------------------------------
#
# RATIONALE (post-Batch-1 audit item R1/C1):
# `models/tlob_compact_regression.yaml` chains from `models/tlob_compact_bare.yaml`
# and adds `tlob_use_cvml: false` + `tlob_cvml_out_channels: 0` on top. E5/E6
# configs use `tlob_compact_regression.yaml` and depend on those defaults
# being present (their golden fixtures have these fields). E4 TLOB uses
# `tlob_compact_bare.yaml` directly and depends on those fields being ABSENT
# (its golden fixture does not have them).
#
# If a future maintainer accidentally moves the cvml defaults into
# `tlob_compact_bare.yaml` — thinking they belong there as "defaults" — the
# E4 fingerprint silently changes (golden-fixture match fails with a
# mysterious message about the resolved dict diverging). This test locks
# the separation so the regression fails loudly with a clear cause.


class TestChainedInheritancePurity:
    """§3.4 / R1 — lock the bare/regression field split for chained bases."""

    def test_tlob_compact_bare_excludes_cvml_fields(self):
        """`tlob_compact_bare.yaml` MUST NOT set cvml fields — E4 relies on absence."""
        bare_path = _BASES_ROOT / "models" / "tlob_compact_bare.yaml"
        assert bare_path.exists()
        with open(bare_path) as f:
            bare = yaml.safe_load(f) or {}

        model_block = bare.get("model", {}) if isinstance(bare.get("model"), dict) else {}
        for field in ("tlob_use_cvml", "tlob_cvml_out_channels"):
            assert field not in model_block, (
                f"`tlob_compact_bare.yaml` must NOT set `model.{field}` — "
                f"E4 TLOB's golden fixture depends on this field being ABSENT "
                f"from the resolved dict. Adding it here silently changes the "
                f"E4 fingerprint. If you want this default, put it in "
                f"`tlob_compact_regression.yaml` (which chains from bare) "
                f"instead."
            )

    def test_tlob_compact_regression_adds_cvml_fields_only(self):
        """`tlob_compact_regression.yaml` adds ONLY the cvml defaults on top of bare."""
        reg_path = _BASES_ROOT / "models" / "tlob_compact_regression.yaml"
        assert reg_path.exists()
        with open(reg_path) as f:
            reg = yaml.safe_load(f) or {}

        # Must chain from bare
        assert reg.get("_base") == "tlob_compact_bare.yaml", (
            "`tlob_compact_regression.yaml` must chain from `tlob_compact_bare.yaml` "
            "via `_base:`. If chaining is removed, move cvml defaults to every "
            "E5/E6 child explicitly (many files — consider the trade-off)."
        )

        # Must declare _partial
        assert reg.get("_partial") is True

        # Own model fields should ONLY be the cvml defaults (everything else
        # comes from chained bare)
        model_own = reg.get("model", {}) if isinstance(reg.get("model"), dict) else {}
        expected_own = {"tlob_use_cvml", "tlob_cvml_out_channels"}
        unexpected = set(model_own.keys()) - expected_own
        assert not unexpected, (
            f"`tlob_compact_regression.yaml` sets unexpected model fields "
            f"beyond the cvml defaults: {unexpected}. If you're adding "
            f"shared TLOB architecture fields, they belong in "
            f"`tlob_compact_bare.yaml` so E4 (which uses bare directly) "
            f"also inherits them."
        )

    def test_hmhp_cascade_bare_model_type_is_hmhp(self):
        """`hmhp_cascade_bare.yaml` sets model_type=hmhp (classification default).
        Regression variant overrides via `hmhp_cascade_regression.yaml` chain."""
        bare_path = _BASES_ROOT / "models" / "hmhp_cascade_bare.yaml"
        assert bare_path.exists()
        with open(bare_path) as f:
            bare = yaml.safe_load(f) or {}
        assert bare.get("model", {}).get("model_type") == "hmhp", (
            "`hmhp_cascade_bare.yaml` must set model_type=hmhp. Regression "
            "variants (configs 7, 8) override via `hmhp_cascade_regression.yaml` "
            "chain which sets model_type=hmhp_regression."
        )

    def test_hmhp_cascade_bare_excludes_regression_fields(self):
        """`hmhp_cascade_bare.yaml` MUST NOT set regression-specific fields.

        If `hmhp_regression_loss_type: huber` leaked into bare, 9 of 11 HMHP
        configs (classification + TB) would get that field injected into their
        resolved dicts — breaking byte-identity with goldens that omit it.
        """
        bare_path = _BASES_ROOT / "models" / "hmhp_cascade_bare.yaml"
        with open(bare_path) as f:
            bare = yaml.safe_load(f) or {}
        model_block = bare.get("model", {}) if isinstance(bare.get("model"), dict) else {}
        forbidden = {"hmhp_regression_loss_type", "hmhp_use_regression"}
        leaked = forbidden & set(model_block.keys())
        assert not leaked, (
            f"`hmhp_cascade_bare.yaml` must NOT set {leaked} — these are "
            f"regression-variant-specific and only belong in "
            f"`hmhp_cascade_regression.yaml` (via chained inheritance) or "
            f"rare child overrides (config 6 for `hmhp_use_regression: true`)."
        )

    def test_hmhp_cascade_regression_adds_regression_fields_only(self):
        """`hmhp_cascade_regression.yaml` adds only model_type override + regression loss."""
        reg_path = _BASES_ROOT / "models" / "hmhp_cascade_regression.yaml"
        assert reg_path.exists()
        with open(reg_path) as f:
            reg = yaml.safe_load(f) or {}

        assert reg.get("_base") == "hmhp_cascade_bare.yaml", (
            "`hmhp_cascade_regression.yaml` must chain from `hmhp_cascade_bare.yaml`."
        )
        assert reg.get("_partial") is True

        model_own = reg.get("model", {}) if isinstance(reg.get("model"), dict) else {}
        expected_own = {"model_type", "hmhp_regression_loss_type"}
        unexpected = set(model_own.keys()) - expected_own
        assert not unexpected, (
            f"`hmhp_cascade_regression.yaml` sets unexpected fields beyond "
            f"{expected_own}: {unexpected}. Shared HMHP arch fields belong "
            f"in `hmhp_cascade_bare.yaml` (so classification + TB configs "
            f"also inherit them)."
        )

    def test_tlob_paper_classification_excludes_cvml_fields(self):
        """`tlob_paper_classification.yaml` MUST NOT set cvml fields.

        All 7 TLOB paper-classification configs (Batch 3) OMIT
        `tlob_use_cvml` and `tlob_cvml_out_channels` from their resolved
        dicts. If this base injected cvml defaults, all 7 goldens would
        break byte-identity. Same E4-lesson pattern locked by
        `test_tlob_compact_bare_excludes_cvml_fields`.
        """
        path = _BASES_ROOT / "models" / "tlob_paper_classification.yaml"
        assert path.exists()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        model_block = data.get("model", {}) if isinstance(data.get("model"), dict) else {}
        forbidden = {"tlob_use_cvml", "tlob_cvml_out_channels"}
        leaked = forbidden & set(model_block.keys())
        assert not leaked, (
            f"`tlob_paper_classification.yaml` must NOT set {leaked} — "
            f"all 7 Batch-3 classification configs rely on these fields "
            f"being ABSENT from the resolved dict (dataclass defaults apply "
            f"at load time). Leaking cvml defaults corrupts all goldens."
        )

    def test_tlob_paper_classification_excludes_regression_fields(self):
        """`tlob_paper_classification.yaml` MUST NOT set regression-specific fields.

        This is a classification base. Regression fields (regression_loss_type,
        regression_loss_delta, model.task_type: regression) belong in
        `tlob_compact_regression.yaml` or future regression variants.
        """
        path = _BASES_ROOT / "models" / "tlob_paper_classification.yaml"
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        model_block = data.get("model", {}) if isinstance(data.get("model"), dict) else {}
        forbidden = {"regression_loss_type", "regression_loss_delta", "task_type"}
        leaked = forbidden & set(model_block.keys())
        assert not leaked, (
            f"`tlob_paper_classification.yaml` must NOT set {leaked} — "
            f"this is a classification base. Regression fields belong in "
            f"a dedicated regression base."
        )

    def test_triple_barrier_label_omits_horizon_idx(self):
        """`triple_barrier_volscaled.yaml` must NOT set `data.horizon_idx`.

        The 3 TB configs' golden fixtures OMIT horizon_idx entirely (TB uses
        `hmhp_horizons: [50,100,200]` instead of a single primary horizon).
        If this base sets horizon_idx=0, TB resolved dicts would contain it
        and break byte-identity.
        """
        label_path = _BASES_ROOT / "labels" / "triple_barrier_volscaled.yaml"
        assert label_path.exists()
        with open(label_path) as f:
            label = yaml.safe_load(f) or {}
        data_block = label.get("data", {}) if isinstance(label.get("data"), dict) else {}
        assert "horizon_idx" not in data_block, (
            "`triple_barrier_volscaled.yaml` must NOT set `data.horizon_idx`. "
            "TB golden fixtures omit this field (spec is `hmhp_horizons`). "
            "Setting it would corrupt byte-identity for all 3 TB configs."
        )
