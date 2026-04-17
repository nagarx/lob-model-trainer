"""End-to-end validation for migrated multi-base experiment configs.

The pre-existing ``test_merge_v1_parity.py`` verifies that v2 ``merge.py``
produces the same RESOLVED DICT as v1 for every in-scope config (byte-
identical golden-fixture match). This file goes one step further: it
asserts that every migrated config actually LOADS via
``ExperimentConfig.from_yaml()`` — i.e., the resolved dict is dacite-
compatible, post-init validation passes, and the config is truly runnable.

Gap rationale (from post-Batch-1 coverage audit item G1): parity tests
only compare pre-dacite resolved dicts. Dacite.from_dict could reject
migrated configs in ways the parity test can't catch (e.g., missing
required field introduced by a base refactor).

Auto-discovery: scans ``configs/experiments/*.yaml`` for configs that use
``_base:`` as a list (v2 migrated form). Parametrizes over each.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from lobtrainer.config import ExperimentConfig


_REPO_ROOT = Path(__file__).resolve().parents[1]   # lob-model-trainer/
_EXPERIMENTS_ROOT = _REPO_ROOT / "configs" / "experiments"


def _is_migrated_multibase(yaml_path: Path) -> bool:
    """True if the YAML uses v2 list-form ``_base:`` (Phase 3 migrated form)."""
    try:
        with open(yaml_path) as f:
            raw = yaml.safe_load(f) or {}
    except Exception:
        return False
    if not isinstance(raw, dict):
        return False
    return isinstance(raw.get("_base"), list)


def _collect_migrated_configs() -> list[tuple[str, Path]]:
    """Collect every migrated multi-base config."""
    out = []
    for yp in sorted(_EXPERIMENTS_ROOT.glob("*.yaml")):
        if _is_migrated_multibase(yp):
            out.append((yp.name, yp))
    return out


_MIGRATED = _collect_migrated_configs()


@pytest.mark.parametrize(
    "fname,yaml_path",
    _MIGRATED,
    ids=[k for k, _ in _MIGRATED],
)
class TestMigratedConfigsInstantiate:
    """Every migrated config must successfully construct an ExperimentConfig.

    This guards against the class of bugs where parity (byte-identical
    resolved dict) passes but dacite rejects the result — e.g., a base
    refactor that produces a structurally valid dict but fails a
    ``__post_init__`` cross-validation.
    """

    def test_from_yaml_succeeds(self, fname: str, yaml_path: Path):
        """`ExperimentConfig.from_yaml(migrated_yaml)` returns a valid config."""
        try:
            cfg = ExperimentConfig.from_yaml(str(yaml_path))
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"ExperimentConfig.from_yaml({fname}) raised "
                f"{type(exc).__name__}: {exc}\n"
                f"This migrated config produces a dict that dacite cannot "
                f"construct — parity test would have missed this."
            )
        assert cfg.name, f"{fname} resolved with empty name"

    def test_resolved_fields_are_concrete(self, fname: str, yaml_path: Path):
        """Spot-check that key fields resolved through the bases are present.

        Catches regressions where a base is deleted/renamed and a child
        silently loses its required fields.
        """
        cfg = ExperimentConfig.from_yaml(str(yaml_path))
        # Common post-Batch-1 fields (present in all E-family migrated configs)
        assert cfg.data.data_dir, f"{fname} has empty data_dir"
        assert cfg.data.feature_count > 0, (
            f"{fname} has feature_count={cfg.data.feature_count}; "
            f"check datasets/ base inheritance"
        )
        assert cfg.model.model_type, f"{fname} has empty model_type"
        assert cfg.train.batch_size > 0, (
            f"{fname} has batch_size={cfg.train.batch_size}; "
            f"check train/ base inheritance"
        )


class TestMigratedConfigsCount:
    """Meta-tests on the migration scope."""

    def test_at_least_batch_1_is_migrated(self):
        """After Batch 1 (2026-04-15), at least 7 configs should be migrated."""
        assert len(_MIGRATED) >= 7, (
            f"Expected ≥7 migrated multi-base configs (post-Batch-1); "
            f"found {len(_MIGRATED)}: {[f for f, _ in _MIGRATED]}"
        )

    def test_all_e_family_migrated(self):
        """Batch 1 scope: E4 TLOB + E5×5 + E6 — 7 configs."""
        expected = {
            "e4_tlob_h60.yaml",
            "e5_60s_huber_cvml.yaml",
            "e5_60s_huber_nocvml.yaml",
            "e5_60s_gmadl_cvml.yaml",
            "e5_30s_huber_cvml.yaml",
            "e5_30s_huber_nocvml.yaml",
            "e6_calibrated_conviction.yaml",
        }
        migrated = {f for f, _ in _MIGRATED}
        missing = expected - migrated
        assert not missing, f"E-family configs not migrated: {missing}"

    def test_all_hmhp_family_migrated(self):
        """Batch 2 scope: 11 HMHP configs (3 TB + 3 128feat classif +
        3 40feat classif + 2 regression). The 2 niche HMHP configs
        (multihorizon_v1, short_term_hmhp_v1) remain standalone by design."""
        expected = {
            # Triple barrier (3)
            "nvda_hmhp_triple_barrier_v1.yaml",
            "nvda_hmhp_triple_barrier_volscaled.yaml",
            "nvda_hmhp_triple_barrier_calibrated.yaml",
            # 128feat classification (3)
            "nvda_hmhp_128feat_h10_primary.yaml",
            "nvda_hmhp_128feat_arcx_h10.yaml",
            "nvda_hmhp_128feat_opportunity_h10.yaml",
            # 40feat classification (3)
            "nvda_hmhp_40feat_h10.yaml",
            "nvda_hmhp_40feat_h60_profit8bps.yaml",
            "nvda_hmhp_40feat_h60_profit8bps_regression.yaml",
            # Regression (2)
            "nvda_hmhp_regression_h10_primary.yaml",
            "nvda_hmhp_regressor_h60.yaml",
        }
        migrated = {f for f, _ in _MIGRATED}
        missing = expected - migrated
        assert not missing, f"HMHP-family configs not migrated: {missing}"

    def test_all_tlob_classif_family_migrated(self):
        """Batch 3 scope: 7 TLOB paper-spec classification configs.

        These are the DeepLOB-paper-horizon benchmarks (event-based,
        hidden=64/layers=4/heads=1). Distinct from E5 TLOB regression
        (time-based, hidden=32/layers=2/heads=2).
        """
        expected = {
            "nvda_tlob_h10_v1.yaml",
            "nvda_tlob_h50_v1.yaml",
            "nvda_tlob_h100_v1.yaml",
            "nvda_tlob_raw_h50_v1.yaml",
            "nvda_tlob_98feat_h100.yaml",
            "nvda_tlob_repo_match_h50.yaml",
            "nvda_tlob_v2_h100.yaml",
        }
        migrated = {f for f, _ in _MIGRATED}
        missing = expected - migrated
        assert not missing, f"TLOB classification configs not migrated: {missing}"

    def test_niche_hmhp_configs_excluded_as_designed(self):
        """The 2 niche HMHP configs MUST remain standalone (design decision)."""
        niche = {
            "nvda_hmhp_multihorizon_v1.yaml",
            "nvda_short_term_hmhp_v1.yaml",
        }
        migrated = {f for f, _ in _MIGRATED}
        accidentally_migrated = niche & migrated
        assert not accidentally_migrated, (
            f"Niche HMHP configs accidentally migrated to multi-base: "
            f"{accidentally_migrated}. These should remain standalone "
            f"per §3.6 (too niche for shared-base composition)."
        )
