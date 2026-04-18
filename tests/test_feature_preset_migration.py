"""Phase 7 Stage 7.1 migration parity tests.

These tests lock the hand-curated FeatureSet artifacts
(``contracts/feature_sets/nvda_short_term_40_src{116,128}_v1.json`` +
``nvda_analysis_ready_119_src128_v1.json``) to the trainer's authoritative
preset definitions in ``lobtrainer.constants.feature_presets``.

**Regression guard**: if a future editor accidentally changes a preset
(e.g., adds/removes an index in ``PRESET_SHORT_TERM_40``) WITHOUT
regenerating the FeatureSet JSON, this test will fail. The failure message
points at the migration script + the two sources that diverged.

**Cross-venv boundary**: this test runs in the trainer venv which has
``lobtrainer.constants.feature_presets`` directly importable (no sys-modules
stub required) AND can read the JSON via stdlib json (no hft-ops writer
needed for read). That's the natural home for parity.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    from lobtrainer.constants.feature_presets import (
        PRESET_ANALYSIS_READY_128,
        PRESET_SHORT_TERM_40,
    )
except DeprecationWarning:
    # Post-2026-06-15 PendingDeprecationWarning escalation may raise rather
    # than warn — use pytest.importorskip as a safety net.
    pytest.importorskip("lobtrainer.constants.feature_presets")


# Monorepo root is 2 levels up from this test file.
_TEST_DIR = Path(__file__).resolve().parent
_MONOREPO_ROOT = _TEST_DIR.parent.parent
_REGISTRY_DIR = _MONOREPO_ROOT / "contracts" / "feature_sets"


def _load_feature_set(name: str) -> dict:
    """Read a FeatureSet JSON from the registry."""
    path = _REGISTRY_DIR / f"{name}.json"
    if not path.exists():
        pytest.skip(
            f"FeatureSet {name!r} not found at {path}. "
            f"Run `hft-ops/scripts/migrate_feature_presets_to_registry.py` "
            f"to regenerate."
        )
    with open(path) as f:
        return json.load(f)


class TestShortTerm40ParityWithTrainerPreset:
    """Both SFC-variant FeatureSets must mirror PRESET_SHORT_TERM_40 exactly."""

    def test_src128_indices_match_preset(self):
        fs = _load_feature_set("nvda_short_term_40_src128_v1")
        expected = sorted({int(i) for i in PRESET_SHORT_TERM_40})
        actual = sorted({int(i) for i in fs["feature_indices"]})
        assert actual == expected, (
            f"nvda_short_term_40_src128_v1 indices diverged from "
            f"PRESET_SHORT_TERM_40:\n"
            f"  expected (trainer): {expected}\n"
            f"  actual   (registry): {actual}\n"
            f"Regenerate via "
            f"`hft-ops/scripts/migrate_feature_presets_to_registry.py`."
        )

    def test_src116_indices_match_preset(self):
        fs = _load_feature_set("nvda_short_term_40_src116_v1")
        expected = sorted({int(i) for i in PRESET_SHORT_TERM_40})
        actual = sorted({int(i) for i in fs["feature_indices"]})
        assert actual == expected

    def test_src128_sfc_is_128(self):
        fs = _load_feature_set("nvda_short_term_40_src128_v1")
        assert fs["source_feature_count"] == 128

    def test_src116_sfc_is_116(self):
        fs = _load_feature_set("nvda_short_term_40_src116_v1")
        assert fs["source_feature_count"] == 116

    def test_src128_src116_share_indices_differ_hash(self):
        """Same indices + different SFC → different content_hash.

        Content-hash includes (sorted_unique_indices, source_feature_count,
        contract_version). Indices are identical, SFC differs → hash MUST
        differ. This regression guard catches SFC accidentally being
        stripped from the hash input.
        """
        fs128 = _load_feature_set("nvda_short_term_40_src128_v1")
        fs116 = _load_feature_set("nvda_short_term_40_src116_v1")
        assert fs128["feature_indices"] == fs116["feature_indices"], (
            "SFC-variant FeatureSets must share indices"
        )
        assert fs128["content_hash"] != fs116["content_hash"], (
            "Content-hash must differ when SFC differs (else SFC is "
            "silently excluded from hash input — contract breach)"
        )


class TestAnalysisReady119ParityWithTrainerPreset:
    """nvda_analysis_ready_119_src128_v1 must mirror PRESET_ANALYSIS_READY_128 exactly."""

    def test_indices_match_preset(self):
        fs = _load_feature_set("nvda_analysis_ready_119_src128_v1")
        expected = sorted({int(i) for i in PRESET_ANALYSIS_READY_128})
        actual = sorted({int(i) for i in fs["feature_indices"]})
        assert actual == expected

    def test_count_is_119(self):
        fs = _load_feature_set("nvda_analysis_ready_119_src128_v1")
        assert len(fs["feature_indices"]) == 119

    def test_dead_features_excluded(self):
        """The 9 dead/constant features must NOT appear in indices."""
        fs = _load_feature_set("nvda_analysis_ready_119_src128_v1")
        dead = {68, 69, 76, 77, 92, 94, 96, 97, 102}
        indices = set(fs["feature_indices"])
        overlap = indices & dead
        assert not overlap, (
            f"Dead features leaked into analysis_ready_119: {sorted(overlap)}"
        )

    def test_sfc_is_128(self):
        fs = _load_feature_set("nvda_analysis_ready_119_src128_v1")
        assert fs["source_feature_count"] == 128


class TestAllFeatureSetsResolve:
    """Each FeatureSet must resolve via the trainer's resolve_feature_set."""

    def _resolver(self):
        from lobtrainer.data.feature_set_resolver import resolve_feature_set

        return resolve_feature_set

    def test_short_term_40_src128_resolves(self):
        fs = self._resolver()(
            "nvda_short_term_40_src128_v1",
            registry_dir=_REGISTRY_DIR,
            expected_contract_version="2.2",
            expected_source_feature_count=128,
        )
        assert len(fs.feature_indices) == 40

    def test_short_term_40_src116_resolves(self):
        fs = self._resolver()(
            "nvda_short_term_40_src116_v1",
            registry_dir=_REGISTRY_DIR,
            expected_contract_version="2.2",
            expected_source_feature_count=116,
        )
        assert len(fs.feature_indices) == 40

    def test_analysis_ready_119_src128_resolves(self):
        fs = self._resolver()(
            "nvda_analysis_ready_119_src128_v1",
            registry_dir=_REGISTRY_DIR,
            expected_contract_version="2.2",
            expected_source_feature_count=128,
        )
        assert len(fs.feature_indices) == 119

    def test_sfc_mismatch_rejected(self):
        """Trainer configured for SFC=98 must NOT accept SFC=128 FeatureSet."""
        from lobtrainer.data.feature_set_resolver import (
            FeatureSetContractMismatch,
        )

        with pytest.raises(FeatureSetContractMismatch, match="source_feature_count"):
            self._resolver()(
                "nvda_analysis_ready_119_src128_v1",
                registry_dir=_REGISTRY_DIR,
                expected_contract_version="2.2",
                expected_source_feature_count=98,  # deliberate mismatch
            )


class TestContentHashDeterminism:
    """Re-running the migration script must produce identical content_hash.

    Phase 4 PRODUCT-only hash means the hash depends ONLY on
    (sorted_unique_indices, source_feature_count, contract_version).
    Provenance fields (tool, config_path, created_at, criteria) MUST NOT
    affect the hash. This test locks that invariant.
    """

    def test_hash_matches_recomputed_product(self):
        from hft_contracts.feature_sets.hashing import compute_feature_set_hash

        fs = _load_feature_set("nvda_analysis_ready_119_src128_v1")
        recomputed = compute_feature_set_hash(
            feature_indices=fs["feature_indices"],
            source_feature_count=fs["source_feature_count"],
            contract_version=fs["contract_version"],
        )
        assert fs["content_hash"] == recomputed, (
            f"Stored content_hash {fs['content_hash']!r} disagrees with "
            f"recomputed {recomputed!r}. Phase 4 integrity contract breach."
        )
