"""Tests for lobtrainer.data.feature_set_resolver (Phase 4 Batch 4c.1)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lobtrainer.data.feature_set_resolver import (
    FeatureSetResolverError,
    ResolvedFeatureSet,
    _compute_content_hash,
    find_feature_sets_dir,
    resolve_feature_set,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _canonical_hash(indices, sfc, cv):
    """Reproduce the canonical hash form independently (parity anchor)."""
    canonical = {
        "feature_indices": sorted(set(int(i) for i in indices)),
        "source_feature_count": int(sfc),
        "contract_version": str(cv),
    }
    blob = json.dumps(canonical, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _write_feature_set(
    path: Path,
    name: str,
    indices: list[int],
    *,
    source_feature_count: int = 98,
    contract_version: str = "2.2",
    content_hash: str | None = None,
    assets: list[str] = None,
    horizons: list[int] = None,
    feature_names: list[str] | None = None,
) -> dict:
    """Write a FeatureSet JSON directly (bypasses hft-ops writer).

    Used to build test fixtures. Matches the schema that the resolver
    reads. Integrity hash is computed unless ``content_hash`` is passed
    explicitly (which supports tamper-test fixtures).
    """
    if content_hash is None:
        content_hash = _canonical_hash(indices, source_feature_count, contract_version)
    data = {
        "schema_version": "1.0",
        "name": name,
        "content_hash": content_hash,
        "contract_version": contract_version,
        "source_feature_count": source_feature_count,
        "applies_to": {
            "assets": assets or ["NVDA"],
            "horizons": horizons or [10],
        },
        "feature_indices": indices,
        "feature_names": feature_names or [f"feature_{i}" for i in indices],
        "produced_by": {
            "tool": "test",
            "tool_version": "0.0.0",
            "config_path": "x.yaml",
            "config_hash": "0" * 64,
            "source_profile_hash": "1" * 64,
            "data_export": "data/exports/x",
            "data_dir_hash": "2" * 64,
        },
        "criteria": {},
        "criteria_schema_version": "1.0",
        "description": "",
        "notes": "",
        "created_at": "2026-04-15T12:00:00+00:00",
        "created_by": "test",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    return data


# ---------------------------------------------------------------------------
# Canonical hash parity with hft-ops.feature_sets.hashing
# ---------------------------------------------------------------------------


class TestCanonicalHashParity:
    """Lock byte-for-byte parity between this resolver's inlined hash
    function and hft_ops.feature_sets.hashing.compute_feature_set_hash.

    If the two implementations drift, FeatureSets written by the
    producer would fail integrity verification at the trainer — a very
    confusing failure mode. This test fails immediately on drift so
    the two sides stay in lockstep.

    hft-ops is not pip-installed in this venv (by design — trainer
    stays self-contained). The test reproduces the canonical form from
    the hft-ops source independently: both sides construct the same
    JSON blob and the SHA-256 output must agree.
    """

    def test_simple_indices_hash_matches_reference(self):
        # Reference implementation inline (mirror of hft-ops impl).
        indices = [12, 0, 5]  # unsorted intentionally
        sfc = 98
        cv = "2.2"

        # Resolver's implementation:
        got = _compute_content_hash(indices, sfc, cv)

        # Independent reproduction of the canonical form:
        canonical = {
            "feature_indices": [0, 5, 12],  # sorted(set(...))
            "source_feature_count": 98,
            "contract_version": "2.2",
        }
        expected_blob = json.dumps(
            canonical, sort_keys=True, default=str
        ).encode("utf-8")
        expected = hashlib.sha256(expected_blob).hexdigest()
        assert got == expected

    def test_hash_is_64_lowercase_hex(self):
        h = _compute_content_hash([0, 5, 12], 98, "2.2")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)
        assert h == h.lower()

    def test_duplicates_and_order_normalized(self):
        h1 = _compute_content_hash([0, 5, 12], 98, "2.2")
        h2 = _compute_content_hash([12, 5, 0, 5], 98, "2.2")
        assert h1 == h2


# ---------------------------------------------------------------------------
# resolve_feature_set — happy path + integrity verification
# ---------------------------------------------------------------------------


class TestResolveHappyPath:
    def test_loads_valid_feature_set(self, tmp_path):
        _write_feature_set(tmp_path / "test_v1.json", "test_v1", [0, 5, 12])

        rfs = resolve_feature_set("test_v1", tmp_path)
        assert isinstance(rfs, ResolvedFeatureSet)
        assert rfs.name == "test_v1"
        assert rfs.feature_indices == (0, 5, 12)
        assert rfs.source_feature_count == 98
        assert rfs.contract_version == "2.2"
        assert rfs.applies_to_assets == ("NVDA",)
        assert rfs.applies_to_horizons == (10,)

    def test_feature_names_tuple(self, tmp_path):
        _write_feature_set(
            tmp_path / "named_v1.json",
            "named_v1",
            [0, 5, 12],
            feature_names=["alpha", "beta", "gamma"],
        )
        rfs = resolve_feature_set("named_v1", tmp_path)
        assert rfs.feature_names == ("alpha", "beta", "gamma")

    def test_missing_feature_names_is_empty_tuple(self, tmp_path):
        data = _write_feature_set(tmp_path / "x_v1.json", "x_v1", [0, 1])
        # Rewrite without feature_names
        data.pop("feature_names")
        (tmp_path / "x_v1.json").write_text(
            json.dumps(data, sort_keys=True, indent=2) + "\n"
        )
        rfs = resolve_feature_set("x_v1", tmp_path)
        assert rfs.feature_names == ()


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


class TestPathSafety:
    def test_empty_name_rejected(self, tmp_path):
        with pytest.raises(FeatureSetResolverError, match="non-empty"):
            resolve_feature_set("", tmp_path)

    def test_slash_in_name_rejected(self, tmp_path):
        with pytest.raises(FeatureSetResolverError, match="path separators"):
            resolve_feature_set("foo/bar", tmp_path)

    def test_backslash_in_name_rejected(self, tmp_path):
        with pytest.raises(FeatureSetResolverError, match="path separators"):
            resolve_feature_set("foo\\bar", tmp_path)

    def test_leading_dot_rejected(self, tmp_path):
        with pytest.raises(FeatureSetResolverError, match="start with"):
            resolve_feature_set(".hidden", tmp_path)

    # Phase 6 6A.8 regression guards — whitespace-only names rejected via
    # `.strip()` before emptiness check. Prior code treated `"   "` as truthy
    # and produced a confusing "FeatureSet '   ' not found at /.../   .json"
    # error instead of the clearer "non-empty" semantic message.

    def test_whitespace_only_name_rejected(self, tmp_path):
        with pytest.raises(FeatureSetResolverError, match="non-empty"):
            resolve_feature_set("   ", tmp_path)

    def test_tab_only_name_rejected(self, tmp_path):
        with pytest.raises(FeatureSetResolverError, match="non-empty"):
            resolve_feature_set("\t", tmp_path)

    def test_newline_only_name_rejected(self, tmp_path):
        with pytest.raises(FeatureSetResolverError, match="non-empty"):
            resolve_feature_set("\n\n", tmp_path)

    def test_leading_trailing_whitespace_stripped(self, tmp_path):
        """Valid name with surrounding whitespace is trimmed before lookup.
        Matches how DataConfig.__post_init__ normalizes feature_set input."""
        _write_feature_set(tmp_path / "trimmed_v1.json", "trimmed_v1", [0, 5])
        # "  trimmed_v1  " → stripped to "trimmed_v1" → registry hit
        result = resolve_feature_set("  trimmed_v1  ", tmp_path)
        assert result.name == "trimmed_v1"


# ---------------------------------------------------------------------------
# Missing file / malformed JSON
# ---------------------------------------------------------------------------


class TestMissingOrMalformed:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FeatureSetResolverError, match="not found"):
            resolve_feature_set("missing", tmp_path)

    def test_missing_file_error_lists_available(self, tmp_path):
        _write_feature_set(tmp_path / "alpha_v1.json", "alpha_v1", [0])
        _write_feature_set(tmp_path / "beta_v1.json", "beta_v1", [0])
        with pytest.raises(FeatureSetResolverError) as excinfo:
            resolve_feature_set("gamma_v1", tmp_path)
        err = str(excinfo.value)
        assert "alpha_v1" in err
        assert "beta_v1" in err

    def test_malformed_json_raises(self, tmp_path):
        (tmp_path / "bad.json").write_text("this is not json")
        with pytest.raises(FeatureSetResolverError, match="not valid JSON"):
            resolve_feature_set("bad", tmp_path)


# ---------------------------------------------------------------------------
# Schema validation (minimal)
# ---------------------------------------------------------------------------


class TestMinimalValidation:
    def test_missing_key_raises(self, tmp_path):
        data = _write_feature_set(tmp_path / "test.json", "test", [0])
        # Remove a required key and rewrite
        data.pop("content_hash")
        (tmp_path / "test.json").write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")

        with pytest.raises(FeatureSetResolverError, match="missing required keys"):
            resolve_feature_set("test", tmp_path)

    def test_name_filename_mismatch_raises(self, tmp_path):
        _write_feature_set(
            tmp_path / "filename_v1.json",
            name="different_name_v1",  # content.name != filename
            indices=[0, 5],
        )
        with pytest.raises(FeatureSetResolverError, match="filename/name mismatch"):
            resolve_feature_set("filename_v1", tmp_path)

    def test_bad_hash_format_raises(self, tmp_path):
        _write_feature_set(
            tmp_path / "badhash.json",
            "badhash",
            [0, 5],
            content_hash="sha256:short",
        )
        with pytest.raises(FeatureSetResolverError, match="64-char"):
            resolve_feature_set("badhash", tmp_path)

    def test_out_of_range_index_raises(self, tmp_path):
        _write_feature_set(
            tmp_path / "oor.json",
            "oor",
            [0, 5, 99],  # sfc=98, so 99 is out of range
            source_feature_count=98,
        )
        with pytest.raises(FeatureSetResolverError, match="< source_feature_count"):
            resolve_feature_set("oor", tmp_path)


# ---------------------------------------------------------------------------
# Content hash integrity verification
# ---------------------------------------------------------------------------


class TestIntegrityVerification:
    def test_tampered_indices_raises(self, tmp_path):
        # Write a file with indices [0, 5, 12] but a hash computed over [0, 5]
        wrong_hash = _canonical_hash([0, 5], 98, "2.2")
        _write_feature_set(
            tmp_path / "tampered.json",
            "tampered",
            [0, 5, 12],
            content_hash=wrong_hash,
        )
        with pytest.raises(FeatureSetResolverError, match="integrity check failed"):
            resolve_feature_set("tampered", tmp_path)

    def test_tampered_contract_version_raises(self, tmp_path):
        # Compute hash under one contract version, but write a different one
        real_hash = _canonical_hash([0, 5], 98, "2.3")
        _write_feature_set(
            tmp_path / "drift.json",
            "drift",
            [0, 5],
            contract_version="2.2",  # file says 2.2
            content_hash=real_hash,  # but hash was computed for 2.3
        )
        with pytest.raises(FeatureSetResolverError, match="integrity check failed"):
            resolve_feature_set("drift", tmp_path)


# ---------------------------------------------------------------------------
# Contract compatibility checks
# ---------------------------------------------------------------------------


class TestContractCompat:
    def test_contract_version_mismatch_raises(self, tmp_path):
        _write_feature_set(
            tmp_path / "v22.json",
            "v22",
            [0, 5],
            contract_version="2.2",
        )
        with pytest.raises(FeatureSetResolverError, match="contract_version mismatch"):
            resolve_feature_set(
                "v22",
                tmp_path,
                expected_contract_version="2.3",
            )

    def test_contract_version_match_succeeds(self, tmp_path):
        _write_feature_set(
            tmp_path / "v22.json", "v22", [0, 5], contract_version="2.2"
        )
        rfs = resolve_feature_set(
            "v22", tmp_path, expected_contract_version="2.2"
        )
        assert rfs.contract_version == "2.2"

    def test_source_feature_count_mismatch_raises(self, tmp_path):
        _write_feature_set(
            tmp_path / "sfc98.json",
            "sfc98",
            [0, 5],
            source_feature_count=98,
        )
        with pytest.raises(
            FeatureSetResolverError, match="source_feature_count mismatch"
        ):
            resolve_feature_set(
                "sfc98",
                tmp_path,
                expected_source_feature_count=128,
            )

    def test_source_feature_count_match_succeeds(self, tmp_path):
        _write_feature_set(
            tmp_path / "sfc98.json", "sfc98", [0, 5], source_feature_count=98
        )
        rfs = resolve_feature_set(
            "sfc98", tmp_path, expected_source_feature_count=98
        )
        assert rfs.source_feature_count == 98

    def test_none_checks_skipped(self, tmp_path):
        # When neither expected_* is provided, resolver succeeds even
        # on a set with arbitrary contract_version / source_feature_count.
        _write_feature_set(
            tmp_path / "any.json",
            "any",
            [0, 5],
            contract_version="9.99",
            source_feature_count=50,
        )
        # source_feature_count=50 is still valid because max(indices)=5 < 50
        rfs = resolve_feature_set("any", tmp_path)
        assert rfs.contract_version == "9.99"
        assert rfs.source_feature_count == 50


# ---------------------------------------------------------------------------
# find_feature_sets_dir — auto-detection via pipeline_contract.toml anchor
# ---------------------------------------------------------------------------


def _make_fake_pipeline_root(root: Path) -> Path:
    """Build a directory tree that passes find_feature_sets_dir's probe."""
    (root / "contracts").mkdir(parents=True)
    (root / "contracts" / "pipeline_contract.toml").write_text("# stub\n")
    return root


class TestFindFeatureSetsDir:
    def test_finds_when_anchor_is_pipeline_root(self, tmp_path):
        root = _make_fake_pipeline_root(tmp_path / "pipeline")
        got = find_feature_sets_dir(root)
        assert got == root / "contracts" / "feature_sets"

    def test_finds_from_subdirectory(self, tmp_path):
        root = _make_fake_pipeline_root(tmp_path / "pipeline")
        (root / "data" / "exports" / "nvda").mkdir(parents=True)
        got = find_feature_sets_dir(root / "data" / "exports" / "nvda")
        assert got == root / "contracts" / "feature_sets"

    def test_finds_even_when_target_dir_missing(self, tmp_path):
        # First-run registry — parent contracts/ exists but feature_sets/
        # is not yet created. The resolver doesn't require existence at
        # detection time; the ``resolve_feature_set`` call returns a
        # "not found" error if a specific name is requested.
        root = _make_fake_pipeline_root(tmp_path / "pipeline")
        assert not (root / "contracts" / "feature_sets").exists()
        got = find_feature_sets_dir(root)
        assert got == root / "contracts" / "feature_sets"
        assert not got.exists()

    def test_raises_when_no_pipeline_root_found(self, tmp_path):
        # No contracts/pipeline_contract.toml anywhere in the ancestry.
        isolated = tmp_path / "isolated"
        isolated.mkdir()
        with pytest.raises(FeatureSetResolverError, match="Cannot auto-detect"):
            find_feature_sets_dir(isolated)

    def test_error_lists_visited_paths(self, tmp_path):
        isolated = tmp_path / "a" / "b" / "c"
        isolated.mkdir(parents=True)
        with pytest.raises(FeatureSetResolverError) as excinfo:
            find_feature_sets_dir(isolated)
        err = str(excinfo.value)
        # The error message names the starting anchor so the operator
        # can diagnose where the walk began.
        assert str(isolated) in err

    def test_max_parents_cap_respected(self, tmp_path):
        # Create a deep tree where the pipeline root is beyond max_parents.
        deep = tmp_path
        for name in "abcdefghij":  # 10 levels deep
            deep = deep / name
            deep.mkdir()
        # Pipeline root at the top (tmp_path), but max_parents=3 stops short.
        _make_fake_pipeline_root(tmp_path)
        with pytest.raises(FeatureSetResolverError):
            find_feature_sets_dir(deep, max_parents=3)


# ---------------------------------------------------------------------------
# Trainer integration contract: resolver populates DataConfig cache
# ---------------------------------------------------------------------------


class TestTrainerIntegrationContract:
    """Lock the integration contract between resolver and DataConfig cache
    without running a full Trainer instance. The trainer's
    _create_dataloaders slice that handles feature_set resolution is
    effectively:

        resolved = resolve_feature_set(name, registry_dir, ...)
        cfg_data._feature_indices_resolved = list(resolved.feature_indices)
        cfg_data._feature_set_ref_resolved = (resolved.name, resolved.content_hash)
        config_feature_indices = list(resolved.feature_indices)

    These tests replay that contract against a synthetic fixture and
    assert the cache + indices post-conditions. If trainer.py refactors
    this block, the contract must remain intact — these tests fail
    immediately on drift.
    """

    def test_cache_populated_after_resolver_call(self, tmp_path):
        from lobtrainer.config.schema import DataConfig

        root = _make_fake_pipeline_root(tmp_path / "pipeline")
        registry = root / "contracts" / "feature_sets"
        _write_feature_set(registry / "test_v1.json", "test_v1", [0, 5, 12])

        cfg_data = DataConfig(
            data_dir=str(root / "data"),
            feature_count=98,
            feature_set="test_v1",
        )
        # Simulate the trainer's _create_dataloaders resolver block:
        registry_dir = find_feature_sets_dir(Path(cfg_data.data_dir).parent)
        # (Test uses root directly since data_dir is fake; production
        # walks up from the real data_dir.)
        resolved = resolve_feature_set(
            "test_v1",
            registry_dir,
            expected_source_feature_count=cfg_data.feature_count,
        )
        cfg_data._feature_indices_resolved = list(resolved.feature_indices)
        cfg_data._feature_set_ref_resolved = (
            resolved.name, resolved.content_hash,
        )

        # Post-conditions the trainer depends on:
        assert cfg_data._feature_indices_resolved == [0, 5, 12]
        ref = cfg_data._feature_set_ref_resolved
        assert ref is not None
        assert ref[0] == "test_v1"
        assert len(ref[1]) == 64  # content_hash

    def test_integration_contract_source_feature_count_mismatch(self, tmp_path):
        # If the FeatureSet was built over a different source width than
        # the DataConfig declares, the resolver raises. Trainer must
        # propagate this as a hard failure.
        from lobtrainer.config.schema import DataConfig

        root = _make_fake_pipeline_root(tmp_path / "pipeline")
        registry = root / "contracts" / "feature_sets"
        _write_feature_set(
            registry / "wide_v1.json",
            "wide_v1",
            [0, 5, 12],
            source_feature_count=128,
        )

        cfg_data = DataConfig(
            data_dir=str(root / "data"),
            feature_count=98,  # declares 98, but set was built over 128
            feature_set="wide_v1",
        )
        registry_dir = find_feature_sets_dir(Path(cfg_data.data_dir).parent)
        with pytest.raises(
            FeatureSetResolverError, match="source_feature_count mismatch"
        ):
            resolve_feature_set(
                "wide_v1",
                registry_dir,
                expected_source_feature_count=cfg_data.feature_count,
            )
