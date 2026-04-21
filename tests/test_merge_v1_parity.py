"""v1 ↔ v2 merge.py parity test — Phase 3 non-negotiable acceptance criterion.

For every current config (~41 fixtures), load it through BOTH the archived v1
``merge.py`` and the new v2 ``merge.py`` and assert byte-identical results
(status + resolved-dict OR exc_type + message).

Loading the archive: since the archive deliberately has no ``__init__.py``,
Python cannot import it as a package. We use ``importlib.util.spec_from_file_location``
to load the archived module from its path.

This test locks the v1 → v2 migration bit-for-bit. If it passes, every
non-migrated config produces the same effective resolved dict under v2 as
it did under v1 — meaning researchers see ZERO behavior change, ledger
fingerprints stay identical, and downstream tests don't need to be updated.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml


# -----------------------------------------------------------------------------
# Paths (resolved once at module import)
# -----------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]  # lob-model-trainer/
_CONFIGS_ROOT = _REPO_ROOT / "configs"
_GOLDEN_ROOT = _REPO_ROOT / "tests" / "fixtures" / "golden"
_ARCHIVE_MERGE_V1 = (
    _REPO_ROOT
    / "src"
    / "lobtrainer"
    / "config"
    / "archive"
    / "merge-v1"
    / "merge.py"
)


# -----------------------------------------------------------------------------
# Archive loader — uses spec_from_file_location because the archive has no
# __init__.py by design (matches monolith-v1 precedent).
# -----------------------------------------------------------------------------


def _load_archive_v1():
    """Load the archived v1 merge.py module via importlib.util.

    The archive directory deliberately has no ``__init__.py`` so it is not
    importable as a package. We load it directly from its file path so the
    parity test can exercise both implementations side-by-side.
    """
    assert _ARCHIVE_MERGE_V1.exists(), (
        f"Archive merge.py not found at {_ARCHIVE_MERGE_V1}. "
        "Phase 3.1 (archive) may not have been completed."
    )
    spec = importlib.util.spec_from_file_location(
        "archive_merge_v1", _ARCHIVE_MERGE_V1
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_V1 = _load_archive_v1()


# -----------------------------------------------------------------------------
# Per-config snapshot helpers — must mirror tests/fixtures/golden/generate_snapshots.py
# -----------------------------------------------------------------------------


def _normalize_abs_paths(text: str) -> str:
    """Replace absolute repo-path prefixes with a stable ``<REPO>`` placeholder.

    v1 and v2 merge.py raise exceptions that sometimes embed absolute paths
    (e.g., YAML ``ScannerError`` messages include the file path being parsed).
    Golden fixtures generated on the developer's machine bake in that
    developer's absolute path (e.g., ``/Users/knight/code_local/HFT-pipeline-v2/...``).
    CI runs under a different absolute path (e.g.,
    ``/home/runner/work/lob-model-trainer/lob-model-trainer/...``), so a naive
    string comparison of error messages diverges — not because the parity
    contract is broken, but because the paths are environment-specific.

    This helper strips the repo-root absolute prefix from any string, mapping
    both dev and CI paths to the same ``<REPO>`` token. The parity invariant
    (same YAML input → same exception type + same structural message) is
    preserved; only the environment-specific path prefix is normalized away.

    Used symmetrically in ``_normalize_snapshot`` — both the golden (loaded
    from JSON written on dev) and the actual (produced in CI) get normalized
    before comparison. This preserves the goldens as-is (no regeneration
    needed).

    Fix introduced: Phase V.A.0 (2026-04-21). V.A.2 CI surfaced the latent
    path-leakage issue; 18 parity test failures (6 configs × 3 Python
    versions × ScannerError case) traced to absolute-path divergence.
    """
    return text.replace(str(_REPO_ROOT), "<REPO>")


def _normalize_snapshot(s: Dict[str, Any]) -> Dict[str, Any]:
    """Strip environment-specific absolute paths from a snapshot's error message.

    ok-status snapshots pass through unchanged (no path leakage in resolved
    dicts — PyYAML doesn't embed paths in the parsed result itself, only in
    exception messages).

    error-status snapshots have their ``message`` field normalized via
    ``_normalize_abs_paths``. ``exc_type`` and ``status`` are untouched —
    those are structural invariants the parity test MUST lock.
    """
    if s.get("status") == "error" and "message" in s:
        return {**s, "message": _normalize_abs_paths(s["message"])}
    return s


def _snapshot(resolve_fn, yaml_path: Path) -> Dict[str, Any]:
    """Run ``resolve_fn`` on ``yaml_path``; return {status, resolved|exc} dict.

    Matches the serialization contract of ``generate_snapshots.py``:
    - JSON-roundtripped resolved dict (so non-serializable values fail here)
    - default=str for Path / Enum / anything exotic
    """
    try:
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        if raw is None:
            raw = {}
        resolved = resolve_fn(raw, yaml_path.resolve())
        roundtrip = json.loads(json.dumps(resolved, sort_keys=True, default=str))
        return {"status": "ok", "resolved": roundtrip}
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "exc_type": type(exc).__name__,
            "message": str(exc),
        }


def _golden_fixture_path(rel_key: str) -> Path:
    """Return the golden fixture JSON path for a config relative-key."""
    return (_GOLDEN_ROOT / rel_key).with_suffix(".json")


def _load_golden(rel_key: str) -> Dict[str, Any]:
    path = _golden_fixture_path(rel_key)
    assert path.exists(), f"Golden fixture missing: {path}"
    with open(path) as f:
        return json.load(f)


def _is_v1_compatible_yaml(yaml_path: Path) -> bool:
    """True if the YAML is readable by v1 merge.py (single-string `_base:` or none).

    After Phase 3.5 migrations, configs use `_base: [list]` which is a v2-only
    feature. v1 raises `ValueError("_base must be a non-empty file path string")`
    on list form. The `test_v2_matches_golden_fixture` test is the universal
    correctness gate (applies to ALL configs); the v1-archive tests only apply
    to configs that v1 could have resolved in the first place.

    Detection: parse YAML, check if `_base` is a list (v2-only) vs
    str/absent (v1-compatible).
    """
    try:
        with open(yaml_path) as f:
            raw = yaml.safe_load(f) or {}
    except Exception:
        # If we can't even parse, let the actual test raise the real error.
        return True
    if not isinstance(raw, dict):
        return True
    base = raw.get("_base")
    if base is None:
        return True   # no inheritance — v1 + v2 both return identity
    if isinstance(base, str):
        return True   # v1 form
    return False      # list (or other non-str) — v2-only


# -----------------------------------------------------------------------------
# Test cases — parametrized over every in-scope config
# -----------------------------------------------------------------------------

# Mirror the EXCLUDED_EXPERIMENTS set in generate_snapshots.py
_EXCLUDED = {
    "nvda_xgboost_baseline_h60.yaml",
    "nvda_xgboost_baseline_arcx_h60.yaml",
}


def _collect_in_scope_configs() -> list[tuple[str, Path]]:
    """Collect every config that is in-scope for parity (rel_key, abs_path).

    Excludes the 2 XGBoost configs (different schema). Includes configs/archive/
    is NOT — those legacy configs are excluded from Phase 3 migration per §3.6.
    """
    out: list[tuple[str, Path]] = []
    for yp in sorted((_CONFIGS_ROOT / "experiments").glob("*.yaml")):
        if yp.name in _EXCLUDED:
            continue
        out.append((str(yp.relative_to(_CONFIGS_ROOT)), yp))
    for yp in sorted((_CONFIGS_ROOT / "bases").glob("*.yaml")):
        out.append((str(yp.relative_to(_CONFIGS_ROOT)), yp))
    return out


_IN_SCOPE = _collect_in_scope_configs()


@pytest.mark.parametrize(
    "rel_key,yaml_path",
    _IN_SCOPE,
    ids=[k for k, _ in _IN_SCOPE],
)
class TestV1V2Parity:
    """For every in-scope config, v1 and v2 must produce identical output."""

    def test_v2_matches_golden_fixture(
        self,
        rel_key: str,
        yaml_path: Path,
    ):
        """New v2 resolve_inheritance reproduces the pre-migration golden JSON."""
        from lobtrainer.config.merge import resolve_inheritance

        golden = _load_golden(rel_key)
        actual = _snapshot(resolve_inheritance, yaml_path)

        # Normalize absolute paths on both sides so the parity invariant
        # (same input → same {status, exc_type, message}) is checked without
        # environment-specific path prefixes breaking the comparison.
        assert _normalize_snapshot(actual) == _normalize_snapshot(golden), (
            f"v2 diverges from golden fixture for {rel_key}.\n"
            f"Expected (v1 golden, normalized): {json.dumps(_normalize_snapshot(golden), sort_keys=True, indent=2)[:500]}\n"
            f"Actual (v2, normalized): {json.dumps(_normalize_snapshot(actual), sort_keys=True, indent=2)[:500]}"
        )

    def test_v1_archive_matches_golden_fixture(
        self,
        rel_key: str,
        yaml_path: Path,
    ):
        """Sanity: the archived v1 ALSO reproduces the golden fixture — but
        only for v1-compatible configs (no list-form `_base:`).

        This guards against a scenario where we accidentally archived a
        MODIFIED merge.py instead of the true v1 — the archive must
        still produce the same snapshots it was derived from, on at least
        the fixtures that were v1-compatible at snapshot time.

        After Phase 3.5 migrations, configs use v2-only `_base: [list]` form;
        v1 cannot parse those, so this test skips them. The universal
        correctness gate is ``test_v2_matches_golden_fixture`` above.
        """
        if not _is_v1_compatible_yaml(yaml_path):
            pytest.skip(
                f"{rel_key} uses v2-only `_base: [list]` form; "
                f"v1 archive cannot parse it (expected — Phase 3.5 migration)."
            )
        golden = _load_golden(rel_key)
        actual = _snapshot(_V1.resolve_inheritance, yaml_path)

        # Normalize absolute paths on both sides — see test_v2_matches_golden_fixture.
        assert _normalize_snapshot(actual) == _normalize_snapshot(golden), (
            f"Archived v1 diverges from golden fixture for {rel_key}. "
            f"Has the archive been tampered with?"
        )

    def test_v1_and_v2_agree(
        self,
        rel_key: str,
        yaml_path: Path,
    ):
        """Direct v1 vs v2 diff — only for v1-compatible configs. Migrated
        configs (v2-only list form) are covered by ``test_v2_matches_golden_fixture``
        against the same frozen golden JSON."""
        if not _is_v1_compatible_yaml(yaml_path):
            pytest.skip(
                f"{rel_key} uses v2-only `_base: [list]` form; v1 can't parse. "
                f"Parity is enforced via golden-fixture match (test_v2_matches_golden_fixture)."
            )
        from lobtrainer.config.merge import resolve_inheritance

        v1_out = _snapshot(_V1.resolve_inheritance, yaml_path)
        v2_out = _snapshot(resolve_inheritance, yaml_path)

        # Normalize absolute paths — v1 and v2 may cite different YAML error
        # paths (mostly when both fail on the same YAML), but they should
        # produce identical structure.
        assert _normalize_snapshot(v1_out) == _normalize_snapshot(v2_out), (
            f"v1 and v2 disagree on {rel_key}.\n"
            f"v1 (normalized): {_normalize_snapshot(v1_out)}\n"
            f"v2 (normalized): {_normalize_snapshot(v2_out)}"
        )


class TestFixtureCompleteness:
    """Meta-tests on the golden fixture set."""

    def test_fixture_count_matches_in_scope_configs(self):
        """Every in-scope config has a golden fixture."""
        for rel_key, _ in _IN_SCOPE:
            assert _golden_fixture_path(rel_key).exists(), (
                f"Missing golden fixture for {rel_key}"
            )

    def test_no_xgboost_in_fixtures(self):
        """The 2 XGBoost configs must NOT have golden fixtures."""
        for excluded in _EXCLUDED:
            rel_key = f"experiments/{excluded}"
            assert not _golden_fixture_path(rel_key).exists(), (
                f"XGBoost config {excluded} should be excluded from fixtures"
            )

    def test_zero_pre_existing_bases_after_monolith_retirement(self):
        """Post-monolith-retirement (Batch 3 prep 2026-04-15): the bases/
        golden directory should be empty. All 5 E5 consumers were migrated
        to multi-base in Batch 1a, then the monolith `e5_tlob_regression.yaml`
        was deleted alongside its golden fixture. Axis-partitioned bases
        under `configs/bases/{models,datasets,labels,train}/` are PARTIAL
        (each declares `_partial: true`) — they aren't standalone-loadable
        configs, so they don't get captured as golden fixtures at the
        bases/ level."""
        base_fixtures = list(
            (_GOLDEN_ROOT / "bases").glob("*.json")
        )
        assert len(base_fixtures) == 0, (
            f"Expected 0 pre-existing base fixtures (monolith retired); "
            f"got {len(base_fixtures)}: {[p.name for p in base_fixtures]}"
        )


# -----------------------------------------------------------------------------
# Unit tests for the _is_v1_compatible_yaml helper (audit item G9)
# -----------------------------------------------------------------------------
#
# A bug in this helper could mask real failures: if it incorrectly returns
# True for a v2-only config, the archive test runs v1 on unparseable YAML
# and either crashes or silently passes — corrupting the safety net.


class TestIsV1CompatibleYaml:
    """Unit tests for the v1/v2 compatibility detector."""

    def test_no_base_key_is_v1_compatible(self, tmp_path):
        """Config without _base: → v1 can handle (returns data as-is)."""
        p = tmp_path / "no_base.yaml"
        p.write_text(yaml.dump({"name": "plain", "train": {"epochs": 30}}))
        assert _is_v1_compatible_yaml(p) is True

    def test_string_base_is_v1_compatible(self, tmp_path):
        """`_base: "path.yaml"` (string form) → v1 native form."""
        p = tmp_path / "string_base.yaml"
        p.write_text(yaml.dump({"_base": "../bases/thing.yaml", "name": "x"}))
        assert _is_v1_compatible_yaml(p) is True

    def test_null_base_is_v1_compatible(self, tmp_path):
        """`_base: null` — v1 treats same as "no _base" (pop returns None)."""
        p = tmp_path / "null_base.yaml"
        p.write_text(yaml.dump({"_base": None, "name": "x"}))
        assert _is_v1_compatible_yaml(p) is True

    def test_list_base_is_NOT_v1_compatible(self, tmp_path):
        """List-form `_base: [a, b]` is v2-only; v1 raises on it."""
        p = tmp_path / "list_base.yaml"
        p.write_text(yaml.dump({"_base": ["a.yaml", "b.yaml"], "name": "x"}))
        assert _is_v1_compatible_yaml(p) is False

    def test_dict_base_is_NOT_v1_compatible(self, tmp_path):
        """Dict-form `_base: {k: v}` — both v1 and v2 reject, but classifier
        says 'not v1 compatible' (conservative — v2's error message is clearer)."""
        p = tmp_path / "dict_base.yaml"
        p.write_text(yaml.dump({"_base": {"key": "val"}, "name": "x"}))
        assert _is_v1_compatible_yaml(p) is False

    def test_int_base_is_NOT_v1_compatible(self, tmp_path):
        """Int `_base: 42` — not a string, so classified as v2-only.
        (Both v1 and v2 raise; v2's diagnostic is better — skip v1 test.)"""
        p = tmp_path / "int_base.yaml"
        p.write_text(yaml.dump({"_base": 42, "name": "x"}))
        assert _is_v1_compatible_yaml(p) is False

    def test_empty_string_base_is_v1_compatible(self, tmp_path):
        """`_base: ""` — v1 raises a specific error; still v1-compatible
        (both v1 and v2 reject with identical intent — parity preserved)."""
        p = tmp_path / "empty_base.yaml"
        p.write_text(yaml.dump({"_base": "", "name": "x"}))
        assert _is_v1_compatible_yaml(p) is True

    def test_malformed_yaml_is_v1_compatible(self, tmp_path):
        """YAML parse failure → default True (let actual test surface the real error)."""
        p = tmp_path / "broken.yaml"
        p.write_text("this: is not: valid: yaml:\n  - [broken")
        assert _is_v1_compatible_yaml(p) is True

    def test_missing_file_is_v1_compatible(self, tmp_path):
        """Missing file → default True (defer to the actual test for diagnostic)."""
        p = tmp_path / "nope.yaml"
        assert _is_v1_compatible_yaml(p) is True

    def test_yaml_that_is_not_a_dict_is_v1_compatible(self, tmp_path):
        """Top-level non-dict YAML → classifier says v1-compatible (defer to real test).
        Example: a YAML file containing a bare list."""
        p = tmp_path / "list_top.yaml"
        p.write_text(yaml.dump(["just", "a", "list"]))
        assert _is_v1_compatible_yaml(p) is True
