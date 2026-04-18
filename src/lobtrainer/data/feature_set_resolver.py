"""
FeatureSet resolver — trainer-side consumer of the Phase 4 registry.

Loads ``<name>.json`` from ``contracts/feature_sets/``, verifies integrity
(recomputes the content hash from PRODUCT fields and refuses tampered
files), and returns a ``ResolvedFeatureSet`` dataclass carrying the
indices + metadata the trainer needs.

Design notes (Phase 4 Batch 4c, 2026-04-15):

- **Canonical hash form is inlined** (~10 LOC in ``_compute_content_hash``)
  rather than imported from ``hft-ops.feature_sets.hashing``. Rationale:
  keeps lob-model-trainer free of an hft-ops pyproject dependency
  (trainer venvs must stay runnable without the orchestrator). Behavioral
  parity with ``hft_ops.feature_sets.hashing.compute_feature_set_hash``
  is LOCKED by ``tests/test_feature_set_resolver_parity.py`` — any
  canonical-form drift fails CI on both sides.

- **Minimal schema validation** — the resolver checks only the keys it
  reads (name, content_hash, feature_indices, source_feature_count,
  contract_version, applies_to). Full schema validation is the
  producer's responsibility; the writer already validates before write.
  The resolver's job is defensive — reject tampered / corrupt files.

- **Contract compatibility checks** — callers can pass
  ``expected_contract_version`` and ``expected_source_feature_count``
  to have the resolver fail loudly on mismatch (e.g., a FeatureSet
  built over a 128-feature export is consumed by an experiment on a
  98-feature export). Both are optional; passing ``None`` skips the check.

- **Path traversal is guarded** — the ``name`` argument must not contain
  path separators or leading ``.`` so resolver output cannot be coerced
  to escape ``registry_dir``.

Canonical form (must match hft_ops.feature_sets.hashing):

.. code-block:: python

    canonical = {
        "feature_indices": sorted(set(int(i) for i in feature_indices)),
        "source_feature_count": int(source_feature_count),
        "contract_version": str(contract_version),
    }
    blob = json.dumps(canonical, sort_keys=True, default=str).encode("utf-8")
    hash = hashlib.sha256(blob).hexdigest()
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


class FeatureSetResolverError(ValueError):
    """Base class for all FeatureSet resolver failures.

    Subclasses distinguish failure modes so callers can ``except`` the
    specific case they know how to handle (e.g., ``FeatureSetNotFound``
    for "create-and-retry" workflows). Every subclass inherits
    ``ValueError`` via this class, so catch-all ``except ValueError``
    also works for callers that don't need discrimination.
    """


class FeatureSetNotFound(FeatureSetResolverError):
    """The requested FeatureSet name does not exist in the registry.

    Raised when ``<name>.json`` is not present in ``registry_dir``.
    Callers implementing create-on-missing workflows can catch this
    specifically.
    """


class FeatureSetMalformed(FeatureSetResolverError):
    """FeatureSet file exists but fails schema validation.

    Raised on: malformed JSON, missing required keys, filename/name
    mismatch, wrong hash format, feature_indices out of range or with
    duplicates, bad applies_to structure. Distinct from
    ``FeatureSetIntegrityError`` (stored hash ≠ recomputed hash).
    """


class FeatureSetIntegrityError(FeatureSetResolverError):
    """Stored content_hash disagrees with the recomputed hash.

    Detects tampering or disk corruption of the product fields
    (feature_indices, source_feature_count, contract_version). The
    remediation message tells users to regenerate via
    ``hft-ops evaluate --save-feature-set ...`` or restore from git.
    """


class FeatureSetContractMismatch(FeatureSetResolverError):
    """The FeatureSet's contract_version or source_feature_count
    disagrees with what the caller expected.

    Raised only when the caller passes ``expected_contract_version``
    or ``expected_source_feature_count`` and the stored value differs.
    """


@dataclass(frozen=True)
class ResolvedFeatureSet:
    """In-memory representation of a resolved FeatureSet.

    Subset of the full ``hft_ops.feature_sets.FeatureSet`` dataclass —
    only carries fields the trainer needs. The ``content_hash`` is
    retained so callers can propagate it into ledger records
    (``ExperimentRecord.feature_set_ref``) and signal-export metadata.

    Field order (Phase 4 Batch 4c hardening): **identity → contract →
    product → applicability**. Identity (name, hash) first so ``repr()``
    output leads with the cheap, unambiguous fields. Contract
    (contract_version, source_feature_count) next so mismatches surface
    early. Product (feature_indices, feature_names) in the middle
    because downstream consumers iterate these most. Applicability
    (assets, horizons) last because it is advisory metadata.

    Attributes:
        name: FeatureSet identifier (matches the JSON filename).
        content_hash: 64-char lowercase hex SHA-256 of the product fields.
        contract_version: Pipeline contract version at production time.
        source_feature_count: Source feature axis width.
        feature_indices: Sorted-unique tuple of feature indices to use.
        feature_names: Parallel tuple of feature names (metadata only —
            the trainer uses ``feature_indices`` for slicing).
        applies_to_assets: Assets this set was built for (advisory).
        applies_to_horizons: Horizons this set targets (advisory).
    """

    name: str
    content_hash: str
    contract_version: str
    source_feature_count: int
    feature_indices: tuple[int, ...]
    feature_names: tuple[str, ...]
    applies_to_assets: tuple[str, ...]
    applies_to_horizons: tuple[int, ...]


def resolve_feature_set(
    name: str,
    registry_dir: Path,
    *,
    expected_contract_version: Optional[str] = None,
    expected_source_feature_count: Optional[int] = None,
) -> ResolvedFeatureSet:
    """Load, validate, and integrity-verify a FeatureSet from the registry.

    Args:
        name: FeatureSet identifier. Must match ``<name>.json`` in the
            registry directory and the ``name`` field inside the JSON.
            Path separators and leading ``.`` are rejected.
        registry_dir: Directory holding FeatureSet JSON artifacts.
            Typically ``<pipeline_root>/contracts/feature_sets/``.
        expected_contract_version: If provided, resolver refuses to load
            a FeatureSet whose ``contract_version`` field differs. Use
            this when the trainer knows its current contract version
            and wants to fail early rather than train with an
            incompatible feature-index schema.
        expected_source_feature_count: Same as above but for
            ``source_feature_count`` — refuse to use a FeatureSet built
            over a different source width.

    Returns:
        A ``ResolvedFeatureSet`` suitable for populating
        ``DataConfig._feature_indices_resolved``.

    Raises:
        FeatureSetResolverError: On any failure (missing file,
            malformed JSON, schema violation, tampered file,
            name/filename disagreement, contract mismatch).
    """
    # Phase 6 6A.8 (2026-04-17): strip whitespace before emptiness check.
    # Prior behavior treated "   " as truthy, producing a confusing
    # FileNotFoundError on `   .json` instead of the clearer "name empty".
    name = name.strip() if isinstance(name, str) else name
    if not name:
        raise FeatureSetMalformed("FeatureSet name must be non-empty")
    if "/" in name or "\\" in name or name.startswith("."):
        raise FeatureSetMalformed(
            f"FeatureSet name must not contain path separators or start "
            f"with '.', got: {name!r}"
        )

    path = Path(registry_dir) / f"{name}.json"
    if not path.exists():
        available = _list_available_names(registry_dir)
        raise FeatureSetNotFound(
            f"FeatureSet '{name}' not found at {path}. "
            f"Available: {available[:10]}"
            f"{'...' if len(available) > 10 else ''}"
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FeatureSetMalformed(
            f"FeatureSet '{name}' at {path} is not valid JSON: {exc}"
        ) from exc

    _validate_minimal(data, name, path)
    _verify_content_hash(data, name)
    _verify_contract_compat(
        data,
        name,
        expected_contract_version=expected_contract_version,
        expected_source_feature_count=expected_source_feature_count,
    )

    # Field order mirrors ResolvedFeatureSet declaration (identity →
    # contract → product → applicability). See dataclass docstring.
    return ResolvedFeatureSet(
        name=data["name"],
        content_hash=data["content_hash"],
        contract_version=str(data["contract_version"]),
        source_feature_count=int(data["source_feature_count"]),
        feature_indices=tuple(int(i) for i in data["feature_indices"]),
        feature_names=tuple(str(n) for n in data.get("feature_names", ())),
        applies_to_assets=tuple(str(a) for a in data["applies_to"]["assets"]),
        applies_to_horizons=tuple(
            int(h) for h in data["applies_to"]["horizons"]
        ),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_content_hash(
    feature_indices: Iterable[int],
    source_feature_count: int,
    contract_version: str,
) -> str:
    """Mirror of ``hft_ops.feature_sets.hashing.compute_feature_set_hash``.

    Must match byte-for-byte. Parity is locked by a cross-module test.
    """
    canonical = {
        "feature_indices": sorted(set(int(i) for i in feature_indices)),
        "source_feature_count": int(source_feature_count),
        "contract_version": str(contract_version),
    }
    blob = json.dumps(canonical, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _validate_minimal(data: dict, name: str, path: Path) -> None:
    """Minimal schema check — only fields the resolver reads."""
    if not isinstance(data, dict):
        raise FeatureSetMalformed(
            f"FeatureSet {path}: top-level must be a JSON object, "
            f"got {type(data).__name__}"
        )

    required = {
        "name",
        "content_hash",
        "feature_indices",
        "source_feature_count",
        "contract_version",
        "applies_to",
    }
    missing = required - set(data.keys())
    if missing:
        raise FeatureSetMalformed(
            f"FeatureSet {path}: missing required keys {sorted(missing)}"
        )

    if data["name"] != name:
        raise FeatureSetMalformed(
            f"FeatureSet filename/name mismatch at {path}: "
            f"filename='{name}' but content['name']='{data['name']!r}'. "
            f"Every <name>.json MUST have content.name == name."
        )

    ch = data["content_hash"]
    if (
        not isinstance(ch, str)
        or len(ch) != 64
        or not all(c in "0123456789abcdef" for c in ch)
    ):
        raise FeatureSetMalformed(
            f"FeatureSet '{name}': content_hash must be 64-char lowercase "
            f"hex SHA-256, got: {ch!r}"
        )

    indices = data["feature_indices"]
    if not isinstance(indices, list) or not indices:
        raise FeatureSetMalformed(
            f"FeatureSet '{name}': feature_indices must be a non-empty list"
        )
    if any(not isinstance(i, int) or isinstance(i, bool) for i in indices):
        raise FeatureSetMalformed(
            f"FeatureSet '{name}': feature_indices must be integers, "
            f"got: {indices}"
        )
    if any(i < 0 for i in indices):
        raise FeatureSetMalformed(
            f"FeatureSet '{name}': feature_indices must be non-negative"
        )
    if len(set(indices)) != len(indices):
        raise FeatureSetMalformed(
            f"FeatureSet '{name}': feature_indices must be unique"
        )

    sfc = data["source_feature_count"]
    if not isinstance(sfc, int) or isinstance(sfc, bool) or sfc <= 0:
        raise FeatureSetMalformed(
            f"FeatureSet '{name}': source_feature_count must be positive int, "
            f"got {sfc!r}"
        )
    if max(indices) >= sfc:
        raise FeatureSetMalformed(
            f"FeatureSet '{name}': max(feature_indices)={max(indices)} "
            f"must be < source_feature_count={sfc}"
        )

    applies_to = data["applies_to"]
    if not isinstance(applies_to, dict):
        raise FeatureSetMalformed(
            f"FeatureSet '{name}': applies_to must be a dict, "
            f"got {type(applies_to).__name__}"
        )
    for key in ("assets", "horizons"):
        if key not in applies_to:
            raise FeatureSetMalformed(
                f"FeatureSet '{name}': applies_to missing key '{key}'"
            )
        if not isinstance(applies_to[key], list):
            raise FeatureSetMalformed(
                f"FeatureSet '{name}': applies_to.{key} must be a list"
            )


def _verify_content_hash(data: dict, name: str) -> None:
    """Recompute hash and raise if it disagrees with the stored value."""
    expected = _compute_content_hash(
        feature_indices=data["feature_indices"],
        source_feature_count=data["source_feature_count"],
        contract_version=data["contract_version"],
    )
    if expected != data["content_hash"]:
        raise FeatureSetIntegrityError(
            f"FeatureSet '{name}' integrity check failed. "
            f"Stored content_hash:    {data['content_hash']}. "
            f"Recomputed from fields: {expected}. "
            f"This file was likely edited without a matching hash update. "
            f"Regenerate via `hft-ops evaluate --save-feature-set {name} ...` "
            f"or restore from git."
        )


def _verify_contract_compat(
    data: dict,
    name: str,
    *,
    expected_contract_version: Optional[str],
    expected_source_feature_count: Optional[int],
) -> None:
    """Fail loudly on consumer/producer contract mismatch."""
    if expected_contract_version is not None:
        got = str(data["contract_version"])
        if got != str(expected_contract_version):
            raise FeatureSetContractMismatch(
                f"FeatureSet '{name}' contract_version mismatch: "
                f"set built at '{got}', current pipeline at "
                f"'{expected_contract_version}'. Feature index semantics "
                f"are contract-version-bound; the set must be "
                f"regenerated against the current contract."
            )
    if expected_source_feature_count is not None:
        got = int(data["source_feature_count"])
        if got != int(expected_source_feature_count):
            raise FeatureSetContractMismatch(
                f"FeatureSet '{name}' source_feature_count mismatch: "
                f"set built over {got} features, current export has "
                f"{expected_source_feature_count}. Index semantics differ; "
                f"regenerate the set against the current export."
            )


def _list_available_names(registry_dir: Path) -> list[str]:
    """Best-effort listing of registry names for error messages."""
    try:
        return sorted(p.stem for p in Path(registry_dir).glob("*.json"))
    except OSError:
        return []


def find_feature_sets_dir(anchor: Path, *, max_parents: int = 8) -> Path:
    """Auto-detect ``contracts/feature_sets/`` by walking up from ``anchor``.

    The trainer needs to locate the FeatureSet registry without requiring
    users to set an explicit path in every YAML. This helper walks up
    from a filesystem anchor (typically ``DataConfig.data_dir`` or a
    resolved absolute export path) until it finds a directory that
    contains ``contracts/pipeline_contract.toml`` — the unambiguous
    pipeline-root marker — and returns ``<root>/contracts/feature_sets``.

    Returns even if the target directory does not yet exist (first-run
    registries are legitimate; the resolver will raise a
    ``FeatureSetResolverError`` with ``not found`` on a subsequent
    ``resolve_feature_set`` call if an actual FeatureSet name is
    requested). This separation keeps "where is the registry?" and "is
    this name present?" as distinct failures with distinct messages.

    Args:
        anchor: Absolute or relative path to start walking from. If
            relative, resolved against CWD first. ``Path(data_dir).resolve()``
            is the typical caller expression.
        max_parents: Safety cap on how many ``..`` hops to try before
            giving up. Default 8 covers any reasonable monorepo layout.

    Returns:
        Absolute path to ``<pipeline_root>/contracts/feature_sets``.

    Raises:
        FeatureSetResolverError: If no pipeline-root ancestor is found
            within ``max_parents`` hops. The caller can override via
            an explicit ``feature_sets_dir`` argument to whatever
            orchestrator is calling this.
    """
    current = Path(anchor).resolve()
    visited: list[Path] = []
    for _ in range(max_parents + 1):
        visited.append(current)
        if (current / "contracts" / "pipeline_contract.toml").exists():
            return current / "contracts" / "feature_sets"
        if current.parent == current:
            break  # reached filesystem root
        current = current.parent
    raise FeatureSetResolverError(
        f"Cannot auto-detect the FeatureSet registry: no "
        f"'contracts/pipeline_contract.toml' found walking up from "
        f"{anchor!s}. Checked: {[str(p) for p in visited]}. "
        f"Ensure the trainer's data_dir has the pipeline root as an "
        f"ancestor, or pass a feature_sets_dir explicitly."
    )
