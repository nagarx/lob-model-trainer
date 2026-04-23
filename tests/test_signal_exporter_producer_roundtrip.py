"""Producer→consumer roundtrip test for SignalExporter's CompatibilityContract emission.

Closes the validation gap in Phase II (2026-04-20) where all tests used
fixture-constructed ``CompatibilityContract`` objects, bypassing the producer-path
code at ``_build_compatibility_contract()``. That producer path silently returned
``None`` on every ``ExperimentConfig`` due to canonical-path drift
(``config.labels`` vs ``config.data.labels``).

This test constructs a real ``ExperimentConfig`` (no fixture ``CompatibilityContract``),
invokes ``_build_compatibility_contract()`` directly, pipes the result through
``build_signal_metadata()``, and parses the produced metadata dict via
``SignalManifest.from_dict()``. The chain mirrors what ``SignalExporter.export()``
does at runtime, minus the Trainer+DataLoader infrastructure.

Phase A (2026-04-23) TDD red-gate:
    Runs RED pre-fix (broad ``except Exception`` silently swallows AttributeError
    from ``config.labels`` access → returns None → no fingerprint in manifest).
    The ``@pytest.mark.xfail(strict=True)`` marker keeps CI green while exposing
    the bug. When C1 removes the xfail AND applies the fixes, this test turns
    RED→GREEN visibly in the PR diff.

Invariants locked (post-C1):
    1. ``_build_compatibility_contract(ExperimentConfig(), ...)`` returns a valid
       ``CompatibilityContract`` (not None).
    2. The emitted fingerprint is a 64-character lowercase hex string.
    3. ``primary_horizon_idx`` threads from ``config.data.labels.primary_horizon_idx``
       (non-zero value confirms canonical-path fix, not hardcoded-0 regression).
    4. The 3-way roundtrip (producer → serialize → consumer parse → recompute
       fingerprint) produces identical hashes — no silent contract drift.
"""

import re
from typing import Any

import pytest

_hft_contracts = pytest.importorskip("hft_contracts")
from lobtrainer.config.schema import (  # noqa: E402
    DataConfig,
    ExperimentConfig,
    LabelsConfig,
    ModelConfig,
    TrainConfig,
)
from lobtrainer.export.exporter import _build_compatibility_contract  # noqa: E402
from lobtrainer.export.metadata import build_signal_metadata  # noqa: E402
from hft_contracts.signal_manifest import SignalManifest  # noqa: E402


_FINGERPRINT_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def _build_config_with_primary_horizon_idx(primary_horizon_idx: int) -> ExperimentConfig:
    """Build a minimal ExperimentConfig that exercises the producer-path bug.

    Uses ``primary_horizon_idx != 0`` by default so the test catches both:
      - Bug #1/#2 (``config.labels`` path drift → silent None return).
      - Bug #3 (``config.data.horizon_idx`` deprecated-field read → wrong value
        in the fingerprint).

    Parameters
    ----------
    primary_horizon_idx:
        Index into ``config.data.labels.horizons``. Non-zero values verify the
        producer correctly threads the canonical path.
    """
    return ExperimentConfig(
        name="phase_a_producer_roundtrip",
        data=DataConfig(
            feature_count=98,
            labels=LabelsConfig(
                horizons=[10, 60, 300],
                primary_horizon_idx=primary_horizon_idx,
                task="classification",
            ),
        ),
        model=ModelConfig(model_type="logistic_lob", input_size=98, num_classes=3),
        train=TrainConfig(epochs=1),
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Phase A (2026-04-23) TDD red-gate: producer-path bug cluster. "
        "`_build_compatibility_contract()` silently returns None because it reads "
        "`config.labels.horizons` (canonical is `config.data.labels.horizons`). "
        "The broad `except Exception` at exporter.py:118-129 masks the AttributeError. "
        "Removed in C1 when `resolve_labels_config()` helper lands and the inner "
        "try/except is eliminated."
    ),
)
class TestPhaseAProducerRoundtrip:
    """Producer→consumer CompatibilityContract roundtrip.

    These assertions fail pre-Phase-A because the producer path silently returns
    None. When C1 ships, the xfail marker is removed and all assertions pass.
    """

    def test_build_compatibility_contract_returns_non_none(self) -> None:
        """Smoke: the producer returns a valid contract (not silent None)."""
        config = _build_config_with_primary_horizon_idx(primary_horizon_idx=1)
        contract = _build_compatibility_contract(
            config=config,
            feature_set_ref=None,
            calibration_method=None,
        )
        assert contract is not None, (
            "Producer silently returned None — the canonical-path-drift bug "
            "(config.labels vs config.data.labels) masked by the broad "
            "Exception catch. See Phase A plan + exporter.py:76, 84, 118-129."
        )

    def test_fingerprint_is_valid_sha256_hex(self) -> None:
        """The fingerprint is a canonical 64-char lowercase hex SHA-256 digest."""
        config = _build_config_with_primary_horizon_idx(primary_horizon_idx=1)
        contract = _build_compatibility_contract(
            config=config,
            feature_set_ref=None,
            calibration_method=None,
        )
        assert contract is not None, "Pre-fix: producer returned None"
        fingerprint = contract.fingerprint()
        assert _FINGERPRINT_HEX_RE.match(fingerprint), (
            f"Fingerprint must match ^[0-9a-f]{{64}}$; got {fingerprint!r}"
        )

    def test_primary_horizon_idx_threads_through_from_labels_config(self) -> None:
        """Non-zero ``primary_horizon_idx`` flows from config.data.labels → contract.

        Pre-fix, exporter.py:112 reads ``config.data.horizon_idx`` (DEPRECATED
        legacy field that defaults to 0), so the contract would carry 0 even if
        the canonical path has a different value. Post-fix, the helper sources
        the value from ``config.data.labels.primary_horizon_idx``.
        """
        config = _build_config_with_primary_horizon_idx(primary_horizon_idx=2)
        contract = _build_compatibility_contract(
            config=config,
            feature_set_ref=None,
            calibration_method=None,
        )
        assert contract is not None, "Pre-fix: producer returned None"
        assert contract.primary_horizon_idx == 2, (
            f"Expected primary_horizon_idx=2 (from config.data.labels); "
            f"got {contract.primary_horizon_idx!r}. Suggests legacy-field read "
            f"or hardcoded-0 regression."
        )

    def test_horizons_tuple_sourced_from_data_labels(self) -> None:
        """Horizons come from ``config.data.labels.horizons`` (not config.labels)."""
        config = _build_config_with_primary_horizon_idx(primary_horizon_idx=0)
        contract = _build_compatibility_contract(
            config=config,
            feature_set_ref=None,
            calibration_method=None,
        )
        assert contract is not None, "Pre-fix: producer returned None"
        assert contract.horizons == (10, 60, 300), (
            f"Expected horizons=(10, 60, 300) from config.data.labels; "
            f"got {contract.horizons!r}"
        )

    def test_metadata_dict_carries_fingerprint_key(self) -> None:
        """``build_signal_metadata(compatibility=contract)`` emits the fingerprint."""
        config = _build_config_with_primary_horizon_idx(primary_horizon_idx=1)
        contract = _build_compatibility_contract(
            config=config,
            feature_set_ref=None,
            calibration_method=None,
        )
        assert contract is not None, "Pre-fix: producer returned None"
        meta = build_signal_metadata(
            model_type="logistic_lob",
            model_name="test",
            parameters=294,
            signal_type="classification",
            split="val",
            total_samples=10,
            checkpoint="dummy.pt",
            compatibility=contract,
        )
        assert "compatibility_fingerprint" in meta, (
            "Metadata dict missing compatibility_fingerprint — producer path "
            "returned None, so build_signal_metadata did not emit the block."
        )
        assert meta["compatibility_fingerprint"] == contract.fingerprint()

    def test_signal_manifest_parses_producer_output_successfully(self) -> None:
        """Consumer (SignalManifest) parses the producer's metadata dict — 3-way roundtrip.

        This is the load-bearing invariant: producer emits → consumer parses →
        recomputes fingerprint from the embedded block → asserts identical hash.
        If any of these drift, downstream validators raise ContractError.
        """
        config = _build_config_with_primary_horizon_idx(primary_horizon_idx=1)
        contract = _build_compatibility_contract(
            config=config,
            feature_set_ref=None,
            calibration_method=None,
        )
        assert contract is not None, "Pre-fix: producer returned None"
        meta = build_signal_metadata(
            model_type="logistic_lob",
            model_name="test",
            parameters=294,
            signal_type="classification",
            split="val",
            total_samples=10,
            checkpoint="dummy.pt",
            compatibility=contract,
        )
        manifest = SignalManifest.from_dict(meta)
        assert manifest.compatibility_fingerprint is not None
        assert manifest.compatibility is not None
        assert manifest.compatibility_fingerprint == manifest.compatibility.fingerprint(), (
            "Producer-consumer drift: the stored fingerprint disagrees with the "
            "canonical recompute over the serialized block."
        )
        assert manifest.compatibility.primary_horizon_idx == 1
        assert manifest.compatibility.horizons == (10, 60, 300)
