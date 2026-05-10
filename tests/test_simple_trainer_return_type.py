"""Phase Y / γ-1 LITE / #PY-88 (2026-05-10) parity tests for sklearn
``_resolve_labels_for_day`` + ``_load_split`` honoring ``LabelsConfig.return_type``.

Pre-#PY-88, sklearn's ``_load_split`` read cached ``*_regression_labels.npy``
regardless of ``LabelsConfig.return_type``. γ-1 LITE empirical gate
2026-05-09 night confirmed 6 sklearn arms produced **bit-exact identical**
metrics across point_return / smoothed_return / mean_return / peak_return
axis values — proving the data-path was bypassing the configured return_type.

Post-#PY-88 (Phase 1 surgical edit at simple_trainer.py:64-263), sklearn
mirrors the PyTorch path's 3-way dispatch on ``LabelsConfig.source``
(``auto`` / ``forward_prices`` / ``labels``). When source resolves to
``forward_prices``, labels are recomputed via ``LabelFactory.multi_horizon``
which honors return_type.

These tests lock the dispatch contract end-to-end. Failure of any test
indicates either:
- regression of the surgical edit (Branch 1/2/3 dispatch logic broken)
- divergence from the PyTorch path's LabelFactory consumption (cross-impl drift)
- silent ignore of LabelsConfig.return_type (the original bug class returning)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


# =============================================================================
# Constants — keep in sync with test_simple_trainer.py for fixture parity
# =============================================================================

NUM_SEQS = 20
SEQ_LEN = 100
NUM_FEATURES = 98
NUM_HORIZONS = 3
HORIZONS = [10, 60, 300]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fp_data_dir(tmp_path):
    """Synthetic data dir with forward_prices.npy + metadata.forward_prices
    block + horizons (mirrors v3p0 export contract).

    Pseudo-random walk around 130.0 USD ensures non-degenerate labels for
    all 4 return_type variants. Single-day fixture (smaller than the
    train/val/test 6-day fixture in test_simple_trainer.py) for fast
    parametric testing.
    """
    rng = np.random.default_rng(42)
    K = 5
    MAX_H = max(HORIZONS)
    N_FP_COLS = K + MAX_H + 1  # 306

    split_dir = tmp_path / "test"
    split_dir.mkdir()

    n = NUM_SEQS
    sequences = rng.standard_normal((n, SEQ_LEN, NUM_FEATURES)).astype(np.float32)
    sequences[:, -1, 40] = 130.0  # mid_price
    sequences[:, -1, 42] = 2.5  # spread_bps

    # Cached regression labels (used by Branch 1 legacy / "labels" source)
    cached_reg = np.full((n, NUM_HORIZONS), 99.0, dtype=np.float64)

    # Forward prices: pseudo-random walk
    forward_prices = (
        130.0
        + np.cumsum(rng.standard_normal((n, N_FP_COLS)) * 0.01, axis=1)
    ).astype(np.float64)

    np.save(split_dir / "2025-01-01_sequences.npy", sequences)
    np.save(split_dir / "2025-01-01_regression_labels.npy", cached_reg)
    np.save(split_dir / "2025-01-01_forward_prices.npy", forward_prices)

    metadata = {
        "day": "2025-01-01",
        "n_sequences": n,
        "n_features": NUM_FEATURES,
        "schema_version": "3.0",
        "horizons": HORIZONS,
        "forward_prices": {
            "exported": True,
            "smoothing_window_offset": K,
            "max_horizon": MAX_H,
            "n_columns": N_FP_COLS,
        },
    }
    with open(split_dir / "2025-01-01_metadata.json", "w") as f:
        json.dump(metadata, f)

    return tmp_path, split_dir, "2025-01-01", forward_prices, cached_reg, K


def _make_labels_config(return_type: str = "smoothed_return", source: str = "forward_prices"):
    """Build a minimal LabelsConfig with the requested return_type + source."""
    from lobtrainer.config.schema import LabelsConfig
    return LabelsConfig(
        primary_horizon_idx=0,
        horizons=HORIZONS,
        source=source,
        return_type=return_type,
        task="regression",
    )


# =============================================================================
# Test class: dispatch contract on LabelsConfig.source
# =============================================================================


class TestSourceDispatch:
    """Lock the 3-way dispatch on ``LabelsConfig.source`` (auto/forward_prices/labels)."""

    def test_legacy_path_when_labels_config_none(self, fp_data_dir):
        """Branch 1: labels_config=None → cached ``*_regression_labels.npy``.

        This is the back-compat path used by the pre-Phase-Q.6 legacy
        flat-keyword constructor (``SimpleModelTrainer(...)`` directly,
        without ``from_config``). The cached labels MUST be returned
        verbatim — no LabelFactory dispatch.
        """
        from lobtrainer.training.simple_trainer import _resolve_labels_for_day

        _, split_dir, day, _, cached_reg, _ = fp_data_dir
        with open(split_dir / f"{day}_metadata.json") as f:
            metadata = json.load(f)

        reg = _resolve_labels_for_day(
            split_dir=split_dir,
            day=day,
            metadata=metadata,
            labels_config=None,
        )
        np.testing.assert_array_equal(reg, cached_reg), \
            "labels_config=None must return cached *_regression_labels.npy verbatim"

    def test_explicit_precomputed_source_uses_cached(self, fp_data_dir):
        """Branch 1: labels_config.source="precomputed" → cached path.

        Same as labels_config=None but explicit. Used when caller wants
        to force the cached path even with from_config construction.
        Note: Pydantic ``LabelsConfig._VALID_SOURCES`` enum is
        ``{"auto", "precomputed", "forward_prices"}`` — ``"precomputed"``
        is the canonical value (verified at schema.py:322-323).
        """
        from lobtrainer.training.simple_trainer import _resolve_labels_for_day

        _, split_dir, day, _, cached_reg, _ = fp_data_dir
        with open(split_dir / f"{day}_metadata.json") as f:
            metadata = json.load(f)

        labels_cfg = _make_labels_config(source="precomputed")
        reg = _resolve_labels_for_day(
            split_dir=split_dir,
            day=day,
            metadata=metadata,
            labels_config=labels_cfg,
        )
        np.testing.assert_array_equal(reg, cached_reg)

    def test_forward_prices_source_recomputes_via_labelfactory(self, fp_data_dir):
        """Branch 2: labels_config.source="forward_prices" → LabelFactory.multi_horizon.

        Output is bit-exact equal to a direct LabelFactory.multi_horizon
        call with the same (forward_prices, horizons, k, return_type) args.
        Cached labels are IGNORED (verified by checking output != cached_reg
        which we filled with constant 99.0 in the fixture).
        """
        from hft_contracts import LabelFactory
        from lobtrainer.training.simple_trainer import _resolve_labels_for_day

        _, split_dir, day, fp, cached_reg, k = fp_data_dir
        with open(split_dir / f"{day}_metadata.json") as f:
            metadata = json.load(f)

        labels_cfg = _make_labels_config(return_type="smoothed_return", source="forward_prices")
        reg = _resolve_labels_for_day(
            split_dir=split_dir,
            day=day,
            metadata=metadata,
            labels_config=labels_cfg,
        )

        # Bit-exact match against direct LabelFactory call (proves dispatch
        # consumes the SSoT primitive correctly).
        expected = LabelFactory.multi_horizon(fp, HORIZONS, k, "smoothed_return")
        np.testing.assert_array_equal(reg, expected), \
            "forward_prices source must produce LabelFactory.multi_horizon output bit-exact"

        # Confirm cached path was BYPASSED (cached_reg is constant 99.0 sentinel).
        assert not np.array_equal(reg, cached_reg), \
            "forward_prices source must NOT return cached labels (cosmetic-axis bug regression)"

    def test_auto_source_with_metadata_block_uses_forward_prices(self, fp_data_dir):
        """Branch 3a: labels_config.source="auto" + metadata declares fp → forward_prices path.

        Auto-detection: when metadata has ``forward_prices.exported=True``
        AND the file exists, use forward_prices; output matches LabelFactory.
        """
        from hft_contracts import LabelFactory
        from lobtrainer.training.simple_trainer import _resolve_labels_for_day

        _, split_dir, day, fp, _, k = fp_data_dir
        with open(split_dir / f"{day}_metadata.json") as f:
            metadata = json.load(f)

        labels_cfg = _make_labels_config(return_type="smoothed_return", source="auto")
        reg = _resolve_labels_for_day(
            split_dir=split_dir,
            day=day,
            metadata=metadata,
            labels_config=labels_cfg,
        )
        expected = LabelFactory.multi_horizon(fp, HORIZONS, k, "smoothed_return")
        np.testing.assert_array_equal(reg, expected)

    def test_auto_source_without_metadata_block_falls_back_to_cached(self, tmp_path):
        """Branch 3b: labels_config.source="auto" + no fp.npy → cached path.

        Metadata lacks ``forward_prices`` block → auto-detection chooses
        the cached labels path (legacy v2.2 export back-compat).
        """
        from lobtrainer.training.simple_trainer import _resolve_labels_for_day

        rng = np.random.default_rng(42)
        split_dir = tmp_path / "test"
        split_dir.mkdir()
        day = "2025-01-01"
        cached_reg = np.full((10, NUM_HORIZONS), 7.0, dtype=np.float64)
        # Sequences + cached labels only (NO forward_prices.npy)
        sequences = rng.standard_normal((10, SEQ_LEN, NUM_FEATURES)).astype(np.float32)
        np.save(split_dir / f"{day}_sequences.npy", sequences)
        np.save(split_dir / f"{day}_regression_labels.npy", cached_reg)
        # Metadata has NO forward_prices block — pre-Phase-O v2.2-style export
        metadata = {
            "day": day,
            "n_sequences": 10,
            "n_features": NUM_FEATURES,
            "schema_version": "3.0",
            "horizons": HORIZONS,
        }
        with open(split_dir / f"{day}_metadata.json", "w") as f:
            json.dump(metadata, f)

        labels_cfg = _make_labels_config(source="auto")
        reg = _resolve_labels_for_day(
            split_dir=split_dir,
            day=day,
            metadata=metadata,
            labels_config=labels_cfg,
        )
        np.testing.assert_array_equal(reg, cached_reg), \
            "auto source must fall back to cached labels when metadata.forward_prices absent"


# =============================================================================
# Test class: cosmetic-axis bug fix proof
# =============================================================================


class TestReturnTypeDispatch:
    """Phase Y / γ-1 LITE / #PY-88 cosmetic-axis bug fix proof.

    Pre-#PY-88: 6 sklearn arms with different return_type values produced
    BIT-EXACT identical metrics. Post-#PY-88: each return_type yields a
    distinct labels array (and downstream distinct training metrics).
    """

    def test_point_return_differs_from_smoothed_return(self, fp_data_dir):
        """Same fixture, different return_type → DIFFERENT outputs.

        This is the #PY-88 bug-fix proof: pre-fix, both calls returned
        identical cached labels (regardless of return_type). Post-fix,
        each return_type drives LabelFactory to compute a different
        function over the same forward_prices.
        """
        from lobtrainer.training.simple_trainer import _resolve_labels_for_day

        _, split_dir, day, _, _, _ = fp_data_dir
        with open(split_dir / f"{day}_metadata.json") as f:
            metadata = json.load(f)

        reg_point = _resolve_labels_for_day(
            split_dir=split_dir, day=day, metadata=metadata,
            labels_config=_make_labels_config(return_type="point_return"),
        )
        reg_smoothed = _resolve_labels_for_day(
            split_dir=split_dir, day=day, metadata=metadata,
            labels_config=_make_labels_config(return_type="smoothed_return"),
        )

        assert not np.array_equal(reg_point, reg_smoothed), (
            "point_return and smoothed_return MUST produce different labels "
            "on the same forward_prices fixture (this is the #PY-88 cosmetic-"
            "axis bug fix proof). If they're equal, the dispatch is broken."
        )

    @pytest.mark.parametrize("return_type", [
        "smoothed_return", "point_return", "mean_return", "peak_return",
    ])
    @pytest.mark.parametrize("horizon_idx", [0, 1, 2])
    def test_sklearn_dispatch_parity_with_labelfactory(self, fp_data_dir, return_type, horizon_idx):
        """Parametric parity: sklearn ``_resolve_labels_for_day`` output is
        bit-exact equal to direct ``LabelFactory.multi_horizon`` for all
        4 return_types × 3 horizons (12 cases).

        This is the canonical sklearn↔LabelFactory parity lock. The
        PyTorch path also calls LabelFactory.multi_horizon (verified at
        ``dataset.py:459-461``), so equivalence with LabelFactory IS
        equivalence with the PyTorch path.
        """
        from hft_contracts import LabelFactory
        from lobtrainer.training.simple_trainer import _resolve_labels_for_day

        _, split_dir, day, fp, _, k = fp_data_dir
        with open(split_dir / f"{day}_metadata.json") as f:
            metadata = json.load(f)

        labels_cfg = _make_labels_config(return_type=return_type, source="forward_prices")
        reg = _resolve_labels_for_day(
            split_dir=split_dir, day=day, metadata=metadata, labels_config=labels_cfg,
        )
        expected = LabelFactory.multi_horizon(fp, HORIZONS, k, return_type)

        # Bit-exact (no rtol/atol — both go through the same SSoT primitive).
        np.testing.assert_array_equal(
            reg[:, horizon_idx], expected[:, horizon_idx],
            err_msg=(
                f"Dispatch parity broken for return_type={return_type!r} "
                f"horizon_idx={horizon_idx} (horizon={HORIZONS[horizon_idx]}). "
                f"Sklearn _resolve_labels_for_day output diverged from "
                f"LabelFactory.multi_horizon — same SSoT, must be bit-exact."
            ),
        )


# =============================================================================
# Test class: horizons truth-pinning
# =============================================================================


class TestHorizonsTruthPin:
    """Phase C.1 horizons truth-pin invariant: LabelsConfig.horizons (when
    set) overrides metadata.horizons. When LabelsConfig.horizons is empty,
    metadata.horizons is consulted (with fallback chain through metadata.
    labeling.horizons + metadata.label_config.horizons)."""

    def test_labels_config_horizons_overrides_metadata(self, fp_data_dir):
        """When LabelsConfig.horizons is set, dispatch uses those values.

        Set LabelsConfig.horizons=[10, 60] (subset of metadata.horizons=[10,60,300]).
        Verify output shape is (N, 2) not (N, 3).
        """
        from hft_contracts import LabelFactory
        from lobtrainer.training.simple_trainer import _resolve_labels_for_day
        from lobtrainer.config.schema import LabelsConfig

        _, split_dir, day, fp, _, k = fp_data_dir
        with open(split_dir / f"{day}_metadata.json") as f:
            metadata = json.load(f)

        custom_horizons = [10, 60]  # subset
        labels_cfg = LabelsConfig(
            primary_horizon_idx=0,
            horizons=custom_horizons,
            source="forward_prices",
            return_type="smoothed_return",
            task="regression",
        )
        reg = _resolve_labels_for_day(
            split_dir=split_dir, day=day, metadata=metadata, labels_config=labels_cfg,
        )

        assert reg.shape == (NUM_SEQS, len(custom_horizons)), (
            f"LabelsConfig.horizons={custom_horizons} must drive output shape "
            f"(got {reg.shape}, expected {(NUM_SEQS, len(custom_horizons))})"
        )
        # Bit-exact equivalence with direct LabelFactory call using the
        # OVERRIDE horizons (not metadata.horizons).
        expected = LabelFactory.multi_horizon(fp, custom_horizons, k, "smoothed_return")
        np.testing.assert_array_equal(reg, expected)


# =============================================================================
# Test class: fail-loud guards (mid-impl gate MED-2 closure)
# =============================================================================


class TestFailLoudGuards:
    """Phase Y / γ-1 LITE / #PY-88 (2026-05-10) mid-impl gate MED-2 closure.

    Locks the 4 fail-loud guard branches in ``_resolve_labels_for_day``:
    1. source="forward_prices" + missing fp.npy → ``FileNotFoundError``
    2. source="forward_prices" + missing forward_prices block in metadata
       → ``ValueError`` (HIGH-1 fix; mirrors PyTorch path symmetry)
    3. LabelFactory output non-finite → ``ValueError``
    4. Unknown source value → ``ValueError`` (defensive — unreachable when
       Pydantic validates at construction, but lock the defensive branch)
    """

    def test_forward_prices_source_missing_fp_npy_raises(self, tmp_path):
        """Branch 2: source="forward_prices" + fp.npy file missing → FileNotFoundError."""
        from lobtrainer.training.simple_trainer import _resolve_labels_for_day

        rng = np.random.default_rng(42)
        split_dir = tmp_path / "test"
        split_dir.mkdir()
        day = "2025-01-01"
        # Sequences + cached labels exist; fp.npy DOES NOT
        np.save(split_dir / f"{day}_sequences.npy",
                rng.standard_normal((10, SEQ_LEN, NUM_FEATURES)).astype(np.float32))
        np.save(split_dir / f"{day}_regression_labels.npy",
                rng.standard_normal((10, NUM_HORIZONS)).astype(np.float64))
        K, MAX_H = 5, max(HORIZONS)
        N_FP_COLS = K + MAX_H + 1
        metadata = {
            "day": day,
            "n_sequences": 10,
            "n_features": NUM_FEATURES,
            "schema_version": "3.0",
            "horizons": HORIZONS,
            "forward_prices": {
                "exported": True,
                "smoothing_window_offset": K,
                "max_horizon": MAX_H,
                "n_columns": N_FP_COLS,
            },
        }
        with open(split_dir / f"{day}_metadata.json", "w") as f:
            json.dump(metadata, f)

        labels_cfg = _make_labels_config(source="forward_prices")
        with pytest.raises(FileNotFoundError):
            _resolve_labels_for_day(
                split_dir=split_dir, day=day, metadata=metadata,
                labels_config=labels_cfg,
            )

    def test_forward_prices_source_missing_metadata_block_raises_valueerror(self, tmp_path):
        """Branch 2 HIGH-1: source="forward_prices" + missing fp metadata block
        → explicit ValueError (mirrors PyTorch dataset.py:657-661 symmetry).

        Pre-HIGH-1 fix: would raise KeyError from ForwardPriceContract.from_metadata.
        Post-HIGH-1: explicit ValueError BEFORE the from_metadata call.
        """
        from lobtrainer.training.simple_trainer import _resolve_labels_for_day

        rng = np.random.default_rng(42)
        split_dir = tmp_path / "test"
        split_dir.mkdir()
        day = "2025-01-01"
        # fp.npy EXISTS but metadata does NOT declare forward_prices block
        K, MAX_H = 5, max(HORIZONS)
        N_FP_COLS = K + MAX_H + 1
        np.save(split_dir / f"{day}_sequences.npy",
                rng.standard_normal((10, SEQ_LEN, NUM_FEATURES)).astype(np.float32))
        np.save(split_dir / f"{day}_regression_labels.npy",
                rng.standard_normal((10, NUM_HORIZONS)).astype(np.float64))
        np.save(split_dir / f"{day}_forward_prices.npy",
                (130.0 + np.cumsum(rng.standard_normal((10, N_FP_COLS)) * 0.01, axis=1)).astype(np.float64))
        # Metadata lacks `forward_prices` block — pre-Phase-O v2.2-style
        metadata = {
            "day": day,
            "n_sequences": 10,
            "n_features": NUM_FEATURES,
            "schema_version": "3.0",
            "horizons": HORIZONS,
            # NO `forward_prices` block
        }
        with open(split_dir / f"{day}_metadata.json", "w") as f:
            json.dump(metadata, f)

        labels_cfg = _make_labels_config(source="forward_prices")
        with pytest.raises(ValueError, match=r"forward_prices\.exported"):
            _resolve_labels_for_day(
                split_dir=split_dir, day=day, metadata=metadata,
                labels_config=labels_cfg,
            )

    def test_forward_prices_source_block_exported_false_raises_valueerror(self, tmp_path):
        """Branch 2 HIGH-1: source="forward_prices" + block.exported=False
        → explicit ValueError (declared NOT-exported even though file exists).
        """
        from lobtrainer.training.simple_trainer import _resolve_labels_for_day

        rng = np.random.default_rng(42)
        split_dir = tmp_path / "test"
        split_dir.mkdir()
        day = "2025-01-01"
        K, MAX_H = 5, max(HORIZONS)
        N_FP_COLS = K + MAX_H + 1
        np.save(split_dir / f"{day}_sequences.npy",
                rng.standard_normal((10, SEQ_LEN, NUM_FEATURES)).astype(np.float32))
        np.save(split_dir / f"{day}_regression_labels.npy",
                rng.standard_normal((10, NUM_HORIZONS)).astype(np.float64))
        np.save(split_dir / f"{day}_forward_prices.npy",
                (130.0 + np.cumsum(rng.standard_normal((10, N_FP_COLS)) * 0.01, axis=1)).astype(np.float64))
        metadata = {
            "day": day,
            "n_sequences": 10,
            "n_features": NUM_FEATURES,
            "schema_version": "3.0",
            "horizons": HORIZONS,
            "forward_prices": {
                "exported": False,  # ← explicitly false
                "smoothing_window_offset": K,
                "max_horizon": MAX_H,
                "n_columns": N_FP_COLS,
            },
        }
        with open(split_dir / f"{day}_metadata.json", "w") as f:
            json.dump(metadata, f)

        labels_cfg = _make_labels_config(source="forward_prices")
        with pytest.raises(ValueError, match=r"forward_prices\.exported"):
            _resolve_labels_for_day(
                split_dir=split_dir, day=day, metadata=metadata,
                labels_config=labels_cfg,
            )

    def test_unknown_source_value_raises_valueerror_defensive(self, fp_data_dir):
        """Branch defensive: unknown LabelsConfig.source value → ValueError.

        Unreachable when Pydantic validates at construction, but defensive
        branch must remain to prevent silent acceptance if a non-Pydantic
        caller bypasses validation (mock object, test fixture, etc.).
        Test uses a Mock-like object that bypasses Pydantic validation.
        """
        from lobtrainer.training.simple_trainer import _resolve_labels_for_day

        _, split_dir, day, _, _, _ = fp_data_dir
        with open(split_dir / f"{day}_metadata.json") as f:
            metadata = json.load(f)

        # Synthetic non-Pydantic config object — has only the attributes
        # the dispatch reads. Bypasses _VALID_SOURCES validation.
        class FakeLabelsConfig:
            source = "garbage_value"
            return_type = "smoothed_return"
            horizons = HORIZONS

        with pytest.raises(ValueError, match=r"unknown LabelsConfig\.source"):
            _resolve_labels_for_day(
                split_dir=split_dir, day=day, metadata=metadata,
                labels_config=FakeLabelsConfig(),
            )


# =============================================================================
# Test class: signal_metadata return_type emission (Step 3)
# =============================================================================


class TestSignalMetadataReturnTypeEmission:
    """Phase Y / γ-1 LITE / #PY-88 (2026-05-10): top-level ``return_type``
    field on signal_metadata.json — emitted by ``simple_trainer.export_signals``
    + ``exporter.SignalExporter`` from ``LabelsConfig.return_type``.

    Additive cosmetic surface for ledger / backtester / dashboard queries
    that filter by return_type axis. Pre-#PY-88 the field was absent;
    post-#PY-88 it's present whenever ``self.config`` is non-None.
    """

    @pytest.mark.parametrize("return_type", [
        "smoothed_return", "point_return", "mean_return", "peak_return",
    ])
    def test_return_type_emitted_at_top_level(self, return_type, tmp_path):
        """Full from_config → setup → train → evaluate → export_signals chain.

        Verify ``signal_metadata.json["return_type"]`` matches the
        ``LabelsConfig.return_type`` value passed in the config.
        """
        from lobtrainer.training.simple_trainer import SimpleModelTrainer
        from lobtrainer.config.schema import (
            DataConfig,
            ExperimentConfig,
            LabelsConfig,
            LossType,
            ModelConfig,
            ModelType,
            NormalizationConfig,
            SequenceConfig,
            TaskType,
            TrainConfig,
        )

        # Build minimal data dir with fp.npy + metadata for from_config path
        rng = np.random.default_rng(42)
        K, MAX_H = 5, max(HORIZONS)
        N_FP_COLS = K + MAX_H + 1
        for split in ["train", "val", "test"]:
            split_dir = tmp_path / "data" / split
            split_dir.mkdir(parents=True)
            for i, day in enumerate(["2025-01-01", "2025-01-02"]):
                n = NUM_SEQS + i * 5
                sequences = rng.standard_normal((n, SEQ_LEN, NUM_FEATURES)).astype(np.float32)
                sequences[:, -1, 40] = 130.0
                sequences[:, -1, 42] = 2.5
                reg_labels = rng.standard_normal((n, NUM_HORIZONS)).astype(np.float64)
                forward_prices = (
                    130.0
                    + np.cumsum(rng.standard_normal((n, N_FP_COLS)) * 0.01, axis=1)
                ).astype(np.float64)
                np.save(split_dir / f"{day}_sequences.npy", sequences)
                np.save(split_dir / f"{day}_regression_labels.npy", reg_labels)
                np.save(split_dir / f"{day}_forward_prices.npy", forward_prices)
                with open(split_dir / f"{day}_metadata.json", "w") as f:
                    json.dump({
                        "day": day,
                        "n_sequences": n,
                        "n_features": NUM_FEATURES,
                        "schema_version": "3.0",
                        "horizons": HORIZONS,
                        "forward_prices": {
                            "exported": True,
                            "smoothing_window_offset": K,
                            "max_horizon": MAX_H,
                            "n_columns": N_FP_COLS,
                        },
                    }, f)

        # Build full ExperimentConfig
        data = DataConfig(
            data_dir=str(tmp_path / "data"),
            feature_count=NUM_FEATURES,
            sequence=SequenceConfig(window_size=SEQ_LEN, stride=1),
            normalization=NormalizationConfig(strategy="none"),
            labels=LabelsConfig(
                primary_horizon_idx=0,
                horizons=HORIZONS,
                source="forward_prices",
                return_type=return_type,  # parametric
                task="regression",
            ),
        )
        model = ModelConfig(
            model_type=ModelType.TEMPORAL_RIDGE,
            input_size=NUM_FEATURES,
            params={"alpha": 1.0},
        )
        train = TrainConfig(task_type=TaskType.REGRESSION, loss_type=LossType.HUBER)
        config = ExperimentConfig(
            name=f"py88_return_type_{return_type}",
            data=data,
            model=model,
            train=train,
            output_dir=str(tmp_path / "output"),
        )

        trainer = SimpleModelTrainer.from_config(config)
        trainer.setup()
        trainer.train()
        trainer.evaluate("test")
        signal_dir = trainer.export_signals("test")

        with open(signal_dir / "signal_metadata.json") as f:
            meta = json.load(f)

        assert meta.get("return_type") == return_type, (
            f"signal_metadata.json must emit top-level return_type={return_type!r} "
            f"(got {meta.get('return_type')!r}). #PY-88 Step 3 broken."
        )
