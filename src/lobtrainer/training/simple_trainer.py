"""
SimpleModelTrainer: training loop for non-PyTorch (sklearn-style) models.

Handles the full pipeline for simple models:
    1. Load sequences and labels from exported .npy files
    2. Engineer temporal features via TemporalFeatureConfig
    3. Train model via fit()
    4. Evaluate on val/test splits
    5. Export predictions for backtester
    6. Save model checkpoint

Output format matches the PyTorch trainer so the backtester works unchanged.

Phase Q (2026-05-04): conforms to ``BaseTrainer`` Protocol via four
methods (``train``, ``evaluate(split)``, ``save_checkpoint``,
``load_checkpoint``) so the framework-aware factory at
``lobtrainer.training.create_trainer`` can return this class for
sklearn models. ``from_config`` classmethod bridges the canonical
``ExperimentConfig`` entry-point (``scripts/train.py``, ``hft-ops``)
to the legacy flat-keyword constructor.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from lobtrainer.config import ExperimentConfig
    from lobtrainer.config.schema import LabelsConfig

# Phase IV (2026-04-20): canonical SSoT is ``hft_metrics.temporal``. The legacy
# ``lobmodels.features.temporal`` path is a re-export shim that emits a
# DeprecationWarning on every call to ``engineer_temporal_features``; importing
# directly from hft_metrics bypasses the shim entirely. The local alias keeps
# the rest of this file unchanged during the rename window (hft_metrics uses
# ``engineer_features`` as the canonical verb-noun name; the legacy
# ``engineer_temporal_features`` alias will be removed 2026-10-31).
from hft_metrics import TemporalFeatureConfig
from hft_metrics import engineer_features as engineer_temporal_features
from lobmodels.models.simple import (
    BaseSimpleModel,
    TemporalRidge,
    TemporalRidgeConfig,
    TemporalGradBoost,
    TemporalGradBoostConfig,
)
from lobtrainer.data.dataset import _validate_day_metadata

logger = logging.getLogger(__name__)

try:
    from hft_contracts import SIGNAL_PRICE_FEATURE_INDEX as MID_PRICE_IDX
    from hft_contracts import SIGNAL_SPREAD_FEATURE_INDEX as SPREAD_BPS_IDX
except ImportError:
    MID_PRICE_IDX = 40
    SPREAD_BPS_IDX = 42

from hft_contracts import SCHEMA_VERSION as _CONTRACT_SCHEMA_VERSION


def _load_split(
    data_dir: Path,
    split: str,
    horizon_idx: int = 0,
    max_days: int = None,
    *,
    labels_config: Optional["LabelsConfig"] = None,
):
    """Load sequences and regression labels for one split.

    Validates each day's metadata against the pipeline contract before
    loading any numpy arrays (fail-fast at the boundary per hft-rules §8).
    A schema_version mismatch (e.g., loading pre-Phase-O v2.2 exports against
    the current v3.0 contract) raises ContractError up to the caller.

    Phase Y / γ-1 LITE / #PY-88 (2026-05-10): when ``labels_config`` is
    provided AND its ``source`` resolves to forward_prices (explicit
    ``"forward_prices"`` OR ``"auto"`` with
    ``metadata.forward_prices.exported``), labels are RECOMPUTED via
    ``LabelFactory.multi_horizon(forward_prices, horizons, k,
    labels_config.return_type)`` mirroring the PyTorch path at
    ``dataset.py:387-510``. This honors ``LabelsConfig.return_type``
    (``point_return`` / ``smoothed_return`` / ``mean_return`` /
    ``peak_return``) which was COSMETIC for sklearn pre-#PY-88 (γ-1
    LITE empirical confirmation 2026-05-09 night: 6 sklearn arms
    produced bit-exact identical metrics across the return_type axis
    because this code path read cached ``*_regression_labels.npy``
    regardless of return_type).

    Back-compat: when ``labels_config is None`` (legacy flat-keyword
    construction without ``from_config``) OR ``labels_config.source ==
    "precomputed"`` (explicit cache opt-in matching
    ``LabelsConfig._VALID_SOURCES``), the cached
    ``*_regression_labels.npy`` path is preserved.

    ``k`` (smoothing_window_offset) is ALWAYS read from
    ``ForwardPriceContract.smoothing_window_offset`` (per-day metadata),
    NEVER from user config — same Bug-B2 invariant the PyTorch path
    enforces at ``dataset.py:421-422``.
    """
    split_dir = data_dir / split
    meta_files = sorted(split_dir.glob("*_metadata.json"))
    if max_days:
        meta_files = meta_files[:max_days]

    all_seqs, all_labels, all_spreads, all_prices = [], [], [], []
    for mf in meta_files:
        with open(mf) as f:
            m = json.load(f)
        day = m["day"]
        _validate_day_metadata(m, day)
        # Phase Z.1 / #PY-1 (2026-05-05): closes Phase D orphan validator
        # on the sklearn data-load path (parallel to dataset.py:898-906
        # PyTorch path). Validates idx 97 RESERVED 0.0 in sequences NPY
        # via mmap_mode='r' O(1) header-only read. strict=False matches
        # the trainer-side warning convention; Phase X.4 may flip to
        # strict=True for production-gate fail-loud.
        seq_path = split_dir / f"{day}_sequences.npy"
        from hft_contracts.validation import validate_idx_97_reserved
        for _w in validate_idx_97_reserved(seq_path, strict=False):
            logger.warning("idx-97 contract warning (%s): %s", day, _w)
        seq = np.load(seq_path, mmap_mode="r")

        # Phase Y / γ-1 LITE / #PY-88 (2026-05-10): source-dispatched
        # label derivation. Mirrors PyTorch dataset.py:650-693 3-way
        # dispatch (auto / forward_prices / labels).
        reg = _resolve_labels_for_day(
            split_dir=split_dir,
            day=day,
            metadata=m,
            labels_config=labels_config,
        )

        all_seqs.append(seq)
        all_labels.append(reg[:, horizon_idx])
        all_spreads.append(seq[:, -1, SPREAD_BPS_IDX].astype(np.float64))
        all_prices.append(seq[:, -1, MID_PRICE_IDX].astype(np.float64))

    return (
        np.concatenate(all_seqs, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_spreads, axis=0),
        np.concatenate(all_prices, axis=0),
    )


def _resolve_labels_for_day(
    *,
    split_dir: Path,
    day: str,
    metadata: Dict[str, Any],
    labels_config: Optional["LabelsConfig"],
) -> np.ndarray:
    """Resolve regression labels for a single day, dispatching on
    ``LabelsConfig.source``.

    Phase Y / γ-1 LITE / #PY-88 (2026-05-10) helper extracted so the
    dispatch logic is visible at the per-day grain (mirrors PyTorch
    ``dataset.py:650-693``). Three branches mapping to
    ``LabelsConfig._VALID_SOURCES = {"auto", "precomputed", "forward_prices"}``:

    1. ``labels_config is None`` OR ``source == "precomputed"`` →
       legacy cached ``*_regression_labels.npy`` path. Preserves
       pre-#PY-88 sklearn behavior for back-compat with flat-keyword
       construction (no ``from_config``).
    2. ``source == "forward_prices"`` → ALWAYS recompute via
       ``LabelFactory.multi_horizon`` (raises if forward_prices.npy
       missing or metadata lacks horizons).
    3. ``source == "auto"`` → recompute IF metadata declares
       ``forward_prices.exported = true`` AND the file exists; else
       fall back to cached labels.

    ``k`` ALWAYS from ``ForwardPriceContract.smoothing_window_offset``
    (Bug-B2 invariant). Returns ``[N, n_horizons]`` float64 bps.

    Raises:
        ContractError: metadata missing horizons + source=forward_prices.
        ValueError: LabelFactory produces non-finite values, or
            ``source`` is unknown.
        FileNotFoundError: forward_prices required but file missing.
    """
    fp_path = split_dir / f"{day}_forward_prices.npy"
    cached_path = split_dir / f"{day}_regression_labels.npy"

    # Branch 1: legacy / explicit cache opt-in. Matches PyTorch path's
    # `else` branch in dataset.py:650-693 which handles "precomputed"
    # uniformly with auto-without-fp via the cached labels code path.
    if labels_config is None or labels_config.source == "precomputed":
        return np.load(cached_path)

    # Branches 2 + 3: determine if we use forward_prices.
    if labels_config.source == "forward_prices":
        # Phase Y / γ-1 LITE / #PY-88 mid-impl HIGH-1 (2026-05-10):
        # explicit fail-loud BEFORE attempting LabelFactory dispatch
        # mirrors PyTorch path's `dataset.py:657-661` ValueError exactly.
        # Pre-fix: would raise KeyError from ForwardPriceContract.from_metadata
        # when metadata lacks `forward_prices` block — same fail-loud
        # behavior but breaks cross-trainer UX symmetry (sklearn KeyError vs
        # PyTorch ValueError on identical misconfig).
        fp_meta = metadata.get("forward_prices", {})
        if not (isinstance(fp_meta, dict) and fp_meta.get("exported", False)):
            raise ValueError(
                f"#PY-88 sklearn label dispatch for {day}: "
                f"LabelsConfig.source='forward_prices' but metadata does NOT "
                f"declare forward_prices.exported=True. Either re-export the "
                f"corpus with forward_prices enabled OR set "
                f"LabelsConfig.source='precomputed' to use cached labels. "
                f"(Mirrors PyTorch path at dataset.py:657-661 for "
                f"cross-trainer UX symmetry.)"
            )
        use_fp = True  # mandatory; will raise on missing file
    elif labels_config.source == "auto":
        fp_meta = metadata.get("forward_prices", {})
        meta_declares_fp = (
            isinstance(fp_meta, dict) and fp_meta.get("exported", False)
        )
        use_fp = meta_declares_fp and fp_path.exists()
    else:
        # Defensive — unreachable when LabelsConfig validates source at
        # construction (Pydantic _VALID_SOURCES enum). Locked by
        # tests/test_simple_trainer_return_type.py::TestFailLoudGuards.
        raise ValueError(
            f"#PY-88 sklearn label dispatch: unknown LabelsConfig.source "
            f"{labels_config.source!r} for {day}. Expected 'auto', "
            f"'forward_prices', or 'precomputed' (per "
            f"LabelsConfig._VALID_SOURCES)."
        )

    if not use_fp:
        # auto-fallback to cached labels when metadata doesn't declare
        # forward_prices (legacy v2.2 exports or partial v3p0).
        return np.load(cached_path)

    # Mirrors dataset.py:_compute_labels_from_forward_prices.
    from hft_contracts import ForwardPriceContract, LabelFactory
    from hft_contracts.validation import ContractError

    # Memory-bound load: forward_prices.npy may be ~10x larger than
    # cached labels (300+ horizon columns vs 3-5). mmap keeps RAM bounded
    # across 230-day train concatenation.
    fp = np.load(fp_path, mmap_mode="r")
    contract = ForwardPriceContract.from_metadata(metadata)
    k = contract.smoothing_window_offset

    # Resolve horizons: user override > metadata.horizons > metadata.labeling.horizons.
    if labels_config.horizons:
        horizons = list(labels_config.horizons)
    else:
        horizons = list(metadata.get("horizons") or [])
        if not horizons:
            labeling = metadata.get("labeling", {})
            if isinstance(labeling, dict):
                horizons = list(labeling.get("horizons") or [])
        if not horizons:
            label_section = metadata.get("label_config", {})
            if isinstance(label_section, dict):
                horizons = list(label_section.get("horizons") or [])

    if not horizons:
        raise ContractError(
            f"#PY-88 sklearn label dispatch for {day}: metadata is "
            f"missing horizons (checked: metadata['horizons'], "
            f"metadata['labeling']['horizons'], "
            f"metadata['label_config']['horizons']). Cannot dispatch "
            f"LabelFactory.multi_horizon."
        )

    # np.asarray converts the mmap view to a writable ndarray expected
    # by LabelFactory; no copy occurs unless LabelFactory mutates.
    reg = LabelFactory.multi_horizon(
        np.asarray(fp), horizons, k, labels_config.return_type
    )

    # Defense-in-depth — Pre-T9 validation in LabelFactory.multi_horizon
    # should catch this, but explicit check here mirrors PyTorch path
    # at dataset.py:466-475.
    if not np.isfinite(reg).all():
        raise ValueError(
            f"#PY-88 LabelFactory.multi_horizon produced non-finite "
            f"values for {day}: horizons={horizons}, k={k}, "
            f"return_type={labels_config.return_type!r}. Verify "
            f"forward_prices contains no NaN/Inf."
        )

    return reg


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics matching the PyTorch trainer.

    Delegates to hft-metrics via the regression_metrics adapter (Rule 0: no duplication).
    """
    from lobtrainer.training.regression_metrics import compute_all_regression_metrics

    return compute_all_regression_metrics(y_true, y_pred)


class SimpleModelTrainer:
    """Training pipeline for non-PyTorch models.

    Mirrors the PyTorch Trainer's output format so the backtester,
    signal export, and experiment tracking work unchanged.

    Args:
        config: Parsed experiment configuration dict or object with
                data_dir, model_type, horizon_idx, output_dir fields.
    """

    def __init__(
        self,
        data_dir: str,
        model_type: str,
        model_config: dict,
        feature_config: dict = None,
        horizon_idx: int = 0,
        output_dir: str = "outputs/experiments/simple_model",
    ):
        self.data_dir = Path(data_dir)
        self.model_type = model_type
        self.model_config = model_config
        self.feature_config_dict = feature_config or {}
        self.horizon_idx = horizon_idx
        self.output_dir = Path(output_dir)

        self.model: Optional[BaseSimpleModel] = None
        self.feat_config: Optional[TemporalFeatureConfig] = None
        self.train_metrics: Dict[str, float] = {}
        self.val_metrics: Dict[str, float] = {}
        self.test_metrics: Dict[str, float] = {}
        # Optional ExperimentConfig reference set by ``from_config`` for
        # traceability; absent on the legacy flat-keyword construction path.
        self.config: Optional["ExperimentConfig"] = None

    @classmethod
    def from_config(
        cls,
        config: "ExperimentConfig",
        **kwargs: Any,
    ) -> "SimpleModelTrainer":
        """Build a SimpleModelTrainer from an ``ExperimentConfig``.

        Phase Q.6 (2026-05-04): closes the dispatch gap where sklearn
        models were unreachable through ``create_trainer(config)``. Reads
        ``config.data.data_dir``, ``config.model.model_type``,
        ``config.model.params``, the canonical
        ``config.data.labels.primary_horizon_idx`` (with legacy
        ``config.data.horizon_idx`` fallback per the H-6 backlog item),
        and ``config.output_dir``.

        The YAML convention ``model.params.features: {...}`` is mapped
        to the constructor's ``feature_config`` argument; the alternate
        key ``feature_config:`` is also accepted.
        """
        # Resolve primary horizon index: prefer the Phase A.5 canonical
        # location, fall back to the deprecated ``data.horizon_idx`` field.
        horizon_idx: Optional[int] = None
        labels_obj = getattr(getattr(config, "data", None), "labels", None)
        if labels_obj is not None:
            horizon_idx = getattr(labels_obj, "primary_horizon_idx", None)
        if horizon_idx is None:
            horizon_idx = getattr(config.data, "horizon_idx", 0) or 0

        # Extract feature_config from params under either YAML convention.
        # Phase Q.6 post-audit fix: the previous nested-pop pattern
        # `params.pop("features", params.pop("feature_config", None))`
        # silently discarded `feature_config` when BOTH keys were present
        # because Python evaluates the inner pop unconditionally before
        # the outer. Replace with explicit precedence + fail-loud on
        # ambiguous configs per hft-rules §5.
        params = dict(config.model.params) if config.model.params else {}
        has_features = "features" in params
        has_feature_config = "feature_config" in params
        if has_features and has_feature_config:
            raise ValueError(
                "model.params contains BOTH 'features' AND 'feature_config' "
                "keys; provide only one. These are alternate YAML conventions "
                "for the same SimpleModelTrainer feature_config argument."
            )
        if has_features:
            feature_config = params.pop("features")
        elif has_feature_config:
            feature_config = params.pop("feature_config")
        else:
            feature_config = None

        # Resolve model_type to its registry-key string.
        mt = config.model.model_type
        model_type_str = mt.value if hasattr(mt, "value") else str(mt)

        instance = cls(
            data_dir=str(config.data.data_dir),
            model_type=model_type_str,
            model_config=params,
            feature_config=feature_config,
            horizon_idx=int(horizon_idx),
            output_dir=str(config.output_dir),
            **kwargs,
        )
        instance.config = config
        return instance

    def setup(self):
        """Load data, create model, engineer features.

        Phase X.3 Empirical Trust (2026-05-05) — Phase C.1: auto-derive
        ``data.labels.horizons`` from the export's ``*_horizons.json``
        files BEFORE data loading, so signal_metadata.compatibility.horizons
        reflects the actual data (not classification fallback). Mirrors
        the same hook in ``Trainer.setup()`` for cross-trainer parity.
        """
        # Phase C.1 horizons truth-pin (sklearn path)
        if self.config is not None and not self.config.data.labels.horizons:
            from lobtrainer.data.horizons_resolver import resolve_horizons_from_export

            try:
                actual_horizons = resolve_horizons_from_export(
                    self.config.data.data_dir, split="train"
                )
                new_labels = self.config.data.labels.model_copy(
                    update={"horizons": actual_horizons}
                )
                new_data = self.config.data.model_copy(
                    update={"labels": new_labels}
                )
                self.config = self.config.model_copy(
                    update={"data": new_data}
                )
                logger.info(
                    f"Auto-resolved data.labels.horizons={list(actual_horizons)} "
                    f"from {self.config.data.data_dir}/train/*_horizons.json "
                    f"(sklearn path; Phase X.3 / Phase C.1 truth-pinning)."
                )
            except FileNotFoundError as exc:
                logger.debug(f"Horizons auto-resolution skipped (sklearn): {exc}")

        # Phase Y composability / #PY-85 (2026-05-09): resolve FeatureSet to
        # populate `_feature_set_ref_resolved` cache so signal_metadata.json
        # carries feature_set_ref. Without this, harvester returns None →
        # ExperimentRecord.feature_set_ref=null → experiment_provenance_hash
        # composer skips with warning, defeating Phase Y for sklearn arms.
        # Smoke-caught 2026-05-09 (γ-1 Cycle 5 1-arm sklearn produced ledger
        # record without experiment_provenance_hash). Mirrors PyTorch Trainer
        # logic at trainer.py:460-516.
        #
        # NOTE: sklearn path does NOT subset features by index — it loads all
        # NPY columns + applies temporal engineering on top. This resolution
        # is purely for Phase Y trust-column propagation. Resolved indices are
        # stored on _feature_indices_resolved for symmetry with PyTorch but
        # unused by sklearn data loading.
        if self.config is not None:
            cfg_data = self.config.data
            feature_set_name = getattr(cfg_data, "feature_set", None)
            # Idempotency guard (mid-impl audit fix HIGH-1): if cache already
            # populated (e.g., setup() called twice via test fixtures or auto-
            # setup race in train()), skip re-resolution to avoid redundant
            # JSON parse + SHA recompute. Mirrors PyTorch where _create_data-
            # loaders is called once per Trainer lifecycle.
            already_resolved = (
                getattr(cfg_data, "_feature_set_ref_resolved", None)
                is not None
            )
            if feature_set_name is not None and not already_resolved:
                # T12 multi-source guard (mid-impl audit fix LOW-5): match
                # PyTorch trainer.py:464-469 — feature_set + sources is an
                # unsupported combination; fail-loud per hft-rules §5.
                if (
                    getattr(cfg_data, "sources", None) is not None
                ):
                    raise ValueError(
                        "feature_set is not supported with multi-source mode "
                        "(data.sources). Use per-source feature_indices on "
                        "each SourceConfig instead. (T12 + Phase 4 boundary; "
                        "sklearn parity with trainer.py:464-469)"
                    )
                from lobtrainer.data.feature_set_resolver import (
                    find_feature_sets_dir,
                    resolve_feature_set,
                )
                # Phase α-3 / #PY-79 (2026-05-10): .absolute() preserves
                # symlink-source for find_feature_sets_dir walk-up (parity
                # with trainer.py:489-494).
                if cfg_data.feature_sets_dir is not None:
                    registry_dir = Path(cfg_data.feature_sets_dir).absolute()
                else:
                    registry_dir = find_feature_sets_dir(
                        Path(cfg_data.data_dir).absolute()
                    )
                from hft_contracts import (
                    SCHEMA_VERSION as _CURRENT_CONTRACT_VERSION,
                )
                # Mid-impl audit fix HIGH-2: NO try/except wrapper. Match
                # PyTorch trainer.py:496-507 fail-loud discipline. Catching
                # FeatureSetIntegrityError (data corruption) or
                # FeatureSetContractMismatch (schema_version drift = silent-
                # wrong-result class) violates hft-rules §5 + §8. Config
                # errors and data corruption MUST surface — Phase Y
                # composability is downstream of correctness.
                resolved = resolve_feature_set(
                    name=feature_set_name,
                    registry_dir=registry_dir,
                    expected_source_feature_count=cfg_data.feature_count,
                    expected_contract_version=_CURRENT_CONTRACT_VERSION,
                )
                cfg_data._feature_indices_resolved = list(
                    resolved.feature_indices
                )
                cfg_data._feature_set_ref_resolved = (
                    resolved.name,
                    resolved.content_hash,
                )
                logger.info(
                    f"FeatureSet resolved (sklearn): '{resolved.name}' → "
                    f"{len(resolved.feature_indices)} indices "
                    f"(content_hash={resolved.content_hash[:16]}...). "
                    f"Sklearn does NOT subset features by index; "
                    f"resolution is for Phase Y "
                    f"experiment_provenance_hash composability."
                )

        logger.info("Loading data...")
        # Phase Y / γ-1 LITE / #PY-88 (2026-05-10): pass labels_config to
        # activate source-dispatched label derivation. When config is
        # populated (from_config path) AND source resolves to forward_prices,
        # _load_split recomputes labels via LabelFactory.multi_horizon
        # honoring LabelsConfig.return_type. When config is None (legacy
        # flat-keyword construction) OR source="precomputed" (per
        # LabelsConfig._VALID_SOURCES), legacy cached
        # *_regression_labels.npy path preserved for back-compat.
        labels_config_for_dispatch = (
            self.config.data.labels if self.config else None
        )
        self._seq_train, self._y_train, self._spreads_train, self._prices_train = \
            _load_split(
                self.data_dir, "train", self.horizon_idx,
                labels_config=labels_config_for_dispatch,
            )
        self._seq_val, self._y_val, _, _ = \
            _load_split(
                self.data_dir, "val", self.horizon_idx,
                labels_config=labels_config_for_dispatch,
            )
        self._seq_test, self._y_test, self._spreads_test, self._prices_test = \
            _load_split(
                self.data_dir, "test", self.horizon_idx,
                labels_config=labels_config_for_dispatch,
            )

        logger.info(f"Train: {len(self._y_train):,}, Val: {len(self._y_val):,}, Test: {len(self._y_test):,}")

        self.feat_config = TemporalFeatureConfig(**self.feature_config_dict) \
            if self.feature_config_dict else TemporalFeatureConfig()

        logger.info(f"Engineering {self.feat_config.num_features} temporal features...")
        self._X_train = engineer_temporal_features(self._seq_train, self.feat_config)
        self._X_val = engineer_temporal_features(self._seq_val, self.feat_config)
        self._X_test = engineer_temporal_features(self._seq_test, self.feat_config)

        if self.model_type == "temporal_ridge":
            cfg = TemporalRidgeConfig(
                alpha=self.model_config.get("alpha", 1.0),
                feature_config=self.feat_config,
            )
            self.model = TemporalRidge(cfg)
        elif self.model_type == "temporal_gradboost":
            cfg = TemporalGradBoostConfig(
                n_estimators=self.model_config.get("n_estimators", 200),
                max_depth=self.model_config.get("max_depth", 5),
                learning_rate=self.model_config.get("learning_rate", 0.05),
                subsample=self.model_config.get("subsample", 0.8),
                min_samples_leaf=self.model_config.get("min_samples_leaf", 50),
                loss_type=self.model_config.get("loss_type", "huber"),
                huber_delta=self.model_config.get("huber_delta", 0.9),
                max_train_samples=self.model_config.get("max_train_samples", 50000),
                feature_config=self.feat_config,
            )
            self.model = TemporalGradBoost(cfg)
        else:
            raise ValueError(f"Unknown simple model type: {self.model_type}")

        logger.info(f"Model: {self.model.name}")

    def train(self) -> Dict[str, Any]:
        """Fit the model and compute train + val metrics.

        Returns a dict shaped to match the PyTorch ``Trainer.train``
        return value so callers (``scripts/train.py``) can read
        ``total_epochs`` / ``best_val_metric`` / ``best_epoch`` keys
        polymorphically. Sklearn fit is one-shot, so sentinel values
        are emitted: ``total_epochs=1``, ``best_epoch=0``, and
        ``best_val_metric`` falls back to the negative R² (so smaller
        is better, mirroring loss conventions).

        Phase Q.6 post-live-experiment fix (2026-05-04): mirrors the
        PyTorch ``Trainer.train`` pattern of calling ``self.setup()``
        if not already done — ``scripts/train.py:429`` calls
        ``trainer.train()`` directly without an explicit setup, so the
        sklearn path must auto-setup to be a drop-in replacement.
        """
        # Auto-setup if not already done (matches Trainer.train at
        # trainer.py:797). Idempotent: setup() reloads data each call,
        # so we guard via the populated-model check.
        if self.model is None:
            self.setup()

        t0 = time.time()
        self.model.fit(self._X_train, self._y_train)
        fit_time = time.time() - t0

        y_pred_train = self.model.predict(self._X_train)
        self.train_metrics = _compute_metrics(self._y_train, y_pred_train)
        self.train_metrics["fit_time_seconds"] = round(fit_time, 2)

        y_pred_val = self.model.predict(self._X_val)
        self.val_metrics = _compute_metrics(self._y_val, y_pred_val)

        logger.info(
            f"Train: R²={self.train_metrics['r2']:.4f}, IC={self.train_metrics['ic']:.4f} | "
            f"Val: R²={self.val_metrics['r2']:.4f}, IC={self.val_metrics['ic']:.4f} | "
            f"Fit: {fit_time:.1f}s"
        )
        # Phase Q.6 (2026-05-04): augment return dict with PyTorch-shaped
        # keys for cross-trainer caller compatibility.
        result: Dict[str, Any] = {
            **self.val_metrics,
            "total_epochs": 1,
            "best_epoch": 0,
            "best_val_metric": float(-self.val_metrics.get("r2", 0.0)),
            "val_metrics": dict(self.val_metrics),
            "train_metrics": dict(self.train_metrics),
        }
        return result

    def evaluate(self, split: str = "test") -> Dict[str, float]:
        """Evaluate the trained model on the named split.

        Phase Q.6 (2026-05-04): accepts a ``split`` argument to satisfy
        ``BaseTrainer`` Protocol parity with ``Trainer.evaluate(split)``.
        """
        if split == "test":
            y_pred = self.model.predict(self._X_test)
            self.test_metrics = _compute_metrics(self._y_test, y_pred)
            return self.test_metrics
        elif split == "val":
            # If train() already ran, return cached val_metrics; else compute.
            if not self.val_metrics:
                y_pred_val = self.model.predict(self._X_val)
                self.val_metrics = _compute_metrics(self._y_val, y_pred_val)
            return self.val_metrics
        elif split == "train":
            if not self.train_metrics:
                y_pred_train = self.model.predict(self._X_train)
                self.train_metrics = _compute_metrics(self._y_train, y_pred_train)
            return self.train_metrics
        else:
            raise ValueError(
                f"Unknown split '{split}' for SimpleModelTrainer.evaluate; "
                f"expected one of 'train', 'val', 'test'."
            )

    def export_signals(
        self,
        split: str = "test",
        *,
        output_dir: Optional[Path] = None,
        calibration: str = "none",
    ) -> Path:
        """Export predictions in the backtester-compatible format.

        Phase Q.7 (2026-05-04): SSoT migration. Previously SimpleModelTrainer
        built an inline 9-field metadata dict; now uses canonical
        ``build_signal_metadata`` so sklearn signals carry the Phase II
        + Phase 4c.4 surfaces and (post-Q.6.5.A) the Phase II
        ``compatibility`` block at full parity with the PyTorch path.
        Phase Q.8 (2026-05-04): write through ``atomic_write_json``
        (tmp + fsync + os.replace) so a SIGKILL mid-write doesn't poison
        a previously-good signal directory with a partial JSON.

        Phase Q.6.5.A (2026-05-04 night): wires
        ``build_compatibility_contract`` (Phase X.1.A SSoT at
        ``lobtrainer.training.compatibility``) so sklearn
        signal_metadata.json now carries the Phase II 11-field
        ``compatibility`` block + ``compatibility_fingerprint`` +
        ``feature_set_ref`` (Phase 4 4c.4) + ``data_source`` — closes
        F-18 and unblocks Phase Y ``experiment_provenance_hash``
        composition for sklearn experiments.

        Phase Q.6.5.B (2026-05-04 night): signature extended with
        ``output_dir`` + ``calibration`` keyword-only kwargs for parity
        with ``Trainer.export_signals``. Sklearn currently rejects
        non-``"none"`` calibration per hft-rules §5 fail-fast —
        ``variance_match`` is not yet wired for the sklearn pipeline.

        Args:
            split: Data split. Sklearn currently restricts to ``"test"``
                only — ``setup()`` does not extract ``_spreads_val`` /
                ``_prices_val`` (val arrays for X/y ARE loaded but the
                spread/price columns are discarded at simple_trainer.py:218).
                Extending to ``"val"`` is a small follow-up.
            output_dir: Override default location. ``None`` uses
                ``self.output_dir / "signals" / split``.
            calibration: ``"none"`` (default) or ``"variance_match"``.
                Sklearn raises on ``"variance_match"`` until the
                calibration pipeline is wired (Phase X.6 candidate).

        Returns:
            Output directory path (``Path``).

        Raises:
            ValueError: For non-``"test"`` split or non-``"none"``
                calibration (sklearn-specific limits).
        """
        # Phase Q.6.5.B: validate calibration BEFORE any work — fail-fast
        # per hft-rules §5. variance_match for sklearn is a Phase X.6
        # candidate; document the exact restriction so operators know it's
        # not silently downgraded.
        if calibration != "none":
            raise ValueError(
                f"SimpleModelTrainer (sklearn path) does not yet support "
                f"calibration={calibration!r}. Only 'none' is wired today. "
                f"variance_match is regression-only and pending Phase X.6 "
                f"sklearn calibration cycle."
            )

        if split == "test":
            X, y, spreads, prices = self._X_test, self._y_test, self._spreads_test, self._prices_test
        else:
            raise ValueError(
                f"Only 'test' split export supported by sklearn, got {split!r}. "
                f"setup() loads val sequences/labels but discards spread/price "
                f"columns at simple_trainer.py:218 (_, _ unpacking) — extending "
                f"to 'val' requires extracting them too. Open follow-up task."
            )

        y_pred = self.model.predict(X)

        # Phase Q.6.5.B: respect explicit output_dir override (Protocol parity).
        # When None, default to <self.output_dir>/signals/<split>/ matching
        # pre-Q.6.5.B behavior.
        if output_dir is None:
            signal_dir = self.output_dir / "signals" / split
        else:
            signal_dir = Path(output_dir)
        signal_dir.mkdir(parents=True, exist_ok=True)

        # #PY-63 (2026-05-07): producer fail-loud per hft-rules §8 — silent
        # NaN propagation downstream would mask "metric crashed" as "no skill".
        from hft_contracts.validation import assert_finite_array
        assert_finite_array(
            y_pred,
            name="SimpleModelTrainer.export_signals.predicted_returns",
            extra_diagnostic=(
                f"Refusing to save corrupt predicted_returns.npy. "
                f"Investigate model_type={self.model_type!r} for numerical "
                f"divergence (loss/learning rate/feature normalization)."
            ),
        )
        np.save(signal_dir / "predicted_returns.npy", y_pred.astype(np.float64))
        np.save(signal_dir / "regression_labels.npy", y.astype(np.float64))
        np.save(signal_dir / "spreads.npy", spreads)
        np.save(signal_dir / "prices.npy", prices)

        # Lazy-import the SSoTs to avoid pulling exporter.py's PyTorch
        # dependencies into the sklearn path's module-load surface.
        from lobtrainer.export.metadata import build_signal_metadata
        from hft_contracts.atomic_io import atomic_write_json
        from lobtrainer.training.compatibility import (
            build_compatibility_contract,
            compute_model_config_hash,
        )

        # Phase Q.6.5.B (2026-05-04 night): resolve feature_set_ref via
        # ``lobtrainer.training.compatibility.feature_set_ref_to_dict``
        # SSoT (HIGH-1 lift from Q.6.5.A audit; closes 3-site duplication
        # per hft-rules §0 reuse-first). The SSoT defensively rejects
        # malformed cache values (wrong arity / empty-string components)
        # so the export does not crash on cache poisoning.
        from lobtrainer.training.compatibility import feature_set_ref_to_dict
        fs_ref_dict = (
            feature_set_ref_to_dict(self.config.data)
            if self.config is not None
            else None
        )

        # Phase Q.6.5.A (2026-05-04 night): closes F-18 — wire
        # CompatibilityContract via Phase X.1.A SSoT
        # (build_compatibility_contract). When ``self.config`` is None
        # (legacy flat-keyword construction), ``compat=None`` preserves
        # back-compat behavior and ``build_signal_metadata`` omits the
        # compatibility/fingerprint/data_source/calibration_method keys
        # (additive emission per metadata.py:154-169). The function
        # itself returns None when ``hft_contracts`` is unavailable
        # (compatibility.py:159-163), so the sklearn path gracefully
        # degrades pre-Phase-II environments.
        compat = (
            build_compatibility_contract(
                self.config,
                feature_set_ref=fs_ref_dict,
                calibration_method=None,
            )
            if self.config is not None
            else None
        )

        # Phase Y deployment (2026-05-05): compute model_config_hash for the
        # signal_metadata.json producer-side emission. Closes the harvest
        # gap that blocked Phase D experiment_provenance_hash composition
        # (model_config_hash was written only to the checkpoint sidecar
        # pre-Phase-Y; hft-ops never reads sidecars). Reuses Phase X.1.A
        # SSoT compute_model_config_hash which filters _LOSS_TUNING_KEYS
        # so changing loss-tuning hyperparams doesn't churn the hash.
        model_cfg_hash = (
            compute_model_config_hash(self.config.model)
            if self.config is not None
            else None
        )

        # Phase Y / γ-1 LITE / #PY-88 (2026-05-10): top-level human-visible
        # ``return_type`` field. Read from ``LabelsConfig.return_type`` when
        # ``self.config`` is non-None; legacy flat-keyword construction
        # passes None (additive emission — pre-#PY-88 consumers ignore).
        # The compat fingerprint already encodes return_type via
        # ``compute_label_strategy_hash`` opaque SHA — the top-level field
        # adds operator-readable surface for ledger queries.
        rt_string = (
            self.config.data.labels.return_type
            if (
                self.config is not None
                and getattr(self.config.data, "labels", None) is not None
                and getattr(self.config.data.labels, "return_type", None) is not None
            )
            else None
        )

        metadata = build_signal_metadata(
            model_type=self.model_type,
            model_name=self.model.name,
            parameters=self.model.num_parameters,
            signal_type="regression",
            split=split,
            total_samples=int(len(y_pred)),
            checkpoint=str(self.output_dir / "checkpoints" / "best.pkl"),
            horizon_idx=self.horizon_idx,
            feature_config=self.feat_config.to_dict() if self.feat_config else None,
            metrics={k: round(v, 6) for k, v in self.test_metrics.items()} if self.test_metrics else None,
            model_config_hash=model_cfg_hash,
            # Phase Q.6.5.A (2026-05-04 night): F-18 closure — sklearn
            # signal_metadata now carries Phase II compatibility block +
            # Phase 4c.4 feature_set_ref + Phase II data_source.
            # ``build_signal_metadata`` is additive — when
            # ``compatibility=None``, these keys are omitted (preserves
            # legacy flat-keyword construction back-compat).
            compatibility=compat,
            feature_set_ref=fs_ref_dict,
            data_source=compat.data_source if compat is not None else None,
            calibration_method=None,  # sklearn variance_match not yet wired
            return_type=rt_string,
        )
        atomic_write_json(signal_dir / "signal_metadata.json", metadata)

        logger.info(f"Exported {len(y_pred):,} signals to {signal_dir}")
        return signal_dir

    def save(self) -> Path:
        """Legacy save entry-point (preserved for back-compat).

        New callers should use ``save_checkpoint(path)`` (Phase Q.6
        Protocol method); this remains as a thin wrapper that delegates
        to ``save_checkpoint`` with the default location and also writes
        the ``test_metrics.json`` / ``training_history.json`` sidecar
        files.
        """
        ckpt_path = self.save_checkpoint()

        if self.test_metrics:
            with open(self.output_dir / "test_metrics.json", "w") as f:
                json.dump({f"test_{k}": v for k, v in self.test_metrics.items()}, f, indent=2)

        history = {
            "model_type": self.model_type,
            "model_name": self.model.name,
            "parameters": self.model.num_parameters,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "test_metrics": self.test_metrics,
            "feature_config": self.feat_config.to_dict() if self.feat_config else {},
        }
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        logger.info(f"Saved to {self.output_dir}")
        return ckpt_path

    def save_checkpoint(self, path: Optional[Path] = None) -> Path:
        """Persist the fitted sklearn model to disk via pickle.

        Phase Q.6 (2026-05-04): satisfies the ``BaseTrainer`` Protocol
        contract that ``scripts/train.py`` relies on. Defaults to
        ``<output_dir>/checkpoints/best.pkl`` matching the legacy
        ``save()`` location.

        Args:
            path: Override default location. Parent directory is
                created if missing.

        Returns:
            Absolute path actually written.
        """
        if path is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            ckpt_dir = self.output_dir / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            path = ckpt_dir / "best.pkl"
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

        if self.model is None:
            raise RuntimeError(
                "Cannot save_checkpoint before train(): self.model is None."
            )
        self.model.save(path)

        # Phase X.1 v2 (2026-05-04): write atomic sidecar with
        # CompatibilityContract + model_config_hash for cross-checkpoint
        # validation on load. Mirrors the PyTorch path's checkpoint dict
        # contract — same primitives, same fingerprint format.
        # Sidecar has its own schema_version='1.0' so future schema bumps
        # can migrate older sidecars.
        if self.config is not None:
            from hft_contracts.atomic_io import atomic_write_json
            from lobtrainer.training.compatibility import (
                build_compatibility_contract,
                compute_model_config_hash,
            )
            compat = build_compatibility_contract(self.config)
            sidecar_path = path.with_suffix(path.suffix + ".config.json")
            sidecar = {
                "schema_version": "1.0",  # Phase X.1 v2 sidecar schema
                "compatibility": compat.to_canonical_dict() if compat is not None else None,
                "compatibility_fingerprint": compat.fingerprint() if compat is not None else None,
                "model_config_hash": compute_model_config_hash(
                    self.config.model
                ),
                "config": self.config.to_dict(),
            }
            atomic_write_json(sidecar_path, sidecar)
            logger.info(f"Saved Phase X.1 v2 sidecar to {sidecar_path}")
        else:
            # Phase Q.6.5.F (2026-05-04 night): N-7 closure — promote silent
            # warning to DeprecationWarning so the legacy flat-keyword
            # construction path is loud per hft-rules §5 fail-fast. Operators
            # using ad-hoc Python (NOT scripts/train.py / hft-ops orchestrator)
            # who construct ``SimpleModelTrainer(data_dir=..., model_type=...,
            # ...)`` directly without ``from_config(config)`` lose Phase X.1 v2
            # sidecar coverage. Migrate to ``SimpleModelTrainer.from_config(config)``
            # before the legacy constructor's removal date (2027-04-01).
            import warnings
            warnings.warn(
                f"SimpleModelTrainer.save_checkpoint at {path} did not write a "
                f"sidecar — self.config is None (legacy flat-keyword "
                f"construction path). Future load_checkpoint cannot validate "
                f"against the active ExperimentConfig. Migrate to "
                f"SimpleModelTrainer.from_config(config) for full Phase X.1 v2 "
                f"sidecar coverage. Legacy flat-keyword constructor removal "
                f"target: 2027-04-01.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Also keep the logger.warning so log-aggregation tooling still
            # surfaces this in CI logs (Python warnings can be filtered).
            logger.warning(
                f"SimpleModelTrainer.save_checkpoint at {path} did not write a "
                f"Phase X.1 v2 sidecar (legacy flat-keyword construction)."
            )

        return path

    def load_checkpoint(
        self,
        path: Path,
        load_optimizer: bool = True,
        strict_config: bool = False,
    ) -> None:
        """Restore the sklearn model from a pickle checkpoint.

        Phase X.1 v2 (2026-05-04): validates the sidecar at
        ``<path>.config.json`` against the active config. Mirrors
        ``Trainer.load_checkpoint`` semantics:
          * ``strict_config=False`` (default): warn on mismatch.
          * ``strict_config=True``: raise CheckpointConfigMismatchError.
          * Sidecar missing: warn CheckpointMissingFingerprintWarning.
          * Sidecar parse error: warn + skip validation.
          * Sidecar parses but lacks contract keys (partial-write or
            pre-X.1 v2 schema): warn (Agent 4 sanity-check Q10 fix).

        Phase Q.6.5.B (2026-05-04 night): ``load_optimizer`` kwarg added
        to satisfy the unified ``BaseTrainer`` Protocol signature. On
        sklearn (this class), it is a documented no-op — there is no
        optimizer state to load from the pickle. Without this kwarg, a
        polymorphic caller passing ``load_optimizer=False`` (the canonical
        signal-export pattern at ``scripts/export_signals.py``) would
        TypeError on the sklearn path. Closes N-6 signature drift.

        Args:
            path: Pickle file written by ``save_checkpoint`` or
                legacy ``save``.
            load_optimizer: PyTorch-only. Documented no-op on sklearn —
                this class has no optimizer state to load. Accepted for
                Protocol parity with ``Trainer.load_checkpoint``.
            strict_config: When True, raise on fingerprint mismatch.

        Raises:
            ValueError: If ``self.model_type`` is not a known sklearn
                registry entry.
        """
        # load_optimizer is a documented no-op on the sklearn path
        # (no optimizer in the pickle). Reference the kwarg here so the
        # signature is exercised — silent-discard would surface as an
        # unused-argument lint warning.
        del load_optimizer  # explicit no-op: sklearn pickle has no optimizer state
        import warnings
        import json
        from lobtrainer.training.compatibility import (
            build_compatibility_contract,
            compute_model_config_hash,
            CheckpointConfigMismatchError,
            CheckpointConfigMismatchWarning,
            CheckpointMissingFingerprintWarning,
        )

        path = Path(path)

        # Phase X.1 v2 sidecar validation BEFORE pickle load.
        sidecar_path = path.with_suffix(path.suffix + ".config.json")
        if not sidecar_path.exists():
            warnings.warn(
                f"SimpleModelTrainer checkpoint at {path} lacks sidecar "
                f"({sidecar_path}). Pre-X.1 v2 artifact — cannot validate "
                f"against active config.",
                CheckpointMissingFingerprintWarning,
                stacklevel=2,
            )
        else:
            try:
                with open(sidecar_path) as f:
                    sidecar = json.load(f)
            except json.JSONDecodeError as exc:
                warnings.warn(
                    f"Sidecar at {sidecar_path} is corrupt JSON: {exc}. "
                    f"Skipping validation.",
                    CheckpointMissingFingerprintWarning,
                    stacklevel=2,
                )
                sidecar = {}

            ckpt_compat_dict = sidecar.get("compatibility")
            ckpt_compat_fingerprint = sidecar.get("compatibility_fingerprint")
            ckpt_model_cfg_hash = sidecar.get("model_config_hash")

            # Per Agent 4 sanity check Q10: warn when sidecar parses cleanly but
            # lacks BOTH contract keys (partial-write or pre-X.1 v2 schema).
            if ckpt_compat_fingerprint is None and ckpt_model_cfg_hash is None and sidecar:
                warnings.warn(
                    f"Sidecar at {sidecar_path} parsed but lacks both "
                    f"'compatibility_fingerprint' and 'model_config_hash' keys. "
                    f"Pre-X.1 v2 schema or partial-write. Skipping validation.",
                    CheckpointMissingFingerprintWarning,
                    stacklevel=2,
                )
            elif self.config is not None:
                # Build active contracts for comparison
                active_compat = build_compatibility_contract(self.config)
                active_model_cfg_hash = compute_model_config_hash(
                    self.config.model
                )

                # Compatibility-contract mismatch
                if (
                    ckpt_compat_fingerprint is not None
                    and active_compat is not None
                    and ckpt_compat_fingerprint != active_compat.fingerprint()
                ):
                    from hft_contracts.compatibility import CompatibilityContract
                    try:
                        # Phase X.1 v2 sanity-check fix: no from_dict classmethod
                        # exists; reconstruct via dict expansion (post_init validates).
                        ckpt_compat = CompatibilityContract(
                            **(ckpt_compat_dict or {})
                        )
                        diff = active_compat.diff(ckpt_compat)
                        diff_msg = (
                            f"Sidecar compatibility mismatch at {sidecar_path}.\n"
                            f"  Differing fields (active vs sidecar): {diff}"
                        )
                    except Exception:
                        diff_msg = (
                            f"Sidecar compatibility_fingerprint mismatch at "
                            f"{sidecar_path}: ckpt={ckpt_compat_fingerprint[:16]}, "
                            f"active={active_compat.fingerprint()[:16]}"
                        )
                    if strict_config:
                        raise CheckpointConfigMismatchError(diff_msg)
                    else:
                        warnings.warn(
                            diff_msg,
                            CheckpointConfigMismatchWarning,
                            stacklevel=2,
                        )

                # Model-config-hash mismatch
                if (
                    ckpt_model_cfg_hash is not None
                    and ckpt_model_cfg_hash != active_model_cfg_hash
                ):
                    msg = (
                        f"Sidecar model_config_hash mismatch at {sidecar_path}: "
                        f"ckpt={ckpt_model_cfg_hash[:16]}, "
                        f"active={active_model_cfg_hash[:16]}. "
                        f"Likely cause: model_type/hidden_dim/num_layers/dropout "
                        f"or other architectural keys differ."
                    )
                    if strict_config:
                        raise CheckpointConfigMismatchError(msg)
                    else:
                        warnings.warn(
                            msg,
                            CheckpointConfigMismatchWarning,
                            stacklevel=2,
                        )

        # Existing sklearn pickle load (preserved verbatim).
        if self.model_type == "temporal_ridge":
            from lobmodels.models.simple import TemporalRidge
            self.model = TemporalRidge.load(path)
        elif self.model_type == "temporal_gradboost":
            from lobmodels.models.simple import TemporalGradBoost
            self.model = TemporalGradBoost.load(path)
        else:
            raise ValueError(
                f"Cannot load_checkpoint for unknown sklearn model_type "
                f"'{self.model_type}'; expected 'temporal_ridge' or "
                f"'temporal_gradboost'."
            )
        logger.info(f"Loaded {self.model_type} checkpoint from {path}")
