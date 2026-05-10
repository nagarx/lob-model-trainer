"""
Unified signal exporter for all model types.

Replaces the 3 separate signal export scripts with a single Trainer-integrated
export path. Uses the Trainer's DataLoader for normalized inference and
RawFeatureExtractor for raw spread/price from disk.

Produces backtester-compatible signal files matching the contract in
lob-backtester/src/lobbacktest/data/signal_manifest.py.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from lobtrainer.config.paths import resolve_labels_config
from lobtrainer.export.raw_features import RawFeatureExtractor
from lobtrainer.export.metadata import build_signal_metadata
# Phase X.1 v2 (2026-05-04): build_compatibility_contract + derive_data_source moved
# to lobtrainer.training.compatibility (shared with Trainer.save_checkpoint +
# SimpleModelTrainer.save_checkpoint sidecar). NO behavior change here — same body.
from lobtrainer.training.compatibility import (
    build_compatibility_contract,
    compute_model_config_hash,
    derive_data_source,
)

# Phase X.1 v2 (2026-05-04): back-compat aliases for callers that still import
# the pre-X.1 underscore-prefixed names. New callers should use the public
# names re-exported above. These aliases will be retained until 2026-10-31
# (matching hft-contracts._atomic_io shim deadline pattern).
_build_compatibility_contract = build_compatibility_contract
_derive_data_source = derive_data_source

logger = logging.getLogger(__name__)


def _feature_set_ref_dict(data_config: Any) -> Optional[Dict[str, str]]:
    """Extract feature_set_ref dict from DataConfig private cache.

    Phase Q.6.5.B (2026-05-04 night): retained as a 1-line delegation to
    the canonical SSoT at ``lobtrainer.training.compatibility``. Direct
    callers should migrate to
    ``compatibility.feature_set_ref_to_dict(data_config)``; this shim
    preserves any test/code that imports ``_feature_set_ref_dict`` from
    exporter.py (e.g., ``tests/test_signal_metadata_feature_set_ref.py``).

    The SSoT is more defensive than the original — it returns None on
    malformed cache (wrong arity / non-tuple) and on empty-string
    components, instead of raising ValueError. Tests that constructed
    ``_feature_set_ref_resolved = ("name", "")`` previously got the
    malformed dict; post-Q.6.5.B they get None (defensive cache-poisoning
    rejection per hft-rules §5).
    """
    from lobtrainer.training.compatibility import feature_set_ref_to_dict
    return feature_set_ref_to_dict(data_config)


@dataclass
class ExportResult:
    """Result of a signal export operation."""
    output_dir: Path
    n_samples: int
    signal_type: str
    files_written: List[str]
    metadata: Dict[str, Any]


class SignalExporter:
    """Unified signal exporter for all strategy types.

    Delegates to the Trainer for model setup, normalization, feature selection,
    and DataLoader construction. Extracts raw spread/price from disk via
    RawFeatureExtractor. Handles all 4 strategy types:

    - ClassificationStrategy: predictions [N] int + labels
    - HMHPClassificationStrategy: + agreement, confidence
    - RegressionStrategy: predicted_returns [N] float + regression labels
    - HMHPRegressionStrategy: predicted_returns [N,H] float (multi-horizon)

    Usage:
        trainer = Trainer(config)
        trainer.setup()
        trainer.load_checkpoint(checkpoint_path, load_optimizer=False)

        exporter = SignalExporter(trainer)
        result = exporter.export(split="test", output_dir=output_dir)
    """

    def __init__(
        self,
        trainer: "Trainer",
        *,
        calibration: str = "none",
    ):
        self._trainer = trainer
        self._calibration = calibration

    def export(
        self,
        split: str = "test",
        output_dir: Optional[Path] = None,
    ) -> ExportResult:
        """Export signals for a data split.

        Args:
            split: Data split ("val" or "test"). Training split is refused.
            output_dir: Output directory. Created if it doesn't exist.

        Returns:
            ExportResult with file list and metadata.
        """
        if split == "train":
            raise ValueError(
                "Cannot export training split — DataLoader uses drop_last=True, "
                "causing prediction/raw-feature alignment mismatch. Use 'val' or 'test'."
            )

        loader = self._trainer.get_loader(split)
        if loader is None:
            raise ValueError(
                f"No DataLoader for split '{split}'. "
                f"Did you call trainer.setup()?"
            )

        if output_dir is None:
            output_dir = Path(self._trainer.config.output_dir) / "signals" / split
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Run inference through the Trainer's DataLoader
        logger.info(f"Running inference on '{split}' split...")
        inference = self._run_inference(loader)

        # 2. Extract raw spread/price from disk
        logger.info("Extracting raw spread/price from disk...")
        data_dir = Path(self._trainer.config.data.data_dir)
        raw = RawFeatureExtractor(data_dir, split).extract()

        # 3. Alignment verification
        n_inf = inference["n_samples"]
        n_raw = raw.n_samples
        if n_inf != n_raw:
            raise RuntimeError(
                f"Alignment mismatch: inference produced {n_inf:,} samples "
                f"but raw features have {n_raw:,} samples. "
                f"This indicates a DataLoader/file ordering discrepancy."
            )
        logger.info(f"Alignment verified: {n_inf:,} samples")

        # 4. Apply calibration if requested
        signal_type = inference["signal_type"]
        calibration_result = None
        if self._calibration == "variance_match" and signal_type == "regression":
            calibration_result = self._apply_calibration(inference)

        # 5. Write signal files
        files_written = self._write_files(
            output_dir, inference, raw, calibration_result
        )

        # 6. Build and write metadata
        # Phase Q.8 (2026-05-04): atomic write via SSoT. Previously a
        # naive ``with open(...) as f; json.dump`` could leave a partial
        # JSON file on SIGKILL/ENOSPC mid-write, poisoning a
        # previously-good signal directory. ``atomic_write_json``
        # (tmp + fsync + os.replace + cleanup) is the canonical writer
        # convention used by ``ExperimentRecord.save``,
        # ``hft_ops.feature_sets.writer``, and ``hft_ops.ledger._save_index``.
        from hft_contracts.atomic_io import atomic_write_json
        metadata = self._build_metadata(
            inference, raw, split, output_dir, calibration_result
        )
        meta_path = output_dir / "signal_metadata.json"
        atomic_write_json(meta_path, metadata)
        files_written.append("signal_metadata.json")

        logger.info(
            f"Exported {n_inf:,} {signal_type} signals to {output_dir} "
            f"({len(files_written)} files)"
        )

        return ExportResult(
            output_dir=output_dir,
            n_samples=n_inf,
            signal_type=signal_type,
            files_written=files_written,
            metadata=metadata,
        )

    @torch.no_grad()
    def _run_inference(self, loader) -> Dict[str, Any]:
        """Run model inference and collect per-sample outputs.

        Dispatches on strategy type to extract the correct ModelOutput fields.
        """
        from lobtrainer.training.strategies.classification import ClassificationStrategy
        from lobtrainer.training.strategies.regression import RegressionStrategy
        from lobtrainer.training.strategies.hmhp_classification import HMHPClassificationStrategy
        from lobtrainer.training.strategies.hmhp_regression import HMHPRegressionStrategy

        model = self._trainer.model
        device = self._trainer.device
        strategy = self._trainer.strategy
        model.eval()

        result: Dict[str, Any] = {"n_samples": 0}

        if isinstance(strategy, HMHPClassificationStrategy):
            result.update(self._infer_hmhp_classification(model, device, loader))
        elif isinstance(strategy, HMHPRegressionStrategy):
            result.update(self._infer_hmhp_regression(model, device, loader))
        elif isinstance(strategy, RegressionStrategy):
            result.update(self._infer_regression(model, device, loader))
        elif isinstance(strategy, ClassificationStrategy):
            result.update(self._infer_classification(model, device, loader))
        else:
            raise ValueError(f"Unknown strategy type: {type(strategy).__name__}")

        return result

    def _infer_classification(self, model, device, loader) -> Dict[str, Any]:
        """Standard classification: logits → argmax predictions."""
        all_preds = []
        all_labels = []

        for features, labels in loader:
            features = features.to(device)
            output = model(features)
            preds = output.logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())

        predictions = np.concatenate(all_preds).astype(np.int32)
        labels = np.concatenate(all_labels).astype(np.int32)

        return {
            "signal_type": "classification",
            "n_samples": len(predictions),
            "predictions": predictions,
            "labels": labels,
        }

    def _infer_hmhp_classification(self, model, device, loader) -> Dict[str, Any]:
        """HMHP classification: predictions + agreement + confidence."""
        all_preds = []
        all_labels = []
        all_agreement = []
        all_confidence = []

        for batch_data in loader:
            if len(batch_data) == 3:
                features, label_dict, _ = batch_data
            else:
                features, label_dict = batch_data

            features = features.to(device)
            output = model(features)

            preds = output.logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)

            # Primary horizon labels
            first_h = sorted(label_dict.keys())[0]
            all_labels.append(label_dict[first_h].numpy())

            # Agreement (always populated by HMHP)
            if output.agreement is not None:
                all_agreement.append(output.agreement.squeeze(-1).cpu().numpy())

            # Confidence (always populated by HMHP classifier, None for HMHP-R)
            if output.confidence is not None:
                all_confidence.append(output.confidence.squeeze(-1).cpu().numpy())

        predictions = np.concatenate(all_preds).astype(np.int32)
        labels = np.concatenate(all_labels).astype(np.int32)

        result = {
            "signal_type": "classification",
            "n_samples": len(predictions),
            "predictions": predictions,
            "labels": labels,
        }
        if all_agreement:
            result["agreement_ratio"] = np.concatenate(all_agreement).astype(np.float64)
        if all_confidence:
            result["confirmation_score"] = np.concatenate(all_confidence).astype(np.float64)

        return result

    def _infer_regression(self, model, device, loader) -> Dict[str, Any]:
        """Standard regression: single-horizon predictions."""
        all_preds = []
        all_labels = []

        for features, labels in loader:
            features = features.to(device)
            output = model(features)
            all_preds.append(output.predictions.cpu().numpy())
            all_labels.append(labels.numpy())

        predicted_returns = np.concatenate(all_preds).astype(np.float64)
        regression_labels = np.concatenate(all_labels).astype(np.float64)

        # #PY-63 (2026-05-07): producer fail-loud per hft-rules §8.
        from hft_contracts.validation import assert_finite_array
        assert_finite_array(
            predicted_returns,
            name="SignalExporter._infer_regression.predicted_returns",
            extra_diagnostic=(
                "Refusing to emit corrupt inference output. Investigate "
                "numerical stability (loss divergence, learning rate, "
                "feature normalization)."
            ),
        )

        return {
            "signal_type": "regression",
            "n_samples": len(predicted_returns),
            "predicted_returns": predicted_returns,
            "regression_labels": regression_labels,
        }

    def _infer_hmhp_regression(self, model, device, loader) -> Dict[str, Any]:
        """HMHP regression: multi-horizon predictions [N, H]."""
        horizon_preds: Dict[int, List[np.ndarray]] = {}
        horizon_labels: Dict[int, List[np.ndarray]] = {}
        all_agreement = []

        for batch_data in loader:
            if len(batch_data) == 3:
                features, _, reg_targets = batch_data
            else:
                raise ValueError(
                    "HMHPRegressionStrategy requires 3-tuple batch "
                    "(features, labels_dict, reg_targets_dict)"
                )

            features = features.to(device)
            output = model(features)

            # Collect per-horizon predictions
            if output.horizon_predictions is not None:
                for h, pred in output.horizon_predictions.items():
                    if h not in horizon_preds:
                        horizon_preds[h] = []
                        horizon_labels[h] = []
                    horizon_preds[h].append(pred.squeeze(-1).cpu().numpy())
                    if h in reg_targets:
                        horizon_labels[h].append(reg_targets[h].numpy())

            if output.agreement is not None:
                all_agreement.append(output.agreement.squeeze(-1).cpu().numpy())

        # Stack into [N, H] arrays, sorted by horizon
        sorted_horizons = sorted(horizon_preds.keys())
        predicted_returns = np.column_stack(
            [np.concatenate(horizon_preds[h]) for h in sorted_horizons]
        ).astype(np.float64)

        # #PY-63 (2026-05-07): producer fail-loud (multi-horizon variant).
        from hft_contracts.validation import assert_finite_array
        assert_finite_array(
            predicted_returns,
            name="SignalExporter._infer_hmhp_regression.predicted_returns",
            extra_diagnostic=(
                "Refusing to emit corrupt multi-horizon inference output."
            ),
        )

        regression_labels = None
        if all(h in horizon_labels and horizon_labels[h] for h in sorted_horizons):
            regression_labels = np.column_stack(
                [np.concatenate(horizon_labels[h]) for h in sorted_horizons]
            ).astype(np.float64)

        result: Dict[str, Any] = {
            "signal_type": "regression",
            "n_samples": predicted_returns.shape[0],
            "predicted_returns": predicted_returns,
            "horizons": sorted_horizons,
        }
        if regression_labels is not None:
            result["regression_labels"] = regression_labels
        if all_agreement:
            agreement_ratio_arr = np.concatenate(all_agreement).astype(np.float64)
            # #PY-63 (2026-05-07): agreement_ratio is HMHP-R model output —
            # producer fail-loud per §8 (cross-horizon decoder confidence;
            # NaN here would silently degrade ReadabilityHybridStrategy
            # gate computation downstream).
            from hft_contracts.validation import assert_finite_array
            assert_finite_array(
                agreement_ratio_arr,
                name="SignalExporter._infer_hmhp_regression.agreement_ratio",
                extra_diagnostic=(
                    "Refusing to emit corrupt cross-horizon agreement output."
                ),
            )
            result["agreement_ratio"] = agreement_ratio_arr

        return result

    def _apply_calibration(self, inference: Dict) -> Optional[Dict]:
        """Apply variance-matching calibration to regression predictions.

        Returns dict with calibration stats AND the calibrated array:
        {**CalibrationResult.to_dict(), "calibrated": np.ndarray}.
        CalibrationResult.to_dict() intentionally omits the ndarray,
        so we add it back for _write_files to save as calibrated_returns.npy.
        """
        from lobtrainer.calibration.variance import (
            calibrate_variance,
            VarianceCalibrationConfig,
        )

        preds = inference["predicted_returns"]
        labels = inference.get("regression_labels")

        # Phase A (2026-04-23): respect config.data.labels.primary_horizon_idx
        # when slicing 2-D predictions/labels. Pre-Phase-A hardcoded [:, 0]
        # silently mis-calibrated every HMHP-R experiment with
        # primary_horizon_idx != 0. `calibrate_variance` is strict 1-D now —
        # 2-D input raises ValueError (hft-rules §8 fail-loud).
        #
        # Phase A.5.4 (2026-04-24): use LabelsConfig.validate_primary_horizon_idx_for
        # instead of ``or 0`` pattern. Closes plan v4 bugs #2 (negative idx
        # silently picks last-N) + #5 (out-of-bounds silently returns wrong
        # column). The validator raises ValueError with diagnostic on any
        # invalid idx — the caller gets fail-loud BEFORE slicing.
        labels_cfg = resolve_labels_config(self._trainer.config)
        if preds.ndim == 2:
            primary_idx = labels_cfg.validate_primary_horizon_idx_for(
                n_horizons=preds.shape[-1]
            )
            preds_1d = preds[:, primary_idx]
            labels_1d = labels[:, primary_idx] if labels is not None else None
        else:
            primary_idx = labels_cfg.primary_horizon_idx or 0
            preds_1d = preds
            labels_1d = labels

        config = VarianceCalibrationConfig(method="variance_match")
        # Phase A.5.5 (2026-04-24): ``metadata=`` kwarg renamed to
        # ``context=`` + typed via ``CalibrationContext`` TypedDict.
        # Wire-format JSON key in signal_metadata.json stays ``"metadata"``
        # (preserved for downstream consumer compat).
        cal_result = calibrate_variance(
            preds_1d,
            labels_1d,
            config,
            context={"primary_horizon_idx": primary_idx},
        )

        logger.info(
            f"Calibration: scale={cal_result.scale_factor:.4f}, "
            f"pred_std={cal_result.pred_std:.2f} → target_std={cal_result.target_std:.2f}"
        )

        # to_dict() omits the ndarray — add it back for file writing
        result = cal_result.to_dict()

        # #PY-63 (2026-05-07): calibration producer fail-loud — variance_match
        # produces NaN/Inf when pred_std=0 (constant predictions) or label_std=0.
        # Without this, corrupt calibrated_returns.npy silently overrides
        # predicted_returns.npy at backtester load per signal_manifest D10
        # calibration precedence, biasing P&L computation downstream.
        from hft_contracts.validation import assert_finite_array
        assert_finite_array(
            np.asarray(cal_result.calibrated),
            name="SignalExporter._apply_calibration.calibrated",
            extra_diagnostic=(
                "Calibration produced corrupt output (likely "
                "scale_factor=NaN/Inf from zero-variance inputs). "
                "Investigate variance_match input data quality."
            ),
        )

        result["calibrated"] = cal_result.calibrated
        return result

    def _write_files(
        self,
        output_dir: Path,
        inference: Dict,
        raw: "RawFeatures",
        calibration_result: Optional[Dict],
    ) -> List[str]:
        """Write .npy signal files to output directory."""
        files = []

        # #PY-73 atomic writes — SIGKILL mid-write would leave partial .npy
        # files that downstream backtester silently consumes. Migrated
        # 2026-05-11 (hft-contracts v2.7.0).
        from hft_contracts.atomic_io import atomic_write_npy

        # Always write prices and spreads
        atomic_write_npy(output_dir / "prices.npy", raw.prices)
        files.append("prices.npy")
        atomic_write_npy(output_dir / "spreads.npy", raw.spreads)
        files.append("spreads.npy")

        # Classification outputs
        if "predictions" in inference:
            atomic_write_npy(output_dir / "predictions.npy", inference["predictions"])
            files.append("predictions.npy")
        if "labels" in inference:
            atomic_write_npy(output_dir / "labels.npy", inference["labels"])
            files.append("labels.npy")

        # Regression outputs
        if "predicted_returns" in inference:
            atomic_write_npy(output_dir / "predicted_returns.npy", inference["predicted_returns"])
            files.append("predicted_returns.npy")
        if "regression_labels" in inference:
            atomic_write_npy(output_dir / "regression_labels.npy", inference["regression_labels"])
            files.append("regression_labels.npy")

        # HMHP-specific
        if "agreement_ratio" in inference:
            atomic_write_npy(output_dir / "agreement_ratio.npy", inference["agreement_ratio"])
            files.append("agreement_ratio.npy")
        if "confirmation_score" in inference:
            atomic_write_npy(output_dir / "confirmation_score.npy", inference["confirmation_score"])
            files.append("confirmation_score.npy")

        # Calibrated returns
        if calibration_result is not None and "calibrated" in calibration_result:
            cal_arr = np.array(calibration_result["calibrated"], dtype=np.float64)
            atomic_write_npy(output_dir / "calibrated_returns.npy", cal_arr)
            files.append("calibrated_returns.npy")

        return files

    def _build_metadata(
        self,
        inference: Dict,
        raw: "RawFeatures",
        split: str,
        output_dir: Path,
        calibration_result: Optional[Dict],
    ) -> Dict[str, Any]:
        """Build comprehensive signal metadata."""
        config = self._trainer.config
        model = self._trainer.model

        model_name = getattr(model, "name", model.__class__.__name__)
        num_params = sum(p.numel() for p in model.parameters())

        # Determine model_type string
        from lobtrainer.training.strategies.hmhp_classification import HMHPClassificationStrategy
        from lobtrainer.training.strategies.hmhp_regression import HMHPRegressionStrategy

        strategy = self._trainer.strategy
        if isinstance(strategy, HMHPRegressionStrategy):
            model_type_str = "hmhp_regression"
        elif isinstance(strategy, HMHPClassificationStrategy):
            model_type_str = "hmhp"
        else:
            model_type_str = config.model.name

        # Spread stats
        spread_stats = {
            "mean": float(np.mean(raw.spreads)),
            "median": float(np.median(raw.spreads)),
            "p90": float(np.percentile(raw.spreads, 90)),
        }

        # Price stats
        price_stats = {
            "mean": float(np.mean(raw.prices)),
            "min": float(np.min(raw.prices)),
            "max": float(np.max(raw.prices)),
        }

        # Classification-specific metadata
        predictions_distribution = None
        directional_rate = None
        agreement_stats = None
        confirmation_percentiles = None

        if inference["signal_type"] == "classification" and "predictions" in inference:
            preds = inference["predictions"]
            predictions_distribution = {
                "Down": int(np.sum(preds == 0)),
                "Stable": int(np.sum(preds == 1)),
                "Up": int(np.sum(preds == 2)),
            }
            directional_rate = float(np.mean(preds != 1))

            if "agreement_ratio" in inference:
                ag = inference["agreement_ratio"]
                agreement_stats = {
                    "mean": float(np.mean(ag)),
                    "full_agreement": int(np.sum(ag > 0.99)),
                    "partial": int(np.sum(ag <= 0.99)),
                }
            if "confirmation_score" in inference:
                cs = inference["confirmation_score"]
                confirmation_percentiles = {
                    "p25": float(np.percentile(cs, 25)),
                    "p50": float(np.percentile(cs, 50)),
                    "p75": float(np.percentile(cs, 75)),
                    "p99": float(np.percentile(cs, 99)),
                }

        # Regression-specific metadata
        metrics_dict = None
        prediction_stats = None

        if "predicted_returns" in inference:
            pr = inference["predicted_returns"]
            # Phase A (2026-04-23) — bug #6b: stats for 2-D predictions MUST
            # slice by the configured primary_horizon_idx, not hardcoded [:, 0].
            # The stats land in signal_metadata.json and are read by
            # hft-ops statistical_compare; wrong-column slicing silently
            # misrepresented the primary horizon for every HMHP-R experiment.
            #
            # Phase A.5.4 (2026-04-24): replaced ``or 0`` pattern with
            # ``validate_primary_horizon_idx_for`` — raises on negative /
            # out-of-bounds idx (plan v4 bug #2). Single canonical source;
            # stats_idx, metrics slice, AND calibration slice all share
            # validation via the method.
            #
            # Phase 1 N6 forensic-bug closure (#PY-10, 2026-05-06):
            # When ``--calibrate variance_match`` was applied, primary metrics
            # MUST describe the calibrated array (which is what backtester
            # loads via signal_dir/calibrated_returns.npy). Pre-fix used raw
            # pr unconditionally — signal_metadata.json silently reported
            # raw-prediction characteristics for experiments claiming
            # calibration_method != None. Now: when calibration_result has
            # "calibrated" key, primary stats + metrics describe calibrated
            # array; raw-prediction stats preserved under "raw_prediction_stats"
            # for back-compat / transparency.
            if calibration_result is not None and "calibrated" in calibration_result:
                primary_arr = np.asarray(
                    calibration_result["calibrated"], dtype=np.float64
                )
                raw_arr_for_stats = pr  # preserved for raw_prediction_stats
            else:
                primary_arr = pr
                raw_arr_for_stats = None

            labels_cfg_stats = resolve_labels_config(config)
            if primary_arr.ndim == 1:
                prediction_stats = {
                    "mean": float(np.mean(primary_arr)),
                    "std": float(np.std(primary_arr)),
                    "min": float(np.min(primary_arr)),
                    "max": float(np.max(primary_arr)),
                }
            else:
                stats_idx = labels_cfg_stats.validate_primary_horizon_idx_for(
                    n_horizons=primary_arr.shape[-1]
                )
                prediction_stats = {
                    "mean": float(np.mean(primary_arr[:, stats_idx])),
                    "std": float(np.std(primary_arr[:, stats_idx])),
                }

            # N6 closure: also record raw-prediction stats when calibration applied
            # so reviewers can compare raw vs calibrated distributions.
            if raw_arr_for_stats is not None:
                if raw_arr_for_stats.ndim == 1:
                    prediction_stats["raw_mean"] = float(np.mean(raw_arr_for_stats))
                    prediction_stats["raw_std"] = float(np.std(raw_arr_for_stats))
                else:
                    prediction_stats["raw_mean"] = float(
                        np.mean(raw_arr_for_stats[:, stats_idx])
                    )
                    prediction_stats["raw_std"] = float(
                        np.std(raw_arr_for_stats[:, stats_idx])
                    )

            if "regression_labels" in inference:
                rl = inference["regression_labels"]
                try:
                    from lobtrainer.training.regression_metrics import (
                        compute_all_regression_metrics,
                    )
                    if primary_arr.ndim == 1:
                        metrics_dict = compute_all_regression_metrics(rl, primary_arr)
                    else:
                        # stats_idx already validated above for primary_arr.ndim==2 branch.
                        metrics_dict = compute_all_regression_metrics(
                            rl[:, stats_idx], primary_arr[:, stats_idx]
                        )
                except Exception as e:
                    logger.warning(f"Could not compute regression metrics: {e}")

        # Horizons
        horizons = inference.get("horizons") or getattr(config.model, "hmhp_horizons", None)
        # Phase A (2026-04-23): canonical source is ``config.data.labels.primary_horizon_idx``
        # (via helper), not the deprecated ``DataConfig.horizon_idx`` field. See
        # schema.py:1472 for the deprecation. The helper also raises ``AttributeError``
        # loudly when the canonical path is unavailable, so ``horizon_idx is None``
        # here strictly means ``primary_horizon_idx`` was explicitly set to ``None``.
        #
        # Phase A.5.4 (2026-04-24) — plan v4 bug #5: pre-A.5.4 had a silent
        # fallback ``primary_horizon = None`` when ``horizon_idx >= len(horizons)``
        # (no diagnostic logged). hft-rules §8 — "never silently drop": now
        # use validate_primary_horizon_idx_for which raises with an actionable
        # diagnostic. Wrap in try/except so the metadata still emits
        # (primary_horizon=None) but the WARN log surfaces the misconfiguration.
        horizon_idx = resolve_labels_config(config).primary_horizon_idx
        primary_horizon = None
        if horizons and horizon_idx is not None:
            try:
                valid_idx = resolve_labels_config(config).validate_primary_horizon_idx_for(
                    n_horizons=len(horizons)
                )
                primary_horizon = f"H{horizons[valid_idx]}"
            except ValueError as e:
                logger.warning(
                    "Primary horizon label emission skipped — "
                    "primary_horizon_idx=%r vs len(horizons)=%d: %s. "
                    "Metadata will record primary_horizon=None; fix "
                    "config.data.labels.primary_horizon_idx to eliminate.",
                    horizon_idx, len(horizons), e,
                )

        # Phase II (2026-04-20): construct CompatibilityContract BEFORE passing to
        # build_signal_metadata so the producer fingerprint is pinned at export time.
        # The manifest consumer validates by recomputing from the serialized block —
        # any tampering between producer and consumer is detected.
        fs_ref = _feature_set_ref_dict(config.data)
        # SB-3 Phase II hardening (2026-04-20): derive calibration_method from the
        # APPLIED state (calibration_result is not None) rather than the configured
        # flag (self._calibration != "none"). The configured flag was set to
        # "variance_match" even on classification paths, where `_apply_calibration`
        # short-circuits (regression-only). That produced a manifest claiming
        # calibration with no calibrated_returns.npy file — which raises at LOAD
        # time via the D10 orphan-file rule. Now: calibration_method reflects
        # what actually happened, not what the caller asked for. A WARN log
        # surfaces the mismatch so the operator knows the --calibrate flag was
        # a no-op on their signal_type.
        configured_calibration = getattr(self, "_calibration", "none")
        calibration_applied = calibration_result is not None
        signal_type = inference.get("signal_type", "unknown")
        if configured_calibration != "none" and not calibration_applied:
            logger.warning(
                "Calibration flag %r was configured but not applied (signal_type=%r). "
                "variance_match is regression-only. Setting calibration_method=None "
                "in signal_metadata.json so the manifest matches the files on disk.",
                configured_calibration, signal_type,
            )
        calibration_method = (
            configured_calibration if calibration_applied else None
        )
        compatibility = build_compatibility_contract(
            config=config,
            feature_set_ref=fs_ref,
            calibration_method=calibration_method,
        )
        data_source_tag = derive_data_source(config.data.data_dir)

        # Phase Y deployment (2026-05-05): emit model_config_hash at signal-
        # metadata root so hft-ops harvester can read it back via the
        # signal_metadata.json contract (mirrors compatibility_fingerprint).
        # Closes the cross-repo harvest gap that blocked Phase D
        # experiment_provenance_hash composition (model_config_hash was
        # written only to the .pt checkpoint dict pre-Phase-Y; hft-ops never
        # reads checkpoints). _LOSS_TUNING_KEYS denylist preserves stability
        # under loss-tuning param changes.
        model_cfg_hash = compute_model_config_hash(config.model)

        # Phase Y / γ-1 LITE / #PY-88 (2026-05-10): top-level human-visible
        # ``return_type`` symmetric with sklearn path's
        # ``simple_trainer.export_signals`` emission. The fingerprint
        # already encodes return_type opaquely; this adds a string-readable
        # surface for ledger / backtester / dashboard queries that filter
        # by return_type axis.
        rt_string = getattr(
            getattr(config.data, "labels", None), "return_type", None
        )

        return build_signal_metadata(
            model_type=model_type_str,
            model_name=str(model_name),
            parameters=num_params,
            signal_type=inference["signal_type"],
            split=split,
            total_samples=inference["n_samples"],
            checkpoint=str(output_dir.parent / "checkpoints" / "best.pt"),
            config_path=None,  # Set by CLI if known; ExperimentConfig doesn't store path
            horizons=horizons,
            primary_horizon=primary_horizon,
            horizon_idx=horizon_idx,
            data_dir=str(config.data.data_dir),
            feature_preset=getattr(config.data, "feature_preset", None),
            feature_set_ref=fs_ref,
            normalization_strategy=str(getattr(config.data.normalization, "strategy", "unknown")),
            prediction_stats=prediction_stats,
            spread_stats=spread_stats,
            price_stats=price_stats,
            predictions_distribution=predictions_distribution,
            agreement_stats=agreement_stats,
            confirmation_percentiles=confirmation_percentiles,
            directional_rate=directional_rate,
            metrics=metrics_dict,
            calibration=calibration_result,
            # Phase II additions
            compatibility=compatibility,
            data_source=data_source_tag,
            calibration_method=calibration_method,
            # Phase Y deployment (2026-05-05): model-axis identity hash
            model_config_hash=model_cfg_hash,
            # Phase Y / γ-1 LITE / #PY-88 (2026-05-10): top-level return_type
            return_type=rt_string,
        )
