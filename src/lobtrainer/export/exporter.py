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

from lobtrainer.export.raw_features import RawFeatureExtractor
from lobtrainer.export.metadata import build_signal_metadata

logger = logging.getLogger(__name__)


def _feature_set_ref_dict(data_config: Any) -> Optional[Dict[str, str]]:
    """Extract feature_set_ref dict from DataConfig private cache (Phase 4 Batch 4c.4).

    The resolver populates `DataConfig._feature_set_ref_resolved` as a
    `(name, content_hash)` tuple at dataloader construction (see
    `Trainer._create_dataloaders`). This helper converts the tuple to the
    JSON-shape `{"name": ..., "content_hash": ...}` for signal_metadata.json.
    Returns None when no FeatureSet was resolved (feature_preset /
    feature_indices / no selection paths).

    Cross-subprocess invariant: this field is populated only when
    `_create_dataloaders` has run in the current process; it is NOT
    serialized across subprocess boundaries (the `_`-prefix + R3 `to_dict`
    filter strip it). The signal-export subprocess re-runs
    `_create_dataloaders` on its copy of the resolved config, which
    re-populates the cache. See `test_feature_set_ref_subprocess_invariant.py`.
    """
    resolved = getattr(data_config, "_feature_set_ref_resolved", None)
    if resolved is None:
        return None
    # Tuple form: (name, content_hash). Convert to dict.
    name, content_hash = resolved
    return {"name": str(name), "content_hash": str(content_hash)}


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
        metadata = self._build_metadata(
            inference, raw, split, output_dir, calibration_result
        )
        meta_path = output_dir / "signal_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
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
            result["agreement_ratio"] = np.concatenate(all_agreement).astype(np.float64)

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

        # calibrate_variance accepts only 1D — use primary horizon for multi-horizon
        if preds.ndim == 2:
            preds_1d = preds[:, 0]
            labels_1d = labels[:, 0] if labels is not None else None
        else:
            preds_1d = preds
            labels_1d = labels

        config = VarianceCalibrationConfig(method="variance_match")
        cal_result = calibrate_variance(preds_1d, labels_1d, config)

        logger.info(
            f"Calibration: scale={cal_result.scale_factor:.4f}, "
            f"pred_std={cal_result.pred_std:.2f} → target_std={cal_result.target_std:.2f}"
        )

        # to_dict() omits the ndarray — add it back for file writing
        result = cal_result.to_dict()
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

        # Always write prices and spreads
        np.save(output_dir / "prices.npy", raw.prices)
        files.append("prices.npy")
        np.save(output_dir / "spreads.npy", raw.spreads)
        files.append("spreads.npy")

        # Classification outputs
        if "predictions" in inference:
            np.save(output_dir / "predictions.npy", inference["predictions"])
            files.append("predictions.npy")
        if "labels" in inference:
            np.save(output_dir / "labels.npy", inference["labels"])
            files.append("labels.npy")

        # Regression outputs
        if "predicted_returns" in inference:
            np.save(output_dir / "predicted_returns.npy", inference["predicted_returns"])
            files.append("predicted_returns.npy")
        if "regression_labels" in inference:
            np.save(output_dir / "regression_labels.npy", inference["regression_labels"])
            files.append("regression_labels.npy")

        # HMHP-specific
        if "agreement_ratio" in inference:
            np.save(output_dir / "agreement_ratio.npy", inference["agreement_ratio"])
            files.append("agreement_ratio.npy")
        if "confirmation_score" in inference:
            np.save(output_dir / "confirmation_score.npy", inference["confirmation_score"])
            files.append("confirmation_score.npy")

        # Calibrated returns
        if calibration_result is not None and "calibrated" in calibration_result:
            cal_arr = np.array(calibration_result["calibrated"], dtype=np.float64)
            np.save(output_dir / "calibrated_returns.npy", cal_arr)
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
            if pr.ndim == 1:
                prediction_stats = {
                    "mean": float(np.mean(pr)),
                    "std": float(np.std(pr)),
                    "min": float(np.min(pr)),
                    "max": float(np.max(pr)),
                }
            else:
                prediction_stats = {
                    "mean": float(np.mean(pr[:, 0])),
                    "std": float(np.std(pr[:, 0])),
                }

            if "regression_labels" in inference:
                rl = inference["regression_labels"]
                try:
                    from lobtrainer.training.regression_metrics import (
                        compute_all_regression_metrics,
                    )
                    if pr.ndim == 1:
                        metrics_dict = compute_all_regression_metrics(rl, pr)
                    else:
                        metrics_dict = compute_all_regression_metrics(rl[:, 0], pr[:, 0])
                except Exception as e:
                    logger.warning(f"Could not compute regression metrics: {e}")

        # Horizons
        horizons = inference.get("horizons") or getattr(config.model, "hmhp_horizons", None)
        horizon_idx = getattr(config.data, "horizon_idx", None)
        primary_horizon = None
        if horizons and horizon_idx is not None and horizon_idx < len(horizons):
            primary_horizon = f"H{horizons[horizon_idx]}"

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
            feature_set_ref=_feature_set_ref_dict(config.data),
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
        )
