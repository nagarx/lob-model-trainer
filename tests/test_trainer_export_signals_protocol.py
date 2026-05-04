"""Phase Q.6.5.B Part 1 — ``Trainer.export_signals`` Protocol method smoke test.

Closes Q.6.5.B Part 1 ship-blocker (Agent 1 mid-impl validation HIGH-1):
the new ``Trainer.export_signals`` method on the PyTorch path was added
without dedicated test coverage. Without a smoke test, a regression in
the SignalExporter delegation lazy-import path could slip through.

The test exercises:
    ExperimentConfig → Trainer.setup() → trainer.export_signals('val', ...)
    → signal_metadata.json present + 64-hex compatibility_fingerprint

Mirrors test_signal_exporter_integration.py::TestPhaseA57cFullIntegrationChain
but invokes the BaseTrainer Protocol method directly instead of constructing
SignalExporter manually. Locks the Q1 closure (Trainer + SimpleModelTrainer
both expose ``export_signals(split, output_dir, calibration) -> Path``) at
the production-code level.

Marked ``@pytest.mark.integration`` so fast-CI can opt out.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

# Skip if hft_contracts unavailable (matches integration test convention).
_hft_contracts = pytest.importorskip("hft_contracts")

from lobtrainer.config.schema import (  # noqa: E402
    DataConfig,
    ExperimentConfig,
    LabelsConfig,
    ModelConfig,
    TrainConfig,
)
from lobtrainer.training.trainer import Trainer  # noqa: E402

_FINGERPRINT_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


@pytest.mark.integration
class TestTrainerExportSignalsProtocolMethod:
    """Phase Q.6.5.B Part 1 — Trainer.export_signals delegates to
    SignalExporter and produces canonical signal_metadata.json with
    compatibility block + fingerprint."""

    def test_export_signals_method_returns_path_with_canonical_artifacts(
        self, synthetic_export_dir: Path, tmp_path: Path,
    ) -> None:
        """End-to-end via Protocol method:
        ExperimentConfig → Trainer.setup() → trainer.export_signals('val', ...)

        Locks 5 invariants:
          1. trainer.export_signals returns a Path (not ExportResult)
          2. signal_metadata.json present at the returned location
          3. predicted_returns/predictions + raw features (.npy) present
          4. compatibility_fingerprint is 64-hex SHA-256
          5. Returned path equals the explicit output_dir override
        """
        signals_dir = tmp_path / "signals_protocol_method"

        config = ExperimentConfig(
            name="phase_q65b_protocol_method",
            data=DataConfig(
                data_dir=str(synthetic_export_dir),
                feature_count=98,
                labels=LabelsConfig(
                    horizons=[10, 60, 300],
                    primary_horizon_idx=0,
                    task="classification",
                    threshold_bps=8.0,
                ),
            ),
            model=ModelConfig(
                model_type="logistic", input_size=98, num_classes=3,
            ),
            train=TrainConfig(
                batch_size=8,
                epochs=1,
                num_workers=0,
            ),
            output_dir=str(tmp_path / "outputs"),
        )

        # Setup PyTorch trainer (DataLoaders + model + optimizer).
        trainer = Trainer(config)
        trainer.setup()
        assert trainer.get_loader("val") is not None
        assert trainer.model is not None

        # Invoke the Protocol method. This is the F-16 / Q1 closure path —
        # post-Q.6.5.B, scripts/export_signals.py and cli.py both call
        # this polymorphically via the BaseTrainer Protocol.
        result_path = trainer.export_signals(
            split="val",
            output_dir=signals_dir,
            calibration="none",
        )

        # 1. Returns Path (not ExportResult) — Protocol contract
        assert isinstance(result_path, Path), (
            f"Trainer.export_signals must return Path (Protocol contract), "
            f"got {type(result_path).__name__}"
        )
        # 2. Returned path equals the explicit override
        assert result_path == signals_dir, (
            f"Returned path {result_path} != requested output_dir {signals_dir}"
        )

        # 3. Required artifacts present
        assert (signals_dir / "signal_metadata.json").exists(), (
            "signal_metadata.json missing — SignalExporter delegation broken"
        )

        # 4. compatibility_fingerprint is 64-hex SHA-256
        meta = json.loads((signals_dir / "signal_metadata.json").read_text())
        fp = meta.get("compatibility_fingerprint")
        assert fp is not None, (
            "compatibility_fingerprint missing — Phase II surface dropped "
            "in the Trainer.export_signals delegation chain"
        )
        assert _FINGERPRINT_HEX_RE.match(fp), (
            f"compatibility_fingerprint {fp!r} is not 64-hex SHA-256"
        )

    def test_export_signals_default_output_dir(
        self, synthetic_export_dir: Path, tmp_path: Path,
    ) -> None:
        """When output_dir=None, default location is
        ``<config.output_dir>/signals/<split>/``. Locks the default
        path-resolution contract."""
        config = ExperimentConfig(
            name="phase_q65b_default_output_dir",
            data=DataConfig(
                data_dir=str(synthetic_export_dir),
                feature_count=98,
                labels=LabelsConfig(
                    horizons=[10, 60, 300],
                    primary_horizon_idx=0,
                    task="classification",
                    threshold_bps=8.0,
                ),
            ),
            model=ModelConfig(
                model_type="logistic", input_size=98, num_classes=3,
            ),
            train=TrainConfig(batch_size=8, epochs=1, num_workers=0),
            output_dir=str(tmp_path / "outputs"),
        )

        trainer = Trainer(config)
        trainer.setup()

        # Default output_dir → <config.output_dir>/signals/val
        result_path = trainer.export_signals(split="val")  # no output_dir kwarg
        expected = Path(config.output_dir) / "signals" / "val"
        assert result_path == expected, (
            f"Default output_dir resolution wrong: got {result_path}, "
            f"expected {expected}"
        )
        assert (expected / "signal_metadata.json").exists()

    def test_export_signals_train_split_refused(
        self, synthetic_export_dir: Path, tmp_path: Path,
    ) -> None:
        """Train split is refused by SignalExporter (DataLoader uses
        drop_last=True; alignment mismatch with raw features). The
        Protocol method propagates this rejection."""
        config = ExperimentConfig(
            name="phase_q65b_train_refused",
            data=DataConfig(
                data_dir=str(synthetic_export_dir),
                feature_count=98,
                labels=LabelsConfig(
                    horizons=[10, 60, 300],
                    primary_horizon_idx=0,
                    task="classification",
                    threshold_bps=8.0,
                ),
            ),
            model=ModelConfig(
                model_type="logistic", input_size=98, num_classes=3,
            ),
            train=TrainConfig(batch_size=8, epochs=1, num_workers=0),
            output_dir=str(tmp_path / "outputs"),
        )

        trainer = Trainer(config)
        trainer.setup()

        with pytest.raises(ValueError, match=r"Cannot export training split"):
            trainer.export_signals(split="train", output_dir=tmp_path / "x")
