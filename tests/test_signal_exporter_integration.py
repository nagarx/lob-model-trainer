"""Phase A.5.7c (2026-04-25) — full Trainer.setup() + SignalExporter.export() integration test.

Closes plan v4 §A.5.6 ship-blocker test gap (SB-3 from 5-agent validation).

The realistic production chain is:

    ExperimentConfig
      → Trainer(config).setup()        # DataLoader construction + model init
        → resolver populates PrivateAttr
        → model factory dispatch
      → SignalExporter(trainer).export(split="val", output_dir=tmp)
        → _run_inference(loader)        # forward pass
        → RawFeatureExtractor.extract() # disk-side spread/price
        → _build_metadata + CompatibilityContract
        → _write_files (signal_metadata.json + .npy files)
      → SignalManifest.from_signal_dir(output_dir)
        → manifest.compatibility.fingerprint() == producer fingerprint

Prior to A.5.7c, the closest test was ``test_signal_exporter_producer_roundtrip.py``,
which directly invoked ``_build_compatibility_contract()`` + ``build_signal_metadata()``
+ wrote signal_metadata.json by hand + parsed it back via ``SignalManifest`` —
SKIPPING ``Trainer.setup()`` and ``SignalExporter.export()`` entirely.

That left a coverage gap at the seam between Trainer state and SignalExporter
state: a regression that hides between ``trainer.setup()`` and
``exporter.export()`` (e.g., DataLoader-config mismatch, model-output-shape
drift, raw-feature alignment break) would slip through.

This integration test exercises the FULL chain. Marked ``@pytest.mark.integration``
so fast-CI can opt out via ``pytest -m "not integration"`` (test runtime is
~5-10s for ``Trainer.setup()`` on synthetic data + 2 epochs).

Model choice: ``logistic`` — smallest model in the registry (~294 params),
no GPU dependency, deterministic. The test does NOT train (no
``trainer.train()`` call) — only ``trainer.setup()`` + 1 forward pass via
``SignalExporter.export()``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

# Skip the entire module when hft_contracts isn't available (matches existing
# producer_roundtrip test convention).
_hft_contracts = pytest.importorskip("hft_contracts")

from lobtrainer.config.schema import (  # noqa: E402
    DataConfig,
    ExperimentConfig,
    LabelsConfig,
    ModelConfig,
    TrainConfig,
)
from lobtrainer.training.trainer import Trainer  # noqa: E402
from lobtrainer.export.exporter import SignalExporter  # noqa: E402
from hft_contracts.signal_manifest import SignalManifest  # noqa: E402


_FINGERPRINT_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


@pytest.mark.integration
class TestPhaseA57cFullIntegrationChain:
    """Phase A.5.7c — full Trainer.setup() + SignalExporter.export() chain.

    Locks the production chain end-to-end. Pre-A.5.7c, the chain was
    tested only at sub-component levels:

      - ``test_signal_export.py`` — _run_inference only (MockTrainer, no setup)
      - ``test_signal_exporter_calibration_horizon.py`` — _apply_calibration only
        (_DummyTrainer, skips setup + export)
      - ``test_signal_exporter_producer_roundtrip.py`` — _build_compatibility_contract
        + build_signal_metadata + SignalManifest.from_signal_dir (NO setup,
        NO export — JSON written by hand)

    No prior test exercised the seam between trainer.setup() and
    exporter.export() — a regression there would slip through.
    """

    def test_full_chain_classification_logistic(
        self, synthetic_export_dir: Path, tmp_path: Path,
    ) -> None:
        """End-to-end chain: ExperimentConfig → Trainer.setup() →
        SignalExporter.export() → SignalManifest.from_signal_dir().

        Uses the smallest model (logistic, ~294 params), the existing
        ``synthetic_export_dir`` fixture (2 train days + 1 val day,
        98 features, classification labels), and exports via
        ``split='val'`` (smaller than test).

        The test does NOT call ``trainer.train()`` — only ``trainer.setup()``
        is needed to construct DataLoaders + initialize the model so
        ``SignalExporter.export()`` can run inference.

        Locks 5 invariants in a SINGLE test (consolidated to keep
        Trainer.setup() runtime amortized — ~5-10s):

          1. Trainer.setup() succeeds with synthetic_export_dir input
          2. SignalExporter(trainer).export(split='val') produces files
          3. signal_metadata.json carries valid compatibility_fingerprint
          4. SignalManifest.from_signal_dir loads + 3-way validates
          5. Recomputed fingerprint matches stored fingerprint
             (producer-consumer byte-identity contract)
        """
        # Arrange: minimal classification ExperimentConfig pointing at the
        # synthetic export. Defaults preserved everywhere except
        # data_dir/output_dir (per-run paths) and a small batch_size
        # to keep the synthetic data loadable.
        output_dir = tmp_path / "outputs"
        signals_dir = tmp_path / "signals"

        config = ExperimentConfig(
            name="phase_a57c_integration",
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
                batch_size=8,  # synthetic days are small
                epochs=1,      # we don't actually train
                num_workers=0, # avoid pickle-of-config worker contract issue
            ),
            output_dir=str(output_dir),
        )

        # Act 1 — full Trainer.setup() (DataLoaders + model + optimizer)
        trainer = Trainer(config)
        trainer.setup()

        # Sanity: setup populated loaders + model
        assert trainer.get_loader("val") is not None, (
            "Trainer.setup() did not produce a val DataLoader"
        )
        assert trainer.model is not None, "Trainer.setup() did not init model"

        # Act 2 — full SignalExporter.export() pipeline
        exporter = SignalExporter(trainer)
        result = exporter.export(split="val", output_dir=signals_dir)

        # Assert 1 — export produced files
        assert result.output_dir == signals_dir
        assert (signals_dir / "signal_metadata.json").exists()

        # Assert 2 — signal_metadata.json carries a valid fingerprint
        metadata = json.loads((signals_dir / "signal_metadata.json").read_text())
        assert "compatibility_fingerprint" in metadata, (
            "signal_metadata.json missing compatibility_fingerprint key — "
            "the producer chain (trainer config → ExperimentConfig → "
            "_build_compatibility_contract → build_signal_metadata) silently "
            "dropped the fingerprint."
        )
        producer_fp = metadata["compatibility_fingerprint"]
        assert _FINGERPRINT_HEX_RE.match(producer_fp), (
            f"Fingerprint not 64-char lowercase hex: {producer_fp!r}"
        )

        # Assert 3 — SignalManifest.from_signal_dir loads without error
        manifest = SignalManifest.from_signal_dir(signals_dir)
        assert manifest.compatibility is not None, (
            "Manifest loaded but compatibility block is None — load-side "
            "parsing failed silently."
        )
        assert manifest.compatibility_fingerprint is not None

        # Assert 4 — 3-way byte-identity: stored fingerprint == recomputed
        # over the serialized block (the canonical producer-consumer contract)
        recomputed_fp = manifest.compatibility.fingerprint()
        assert manifest.compatibility_fingerprint == recomputed_fp, (
            f"Producer-consumer fingerprint drift: "
            f"stored={manifest.compatibility_fingerprint[:16]}... "
            f"recomputed={recomputed_fp[:16]}... — the canonical-form "
            f"computation has drifted between producer and consumer paths."
        )

        # Assert 5 — fingerprint also matches what the producer emitted
        # (full chain integrity)
        assert manifest.compatibility_fingerprint == producer_fp, (
            "Manifest's stored fingerprint differs from what the producer "
            "wrote into signal_metadata.json — file-write or load drift."
        )

        # Assert 6 — labels-config threading: primary_horizon_idx flows
        # from config.data.labels.primary_horizon_idx through to the
        # contract. Defaulted to 0 above; verifies the producer reads
        # the canonical path (not the deprecated DataConfig.horizon_idx).
        assert manifest.compatibility.primary_horizon_idx == 0
        assert manifest.compatibility.horizons == (10, 60, 300)

        # Assert 7 — feature_count, window_size threaded correctly
        assert manifest.compatibility.feature_count == 98
        assert manifest.compatibility.window_size == 100  # SequenceConfig default
