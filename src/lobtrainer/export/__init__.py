"""
Signal export module for the LOB training pipeline.

Provides a unified `SignalExporter` that replaces the 3 separate signal
export scripts with a single, Trainer-integrated export path. Produces
backtester-compatible signal files (.npy + metadata JSON).

Usage:
    trainer = Trainer(config)
    trainer.setup()
    trainer.load_checkpoint(checkpoint_path, load_optimizer=False)

    exporter = SignalExporter(trainer)
    result = exporter.export(split="test", output_dir=output_dir)
"""

from lobtrainer.export.exporter import SignalExporter, ExportResult
from lobtrainer.export.raw_features import RawFeatureExtractor, RawFeatures
from lobtrainer.export.metadata import build_signal_metadata

__all__ = [
    "SignalExporter",
    "ExportResult",
    "RawFeatureExtractor",
    "RawFeatures",
    "build_signal_metadata",
]
