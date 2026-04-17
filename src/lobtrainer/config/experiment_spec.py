"""Experiment Specification — single-YAML experiment orchestration (T14).

ExperimentSpec is a unified config that combines data sources, labels,
model, training, gates, and CV into one reproducible YAML. It generates
a standard ExperimentConfig for the existing Trainer/CVTrainer pipeline.

This is a PARALLEL entry point alongside hft-ops ExperimentManifest.
ExperimentSpec assumes data is already exported (no Rust extraction stage)
and focuses on the "configure and train" workflow.

Usage:
    spec = ExperimentSpec.from_yaml("experiments/e17_fusion.yaml")
    spec.validate()
    config = spec.to_experiment_config()
    trainer = Trainer(config)
    trainer.train()

Reference: plan/EXPERIMENTATION_FIRST_ARCHITECTURE.md §20
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class SignalQualityGateConfig:
    """Pre-training signal quality gate configuration.

    Checks that at least ``min_features_passing`` features have
    Spearman IC > ``min_ic`` against the target regression label.
    Prevents wasting compute on experiments with no signal.

    Reference: hft-rules.md §13 (mandatory signal quality gate).
    """

    enabled: bool = True
    min_ic: float = 0.05
    min_features_passing: int = 1


@dataclass
class CostGateConfig:
    """Cost feasibility gate configuration (optional)."""

    enabled: bool = False
    breakeven_bps: float = 1.4  # Deep ITM default


@dataclass
class GateConfig:
    """All pre-training gates."""

    signal_quality: SignalQualityGateConfig = field(
        default_factory=SignalQualityGateConfig
    )
    cost: CostGateConfig = field(default_factory=CostGateConfig)


@dataclass
class ExperimentMetadata:
    """Experiment identification and tracking."""

    name: str = "unnamed"
    hypothesis: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class ExperimentSpec:
    """Single-YAML experiment specification.

    Combines data sources, labels, model, training, gates, and CV into
    one reproducible config. Generates ExperimentConfig for the existing
    Trainer/CVTrainer pipeline via ``to_experiment_config()``.

    All fields mirror the ExperimentConfig hierarchy but are stored as
    plain dicts for flexible YAML parsing. The ``to_experiment_config()``
    method passes them through ``ExperimentConfig.from_dict()`` which
    handles all dataclass construction via dacite.
    """

    experiment: ExperimentMetadata = field(
        default_factory=ExperimentMetadata
    )
    data: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    train: Dict[str, Any] = field(default_factory=dict)
    gates: GateConfig = field(default_factory=GateConfig)
    cv: Optional[Dict[str, Any]] = None
    output_dir: str = "outputs"

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentSpec":
        """Load ExperimentSpec from a YAML file.

        Args:
            path: Path to the spec YAML file.

        Returns:
            ExperimentSpec instance.

        Raises:
            FileNotFoundError: If the YAML file doesn't exist.
            yaml.YAMLError: If the YAML is malformed.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Spec file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ValueError(f"Spec YAML must be a dict, got {type(raw)}")

        # Parse experiment metadata
        exp_raw = raw.get("experiment", {})
        experiment = ExperimentMetadata(
            name=exp_raw.get("name", "unnamed"),
            hypothesis=exp_raw.get("hypothesis", ""),
            tags=exp_raw.get("tags", []),
        )

        # Parse gates
        gates_raw = raw.get("gates", {})
        sq_raw = gates_raw.get("signal_quality", {})
        cost_raw = gates_raw.get("cost", {})
        gates = GateConfig(
            signal_quality=SignalQualityGateConfig(
                enabled=sq_raw.get("enabled", True),
                min_ic=sq_raw.get("min_ic", 0.05),
                min_features_passing=sq_raw.get("min_features_passing", 1),
            ),
            cost=CostGateConfig(
                enabled=cost_raw.get("enabled", False),
                breakeven_bps=cost_raw.get("breakeven_bps", 1.4),
            ),
        )

        return cls(
            experiment=experiment,
            data=raw.get("data", {}),
            model=raw.get("model", {}),
            train=raw.get("train", {}),
            gates=gates,
            cv=raw.get("cv"),
            output_dir=raw.get("output_dir", "outputs"),
        )

    def to_experiment_config(self) -> "ExperimentConfig":
        """Convert to a standard ExperimentConfig for Trainer/CVTrainer.

        Builds a dict matching the ExperimentConfig.from_dict() schema
        and delegates to dacite for full dataclass construction.

        Returns:
            ExperimentConfig ready for Trainer(config).
        """
        from lobtrainer.config.schema import ExperimentConfig

        config_dict = {
            "name": self.experiment.name,
            "description": self.experiment.hypothesis,
            "data": dict(self.data),
            "model": dict(self.model),
            "train": dict(self.train),
            "output_dir": self.output_dir,
            "tags": self.experiment.tags,
        }

        # Add CV config if specified
        if self.cv is not None:
            config_dict["cv"] = dict(self.cv)

        return ExperimentConfig.from_dict(config_dict)

    def validate(self) -> List[str]:
        """Pre-flight validation. Returns list of warnings.

        Checks:
        - Experiment name is set
        - Data section has required fields
        - Gates config is valid
        - Can produce a valid ExperimentConfig (dry run)

        Raises:
            ValueError: For hard errors that prevent execution.
        """
        warnings = []

        if self.experiment.name == "unnamed":
            warnings.append("Experiment name is 'unnamed' — set a descriptive name")

        if not self.experiment.hypothesis:
            warnings.append(
                "No hypothesis specified — per hft-rules.md §13, "
                "never run an experiment without a clear hypothesis"
            )

        if not self.data:
            raise ValueError("ExperimentSpec.data section is empty")

        if self.gates.signal_quality.enabled:
            if self.gates.signal_quality.min_ic <= 0:
                raise ValueError(
                    f"signal_quality.min_ic must be > 0, "
                    f"got {self.gates.signal_quality.min_ic}"
                )

        # Dry-run config generation to catch schema errors early
        try:
            self.to_experiment_config()
        except Exception as e:
            raise ValueError(
                f"ExperimentSpec cannot produce a valid ExperimentConfig: {e}"
            ) from e

        return warnings
