"""
Tests for configuration schema.
"""

import tempfile
from pathlib import Path
import pytest

from lobtrainer.config import (
    DataConfig,
    SequenceConfig,
    NormalizationConfig,
    ModelConfig,
    TrainConfig,
    ExperimentConfig,
    load_config,
    save_config,
)
from lobtrainer.config.schema import NormalizationStrategy, ModelType, LabelEncoding, TaskType, LossType


class TestSequenceConfig:
    """Test SequenceConfig validation."""
    
    def test_default_values(self):
        """Default values should be valid."""
        config = SequenceConfig()
        assert config.window_size == 100
        assert config.stride == 10
    
    def test_invalid_window_size(self):
        """Window size must be >= 1.

        Phase A.5.3b (2026-04-24): post-migration SequenceConfig is a Pydantic
        BaseModel; ``raise ValueError(...)`` inside ``@model_validator`` is
        auto-wrapped as ``pydantic.ValidationError``. Original error text
        preserved in the wrapped message — ``match=`` substring still fires.
        """
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="window_size must be >= 1"):
            SequenceConfig(window_size=0)

    def test_invalid_stride(self):
        """Stride must be >= 1."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="stride must be >= 1"):
            SequenceConfig(stride=0)

    def test_stride_exceeds_window(self):
        """Stride should not exceed window size."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="stride.*should not exceed window_size"):
            SequenceConfig(window_size=50, stride=100)


class TestNormalizationConfig:
    """Test NormalizationConfig."""
    
    def test_default_strategy(self):
        """Default is per-day Z-score."""
        config = NormalizationConfig()
        assert config.strategy == NormalizationStrategy.ZSCORE_PER_DAY
    
    def test_default_exclude_features_empty(self):
        """Default exclude_features is empty (avoids out-of-bounds on 40-feature data).

        Phase A.5.3c (2026-04-24): exclude_features type changed from
        List[int] to Tuple[int, ...] for true immutability. Empty default
        is now ``()`` not ``[]``.
        """
        config = NormalizationConfig()
        assert config.exclude_features == ()

    def test_invalid_eps(self):
        """Eps must be > 0.

        Phase A.5.3c: Pydantic wraps `raise ValueError(...)` from
        @model_validator as pydantic.ValidationError — `match=` substring
        still fires on the wrapped message.
        """
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="eps must be > 0"):
            NormalizationConfig(eps=0)


class TestModelConfig:
    """Test ModelConfig."""
    
    def test_default_values(self):
        """Default values should be valid."""
        config = ModelConfig()
        assert config.model_type == ModelType.LSTM
        assert config.hidden_size == 64
        assert config.num_layers == 2
    
    def test_invalid_dropout(self):
        """Dropout must be in [0, 1]."""
        with pytest.raises(ValueError, match="dropout must be in"):
            ModelConfig(dropout=1.5)
        with pytest.raises(ValueError, match="dropout must be in"):
            ModelConfig(dropout=-0.1)


class TestTrainConfig:
    """Test TrainConfig."""
    
    def test_default_seed(self):
        """Default seed is 42."""
        config = TrainConfig()
        assert config.seed == 42
    
    def test_invalid_batch_size(self):
        """Batch size must be >= 1.

        Phase A.5.3e (2026-04-24): Pydantic wraps ValueError as
        ValidationError; match= substring still fires.
        """
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="batch_size must be >= 1"):
            TrainConfig(batch_size=0)

    def test_invalid_learning_rate(self):
        """Learning rate must be > 0."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="learning_rate must be > 0"):
            TrainConfig(learning_rate=0)

    def test_classification_task_rejects_huber_loss(self):
        """Regression loss_type on classification task_type must raise."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="regression loss"):
            TrainConfig(task_type=TaskType.MULTICLASS, loss_type=LossType.HUBER)

    def test_regression_task_rejects_cross_entropy_loss(self):
        """Classification loss_type on regression task_type must raise."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="classification loss"):
            TrainConfig(task_type=TaskType.REGRESSION, loss_type=LossType.CROSS_ENTROPY)

    def test_regression_task_accepts_huber_loss(self):
        """Regression loss with regression task should pass validation."""
        config = TrainConfig(task_type=TaskType.REGRESSION, loss_type=LossType.HUBER)
        assert config.task_type == TaskType.REGRESSION
        assert config.loss_type == LossType.HUBER

    def test_classification_task_accepts_focal_loss(self):
        """Classification loss with classification task should pass validation."""
        config = TrainConfig(task_type=TaskType.MULTICLASS, loss_type=LossType.FOCAL)
        assert config.task_type == TaskType.MULTICLASS
        assert config.loss_type == LossType.FOCAL


class TestExperimentConfig:
    """Test ExperimentConfig serialization."""
    
    def test_default_config(self):
        """Default config should be valid."""
        config = ExperimentConfig()
        assert config.data.feature_count == 98
        assert config.model.input_size == 98
    
    def test_to_dict(self):
        """Config should serialize to dict."""
        config = ExperimentConfig(name="test")
        data = config.to_dict()
        assert data["name"] == "test"
        assert data["data"]["feature_count"] == 98
        assert data["model"]["model_type"] == "lstm"  # Enum to string
    
    def test_yaml_round_trip(self):
        """Config should survive YAML serialization."""
        config = ExperimentConfig(
            name="test_experiment",
            description="Test description",
            tags=["test", "baseline"],
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            config.to_yaml(str(path))
            loaded = ExperimentConfig.from_yaml(str(path))
        
        assert loaded.name == config.name
        assert loaded.description == config.description
        assert loaded.tags == config.tags
        assert loaded.data.feature_count == config.data.feature_count
    
    def test_json_round_trip(self):
        """Config should survive JSON serialization."""
        config = ExperimentConfig(name="json_test")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            config.to_json(str(path))
            loaded = ExperimentConfig.from_json(str(path))
        
        assert loaded.name == config.name


class TestLoadSaveConfig:
    """Test convenience load/save functions."""
    
    def test_load_yaml(self):
        """load_config detects YAML format."""
        config = ExperimentConfig(name="yaml_test")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            save_config(config, str(path))
            loaded = load_config(str(path))
        
        assert loaded.name == "yaml_test"
    
    def test_load_json(self):
        """load_config detects JSON format."""
        config = ExperimentConfig(name="json_test")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            save_config(config, str(path))
            loaded = load_config(str(path))
        
        assert loaded.name == "json_test"
    
    def test_unsupported_format(self):
        """Unsupported format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported config format"):
            load_config("config.txt")


# =============================================================================
# Phase A.5.3a (2026-04-24): LabelsConfig → Pydantic v2 BaseModel migration.
#
# Locks 4 bug-class retirements at the TYPE layer:
#   1. Canonical-path-drift (config.labels vs config.data.labels)
#   2. Silent mutation (frozen=True)
#   3. Extra-field acceptance (extra="forbid")
#   4. Silent-None field access (Pydantic validators fire at construction)
#
# Plus the v3-A ClassVar discipline regression — the four class-level
# frozenset constants MUST NOT leak into model_dump() output (would rotate
# every post-migration compatibility_fingerprint).
#
# Plus byte-identity parity vs the hft-contracts A.5.1 frozen fixture —
# same logical LabelsConfig content produces the same SHA-256 hash before
# and after the dacite→Pydantic migration.
# =============================================================================


class TestLabelsConfigPydantic:
    """A.5.3a Pydantic migration locks.

    See ``LabelsConfig`` class docstring for the full rationale of the 4
    bug-class retirements. These tests LOCK each layer of the retirement so
    a future regression (e.g. someone adding a new attribute without
    ClassVar annotation) surfaces at CI time, not in production.
    """

    def test_frozen_rejects_mutation(self):
        """`frozen=True` MUST raise on any field assignment post-construction.

        Retires bug class #2 (silent mutation). Pre-migration dataclass
        accepted `config.labels.task = "..."` silently.
        """
        from lobtrainer.config.schema import LabelsConfig
        from pydantic import ValidationError

        cfg = LabelsConfig()
        with pytest.raises(ValidationError):
            cfg.source = "forward_prices"  # type: ignore[misc]
        with pytest.raises(ValidationError):
            cfg.task = "regression"  # type: ignore[misc]

    def test_extra_forbid_rejects_typo(self):
        """`extra="forbid"` MUST raise on unknown kwargs.

        Retires bug class #3. Pre-migration dacite silently dropped unknown
        fields, producing a config with ZERO user-intended behavior from
        the typo'd YAML.
        """
        from lobtrainer.config.schema import LabelsConfig
        from pydantic import ValidationError

        # Common operator typos
        with pytest.raises(ValidationError, match="horizen_idx|extra"):
            LabelsConfig(horizen_idx=0)  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            LabelsConfig(soruce="auto")  # type: ignore[call-arg]

    def test_validators_preserved_source_invalid(self):
        """Legacy `__post_init__` validators must fire under `@model_validator`.

        Pre-migration these were `raise ValueError(...)` in `__post_init__`;
        post-migration Pydantic wraps them as `ValidationError` but the
        same invariants hold.
        """
        from lobtrainer.config.schema import LabelsConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="source"):
            LabelsConfig(source="invalid_source_name")
        with pytest.raises(ValidationError, match="threshold_bps"):
            LabelsConfig(threshold_bps=-1.0)
        with pytest.raises(ValidationError, match="horizons"):
            LabelsConfig(horizons=[0, 1])  # 0 is degenerate
        with pytest.raises(ValidationError, match="duplicates"):
            LabelsConfig(horizons=[10, 10])
        with pytest.raises(ValidationError, match="sample_weights"):
            LabelsConfig(sample_weights="bogus")

    def test_class_constants_not_in_model_dump(self):
        """v3-A regression: `ClassVar[frozenset[str]]` constants MUST be
        excluded from ``model_dump()`` — otherwise they'd appear as model
        FIELDS, leaking into every stored ``compatibility_fingerprint`` and
        rotating the hash across the dacite→Pydantic migration.

        This is the ship-blocker for fingerprint byte-identity. See the
        ``LabelsConfig`` docstring + ``compute_label_strategy_hash`` in
        hft-contracts for the cross-module coupling.
        """
        from lobtrainer.config.schema import LabelsConfig

        cfg = LabelsConfig()
        dump = cfg.model_dump()
        forbidden_keys = {
            "_VALID_SOURCES",
            "_VALID_RETURN_TYPES",
            "_VALID_TASKS",
            "_VALID_SAMPLE_WEIGHTS",
        }
        leaked = forbidden_keys & dump.keys()
        assert not leaked, (
            f"ClassVar[frozenset] constants leaked into model_dump: {leaked}. "
            f"The v3-A ClassVar annotation is the fix — without it, Pydantic "
            f"treats class-level attributes as model fields and they appear "
            f"in serialized output, breaking compatibility_fingerprint "
            f"byte-identity across the dataclass→BaseModel migration. "
            f"Fix: annotate each as `_VALID_X: ClassVar[frozenset[str]] = "
            f"frozenset({{...}})` in LabelsConfig class body."
        )

    def test_class_constants_still_accessible_as_class_attrs(self):
        """Defensive: ``ClassVar`` annotation MUST NOT break class-level
        attribute access. The 4 constants remain usable by the validator
        at ``__post_init__`` / ``@model_validator`` run time.
        """
        from lobtrainer.config.schema import LabelsConfig

        assert "auto" in LabelsConfig._VALID_SOURCES
        assert "forward_prices" in LabelsConfig._VALID_SOURCES
        assert "smoothed_return" in LabelsConfig._VALID_RETURN_TYPES
        assert "classification" in LabelsConfig._VALID_TASKS
        assert "concurrent_overlap" in LabelsConfig._VALID_SAMPLE_WEIGHTS

    def test_label_strategy_hash_real_pydantic_parity(self):
        """Byte-identity lock against the A.5.1 frozen fixture.

        The REAL migrated ``LabelsConfig`` (now BaseModel) MUST produce the
        same ``compute_label_strategy_hash`` output as the pre-migration
        dataclass fixture in hft-contracts (generated by the one-off
        ``generate_pre_migration_snapshots.py`` script on 2026-04-24).

        Ship-blocker: if this test fails, every post-migration
        ``compatibility_fingerprint`` stored in the hft-ops ledger would
        rotate silently — breaking
        ``hft-ops ledger list --compatibility-fp <hex>`` cross-experiment
        query comparability.
        """
        from hft_contracts._testing import require_monorepo_root
        from hft_contracts.compatibility import compute_label_strategy_hash
        from lobtrainer.config.schema import LabelsConfig
        import json as _json

        monorepo_root = require_monorepo_root(
            "hft-contracts/tests/fixtures/pre_pydantic_label_strategy_hash.json"
        )
        fixture_path = (
            monorepo_root
            / "hft-contracts"
            / "tests"
            / "fixtures"
            / "pre_pydantic_label_strategy_hash.json"
        )
        with open(fixture_path) as f:
            frozen = _json.load(f)
        expected_hash = frozen["label_strategy_hash"]

        # Real LabelsConfig with default field values — MUST match the
        # fixture which was generated from the dataclass mirror with
        # identical defaults (source="auto", task="auto", horizons=[], ...).
        real_labels = LabelsConfig()
        real_hash = compute_label_strategy_hash(real_labels)

        assert real_hash == expected_hash, (
            f"Real Pydantic LabelsConfig hash {real_hash!r} != "
            f"frozen dataclass fixture hash {expected_hash!r}. "
            f"The A.5.3a migration has rotated compatibility_fingerprint. "
            f"Possible causes: (1) new field added to LabelsConfig without "
            f"regenerating the fixture; (2) _VALID_* ClassVar annotation "
            f"missing (leaks class constants into model_dump); (3) model_dump "
            f"divergence from asdict for some field type. "
            f"Fixture path: {fixture_path}"
        )

    def test_labels_config_to_dict_via_experiment_config(self):
        """Mixed-state to_dict parity: DataConfig (still @dataclass) + nested
        LabelsConfig (now BaseModel) must serialize cleanly via
        ``ExperimentConfig.to_dict()``.

        Without the BaseModel branch patch in ``_convert``, the recursive
        walker would fall through to the passthrough case for BaseModel
        instances, leaving a raw ``LabelsConfig`` object in the output dict
        — yaml.dump would then crash with `cannot represent an object` or
        emit a useless !!python/object tag.

        The patch dispatches to ``model_dump(exclude_none=False)`` BEFORE
        the dataclass check, producing a clean nested dict.
        """
        from lobtrainer.config.schema import (
            DataConfig,
            ExperimentConfig,
            LabelsConfig,
        )

        ec = ExperimentConfig(
            name="mixed_state_test",
            data=DataConfig(
                feature_count=98,
                labels=LabelsConfig(
                    source="forward_prices",
                    task="regression",
                    horizons=[10, 60],
                    primary_horizon_idx=0,
                ),
            ),
        )
        d = ec.to_dict()
        # Labels must be a dict (not a raw BaseModel instance).
        assert isinstance(d["data"]["labels"], dict), (
            f"labels serialized as {type(d['data']['labels']).__name__}, "
            f"expected dict. _convert's BaseModel branch isn't firing."
        )
        # Key fields present + no Pydantic internals leaked.
        labels_dict = d["data"]["labels"]
        assert labels_dict["source"] == "forward_prices"
        assert labels_dict["task"] == "regression"
        # Phase A.5.3a.1: horizons is Tuple[int, ...] internally; _convert's
        # `isinstance(obj, (list, tuple))` branch normalizes to list for JSON/YAML
        # friendly output. Matches sanitize_for_hash canonical-form convention.
        assert labels_dict["horizons"] == [10, 60]
        assert labels_dict["primary_horizon_idx"] == 0
        # Pydantic internals MUST NOT appear anywhere in the nested dict.
        def _walk(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(k, str) and k.startswith("__pydantic_"):
                        return True
                    if _walk(v):
                        return True
            elif isinstance(obj, list):
                return any(_walk(item) for item in obj)
            return False
        assert not _walk(d), (
            "Pydantic internals (__pydantic_*) leaked into to_dict output; "
            "_convert's BaseModel branch should have stripped them via model_dump."
        )
        # ClassVar constants MUST NOT appear either.
        assert "_VALID_SOURCES" not in labels_dict


# =============================================================================
# Phase A.5.3a.1 (2026-04-24) post-audit hardening regression locks.
#
# Three specialized agents (code-reviewer + hft-architect + general-purpose)
# identified 4 ship-blocker-class bugs in the A.5.3a migration that this
# hardening commit closes:
#
#   1. `model_copy(update={invalid})` bypasses validators (bug class #3
#      re-opened through a common Pydantic idiom — Pydantic v2's
#      model_copy uses model_construct which SKIPS validation). Fixed by
#      custom model_copy override that dispatches to model_validate when
#      update is provided.
#
#   2. Pydantic v2 lax-mode silent type coercion (NEW bug class the
#      migration introduced, not one it retired):
#        - `LabelsConfig(horizons=["10","60"])` silently coerced strings → ints
#        - `LabelsConfig(primary_horizon_idx=True)` coerced bool → 1
#      Fixed by `strict=True` in model_config.
#
#   3. NaN/Inf `threshold_bps` silently accepted (IEEE 754 considers them
#      valid floats; strict mode does NOT reject). Fixed by explicit
#      math.isfinite check in _validate_all.
#
#   4. `cfg.horizons.append(99)` bypassed frozen=True (mutable container).
#      Fixed by changing horizons type List[int] → Tuple[int, ...] +
#      @field_validator(mode="before") for list→tuple coercion from YAML.
# =============================================================================


class TestLabelsConfigPydanticHardening:
    """A.5.3a.1 post-audit hardening: lock the 4 bug-class closures.

    These tests MUST pass or the Phase A.5 migration's promised bug-class
    retirements are incomplete.
    """

    def test_model_copy_with_invalid_update_raises(self):
        """Bug #1: model_copy(update={...}) MUST re-run validators.

        Empirically confirmed pre-A.5.3a.1: default Pydantic v2 model_copy
        uses model_construct internally, which SKIPS validation. Invalid
        updates pass silently. Custom override at LabelsConfig.model_copy
        forces re-validation via model_validate.
        """
        from lobtrainer.config.schema import LabelsConfig
        from pydantic import ValidationError

        cfg = LabelsConfig()
        with pytest.raises(ValidationError, match="source"):
            cfg.model_copy(update={"source": "invalid_source_name"})
        with pytest.raises(ValidationError, match="task"):
            cfg.model_copy(update={"task": "not_a_real_task"})
        with pytest.raises(ValidationError, match="threshold_bps"):
            cfg.model_copy(update={"threshold_bps": -1.0})

    def test_model_copy_without_update_preserves_fast_path(self):
        """Positive control: model_copy() with no update doesn't incur
        re-validation cost — delegates to super().model_copy for the pure
        copy case.
        """
        from lobtrainer.config.schema import LabelsConfig

        cfg = LabelsConfig(source="forward_prices", task="regression")
        cfg2 = cfg.model_copy()
        assert cfg2.source == "forward_prices"
        assert cfg2.task == "regression"
        # Deep copy flag still works
        cfg3 = cfg.model_copy(deep=True)
        assert cfg3.source == "forward_prices"

    def test_model_copy_with_valid_update_succeeds(self):
        """Positive control: model_copy(update={valid}) passes validators."""
        from lobtrainer.config.schema import LabelsConfig

        cfg = LabelsConfig()
        cfg2 = cfg.model_copy(update={
            "source": "forward_prices",
            "task": "regression",
            "horizons": [10, 60, 300],
        })
        assert cfg2.source == "forward_prices"
        assert cfg2.task == "regression"
        assert cfg2.horizons == (10, 60, 300)  # Tuple[int, ...] post-migration

    def test_strict_mode_rejects_string_to_int_coercion(self):
        """Bug #2a: Pydantic lax mode would coerce string → int silently.
        strict=True rejects to preserve hft-rules §5 fail-fast.

        Empirically confirmed pre-A.5.3a.1: `LabelsConfig(horizons=["10","60"])`
        produced horizons=[10, 60] without any error.
        """
        from lobtrainer.config.schema import LabelsConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="horizons"):
            LabelsConfig(horizons=["10", "60"])
        with pytest.raises(ValidationError, match="horizons"):
            LabelsConfig(horizons=["10"])

    def test_strict_mode_rejects_bool_as_int(self):
        """Bug #2b: bool is an int-subclass in Python, so lax mode would
        coerce `primary_horizon_idx=True` → 1 silently. strict=True
        rejects.
        """
        from lobtrainer.config.schema import LabelsConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="primary_horizon_idx"):
            LabelsConfig(primary_horizon_idx=True)
        with pytest.raises(ValidationError, match="primary_horizon_idx"):
            LabelsConfig(primary_horizon_idx=False)

    def test_nan_threshold_bps_rejected(self):
        """Bug #3: NaN/Inf threshold_bps passed lax mode AND the
        `threshold_bps < 0` check (NaN comparisons return False).

        Explicit math.isfinite() check closes this — a NaN classification
        threshold would silently skip EVERY sample (abs(x) < NaN is False
        for all x in Python).
        """
        import math
        from lobtrainer.config.schema import LabelsConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="finite"):
            LabelsConfig(threshold_bps=float("nan"))
        with pytest.raises(ValidationError, match="finite"):
            LabelsConfig(threshold_bps=float("inf"))
        with pytest.raises(ValidationError, match="finite"):
            LabelsConfig(threshold_bps=-math.inf)

    def test_horizons_tuple_immutable_list_append_blocked(self):
        """Bug #4: Pydantic frozen=True blocks field ASSIGNMENT but NOT
        mutation of mutable containers. With horizons: List[int], the
        call `cfg.horizons.append(99)` silently mutated the instance.

        Fixed by changing type to Tuple[int, ...] (immutable container).
        Tuple has no .append method — raises AttributeError on the bypass
        attempt.
        """
        from lobtrainer.config.schema import LabelsConfig

        cfg = LabelsConfig(horizons=[10, 60])
        # Post-migration: horizons is always a tuple (coerced from list input
        # via @field_validator(mode="before")).
        assert isinstance(cfg.horizons, tuple), (
            f"horizons should be tuple post-A.5.3a.1; got {type(cfg.horizons).__name__}"
        )
        # .append on tuple raises AttributeError — container-mutation
        # bypass of frozen is now structurally blocked.
        with pytest.raises(AttributeError):
            cfg.horizons.append(99)  # type: ignore[attr-defined]
        # Bonus: __setitem__ also raises (tuples are fully immutable).
        with pytest.raises(TypeError):
            cfg.horizons[0] = 999  # type: ignore[index]

    def test_horizons_accepts_yaml_list_input_coerced_to_tuple(self):
        """Positive control for the @field_validator(mode="before"):
        YAML loaders always emit lists; we must accept list input while
        producing an immutable tuple internally.
        """
        from lobtrainer.config.schema import LabelsConfig

        cfg = LabelsConfig(horizons=[10, 60, 300])
        assert cfg.horizons == (10, 60, 300)
        assert isinstance(cfg.horizons, tuple)
        # Empty list default also works
        cfg_default = LabelsConfig()
        assert cfg_default.horizons == ()
        assert isinstance(cfg_default.horizons, tuple)
        # Tuple input is accepted directly (no conversion needed)
        cfg_tuple = LabelsConfig(horizons=(5, 10))
        assert cfg_tuple.horizons == (5, 10)

    def test_hardening_preserves_byte_identity_fixture_parity(self):
        """The A.5.3a.1 hardening (strict + isfinite + tuple coercion +
        model_copy override) MUST NOT rotate the fixture hash.

        Byte-identity depends on canonical-form equality, NOT Python type
        equality. sanitize_for_hash normalizes tuples → lists before
        hashing, so the canonical form is the same whether horizons is
        List[int] or Tuple[int, ...]. Ship-blocker lock.
        """
        from hft_contracts._testing import require_monorepo_root
        from hft_contracts.compatibility import compute_label_strategy_hash
        from lobtrainer.config.schema import LabelsConfig
        import json as _json

        monorepo_root = require_monorepo_root(
            "hft-contracts/tests/fixtures/pre_pydantic_label_strategy_hash.json"
        )
        fixture_path = (
            monorepo_root
            / "hft-contracts"
            / "tests"
            / "fixtures"
            / "pre_pydantic_label_strategy_hash.json"
        )
        with open(fixture_path) as f:
            frozen = _json.load(f)
        expected_hash = frozen["label_strategy_hash"]

        real_hash = compute_label_strategy_hash(LabelsConfig())
        assert real_hash == expected_hash, (
            f"A.5.3a.1 hardening rotated the fixture hash — byte-identity "
            f"broken. Real: {real_hash}, fixture: {expected_hash}. Possible "
            f"cause: new field, removed field, changed canonical form, or "
            f"Pydantic field type change that leaks into model_dump."
        )


# =============================================================================
# Phase A.5.3b (2026-04-24) regression locks — SafeBaseModel + SequenceConfig.
#
# Two new lock-in test classes:
#
#   1. TestSafeBaseModel — directly exercises the shared base class on a
#      minimal fixture subclass (no LabelsConfig-specific fields), proving
#      the 4 hardening patterns fire correctly at the base-class level.
#      Prevents silent stripping of patterns if future Pydantic-version
#      bumps change ConfigDict inheritance semantics.
#
#   2. TestSequenceConfigPydantic — exercises SequenceConfig-specific
#      invariants (cross-field stride ≤ window_size) + confirms the
#      inherited hardening patterns work identically on the SECOND
#      class to inherit from SafeBaseModel (validates abstraction fit
#      with N≥2 subclasses).
#
# LabelsConfig tests (TestLabelsConfigPydantic + TestLabelsConfigPydanticHardening)
# are UNCHANGED — they continue to exercise identical behavior now that
# LabelsConfig inherits from SafeBaseModel. If byte-identity parity test
# still passes (verified), the refactor is semantically sound.
# =============================================================================


class TestSafeBaseModel:
    """Direct tests on the shared base class via minimal fixture subclass.

    Prevents silent pattern-stripping if Pydantic's ConfigDict inheritance
    semantics ever change. Any test here failing means SafeBaseModel isn't
    packaging the 4 hardening patterns correctly, and EVERY consumer
    subclass (LabelsConfig + SequenceConfig + future A.5.3c-i classes)
    would silently lose the guarantee.
    """

    @staticmethod
    def _make_fixture_subclass():
        """Minimal SafeBaseModel subclass for direct base-class testing.

        Defined as a factory (not module-level class) to keep this test file
        from polluting its own namespace with test-only config classes.
        """
        from lobtrainer.config.base import SafeBaseModel

        class _FixtureConfig(SafeBaseModel):
            x: int = 0
            y: str = "default"

        return _FixtureConfig

    def test_frozen_inherited_rejects_mutation(self):
        """``frozen=True`` inherited from SafeBaseModel — field assignment
        raises ValidationError on ANY subclass."""
        from pydantic import ValidationError

        Config = self._make_fixture_subclass()
        cfg = Config()
        with pytest.raises(ValidationError):
            cfg.x = 42  # type: ignore[misc]
        with pytest.raises(ValidationError):
            cfg.y = "changed"  # type: ignore[misc]

    def test_extra_forbid_inherited_rejects_typo(self):
        """``extra="forbid"`` inherited — unknown kwargs raise
        ValidationError at construction."""
        from pydantic import ValidationError

        Config = self._make_fixture_subclass()
        with pytest.raises(ValidationError):
            Config(z=99)  # type: ignore[call-arg]

    def test_strict_inherited_rejects_string_to_int(self):
        """``strict=True`` inherited — string-to-int coercion rejected."""
        from pydantic import ValidationError

        Config = self._make_fixture_subclass()
        with pytest.raises(ValidationError):
            Config(x="42")  # string → int would be silent under lax

    def test_strict_inherited_rejects_bool_as_int(self):
        """Bool is int-subclass but strict mode inherited from SafeBaseModel
        rejects the coercion (ship-blocker #3 from A.5.3a.1)."""
        from pydantic import ValidationError

        Config = self._make_fixture_subclass()
        with pytest.raises(ValidationError):
            Config(x=True)

    def test_model_copy_inherited_revalidates_on_update(self):
        """Inherited ``model_copy`` override re-runs validators on update.

        Without the override, default Pydantic v2 model_copy would accept
        ``update={"x": "not_an_int"}`` silently (bypasses validators).
        Ship-blocker #1 from A.5.3a.1 closed at SafeBaseModel level.
        """
        from pydantic import ValidationError

        Config = self._make_fixture_subclass()
        cfg = Config(x=10)
        with pytest.raises(ValidationError):
            cfg.model_copy(update={"x": "not_an_int"})
        with pytest.raises(ValidationError):
            cfg.model_copy(update={"y": 42})  # int where str expected under strict

    def test_model_copy_inherited_no_update_fast_path(self):
        """Positive control: ``model_copy()`` with no update does NOT trigger
        validation (delegates to super() — matches Pydantic default semantics
        for pure copy)."""
        Config = self._make_fixture_subclass()
        cfg = Config(x=42, y="preserved")
        cfg2 = cfg.model_copy()
        assert cfg2.x == 42
        assert cfg2.y == "preserved"
        # Deep copy flag still works
        cfg3 = cfg.model_copy(deep=True)
        assert cfg3.x == 42

    def test_model_copy_inherited_valid_update_succeeds(self):
        """Valid update via model_copy works normally — no unexpected
        rejection of legitimate updates."""
        Config = self._make_fixture_subclass()
        cfg = Config(x=10, y="a")
        cfg2 = cfg.model_copy(update={"x": 20, "y": "b"})
        assert cfg2.x == 20
        assert cfg2.y == "b"

    def test_subclass_model_config_inherits_not_required_to_redeclare(self):
        """Defensive: subclass without inline ``model_config`` MUST inherit
        frozen+extra_forbid+strict. Test verifies Pydantic's inheritance
        of ConfigDict works as documented.

        (If this ever fails, Pydantic semantics have changed and every
        migrated class in A.5.3a-i risks silent pattern-stripping.)
        """
        Config = self._make_fixture_subclass()
        # Check ConfigDict keys are inherited
        assert Config.model_config.get("frozen") is True
        assert Config.model_config.get("extra") == "forbid"
        assert Config.model_config.get("strict") is True


class TestSequenceConfigPydantic:
    """SequenceConfig-specific Pydantic regression tests.

    Confirms inherited SafeBaseModel hardening patterns fire correctly on
    the SECOND class to inherit (N≥2 validates abstraction fit) + locks
    SequenceConfig-specific invariants (cross-field stride ≤ window_size).
    """

    def test_frozen_rejects_mutation(self):
        """Field assignment raises ValidationError (inherited from SafeBaseModel)."""
        from pydantic import ValidationError
        cfg = SequenceConfig()
        with pytest.raises(ValidationError):
            cfg.window_size = 200  # type: ignore[misc]

    def test_extra_forbid_rejects_typo(self):
        """Typo ``stide`` (for ``stride``) raises ValidationError (inherited)."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SequenceConfig(stide=10)  # type: ignore[call-arg]

    def test_strict_rejects_string_window_size(self):
        """Strict mode inherited — ``window_size="100"`` rejected (bug #2 from A.5.3a.1)."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SequenceConfig(window_size="100")  # type: ignore[arg-type]

    def test_cross_field_validator_stride_ge_window(self):
        """Cross-field invariant ``stride ≤ window_size`` enforced at
        ``@model_validator(mode="after")`` level (can't use ``@field_validator``
        — single-field validators can't access sibling fields)."""
        from pydantic import ValidationError
        # stride > window_size must raise
        with pytest.raises(ValidationError, match="stride.*should not exceed"):
            SequenceConfig(window_size=10, stride=20)
        # stride == window_size is allowed (boundary)
        cfg = SequenceConfig(window_size=50, stride=50)
        assert cfg.stride == 50

    def test_model_copy_inherited_revalidates_stride_invariant(self):
        """Inherited ``model_copy`` override re-runs the cross-field
        validator on update. Without this, ``cfg.model_copy(update={"stride": 999})``
        would silently produce a config violating the stride ≤ window_size
        invariant — critical for sweep-axis parameter mutation paths."""
        from pydantic import ValidationError
        cfg = SequenceConfig(window_size=100, stride=10)
        # Invalid update — stride exceeds window_size
        with pytest.raises(ValidationError, match="stride.*should not exceed"):
            cfg.model_copy(update={"stride": 200})
        # Valid update
        cfg2 = cfg.model_copy(update={"stride": 50})
        assert cfg2.stride == 50
        assert cfg2.window_size == 100  # unchanged


class TestNormalizationConfigPydantic:
    """Phase A.5.3c (2026-04-24) regression locks for NormalizationConfig migration.

    NormalizationConfig is the FIRST Pydantic-migrated class in the Phase A.5
    cycle with an Enum field (``strategy: NormalizationStrategy``). The
    ``@field_validator(mode="before")`` pattern below is the precedent for
    all future enum-field migrations (A.5.3e TrainConfig.task_type,
    A.5.3g DataConfig.labeling_strategy, A.5.3h ModelConfig.model_type/
    deeplob_mode).

    Also exercises:
    - Tuple[int, ...] immutability on exclude_features (A.5.3a.1 pattern)
    - NaN-finite check on eps + clip_value (A.5.3a.1 pattern)
    - Inherited SafeBaseModel semantics (frozen/extra_forbid/strict/model_copy)
    """

    def test_frozen_rejects_mutation(self):
        """Field assignment raises ValidationError (inherited from SafeBaseModel)."""
        from pydantic import ValidationError
        cfg = NormalizationConfig()
        with pytest.raises(ValidationError):
            cfg.eps = 1e-5  # type: ignore[misc]

    def test_extra_forbid_rejects_typo(self):
        """Typo ``clippy_value`` (for ``clip_value``) raises ValidationError."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            NormalizationConfig(clippy_value=5.0)  # type: ignore[call-arg]

    def test_strict_rejects_string_eps(self):
        """Strict mode inherited — ``eps="1e-8"`` rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            NormalizationConfig(eps="1e-8")  # type: ignore[arg-type]

    def test_strict_rejects_string_in_exclude_features_items(self):
        """Strict Tuple[int, ...] rejects string items.

        Pre-A.5.3c this would silently coerce to int (ship-blocker #2 from
        A.5.3a.1). The @field_validator(mode="before") coerces list→tuple
        BUT the strict type check still fires on the inner int items.
        """
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            NormalizationConfig(exclude_features=["93"])

    def test_nan_eps_rejected(self):
        """NaN eps passes strict float type check but fails explicit
        math.isfinite check in @model_validator (IEEE 754 considers NaN
        a valid float — per A.5.3a.1 pattern, explicit check is required)."""
        import math
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="finite"):
            NormalizationConfig(eps=float("nan"))
        with pytest.raises(ValidationError, match="finite"):
            NormalizationConfig(eps=float("inf"))

    def test_nan_clip_value_rejected(self):
        """Same NaN-finite check applies to clip_value (when not None)."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="finite"):
            NormalizationConfig(clip_value=float("nan"))

    def test_clip_value_none_allowed(self):
        """Positive control: clip_value=None is valid (disables clipping)."""
        cfg = NormalizationConfig(clip_value=None)
        assert cfg.clip_value is None

    def test_clip_value_negative_rejected(self):
        """clip_value must be > 0 (legacy invariant, preserved)."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="clip_value must be > 0"):
            NormalizationConfig(clip_value=-1.0)

    def test_strategy_accepts_yaml_string_input(self):
        """YAML loads strategies as strings. Pre-A.5.3c dacite had
        ``cast=[Enum, ...]`` that converted strings to enums BEFORE
        construction. Under strict=True Pydantic, strings would normally
        reject. The @field_validator(mode="before") converts string→Enum
        BEFORE the strict type check fires.

        This is the PATTERN for all subsequent enum-field migrations.
        """
        from lobtrainer.config.schema import NormalizationStrategy

        # String input (YAML-style) → coerced to Enum instance
        cfg = NormalizationConfig(strategy="zscore_per_day")
        assert cfg.strategy == NormalizationStrategy.ZSCORE_PER_DAY
        assert isinstance(cfg.strategy, NormalizationStrategy)

        # Enum instance input (programmatic) → passthrough
        cfg2 = NormalizationConfig(strategy=NormalizationStrategy.NONE)
        assert cfg2.strategy == NormalizationStrategy.NONE

        # Invalid string → ValidationError (wrapped ValueError from Enum())
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            NormalizationConfig(strategy="not_a_real_strategy")

    def test_exclude_features_tuple_immutable(self):
        """exclude_features is Tuple[int, ...] post-migration — .append raises.

        Pre-A.5.3c this was List[int] which would silently allow
        ``cfg.exclude_features.append(99)`` to mutate the frozen config.
        Tuple is truly immutable — closes the frozen-bypass for this field
        identically to how LabelsConfig.horizons was closed in A.5.3a.1.
        """
        cfg = NormalizationConfig(exclude_features=[93])
        assert isinstance(cfg.exclude_features, tuple)
        assert cfg.exclude_features == (93,)
        with pytest.raises(AttributeError):
            cfg.exclude_features.append(99)  # type: ignore[attr-defined]

    def test_exclude_features_accepts_yaml_list(self):
        """Positive control — YAML list input gets coerced to tuple via
        @field_validator(mode="before")."""
        cfg = NormalizationConfig(exclude_features=[10, 93, 111])
        assert cfg.exclude_features == (10, 93, 111)
        assert isinstance(cfg.exclude_features, tuple)

    def test_model_copy_revalidates_enum_field(self):
        """Inherited model_copy override re-runs validators including
        string→Enum coercion + Enum-membership check on update."""
        from pydantic import ValidationError
        cfg = NormalizationConfig()
        # Invalid strategy string → ValidationError
        with pytest.raises(ValidationError):
            cfg.model_copy(update={"strategy": "not_a_real_strategy"})
        # Valid string update — coerced via pre-validator on the update path
        cfg2 = cfg.model_copy(update={"strategy": "none"})
        from lobtrainer.config.schema import NormalizationStrategy
        assert cfg2.strategy == NormalizationStrategy.NONE


class TestSourceConfigPydantic:
    """Phase A.5.3d (2026-04-24) regression locks for SourceConfig migration.

    SourceConfig is a simple leaf (4 scalar fields: name, data_dir, role,
    feature_count) with no Enum, no mutable containers. Uses the
    LabelsConfig-pattern ``ClassVar[frozenset[str]] _VALID_ROLES`` for
    role-value discoverability + v3-A ClassVar-leak regression coverage.
    """

    def test_frozen_rejects_mutation(self):
        """Field assignment raises ValidationError (inherited from SafeBaseModel)."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import SourceConfig
        cfg = SourceConfig(name="mbo", data_dir="/tmp", role="primary")
        with pytest.raises(ValidationError):
            cfg.name = "different"  # type: ignore[misc]

    def test_extra_forbid_rejects_typo(self):
        """Typo rejected by extra='forbid' (inherited)."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import SourceConfig
        with pytest.raises(ValidationError):
            SourceConfig(nmae="mbo")  # type: ignore[call-arg]

    def test_strict_rejects_string_feature_count(self):
        """Strict mode — string-to-int rejected (inherited)."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import SourceConfig
        with pytest.raises(ValidationError):
            SourceConfig(feature_count="98")  # type: ignore[arg-type]

    def test_strict_rejects_bool_feature_count(self):
        """bool-to-int rejected (inherited — bug #3 from A.5.3a.1)."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import SourceConfig
        with pytest.raises(ValidationError):
            SourceConfig(feature_count=True)  # type: ignore[arg-type]

    def test_invalid_role_rejected(self):
        """Role must be 'primary' or 'auxiliary' per _VALID_ROLES frozenset."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import SourceConfig
        with pytest.raises(ValidationError, match="role must be one of"):
            SourceConfig(role="bogus")

    def test_negative_feature_count_rejected(self):
        """feature_count must be >= 0 (legacy invariant preserved)."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import SourceConfig
        with pytest.raises(ValidationError, match="feature_count"):
            SourceConfig(feature_count=-1)

    def test_class_constant_not_in_model_dump(self):
        """v3-A ClassVar discipline: _VALID_ROLES annotated as
        ClassVar[frozenset[str]] MUST be excluded from model_dump output.
        Without the ClassVar annotation, Pydantic would treat it as a
        model field, leaking into YAML round-trips."""
        from lobtrainer.config.schema import SourceConfig
        cfg = SourceConfig(name="mbo", data_dir="/tmp", role="primary")
        dump = cfg.model_dump()
        assert "_VALID_ROLES" not in dump, (
            f"ClassVar _VALID_ROLES leaked into model_dump: {list(dump.keys())}. "
            f"Annotation as `ClassVar[frozenset[str]]` is LOAD-BEARING."
        )
        # Positive control — legitimate fields ARE in model_dump.
        assert "role" in dump
        assert "feature_count" in dump

    def test_class_constant_accessible_as_class_attr(self):
        """Defensive — ClassVar annotation must NOT break class-attr access."""
        from lobtrainer.config.schema import SourceConfig
        assert "primary" in SourceConfig._VALID_ROLES
        assert "auxiliary" in SourceConfig._VALID_ROLES

    def test_model_copy_revalidates_role(self):
        """Inherited model_copy override re-runs _validate_all on update,
        rejecting invalid role mutations."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import SourceConfig
        cfg = SourceConfig(name="mbo", data_dir="/tmp", role="primary")
        with pytest.raises(ValidationError, match="role must be one of"):
            cfg.model_copy(update={"role": "bogus"})
        # Valid update passes
        cfg2 = cfg.model_copy(update={"role": "auxiliary"})
        assert cfg2.role == "auxiliary"

    def test_multi_source_via_dataconfig(self):
        """Integration: SourceConfig held as Optional[List[SourceConfig]] on
        DataConfig (@dataclass) — mixed-state List[BaseModel] construction.

        Verifies dacite type_hooks correctly route each list item through
        model_validate (critical for T12 multi-source fusion YAML configs)."""
        from lobtrainer.config.schema import DataConfig, SourceConfig
        # Programmatic construction — BaseModel instances in List
        dc = DataConfig(
            sources=[
                SourceConfig(name="mbo", data_dir="/tmp", role="primary", feature_count=98),
                SourceConfig(name="basic", data_dir="/tmp", role="auxiliary", feature_count=34),
            ],
        )
        assert dc.sources is not None
        assert len(dc.sources) == 2
        assert dc.sources[0].role == "primary"
        assert dc.sources[1].role == "auxiliary"


class TestTrainConfigPydantic:
    """Phase A.5.3e (2026-04-24) regression locks for TrainConfig migration.

    TrainConfig has 2 Enum fields (task_type, loss_type) + cross-field
    task↔loss compatibility. Uses the NormalizationConfig.strategy pattern
    for string→Enum coercion (A.5.3c precedent) + the LabelsConfig pattern
    for ClassVar frozenset constants (_CLASSIFICATION_LOSSES / _REGRESSION_LOSSES).
    """

    def test_frozen_rejects_mutation(self):
        """Field assignment raises ValidationError (inherited from SafeBaseModel)."""
        from pydantic import ValidationError
        cfg = TrainConfig()
        with pytest.raises(ValidationError):
            cfg.learning_rate = 1e-5  # type: ignore[misc]

    def test_extra_forbid_rejects_typo(self):
        """Typo ``leraning_rate`` (for ``learning_rate``) rejected (inherited)."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            TrainConfig(leraning_rate=1e-3)  # type: ignore[call-arg]

    def test_strict_rejects_string_batch_size(self):
        """Strict mode — string-to-int rejected (inherited)."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            TrainConfig(batch_size="64")  # type: ignore[arg-type]

    def test_strict_rejects_bool_batch_size(self):
        """bool-to-int rejected (inherited — bug #3 from A.5.3a.1)."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            TrainConfig(batch_size=True)  # type: ignore[arg-type]

    def test_nan_learning_rate_rejected(self):
        """NaN learning_rate passes strict float type check but fails
        explicit math.isfinite (A.5.3a.1 pattern). A NaN learning_rate
        would silently produce NaN gradients on first backward pass —
        poisons every metric downstream."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="finite"):
            TrainConfig(learning_rate=float("nan"))
        with pytest.raises(ValidationError, match="finite"):
            TrainConfig(learning_rate=float("inf"))

    def test_nan_focal_gamma_rejected(self):
        """focal_gamma NaN check (A.5.3a.1 pattern)."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="finite"):
            TrainConfig(focal_gamma=float("nan"))

    def test_nan_focal_alpha_rejected(self):
        """focal_alpha NaN check — Optional[float] so None path doesn't
        hit isfinite; non-None NaN must be rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="finite"):
            TrainConfig(
                task_type=TaskType.MULTICLASS,
                loss_type=LossType.FOCAL,
                focal_alpha=float("nan"),
            )

    def test_task_type_accepts_yaml_string_input(self):
        """YAML-style string → Enum coercion (A.5.3c pattern)."""
        cfg = TrainConfig(task_type="regression", loss_type="huber")
        assert cfg.task_type == TaskType.REGRESSION
        assert cfg.loss_type == LossType.HUBER

    def test_loss_type_invalid_string_rejected(self):
        """Invalid Enum value raises (clean error from Enum constructor)."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            TrainConfig(loss_type="not_a_real_loss")

    def test_model_copy_revalidates_cross_field_task_loss(self):
        """Inherited model_copy override MUST re-run _validate_all,
        rejecting task↔loss incompatibility on update.

        Without this, ``cfg.model_copy(update={"loss_type": LossType.HUBER})``
        on a classification config would silently produce an invalid
        TrainConfig (HUBER is regression-only). Sweep axis mutation paths
        are the primary consumer of this guard."""
        from pydantic import ValidationError
        cfg = TrainConfig(task_type=TaskType.MULTICLASS, loss_type=LossType.FOCAL)
        # Invalid: switching loss to regression on classification task
        with pytest.raises(ValidationError, match="regression loss"):
            cfg.model_copy(update={"loss_type": LossType.HUBER})
        # Valid: consistent pair — classification + CE
        cfg2 = cfg.model_copy(update={"loss_type": LossType.CROSS_ENTROPY})
        assert cfg2.loss_type == LossType.CROSS_ENTROPY

    def test_class_constants_not_in_model_dump(self):
        """v3-A ClassVar discipline — promoted frozensets MUST NOT leak
        into model_dump output. Without ClassVar annotations, Pydantic
        would expose these as model fields, breaking YAML round-trips."""
        cfg = TrainConfig()
        dump = cfg.model_dump()
        assert "_CLASSIFICATION_LOSSES" not in dump, (
            f"_CLASSIFICATION_LOSSES leaked into model_dump: {list(dump.keys())}"
        )
        assert "_REGRESSION_LOSSES" not in dump
        # Positive control: real fields ARE in model_dump
        assert "task_type" in dump
        assert "loss_type" in dump

    def test_class_constants_accessible_as_class_attrs(self):
        """Defensive — ClassVar annotation must NOT break class-level access."""
        assert LossType.CROSS_ENTROPY in TrainConfig._CLASSIFICATION_LOSSES
        assert LossType.FOCAL in TrainConfig._CLASSIFICATION_LOSSES
        assert LossType.HUBER in TrainConfig._REGRESSION_LOSSES
        assert LossType.MSE in TrainConfig._REGRESSION_LOSSES


class TestCVConfigPydantic:
    """Phase A.5.3f (2026-04-24) regression locks for CVConfig migration.

    CVConfig is a simple leaf (2 scalar int fields) with no Enum, no
    mutable containers, no class-level constants. Inherited SafeBaseModel
    semantics (frozen/extra_forbid/strict/model_copy) all apply identically.
    """

    def test_frozen_rejects_mutation(self):
        """Field assignment raises ValidationError (inherited from SafeBaseModel)."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import CVConfig
        cv = CVConfig()
        with pytest.raises(ValidationError):
            cv.n_splits = 10  # type: ignore[misc]

    def test_extra_forbid_rejects_typo(self):
        """Typo ``n_split`` (for ``n_splits``) rejected by extra='forbid'."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import CVConfig
        with pytest.raises(ValidationError):
            CVConfig(n_split=5)  # type: ignore[call-arg]

    def test_strict_rejects_string_n_splits(self):
        """Strict mode — string-to-int rejected."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import CVConfig
        with pytest.raises(ValidationError):
            CVConfig(n_splits="5")  # type: ignore[arg-type]

    def test_strict_rejects_bool_embargo_days(self):
        """bool-to-int rejected (inherited from SafeBaseModel strict mode)."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import CVConfig
        with pytest.raises(ValidationError):
            CVConfig(embargo_days=True)  # type: ignore[arg-type]

    def test_model_copy_revalidates_invariants(self):
        """Inherited model_copy override re-runs validators on update."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import CVConfig
        cv = CVConfig(n_splits=5, embargo_days=1)
        # Invalid update
        with pytest.raises(ValidationError, match="n_splits must be >= 2"):
            cv.model_copy(update={"n_splits": 1})
        # Valid update
        cv2 = cv.model_copy(update={"n_splits": 10, "embargo_days": 2})
        assert cv2.n_splits == 10
        assert cv2.embargo_days == 2

    def test_defaults_preserved(self):
        """Positive control — defaults match pre-migration dataclass."""
        from lobtrainer.config.schema import CVConfig
        cv = CVConfig()
        assert cv.n_splits == 5
        assert cv.embargo_days == 1
