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


class TestDataConfigPydantic:
    """Phase A.5.3g (2026-04-24) regression locks for DataConfig migration.

    DataConfig is the most complex leaf migration in Phase A.5:
    - Composite holding 4 SafeBaseModel children (LabelsConfig, SequenceConfig,
      NormalizationConfig, SourceConfig)
    - 2 PrivateAttr resolver caches (_feature_indices_resolved,
      _feature_set_ref_resolved) — mutable under frozen, excluded from
      model_dump AND __eq__ (see SafeBaseModel.__eq__ override).
    - 2 Enum coercers (labeling_strategy + label_encoding)
    - 1 Tuple immutability pattern (feature_indices: List → Tuple)
    - T9 labels auto-derivation via object.__setattr__ (in-validator
      cross-field self-mutation)
    - T12 sources exactly-one-primary + unique-names validation
    - 3-field mutual exclusion (feature_set / feature_indices / feature_preset)
    - feature_preset DeprecationWarning emission
    """

    # --- Core hardening (inherited SafeBaseModel semantics) -----------------

    def test_frozen_rejects_mutation(self):
        """Public field assignment raises ValidationError under frozen=True."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import DataConfig
        cfg = DataConfig()
        with pytest.raises(ValidationError):
            cfg.data_dir = "/somewhere/else"  # type: ignore[misc]
        with pytest.raises(ValidationError):
            cfg.feature_count = 40  # type: ignore[misc]
        with pytest.raises(ValidationError):
            cfg.feature_set = "foo_v1"  # type: ignore[misc]

    def test_extra_forbid_rejects_typo(self):
        """Typo ``feature_cont`` (for ``feature_count``) rejected."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import DataConfig
        with pytest.raises(ValidationError):
            DataConfig(feature_cont=98)  # type: ignore[call-arg]

    def test_strict_rejects_string_feature_count(self):
        """Strict mode — string-to-int rejected on feature_count."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import DataConfig
        with pytest.raises(ValidationError):
            DataConfig(feature_count="98")  # type: ignore[arg-type]

    def test_strict_rejects_bool_feature_count(self):
        """Strict mode — bool rejected (no bool-is-int coercion)."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import DataConfig
        with pytest.raises(ValidationError):
            DataConfig(feature_count=True)  # type: ignore[arg-type]

    # --- Enum coercion for YAML string input --------------------------------

    def test_labeling_strategy_accepts_yaml_string(self):
        """YAML ``labeling_strategy: "tlob"`` → LabelingStrategy.TLOB
        (pre-validator bridge — strict mode would otherwise reject string).
        """
        from lobtrainer.config.schema import DataConfig, LabelingStrategy
        cfg = DataConfig(labeling_strategy="regression")
        assert cfg.labeling_strategy == LabelingStrategy.REGRESSION

    def test_label_encoding_accepts_yaml_string(self):
        """YAML ``label_encoding: "binary_up"`` → LabelEncoding.BINARY_UP
        (pre-validator bridge — strict mode would otherwise reject string).
        """
        from lobtrainer.config.schema import DataConfig, LabelEncoding
        cfg = DataConfig(label_encoding="binary_up")
        assert cfg.label_encoding == LabelEncoding.BINARY_UP

    def test_labeling_strategy_invalid_string_rejected(self):
        """Unknown ``labeling_strategy: "bogus"`` rejected — Enum coercer
        raises when the string is not a valid variant.
        """
        from pydantic import ValidationError
        from lobtrainer.config.schema import DataConfig
        with pytest.raises(ValidationError):
            DataConfig(labeling_strategy="bogus_strategy")

    # --- Tuple immutability + list→tuple coercion ---------------------------

    def test_feature_indices_accepts_yaml_list(self):
        """YAML ``feature_indices: [0, 5, 12]`` coerced to tuple."""
        from lobtrainer.config.schema import DataConfig
        cfg = DataConfig(feature_indices=[0, 5, 12])
        assert cfg.feature_indices == (0, 5, 12)
        assert isinstance(cfg.feature_indices, tuple)

    def test_feature_indices_is_immutable_post_coercion(self):
        """After list→tuple coercion, the field IS tuple (no .append)."""
        from lobtrainer.config.schema import DataConfig
        cfg = DataConfig(feature_indices=[0, 5, 12])
        with pytest.raises(AttributeError):
            cfg.feature_indices.append(99)  # type: ignore[union-attr]

    # --- 3-field mutual exclusion (Phase 4 Batch 4c) ------------------------

    def test_feature_set_plus_indices_rejected(self):
        """``feature_set`` + ``feature_indices`` both set rejects at construction."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import DataConfig
        with pytest.raises(ValidationError, match="At most one feature-selection field"):
            DataConfig(feature_set="foo_v1", feature_indices=[0, 5])

    def test_feature_set_plus_preset_rejected(self):
        """``feature_set`` + ``feature_preset`` both set rejects."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import DataConfig
        with pytest.raises(ValidationError, match="At most one feature-selection field"):
            # Use a preset that exists so we test the mutual-exclusion
            # check, not the "unknown preset" check.
            DataConfig(feature_set="foo_v1", feature_preset="short_term_40")

    # --- feature_preset DeprecationWarning (Phase 4 4c) ---------------------

    def test_feature_preset_emits_deprecation_warning(self):
        """``feature_preset`` still works but emits DeprecationWarning."""
        from lobtrainer.config.schema import DataConfig
        with pytest.warns(DeprecationWarning, match="DEPRECATED"):
            DataConfig(feature_preset="short_term_40")

    # --- T12 sources cross-field validation ---------------------------------

    def test_sources_exactly_one_primary_enforced(self):
        """T12: sources must have exactly ONE role='primary'."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import DataConfig, SourceConfig
        # Zero primary
        with pytest.raises(ValidationError, match="Exactly one source"):
            DataConfig(sources=[
                SourceConfig(name="a", data_dir="/a", role="auxiliary"),
                SourceConfig(name="b", data_dir="/b", role="auxiliary"),
            ])
        # Two primary
        with pytest.raises(ValidationError, match="Exactly one source"):
            DataConfig(sources=[
                SourceConfig(name="a", data_dir="/a", role="primary"),
                SourceConfig(name="b", data_dir="/b", role="primary"),
            ])

    def test_sources_unique_names_enforced(self):
        """T12: sources must have unique ``name`` values."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import DataConfig, SourceConfig
        with pytest.raises(ValidationError, match="Duplicate source names"):
            DataConfig(sources=[
                SourceConfig(name="dup", data_dir="/a", role="primary"),
                SourceConfig(name="dup", data_dir="/b", role="auxiliary"),
            ])

    # --- T9 labels auto-derivation via object.__setattr__ -------------------

    def test_labels_auto_derived_from_labeling_strategy(self):
        """When ``labels is None``, T9 validator derives LabelsConfig from
        legacy fields via ``object.__setattr__`` (in-validator cross-field
        mutation under frozen=True)."""
        from lobtrainer.config.schema import DataConfig, LabelingStrategy
        # Default (tlob) → classification
        cfg = DataConfig()
        assert cfg.labels is not None
        assert cfg.labels.task == "classification"
        # Regression strategy → regression labels
        cfg_reg = DataConfig(labeling_strategy="regression")
        assert cfg_reg.labels is not None
        assert cfg_reg.labels.task == "regression"

    # --- PrivateAttr semantics (mutable, excluded from dump + eq) -----------

    def test_private_attr_is_mutable_under_frozen(self):
        """PrivateAttr fields mutable even with frozen=True (Pydantic design)."""
        from lobtrainer.config.schema import DataConfig
        cfg = DataConfig(feature_set="foo_v1")
        # Both resolver writes succeed (resolver pattern at trainer.py:416-417)
        cfg._feature_indices_resolved = [0, 5, 12]
        cfg._feature_set_ref_resolved = ("foo_v1", "a" * 64)
        assert cfg._feature_indices_resolved == [0, 5, 12]
        assert cfg._feature_set_ref_resolved == ("foo_v1", "a" * 64)

    def test_private_attr_excluded_from_model_dump(self):
        """PrivateAttr MUST NOT leak into model_dump() — Phase 4 R3 invariant."""
        from lobtrainer.config.schema import DataConfig
        cfg = DataConfig(feature_set="foo_v1")
        cfg._feature_indices_resolved = [0, 5, 12]
        cfg._feature_set_ref_resolved = ("foo_v1", "a" * 64)
        dumped = cfg.model_dump()
        assert "_feature_indices_resolved" not in dumped
        assert "_feature_set_ref_resolved" not in dumped
        assert "feature_set" in dumped  # public field IS preserved

    def test_private_attr_excluded_from_eq(self):
        """PrivateAttr MUST NOT affect __eq__ — Phase 4 R3 "cache is not
        semantic identity". SafeBaseModel.__eq__ override ensures this."""
        from lobtrainer.config.schema import DataConfig
        c1 = DataConfig(feature_set="x_v1")
        c2 = DataConfig(feature_set="x_v1")
        c2._feature_indices_resolved = [0, 5, 12]
        c2._feature_set_ref_resolved = ("x_v1", "a" * 64)
        assert c1 == c2, (
            "Cache state must not affect equality — violates Phase 4 R3 "
            "if two configs with same user fields but different cache state "
            "compare unequal. See SafeBaseModel.__eq__ override."
        )

    def test_eq_distinguishes_public_fields(self):
        """Positive control — __eq__ DOES distinguish public-field changes."""
        from lobtrainer.config.schema import DataConfig
        c1 = DataConfig(feature_count=98)
        c2 = DataConfig(feature_count=40)
        assert c1 != c2

    # --- Class-level constants (v3-A ClassVar discipline) -------------------
    #
    # DataConfig has NO class-level frozenset/Tuple constants. The VALID_* sets
    # (VALID_FEATURE_COUNTS) live INSIDE ``_validate_all`` as local vars,
    # NOT class-level — they do not need ClassVar annotation. This test
    # documents that absence + locks the baseline.

    def test_no_class_constants_leak_into_model_dump(self):
        """Baseline check — DataConfig has no class-level constants, so
        model_dump() has only public fields + inherited PrivateAttr exclusion.
        """
        from lobtrainer.config.schema import DataConfig
        cfg = DataConfig()
        dumped = cfg.model_dump()
        # No accidental class constants leaking in
        assert "VALID_FEATURE_COUNTS" not in dumped
        # Only expected public fields
        assert "data_dir" in dumped
        assert "feature_count" in dumped


class TestExperimentConfigPydantic:
    """Phase A.5.3i (2026-04-24 KEYSTONE) regression locks for ExperimentConfig.

    KEYSTONE commit of Phase A.5 Scope D — closes the entire migration
    cycle. ExperimentConfig is the top-level composite holding all 8
    already-migrated sub-configs. Exercises the complete trust-chain:
    ``from_dict`` → ``model_validate`` → recursive nested-BaseModel
    construction with full validator fire-through.

    Retires 4 bug classes at the TYPE layer:

    1. Silent mutation (``config.output_dir = X`` raises)
    2. Extra-field acceptance (``ExperimentConfig(names='foo')`` raises)
    3. Canonical-path-drift (``config.labels`` AttributeError — no such
       field on ExperimentConfig; canonical is ``config.data.labels``)
    4. Silent-None field access (typed fields enforced by Pydantic)

    Key NEW patterns introduced in A.5.3i (not present in A.5.3a-h):

    - ``@field_validator("importance", mode="before")`` replaces the
      ``_coerce_importance(self)`` helper that mutated post-construction.
    - ``object.__setattr__(self, "model", ...)`` for T13 auto-derive
      under frozen=True (matches DataConfig's T9 + ModelConfig's params
      self-mutation pattern).
    - ``from_dict`` body rewritten to ``cls.model_validate(data)``; dacite
      retired from dependency tree.
    - CLI override pattern: outer ``config.model_copy(update={...})``
      with ``_top_overrides`` accumulator dict (3 production sites:
      cli.py, scripts/train.py, scripts/export_signals.py).
    """

    # --- Core hardening (inherited SafeBaseModel semantics) -----------------

    def test_frozen_rejects_mutation(self):
        """Top-level public field assignment raises ValidationError."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import ExperimentConfig
        cfg = ExperimentConfig()
        with pytest.raises(ValidationError):
            cfg.output_dir = "/tmp/foo"  # type: ignore[misc]
        with pytest.raises(ValidationError):
            cfg.name = "bar"  # type: ignore[misc]
        with pytest.raises(ValidationError):
            cfg.data = cfg.data  # even reassigning same value raises  # type: ignore[misc]

    def test_extra_forbid_rejects_typo(self):
        """Typo ``names`` (for ``name``) rejected by extra='forbid'."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import ExperimentConfig
        with pytest.raises(ValidationError):
            ExperimentConfig(names="foo")  # type: ignore[call-arg]

    def test_canonical_path_drift_catches_config_labels(self):
        """Bug class #1 (canonical-path-drift) retired at type layer:
        ``config.labels`` does NOT exist (canonical is ``config.data.labels``).
        Pydantic returns AttributeError on missing field access — same
        mode as Python default, but with a clear "no such attribute"
        diagnostic (no silent None-coalesce).
        """
        from lobtrainer.config.schema import ExperimentConfig
        cfg = ExperimentConfig()
        with pytest.raises(AttributeError):
            cfg.labels  # type: ignore[attr-defined]

    def test_strict_rejects_string_name(self):
        """Strict mode — for typed str field, string value is valid;
        testing that int is rejected under strict."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import ExperimentConfig
        with pytest.raises(ValidationError):
            ExperimentConfig(name=42)  # type: ignore[arg-type]

    # --- from_dict delegates to model_validate ------------------------------

    def test_from_dict_delegates_to_model_validate(self):
        """A.5.3i dacite retirement — from_dict body is now
        ``cls.model_validate(data)``. Still returns valid ExperimentConfig."""
        from lobtrainer.config.schema import ExperimentConfig
        cfg = ExperimentConfig.from_dict({
            "name": "smoke",
            "data": {"feature_count": 98},
            "model": {"input_size": 98},
        })
        assert cfg.name == "smoke"
        assert cfg.data.feature_count == 98
        assert cfg.model.input_size == 98

    def test_from_dict_rejects_base_key(self):
        """``_base`` inheritance belongs to from_yaml; from_dict rejects."""
        from lobtrainer.config.schema import ExperimentConfig
        with pytest.raises(ValueError, match="_base key found"):
            ExperimentConfig.from_dict({"_base": "foo.yaml"})

    def test_from_dict_normalizes_scientific_notation_strings(self):
        """Preserved behavior: YAML parsers sometimes emit scientific
        notation as string (e.g., "1e-8"). Our _normalize_config_types
        pre-processor converts before strict-mode type check fires."""
        from lobtrainer.config.schema import ExperimentConfig
        cfg = ExperimentConfig.from_dict({
            "name": "sci",
            "data": {"feature_count": 98, "normalization": {"eps": "1e-8"}},
            "model": {"input_size": 98},
        })
        assert cfg.data.normalization.eps == 1e-8
        assert isinstance(cfg.data.normalization.eps, float)

    # --- importance @field_validator(mode="before") -------------------------

    def test_importance_field_coerces_dict_to_config(self):
        """A.5.3i moved _coerce_importance from __post_init__ mutation
        to @field_validator(mode='before'). Dict → ImportanceConfig
        coercion preserved; ImportanceConfig's own validators fire."""
        from lobtrainer.config.schema import ExperimentConfig
        cfg = ExperimentConfig(
            name="imp",
            data={"feature_count": 98},
            model={"input_size": 98},
            importance={
                "enabled": True,
                "method": "permutation",
                "n_permutations": 5,
                "block_length_samples": 10,
                "eval_split": "test",
            },
        )
        assert type(cfg.importance).__name__ == "ImportanceConfig"
        assert cfg.importance.enabled is True

    def test_importance_field_none_passes_through(self):
        """Default None (no importance analysis) stays None."""
        from lobtrainer.config.schema import ExperimentConfig
        cfg = ExperimentConfig()
        assert cfg.importance is None

    def test_importance_field_rejects_bad_type(self):
        """TypeError for anything other than None / dict / ImportanceConfig."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import ExperimentConfig
        with pytest.raises(ValidationError):
            ExperimentConfig(importance="invalid_string")

    # --- T13 auto-derive via object.__setattr__ -----------------------------

    def test_t13_auto_derive_model_input_size(self):
        """With model.input_size=0 (sentinel), T13 auto-derives from
        data.feature_count. Uses object.__setattr__ inside
        @model_validator(mode='after') — frozen bypass pattern."""
        from lobtrainer.config.schema import ExperimentConfig
        cfg = ExperimentConfig.from_dict({
            "name": "t13",
            "data": {"feature_count": 40},
            "model": {"input_size": 0},  # auto-derive sentinel
        })
        assert cfg.model.input_size == 40
        # Params dict also updated in-place for num_features / input_size keys
        # (preserved behavior from pre-migration)
        if "num_features" in cfg.model.params:
            assert cfg.model.params["num_features"] == 40

    def test_t13_raises_on_explicit_mismatch(self):
        """Pre-migration behavior preserved: explicit input_size ≠ resolved
        feature_count raises ValueError (wrapped by Pydantic as
        ValidationError)."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import ExperimentConfig
        with pytest.raises(ValidationError, match="!= resolved"):
            ExperimentConfig.from_dict({
                "name": "mismatch",
                "data": {"feature_count": 40},
                "model": {"input_size": 98},
            })

    # --- model_copy + CLI override pattern ----------------------------------

    def test_model_copy_re_fires_validators(self):
        """Inherited SafeBaseModel.model_copy(update=...) override re-runs
        validators — catches invalid user overrides at CLI time."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import ExperimentConfig
        cfg = ExperimentConfig.from_dict({
            "name": "orig", "data": {"feature_count": 98}, "model": {"input_size": 98}
        })
        # Valid update — works
        cfg2 = cfg.model_copy(update={"output_dir": "/new/path"})
        assert cfg2.output_dir == "/new/path"
        # Invalid update — re-firing validators catches it
        with pytest.raises(ValidationError):
            cfg.model_copy(update={"name": 42})  # int rejected under strict str

    def test_cli_override_pattern_atomic(self):
        """Two-layer override pattern: inner sub-config model_copy + outer
        ExperimentConfig model_copy — matches cli.py apply_overrides path
        (post-A.5.3i). Both layers' validators fire."""
        from lobtrainer.config.schema import ExperimentConfig
        cfg = ExperimentConfig()
        # Build new TrainConfig inline
        new_train = cfg.train.model_copy(update={"epochs": 50, "batch_size": 32})
        # Swap into outer via single model_copy
        cfg2 = cfg.model_copy(update={
            "train": new_train,
            "output_dir": "/new/out",
        })
        assert cfg2.train.epochs == 50
        assert cfg2.train.batch_size == 32
        assert cfg2.output_dir == "/new/out"
        # Original untouched (pure semantics)
        assert cfg.train.epochs == 100
        assert cfg.output_dir == "outputs"

    # --- Live-YAML corpus regression (ship-blocker per plan v4) -------------

    def test_live_yaml_corpus_loads_without_validation_error(self):
        """CRITICAL (plan v4 ship-blocker): every production + test-fixture
        YAML under configs/experiments/ must load via
        ``ExperimentConfig.from_yaml(path)`` WITHOUT ValidationError.

        Under ``extra='forbid'``, any YAML with a latent typo (or a field
        post-A.5.3i rejects) would silently accept pre-migration but fail
        post-migration. This test catches operator-facing regressions at
        CI time, not in production. Partial bases are excluded (they
        require merge resolution; tested separately by test_partial_base_*).

        Exclusions:
          - ``configs/archive/**`` — historical, pre-migration schemas
            (operator has explicitly moved out of active use)
          - ``nvda_tlob_triple_barrier_11mo_v1.yaml`` — pre-existing
            yaml.SafeLoader syntax error (line 124 uses Python-style
            ``\"\"\"`` triple-quote comments which are NOT valid YAML).
            This file has been broken since before Phase A.5; needs a
            full rewrite (descriptive comments → YAML # comments) in a
            separate follow-up. Unrelated to A.5.3i migration.
        """
        import glob
        from pathlib import Path
        from lobtrainer.config.schema import ExperimentConfig
        from lobtrainer.config.merge import is_partial_base

        configs_dir = Path(__file__).parent.parent / "configs"
        yaml_paths = sorted(configs_dir.glob("**/*.yaml"))
        assert len(yaml_paths) > 0, "no YAMLs found under configs/"

        # Pre-existing yaml-syntax-error YAMLs (not related to A.5.3i).
        # Document each exclusion so future contributors can resurrect.
        _KNOWN_YAML_SYNTAX_BROKEN = {
            "nvda_tlob_triple_barrier_11mo_v1.yaml",
        }

        errors = []
        for path in yaml_paths:
            if is_partial_base(path):
                continue  # partial bases are standalone-invalid by design
            # Skip archive (historical, may have pre-migration schemas)
            if "archive" in path.parts:
                continue
            if path.name in _KNOWN_YAML_SYNTAX_BROKEN:
                continue
            try:
                _ = ExperimentConfig.from_yaml(str(path))
            except Exception as e:
                errors.append(f"{path.relative_to(configs_dir)}: {type(e).__name__}: {str(e)[:200]}")
        assert not errors, (
            "\nLive-YAML corpus regression post-A.5.3i:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    def test_partial_base_rejected_on_direct_load(self):
        """Partial bases (``_partial: true``) cannot be loaded standalone —
        must be composed via multi-base inheritance.

        Phase A.5.7b SB-4: this test is the PRESENCE smoke check. The
        parametric test below (``test_all_partial_bases_rejected_on_direct_load``)
        exercises ALL 22 partial bases — this single-input test stays as a
        readability anchor (faster failure when the corpus is empty).
        """
        import glob
        from pathlib import Path
        from lobtrainer.config.schema import ExperimentConfig
        from lobtrainer.config.merge import is_partial_base

        configs_dir = Path(__file__).parent.parent / "configs"
        partials = [p for p in configs_dir.glob("**/*.yaml") if is_partial_base(p)]
        if not partials:
            pytest.skip("no partial bases in corpus")
        # Try loading ONE partial directly — must raise clearly
        with pytest.raises(ValueError, match="Partial base"):
            ExperimentConfig.from_yaml(str(partials[0]))

    @pytest.mark.parametrize(
        "partial_path",
        [
            p for p in (Path(__file__).parent.parent / "configs").glob("**/*.yaml")
            if (lambda p: p.exists() and p.is_file() and __import__(
                "lobtrainer.config.merge", fromlist=["is_partial_base"]
            ).is_partial_base(p))(p)
        ],
        ids=lambda p: str(p.relative_to(Path(__file__).parent.parent / "configs")),
    )
    def test_all_partial_bases_rejected_on_direct_load(self, partial_path):
        """Phase A.5.7b SB-4: EVERY partial base under configs/ MUST raise
        ValueError when loaded directly via ``ExperimentConfig.from_yaml``.

        Pre-A.5.7b, only ``partials[0]`` (one of 22) was tested — a
        regression that affected only specific axis types (model / dataset /
        label) would silently break 21 of 22 with no signal.

        Parametric expansion catches:
          - Per-axis schema drift (model bases vs dataset bases vs label
            bases vs train bases differ structurally)
          - YAML parse-error masking ``_partial: true`` detection
          - Future partial-base additions (CI fails when a new partial
            doesn't raise as expected)
        """
        from lobtrainer.config.schema import ExperimentConfig

        with pytest.raises(ValueError, match="Partial base"):
            ExperimentConfig.from_yaml(str(partial_path))

    # --- Fingerprint byte-stability (hft-contracts cross-repo invariant) ---

    def test_to_dict_produces_pure_dict(self):
        """to_dict() output is pure nested dict — no BaseModel leak, no
        Enum leak, no tuple-with-yaml-tag leak. Preserves YAML round-trip
        byte-identity + ensures hft-ops fingerprint computation (via
        compute_fingerprint → sanitize_for_hash on this dict) is stable."""
        from lobtrainer.config.schema import ExperimentConfig
        cfg = ExperimentConfig()
        d = cfg.to_dict()
        # Top-level is dict
        assert isinstance(d, dict)
        # All values pure — walk recursively
        def _walk(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    assert isinstance(k, (str, int, float, bool, type(None))), (
                        f"non-primitive dict key: {k!r} ({type(k).__name__})"
                    )
                    _walk(v)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    _walk(item)
            else:
                assert isinstance(obj, (str, int, float, bool, type(None))), (
                    f"non-primitive leaf: {obj!r} ({type(obj).__name__})"
                )
        _walk(d)

    # --- Registry completeness check ----------------------------------------

    def test_registry_contains_experiment_config(self):
        """Regression: ensure _PYDANTIC_CONFIG_CLASSES includes the full
        9-class set post-A.5.3i. Guards against future contributors
        forgetting to append new classes."""
        from lobtrainer.config.schema import _PYDANTIC_CONFIG_CLASSES, ExperimentConfig
        assert ExperimentConfig in _PYDANTIC_CONFIG_CLASSES
        # All 9 migrated classes
        assert len(_PYDANTIC_CONFIG_CLASSES) == 9


class TestModelConfigPydantic:
    """Phase A.5.3h (2026-04-24) regression locks for ModelConfig migration.

    ModelConfig is the last leaf before the A.5.3i ExperimentConfig keystone.
    Most field-heavy class in the cycle (~40 fields), exercising 4 distinct
    hardening patterns simultaneously:

    1. ModelType Enum string→instance coercer (A.5.3c pattern)
    2. hmhp_horizons: List[int] → Tuple[int, ...] with default + None-→-default
       + list-→-tuple coercer (A.5.3a.1 immutability + legacy None contract)
    3. logistic_feature_indices: Optional[Tuple[int, ...]] + coercer
    4. hmhp_cascade_connections: Optional[Tuple[Tuple[int, int], ...]] NESTED
       coercer (YAML yields list-of-lists; strict rejects)

    Plus: critical self-mutation via object.__setattr__ for
    _build_params_from_legacy() in _validate_all (under frozen=True,
    ``self.params = ...`` raises; same pattern as DataConfig's T9 labels
    derivation).
    """

    # --- Core hardening (inherited SafeBaseModel semantics) -----------------

    def test_frozen_rejects_mutation(self):
        """Public field assignment raises ValidationError under frozen=True."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import ModelConfig
        m = ModelConfig()
        with pytest.raises(ValidationError):
            m.input_size = 40  # type: ignore[misc]
        with pytest.raises(ValidationError):
            m.num_classes = 5  # type: ignore[misc]
        with pytest.raises(ValidationError):
            m.hmhp_horizons = (100,)  # type: ignore[misc]

    def test_extra_forbid_rejects_typo(self):
        """Typo ``input_sze`` (for ``input_size``) rejected by extra='forbid'."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import ModelConfig
        with pytest.raises(ValidationError):
            ModelConfig(input_sze=98)  # type: ignore[call-arg]

    def test_strict_rejects_string_input_size(self):
        """Strict mode — string-to-int rejected on input_size."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import ModelConfig
        with pytest.raises(ValidationError):
            ModelConfig(input_size="98")  # type: ignore[arg-type]

    def test_strict_rejects_bool_num_classes(self):
        """Strict mode — bool rejected (no bool-is-int coercion)."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import ModelConfig
        with pytest.raises(ValidationError):
            ModelConfig(num_classes=True)  # type: ignore[arg-type]

    # --- Domain invariants preserved from @dataclass __post_init__ ----------

    def test_input_size_negative_rejected(self):
        """T13 sentinel: input_size=0 allowed (auto-derive); negative rejected."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import ModelConfig
        # Zero ALLOWED (auto-derive sentinel)
        m = ModelConfig(input_size=0)
        assert m.input_size == 0
        # Negative REJECTED
        with pytest.raises(ValidationError, match="input_size must be >= 0"):
            ModelConfig(input_size=-1)

    def test_num_classes_too_small_rejected(self):
        """num_classes >= 2 invariant preserved."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import ModelConfig
        with pytest.raises(ValidationError, match="num_classes must be >= 2"):
            ModelConfig(num_classes=1)

    def test_dropout_range_enforced(self):
        """dropout ∈ [0, 1] invariant preserved."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import ModelConfig
        with pytest.raises(ValidationError, match="dropout must be in"):
            ModelConfig(dropout=-0.1)
        with pytest.raises(ValidationError, match="dropout must be in"):
            ModelConfig(dropout=1.5)

    # --- Enum coercion for YAML string input --------------------------------

    def test_model_type_accepts_yaml_string(self):
        """YAML ``model_type: "tlob"`` → ModelType.TLOB (pre-validator bridge)."""
        from lobtrainer.config.schema import ModelConfig, ModelType
        m = ModelConfig(model_type="tlob")
        assert m.model_type == ModelType.TLOB

    def test_model_type_invalid_string_rejected(self):
        """Unknown model_type string rejected — Enum coercer raises."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import ModelConfig
        with pytest.raises(ValidationError):
            ModelConfig(model_type="bogus_architecture")

    # --- hmhp_horizons Tuple[int,...] immutability pattern ------------------

    def test_hmhp_horizons_default_is_canonical_tuple(self):
        """Legacy contract preserved: default hmhp_horizons is the canonical 5-horizon tuple."""
        from lobtrainer.config.schema import ModelConfig
        m = ModelConfig()
        assert m.hmhp_horizons == (10, 20, 50, 100, 200)
        assert isinstance(m.hmhp_horizons, tuple)

    def test_hmhp_horizons_accepts_yaml_list(self):
        """YAML ``hmhp_horizons: [10, 60, 300]`` coerced to tuple."""
        from lobtrainer.config.schema import ModelConfig
        m = ModelConfig(hmhp_horizons=[10, 60, 300])
        assert m.hmhp_horizons == (10, 60, 300)
        assert isinstance(m.hmhp_horizons, tuple)

    def test_hmhp_horizons_none_becomes_default(self):
        """Legacy contract: explicit None (from YAML ``null``) → canonical default tuple."""
        from lobtrainer.config.schema import ModelConfig
        m = ModelConfig(hmhp_horizons=None)
        assert m.hmhp_horizons == (10, 20, 50, 100, 200)

    def test_hmhp_horizons_is_immutable_post_coercion(self):
        """After list→tuple coercion, the field IS tuple (no .append)."""
        from lobtrainer.config.schema import ModelConfig
        m = ModelConfig(hmhp_horizons=[10, 60])
        with pytest.raises(AttributeError):
            m.hmhp_horizons.append(300)  # type: ignore[union-attr]

    # --- logistic_feature_indices Optional[Tuple[int,...]] ------------------

    def test_logistic_feature_indices_accepts_yaml_list(self):
        """YAML ``logistic_feature_indices: [0, 5, 12]`` coerced to tuple."""
        from lobtrainer.config.schema import ModelConfig
        m = ModelConfig(model_type="logistic", logistic_feature_indices=[0, 5, 12])
        assert m.logistic_feature_indices == (0, 5, 12)
        assert isinstance(m.logistic_feature_indices, tuple)

    def test_logistic_feature_indices_none_stays_none(self):
        """Optional: None stays None (not converted to tuple)."""
        from lobtrainer.config.schema import ModelConfig
        m = ModelConfig(logistic_feature_indices=None)
        assert m.logistic_feature_indices is None

    # --- hmhp_cascade_connections Optional[Tuple[Tuple[int, int], ...]] -----

    def test_hmhp_cascade_connections_accepts_nested_yaml_lists(self):
        """YAML ``hmhp_cascade_connections: [[0, 1], [1, 2]]`` — outer + inner
        list-to-tuple coercion both fire (strict rejects both without coercer).
        """
        from lobtrainer.config.schema import ModelConfig
        m = ModelConfig(hmhp_cascade_connections=[[0, 1], [1, 2]])
        assert m.hmhp_cascade_connections == ((0, 1), (1, 2))
        assert isinstance(m.hmhp_cascade_connections, tuple)
        assert all(isinstance(p, tuple) for p in m.hmhp_cascade_connections)

    def test_hmhp_cascade_connections_none_stays_none(self):
        """Optional: None stays None."""
        from lobtrainer.config.schema import ModelConfig
        m = ModelConfig(hmhp_cascade_connections=None)
        assert m.hmhp_cascade_connections is None

    # --- params auto-derivation via object.__setattr__ ----------------------

    def test_params_auto_derived_when_empty(self):
        """Legacy contract: empty params populated from legacy flat fields
        via ``object.__setattr__`` in _validate_all."""
        from lobtrainer.config.schema import ModelConfig
        # Default (LSTM) → params built from legacy lstm_* fields
        m = ModelConfig(model_type="lstm", hidden_size=128, num_layers=3)
        assert m.params  # non-empty
        assert m.params.get("hidden_size") == 128
        assert m.params.get("num_layers") == 3

    def test_params_preserved_when_explicit(self):
        """If user supplies explicit params, _validate_all does NOT overwrite."""
        from lobtrainer.config.schema import ModelConfig
        explicit = {"hidden_dim": 999, "custom_key": "value"}
        m = ModelConfig(model_type="tlob", params=explicit)
        assert m.params == explicit  # unchanged

    def test_params_dict_contents_remain_mutable(self):
        """Load-bearing: under frozen=True, ``self.params = {}`` raises, but
        ``self.params["key"] = value`` succeeds (Python does not auto-freeze
        mutable-container CONTENTS). This is required for T13 auto-derivation
        at ExperimentConfig._validate_all which updates params in-place."""
        from lobtrainer.config.schema import ModelConfig
        m = ModelConfig(model_type="tlob", input_size=0)
        # Dict-content mutation succeeds (not a field assignment)
        m.params["num_features"] = 40
        assert m.params["num_features"] == 40

    # --- model_copy re-fires validators (A.5.3h correctness) ----------------

    def test_model_copy_revalidates_on_invalid_update(self):
        """Inherited model_copy override re-runs validators on update."""
        from pydantic import ValidationError
        from lobtrainer.config.schema import ModelConfig
        m = ModelConfig(model_type="tlob", input_size=98)
        # Invalid update — num_classes < 2
        with pytest.raises(ValidationError, match="num_classes must be >= 2"):
            m.model_copy(update={"num_classes": 1})

    def test_model_copy_preserves_params_on_valid_update(self):
        """Valid model_copy update preserves the rebuilt params dict.
        Load-bearing for T13 auto-derivation: when
        ``config.model = config.model.model_copy(update={"input_size": 98})``
        the ModelConfig validator re-fires, but params is already non-empty
        (populated by the prior construction), so _build_params_from_legacy
        is skipped — the caller-supplied update's ``params`` (if any) takes
        precedence."""
        from lobtrainer.config.schema import ModelConfig
        m = ModelConfig(model_type="tlob", input_size=0)
        assert m.params  # non-empty post-construction
        m2 = m.model_copy(update={"input_size": 98, "params": {"hidden_dim": 128}})
        assert m2.input_size == 98
        assert m2.params == {"hidden_dim": 128}

    # --- Full YAML-round-trip positive smoke --------------------------------

    def test_defaults_preserved(self):
        """Positive control — defaults match pre-migration @dataclass."""
        from lobtrainer.config.schema import ModelConfig, ModelType
        m = ModelConfig()
        assert m.model_type == ModelType.LSTM
        assert m.input_size == 98
        assert m.num_classes == 3
        assert m.dropout == 0.2
        assert m.hmhp_horizons == (10, 20, 50, 100, 200)


# =============================================================================
# Phase A.5.3f.1 post-audit hardening regression locks (2026-04-24).
#
# Three-agent audit of A.5.3b-f caught 3 ship-blocker mutation sites (cli.py
# epochs field, scripts/train.py, scripts/export_signals.py) + 3 test coverage
# gaps. Mutation sites fixed in the corresponding source files. This class
# locks the 3 coverage gaps:
#
#   1. Pickle round-trip — DataLoader workers spawn subprocesses that pickle
#      configs under num_workers > 0. Silent crash in worker would be hard
#      to debug; explicit test here fails loudly at CI time.
#
#   2. deepcopy round-trip — CVTrainer._build_fold_config uses copy.deepcopy
#      on an ExperimentConfig that embeds 6 SafeBaseModel subclasses. If any
#      deepcopy path is broken by Pydantic internals, per-fold configs would
#      silently share state.
#
#   3. Enum-serialization byte-identity — Pydantic model_dump() output for
#      Enum fields MUST be the .value string (not the Enum instance), so YAML
#      round-trips byte-identical + ledger fingerprints stay stable. A
#      Pydantic v3 migration could silently flip this default; test locks it.
# =============================================================================


class TestPydanticHardeningCoverageGaps:
    """A.5.3f.1 post-audit regression locks — 3 coverage gaps identified
    by 3-agent audit (pickle, deepcopy, Enum serialization).
    """

    def test_pickle_round_trip_all_migrated_classes(self):
        """All 6 SafeBaseModel subclasses MUST pickle cleanly.

        PyTorch DataLoader workers pickle the config when num_workers > 0
        (default=4 on TrainConfig). Pickle failure in a worker would be a
        silent subprocess crash or fallback to main-thread loading —
        hard to diagnose. This test locks pickle-ability at CI time.
        """
        import pickle
        from lobtrainer.config.schema import (
            LabelsConfig, SequenceConfig, NormalizationConfig,
            SourceConfig, TrainConfig, CVConfig,
        )

        # Each class with representative non-default field values
        instances = [
            LabelsConfig(source="forward_prices", task="regression", horizons=[10, 60]),
            SequenceConfig(window_size=100, stride=5),
            NormalizationConfig(strategy="zscore_per_day", exclude_features=[93]),
            SourceConfig(name="mbo", data_dir="/tmp", role="primary", feature_count=98),
            TrainConfig(batch_size=64, learning_rate=1e-4, epochs=50),
            CVConfig(n_splits=5, embargo_days=1),
        ]
        for cfg in instances:
            blob = pickle.dumps(cfg)
            restored = pickle.loads(blob)
            # Field-equality: model_dump round-trips byte-identical
            assert restored.model_dump() == cfg.model_dump(), (
                f"{type(cfg).__name__} pickle round-trip altered fields: "
                f"orig={cfg.model_dump()}, restored={restored.model_dump()}"
            )
            # Type preservation
            assert type(restored) is type(cfg)
            # Frozen still applies on restored instance
            from pydantic import ValidationError
            with pytest.raises(ValidationError):
                # Any field assignment on any class MUST raise
                fields = list(restored.model_dump().keys())
                if fields:
                    setattr(restored, fields[0], "mutated_value")

    def test_deepcopy_round_trip_all_migrated_classes(self):
        """All 6 SafeBaseModel subclasses MUST deepcopy cleanly.

        CVTrainer._build_fold_config at cv_trainer.py:250 uses copy.deepcopy
        on ExperimentConfig — if any nested Pydantic model fails deepcopy,
        folds share state silently.
        """
        import copy
        from lobtrainer.config.schema import (
            LabelsConfig, SequenceConfig, NormalizationConfig,
            SourceConfig, TrainConfig, CVConfig,
        )
        instances = [
            LabelsConfig(horizons=[10, 60, 300]),
            SequenceConfig(window_size=100, stride=10),
            NormalizationConfig(exclude_features=[93]),
            SourceConfig(name="mbo", data_dir="/tmp"),
            TrainConfig(),
            CVConfig(),
        ]
        for cfg in instances:
            restored = copy.deepcopy(cfg)
            # Same field state
            assert restored.model_dump() == cfg.model_dump()
            # Different instance identity
            assert restored is not cfg, (
                f"{type(cfg).__name__} deepcopy returned same object — "
                f"PyDantic internals may be sharing mutable state across "
                f"what appears to be independent copies."
            )

    # =========================================================================
    # Phase A.5.7b (2026-04-25) — Composite-class pickle/deepcopy tests (SB-2)
    # =========================================================================
    #
    # Pre-A.5.7b, the round-trip parametric tests above covered only the 6
    # LEAF classes. The 3 COMPOSITES (DataConfig + ModelConfig +
    # ExperimentConfig) were untested — but CVTrainer._build_fold_config
    # deep-copies ExperimentConfig per-fold (the docstring of the deepcopy
    # test even cites cv_trainer.py:250).
    #
    # Composite-specific concerns the leaf tests miss:
    #
    # 1. DataConfig has PrivateAttr fields (_feature_indices_resolved,
    #    _feature_set_ref_resolved). Pydantic excludes PrivateAttr from
    #    model_dump by default — but does pickle/deepcopy preserve them?
    #    Critical because the trainer's resolver writes to the PrivateAttr
    #    AFTER construction (trainer.py:416-419). If pickle drops it, the
    #    DataLoader worker copies lose resolver state.
    #
    # 2. ModelConfig has params: Dict[str, Any] (mutable container). Naive
    #    deepcopy could share dict references across "independent" copies
    #    if Pydantic's __copy__ is shallow.
    #
    # 3. ExperimentConfig embeds ALL 8 sub-configs as nested BaseModel
    #    fields. Recursive pickle/deepcopy must work end-to-end — a
    #    regression in any nested layer would manifest at the composite
    #    level only.
    # =========================================================================

    def test_pickle_round_trip_composite_configs(self):
        """SB-2: composite classes (DataConfig, ModelConfig, ExperimentConfig)
        with non-default state survive pickle round-trip.

        Specifically exercises:
          - DataConfig with PrivateAttr populated (resolver-cache state)
          - ModelConfig with non-empty params dict (mutable container)
          - ExperimentConfig with non-default sub-configs (nested BaseModel)

        Locks the production pickle path used by PyTorch DataLoader workers
        (num_workers > 0) — silent worker pickle failure was the v3 round-3
        ship-blocker concern that A.5.3f.1 partially addressed for leaves.
        """
        import pickle
        from lobtrainer.config.schema import (
            DataConfig, ExperimentConfig, LabelsConfig, ModelConfig,
        )
        from pydantic import ValidationError

        # 1. DataConfig with PrivateAttr populated (resolver post-state)
        dc = DataConfig(feature_set="x_v1")
        dc._feature_indices_resolved = [0, 5, 12, 84, 85]
        dc._feature_set_ref_resolved = ("x_v1", "a" * 64)

        # 2. ModelConfig with non-empty params (mutable dict + nested types)
        mc = ModelConfig(
            model_type="tlob", input_size=98, num_classes=3,
            params={"hidden_dim": 64, "num_layers": 4, "use_bin": True},
        )

        # 3. ExperimentConfig with non-default sub-configs (nested BaseModel chain)
        ec = ExperimentConfig(
            name="composite_pickle_test",
            data=DataConfig(
                feature_count=98,
                labels=LabelsConfig(horizons=[10, 60, 300], primary_horizon_idx=1),
            ),
            model=ModelConfig(model_type="tlob", input_size=98),
            tags=["composite", "pickle", "test"],
        )

        for cfg in [dc, mc, ec]:
            blob = pickle.dumps(cfg)
            restored = pickle.loads(blob)
            assert type(restored) is type(cfg)
            assert restored.model_dump() == cfg.model_dump(), (
                f"{type(cfg).__name__} pickle round-trip altered fields"
            )
            # Frozen invariant survives pickle
            with pytest.raises(ValidationError):
                fields = list(restored.model_dump().keys())
                if fields:
                    setattr(restored, fields[0], "mutated_value")

        # PrivateAttr-specific assertion: DataConfig pickle MUST preserve
        # the resolver cache (Pydantic v2 PrivateAttr docs guarantee this)
        dc_restored = pickle.loads(pickle.dumps(dc))
        assert dc_restored._feature_indices_resolved == [0, 5, 12, 84, 85], (
            "DataConfig pickle dropped PrivateAttr — Pydantic v2 should "
            "preserve _feature_indices_resolved through pickle. Regression "
            "would silently lose resolver-cache state in DataLoader workers."
        )
        assert dc_restored._feature_set_ref_resolved == ("x_v1", "a" * 64)

    def test_deepcopy_round_trip_composite_configs(self):
        """SB-2: composite classes survive deepcopy with NO shared mutable state.

        ``ModelConfig.params`` is a Dict[str, Any] — a mutable container.
        Naive shallow-copy implementations would share the dict reference
        across what appears to be independent copies; mutating one would
        silently affect the other. Locks correct deepcopy semantics.

        ``ExperimentConfig`` is the production deepcopy target at
        cv_trainer.py:250 (per-fold config build). End-to-end round-trip
        must produce structurally-independent copies.
        """
        import copy
        from lobtrainer.config.schema import (
            DataConfig, ExperimentConfig, LabelsConfig, ModelConfig,
        )

        # ModelConfig with mutable dict — verify NO shared reference post-deepcopy
        mc = ModelConfig(
            model_type="tlob", input_size=98,
            params={"hidden_dim": 64, "extra": ["a", "b"]},
        )
        mc_copy = copy.deepcopy(mc)
        assert mc_copy is not mc
        assert mc_copy.model_dump() == mc.model_dump()
        # The params dict MUST be a different object (deep, not shallow)
        assert mc_copy.params is not mc.params, (
            "ModelConfig.params dict was shared post-deepcopy — Pydantic v2 "
            "deepcopy may have produced a shallow copy. Mutating mc_copy.params "
            "would silently affect mc.params, corrupting CVTrainer's per-fold "
            "isolation (cv_trainer.py:250 deepcopy contract)."
        )
        # The nested list inside params should also be a different object
        assert mc_copy.params["extra"] is not mc.params["extra"]

        # DataConfig with PrivateAttr — verify deepcopy preserves it as fresh state
        dc = DataConfig(feature_set="x_v1")
        dc._feature_indices_resolved = [0, 5, 12]
        dc_copy = copy.deepcopy(dc)
        assert dc_copy is not dc
        assert dc_copy.model_dump() == dc.model_dump()
        # PrivateAttr value preserved (semantic content)
        assert dc_copy._feature_indices_resolved == [0, 5, 12]
        # PrivateAttr list is a different object (deep copy)
        assert dc_copy._feature_indices_resolved is not dc._feature_indices_resolved

        # ExperimentConfig with nested BaseModel chain — recursive deepcopy
        ec = ExperimentConfig(
            name="deepcopy_composite_test",
            data=DataConfig(
                feature_count=98,
                labels=LabelsConfig(horizons=[10, 60, 300]),
            ),
            tags=["a", "b", "c"],
        )
        ec_copy = copy.deepcopy(ec)
        assert ec_copy is not ec
        assert ec_copy.data is not ec.data, "Nested DataConfig shared reference"
        assert ec_copy.data.labels is not ec.data.labels, "Nested LabelsConfig shared reference"
        assert ec_copy.model_dump() == ec.model_dump()
        # Mutable list field
        assert ec_copy.tags is not ec.tags, "tags list shared reference"

    def test_enum_serialization_emits_string_value(self):
        """Enum fields MUST serialize as string .value in model_dump().

        Critical for:
        - YAML round-trip byte-identity (YAML writers expect strings,
          not Enum instances that serialize as `!!python/object:...`)
        - Ledger fingerprint stability — CompatibilityContract includes
          normalization_strategy as a string; if model_dump() ever emits
          the Enum instance instead, sha256 hash rotates and cross-
          experiment comparability breaks.

        Ship-blocker if broken. Locks Pydantic v2's default StrEnum
        serialization behavior (these Enums inherit str, so .value
        equality is tight).
        """
        from lobtrainer.config.schema import (
            NormalizationConfig, NormalizationStrategy,
            TrainConfig, TaskType, LossType,
        )

        # NormalizationConfig.strategy
        nc = NormalizationConfig(strategy=NormalizationStrategy.ZSCORE_PER_DAY)
        d = nc.model_dump()
        assert d["strategy"] == "zscore_per_day", (
            f"NormalizationStrategy did not serialize as string value; "
            f"got type={type(d['strategy']).__name__}, value={d['strategy']!r}"
        )
        # Also verify mode="python" returns string (not .value attr access)
        assert isinstance(d["strategy"], str)

        # TrainConfig.task_type + loss_type (2 Enum fields)
        tc = TrainConfig(task_type=TaskType.REGRESSION, loss_type=LossType.HUBER)
        td = tc.model_dump()
        assert td["task_type"] == "regression", (
            f"TaskType did not serialize as string; got {td['task_type']!r}"
        )
        assert td["loss_type"] == "huber", (
            f"LossType did not serialize as string; got {td['loss_type']!r}"
        )
        assert isinstance(td["task_type"], str)
        assert isinstance(td["loss_type"], str)

    def test_yaml_load_horizons_is_tuple_not_list(self):
        """After YAML load + dacite → LabelsConfig.model_validate path,
        horizons MUST be a tuple (Pydantic @field_validator(mode="before")
        coerces list→tuple at parse time).

        Without this lock, a regression that silently reverts
        horizons to List[int] (dropping the A.5.3a.1 immutability guarantee)
        would pass all other tests but re-introduce the .append bypass.
        """
        import tempfile
        from pathlib import Path as _Path
        from lobtrainer.config.schema import ExperimentConfig

        yaml_text = """
name: tuple_coercion_test
data:
  data_dir: "/tmp"
  feature_count: 98
  labels:
    source: forward_prices
    task: regression
    horizons: [10, 60, 300]
    primary_horizon_idx: 0
model:
  input_size: 98
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            yaml_path = f.name
        try:
            cfg = ExperimentConfig.from_yaml(yaml_path)
            horizons = cfg.data.labels.horizons
            assert isinstance(horizons, tuple), (
                f"YAML-loaded horizons is {type(horizons).__name__}, expected tuple. "
                f"The @field_validator(mode='before') list→tuple coercer on "
                f"LabelsConfig.horizons may have regressed — this breaks the "
                f"A.5.3a.1 immutability guarantee."
            )
            assert horizons == (10, 60, 300)
        finally:
            _Path(yaml_path).unlink()

    def test_apply_overrides_epochs_field_post_a53f1(self):
        """A.5.3f.1 regression lock for the cli.py --epochs bug discovered by
        the 3-agent audit. The original A.5.3e cli.py fix missed this field
        (it was in a separate if-block before the _train_overrides dict).
        Every ``lobtrainer train --epochs 10`` invocation would have raised.
        """
        import argparse
        from lobtrainer.cli import apply_overrides
        from lobtrainer.config import ExperimentConfig

        cfg = ExperimentConfig(name="epochs_override_test")
        args = argparse.Namespace(
            epochs=42,
            batch_size=None,
            learning_rate=None,
            seed=None,
            data_dir=None,
            output_dir=None,
        )
        cfg2 = apply_overrides(cfg, args)
        assert cfg2.train.epochs == 42, (
            f"cli.py apply_overrides failed to apply --epochs override. "
            f"Got train.epochs={cfg2.train.epochs}, expected 42."
        )
        # Other fields unchanged
        assert cfg2.train.batch_size == cfg.train.batch_size


# =============================================================================
# Phase A.5.7a (2026-04-25) — SafeBaseModel canonical-form SSoT regression locks
# =============================================================================
#
# Three regression locks for the A.5.7a hardening cycle:
#
#   1. SB-1 fix: ``__hash__`` was order-sensitive on dict-typed fields,
#      while ``__eq__`` (via ``__dict__ ==``) was order-insensitive — silently
#      violating Python's ``a == b ⇒ hash(a) == hash(b)`` invariant.
#      Reachable via ``ModelConfig.params: Dict[str, Any]``. Locked by
#      tests below.
#
#   2. HP-2: ``__pydantic_init_subclass__`` auto-registration. Replaces the
#      hand-maintained ``_PYDANTIC_CONFIG_CLASSES`` list. Registry-completeness
#      test asserts exactly 9 production classes registered.
#
#   3. HP-3: ClassVar discipline. Parametric test walks the auto-registry
#      and asserts no class leaks ``_``-prefixed keys into ``model_dump()``
#      — catches a contributor forgetting the ``ClassVar[...]`` annotation
#      on a class-level constant (which would silently flow into
#      cross-module fingerprints, rotating every byte-identity hash).
# =============================================================================


def _list_safe_base_model_subclasses():
    """Helper for parametric tests over ``SafeBaseModel._registry``.

    Imports ``schema`` to ensure all 9 production config classes have been
    defined (their ``__pydantic_init_subclass__`` hooks fire at definition
    time, populating the registry). Returns a snapshot list so the
    parametric expansion is stable even if test fixtures temporarily
    register additional subclasses (which the hook explicitly excludes
    via leading-underscore name filter).
    """
    from lobtrainer.config import base, schema  # noqa: F401 — populates registry
    return list(base.SafeBaseModel._registry)


class TestSafeBaseModelCanonicalForm:
    """Phase A.5.7a SB-1 regression locks: ``_canonical_form()`` SSoT.

    Pre-A.5.7a, ``__hash__`` used ``hash(tuple(sorted((k, repr(v)) for ...)))``
    which was order-sensitive on dict-typed fields like
    ``ModelConfig.params``. ``__eq__`` (``self.__dict__ == other.__dict__``)
    was order-insensitive. The mismatch silently violated Python's
    ``a == b ⇒ hash(a) == hash(b)`` invariant — corrupting any set/dict
    deduplication using these classes as keys.

    Post-A.5.7a both ``__eq__`` and ``__hash__`` consume the SAME
    ``_canonical_form()`` (sorted-keys JSON of ``model_dump(mode="json")``).
    Aligned by SSoT construction; invariant holds structurally.
    """

    def test_dict_field_insertion_order_invariance_for_hash(self):
        """SB-1: dict-typed field with same logical content but different
        insertion order MUST produce the same hash.

        Reachable via ``ModelConfig.params: Dict[str, Any]``. The smoking
        gun: pre-A.5.7a, ``repr({"a":1,"b":2}) != repr({"b":2,"a":1})``
        but the dicts compared equal. Hash invariant violated.
        """
        from lobtrainer.config.schema import ModelConfig

        mc1 = ModelConfig(model_type="tlob", input_size=98, params={"a": 1, "b": 2})
        mc2 = ModelConfig(model_type="tlob", input_size=98, params={"b": 2, "a": 1})

        assert mc1 == mc2, (
            "Dict order-insensitive equality should hold (params dicts "
            "have same content, different insertion order)."
        )
        assert hash(mc1) == hash(mc2), (
            "Hash MUST be order-insensitive when __eq__ is order-insensitive. "
            "Pre-A.5.7a: __hash__ used repr() over sorted-key tuple, but "
            "repr({'a':1,'b':2}) != repr({'b':2,'a':1}). Python's invariant "
            "a == b ⇒ hash(a) == hash(b) was silently violated. Post-A.5.7a: "
            "_canonical_form() uses json.dumps(sort_keys=True) on the "
            "model_dump output — order-independent by construction."
        )

    def test_set_membership_works_under_order_insensitive_keys(self):
        """SB-1 invariant in production-realistic context: a set built
        from ``ModelConfig`` instances with semantically-equal params
        dicts must have exactly ONE member.

        Pre-A.5.7a: `len({mc1, mc2}) == 2` (broken hash) even though
        `mc1 == mc2`. Post-A.5.7a: `len({mc1, mc2}) == 1` (correct).
        """
        from lobtrainer.config.schema import ModelConfig

        mc1 = ModelConfig(model_type="tlob", input_size=98, params={"a": 1, "b": 2})
        mc2 = ModelConfig(model_type="tlob", input_size=98, params={"b": 2, "a": 1})
        s = {mc1, mc2}
        assert len(s) == 1, (
            f"Set membership broken — pre-A.5.7a __hash__ ordering bug. "
            f"Expected len==1, got {len(s)}."
        )

    def test_eq_excludes_private_attr_via_canonical_form(self):
        """Phase 4 R3 invariant preserved post-refactor: PrivateAttr fields
        do NOT affect equality. ``model_dump`` excludes PrivateAttr by
        default; ``_canonical_form`` is built on ``model_dump`` output ⇒
        PrivateAttr structurally absent from canonical form."""
        from lobtrainer.config.schema import DataConfig

        dc1 = DataConfig(feature_set="x_v1")
        dc2 = DataConfig(feature_set="x_v1")
        dc2._feature_indices_resolved = [0, 5, 12]
        dc2._feature_set_ref_resolved = ("x_v1", "a" * 64)
        assert dc1 == dc2, (
            "DataConfig equality must exclude PrivateAttr fields "
            "(Phase 4 R3 invariant — resolver caches are not part of "
            "semantic identity)."
        )
        assert hash(dc1) == hash(dc2)

    def test_canonical_form_is_deterministic_string(self):
        """``_canonical_form()`` returns a stable JSON string. Same instance
        called twice returns identical bytes (no nondeterministic key
        ordering, no timestamp embedding, no UUID drift)."""
        from lobtrainer.config.schema import LabelsConfig

        lc = LabelsConfig(horizons=[10, 60, 300], primary_horizon_idx=1)
        cf1 = lc._canonical_form()
        cf2 = lc._canonical_form()
        assert cf1 == cf2, "Canonical form must be deterministic"
        # Sanity: it's actual JSON
        import json
        parsed = json.loads(cf1)
        assert parsed["primary_horizon_idx"] == 1

    def test_canonical_form_excludes_private_attr_field(self):
        """Direct verification: PrivateAttr value does NOT appear in the
        canonical form string."""
        from lobtrainer.config.schema import DataConfig

        dc = DataConfig(feature_set="x_v1")
        dc._feature_indices_resolved = [0, 5, 12, 84, 85]
        cf = dc._canonical_form()
        assert "_feature_indices_resolved" not in cf
        assert "84, 85" not in cf  # the value bytes don't leak either


class TestSafeBaseModelRegistry:
    """Phase A.5.7a HP-2 + HP-3 regression locks: ``__pydantic_init_subclass__``
    auto-registration replaces the hand-maintained
    ``_PYDANTIC_CONFIG_CLASSES`` list.
    """

    def test_registry_contains_all_9_production_classes(self):
        """HP-2: registry auto-populates with exactly 9 classes (the
        production config hierarchy). If a contributor adds a 10th class
        and this test still expects 9, the test fails — forcing the
        new class to either (a) be intentionally added to the count or
        (b) be excluded via leading-underscore name (test fixture)."""
        from lobtrainer.config.base import SafeBaseModel
        from lobtrainer.config.schema import (
            LabelsConfig, SequenceConfig, NormalizationConfig, SourceConfig,
            TrainConfig, CVConfig, DataConfig, ModelConfig, ExperimentConfig,
        )
        # Filter to current concrete classes (test fixtures with leading-
        # underscore names should already be excluded by the hook itself).
        production_classes = {
            cls for cls in SafeBaseModel._registry
            if not cls.__name__.startswith("_")
        }
        expected = {
            LabelsConfig, SequenceConfig, NormalizationConfig, SourceConfig,
            TrainConfig, CVConfig, DataConfig, ModelConfig, ExperimentConfig,
        }
        assert production_classes == expected, (
            f"Registry mismatch.\n"
            f"  Missing from registry: {expected - production_classes}\n"
            f"  Unexpected in registry: {production_classes - expected}\n"
            f"If you added a new SafeBaseModel subclass, update this "
            f"test's `expected` set + the count below."
        )
        assert len(production_classes) == 9

    def test_underscore_prefixed_test_fixtures_excluded(self):
        """HP-2: test fixtures defined inside test methods (with
        leading-underscore names like ``_FixtureConfig``) are EXCLUDED
        from the registry. Without this filter, every test method that
        creates a temporary SafeBaseModel subclass would pollute the
        production-class count."""
        from lobtrainer.config.base import SafeBaseModel

        # Force-create a fixture-style subclass; it should NOT appear in the registry.
        class _TempUnregisteredConfig(SafeBaseModel):
            x: int = 0

        registered_names = {cls.__name__ for cls in SafeBaseModel._registry}
        assert "_TempUnregisteredConfig" not in registered_names, (
            "Underscore-prefixed test fixture leaked into the registry. "
            "Check `__pydantic_init_subclass__` filter in base.py."
        )

    def test_pydantic_config_classes_shim_matches_registry(self):
        """A.5.7a back-compat: ``schema._PYDANTIC_CONFIG_CLASSES`` is a
        re-export shim of ``SafeBaseModel._registry``. Any external code
        that imported the hand-list must continue to work — this lock
        catches a regression where the shim drifts from the registry."""
        from lobtrainer.config.base import SafeBaseModel
        from lobtrainer.config.schema import _PYDANTIC_CONFIG_CLASSES

        # Filter both to production classes for stability under test-fixture pollution
        registry_prod = [
            cls for cls in SafeBaseModel._registry
            if not cls.__name__.startswith("_")
        ]
        shim_prod = [
            cls for cls in _PYDANTIC_CONFIG_CLASSES
            if not cls.__name__.startswith("_")
        ]
        assert set(shim_prod) == set(registry_prod), (
            "Re-export shim drifted from canonical SafeBaseModel._registry."
        )

    @pytest.mark.parametrize(
        "cls",
        _list_safe_base_model_subclasses(),
        ids=lambda c: c.__name__,
    )
    def test_no_underscore_prefixed_keys_in_model_dump(self, cls):
        """HP-3: ClassVar discipline. No production class should emit
        ``_``-prefixed keys in its ``model_dump()`` output.

        Why this matters: class-level constants (e.g., LabelsConfig's
        ``_VALID_SOURCES``) MUST be annotated as ``ClassVar[...]`` to be
        recognized by Pydantic as constants rather than fields. Without
        the annotation, Pydantic v2 treats them as mutable fields ⇒
        they leak into ``model_dump()`` ⇒ they appear in serialized
        configs ⇒ they corrupt cross-module fingerprint byte-identity
        (every fingerprint hash rotates).

        Parametric over ALL 9 SafeBaseModel subclasses. A new class added
        to the registry (via auto-registration) is automatically covered.
        """
        instance = cls()  # all 9 classes have all-defaults working construction
        dumped = instance.model_dump()
        leaked = [k for k in dumped if isinstance(k, str) and k.startswith("_")]
        assert not leaked, (
            f"{cls.__name__}.model_dump() leaked private/class-constant key(s): {leaked}.\n"
            f"  - If it's a class-level constant: annotate as `ClassVar[...]`\n"
            f"  - If it's runtime-only state: use `PrivateAttr()`\n"
            f"Phase A.5.7a HP-3 lock: this regression silently rotates "
            f"every cross-module fingerprint hash."
        )
