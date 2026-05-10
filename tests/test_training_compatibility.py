"""Phase X.1 v2 (2026-05-04) tests for the new shared compatibility module.

Verifies the X.1.A surface:
  * ``build_compatibility_contract`` (moved from exporter.py — preserves behavior)
  * ``derive_data_source`` (moved from exporter.py — preserves behavior)
  * ``compute_model_config_hash`` (NEW — sha256 over filtered model.params)
  * ``_LOSS_TUNING_KEYS`` (NEW — denylist of training-axis keys)
  * 3 exception classes (NEW)

Hash sensitivity matrix: architecture keys (hidden_dim, num_layers,
hmhp_pool_mode, num_classes, etc.) MUST trip the hash; loss-tuning keys
(gmadl_a, gmadl_b, regression_loss_*, loss_weights, etc.) and auto-derived
data axes (num_features, sequence_length, input_size, task_type) MUST NOT.

Per Agent 4 sanity check Q7 (post v1→v2 redesign): ``num_classes`` is
INTENTIONALLY NOT in the denylist because it determines output head
dimension (architectural). ``task_type`` IS in the denylist because
CompatibilityContract already captures the same axis via
``compute_label_strategy_hash(LabelsConfig)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lobtrainer.training.compatibility import (
    CheckpointConfigMismatchError,
    CheckpointConfigMismatchWarning,
    CheckpointMissingFingerprintWarning,
    _LOSS_TUNING_KEYS,
    build_compatibility_contract,
    compute_model_config_hash,
    derive_data_source,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_test_config(
    tmp_path: Path,
    *,
    model_type: str = "tlob",
    params: dict | None = None,
    feature_count: int = 98,
    input_size: int | None = None,
    window_size: int = 20,
    normalization_strategy: str = "none",
):
    """Build a minimal ExperimentConfig for hashing tests.

    Mirrors the pattern from test_create_trainer_dispatch.py — doesn't
    require real data on disk because the hash functions only read config
    fields, not data.

    Phase 1 N7 (2026-05-06): added ``normalization_strategy`` kwarg so N7
    tests can construct configs with GLOBAL_ZSCORE / HYBRID strategies that
    derive a real on-disk stats path. Default ``"none"`` preserves
    pre-N7 fixture behavior.
    """
    from lobtrainer.config.schema import (
        ExperimentConfig,
        DataConfig,
        ModelConfig,
        TrainConfig,
        LabelsConfig,
        SequenceConfig,
        NormalizationConfig,
        ModelType,
        TaskType,
        LossType,
    )

    if params is None:
        params = {}
    if input_size is None:
        input_size = feature_count

    data = DataConfig(
        data_dir=str(tmp_path / "fake_data"),
        feature_count=feature_count,
        sequence=SequenceConfig(window_size=window_size, stride=1),
        normalization=NormalizationConfig(strategy=normalization_strategy),
        labels=LabelsConfig(
            primary_horizon_idx=0,
            horizons=[10, 60, 300],
            source="forward_prices",
            task="regression",
        ),
    )
    model = ModelConfig(
        model_type=getattr(ModelType, model_type.upper()),
        input_size=input_size,
        params=params,
    )
    train = TrainConfig(task_type=TaskType.REGRESSION, loss_type=LossType.HUBER)
    return ExperimentConfig(
        name=f"test_{model_type}",
        data=data,
        model=model,
        train=train,
        output_dir=str(tmp_path / "output"),
    )


# ---------------------------------------------------------------------------
# derive_data_source
# ---------------------------------------------------------------------------


class TestDeriveDataSource:
    def test_basic_prefix_returns_off_exchange(self):
        assert derive_data_source(Path("/tmp/data/exports/basic_nvda_60s")) == "off_exchange"

    def test_basic_prefix_relative_path(self):
        assert derive_data_source("basic_arcx_30s") == "off_exchange"

    def test_default_mbo_lob(self):
        assert derive_data_source("/tmp/data/exports/e5_timebased_60s_v3p0") == "mbo_lob"

    def test_default_for_path_without_basic_prefix(self):
        assert derive_data_source("nvda_xnas_128feat_regression_fwd_prices_v3p0") == "mbo_lob"


# ---------------------------------------------------------------------------
# build_compatibility_contract — moved from exporter.py
# ---------------------------------------------------------------------------


class TestBuildCompatibilityContract:
    def test_returns_contract_with_expected_fields(self, tmp_path):
        config = _build_test_config(tmp_path, model_type="tlob")
        compat = build_compatibility_contract(config)
        assert compat is not None, "hft_contracts should be available in test env"
        # Verify key fields
        assert compat.feature_count == 98
        assert compat.window_size == 20
        assert compat.horizons == (10, 60, 300)
        assert compat.primary_horizon_idx == 0
        # NOTE: build_compatibility_contract preserves pre-Phase-X.1 behavior of using
        # str(Enum) which on Python 3.11+ returns the Enum repr form
        # ('NormalizationStrategy.NONE') instead of the value ('none'). This is a
        # pre-existing latent issue documented for a future cycle (Phase X.3 silent-default
        # sweep). Phase X.1 v2 deliberately does NOT change this behavior — it would
        # alter every existing CompatibilityContract fingerprint and break consumers.
        assert "NONE" in compat.normalization_strategy.upper() or compat.normalization_strategy == "none"

    def test_default_feature_layout_is_default_string(self, tmp_path):
        config = _build_test_config(tmp_path)
        compat = build_compatibility_contract(config)
        assert compat.feature_layout == "default"

    def test_feature_set_ref_propagates_content_hash(self, tmp_path):
        config = _build_test_config(tmp_path)
        fs_ref = {"name": "test_set", "content_hash": "a" * 64}
        compat = build_compatibility_contract(config, feature_set_ref=fs_ref)
        assert compat.feature_layout == "a" * 64

    def test_calibration_method_propagates(self, tmp_path):
        config = _build_test_config(tmp_path)
        compat = build_compatibility_contract(config, calibration_method="variance_match")
        assert compat.calibration_method == "variance_match"

    def test_fingerprint_returns_64_hex(self, tmp_path):
        config = _build_test_config(tmp_path)
        compat = build_compatibility_contract(config)
        fp = compat.fingerprint()
        assert isinstance(fp, str)
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)


# ---------------------------------------------------------------------------
# compute_model_config_hash — NEW Phase X.1 v2
# ---------------------------------------------------------------------------


class TestComputeModelConfigHash:
    def test_returns_64_hex_lowercase(self, tmp_path):
        config = _build_test_config(tmp_path, model_type="tlob")
        h = compute_model_config_hash(config.model)
        assert isinstance(h, str)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_stable_across_calls(self, tmp_path):
        """Same config → same hash (deterministic)."""
        config = _build_test_config(tmp_path, model_type="tlob")
        h1 = compute_model_config_hash(config.model)
        h2 = compute_model_config_hash(config.model)
        assert h1 == h2

    def test_different_model_types_produce_different_hashes(self, tmp_path):
        c_tlob = _build_test_config(tmp_path / "a", model_type="tlob")
        c_lstm = _build_test_config(tmp_path / "b", model_type="lstm")
        assert compute_model_config_hash(c_tlob.model) != compute_model_config_hash(c_lstm.model)

    def test_dict_insertion_order_does_not_affect_hash(self, tmp_path):
        """params={'a': 1, 'b': 2} should hash the same as {'b': 2, 'a': 1}.

        canonical_json_blob's sort_keys=True guarantees this. Phase X.1 v2 explicitly
        relies on this property for cross-checkpoint stability.
        """
        c1 = _build_test_config(tmp_path / "a", params={"a": 1, "b": 2})
        c2 = _build_test_config(tmp_path / "b", params={"b": 2, "a": 1})
        assert compute_model_config_hash(c1.model) == compute_model_config_hash(c2.model)

    # Architecture mutation sensitivity — these MUST change the hash
    @pytest.mark.parametrize("arch_key,old_value,new_value", [
        ("hidden_dim", 192, 256),
        ("num_layers", 4, 6),
        ("num_heads", 1, 2),
        ("dropout", 0.1, 0.3),
        ("hmhp_pool_mode", "last", "mean"),
        ("num_classes", 3, 5),  # NOT in denylist — output head shape architectural
    ])
    def test_arch_mutation_changes_hash(self, tmp_path, arch_key, old_value, new_value):
        """Mutating any architectural key MUST invalidate model_config_hash."""
        c1 = _build_test_config(tmp_path / "a", params={arch_key: old_value})
        c2 = _build_test_config(tmp_path / "b", params={arch_key: new_value})
        h1 = compute_model_config_hash(c1.model)
        h2 = compute_model_config_hash(c2.model)
        assert h1 != h2, (
            f"Architecture key '{arch_key}' (old={old_value}, new={new_value}) "
            f"did not change model_config_hash — must be in arch axis, NOT denylist."
        )

    # Loss-tuning insensitivity — these MUST NOT change the hash
    @pytest.mark.parametrize("loss_key,old_value,new_value", [
        ("gmadl_a", 1.0, 10.0),
        ("gmadl_b", 1.0, 1.5),
        ("regression_loss_type", "huber", "mse"),
        ("regression_loss_delta", 5.0, 12.6),
        ("loss_type", "huber", "mse"),  # hmhp_regression-specific variant
        ("loss_weights", {"H10": 0.5}, {"H10": 0.7}),
        ("huber_delta", 5.0, 12.6),
    ])
    def test_loss_tuning_does_not_change_hash(self, tmp_path, loss_key, old_value, new_value):
        """Mutating loss-tuning keys MUST NOT trip model_config_hash."""
        c1 = _build_test_config(tmp_path / "a", params={loss_key: old_value})
        c2 = _build_test_config(tmp_path / "b", params={loss_key: new_value})
        h1 = compute_model_config_hash(c1.model)
        h2 = compute_model_config_hash(c2.model)
        assert h1 == h2, (
            f"Loss-tuning key '{loss_key}' (old={old_value}, new={new_value}) "
            f"changed model_config_hash — should be in _LOSS_TUNING_KEYS denylist."
        )

    # Auto-derived data axes — these MUST NOT change the hash
    # (CompatibilityContract.feature_count / window_size / horizons captures them)
    @pytest.mark.parametrize("auto_key,old_value,new_value", [
        ("num_features", 98, 128),
        ("sequence_length", 20, 100),
        ("input_size", 98, 116),
        ("task_type", "classification", "regression"),
    ])
    def test_auto_derived_data_axes_do_not_change_hash(
        self, tmp_path, auto_key, old_value, new_value,
    ):
        """Auto-derived data-axis keys MUST be in denylist (already in CompatibilityContract)."""
        c1 = _build_test_config(tmp_path / "a", params={auto_key: old_value})
        c2 = _build_test_config(tmp_path / "b", params={auto_key: new_value})
        h1 = compute_model_config_hash(c1.model)
        h2 = compute_model_config_hash(c2.model)
        assert h1 == h2, (
            f"Auto-derived key '{auto_key}' changed model_config_hash — "
            f"must be in _LOSS_TUNING_KEYS denylist (CompatibilityContract covers it)."
        )


class TestLossTuningKeys:
    """Verify the denylist contents — locks the architectural-vs-loss boundary."""

    def test_denylist_is_frozenset(self):
        assert isinstance(_LOSS_TUNING_KEYS, frozenset)

    def test_denylist_contains_all_documented_keys(self):
        """Locks the curated list per §I.4 X.1.A."""
        expected = {
            "gmadl_a", "gmadl_b",
            "regression_loss_type", "regression_loss_delta",
            "loss_type",
            "loss_weights",
            "huber_delta", "pinball_quantiles",
            "task_type",
            "num_features", "sequence_length", "input_size",
        }
        assert _LOSS_TUNING_KEYS == expected, (
            f"Denylist drifted from documented set. "
            f"Missing: {expected - _LOSS_TUNING_KEYS}; "
            f"Extra: {_LOSS_TUNING_KEYS - expected}"
        )

    def test_num_classes_NOT_in_denylist(self):
        """num_classes IS architectural (output head dim) — MUST trip hash."""
        assert "num_classes" not in _LOSS_TUNING_KEYS

    def test_hidden_dim_NOT_in_denylist(self):
        """hidden_dim IS architectural — MUST trip hash."""
        assert "hidden_dim" not in _LOSS_TUNING_KEYS

    def test_hmhp_pool_mode_NOT_in_denylist(self):
        """hmhp_pool_mode IS architectural (Phase S F-9) — MUST trip hash."""
        assert "hmhp_pool_mode" not in _LOSS_TUNING_KEYS


# ---------------------------------------------------------------------------
# Exception classes — verify hierarchy + distinctness
# ---------------------------------------------------------------------------


class TestCheckpointDictEndToEnd:
    """Phase X.1 v2 post-sanity-check addition (per Agent 4 recommended test):
    exercise Trainer._build_checkpoint_dict end-to-end and verify torch.save +
    torch.load round-trip preserves the 3 NEW Phase X.1 v2 keys.

    The 40 unit tests above never invoked the actual checkpoint write path —
    a P0 (compat.to_dict() vs to_canonical_dict() AttributeError) was caught
    only by adversarial review. This test closes that coverage gap.
    """

    def test_trainer_build_checkpoint_dict_embeds_3_keys(self, tmp_path):
        """Verify Trainer._build_checkpoint_dict produces dict with the 3 NEW keys."""
        import torch.nn as nn
        from lobtrainer.training.trainer import Trainer

        config = _build_test_config(tmp_path, model_type="tlob", params={"tlob_hidden_dim": 64})
        # Build trainer with minimal model (avoid full setup() which needs real data)
        trainer = Trainer.__new__(Trainer)  # bypass __init__
        trainer.config = config
        trainer.model = nn.Linear(2, 2)
        trainer._optimizer = None
        trainer._scheduler = None
        # Mimic state attribute
        from lobtrainer.training.trainer import TrainingState
        trainer.state = TrainingState()

        ckpt = trainer._build_checkpoint_dict()
        # 3 NEW Phase X.1 v2 keys MUST be present
        assert 'compatibility' in ckpt, "checkpoint missing 'compatibility' key"
        assert 'compatibility_fingerprint' in ckpt, "checkpoint missing 'compatibility_fingerprint' key"
        assert 'model_config_hash' in ckpt, "checkpoint missing 'model_config_hash' key"
        # Values must be valid
        assert ckpt['compatibility'] is not None
        assert isinstance(ckpt['compatibility'], dict)
        assert isinstance(ckpt['compatibility_fingerprint'], str)
        assert len(ckpt['compatibility_fingerprint']) == 64
        assert isinstance(ckpt['model_config_hash'], str)
        assert len(ckpt['model_config_hash']) == 64

    def test_trainer_save_checkpoint_roundtrip_torch_save_load(self, tmp_path):
        """Sanity: torch.save the checkpoint dict + torch.load preserves keys."""
        import torch
        import torch.nn as nn
        from lobtrainer.training.trainer import Trainer, TrainingState

        config = _build_test_config(tmp_path, model_type="tlob")
        trainer = Trainer.__new__(Trainer)
        trainer.config = config
        trainer.model = nn.Linear(2, 2)
        trainer._optimizer = None
        trainer._scheduler = None
        trainer.state = TrainingState()

        ckpt_dict = trainer._build_checkpoint_dict()
        ckpt_path = tmp_path / "test_checkpoint.pt"
        torch.save(ckpt_dict, ckpt_path)

        # Reload + verify all 3 NEW keys preserved
        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert loaded['compatibility_fingerprint'] == ckpt_dict['compatibility_fingerprint']
        assert loaded['model_config_hash'] == ckpt_dict['model_config_hash']
        assert loaded['compatibility'] is not None


class TestRngStateInvariance:
    """Phase DESIGN-1 Phase E Site 1 (2026-05-10) — lock the structural
    invariant that ``rng_state`` and ``callback_state`` CANNOT enter
    ``compatibility_fingerprint``.

    Background: Phase A.2 added ``rng_state`` to checkpoint dict at
    ``trainer.py:1353`` and Phase G-1 added ``callback_state`` at
    ``trainer.py:1365``. The fingerprint is computed at ``trainer.py:1338``
    via ``compat.fingerprint()`` — BEFORE either is added. Computation
    order makes leakage structurally impossible TODAY.

    But: a future refactor that moves fingerprint computation past line 1353
    (e.g., adding "extended fingerprint" feature, signed checkpoints,
    content-addressed storage) would silently leak ``rng_state`` /
    ``callback_state`` into the fingerprint canonical-form. R-series ledger
    entries become uncomparable. Phase-3-§3.3b-class conflation regression.

    This class catches that future refactor at TYPE level by asserting that
    ``CompatibilityContract`` has NO ``rng_state`` / ``callback_state``
    fields — so ``compat.fingerprint()`` cannot include them by construction.

    Per Wave 2.2 critique 2026-05-10: ship 1 test (compat_fingerprint side).
    DROP redundant ``model_config_hash`` test — ``compute_model_config_hash``
    reads ``model.config.params`` filtered by ``_LOSS_TUNING_KEYS`` denylist,
    NEVER touches ``rng_state``; no plausible leakage path exists.
    """

    def test_compatibility_fingerprint_independent_of_rng_state(self):
        """Structural lock: ``rng_state`` + ``callback_state`` NOT in
        ``CompatibilityContract`` dataclass fields.

        Three-way verification (single test for cohesion):
          1. STRUCTURAL: ``CompatibilityContract`` has no ``rng_state`` /
             ``callback_state`` fields → ``compat.fingerprint()`` cannot
             include them by construction.
          2. IDEMPOTENCY: ``compat.fingerprint()`` is deterministic
             (locks the canonical-form ordering invariant).
          3. SHAPE: fingerprint is 64-hex SHA-256.

        Companion runtime-path lock at G-4
        ``test_two_runs_same_seed_produce_tensor_equal_checkpoint`` —
        if structural lock here passes but runtime side-check there fails,
        a NON-rng-state non-determinism source has entered fingerprint inputs.
        """
        import dataclasses
        from hft_contracts.compatibility import CompatibilityContract

        # 1. STRUCTURAL — verify rng_state + callback_state NOT in dataclass fields
        field_names = {
            f.name for f in dataclasses.fields(CompatibilityContract)
        }
        assert "rng_state" not in field_names, (
            "rng_state leaked into CompatibilityContract dataclass fields — "
            "compat.fingerprint() would now include it, silently breaking "
            "R-series ledger comparability across checkpoints with different "
            "rng_state. Phase-3-§3.3b-class conflation regression. "
            f"Current fields: {sorted(field_names)}"
        )
        assert "callback_state" not in field_names, (
            "callback_state leaked into CompatibilityContract dataclass fields — "
            "G-1 callback state must remain OBSERVATION not TREATMENT in the "
            "fingerprint canonical-form. "
            f"Current fields: {sorted(field_names)}"
        )

        # 2. IDEMPOTENCY — fingerprint is deterministic
        compat = CompatibilityContract(
            contract_version="3.0",
            schema_version="3.0",
            feature_count=98,
            window_size=20,
            feature_layout=(
                "ask_prices_10_ask_sizes_10_bid_prices_10_bid_sizes_10"
            ),
            data_source="MBO",
            label_strategy_hash="a" * 64,
            calibration_method=None,
            primary_horizon_idx=0,
            horizons=(10, 60, 300),
            normalization_strategy="hybrid",
        )
        fp_first = compat.fingerprint()
        fp_second = compat.fingerprint()
        assert fp_first == fp_second, (
            "CompatibilityContract.fingerprint() is non-deterministic across "
            "calls — non-canonical input ordering or hidden state."
        )

        # 3. SHAPE — fingerprint is 64-hex SHA-256
        import re
        assert re.match(r"^[0-9a-f]{64}$", fp_first), (
            f"fingerprint not 64-char lowercase hex: {fp_first!r}"
        )

        # 4. BEHAVIORAL bonus — different documented field → different fingerprint
        # (locks that the fingerprint isn't a constant)
        compat_alt = CompatibilityContract(
            contract_version="3.0",
            schema_version="3.0",
            feature_count=99,  # ← differs
            window_size=20,
            feature_layout=(
                "ask_prices_10_ask_sizes_10_bid_prices_10_bid_sizes_10"
            ),
            data_source="MBO",
            label_strategy_hash="a" * 64,
            calibration_method=None,
            primary_horizon_idx=0,
            horizons=(10, 60, 300),
            normalization_strategy="hybrid",
        )
        assert compat_alt.fingerprint() != fp_first, (
            "Different feature_count produced identical fingerprint — "
            "CompatibilityContract.fingerprint() is not actually reading "
            "field values."
        )


class TestSignatureLock:
    """Phase X.1 v2 post-validation (Agent 3 Q14e + Agent 1 Q1) — lock the
    public API surface so future drift is caught at test time.

    Pre-X.1 v1 plan changed `save_checkpoint` to `Optional[Path] = None) -> Path`,
    breaking back-compat. This test prevents recurrence.
    """

    def test_save_checkpoint_signature_locked(self):
        """Trainer.save_checkpoint must accept (self, path: Union[str, Path]) -> None."""
        import inspect
        from lobtrainer.training.trainer import Trainer
        sig = inspect.signature(Trainer.save_checkpoint)
        params = list(sig.parameters.keys())
        assert params == ["self", "path"], (
            f"Trainer.save_checkpoint signature drifted: got {params}. "
            f"Pre-X.1 v2 callers (ModelCheckpoint callback) only pass (self, path). "
            f"Adding kwargs would break those callers."
        )

    def test_load_checkpoint_includes_strict_config_kwarg(self):
        """Trainer.load_checkpoint MUST expose strict_config kwarg with default False."""
        import inspect
        from lobtrainer.training.trainer import Trainer
        sig = inspect.signature(Trainer.load_checkpoint)
        assert "strict_config" in sig.parameters, (
            "Trainer.load_checkpoint must accept strict_config kwarg "
            "(Phase X.1 v2 contract; Phase X.4 will flip default to True)."
        )
        assert sig.parameters["strict_config"].default is False, (
            "strict_config default MUST be False until Phase X.4 promotion gate clears."
        )

    def test_simple_trainer_load_checkpoint_signature(self):
        """SimpleModelTrainer.load_checkpoint mirrors Trainer's strict_config kwarg."""
        import inspect
        from lobtrainer.training.simple_trainer import SimpleModelTrainer
        sig = inspect.signature(SimpleModelTrainer.load_checkpoint)
        assert "strict_config" in sig.parameters
        assert sig.parameters["strict_config"].default is False


class TestGoldenHash:
    """Phase X.1 v2 post-validation (Agent 3 Q7b — hft-rules §13 mandatory).

    Independent metric validation: cross-check compute_model_config_hash
    against a hand-computed SHA-256 via Python stdlib (NOT canonical_hash SSoT).

    If this test fires, it means EITHER (a) canonical_json_blob ordering
    changed, OR (b) sha256_hex implementation changed, OR (c) the
    _LOSS_TUNING_KEYS denylist changed semantics. ANY of these would
    silently rotate every existing checkpoint's model_config_hash field
    and invalidate cross-experiment fingerprint comparisons.
    """

    def test_compute_model_config_hash_independent_validation(self, tmp_path):
        """Cross-check via stdlib hashlib — locks the canonical-form contract."""
        import hashlib
        import json

        config = _build_test_config(
            tmp_path,
            model_type="tlob",
            params={
                "hidden_dim": 192,
                "num_layers": 4,
                "num_heads": 1,
                "dropout": 0.1,
                # Loss-tuning keys (denylist) — should NOT affect hash
                "gmadl_a": 1.0,
                "regression_loss_type": "huber",
            },
        )

        # SSoT path
        h_ssot = compute_model_config_hash(config.model)

        # Independent path: replicate the canonical form via stdlib only
        from lobtrainer.training.compatibility import _LOSS_TUNING_KEYS
        arch_params = {
            k: v
            for k, v in dict(config.model.params).items()
            if k not in _LOSS_TUNING_KEYS
        }
        independent_canonical_form = {
            "model_type": "tlob",
            "params": arch_params,
        }
        # Match canonical_json_blob discipline: sort_keys=True, default=str
        independent_blob = json.dumps(
            independent_canonical_form, sort_keys=True, default=str
        ).encode("utf-8")
        h_independent = hashlib.sha256(independent_blob).hexdigest()

        assert h_ssot == h_independent, (
            f"compute_model_config_hash drifted from independent stdlib path.\n"
            f"  SSoT (canonical_hash):      {h_ssot}\n"
            f"  Independent (stdlib):       {h_independent}\n"
            f"This rotates every checkpoint's model_config_hash. "
            f"Before regenerating goldens, verify which side changed: "
            f"canonical_json_blob ordering OR sha256_hex impl OR _LOSS_TUNING_KEYS."
        )


class TestEndToEndLifecycle:
    """Phase X.1 v2 post-validation (Agent 3 Q3a — CRITICAL gap).

    The single highest-leverage test: exercises the FULL operator flow
    train → save → load matching → mutate → load mismatching warn → strict raise.

    This test catches every regression in the warn/raise path including
    the Q1 optimizer-None bug, _resumed_from_checkpoint flag lifecycle,
    fingerprint mismatch detection, and CompatibilityContract round-trip.
    """

    def test_save_load_lifecycle_warn_then_strict_then_raise(self, tmp_path):
        """Full lifecycle: same config no warning; mutate config warn;
        mutate config + strict raise."""
        import warnings
        from lobtrainer.training.compatibility import (
            CheckpointConfigMismatchError,
            CheckpointConfigMismatchWarning,
        )
        from lobtrainer.training.trainer import Trainer

        # Phase 1: build trainer1 + save checkpoint with frozen params
        config = _build_test_config(
            tmp_path / "exp1",
            model_type="logistic",
            feature_count=20,
            params={"hidden_dim": 32},
        )
        trainer1 = Trainer.__new__(Trainer)
        trainer1.config = config
        import torch.nn as nn
        trainer1.model = nn.Linear(2, 2)
        trainer1._optimizer = None
        trainer1._scheduler = None
        from lobtrainer.training.trainer import TrainingState
        trainer1.state = TrainingState()
        trainer1.device = "cpu"

        ckpt_path = tmp_path / "lifecycle_test.pt"
        ckpt_dict = trainer1._build_checkpoint_dict()
        import torch
        torch.save(ckpt_dict, ckpt_path)

        # Phase 2: load via SAME config — no warning expected
        trainer2 = Trainer.__new__(Trainer)
        trainer2.config = config
        trainer2.model = nn.Linear(2, 2)
        trainer2._optimizer = None
        trainer2._scheduler = None
        trainer2.state = TrainingState()
        trainer2.device = "cpu"
        trainer2._resumed_from_checkpoint = False

        with warnings.catch_warnings():
            warnings.simplefilter("error", CheckpointConfigMismatchWarning)
            trainer2.load_checkpoint(ckpt_path, load_optimizer=False)
        assert trainer2._resumed_from_checkpoint is True, (
            "Phase X.1.K: flag MUST be True after successful load_checkpoint"
        )

        # Phase 3: load via MUTATED config — warning expected
        mutated_config = _build_test_config(
            tmp_path / "exp1_mut",
            model_type="logistic",
            feature_count=20,
            params={"hidden_dim": 64},  # changed from 32 → 64
        )
        trainer3 = Trainer.__new__(Trainer)
        trainer3.config = mutated_config
        trainer3.model = nn.Linear(2, 2)
        trainer3._optimizer = None
        trainer3._scheduler = None
        trainer3.state = TrainingState()
        trainer3.device = "cpu"
        trainer3._resumed_from_checkpoint = False

        with pytest.warns(CheckpointConfigMismatchWarning):
            trainer3.load_checkpoint(ckpt_path, load_optimizer=False)

        # Phase 4: load via MUTATED config + strict_config=True — raise expected
        trainer4 = Trainer.__new__(Trainer)
        trainer4.config = mutated_config
        trainer4.model = nn.Linear(2, 2)
        trainer4._optimizer = None
        trainer4._scheduler = None
        trainer4.state = TrainingState()
        trainer4.device = "cpu"
        trainer4._resumed_from_checkpoint = False

        with pytest.raises(CheckpointConfigMismatchError):
            trainer4.load_checkpoint(ckpt_path, load_optimizer=False, strict_config=True)


class TestExceptionClasses:
    def test_mismatch_error_is_value_error_subclass(self):
        assert issubclass(CheckpointConfigMismatchError, ValueError)

    def test_mismatch_warning_is_user_warning_subclass(self):
        assert issubclass(CheckpointConfigMismatchWarning, UserWarning)

    def test_missing_warning_is_user_warning_subclass(self):
        assert issubclass(CheckpointMissingFingerprintWarning, UserWarning)

    def test_mismatch_error_and_warning_are_distinct(self):
        """Error class is NOT a UserWarning subclass."""
        assert not issubclass(CheckpointConfigMismatchError, UserWarning)
        assert not issubclass(CheckpointConfigMismatchWarning, ValueError)

    def test_mismatch_warning_and_missing_warning_are_distinct(self):
        """Both inherit UserWarning but are distinct classes (different filter rules)."""
        assert CheckpointConfigMismatchWarning is not CheckpointMissingFingerprintWarning
        assert not issubclass(CheckpointConfigMismatchWarning, CheckpointMissingFingerprintWarning)
        assert not issubclass(CheckpointMissingFingerprintWarning, CheckpointConfigMismatchWarning)


# ---------------------------------------------------------------------------
# Phase 1 N7 (#PY-10, 2026-05-06) — normalization stats SHA bound to checkpoint
# ---------------------------------------------------------------------------


class TestN7NormalizationStatsBoundToCheckpoint:
    """Phase 1 N7 forensic-bug closure (#PY-10, 2026-05-06).

    Pre-fix: checkpoint embeds compatibility_fingerprint + model_config_hash
    (Phase X.1 v2 closed F-13 CONFIG drift) but NOT normalization_stats_sha256.
    Re-extracting the dataset with different per-day stats produces silently
    different inference behavior on resume — DORMANT-PRIMED data-stats drift.

    Post-fix: ``Trainer._build_checkpoint_dict`` embeds SHA-256 of the active
    normalization stats file (via ``hft_contracts.provenance.hash_file`` SSoT
    per #PY-41 anti-pattern recurrence). ``Trainer.load_checkpoint`` validates
    against the active config and raises ``CheckpointConfigMismatchError`` in
    strict mode (or warns ``CheckpointConfigMismatchWarning`` in default mode)
    on three N7 surfaces: hash mismatch, stats file missing at active path,
    or strategy divergence (active produces no stats file). Pre-N7 checkpoints
    (no key) silently skip the check — back-compat preserved.

    SSoT REUSE: ``hft_contracts.provenance.hash_file`` (Phase V.1.5 consolidation).
    NO new canonical-hash site — locked by reuse contract per hft-rules §0.
    """

    @pytest.mark.parametrize("strategy_str,expected_filename", [
        ("global_zscore", "normalization_stats.json"),
        ("hybrid", "hybrid_normalization_stats.json"),
        ("none", None),
    ])
    def test_derive_normalization_stats_path_per_strategy(
        self, strategy_str, expected_filename, tmp_path,
    ):
        """Helper maps (strategy, data_dir) → file path; None for NONE strategy."""
        from lobtrainer.training.trainer import _derive_normalization_stats_path

        config = _build_test_config(
            tmp_path,
            model_type="logistic",
            feature_count=20,
            normalization_strategy=strategy_str,
        )
        result = _derive_normalization_stats_path(config)
        if expected_filename is None:
            assert result is None
        else:
            assert result == Path(config.data.data_dir) / expected_filename

    def test_derive_normalization_stats_path_none_for_no_data_dir(self):
        """Multi-source mode: data_dir is None → stats path is None."""
        from types import SimpleNamespace
        from lobtrainer.training.trainer import _derive_normalization_stats_path
        from lobtrainer.config.schema import NormalizationStrategy

        # Synthetic config — no need to build full ExperimentConfig
        config = SimpleNamespace(
            data=SimpleNamespace(
                data_dir=None,
                normalization=SimpleNamespace(strategy=NormalizationStrategy.GLOBAL_ZSCORE),
            ),
        )
        assert _derive_normalization_stats_path(config) is None

    def _build_trainer_for_n7(self, tmp_path, strategy_str="global_zscore", create_stats_file=True):
        """Build a minimal Trainer with optional fake stats file at the strategy-specific path.

        Returns (trainer, stats_path) where stats_path is the file actually
        written (or None if strategy doesn't produce a path or create_stats_file=False).
        """
        from lobtrainer.training.trainer import (
            Trainer,
            TrainingState,
            _derive_normalization_stats_path,
        )
        import torch.nn as nn

        config = _build_test_config(
            tmp_path,
            model_type="logistic",
            feature_count=20,
            normalization_strategy=strategy_str,
        )
        stats_path = _derive_normalization_stats_path(config)
        if stats_path is not None and create_stats_file:
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            stats_path.write_text(
                '{"means": [0.1, 0.2, 0.3], "stds": [1.1, 1.2, 1.3]}'
            )

        trainer = Trainer.__new__(Trainer)
        trainer.config = config
        trainer.model = nn.Linear(2, 2)
        trainer._optimizer = None
        trainer._scheduler = None
        trainer.state = TrainingState()
        trainer.device = "cpu"
        trainer._resumed_from_checkpoint = False
        return trainer, stats_path

    def test_build_checkpoint_dict_embeds_normalization_stats_sha256(self, tmp_path):
        """Save side: when stats file exists, checkpoint embeds 64-hex SHA-256
        matching ``hash_file`` SSoT output."""
        from hft_contracts.provenance import hash_file

        trainer, stats_path = self._build_trainer_for_n7(tmp_path, "global_zscore")
        assert stats_path is not None and stats_path.exists()

        ckpt = trainer._build_checkpoint_dict()
        assert "normalization_stats_sha256" in ckpt, (
            "Phase 1 N7: _build_checkpoint_dict MUST emit 'normalization_stats_sha256' key"
        )
        sha = ckpt["normalization_stats_sha256"]
        assert sha is not None
        assert isinstance(sha, str)
        assert len(sha) == 64, f"SHA-256 hex must be 64 chars; got {len(sha)}"
        # Independent SSoT verification — direct hash of the file matches embedded value
        assert sha == hash_file(stats_path, missing_ok=False), (
            "Embedded SHA must match hft_contracts.provenance.hash_file output (SSoT)"
        )

    def test_build_checkpoint_dict_norm_sha_none_when_strategy_is_none(self, tmp_path):
        """Save side: NONE strategy → no stats file produced → embedded sha is None."""
        trainer, stats_path = self._build_trainer_for_n7(tmp_path, "none")
        assert stats_path is None  # NONE strategy produces no path

        ckpt = trainer._build_checkpoint_dict()
        assert "normalization_stats_sha256" in ckpt
        assert ckpt["normalization_stats_sha256"] is None

    def test_build_checkpoint_dict_norm_sha_none_when_stats_file_absent(self, tmp_path):
        """Save side: GLOBAL_ZSCORE but stats file not yet computed → embedded sha is None.

        Models the pre-setup() state where the path is derived but no file exists yet.
        """
        trainer, stats_path = self._build_trainer_for_n7(
            tmp_path, "global_zscore", create_stats_file=False,
        )
        assert stats_path is not None and not stats_path.exists()

        ckpt = trainer._build_checkpoint_dict()
        assert ckpt["normalization_stats_sha256"] is None

    def test_load_checkpoint_no_warning_on_norm_sha_match(self, tmp_path):
        """Load side: matching norm_sha emits NO N7 warning (silent success)."""
        import warnings
        import torch
        from lobtrainer.training.trainer import Trainer, TrainingState
        import torch.nn as nn

        # Save with stats file content A
        trainer1, stats_path = self._build_trainer_for_n7(tmp_path, "global_zscore")
        ckpt_dict = trainer1._build_checkpoint_dict()
        ckpt_path = tmp_path / "ckpt.pt"
        torch.save(ckpt_dict, ckpt_path)

        # Load with same config + same stats file — no warning
        trainer2 = Trainer.__new__(Trainer)
        trainer2.config = trainer1.config
        trainer2.model = nn.Linear(2, 2)
        trainer2._optimizer = None
        trainer2._scheduler = None
        trainer2.state = TrainingState()
        trainer2.device = "cpu"
        trainer2._resumed_from_checkpoint = False

        with warnings.catch_warnings():
            warnings.simplefilter("error", CheckpointConfigMismatchWarning)
            # Must NOT raise — round-trip identical config is silent-pass
            trainer2.load_checkpoint(ckpt_path, load_optimizer=False)
        assert trainer2._resumed_from_checkpoint is True

    def test_load_checkpoint_warns_on_norm_sha_mismatch_default(self, tmp_path):
        """Load side: default strict_config=False emits CheckpointConfigMismatchWarning
        when stats file content changes between save and load (data re-extraction)."""
        import torch
        from lobtrainer.training.trainer import Trainer, TrainingState
        import torch.nn as nn

        trainer1, stats_path = self._build_trainer_for_n7(tmp_path, "global_zscore")
        ckpt_dict = trainer1._build_checkpoint_dict()
        ckpt_path = tmp_path / "ckpt.pt"
        torch.save(ckpt_dict, ckpt_path)

        # Mutate stats file (simulate re-extraction with different per-day stats)
        stats_path.write_text('{"means": [99.0, -99.0, 99.0], "stds": [99.0, 99.0, 99.0]}')

        trainer2 = Trainer.__new__(Trainer)
        trainer2.config = trainer1.config
        trainer2.model = nn.Linear(2, 2)
        trainer2._optimizer = None
        trainer2._scheduler = None
        trainer2.state = TrainingState()
        trainer2.device = "cpu"
        trainer2._resumed_from_checkpoint = False

        with pytest.warns(
            CheckpointConfigMismatchWarning, match="Normalization stats SHA mismatch"
        ):
            trainer2.load_checkpoint(ckpt_path, load_optimizer=False)

    def test_load_checkpoint_raises_on_norm_sha_mismatch_strict(self, tmp_path):
        """Load side: strict_config=True raises CheckpointConfigMismatchError on hash mismatch."""
        import torch
        from lobtrainer.training.trainer import Trainer, TrainingState
        import torch.nn as nn

        trainer1, stats_path = self._build_trainer_for_n7(tmp_path, "global_zscore")
        ckpt_dict = trainer1._build_checkpoint_dict()
        ckpt_path = tmp_path / "ckpt.pt"
        torch.save(ckpt_dict, ckpt_path)

        # Mutate stats file
        stats_path.write_text('{"means": [99.0, -99.0, 99.0], "stds": [99.0, 99.0, 99.0]}')

        trainer2 = Trainer.__new__(Trainer)
        trainer2.config = trainer1.config
        trainer2.model = nn.Linear(2, 2)
        trainer2._optimizer = None
        trainer2._scheduler = None
        trainer2.state = TrainingState()
        trainer2.device = "cpu"
        trainer2._resumed_from_checkpoint = False

        with pytest.raises(
            CheckpointConfigMismatchError, match="Normalization stats SHA mismatch"
        ):
            trainer2.load_checkpoint(
                ckpt_path, load_optimizer=False, strict_config=True
            )

    def test_load_checkpoint_warns_on_missing_active_stats(self, tmp_path):
        """Load side: when ckpt has SHA but active stats file deleted, warn (or raise in strict)."""
        import torch
        from lobtrainer.training.trainer import Trainer, TrainingState
        import torch.nn as nn

        trainer1, stats_path = self._build_trainer_for_n7(tmp_path, "global_zscore")
        ckpt_dict = trainer1._build_checkpoint_dict()
        ckpt_path = tmp_path / "ckpt.pt"
        torch.save(ckpt_dict, ckpt_path)

        # Delete the stats file (simulate post-train cleanup or wrong data_dir)
        stats_path.unlink()

        trainer2 = Trainer.__new__(Trainer)
        trainer2.config = trainer1.config
        trainer2.model = nn.Linear(2, 2)
        trainer2._optimizer = None
        trainer2._scheduler = None
        trainer2.state = TrainingState()
        trainer2.device = "cpu"
        trainer2._resumed_from_checkpoint = False

        with pytest.warns(CheckpointConfigMismatchWarning, match="does not exist"):
            trainer2.load_checkpoint(ckpt_path, load_optimizer=False)

    def test_load_checkpoint_pre_n7_checkpoint_silently_skips_norm_check(self, tmp_path):
        """Load side: pre-N7 checkpoint (no normalization_stats_sha256 key) silently skips
        N7 validation. Back-compat lock preventing N7 from accidentally raising on PRE-N7
        artifacts (mirrors the X.1 v2 missing-key back-compat pattern at line 1254-1262)."""
        import warnings as warnings_mod
        import torch
        from lobtrainer.training.trainer import Trainer, TrainingState
        import torch.nn as nn

        # Build trainer + checkpoint dict, then DELETE the N7 key (simulate pre-N7 artifact)
        trainer1, _ = self._build_trainer_for_n7(tmp_path, "global_zscore")
        ckpt_dict = trainer1._build_checkpoint_dict()
        ckpt_dict.pop("normalization_stats_sha256")  # simulate pre-N7
        ckpt_path = tmp_path / "ckpt_pre_n7.pt"
        torch.save(ckpt_dict, ckpt_path)

        trainer2 = Trainer.__new__(Trainer)
        trainer2.config = trainer1.config
        trainer2.model = nn.Linear(2, 2)
        trainer2._optimizer = None
        trainer2._scheduler = None
        trainer2.state = TrainingState()
        trainer2.device = "cpu"
        trainer2._resumed_from_checkpoint = False

        # No N7 warning should be emitted for pre-N7 checkpoint
        with warnings_mod.catch_warnings():
            warnings_mod.simplefilter("error", CheckpointConfigMismatchWarning)
            trainer2.load_checkpoint(ckpt_path, load_optimizer=False)
        assert trainer2._resumed_from_checkpoint is True

    def test_load_checkpoint_warns_on_strategy_divergence(self, tmp_path):
        """Load side: ckpt has norm_sha (saved with GLOBAL_ZSCORE) but active config
        produces no stats path (strategy=NONE). Defense-in-depth gate at
        trainer.py:1405-1416 catches strategy divergence even though
        compatibility_fingerprint also already detects it (normalization_strategy
        is in CompatibilityContract).

        Closes adversarial-review-identified coverage gap at the only uncovered
        N7 branch (`active_stats_path is None`)."""
        import torch
        from lobtrainer.training.trainer import Trainer, TrainingState
        import torch.nn as nn

        # Save with GLOBAL_ZSCORE + stats file
        trainer1, stats_path = self._build_trainer_for_n7(tmp_path, "global_zscore")
        ckpt_dict = trainer1._build_checkpoint_dict()
        assert ckpt_dict["normalization_stats_sha256"] is not None  # ckpt has SHA
        ckpt_path = tmp_path / "ckpt.pt"
        torch.save(ckpt_dict, ckpt_path)

        # Load with strategy=NONE (active config produces no stats path)
        # NOTE: compatibility_fingerprint will also mismatch (different normalization_strategy);
        # we expect BOTH warnings (compat first, then N7). Use match= to scope to the N7 one.
        trainer2, none_stats_path = self._build_trainer_for_n7(
            tmp_path / "exp_none", "none", create_stats_file=False,
        )
        assert none_stats_path is None  # NONE strategy → no path

        with pytest.warns(CheckpointConfigMismatchWarning, match="produces no stats file"):
            trainer2.load_checkpoint(ckpt_path, load_optimizer=False)
