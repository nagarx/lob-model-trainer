"""Cycle 1b.2 (2026-05-07) — trainer schema bridge for #PY-54 closure.

Tests the trainer-side bridge that propagates ``ModelConfig.hmhp_primary_horizon_idx``
through ``ModelConfig._build_params_from_legacy`` into the params dict consumed
by lob-models ``HMHPConfig`` (Cycle 1b.1, commit 83ab54c).

Also tests the cross-config invariant at ``ExperimentConfig._validate_all``:
``data.labels.primary_horizon_idx`` MUST match ``model.hmhp_primary_horizon_idx``
for HMHP/HMHP-R model types — silent divergence would corrupt gradients per
Adversarial Agent 9 §H#1.
"""

import pytest

from lobtrainer.config.schema import (
    DataConfig,
    ExperimentConfig,
    LabelsConfig,
    ModelConfig,
    ModelType,
    SequenceConfig,
    SourceConfig,
    TrainConfig,
)


def _build_minimal_hmhp_config(
    *,
    model_type: str = "hmhp",
    hmhp_primary_horizon_idx: int = 0,
    labels_primary_horizon_idx=0,  # accepts int or None
    horizons=(10, 20, 50, 100, 200),
):
    """Helper: minimal ExperimentConfig for HMHP/HMHP-R cross-config tests.

    Keeps the rest of the config defaults — only varies the two fields under
    test. lob-models HMHPConfig requires ``len(horizons) >= 2``; default 5
    horizons match the lob-models default at base.py:1886-1888.
    """
    labels_kwargs = {"task": "regression"} if model_type == "hmhp_regression" else {}
    return ExperimentConfig(
        name="test_cycle_1b2_bridge",
        output_dir="/tmp/test",
        data=DataConfig(
            data_dir="/tmp/data",
            feature_count=98,
            sequence=SequenceConfig(window_size=20, stride=1),
            labels=LabelsConfig(
                horizons=list(horizons),
                primary_horizon_idx=labels_primary_horizon_idx,
                **labels_kwargs,
            ),
        ),
        model=ModelConfig(
            model_type=ModelType(model_type),
            input_size=98,
            num_classes=3,
            hmhp_horizons=tuple(horizons),
            hmhp_primary_horizon_idx=hmhp_primary_horizon_idx,
        ),
        train=TrainConfig(
            batch_size=64, epochs=1, learning_rate=1e-4,
        ),
    )


class TestBridgePropagation:
    """ModelConfig._build_params_from_legacy must propagate hmhp_primary_horizon_idx."""

    def test_default_zero_propagates_to_params(self):
        """When ``hmhp_primary_horizon_idx=0`` (default), params dict carries it."""
        cfg = _build_minimal_hmhp_config(
            hmhp_primary_horizon_idx=0,
            labels_primary_horizon_idx=0,
        )
        # _build_params_from_legacy is called inside ModelConfig._validate_all
        # and stored on self.model.params
        assert cfg.model.params.get("primary_horizon_idx") == 0

    def test_nonzero_propagates_to_params(self):
        """When ``hmhp_primary_horizon_idx=2``, params dict carries 2."""
        cfg = _build_minimal_hmhp_config(
            hmhp_primary_horizon_idx=2,
            labels_primary_horizon_idx=2,
            horizons=(10, 20, 50, 100, 200),  # idx=2 selects horizon 50
        )
        assert cfg.model.params.get("primary_horizon_idx") == 2

    def test_propagates_for_hmhp_regression(self):
        """Bridge fires for hmhp_regression model type, not just hmhp."""
        cfg = _build_minimal_hmhp_config(
            model_type="hmhp_regression",
            hmhp_primary_horizon_idx=1,
            labels_primary_horizon_idx=1,
        )
        assert cfg.model.params.get("primary_horizon_idx") == 1


class TestCrossConfigInvariant:
    """ExperimentConfig._validate_all enforces label/model alignment."""

    def test_default_zero_zero_passes(self):
        """Both default 0 — no mismatch."""
        cfg = _build_minimal_hmhp_config(
            hmhp_primary_horizon_idx=0,
            labels_primary_horizon_idx=0,
        )
        # Should NOT raise
        assert cfg.model.hmhp_primary_horizon_idx == 0
        assert cfg.data.labels.primary_horizon_idx == 0

    def test_explicit_matching_pair_passes(self):
        """Both idx=1 — explicit match passes."""
        cfg = _build_minimal_hmhp_config(
            hmhp_primary_horizon_idx=1,
            labels_primary_horizon_idx=1,
        )
        assert cfg.model.hmhp_primary_horizon_idx == 1

    def test_explicit_mismatch_raises(self):
        """Different EXPLICIT values trigger cross-config invariant raise.

        Two-tier policy: auto-align when model is at default 0; strict-raise
        only when model is explicitly non-zero AND mismatches labels.
        """
        with pytest.raises(ValueError, match="Cross-config invariant violation"):
            _build_minimal_hmhp_config(
                hmhp_primary_horizon_idx=1,  # model says 1 (explicit non-zero)
                labels_primary_horizon_idx=2,  # labels says 2
            )

    def test_auto_align_when_model_default_zero_labels_nonzero(self):
        """Pre-Cycle-1b.1 production YAMLs (e.g., nvda_hmhp_40feat_h60_*.yaml)
        set ``data.labels.primary_horizon_idx=1`` but did NOT set the new
        ``model.hmhp_primary_horizon_idx`` field (didn't exist). Cycle 1b.2
        auto-aligns model field to labels at construction time, transparently
        fixing the pre-existing silent drift.
        """
        cfg = _build_minimal_hmhp_config(
            hmhp_primary_horizon_idx=0,  # model at default
            labels_primary_horizon_idx=1,  # labels non-default
        )
        # After auto-align, model field == labels field
        assert cfg.model.hmhp_primary_horizon_idx == 1
        assert cfg.data.labels.primary_horizon_idx == 1
        # params dict ALSO carries the auto-aligned value
        assert cfg.model.params.get("primary_horizon_idx") == 1

    def test_explicit_mismatch_error_message_includes_PY54(self):
        """Error message cites #PY-54 for traceability (explicit-mismatch path)."""
        with pytest.raises(ValueError, match="#PY-54"):
            _build_minimal_hmhp_config(
                hmhp_primary_horizon_idx=2,  # model explicit non-zero
                labels_primary_horizon_idx=3,  # mismatch
            )

    def test_labels_none_treated_as_zero(self):
        """``LabelsConfig.primary_horizon_idx=None`` matches ``model=0`` (None ≡ 0)."""
        cfg = _build_minimal_hmhp_config(
            hmhp_primary_horizon_idx=0,
            labels_primary_horizon_idx=None,
        )
        # Should NOT raise (None means "all-horizons HMHP mode" → effective 0)
        assert cfg.model.hmhp_primary_horizon_idx == 0
        assert cfg.data.labels.primary_horizon_idx is None

    def test_labels_none_does_not_match_model_explicit_nonzero(self):
        """``labels=None, model=1`` is EXPLICIT-mismatch path (None → effective
        0 ≠ 1); raises per strict-raise tier of two-tier policy."""
        with pytest.raises(ValueError, match="Cross-config invariant violation"):
            _build_minimal_hmhp_config(
                hmhp_primary_horizon_idx=1,  # explicit non-default
                labels_primary_horizon_idx=None,  # effective 0
            )

    def test_non_hmhp_model_skip_invariant(self):
        """For non-HMHP model types (e.g., tlob), the invariant doesn't fire."""
        # tlob ignores hmhp_primary_horizon_idx entirely — mismatch is harmless.
        # We construct an ExperimentConfig with tlob model_type + mismatched
        # hmhp_primary_horizon_idx (which is irrelevant for tlob) — should NOT raise.
        cfg = ExperimentConfig(
            name="test_tlob_skip_invariant",
            output_dir="/tmp/test",
            data=DataConfig(
                data_dir="/tmp/data",
                feature_count=98,
                sequence=SequenceConfig(window_size=20, stride=1),
                labels=LabelsConfig(
                    horizons=[10, 60, 300],
                    primary_horizon_idx=1,  # labels says 1
                ),
            ),
            model=ModelConfig(
                model_type=ModelType("tlob"),
                input_size=98,
                num_classes=3,
                hmhp_primary_horizon_idx=0,  # model defaults 0 (irrelevant for tlob)
            ),
            train=TrainConfig(
                batch_size=64, epochs=1, learning_rate=1e-4,
            ),
        )
        # No raise — tlob is not in {"hmhp", "hmhp_regression"}
        assert cfg.model.model_type == ModelType.TLOB

    def test_logistic_model_skip_invariant(self):
        """For logistic model_type, the cross-config invariant also skips."""
        cfg = ExperimentConfig(
            name="test_logistic_skip",
            output_dir="/tmp/test",
            data=DataConfig(
                data_dir="/tmp/data",
                feature_count=98,
                sequence=SequenceConfig(window_size=20, stride=1),
                labels=LabelsConfig(
                    horizons=[10, 60, 300],
                    primary_horizon_idx=2,  # labels says 2
                ),
            ),
            model=ModelConfig(
                model_type=ModelType("logistic"),
                input_size=98,
                num_classes=3,
                hmhp_primary_horizon_idx=0,  # different from labels — but tlob ignores
            ),
            train=TrainConfig(
                batch_size=64, epochs=1, learning_rate=1e-4,
            ),
        )
        # No raise
        assert cfg.model.hmhp_primary_horizon_idx == 0


class TestModelConfigField:
    """Field declaration + Pydantic validation behavior on ModelConfig."""

    def test_default_value_is_zero(self):
        """ModelConfig.hmhp_primary_horizon_idx defaults to 0."""
        config = ModelConfig(
            model_type=ModelType.HMHP, input_size=98, num_classes=3,
        )
        assert config.hmhp_primary_horizon_idx == 0

    def test_explicit_value_preserved(self):
        """Explicit value passes through ModelConfig validation."""
        config = ModelConfig(
            model_type=ModelType.HMHP, input_size=98, num_classes=3,
            hmhp_primary_horizon_idx=2,
            hmhp_horizons=(10, 60, 300, 600, 900),  # idx 2 valid
        )
        assert config.hmhp_primary_horizon_idx == 2

    def test_negative_value_caught_by_lobmodels_at_construction(self):
        """When the model is constructed via lob-models HMHPConfig (Cycle 1b.1
        validator catches negative), the bridge still propagates the value
        correctly — we only check trainer side passes the raw int."""
        # Trainer-side ModelConfig accepts the int (no negative-check at trainer);
        # lob-models HMHPConfig validator at base.py raises on negative. This
        # test documents the layered defense — fail-loud at lob-models construction
        # downstream is the safety net.
        config = ModelConfig(
            model_type=ModelType.HMHP, input_size=98, num_classes=3,
            hmhp_primary_horizon_idx=-1,  # trainer accepts; lob-models would reject
        )
        # No raise at trainer ModelConfig construction
        assert config.hmhp_primary_horizon_idx == -1
