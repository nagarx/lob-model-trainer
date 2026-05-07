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


def _build_hmhp_config_with_separate_horizons(
    *,
    model_type: str = "hmhp",
    model_horizons=(10, 60, 300),
    labels_horizons=(10, 60, 300),
    hmhp_primary_horizon_idx: int = 0,
    labels_primary_horizon_idx: int = 0,
):
    """Helper: HMHP ExperimentConfig with INDEPENDENTLY-controlled horizons
    on model.hmhp_horizons vs data.labels.horizons.

    Cycle 2.5b (2026-05-07) — used to test the cross-config horizon-LIST
    mismatch detection. Distinct from ``_build_minimal_hmhp_config`` which
    sets BOTH horizons fields from the same parameter.
    """
    labels_kwargs = {"task": "regression"} if model_type == "hmhp_regression" else {}
    return ExperimentConfig(
        name="test_cycle_2_5b",
        output_dir="/tmp/test",
        data=DataConfig(
            data_dir="/tmp/data",
            feature_count=98,
            sequence=SequenceConfig(window_size=20, stride=1),
            labels=LabelsConfig(
                horizons=list(labels_horizons),
                primary_horizon_idx=labels_primary_horizon_idx,
                **labels_kwargs,
            ),
        ),
        model=ModelConfig(
            model_type=ModelType(model_type),
            input_size=98,
            num_classes=3,
            hmhp_horizons=tuple(model_horizons),
            hmhp_primary_horizon_idx=hmhp_primary_horizon_idx,
        ),
        train=TrainConfig(
            batch_size=64, epochs=1, learning_rate=1e-4,
        ),
    )


class TestCrossConfigHorizonListMismatch:
    """Cycle 2.5b (2026-05-07) — VALUE-check beyond IDX-check.

    Closes Issue 2 from V1-V6 + V7-V12 6-agent re-validation. Pre-Cycle-2.5b,
    the cross-config validator at schema.py:2407-2450 only verified that
    ``model.hmhp_primary_horizon_idx == data.labels.primary_horizon_idx`` —
    catching index-typo bugs but allowing horizon LISTS to silently differ
    (permutations or different lengths) when both indices are in range.

    HMHP cascade has ONE decoder per horizon (lob-models hmhp.py); HMHP
    loss is sum_h(loss_h), so EVERY horizon's value-alignment matters,
    not just primary_horizon_idx. Mismatched lists corrupt gradients at
    non-primary heads.

    Per hft-rules §5 fail-fast + §8 never silently accept corrupt data.
    Defense-in-depth atop:
    - LabelsConfig dup-check (schema.py:411-413)
    - HMHPConfig dup-check (lob-models config/base.py:2093+)
    - Cross-config IDX-check (schema.py:2407-2450)
    """

    def test_identical_lists_pass(self):
        """Golden case — same list on both fields, identical idx."""
        cfg = _build_hmhp_config_with_separate_horizons(
            model_horizons=(10, 60, 300),
            labels_horizons=(10, 60, 300),
        )
        assert tuple(cfg.model.hmhp_horizons) == (10, 60, 300)
        assert tuple(cfg.data.labels.horizons) == (10, 60, 300)

    def test_permutation_at_resolved_idx_raises(self):
        """Same elements, different order at primary idx → RAISE.

        model.hmhp_horizons=(10, 60, 300), labels.horizons=(60, 10, 300).
        idx=0: model means H10, labels means H60. Pre-2.5b passed silently;
        post-2.5b raises with explicit list-mismatch message.
        """
        with pytest.raises(ValueError, match="horizon-LIST mismatch"):
            _build_hmhp_config_with_separate_horizons(
                model_horizons=(10, 60, 300),
                labels_horizons=(60, 10, 300),
            )

    def test_permutation_off_resolved_idx_still_raises(self):
        """Permutation at NON-primary idx ALSO raises (full-tuple equality).

        model.hmhp_horizons=(10, 60, 300), labels.horizons=(10, 300, 60).
        idx=0: BOTH means H10 (primary aligns). But heads 1+2 see permuted
        labels — silent corruption at non-primary heads. Per Agent Y +
        Agent Z V12 review: HMHP loss = sum_h(loss_h), so non-primary head
        misalignment also corrupts gradients. Catches case option (a)
        (resolved-idx-only) would miss.
        """
        with pytest.raises(ValueError, match="horizon-LIST mismatch"):
            _build_hmhp_config_with_separate_horizons(
                model_horizons=(10, 60, 300),
                labels_horizons=(10, 300, 60),
                hmhp_primary_horizon_idx=0,
                labels_primary_horizon_idx=0,
            )

    def test_different_lengths_raise(self):
        """Different-length lists raise — HMHP cascade decoder count must
        match label slicing axis."""
        with pytest.raises(ValueError, match="horizon-LIST mismatch"):
            _build_hmhp_config_with_separate_horizons(
                model_horizons=(10, 60),
                labels_horizons=(10, 60, 300),
            )

    def test_completely_different_lists_raise(self):
        """Disjoint horizon sets raise."""
        with pytest.raises(ValueError, match="horizon-LIST mismatch"):
            _build_hmhp_config_with_separate_horizons(
                model_horizons=(10, 60, 300),
                labels_horizons=(20, 50, 100),
            )

    def test_empty_labels_horizons_skips_check(self):
        """Empty labels.horizons SKIPS the value-check — auto-resolution
        from dataset_manifest at trainer setup time will populate it later
        (see trainer.py:850-866 + simple_trainer.py:240-260). Validator
        cannot prejudge what auto-resolution will produce.
        """
        # Should NOT raise even though model has horizons + labels is empty.
        cfg = _build_hmhp_config_with_separate_horizons(
            model_horizons=(10, 60, 300),
            labels_horizons=(),
        )
        assert tuple(cfg.model.hmhp_horizons) == (10, 60, 300)
        assert tuple(cfg.data.labels.horizons) == ()

    def test_empty_model_horizons_skips_check(self):
        """Symmetric-skip: empty model.hmhp_horizons + populated labels.horizons.

        Cycle 2.5b mid-impl gap closure (Agent verdict GAP-1 2026-05-07):
        defense-in-depth on the symmetric direction. In practice
        ModelConfig.hmhp_horizons has a non-empty default tuple at
        schema.py:1407, so this case is improbable — but locking the skip
        semantics here protects future programmatic constructions that
        explicitly pass an empty tuple.
        """
        cfg = _build_hmhp_config_with_separate_horizons(
            model_horizons=(),
            labels_horizons=(10, 60, 300),
        )
        assert tuple(cfg.model.hmhp_horizons) == ()
        assert tuple(cfg.data.labels.horizons) == (10, 60, 300)

    def test_both_empty_horizons_skip_check(self):
        """Trivial-skip: BOTH empty. Auto-resolution path entirely deferred.

        Cycle 2.5b mid-impl gap closure (Agent verdict GAP-2 2026-05-07):
        locks the both-empty semantics so future refactors don't
        accidentally promote this to a fail-fast path (which would
        regress the auto-resolve workflow at trainer.py:850-866).
        """
        cfg = _build_hmhp_config_with_separate_horizons(
            model_horizons=(),
            labels_horizons=(),
        )
        assert tuple(cfg.model.hmhp_horizons) == ()
        assert tuple(cfg.data.labels.horizons) == ()

    def test_value_check_skipped_for_tlob_model_type(self):
        """Non-HMHP model types skip the entire HMHP cross-config block.

        tlob does not use the hmhp_horizons field; mismatch is harmless.
        """
        cfg = _build_hmhp_config_with_separate_horizons(
            model_type="tlob",
            model_horizons=(10, 60, 300),
            labels_horizons=(10, 300, 60),  # permuted — but tlob ignores
        )
        assert cfg.model.model_type == ModelType.TLOB

    def test_auto_align_then_value_check_runs(self):
        """Auto-align tier (a) fires when model_idx=0 + labels_idx≠0.

        After auto-align, model.hmhp_primary_horizon_idx is mutated to match
        labels. The new VALUE-check runs in the SAME if-block so it fires
        AFTER auto-align resolves indices. With matching lists, it passes.
        """
        cfg = _build_hmhp_config_with_separate_horizons(
            model_horizons=(10, 60, 300),
            labels_horizons=(10, 60, 300),  # SAME list — equality holds
            hmhp_primary_horizon_idx=0,  # default
            labels_primary_horizon_idx=2,  # non-default → triggers auto-align
        )
        # Auto-align mutated idx; value-check passed (lists equal)
        assert cfg.model.hmhp_primary_horizon_idx == 2

    def test_auto_align_with_permuted_horizons_raises(self):
        """Auto-align fires (idx 0 → 2), then VALUE-check catches permutation.

        Locks the post-auto-align value-check semantics: even after the
        index-alignment branch runs, mismatched lists still raise. This is
        the most defensive composition of branches (a)+(c).
        """
        with pytest.raises(ValueError, match="horizon-LIST mismatch"):
            _build_hmhp_config_with_separate_horizons(
                model_horizons=(10, 60, 300),
                labels_horizons=(60, 10, 300),  # permuted
                hmhp_primary_horizon_idx=0,
                labels_primary_horizon_idx=2,  # triggers auto-align first
            )

    def test_error_message_cites_cycle_2_5b_and_PY_54(self):
        """Error message includes 'Cycle 2.5b' AND '#PY-54' tokens for
        traceability — matches existing #PY-54 invariant message style at
        line 2438-2450."""
        with pytest.raises(ValueError) as exc_info:
            _build_hmhp_config_with_separate_horizons(
                model_horizons=(10, 60, 300),
                labels_horizons=(60, 10, 300),
            )
        msg = str(exc_info.value)
        assert "Cycle 2.5b" in msg
        assert "#PY-54-VALUE" in msg
        assert "sum_h(loss_h)" in msg


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
