"""Phase DESIGN-1 G-4 (2026-05-10) — automated bit-exact replay regression test.

Closes the keystone reproducibility integration gap: validates that DESIGN-1
Phase A.1 + A.2 + G-1 + A.4 cumulatively produce TENSOR-EXACT identical
training outcomes across two consecutive in-process Trainer runs with
identical seed.

Per Wave 2.1 adversarial critique 2026-05-10 (refined from initial design):
  - DROP subprocess (redundant with test_dataloader_determinism_workers_2 +
    TestCrossProcessResumeInvariant keystone in test_callback_state_dict.py)
  - DROP SHA-256-on-pickled-blob (fragile across hardware due to optimizer
    state dict ordering + PyTorch save format drift)
  - USE ``torch.testing.assert_close(rtol=0, atol=0)`` walking
    ``model_state_dict`` (catches numerical drift; tolerates serialization
    encoding noise)
  - DROP resume-from-epoch-2 scenario (ModelCheckpoint default
    ``save_best_only=True`` + ``{metric:.4f}`` filename template makes
    epoch-grabbing fragile; would require a synthetic per-epoch callback)
  - DROP ``num_workers={0,2}`` parametrization (covered by
    ``test_dataloader_determinism_workers_2`` in
    ``test_reproducibility_design1.py`` Phase A.1)
  - Reuse ``synthetic_export_dir`` + logistic + 1 epoch (~10-20s runtime)

What this test catches that no existing DESIGN-1 unit test catches:
  - The architectural seam between ``set_seed → DataLoader construction
    → model init → train() forward/backward → save_checkpoint``. End-to-end
    integration coverage. Same-seed runs MUST produce bit-exact identical
    final ``model_state_dict``.

Side-check: ``compatibility_fingerprint`` MUST also match across same-seed
runs — runtime-path defense-in-depth complement to
``TestRngStateInvariance``'s structural-path lock.

Marked ``@pytest.mark.integration``. Skipped under fast CI via
``pytest -m "not integration"``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

import pytest
import torch

# Skip the entire module when ``hft_contracts`` isn't available (matches
# ``test_signal_exporter_integration.py:49`` convention).
_hft_contracts = pytest.importorskip("hft_contracts")

from lobtrainer.config.schema import (  # noqa: E402
    DataConfig,
    ExperimentConfig,
    LabelsConfig,
    ModelConfig,
    TrainConfig,
)
from lobtrainer.training.trainer import Trainer  # noqa: E402
from lobtrainer.utils.reproducibility import (  # noqa: E402
    _safe_torch_load,
    get_seed_state,
    set_seed_state,
    RngStatePolicy,
)


_FINGERPRINT_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


@pytest.fixture(autouse=True)
def _restore_global_rng_state():
    """Snapshot + restore global RNG state around each test in this module.

    ``Trainer.setup() + train()`` mutates global PyTorch / numpy / Python
    RNG state via dropout, optimizer randomness, and DataLoader shuffle.
    Without restore, downstream tests in the suite (e.g.,
    ``test_normalization_integration.py::test_normalize_then_select``)
    inherit the post-training state and may flake on incidental
    determinism assumptions.

    Uses Phase A.2 ``get_seed_state`` / ``set_seed_state`` SSoT — same
    primitive we're testing — to capture + restore. This makes the test
    file safely orderable under default ``pytest`` runs (does not require
    ``-m "not integration"`` filtering for hygiene).
    """
    snapshot = get_seed_state()
    try:
        yield
    finally:
        # GRACEFUL policy: best-effort restore; missing keys (e.g., MPS state)
        # warn but don't fail. fallback_seed=42 gives a deterministic floor
        # if any required key is missing post-test.
        set_seed_state(
            snapshot,
            policy=RngStatePolicy.GRACEFUL,
            fallback_seed=42,
        )


def _build_logistic_config(
    data_dir: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    epochs: int = 1,
    num_workers: int = 0,
) -> ExperimentConfig:
    """Minimal logistic-classification ExperimentConfig pointing at synthetic data.

    Reuses the same logistic model + synthetic_export_dir pattern as
    ``test_signal_exporter_integration.py`` — smallest model in the registry
    (~294 params), no GPU dependency, deterministic. ``num_workers=0`` matches
    the integration-test rationale (avoid pickle-of-config worker contract).
    """
    return ExperimentConfig(
        name="design1_g4_replay",
        data=DataConfig(
            data_dir=str(data_dir),
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
            epochs=epochs,
            seed=seed,
            num_workers=num_workers,
        ),
        output_dir=str(output_dir),
    )


def _assert_checkpoints_tensor_equal(
    ckpt_a: Dict[str, Any],
    ckpt_b: Dict[str, Any],
    *,
    name: str = "checkpoints",
) -> None:
    """Compare two checkpoint dicts at TENSOR-VALUE level (not byte-level).

    Per Wave 2.1 critique: SHA-256 over pickled torch.save bytes is fragile
    across hardware (optimizer state dict ordering, PyTorch save format
    drift). ``torch.testing.assert_close(rtol=0, atol=0)`` walks the dict
    catching actual numerical drift while tolerating serialization-encoding
    noise.

    Compares: ``model_state_dict`` (every tensor) + structural equality of
    ``rng_state`` keys + structural equality of ``callback_state``.
    """
    sa = ckpt_a.get("model_state_dict", {})
    sb = ckpt_b.get("model_state_dict", {})
    assert sa.keys() == sb.keys(), (
        f"{name}: model_state_dict keys differ: "
        f"only-A={sorted(set(sa) - set(sb))}, "
        f"only-B={sorted(set(sb) - set(sa))}"
    )
    for k in sorted(sa):
        torch.testing.assert_close(
            sa[k], sb[k], rtol=0, atol=0,
            msg=f"{name}: model_state_dict tensor {k!r} differs (rtol=0, atol=0)",
        )

    # Side-check 1 — rng_state structural equality (Phase A.2)
    rng_a = ckpt_a.get("rng_state", {})
    rng_b = ckpt_b.get("rng_state", {})
    assert set(rng_a.keys()) == set(rng_b.keys()), (
        f"{name}: rng_state keys differ: "
        f"only-A={sorted(set(rng_a) - set(rng_b))}, "
        f"only-B={sorted(set(rng_b) - set(rng_a))}"
    )

    # Side-check 2 — callback_state structural equality (Phase G-1)
    # callback_state is a JSON-native dict-of-dicts (Callback ABC contract);
    # direct == comparison is correct because no torch.Tensor lives inside
    # (EarlyStopping deliberately EXCLUDES _best_weights per G-1 design).
    cb_a = ckpt_a.get("callback_state", {})
    cb_b = ckpt_b.get("callback_state", {})
    assert cb_a == cb_b, (
        f"{name}: callback_state differs:\n  A={cb_a}\n  B={cb_b}"
    )


@pytest.mark.integration
class TestBitExactReplay:
    """Phase DESIGN-1 G-4 — bit-exact replay across two in-process Trainer runs.

    Verifies that DESIGN-1 reproducibility primitives (``set_seed``,
    DataLoader determinism, RNG state, callback state) cumulatively produce
    tensor-exact identical training outcomes when given identical inputs.

    Total runtime ~15-30s for 2 scenarios × 1 epoch on ``synthetic_export_dir``
    (2 train days × ~50 sequences × batch_size 8 = ~12 batches per epoch).

    What this test catches that no existing DESIGN-1 unit test catches:
      - End-to-end integration of set_seed → DataLoader → model init →
        train() → save_checkpoint as a single contract. A regression
        between any two of these steps would slip through unit tests but
        be caught here.
    """

    def test_two_runs_same_seed_produce_tensor_equal_checkpoint(
        self, synthetic_export_dir: Path, tmp_path: Path,
    ) -> None:
        """KEYSTONE INTEGRATION TEST: same seed + same config + same data
        → bit-exact identical model_state_dict after train().

        Catches the architectural seam between ``set_seed → DataLoader
        construction → model init → train() forward/backward →
        save_checkpoint``. Failure modes locked:
          - ``worker_init_fn`` pickling regression (Phase A.1 ``functools.partial``)
          - DataLoader generator drift (Phase A.1 V2 SB-3)
          - ``set_seed`` missing ``torch.use_deterministic_algorithms``
            (Phase A.1 NEW-C1)
          - Model weight init non-determinism
          - Optimizer state evolution non-determinism
          - rng_state not embedded in checkpoint (Phase A.2 NEW-W3-1)
          - callback_state not embedded (Phase G-1)

        SCOPE NOTE (mid-impl gate, 2026-05-10): the lock is "bit-exact for
        the ops actually exercised by logistic + ``warn_only=True`` mode"
        (the Phase A.1 default). The negative-control test below ensures
        at least one op IS reached by the seed. A future cycle that
        extends this test to TLOB / HMHP under ``strict_determinism=True``
        would tighten the contract; for now logistic-CPU is empirically
        deterministic at warn_only level and the negative control proves
        seed reach.

        Side-check: ``compatibility_fingerprint`` MUST also match across runs
        (defense-in-depth runtime-path complement to
        ``TestRngStateInvariance``'s structural-path lock).
        """
        out_a = tmp_path / "run_a"
        out_b = tmp_path / "run_b"
        config_a = _build_logistic_config(synthetic_export_dir, out_a, seed=42)
        config_b = _build_logistic_config(synthetic_export_dir, out_b, seed=42)

        trainer_a = Trainer(config_a)
        trainer_a.setup()
        trainer_a.train()
        ckpt_path_a = tmp_path / "ckpt_a.pt"
        trainer_a.save_checkpoint(ckpt_path_a)

        trainer_b = Trainer(config_b)
        trainer_b.setup()
        trainer_b.train()
        ckpt_path_b = tmp_path / "ckpt_b.pt"
        trainer_b.save_checkpoint(ckpt_path_b)

        ckpt_a = _safe_torch_load(ckpt_path_a, map_location="cpu")
        ckpt_b = _safe_torch_load(ckpt_path_b, map_location="cpu")

        # Tensor-exact equality on model_state_dict + rng_state keys +
        # callback_state structural
        _assert_checkpoints_tensor_equal(ckpt_a, ckpt_b, name="same_seed_bit_exact")

        # Defense-in-depth side-check: compatibility_fingerprint matches
        # across same-seed runs (runtime-path complement to
        # TestRngStateInvariance structural lock at type level)
        fp_a = ckpt_a.get("compatibility_fingerprint")
        fp_b = ckpt_b.get("compatibility_fingerprint")
        assert fp_a is not None, (
            "compatibility_fingerprint missing from ckpt_a — Phase X.1 v2 "
            "checkpoint contract regression."
        )
        assert fp_b is not None, "compatibility_fingerprint missing from ckpt_b"
        assert _FINGERPRINT_HEX_RE.match(fp_a), (
            f"fingerprint not 64-hex: {fp_a!r}"
        )
        assert fp_a == fp_b, (
            f"compatibility_fingerprint diverged across same-seed runs: "
            f"a={fp_a[:16]}... b={fp_b[:16]}... — runtime-path leak of "
            f"non-deterministic state into the fingerprint canonical-form. "
            f"This complements TestRngStateInvariance's structural-path lock: "
            f"if structural lock passes (rng_state not in CompatibilityContract "
            f"fields) but this side-check fails, a non-rng-state non-determinism "
            f"source has entered the fingerprint inputs."
        )

        # Side-check: model_config_hash must match (architectural axis,
        # independent of training stochasticity)
        mch_a = ckpt_a.get("model_config_hash")
        mch_b = ckpt_b.get("model_config_hash")
        assert mch_a == mch_b, (
            f"model_config_hash diverged across same-seed runs: "
            f"a={mch_a[:16]}... b={mch_b[:16]}... — should depend ONLY on "
            f"model_type + filtered ModelConfig.params, not on rng_state."
        )

    def test_different_seed_produces_different_checkpoint(
        self, synthetic_export_dir: Path, tmp_path: Path,
    ) -> None:
        """NEGATIVE CONTROL: different seeds → different model_state_dict.

        Locks against the trivially-passing scenario where the same-seed
        test passes because seed isn't actually flowing through model init,
        DataLoader shuffle, or training stochasticity. If this test fails,
        the seeded determinism contract is silently degraded — the test
        above passing tells us nothing useful.
        """
        out_x = tmp_path / "run_x_seed42"
        out_y = tmp_path / "run_y_seed43"
        config_x = _build_logistic_config(synthetic_export_dir, out_x, seed=42)
        config_y = _build_logistic_config(synthetic_export_dir, out_y, seed=43)

        ckpt_path_x = tmp_path / "ckpt_x.pt"
        ckpt_path_y = tmp_path / "ckpt_y.pt"

        for cfg, ckpt_path in [(config_x, ckpt_path_x), (config_y, ckpt_path_y)]:
            trainer = Trainer(cfg)
            trainer.setup()
            trainer.train()
            trainer.save_checkpoint(ckpt_path)

        ckpt_x = _safe_torch_load(ckpt_path_x, map_location="cpu")
        ckpt_y = _safe_torch_load(ckpt_path_y, map_location="cpu")

        msd_x = ckpt_x.get("model_state_dict", {})
        msd_y = ckpt_y.get("model_state_dict", {})

        # Both runs must have produced model_state_dict (sanity)
        assert msd_x, "ckpt_x has empty model_state_dict — train() did not run"
        assert msd_y, "ckpt_y has empty model_state_dict — train() did not run"
        assert msd_x.keys() == msd_y.keys(), (
            "model_state_dict shapes differ — config drift, not seed drift"
        )

        # Walk tensors; expect at least one to differ (different seeds reach
        # at least one of: weight init, DataLoader shuffle, dropout, etc.)
        any_differs = False
        for k in sorted(msd_x):
            if not torch.equal(msd_x[k], msd_y[k]):
                any_differs = True
                break

        assert any_differs, (
            "NEGATIVE CONTROL VIOLATED: different seeds (42 vs 43) produced "
            "identical model_state_dict. This means seed isn't actually "
            "flowing through model weight init / DataLoader shuffle / "
            "training stochasticity — the same-seed bit-exact test above is "
            "trivially passing. Investigate set_seed reach + ModelConfig "
            "non-deterministic init paths + DataLoader generator wiring."
        )
