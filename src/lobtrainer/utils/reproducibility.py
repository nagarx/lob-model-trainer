"""
Reproducibility utilities for deterministic experiments.

Ensures that experiments with the same seed produce identical results across:
- Python random module
- NumPy random number generators
- PyTorch (CPU and CUDA)
- CuDNN (if available)

Design principles (RULE.md §6):
- Same inputs MUST produce identical outputs across runs
- Explicit random seeds for any stochastic components
- Document and enforce floating-point comparison tolerances

Reference:
    PyTorch Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html

Usage:
    >>> from lobtrainer.utils import set_seed
    >>> set_seed(42)  # Call at start of training
"""

import random
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union
import logging
import warnings

import numpy as np
import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Phase DESIGN-1 A.2 — RngStatePolicy + exception types
# =============================================================================


class RngStatePolicy(str, Enum):
    """Policy for ``set_seed_state`` behavior on missing/incomplete state.

    Phase DESIGN-1 A.2 (2026-05-10) per V2 SB-2. Closes the partial-state
    silent-corruption hazard surfaced during Wave 4 adversarial review:
    pre-fix ``set_seed_state`` did direct subscription (``state['python']``)
    and raised ``KeyError`` on missing keys with an unactionable message.
    The enum provides three explicit modes:

    - **STRICT**: raise ``RngStatePolicyError`` with actionable diagnostic
      when any required key is missing. Use for production-gate runs that
      must fail-loud on incomplete state.
    - **GRACEFUL** (DEFAULT): emit ``CheckpointMissingRngStateWarning``
      naming the missing keys + reseed missing components from
      ``fallback_seed`` (typically ``config.train.seed``). Production-safe
      back-compat for pre-A.2 checkpoints lacking ``rng_state``.
    - **IGNORE**: skip restoration entirely (cross-platform CI replays
      where bit-exact resume isn't expected).
    """

    STRICT = "strict"
    GRACEFUL = "graceful"
    IGNORE = "ignore"


class RngStatePolicyError(ValueError):
    """Raised by ``set_seed_state`` when state is incomplete and policy=STRICT.

    Phase DESIGN-1 A.2 (2026-05-10). The error message names the missing
    keys + suggests the GRACEFUL fallback path so operators have an
    actionable remediation.
    """


class CheckpointMissingRngStateWarning(UserWarning):
    """Emitted when ``set_seed_state`` runs in GRACEFUL mode with missing keys
    OR when ``Trainer.load_checkpoint`` reads a pre-A.2 checkpoint lacking
    the ``rng_state`` key.

    Phase DESIGN-1 A.2 (2026-05-10). Mirrors
    ``compatibility.CheckpointMissingFingerprintWarning`` semantics:
    surfaces the gap to ops without raising. Phase X.4 may promote to
    STRICT once all in-flight checkpoints have ``rng_state`` populated.
    """


# =============================================================================
# Core Seed Management
# =============================================================================


def validate_seed(seed: int) -> None:
    """Validate a seed value conforms to NumPy legacy RandomState constraints.

    Phase DESIGN-1 A.4 (2026-05-10): extracted from ``set_seed`` body so that
    callers OTHER than ``set_seed`` can validate seeds at the boundary —
    e.g., ``cv_trainer._build_fold_config`` should fail-loud BEFORE
    materializing a Pydantic ``ExperimentConfig`` with a fold-seed that
    overflows numpy's ``2**32`` ceiling. ``set_seed(seed)`` now delegates
    to this helper (zero behavior change for existing callers).

    Constraints (hft-rules §5 fail-fast + §8 never-silently-drop):
      - Type: must be ``int`` (numpy ``np.random.seed`` accepts only int).
      - Lower bound: ``>= 0`` (numpy accepts only non-negative).
      - Upper bound: ``< 2**32`` (numpy legacy ``RandomState`` silently
        truncates ``seed % 2**32``, causing collisions when sweep configs
        compute ``seed = base + worker_id`` or ``hash(name)`` and the sum
        overflows). Modern ``np.random.default_rng(seed)`` accepts arbitrary
        ints, but ``set_seed`` writes to the legacy global state, so the
        2**32 bound applies pipeline-wide.

    Args:
        seed: Integer seed to validate.

    Raises:
        TypeError: ``seed`` is not an ``int``.
        ValueError: ``seed`` is negative or ``>= 2**32``.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed)}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    if seed >= 2**32:
        raise ValueError(
            f"seed must be < 2**32 (numpy legacy RandomState upper bound), "
            f"got {seed}. NumPy silently truncates seed % 2**32, which causes "
            f"collisions for distinct seeds. Reduce the seed or use "
            f"`np.random.default_rng(seed)` (modern API) at call sites."
        )


def _safe_torch_load(
    path: Union[str, Path],
    *,
    map_location: Optional[Any] = None,
) -> Any:
    """``torch.load`` wrapper with ``weights_only=False`` pinned for numpy-globals safety.

    Phase DESIGN-1 A.4 (2026-05-10): canonical helper for DESIGN-1 era
    checkpoints that embed ``rng_state`` (Phase A.2) — a dict containing
    numpy + python global state via ``get_seed_state()``. PyTorch >=2.6
    default-flipped ``torch.load`` to ``weights_only=True``, which
    rejects numpy globals at deserialize time. This wrapper pins
    ``weights_only=False`` so DESIGN-1 checkpoints load cleanly.

    REUSE-FIRST per hft-rules §0: 4 call sites in lob-model-trainer
    (``Trainer.load_checkpoint:1391``, ``evaluate_model.py:73``, plus
    ``analyze_feature_importance.py``, ``analyze_predictions.py``) are
    past the DRY threshold. Pre-A.4 each site repeated the
    ``weights_only=False`` kwarg explicitly; this helper consolidates.

    Args:
        path: Checkpoint file path.
        map_location: ``torch.load`` ``map_location`` kwarg
            (e.g., ``"cpu"``, ``"cuda:0"``, ``torch.device(...)``). ``None``
            preserves checkpoint's original device assignment.

    Returns:
        The loaded checkpoint object (typically a ``Dict[str, Any]``).
    """
    return torch.load(path, map_location=map_location, weights_only=False)


def set_seed(
    seed: int = 42,
    deterministic_cudnn: bool = True,
    *,
    strict_determinism: bool = False,
) -> None:
    """
    Set random seeds for all random number generators.

    This ensures reproducibility across:
    - Python's random module
    - NumPy's random module
    - PyTorch (CPU operations)
    - PyTorch CUDA (if available)
    - CuDNN (if available and deterministic_cudnn=True)
    - PyTorch global deterministic algorithms (warn_only by default)

    Args:
        seed: Random seed (default: 42). Must satisfy 0 <= seed < 2**32 — numpy
              legacy RandomState silently truncates seed % 2**32 otherwise
              (closes Phase DESIGN-1 NEW-DET-2).
        deterministic_cudnn: If True, configure CuDNN for deterministic behavior.
                            This may impact performance but ensures reproducibility.
                            (hft-rules §7: Determinism is prioritized over speed)
        strict_determinism: Phase DESIGN-1 NEW-C1 keyword-only flag controlling
                            ``torch.use_deterministic_algorithms`` mode.
                            - False (DEFAULT): warn_only=True. Non-deterministic
                              GPU ops (e.g., scatter_add backward, certain
                              reductions) emit a UserWarning but proceed.
                              Matches lob-models test_phase0_forward_pass.py:63
                              pattern. Production-safe.
                            - True: warn_only=False. Non-deterministic ops RAISE
                              RuntimeError. Use for production-gate runs that
                              must hard-fail on any non-deterministic kernel.

    Example:
        >>> set_seed(42)
        >>> torch.rand(3)  # Always produces same values

        >>> # Production-gate run that hard-fails on non-det ops
        >>> set_seed(42, strict_determinism=True)

    Notes:
        - Call this at the start of your training script BEFORE any model/data creation
        - Some operations may still be non-deterministic even with these settings
          (e.g., scatter/gather operations on GPU). With strict_determinism=False
          (default), those emit warnings; with True, they raise.
        - CUBLAS_WORKSPACE_CONFIG env var is set via os.environ.setdefault so
          existing user values are preserved.

    Reference:
        https://pytorch.org/docs/stable/notes/randomness.html
    """
    # Phase DESIGN-1 A.4 (2026-05-10): delegate to validate_seed helper for
    # reuse at non-set_seed call sites (e.g., cv_trainer.py fold-seed
    # computation `seed + fold_idx` should fail-loud BEFORE the model_copy
    # materializes if the sum overflows 2**32). hft-rules §0 reuse-first.
    validate_seed(seed)

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # CuDNN determinism
    if deterministic_cudnn and torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set CUBLAS_WORKSPACE_CONFIG for PyTorch >= 1.8
        # This is needed for some CUDA operations to be deterministic.
        # setdefault preserves user-supplied values.
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    # Phase DESIGN-1 NEW-C1 (2026-05-10): global deterministic-algorithms gate.
    # warn_only=True (default) emits a warning when a non-deterministic op
    # runs, but does NOT raise — matches `lob-models/.../test_phase0_forward_pass.py:63`
    # production-tested pattern. Operators opt-in to strict mode via
    # `set_seed(seed, strict_determinism=True)` for production-gate runs.
    torch.use_deterministic_algorithms(True, warn_only=not strict_determinism)

    logger.debug(
        f"Set seed={seed}, deterministic_cudnn={deterministic_cudnn}, "
        f"strict_determinism={strict_determinism}, "
        f"cuda_available={torch.cuda.is_available()}"
    )


RNG_STATE_SCHEMA_VERSION = 1
"""Schema version for the dict returned by ``get_seed_state``. Phase DESIGN-1
A.2 (2026-05-10). Bumped when the dict shape changes incompatibly. Pre-A.2
checkpoints lack the ``schema_version`` key entirely → treated as schema 0
(legacy: python+numpy+torch only) by ``set_seed_state``."""


_REQUIRED_RNG_KEYS = frozenset({"python", "numpy", "torch"})
"""The 3 keys that MUST be present in any non-empty rng_state dict.
``torch_cuda`` and ``torch_mps`` are device-conditional (best-effort)."""


def get_seed_state() -> Dict[str, Any]:
    """
    Get current state of all random number generators.

    Phase DESIGN-1 A.2 (2026-05-10): returns a schema-versioned dict so
    ``set_seed_state`` can dispatch on shape across versions. Adds
    best-effort MPS RNG capture on Apple Silicon (V2 GAP-1 closure).

    Useful for:
    - Debugging reproducibility issues
    - Saving/restoring RNG state for checkpointing (Trainer.load_checkpoint
      consumes this via Option-C ``_pending_rng_state`` ordering)

    Returns:
        Dict with keys:
        - ``schema_version`` (int): currently 1
        - ``python``: ``random.getstate()`` tuple
        - ``numpy``: ``np.random.get_state()`` tuple
        - ``torch``: ``torch.get_rng_state()`` tensor
        - ``torch_cuda`` (optional): ``torch.cuda.get_rng_state_all()`` list
          of tensors. Present only when ``torch.cuda.is_available()``.
        - ``torch_mps`` (optional): ``torch.mps.get_rng_state()`` tensor.
          Present only when ``torch.backends.mps.is_available()`` AND the
          torch version exposes the MPS RNG API. Capture is best-effort —
          older PyTorch versions silently skip without raising.

    Example:
        >>> state = get_seed_state()
        >>> assert state["schema_version"] == 1
        >>> # ... do some operations ...
        >>> set_seed_state(state)  # Restore to saved state
    """
    state: Dict[str, Any] = {
        "schema_version": RNG_STATE_SCHEMA_VERSION,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()

    # MPS best-effort (V2 GAP-1 / NEW-CI-1 closure). Apple Silicon torch
    # versions ≥ 2.1 expose `torch.mps.get_rng_state`; older versions or
    # non-MPS builds silently skip. Wrap in attribute + capability check
    # to avoid AttributeError on older torch installs.
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            state["torch_mps"] = torch.mps.get_rng_state()
        except (RuntimeError, NotImplementedError, AttributeError):
            # MPS RNG API not available — proceed without raising.
            pass

    return state


def set_seed_state(
    state: Dict[str, Any],
    *,
    policy: RngStatePolicy = RngStatePolicy.GRACEFUL,
    fallback_seed: Optional[int] = None,
) -> None:
    """
    Restore random number generators to a saved state.

    Phase DESIGN-1 A.2 (2026-05-10) per V2 SB-2: replaces the pre-A.2
    direct-subscription path (which raised an unactionable ``KeyError``
    on missing keys, and silently left partial state on partial-write)
    with a policy-driven dispatch. Default ``GRACEFUL`` mirrors the
    Phase X.1 v2 ``CheckpointMissingFingerprintWarning`` discipline.

    Args:
        state: State dict from ``get_seed_state()``. May be incomplete
            (e.g., pre-A.2 checkpoint lacking schema_version + 1+ device
            sub-state). Empty/None state is treated as IGNORE.
        policy: Behavior on missing required keys. See ``RngStatePolicy``.
        fallback_seed: When ``policy=GRACEFUL`` and required keys are
            missing, components without state are reseeded from this
            value via ``set_seed(fallback_seed)``. ``None`` (default)
            means no reseed — the missing component keeps its
            pre-existing in-process RNG state. Production callers (e.g.,
            ``Trainer.load_checkpoint``) should pass
            ``self.config.train.seed`` to guarantee a known starting state.

    Raises:
        RngStatePolicyError: When ``policy=STRICT`` and any required key
            in ``_REQUIRED_RNG_KEYS`` is missing from ``state``.

    Example:
        >>> state = get_seed_state()
        >>> # ... do some operations ...
        >>> set_seed_state(state)  # Restore (default GRACEFUL)

        >>> # Strict mode for production-gate runs
        >>> set_seed_state(state, policy=RngStatePolicy.STRICT)

        >>> # Pre-A.2 checkpoint without schema_version: GRACEFUL warns
        >>> # + reseeds missing components from fallback_seed
        >>> set_seed_state({"python": ..., "torch": ...},
        ...                policy=RngStatePolicy.GRACEFUL,
        ...                fallback_seed=42)
    """
    if policy == RngStatePolicy.IGNORE:
        return  # No-op

    if not state:
        # Empty / None — treat as missing-everything in the policy dispatch.
        state = {}

    missing = _REQUIRED_RNG_KEYS - set(state.keys())

    if missing:
        if policy == RngStatePolicy.STRICT:
            raise RngStatePolicyError(
                f"set_seed_state STRICT mode requires keys "
                f"{sorted(_REQUIRED_RNG_KEYS)}, missing {sorted(missing)}. "
                f"Either set policy=RngStatePolicy.GRACEFUL with "
                f"fallback_seed, or supply a complete state dict from "
                f"get_seed_state()."
            )
        # GRACEFUL: WARN + reseed from fallback_seed when provided.
        warnings.warn(
            f"set_seed_state GRACEFUL: rng_state missing keys "
            f"{sorted(missing)}; "
            + (
                f"reseeding from fallback_seed={fallback_seed}."
                if fallback_seed is not None
                else "no fallback_seed provided — missing components keep "
                "their pre-existing RNG state."
            ),
            CheckpointMissingRngStateWarning,
            stacklevel=2,
        )
        if fallback_seed is not None:
            # set_seed re-seeds python + numpy + torch + (cuda if available);
            # supplied components below will then OVERWRITE these with the
            # checkpoint's state. The fallback is only load-bearing for the
            # MISSING keys.
            set_seed(fallback_seed)

    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if (
        "torch_mps" in state
        and hasattr(torch, "mps")
        and torch.backends.mps.is_available()
    ):
        try:
            torch.mps.set_rng_state(state["torch_mps"])
        except (RuntimeError, NotImplementedError, AttributeError):
            # MPS RNG API not available on this torch — best-effort skip.
            pass


# =============================================================================
# Seed Manager Context
# =============================================================================


@dataclass
class SeedManager:
    """
    Context manager for reproducible code blocks.
    
    Provides a clean way to ensure a code block uses a specific seed
    and optionally restores the original state afterward.
    
    Args:
        seed: Random seed to use within the context
        restore_state: If True, restore original RNG state after exiting
                      (useful for testing, not recommended for training)
    
    Example:
        >>> # Reproducible block that doesn't affect global state
        >>> with SeedManager(42, restore_state=True):
        ...     x = torch.rand(3)  # Always same values
        >>> # After exiting, RNG state is restored
        
        >>> # For training (no state restoration)
        >>> with SeedManager(config.train.seed):
        ...     train_model()
    
    Design note (RULE.md §6):
        - restore_state=True is useful for testing to avoid cross-test contamination
        - restore_state=False (default) is correct for training since we want
          the seed to affect all subsequent operations
    """
    
    seed: int
    restore_state: bool = False
    deterministic_cudnn: bool = True
    
    def __post_init__(self):
        self._saved_state: Optional[Dict[str, Any]] = None
    
    def __enter__(self) -> 'SeedManager':
        if self.restore_state:
            self._saved_state = get_seed_state()
        
        set_seed(self.seed, self.deterministic_cudnn)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.restore_state and self._saved_state is not None:
            set_seed_state(self._saved_state)
        return None  # Don't suppress exceptions


# =============================================================================
# Worker Initialization for DataLoader
# =============================================================================


def worker_init_fn(worker_id: int, base_seed: Optional[int] = None) -> None:
    """
    Initialize random seeds for DataLoader workers.
    
    Each worker should have a different seed to avoid duplicate data augmentations,
    but the seeds should be deterministic for reproducibility.
    
    Args:
        worker_id: Worker ID (0, 1, ..., num_workers-1)
        base_seed: Base seed. If None, uses a fixed default (42).
    
    Usage:
        >>> from functools import partial
        >>> loader = DataLoader(
        ...     dataset,
        ...     num_workers=4,
        ...     worker_init_fn=partial(worker_init_fn, base_seed=42),
        ... )
    
    Reference:
        https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    if base_seed is None:
        base_seed = 42
    
    # Each worker gets a unique but deterministic seed
    worker_seed = base_seed + worker_id
    
    # Set seeds for this worker
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def create_worker_init_fn(base_seed: int):
    """
    Create a worker_init_fn with a fixed base seed.

    Phase DESIGN-1 (2026-05-10): returns ``functools.partial`` instead of a
    nested closure. The closure form is NOT picklable, which breaks
    DataLoader at ``num_workers > 0`` on Python 3.14+ where the default
    multiprocessing start method is ``forkserver`` (workers spawn fresh
    interpreters that pickle their args). ``functools.partial`` over a
    module-level function (``worker_init_fn``) IS picklable.

    Args:
        base_seed: Base seed for worker initialization

    Returns:
        A picklable worker_init_fn callable suitable for DataLoader

    Example:
        >>> loader = DataLoader(
        ...     dataset,
        ...     num_workers=4,
        ...     worker_init_fn=create_worker_init_fn(42),
        ... )
    """
    import functools
    return functools.partial(worker_init_fn, base_seed=base_seed)

