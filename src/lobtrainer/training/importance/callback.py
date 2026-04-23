"""Phase 8C-α Stage C.1 trainer wire-in (2026-04-20 post-audit round-2).

`PermutationImportanceCallback` — invokes `compute_permutation_importance`
at `on_train_end` to produce `outputs/<exp>/feature_importance_v1.json`.
The artifact is then picked up by hft-ops Stage C.3 routing (commit
6db1575 + post-audit 92061ad), content-addressed into
`ledger/feature_importance/{yyyy_mm}/<sha>.json`, and referenced via
`ExperimentRecord.artifacts[]`.

**Why a callback, not inline in `Trainer.train()`?**
- The trainer's existing callback abstraction (`EarlyStoppingCallback`,
  `ModelCheckpointCallback`, etc.) already provides clean isolation of
  orthogonal concerns. Adding importance as another callback reuses the
  pattern + automatically benefits from the callback-ordering invariants
  (runs AFTER best-weights restore, AFTER checkpoint save).
- Future SHAP / IG / attention-map callbacks can be siblings in this
  same `lobtrainer.training.importance` package.

**Why this submodule, not `lobtrainer/training/callbacks.py`?**
- The callback depends on `torch` (for model inference) + on hft_contracts
  (for artifact construction) + on hft_metrics (transitively via
  `compute_permutation_importance`). Co-locating it with the importance
  package keeps all importance code in one namespace — discoverable +
  easy to extend.
- `lobtrainer.training.importance.__init__.py` intentionally does NOT
  re-export this symbol to preserve the `compute_permutation_importance`
  pure-function import path as torch-free. XGBoost/sklearn consumers can
  still `from lobtrainer.training.importance import compute_permutation_importance`
  without pulling torch.

**Dispatch model:** PyTorch-only for Phase 8C-α. XGBoost / sklearn
trainers don't use `Trainer.callbacks`; their own scripts would need
a separate integration. Flagged as Phase 8C-β / 8D scope.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, List, Optional

import numpy as np
import torch

from lobtrainer.config.paths import resolve_labels_config
from lobtrainer.training.callbacks import Callback
from lobtrainer.training.importance.config import (
    ImportanceConfig,
    permutation_importance_enabled,
)
from lobtrainer.training.importance.permutation import (
    compute_permutation_importance,
)

logger = logging.getLogger(__name__)


__all__ = [
    "PermutationImportanceCallback",
    "make_pytorch_predict_fn",
    "make_metric_fn_for_task",
]


# ---------------------------------------------------------------------------
# predict_fn / metric_fn factories (separated for testability + reuse)
# ---------------------------------------------------------------------------


def make_pytorch_predict_fn(
    model: torch.nn.Module,
    device: torch.device,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a numpy-in, numpy-out predict function from a PyTorch model.

    The returned closure:
      - Sets model to eval mode (no dropout / BN update).
      - Transfers input numpy → torch on ``device``.
      - Runs forward pass under ``torch.no_grad()`` (no autograd).
      - Handles tuple-output (HMHP: first element is primary) + dict-
        output (fallback: first value).
      - Returns detached CPU numpy.

    Args:
        model: Torch module with ``model(X) → logits_or_preds`` signature.
        device: Target device (cpu / cuda / mps).

    Returns:
        Callable ``predict_fn(X: ndarray) → ndarray``. Input shape
        matches what the model accepts (typically ``(N, T, F)``).
    """
    def predict_fn(X: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(device)
            out = model(X_tensor)
            if isinstance(out, tuple):
                # HMHP and variants return (primary, aux, confirmation).
                # Use primary for importance — other outputs are secondary
                # heads that don't drive feature importance ranking.
                out = out[0]
            elif isinstance(out, dict):
                # Dict outputs: pick predictions/logits, fall back to
                # first value. Rare in this pipeline.
                out = (
                    out.get("predictions")
                    or out.get("logits")
                    or next(iter(out.values()))
                )
            return out.detach().cpu().numpy()

    return predict_fn


def make_metric_fn_for_task(
    task_type: str,
    primary_horizon_idx: int = 0,
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Build a metric_fn callable for the given task type.

    metric_fn contract: ``metric_fn(preds, y) → float``. Higher is better
    so ``importance = baseline - mean(permuted_null)`` is positive-means-
    better. Caller should negate (wrap) for lower-is-better metrics.

    For multi-horizon outputs (regression with H=[10, 60, 300] or
    classification with per-horizon logits), reduces to the primary
    horizon. Subsequent metric selection is handled at the caller level
    via ``ImportanceConfig.baseline_metric``.

    Args:
        task_type: "regression" or "classification".
        primary_horizon_idx: Index into the per-horizon axis (default 0
            = first / shortest horizon, typically H=10 in this pipeline).

    Returns:
        metric_fn callable.

    Raises:
        ValueError: Unknown task_type.
    """
    if task_type == "regression":
        from hft_metrics.ic import spearman_ic

        def regression_metric(preds: np.ndarray, y: np.ndarray) -> float:
            # Multi-horizon preds/y → reduce to primary horizon.
            if preds.ndim > 1 and preds.shape[-1] > 1:
                preds = preds[:, primary_horizon_idx]
            if y.ndim > 1 and y.shape[-1] > 1:
                y = y[:, primary_horizon_idx]
            preds = preds.ravel()
            y = y.ravel()
            # spearman_ic returns (ic, p_value); take the point estimate.
            ic, _p = spearman_ic(preds, y)
            return float(ic)

        return regression_metric

    if task_type == "classification":
        def classification_metric(preds: np.ndarray, y: np.ndarray) -> float:
            """Multi-horizon classification accuracy on primary horizon.

            Round-3 post-audit Agent-4 C3 + Agent-5 Q6 CRITICAL fix:
            for ``(N, H, C)`` multi-horizon logits, ``argmax(axis=-1)``
            yields ``(N, H)`` — which then needs ``[:, primary_horizon_idx]``
            reduction to ``(N,)`` BEFORE comparing to ``y``. The prior
            code raveled `pred_labels` to `(N*H,)` while `y_arr` was
            reduced to `(N,)` → shape mismatch → silent broadcast error
            caught by the outer try/except, artifact never produced.
            Now indexes pred_labels symmetrically to y.
            """
            preds_arr = np.asarray(preds)
            y_arr = np.asarray(y)

            # Step 1: argmax over CLASS axis (last dim) for multi-class
            # logits; passthrough otherwise.
            if preds_arr.ndim > 1 and preds_arr.shape[-1] > 1:
                pred_labels = preds_arr.argmax(axis=-1)
            else:
                pred_labels = preds_arr.astype(np.int64)

            # Step 2: reduce HORIZON axis symmetrically on BOTH
            # pred_labels and y (if present). This is the fix: pre-fix
            # code only reduced y, causing mismatch on (N, H) preds.
            if pred_labels.ndim > 1 and pred_labels.shape[-1] > 1:
                pred_labels = pred_labels[:, primary_horizon_idx]
            if y_arr.ndim > 1 and y_arr.shape[-1] > 1:
                y_arr = y_arr[:, primary_horizon_idx]

            # Step 3: flatten both to 1-D and compare.
            pred_labels = np.asarray(pred_labels).ravel().astype(np.int64)
            y_labels = y_arr.ravel().astype(np.int64)

            if pred_labels.shape != y_labels.shape:
                raise ValueError(
                    f"classification_metric shape mismatch: "
                    f"pred_labels={pred_labels.shape} vs "
                    f"y_labels={y_labels.shape}. Inputs: "
                    f"preds={preds_arr.shape}, y={y_arr.shape}, "
                    f"primary_horizon_idx={primary_horizon_idx}."
                )
            return float((pred_labels == y_labels).mean())

        return classification_metric

    raise ValueError(
        f"Unknown task_type for metric_fn factory: {task_type!r}. "
        f"Expected 'regression' or 'classification'."
    )


# ---------------------------------------------------------------------------
# Eval-loader → numpy arrays (memory-bounded)
# ---------------------------------------------------------------------------


def _extract_eval_tensors(loader: Any) -> tuple[np.ndarray, np.ndarray]:
    """Stack a DataLoader's batches into dense (X, y) numpy arrays.

    The full eval split is materialized in memory so
    ``compute_permutation_importance`` can slice it (subsample) +
    permute per-feature in-place. Memory cost: O(N × T × F × 4) bytes.

    For 50K samples × 100 timesteps × 98 features × float32 = ~2GB.
    Operators with large eval splits should use ``subsample=5000``
    (default) to keep compute tractable — ``compute_permutation_importance``
    handles the draw internally.

    Accepts batches of shape ``(X, y)`` (standard) or ``(X, y, sw)``
    (sample-weighted training — weights discarded for eval).

    Returns:
        ``(X, y)`` numpy arrays of dtype matching the loader's tensors.
    """
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    for batch in loader:
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            X, y = batch[0], batch[1]
        else:
            raise ValueError(
                f"Unexpected batch shape: {type(batch)} with "
                f"len={len(batch) if hasattr(batch, '__len__') else '?'}. "
                f"Expected (X, y) or (X, y, sample_weights)."
            )
        X_np = X.cpu().numpy() if hasattr(X, "cpu") else np.asarray(X)
        y_np = y.cpu().numpy() if hasattr(y, "cpu") else np.asarray(y)
        X_list.append(X_np)
        y_list.append(y_np)
    if not X_list:
        raise ValueError(
            "Eval loader produced ZERO batches. Cannot compute importance."
        )
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


# ---------------------------------------------------------------------------
# The callback
# ---------------------------------------------------------------------------


class PermutationImportanceCallback(Callback):
    """Trainer callback: on_train_end → compute + save FeatureImportanceArtifact.

    Invoked automatically by `train_from_config` when
    ``ExperimentConfig.importance is not None AND config.importance.enabled``.
    Direct-construct ``Trainer(config, callbacks=[...])`` users must
    append this callback manually; the trainer does NOT auto-wire in
    that path to avoid surprising users with explicit callback lists.

    Flow on ``on_train_end``:
      1. Gate check: ``permutation_importance_enabled(self.config)`` —
         returns early (no-op) when disabled, so operators can leave the
         callback in the list + toggle via config.
      2. Resolve eval loader (``test`` by default, per ``config.eval_split``).
      3. Extract full loader → ``(X, y)`` numpy arrays.
      4. Build ``predict_fn`` via ``make_pytorch_predict_fn``.
      5. Build ``metric_fn`` via ``make_metric_fn_for_task`` (task-dispatch).
      6. Resolve feature metadata (names + indices + feature_set_ref)
         from the trainer's data config + ResolvedFeatureSet cache.
      7. Call ``compute_permutation_importance(...)`` → FeatureImportanceArtifact.
      8. Save via ``artifact.save(output_dir / 'feature_importance_v1.json')``
         — atomic write via hft_contracts.atomic_io SSoT.

    Failure mode: all exceptions are logged + swallowed. Importance is a
    non-critical observation; a failure should NOT kill the training run
    (loss would be irrecoverable — the trained model is valuable even
    without the importance audit). hft-rules §8 compliance: the failure
    is tracked via log.warning + no partial artifact is written.
    """

    def __init__(self, config: ImportanceConfig) -> None:
        super().__init__()
        self.config = config

    def on_train_end(self) -> None:  # noqa: C901 (dispatch complexity intentional)
        """Compute importance on the configured eval split + write the artifact."""
        if not permutation_importance_enabled(self.config):
            logger.debug(
                "PermutationImportanceCallback: gate disabled "
                "(enabled=%s, method=%s); no-op.",
                self.config.enabled, self.config.method,
            )
            return

        try:
            self._compute_and_save()
        except Exception:
            logger.exception(
                "PermutationImportanceCallback: artifact generation "
                "failed. Training run is NOT affected; the artifact "
                "will be missing from this run's output. See traceback "
                "above for root cause."
            )

    # -----------------------------------------------------------------
    # Implementation (kept separate for test-mocking)
    # -----------------------------------------------------------------

    def _compute_and_save(self) -> None:
        from lobtrainer.training.importance import compute_permutation_importance

        trainer = self.trainer
        if trainer is None:
            raise RuntimeError(
                "PermutationImportanceCallback.trainer is None. Callback "
                "must be registered via CallbackList before on_train_end."
            )

        # ---- Resolve eval loader --------------------------------------
        split = self.config.eval_split
        loader = self._resolve_eval_loader(trainer, split)
        if loader is None:
            logger.warning(
                "PermutationImportanceCallback: eval_split=%r loader is "
                "None on trainer. Skipping importance computation. "
                "(Configure `data.test_days` / `data.val_days` or switch "
                "eval_split.)",
                split,
            )
            return

        X_eval, y_eval = _extract_eval_tensors(loader)
        logger.info(
            "PermutationImportanceCallback: extracted %d eval samples "
            "from split=%r; computing importance with n_permutations=%d × "
            "n_seeds=%d × n_features=%d.",
            len(X_eval), split, self.config.n_permutations,
            self.config.n_seeds, X_eval.shape[-1] if X_eval.ndim >= 2 else 1,
        )

        # ---- Build predict_fn + metric_fn -----------------------------
        predict_fn = make_pytorch_predict_fn(trainer.model, trainer.device)
        task_type = self._resolve_task_type(trainer.config)
        # Phase A (2026-04-23) — bug #7b: thread primary_horizon_idx from the
        # canonical ``config.data.labels`` (via ``resolve_labels_config`` helper).
        # Pre-Phase-A, this was hardcoded to 0 — every HMHP-R feature-importance
        # artifact was silently computed against H10 regardless of manifest.
        try:
            primary_idx = (
                resolve_labels_config(trainer.config).primary_horizon_idx or 0
            )
        except AttributeError:
            # Pre-T9 config without LabelsConfig — fall back to safe default.
            primary_idx = 0
        metric_fn = make_metric_fn_for_task(task_type, primary_horizon_idx=primary_idx)

        # ---- Resolve feature metadata ---------------------------------
        feature_names, feature_indices = self._resolve_feature_metadata(
            trainer.config, n_features=X_eval.shape[-1],
        )
        feature_set_ref = self._resolve_feature_set_ref(trainer.config)

        # ---- Compute + save -------------------------------------------
        artifact = compute_permutation_importance(
            X=X_eval,
            y=y_eval,
            feature_names=feature_names,
            feature_indices=feature_indices,
            predict_fn=predict_fn,
            metric_fn=metric_fn,
            config=self.config,
            feature_set_ref=feature_set_ref,
            experiment_id=str(trainer.config.name),
            fingerprint="",  # populated by hft-ops at ledger-register time
            model_type=str(
                getattr(trainer.config.model, "model_type", "unknown")
            ),
        )

        target = trainer.output_dir / "feature_importance_v1.json"
        artifact.save(target)
        logger.info(
            "PermutationImportanceCallback: wrote %d-feature artifact to %s "
            "(content_hash=%s, baseline=%s=%.4f)",
            len(artifact.features),
            target,
            artifact.content_hash()[:12],
            artifact.baseline_metric,
            artifact.baseline_value,
        )

    # -----------------------------------------------------------------
    # Small helpers (keep the flow above readable)
    # -----------------------------------------------------------------

    @staticmethod
    def _resolve_eval_loader(trainer: Any, split: str) -> Optional[Any]:
        """Map ``split`` keyword → the trainer's loader attribute."""
        if split == "test":
            return getattr(trainer, "_test_loader", None)
        if split == "val":
            return getattr(trainer, "_val_loader", None)
        raise ValueError(
            f"PermutationImportanceCallback: unknown eval_split={split!r}. "
            f"Expected 'test' or 'val' (validated at ImportanceConfig "
            f"construction, so reaching here implies state corruption)."
        )

    @staticmethod
    def _resolve_task_type(config: Any) -> str:
        """Map trainer config → ``'regression'`` or ``'classification'``.

        Reads the canonical ``config.data.labels.task`` via
        :func:`resolve_labels_config` (Phase A, 2026-04-23). Falls back to
        a model-type heuristic when the label config is absent or its
        ``task`` is the sentinel ``"auto"``.

        Phase A bug-fix notes:

        * Pre-Phase-A, this read ``config.labels.task_type`` — BOTH wrong:
          the canonical path is ``config.data.labels`` (not ``config.labels``,
          which does not exist on ``ExperimentConfig``), AND the canonical
          attribute on :class:`LabelsConfig` is ``task`` (not ``task_type``,
          per schema.py:263). Both errors resolved by the helper + attribute
          rename.
        * Pre-T9 configs may not have a ``LabelsConfig`` attached (helper
          raises ``AttributeError``). In that case, the legacy model-type
          heuristic is preserved for backward compatibility.
        """
        try:
            labels_cfg = resolve_labels_config(config)
        except AttributeError:
            labels_cfg = None  # Pre-T9 config — use heuristic fallback.
        if labels_cfg is not None:
            task = getattr(labels_cfg, "task", None)
            # LabelsConfig.task = "auto" means "detect from metadata"; don't
            # trust that literal as an authoritative answer here.
            if task and task != "auto":
                return str(task).lower()
        # Legacy / "auto" fallback: HMHP-R / *-regression → regression; else
        # classification.
        model_type = str(
            getattr(getattr(config, "model", None), "model_type", "")
        ).lower()
        if "regression" in model_type or model_type.endswith("-r"):
            return "regression"
        return "classification"

    @staticmethod
    def _resolve_feature_metadata(
        config: Any,
        n_features: int,
    ) -> tuple[List[str], List[int]]:
        """Return (feature_names, feature_indices) aligned with X's
        last-axis ordering.

        Round-3 post-audit Agent-4 C1 CRITICAL fix: the trainer populates
        ``cfg_data._feature_indices_resolved = list(resolved.feature_indices)``
        at `trainer.py:378` — a plain ``List[int]``, NOT a
        ``ResolvedFeatureSet`` object. Reading it as
        ``resolved.feature_names`` always returned empty → every
        experiment silently fell through to synthetic ``feature_i``
        placeholder names. Corrected path:
          1. Read the LIST of indices from `_feature_indices_resolved`.
          2. Derive names from the hft_contracts `FeatureIndex` enum
             (or `ExperimentalFeatureIndex` for indices >= 98).
          3. Fall back to `cfg_data.feature_indices` for ad-hoc
             experiments (no registered FeatureSet); use the same
             enum-based name derivation.
          4. Final fallback (unknown indices): synthetic
             ``feature_i`` names — only for diagnostic continuity.
        """
        data = config.data

        # Prefer the trainer-populated resolved-indices cache
        resolved_indices = getattr(data, "_feature_indices_resolved", None)
        if resolved_indices is not None and len(resolved_indices) == n_features:
            return (
                PermutationImportanceCallback._derive_feature_names(resolved_indices),
                list(resolved_indices),
            )

        # Ad-hoc path: raw feature_indices list on DataConfig
        raw_indices = getattr(data, "feature_indices", None)
        if raw_indices and len(raw_indices) == n_features:
            return (
                PermutationImportanceCallback._derive_feature_names(raw_indices),
                list(raw_indices),
            )

        # Unknown — synthesize placeholders (diagnostic only;
        # downstream Stage C.5 merge cannot reconcile without real names)
        return (
            [f"feature_{i}" for i in range(n_features)],
            list(range(n_features)),
        )

    @staticmethod
    def _derive_feature_names(indices: "Sequence[int]") -> List[str]:
        """Map global feature indices → semantic names via hft_contracts.

        For index in [0, 97]: use `FeatureIndex(i).name` (stable taxonomy).
        For index in [98, 147]: use `ExperimentalFeatureIndex(i).name`.
        For unknown / out-of-range: fall back to `feature_{i}` synthetic.

        Uses enum reverse-lookup — O(1) per index, no re-import cost
        after first call.
        """
        names: List[str] = []
        try:
            from hft_contracts import (
                FeatureIndex,
                ExperimentalFeatureIndex,
            )
        except ImportError:
            # hft_contracts unavailable — fall back to synthetic
            return [f"feature_{i}" for i in indices]

        stable_by_value = {f.value: f.name for f in FeatureIndex}
        try:
            experimental_by_value = {
                f.value: f.name for f in ExperimentalFeatureIndex
            }
        except Exception:
            experimental_by_value = {}

        for i in indices:
            if i in stable_by_value:
                names.append(stable_by_value[i].lower())
            elif i in experimental_by_value:
                names.append(experimental_by_value[i].lower())
            else:
                names.append(f"feature_{i}")
        return names

    @staticmethod
    def _resolve_feature_set_ref(config: Any) -> Optional[dict]:
        """Extract ``{name, content_hash}`` from the trainer's resolver
        cache, or None for ad-hoc / preset-based experiments.

        Round-3 post-audit Agent-4 C2 CRITICAL fix: the correct cache
        attribute is ``_feature_set_ref_resolved`` (set at
        `trainer.py:379-381` as a ``Tuple[str, str]`` of (name, hash)),
        NOT ``_feature_indices_resolved`` (which is a `List[int]`).
        The prior code read the indices list and did attribute-access
        on it, always returning None → every artifact opted out of
        Stage C.5 feedback-merge silently. Corrected: read the correct
        tuple attribute + unpack.

        Returning None is an explicit opt-out from Stage C.5 feedback-
        merge eligibility — emits WARN at artifact construction (see
        `FeatureImportanceArtifact.__post_init__`).
        """
        ref = getattr(config.data, "_feature_set_ref_resolved", None)
        if ref is None:
            return None
        # Expected shape: (name: str, content_hash: str) tuple
        try:
            name, content_hash = ref
        except (TypeError, ValueError):
            return None
        if not name or not content_hash:
            return None
        return {"name": str(name), "content_hash": str(content_hash)}
