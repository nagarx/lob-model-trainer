"""Tests for the Phase 3c point-return-DA tripwire producer (point_return_da)."""

import json

import numpy as np
import pytest

from lobtrainer.training.point_return_da import (
    POINT_RETURN_DA_KEYS,
    compute_point_return_da_scalars,
)


def _make_export(tmp_path, day_fps, horizons, k, *, exported=True, split="test"):
    """Create a minimal {split}/ export: forward_prices + metadata + horizons."""
    max_h = max(horizons)
    n_cols = k + max_h + 1
    split_dir = tmp_path / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for stem, fp in day_fps.items():
        assert fp.shape[1] == n_cols, f"fp cols {fp.shape[1]} != expected {n_cols}"
        np.save(split_dir / f"{stem}_forward_prices.npy", fp.astype(np.float64))
        metadata = {
            "forward_prices": {
                "exported": exported,
                "smoothing_window_offset": k,
                "max_horizon": max_h,
                "n_columns": n_cols,
            }
        }
        (split_dir / f"{stem}_metadata.json").write_text(json.dumps(metadata))
        (split_dir / f"{stem}_horizons.json").write_text(
            json.dumps({"horizons": horizons})
        )
    return tmp_path


def test_happy_path_da_and_n(tmp_path):
    # k=1, horizon=1: point = (fp[:,2] - fp[:,1]) / fp[:,1] * 1e4
    day1 = np.array([[99., 100., 101.],    # +100 bps (+)
                     [99., 100.,  99.],    # -100 bps (-)
                     [99., 100., 100.5]],  # +50 bps  (+)
                    dtype=np.float64)
    day2 = np.array([[99., 100., 100.5],   # +50 (+)
                     [99., 100.,  99.5]],  # -50 (-)
                    dtype=np.float64)
    _make_export(tmp_path, {"2025-01-01": day1, "2025-01-02": day2}, horizons=[1], k=1)
    # point signs (sorted-day concat): +,-,+,+,-  ; preds: +,+,+,+,-
    # hits: 1,0,1,1,1 -> da=0.8, n=5
    y_pred = np.array([0.5, 0.3, 0.2, 0.1, -0.4])
    out = compute_point_return_da_scalars(tmp_path, y_pred, primary_horizon_idx=0)
    assert out is not None
    assert set(out) == set(POINT_RETURN_DA_KEYS)
    assert out["point_return_da"] == pytest.approx(0.8)
    assert out["point_return_n"] == 5.0
    assert -1.0 <= out["point_return_rho1"] <= 1.0

def test_keys_are_unprefixed(tmp_path):
    """Both producer paths prepend 'test_' — the helper must NOT."""
    day1 = np.array([[99., 100., 101.]], dtype=np.float64)
    _make_export(tmp_path, {"2025-01-01": day1}, horizons=[1], k=1)
    out = compute_point_return_da_scalars(tmp_path, np.array([0.5]), primary_horizon_idx=0)
    assert out is not None
    for key in out:
        assert not key.startswith("test_"), f"{key} must be unprefixed"

def test_horizon_idx_selection(tmp_path):
    # horizons=[1,2], idx=1 -> horizon 2: point = (fp[:,3] - fp[:,1]) / fp[:,1] * 1e4
    day = np.array([[99., 100., 100.5, 101.],   # h2: (101-100)/100 -> +100 (+)
                    [99., 100., 100.5,  99.]],  # h2: (99-100)/100  -> -100 (-)
                   dtype=np.float64)
    _make_export(tmp_path, {"2025-01-01": day}, horizons=[1, 2], k=1)
    y_pred = np.array([0.5, 0.5])  # +,+ vs point +,- -> hits 1,0 -> da=0.5
    out = compute_point_return_da_scalars(tmp_path, y_pred, primary_horizon_idx=1)
    assert out is not None
    assert out["point_return_da"] == pytest.approx(0.5)
    assert out["point_return_n"] == 2.0

def test_none_idx_defaults_to_first_horizon(tmp_path):
    day1 = np.array([[99., 100., 101.]], dtype=np.float64)  # point + ; pred + -> hit
    _make_export(tmp_path, {"2025-01-01": day1}, horizons=[1], k=1)
    out = compute_point_return_da_scalars(tmp_path, np.array([0.5]), primary_horizon_idx=None)
    assert out is not None
    assert out["point_return_da"] == pytest.approx(1.0)

def test_skip_when_no_horizons_json(tmp_path):
    day1 = np.array([[99., 100., 101.]], dtype=np.float64)
    _make_export(tmp_path, {"2025-01-01": day1}, horizons=[1], k=1)
    (tmp_path / "test" / "2025-01-01_horizons.json").unlink()
    out = compute_point_return_da_scalars(tmp_path, np.array([0.5]), primary_horizon_idx=0)
    assert out is None

def test_skip_when_forward_prices_not_exported(tmp_path):
    day1 = np.array([[99., 100., 101.]], dtype=np.float64)
    _make_export(tmp_path, {"2025-01-01": day1}, horizons=[1], k=1, exported=False)
    out = compute_point_return_da_scalars(tmp_path, np.array([0.5]), primary_horizon_idx=0)
    assert out is None

def test_skip_when_primary_horizon_idx_out_of_range(tmp_path):
    day1 = np.array([[99., 100., 101.]], dtype=np.float64)
    _make_export(tmp_path, {"2025-01-01": day1}, horizons=[1], k=1)
    out = compute_point_return_da_scalars(tmp_path, np.array([0.5]), primary_horizon_idx=5)
    assert out is None

def test_skip_when_empty_predictions(tmp_path):
    day1 = np.array([[99., 100., 101.]], dtype=np.float64)
    _make_export(tmp_path, {"2025-01-01": day1}, horizons=[1], k=1)
    out = compute_point_return_da_scalars(tmp_path, np.array([]), primary_horizon_idx=0)
    assert out is None

def test_raises_on_alignment_mismatch(tmp_path):
    # 1 row on disk, but 4 predictions -> misalignment -> ValueError (fail-loud)
    day1 = np.array([[99., 100., 101.]], dtype=np.float64)
    _make_export(tmp_path, {"2025-01-01": day1}, horizons=[1], k=1)
    with pytest.raises(ValueError, match="alignment mismatch"):
        compute_point_return_da_scalars(
            tmp_path, np.array([0.1, 0.2, 0.3, 0.4]), primary_horizon_idx=0
        )
