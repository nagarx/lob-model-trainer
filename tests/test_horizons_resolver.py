"""Unit tests for lobtrainer.data.horizons_resolver (Audit 2026-05-27 Batch 4).

Tests all 6 error branches in resolve_horizons_from_export():
missing dir, no files, missing key, non-list, non-int, empty list,
cross-day drift, and the happy path with consistent horizons.
"""

import json

import pytest

from lobtrainer.data.horizons_resolver import resolve_horizons_from_export


def _write_horizons_json(split_dir, day_name, horizons):
    """Helper: write a *_horizons.json file for one day."""
    path = split_dir / f"{day_name}_horizons.json"
    path.write_text(json.dumps({"horizons": horizons}))
    return path


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

class TestHappyPath:

    def test_single_day(self, tmp_path):
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        _write_horizons_json(train_dir, "2025-02-03", [10, 60, 300])

        result = resolve_horizons_from_export(tmp_path, split="train")
        assert result == (10, 60, 300)
        assert isinstance(result, tuple)

    def test_multiple_consistent_days(self, tmp_path):
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        for day in ["2025-02-03", "2025-02-04", "2025-02-05"]:
            _write_horizons_json(train_dir, day, [10, 60, 300])

        result = resolve_horizons_from_export(tmp_path, split="train")
        assert result == (10, 60, 300)

    def test_val_split(self, tmp_path):
        val_dir = tmp_path / "val"
        val_dir.mkdir()
        _write_horizons_json(val_dir, "2025-10-01", [10, 60, 300])

        result = resolve_horizons_from_export(tmp_path, split="val")
        assert result == (10, 60, 300)

    def test_single_horizon(self, tmp_path):
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        _write_horizons_json(train_dir, "2025-02-03", [10])

        result = resolve_horizons_from_export(tmp_path)
        assert result == (10,)


# ---------------------------------------------------------------------------
# Error branches
# ---------------------------------------------------------------------------

class TestErrorBranches:

    def test_missing_split_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="split directory does not exist"):
            resolve_horizons_from_export(tmp_path, split="train")

    def test_no_horizons_files(self, tmp_path):
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        # write a non-horizons file to verify the glob is specific
        (train_dir / "2025-02-03_metadata.json").write_text("{}")

        with pytest.raises(FileNotFoundError, match="no \\*_horizons.json files"):
            resolve_horizons_from_export(tmp_path, split="train")

    def test_missing_horizons_key(self, tmp_path):
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        path = train_dir / "2025-02-03_horizons.json"
        path.write_text(json.dumps({"other_key": [10, 60]}))

        with pytest.raises(ValueError, match="missing 'horizons' key"):
            resolve_horizons_from_export(tmp_path, split="train")

    def test_horizons_not_a_list(self, tmp_path):
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        path = train_dir / "2025-02-03_horizons.json"
        path.write_text(json.dumps({"horizons": "not_a_list"}))

        with pytest.raises(ValueError, match="must be a list"):
            resolve_horizons_from_export(tmp_path, split="train")

    def test_horizons_contain_non_integers(self, tmp_path):
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        path = train_dir / "2025-02-03_horizons.json"
        path.write_text(json.dumps({"horizons": [10, 60.5, 300]}))

        with pytest.raises(ValueError, match="must contain integers"):
            resolve_horizons_from_export(tmp_path, split="train")

    def test_empty_horizons_list(self, tmp_path):
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        path = train_dir / "2025-02-03_horizons.json"
        path.write_text(json.dumps({"horizons": []}))

        with pytest.raises(ValueError, match="list is empty"):
            resolve_horizons_from_export(tmp_path, split="train")

    def test_cross_day_drift(self, tmp_path):
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        _write_horizons_json(train_dir, "2025-02-03", [10, 60, 300])
        _write_horizons_json(train_dir, "2025-02-04", [10, 20, 50])

        with pytest.raises(ValueError, match="Horizons drift"):
            resolve_horizons_from_export(tmp_path, split="train")
