"""
Tests for _hmhp_collate_fn custom collation.

_hmhp_collate_fn is the custom collate function for HMHP training, handling
dict-label batches ({horizon: tensor}) in both 2-tuple and 3-tuple formats.

Silent corruption risks tested:
- Collate dropping horizon keys → model trains on partial horizons
- Value/dtype corruption through stacking → wrong labels
- Single-item batch edge case

Note: Transform ordering (normalize FIRST, feature-select SECOND) is a
critical pipeline contract documented in dataset.py:1027-1036 but tested
indirectly through the normalization integration tests, not in this file.

Design Principles (hft-rules.md):
    - Contract tests: producer → consumer (Rule 6)
    - Edge cases: single item batch (Rule 6)
"""

import pytest
import torch
import numpy as np

from lobtrainer.data.dataset import _hmhp_collate_fn


# =============================================================================
# Tests for _hmhp_collate_fn
# =============================================================================


class TestHMHPCollateFn:
    """Tests for the custom HMHP dict-label collation function."""

    def test_2tuple_collation_shapes(self):
        """2-tuple batch: (seq, {h: label}) → (seqs [B,T,F], {h: [B]})."""
        batch = [
            (torch.randn(20, 10), {10: torch.tensor(1), 20: torch.tensor(2)}),
            (torch.randn(20, 10), {10: torch.tensor(0), 20: torch.tensor(1)}),
            (torch.randn(20, 10), {10: torch.tensor(2), 20: torch.tensor(0)}),
        ]
        result = _hmhp_collate_fn(batch)

        assert len(result) == 2, f"Expected 2-tuple, got {len(result)}-tuple"
        seqs, labels = result
        assert seqs.shape == (3, 20, 10), f"Sequences shape: {seqs.shape}"
        assert set(labels.keys()) == {10, 20}, f"Label keys: {labels.keys()}"
        assert labels[10].shape == (3,), f"H10 labels shape: {labels[10].shape}"
        assert labels[20].shape == (3,), f"H20 labels shape: {labels[20].shape}"

    def test_3tuple_collation_shapes(self):
        """3-tuple batch: (seq, labels_dict, reg_dict) → 3-tuple with dicts."""
        batch = [
            (
                torch.randn(20, 10),
                {10: torch.tensor(1), 20: torch.tensor(2)},
                {10: torch.tensor(0.5), 20: torch.tensor(1.5)},
            ),
            (
                torch.randn(20, 10),
                {10: torch.tensor(0), 20: torch.tensor(1)},
                {10: torch.tensor(-0.5), 20: torch.tensor(-1.5)},
            ),
        ]
        result = _hmhp_collate_fn(batch)

        assert len(result) == 3, f"Expected 3-tuple, got {len(result)}-tuple"
        seqs, labels, reg_targets = result
        assert seqs.shape == (2, 20, 10)
        assert reg_targets[10].shape == (2,), f"Reg targets shape: {reg_targets[10].shape}"
        assert reg_targets[20].shape == (2,)

    def test_horizon_keys_preserved(self):
        """Output dict has exactly the same horizon keys as input."""
        horizons = {5, 15, 30, 60, 120}
        batch = [
            (torch.randn(20, 10), {h: torch.tensor(i % 3) for h in horizons})
            for i in range(4)
        ]
        _, labels = _hmhp_collate_fn(batch)
        assert set(labels.keys()) == horizons, (
            f"Expected horizon keys {horizons}, got {set(labels.keys())}"
        )

    def test_sequence_values_preserved(self):
        """Collated sequences match original per-item values."""
        torch.manual_seed(42)
        item0_seq = torch.randn(20, 10)
        item1_seq = torch.randn(20, 10)
        batch = [
            (item0_seq, {10: torch.tensor(0)}),
            (item1_seq, {10: torch.tensor(1)}),
        ]
        seqs, _ = _hmhp_collate_fn(batch)
        assert torch.allclose(seqs[0], item0_seq), "Sequence 0 not preserved"
        assert torch.allclose(seqs[1], item1_seq), "Sequence 1 not preserved"

    def test_label_values_preserved(self):
        """Collated labels match original per-item values."""
        batch = [
            (torch.randn(20, 10), {10: torch.tensor(2), 20: torch.tensor(0)}),
            (torch.randn(20, 10), {10: torch.tensor(1), 20: torch.tensor(2)}),
        ]
        _, labels = _hmhp_collate_fn(batch)
        assert labels[10][0].item() == 2
        assert labels[10][1].item() == 1
        assert labels[20][0].item() == 0
        assert labels[20][1].item() == 2

    def test_single_item_batch(self):
        """Single-item batch works (batch_size=1)."""
        batch = [
            (torch.randn(20, 10), {10: torch.tensor(1)}),
        ]
        seqs, labels = _hmhp_collate_fn(batch)
        assert seqs.shape == (1, 20, 10)
        assert labels[10].shape == (1,)

    def test_regression_targets_dtype_preserved(self):
        """Regression targets maintain float dtype through collation."""
        batch = [
            (
                torch.randn(20, 10),
                {10: torch.tensor(0)},
                {10: torch.tensor(3.14, dtype=torch.float32)},
            ),
            (
                torch.randn(20, 10),
                {10: torch.tensor(1)},
                {10: torch.tensor(2.72, dtype=torch.float32)},
            ),
        ]
        _, _, reg = _hmhp_collate_fn(batch)
        assert reg[10].dtype == torch.float32, (
            f"Regression target dtype should be float32, got {reg[10].dtype}"
        )
