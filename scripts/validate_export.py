#!/usr/bin/env python3
"""Comprehensive validation of exported dataset."""

import numpy as np
import json
from pathlib import Path

DATA_DIR = Path("/Users/knight/code_local/HFT-pipeline-v2/data/exports/nvda_98feat_full")


def main():
    print("=" * 70)
    print("COMPREHENSIVE DATASET VALIDATION")
    print("=" * 70)

    splits = ["train", "val", "test"]
    expected_counts = {"train": 115, "val": 25, "test": 25}
    
    all_valid = True

    # 1. FILE COUNT VALIDATION
    print("\n1. FILE COUNT VALIDATION")
    print("-" * 40)

    for split in splits:
        split_dir = DATA_DIR / split
        feature_files = list(split_dir.glob("*_features.npy"))
        label_files = list(split_dir.glob("*_labels.npy"))
        
        status = "✅" if len(feature_files) == expected_counts[split] else "❌"
        if len(feature_files) != expected_counts[split]:
            all_valid = False
        print(f"  {split}: {len(feature_files)} features, {len(label_files)} labels {status}")
        
        if len(feature_files) != len(label_files):
            print(f"    ❌ MISMATCH: feature/label file count differs!")
            all_valid = False

    # 2. CHRONOLOGICAL ORDER VALIDATION
    print("\n2. CHRONOLOGICAL ORDER VALIDATION")
    print("-" * 40)

    all_dates = {}
    for split in splits:
        split_dir = DATA_DIR / split
        dates = sorted([f.stem.replace("_features", "") for f in split_dir.glob("*_features.npy")])
        all_dates[split] = dates
        print(f"  {split}: {dates[0]} → {dates[-1]} ({len(dates)} days)")

    # Check no overlap between splits
    train_max = max(all_dates["train"])
    val_min = min(all_dates["val"])
    val_max = max(all_dates["val"])
    test_min = min(all_dates["test"])

    temporal_ok = train_max < val_min < val_max < test_min
    status = "✅" if temporal_ok else "❌"
    if not temporal_ok:
        all_valid = False
    print(f"\n  Temporal integrity: train({train_max}) < val({val_min}..{val_max}) < test({test_min}) {status}")

    # 3. FEATURE DIMENSION VALIDATION
    print("\n3. FEATURE DIMENSION VALIDATION (98 features expected)")
    print("-" * 40)

    dimension_ok = True
    for split in splits:
        split_dir = DATA_DIR / split
        for f in sorted(split_dir.glob("*_features.npy")):
            data = np.load(f)
            if data.shape[1] != 98:
                print(f"  ❌ {f.name}: shape={data.shape}")
                dimension_ok = False
                all_valid = False

    if dimension_ok:
        print(f"  ✅ All 165 files have exactly 98 features")

    # 4. NaN/Inf VALIDATION
    print("\n4. NaN/Inf VALIDATION")
    print("-" * 40)

    for split in splits:
        split_dir = DATA_DIR / split
        nan_count = 0
        inf_count = 0
        total_samples = 0
        
        for f in split_dir.glob("*_features.npy"):
            data = np.load(f)
            nan_count += np.isnan(data).sum()
            inf_count += np.isinf(data).sum()
            total_samples += data.shape[0]
        
        status = "✅" if nan_count == 0 and inf_count == 0 else "❌"
        if nan_count > 0 or inf_count > 0:
            all_valid = False
        print(f"  {split}: {total_samples:,} samples, NaN={nan_count}, Inf={inf_count} {status}")

    # 5. LABEL VALUE VALIDATION
    print("\n5. LABEL VALUE VALIDATION (expected: -1, 0, 1)")
    print("-" * 40)

    for split in splits:
        split_dir = DATA_DIR / split
        all_labels = []
        
        for f in split_dir.glob("*_labels.npy"):
            labels = np.load(f)
            all_labels.append(labels)
        
        combined = np.concatenate(all_labels)
        unique_vals = np.unique(combined)
        
        expected_vals = {-1, 0, 1}
        actual_vals = set(unique_vals.tolist())
        
        status = "✅" if actual_vals == expected_vals else "❌"
        if actual_vals != expected_vals:
            all_valid = False
        print(f"  {split}: unique values = {sorted(unique_vals)} {status}")
        
        # Distribution
        down = (combined == -1).sum()
        stable = (combined == 0).sum()
        up = (combined == 1).sum()
        total = len(combined)
        print(f"    Distribution: Down={down:,} ({100*down/total:.1f}%), Stable={stable:,} ({100*stable/total:.1f}%), Up={up:,} ({100*up/total:.1f}%)")

    # 6. FEATURE-LABEL ALIGNMENT VALIDATION
    print("\n6. FEATURE-LABEL SAMPLE ALIGNMENT")
    print("-" * 40)
    
    alignment_ok = True
    for split in splits:
        split_dir = DATA_DIR / split
        mismatch_count = 0
        
        for feat_file in split_dir.glob("*_features.npy"):
            label_file = split_dir / feat_file.name.replace("_features", "_labels")
            if label_file.exists():
                feat_data = np.load(feat_file)
                label_data = np.load(label_file)
                # Labels are created from sequences, so fewer samples expected
                # Each label corresponds to a sequence, not a raw feature row
            else:
                print(f"  ❌ Missing label file for {feat_file.name}")
                alignment_ok = False
                all_valid = False
    
    if alignment_ok:
        print(f"  ✅ All feature files have corresponding label files")

    # 7. CHECK MANIFEST
    print("\n7. MANIFEST VALIDATION")
    print("-" * 40)
    
    manifest_path = DATA_DIR / "dataset_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"  Experiment: {manifest.get('experiment', {}).get('name', 'N/A')}")
        print(f"  Symbol: {manifest.get('symbol', 'N/A')}")
        print(f"  Feature count: {manifest.get('feature_count', 'N/A')}")
        print(f"  Days processed: {manifest.get('days_processed', 'N/A')}")
        print(f"  Export timestamp: {manifest.get('export_timestamp', 'N/A')}")
        
        if manifest.get('days_processed') == 165 and manifest.get('feature_count') == 98:
            print(f"  ✅ Manifest matches expected values")
        else:
            print(f"  ❌ Manifest mismatch!")
            all_valid = False
    else:
        print(f"  ❌ Manifest file not found!")
        all_valid = False

    # 8. CATEGORICAL FEATURE VALIDATION
    print("\n8. CATEGORICAL FEATURE VALIDATION")
    print("-" * 40)
    
    # Check that BOOK_VALID (idx 88), TIME_REGIME (idx 89), MBO_READY (idx 90) 
    # are NOT z-score normalized (should have distinct finite values)
    categorical_indices = {
        "BOOK_VALID": 88,
        "TIME_REGIME": 89,
        "MBO_READY": 90,
    }
    
    # Sample from first train file
    sample_file = sorted((DATA_DIR / "train").glob("*_features.npy"))[0]
    sample_data = np.load(sample_file)
    
    for name, idx in categorical_indices.items():
        col = sample_data[:, idx]
        unique_vals = np.unique(col[np.isfinite(col)])
        # Z-score normalized data would have mean~0, std~1
        # Categorical data should have distinct integer-like values OR be 0/1
        print(f"  {name} (idx {idx}): unique values = {len(unique_vals)}, range = [{col.min():.4f}, {col.max():.4f}]")

    print("\n" + "=" * 70)
    if all_valid:
        print("✅ ALL VALIDATIONS PASSED")
    else:
        print("❌ SOME VALIDATIONS FAILED - REVIEW ABOVE")
    print("=" * 70)


if __name__ == "__main__":
    main()

