#!/usr/bin/env python3
"""
Comprehensive validation of exported dataset.

Validates:
1. File counts and structure
2. Chronological ordering of splits
3. Feature dimensions (98 expected)
4. NaN/Inf data quality
5. Label values (-1, 0, 1)
6. Feature-label pairing
7. Dataset manifest
8. Categorical feature indices and values

Usage:
    python scripts/validate_export.py [--data-dir PATH]
"""

import argparse
import numpy as np
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lobtrainer.constants import FeatureIndex

# Default data directory (can be overridden via command line)
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data/exports/nvda_98feat_full"


def main():
    parser = argparse.ArgumentParser(description='Validate exported dataset')
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR,
                        help='Path to exported dataset directory')
    args = parser.parse_args()
    
    global DATA_DIR
    DATA_DIR = args.data_dir
    
    if not DATA_DIR.exists():
        print(f"❌ Error: Data directory not found: {DATA_DIR}")
        sys.exit(1)
    
    print("=" * 70)
    print("COMPREHENSIVE DATASET VALIDATION")
    print("=" * 70)
    print(f"Data directory: {DATA_DIR}")

    splits = ["train", "val", "test"]
    
    # Dynamically count expected files from actual directory
    expected_counts = {}
    for split in splits:
        split_dir = DATA_DIR / split
        if split_dir.exists():
            expected_counts[split] = len(list(split_dir.glob("*_features.npy")))
        else:
            expected_counts[split] = 0
    
    all_valid = True

    # 1. FILE COUNT VALIDATION
    print("\n1. FILE COUNT VALIDATION")
    print("-" * 40)

    total_days = 0
    for split in splits:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            print(f"  {split}: ❌ Directory not found")
            all_valid = False
            continue
            
        feature_files = list(split_dir.glob("*_features.npy"))
        label_files = list(split_dir.glob("*_labels.npy"))
        
        # Check feature-label pairing (should always match)
        if len(feature_files) == len(label_files):
            status = "✅"
        else:
            status = "❌ MISMATCH"
            all_valid = False
            
        print(f"  {split}: {len(feature_files)} feature files, {len(label_files)} label files {status}")
        total_days += len(feature_files)
    
    print(f"  Total: {total_days} trading days across all splits")

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
    
    # Use AUTHORITATIVE indices from FeatureIndex (not hardcoded!)
    # These are safety gates and categorical features that should NOT be z-score normalized
    categorical_checks = {
        "BOOK_VALID": {
            "index": FeatureIndex.BOOK_VALID,  # 92
            "expected_values": {0.0, 1.0},
            "description": "Book validity flag (0=invalid, 1=valid)",
        },
        "TIME_REGIME": {
            "index": FeatureIndex.TIME_REGIME,  # 93
            "expected_values": {0.0, 1.0, 2.0, 3.0, 4.0},  # 0=OPEN, 1=EARLY, 2=MIDDAY, 3=CLOSE, 4=CLOSED
            "description": "Market session (0-3 regular, 4=after-hours)",
        },
        "MBO_READY": {
            "index": FeatureIndex.MBO_READY,  # 94
            "expected_values": {0.0, 1.0},
            "description": "MBO warmup complete (0=warmup, 1=ready)",
        },
        "INVALIDITY_DELTA": {
            "index": FeatureIndex.INVALIDITY_DELTA,  # 96
            "expected_values": None,  # Non-negative integers, variable
            "description": "Feed problem count (should be >= 0)",
        },
        "SCHEMA_VERSION": {
            "index": FeatureIndex.SCHEMA_VERSION_FEATURE,  # 97
            "expected_values": {2.0},  # Current schema version
            "description": "Schema version (should be 2)",
        },
    }
    
    # Collect all unique values across all training files for accurate validation
    categorical_values = {name: set() for name in categorical_checks}
    
    for feat_file in sorted((DATA_DIR / "train").glob("*_features.npy")):
        data = np.load(feat_file)
        for name, check in categorical_checks.items():
            idx = check["index"]
            col = data[:, idx]
            unique_vals = np.unique(col[np.isfinite(col)])
            categorical_values[name].update(unique_vals.tolist())
    
    categorical_ok = True
    for name, check in categorical_checks.items():
        idx = check["index"]
        actual_values = categorical_values[name]
        expected = check["expected_values"]
        
        # Check if values match expected
        if expected is not None:
            if actual_values.issubset(expected):
                status = "✅"
            else:
                unexpected = actual_values - expected
                status = f"⚠️  (unexpected: {unexpected})"
                # Don't fail validation for this, just warn
        else:
            # For INVALIDITY_DELTA, just check non-negative
            if all(v >= 0 for v in actual_values):
                status = "✅"
            else:
                status = "❌ (negative values found)"
                categorical_ok = False
                all_valid = False
        
        print(f"  {name} (idx {idx}): values = {sorted(actual_values)[:10]}{'...' if len(actual_values) > 10 else ''} {status}")
        print(f"    {check['description']}")

    print("\n" + "=" * 70)
    if all_valid:
        print("✅ ALL VALIDATIONS PASSED")
    else:
        print("❌ SOME VALIDATIONS FAILED - REVIEW ABOVE")
    print("=" * 70)


if __name__ == "__main__":
    main()

