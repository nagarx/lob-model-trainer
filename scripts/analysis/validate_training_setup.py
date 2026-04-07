#!/usr/bin/env python3
"""
Validation script for HMHP Triple Barrier training setup.

This script performs comprehensive validation of the training pipeline
before starting actual training to catch configuration issues early.

Validates:
1. Data export integrity and label encoding
2. Configuration consistency (horizons, features, etc.)
3. Model instantiation and forward pass
4. Label encoding behavior (no spurious +1 shift for Triple Barrier)

Usage:
    python scripts/validate_training_setup.py configs/experiments/nvda_hmhp_triple_barrier_v1.yaml
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch


def validate_export_metadata(data_dir: Path) -> Dict[str, Any]:
    """Validate exported data metadata."""
    print("\n" + "="*60)
    print("1. VALIDATING EXPORT METADATA")
    print("="*60)
    
    train_dir = data_dir / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    
    # Find first metadata file
    meta_files = list(train_dir.glob("*_metadata.json"))
    if not meta_files:
        raise FileNotFoundError(f"No metadata files found in {train_dir}")
    
    with open(meta_files[0]) as f:
        meta = json.load(f)
    
    print(f"✓ Found metadata: {meta_files[0].name}")
    
    # Check labeling strategy
    labeling_info = meta.get('labeling', {})
    strategy = labeling_info.get('strategy', 'unknown')
    print(f"✓ Labeling strategy: {strategy}")
    
    if strategy != 'triple_barrier':
        print(f"⚠ WARNING: Expected 'triple_barrier', got '{strategy}'")
    
    # Check horizons
    horizons = labeling_info.get('horizons', [])
    print(f"✓ Horizons: {horizons}")
    
    # Check label encoding
    encoding = meta.get('label_encoding', {})
    print(f"✓ Label encoding format: {encoding.get('format', 'unknown')}")
    print(f"✓ Label encoding note: {encoding.get('note', 'N/A')}")
    
    # Check feature count
    features_info = meta.get('features', {})
    feature_count = features_info.get('count', 'unknown')
    print(f"✓ Feature count: {feature_count}")
    
    return {
        'strategy': strategy,
        'horizons': horizons,
        'feature_count': feature_count,
        'label_format': encoding.get('format'),
    }


def validate_label_values(data_dir: Path, expected_horizons: list) -> None:
    """Validate that label values are correct for Triple Barrier."""
    print("\n" + "="*60)
    print("2. VALIDATING LABEL VALUES")
    print("="*60)
    
    train_dir = data_dir / "train"
    label_files = list(train_dir.glob("*_labels.npy"))
    if not label_files:
        raise FileNotFoundError("No label files found")
    
    # Check first file
    labels = np.load(label_files[0])
    print(f"✓ Loaded labels from: {label_files[0].name}")
    print(f"✓ Labels shape: {labels.shape}")
    print(f"✓ Labels dtype: {labels.dtype}")
    
    # Check unique values
    unique_values = np.unique(labels)
    print(f"✓ Unique label values: {unique_values}")
    
    # For Triple Barrier, labels should be {0, 1, 2}
    expected_values = {0, 1, 2}
    actual_values = set(unique_values.tolist())
    
    if actual_values == expected_values:
        print("✓ Label values are correct for Triple Barrier: {0=SL, 1=TO, 2=PT}")
    elif actual_values == {-1, 0, 1}:
        print("✗ ERROR: Labels are in TLOB format {-1, 0, 1}, not Triple Barrier!")
        print("  This will cause incorrect label encoding. Check your export.")
        raise ValueError("Label format mismatch")
    else:
        print(f"⚠ WARNING: Unexpected label values: {actual_values}")
    
    # Check multi-horizon shape
    if labels.ndim == 2:
        num_horizons = labels.shape[1]
        print(f"✓ Multi-horizon labels: {num_horizons} horizons")
        
        if len(expected_horizons) != num_horizons:
            print(f"⚠ WARNING: Expected {len(expected_horizons)} horizons, found {num_horizons}")
    else:
        print(f"✗ ERROR: Expected 2D labels [N, num_horizons], got {labels.ndim}D")


def validate_config_consistency(config_path: Path, export_meta: Dict) -> None:
    """Validate that config matches export metadata."""
    print("\n" + "="*60)
    print("3. VALIDATING CONFIG CONSISTENCY")
    print("="*60)
    
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print(f"✓ Loaded config: {config_path.name}")
    
    # Check labeling strategy
    config_strategy = config.get('data', {}).get('labeling_strategy', 'unknown')
    export_strategy = export_meta.get('strategy', 'unknown')
    
    if config_strategy == export_strategy:
        print(f"✓ Labeling strategy matches: {config_strategy}")
    else:
        print(f"✗ ERROR: Strategy mismatch! Config: {config_strategy}, Export: {export_strategy}")
        raise ValueError("Strategy mismatch")
    
    # Check horizons
    config_horizons = config.get('model', {}).get('hmhp_horizons', [])
    export_horizons = export_meta.get('horizons', [])
    
    if config_horizons == export_horizons:
        print(f"✓ Horizons match: {config_horizons}")
    else:
        print(f"✗ ERROR: Horizons mismatch!")
        print(f"  Config: {config_horizons}")
        print(f"  Export: {export_horizons}")
        raise ValueError("Horizons mismatch")
    
    # Check feature count
    config_features = config.get('data', {}).get('feature_count', 0)
    model_input = config.get('model', {}).get('input_size', 0)
    
    if config_features == model_input:
        print(f"✓ Feature count consistent: {config_features}")
    else:
        print(f"✗ ERROR: Feature count mismatch!")
        print(f"  data.feature_count: {config_features}")
        print(f"  model.input_size: {model_input}")
        raise ValueError("Feature count mismatch")
    
    return config


def validate_model_instantiation(config: Dict) -> None:
    """Validate that model can be instantiated and run forward pass."""
    print("\n" + "="*60)
    print("4. VALIDATING MODEL INSTANTIATION")
    print("="*60)
    
    # Import model factory
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from lobtrainer.models import create_model
    from lobtrainer.config import ModelConfig
    
    model_config = config.get('model', {})
    
    # Create ModelConfig from dict
    mc = ModelConfig(
        model_type=model_config.get('model_type', 'hmhp'),
        input_size=model_config.get('input_size', 98),
        num_classes=model_config.get('num_classes', 3),
        dropout=model_config.get('dropout', 0.1),
        hmhp_horizons=model_config.get('hmhp_horizons', [50, 100, 200]),
        hmhp_cascade_mode=model_config.get('hmhp_cascade_mode', 'full'),
        hmhp_state_fusion=model_config.get('hmhp_state_fusion', 'gate'),
        hmhp_encoder_type=model_config.get('hmhp_encoder_type', 'tlob'),
        hmhp_encoder_hidden_dim=model_config.get('hmhp_encoder_hidden_dim', 64),
        hmhp_num_encoder_layers=model_config.get('hmhp_num_encoder_layers', 2),
        hmhp_decoder_hidden_dim=model_config.get('hmhp_decoder_hidden_dim', 32),
        hmhp_state_dim=model_config.get('hmhp_state_dim', 32),
        hmhp_use_confirmation=model_config.get('hmhp_use_confirmation', True),
        hmhp_loss_weights=model_config.get('hmhp_loss_weights'),
    )
    
    print(f"✓ Model type: {mc.model_type}")
    print(f"✓ Horizons: {mc.hmhp_horizons}")
    
    # Create model
    model = create_model(mc)
    print(f"✓ Model created: {type(model).__name__}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Parameters: {num_params:,} total, {trainable:,} trainable")
    
    # Test forward pass
    batch_size = 4
    seq_len = config.get('data', {}).get('sequence', {}).get('window_size', 100)
    num_features = mc.input_size
    
    dummy_input = torch.randn(batch_size, seq_len, num_features)
    print(f"✓ Test input shape: {dummy_input.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✓ Forward pass successful!")
    print(f"  - horizon_logits: {list(output.horizon_logits.keys())}")
    print(f"  - final_logits shape: {output.logits.shape}")
    print(f"  - agreement_ratio shape: {output.agreement.shape}")


def validate_dataset_label_encoding(data_dir: Path, config: Dict) -> None:
    """Validate that dataset correctly handles label encoding."""
    print("\n" + "="*60)
    print("5. VALIDATING DATASET LABEL ENCODING")
    print("="*60)
    
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from lobtrainer.data import load_split_data, LOBSequenceDataset
    
    # Load a small subset
    days = load_split_data(data_dir, "train", validate=True)[:1]  # Just first day
    print(f"✓ Loaded 1 day for testing")
    
    # Get raw label values from file
    raw_labels = days[0].labels
    raw_unique = np.unique(raw_labels)
    print(f"✓ Raw label unique values: {raw_unique}")
    
    # Create dataset with Triple Barrier strategy
    strategy = config.get('data', {}).get('labeling_strategy', 'triple_barrier')
    dataset = LOBSequenceDataset(
        days,
        horizon_idx=None,
        return_labels_as_dict=True,
        labeling_strategy=strategy,
    )
    
    # Get first sample
    seq, labels_dict = dataset[0]
    print(f"✓ Dataset sample retrieved")
    print(f"  - Sequence shape: {seq.shape}")
    print(f"  - Labels keys: {list(labels_dict.keys())}")
    
    # Check label values after dataset processing
    for horizon, label in labels_dict.items():
        label_val = label.item()
        print(f"  - H{horizon} label: {label_val}")
        
        if label_val < 0 or label_val > 2:
            print(f"✗ ERROR: Label {label_val} is outside expected range [0, 2]!")
            raise ValueError("Invalid label value after encoding")
    
    print("✓ Label encoding is correct (no spurious shift for Triple Barrier)")


def main():
    parser = argparse.ArgumentParser(description="Validate HMHP Triple Barrier training setup")
    parser.add_argument("config", type=str, help="Path to experiment config YAML")
    parser.add_argument("--data-dir", type=str, default=None, 
                       help="Override data directory (default: from config)")
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    # Load config to get data_dir
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    data_dir = Path(args.data_dir or config.get('data', {}).get('data_dir', ''))
    if not data_dir.is_absolute():
        # Resolve relative to lob-model-trainer
        data_dir = Path(__file__).parent.parent / data_dir
    
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)
    
    print("="*60)
    print("HMHP TRIPLE BARRIER TRAINING VALIDATION")
    print("="*60)
    print(f"Config: {config_path}")
    print(f"Data:   {data_dir}")
    
    try:
        # Run validations
        export_meta = validate_export_metadata(data_dir)
        validate_label_values(data_dir, export_meta.get('horizons', []))
        config = validate_config_consistency(config_path, export_meta)
        validate_model_instantiation(config)
        validate_dataset_label_encoding(data_dir, config)
        
        print("\n" + "="*60)
        print("✓✓✓ ALL VALIDATIONS PASSED ✓✓✓")
        print("="*60)
        print("\nYou can now start training with:")
        print(f"  python scripts/train.py {config_path}")
        
    except Exception as e:
        print(f"\n✗ VALIDATION FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
