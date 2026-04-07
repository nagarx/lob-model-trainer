#!/usr/bin/env python3
"""Analyze model predictions to detect majority class collapse."""

import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
# Also add lob-models to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'lob-models' / 'src'))

from lobtrainer.data import HybridNormalizer, HybridNormalizationStats
from lobmodels.config import HMHPConfig
from lobmodels.models.hmhp import HierarchicalMultiHorizonPredictor

def main():
    print("="*70)
    print("VERIFYING MODEL PREDICTIONS (CHECKPOINT ANALYSIS)")
    print("="*70)

    # Load the best checkpoint
    checkpoint_dir = Path("outputs/experiments/nvda_hmhp_triple_barrier_calibrated/checkpoints")
    best_checkpoint = checkpoint_dir / "best.pt"

    if not best_checkpoint.exists():
        print(f"Checkpoint not found: {best_checkpoint}")
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        print(f"Available: {[c.name for c in checkpoints]}")
        return

    print(f"Loading checkpoint: {best_checkpoint}")
    checkpoint = torch.load(best_checkpoint, map_location='cpu', weights_only=False)
    
    # Create config and model
    config = HMHPConfig(
        num_features=98,
        sequence_length=100,
        num_classes=3,
        horizons=[50, 100, 200],
    )
    model = HierarchicalMultiHorizonPredictor(config)
    # Use strict=False in case model architecture evolved slightly
    result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if result.missing_keys:
        print(f"Warning: Missing keys: {result.missing_keys[:5]}...")
    if result.unexpected_keys:
        print(f"Warning: Unexpected keys: {result.unexpected_keys[:5]}...")
    model.eval()
    
    print(f"\nModel loaded: {sum(p.numel() for p in model.parameters()):,} params")
    
    # Load validation data
    data_dir = Path("../data/exports/nvda_11month_triple_barrier_calibrated")
    stats_path = data_dir / "hybrid_normalization_stats.json"
    stats = HybridNormalizationStats.load(stats_path)
    normalizer = HybridNormalizer(stats)
    
    # Load first validation day
    val_dir = data_dir / "val"
    first_val_file = sorted(val_dir.glob("*_sequences.npy"))[0]
    first_val_labels = sorted(val_dir.glob("*_labels.npy"))[0]
    
    sequences = np.load(first_val_file)[:500]
    labels = np.load(first_val_labels)[:500]
    
    print(f"\nLoaded {len(sequences)} validation samples for analysis")
    
    # Normalize and predict
    normalized = np.array([normalizer(seq) for seq in sequences])
    x = torch.FloatTensor(normalized)
    
    with torch.no_grad():
        outputs = model(x)
    
    # Debug output structure
    print(f"\nOutput type: {type(outputs)}")
    if isinstance(outputs, dict):
        print(f"Output keys: {outputs.keys()}")
    elif isinstance(outputs, tuple):
        print(f"Output tuple length: {len(outputs)}")
        for i, o in enumerate(outputs):
            print(f"  [{i}]: type={type(o)}, shape={o.shape if hasattr(o, 'shape') else 'N/A'}")
    
    print("\n" + "="*70)
    print("PREDICTION DISTRIBUTION ANALYSIS")
    print("="*70)
    
    # Handle different output formats
    # HMHPOutput is a named tuple with: per_horizon_logits, per_horizon_confidence, ensemble_logits, etc.
    if hasattr(outputs, 'per_horizon_logits'):
        horizon_outputs = outputs.per_horizon_logits
    elif isinstance(outputs, tuple) and len(outputs) > 0 and isinstance(outputs[0], dict):
        # First element is per_horizon_logits dict
        horizon_outputs = outputs[0]
    elif isinstance(outputs, dict):
        horizon_outputs = outputs
    else:
        print(f"Unknown output format: {type(outputs)}")
        return
    
    for horizon in [50, 100, 200]:
        if horizon not in horizon_outputs:
            print(f"\nHorizon {horizon}: Not found in outputs")
            continue
            
        logits = horizon_outputs[horizon]
        probs = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1).numpy()
        
        h_idx = [50, 100, 200].index(horizon)
        true_labels = labels[:, h_idx]
        
        print(f"\nHorizon {horizon}:")
        print(f"  TRUE class distribution:")
        for c, name in enumerate(['StopLoss', 'Timeout', 'ProfitTarget']):
            count = (true_labels == c).sum()
            pct = count / len(true_labels) * 100
            print(f"    {name:12s}: {count:4d} ({pct:5.1f}%)")
        
        print(f"  PREDICTED class distribution:")
        for c, name in enumerate(['StopLoss', 'Timeout', 'ProfitTarget']):
            count = (predictions == c).sum()
            pct = count / len(predictions) * 100
            print(f"    {name:12s}: {count:4d} ({pct:5.1f}%)")
        
        mean_probs = probs.mean(dim=0).numpy()
        print(f"  Average prediction probabilities:")
        for c, name in enumerate(['StopLoss', 'Timeout', 'ProfitTarget']):
            print(f"    P({name:12s}): {mean_probs[c]:.4f}")
        
        correct = (predictions == true_labels).sum()
        accuracy = correct / len(true_labels) * 100
        print(f"  Accuracy: {accuracy:.1f}%")
        
        print(f"  Per-class recall:")
        for c, name in enumerate(['StopLoss', 'Timeout', 'ProfitTarget']):
            mask = true_labels == c
            if mask.sum() > 0:
                recall = (predictions[mask] == c).mean() * 100
                print(f"    {name:12s}: {recall:5.1f}% ({mask.sum()} samples)")
            else:
                print(f"    {name:12s}: N/A (0 samples)")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
If PREDICTED distribution shows ~100% Timeout while TRUE has minority classes,
the model has collapsed to predicting the majority class only.

This means:
- The "91%+ accuracy" is ILLUSORY - it's just matching the Timeout rate
- StopLoss and ProfitTarget recall will be ~0%
- The model has learned NOTHING useful about minority classes

FIX: Re-export data with smaller barriers to get balanced classes.
""")

if __name__ == "__main__":
    main()
