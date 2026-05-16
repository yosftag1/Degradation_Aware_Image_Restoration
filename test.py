"""Quick start and testing utilities."""

import os
import sys
import torch
import torch.nn as nn
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config
from models.architecture import DegradationAwareRestoration
from data.dataset import DegradationPipeline, ImageRestorationDataset
from losses.loss_functions import RestorationLoss


def test_model_forward_pass():
    """Test model with dummy data."""
    print("Testing model forward pass...")
    
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DegradationAwareRestoration(config).to(device)
    model.eval()
    
    # Dummy input
    x = torch.randn(2, 3, 256, 256).to(device)
    
    with torch.no_grad():
        restored, deg_type, severity, deg_embedding = model(x)
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Restored shape: {restored.shape}")
    print(f"✓ Deg type logits: {deg_type.shape}")
    print(f"✓ Severity logits: {severity.shape}")
    print(f"✓ Deg embedding: {deg_embedding.shape}")
    print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()


def test_degradation_pipeline():
    """Test degradation pipeline."""
    print("Testing degradation pipeline...")
    
    config = Config()
    pipeline = DegradationPipeline(config)
    
    # Create dummy image
    image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    for i in range(3):
        degraded, deg_type, severity = pipeline.apply_degradation(image)
        
        deg_name = config.DEGRADATION_TYPES[deg_type]
        severity_val = pipeline.severity_levels[severity]
        
        print(f"✓ Test {i+1}: {deg_name} (severity={severity_val:.1f})")
    
    print()


def test_loss_functions():
    """Test loss functions."""
    print("Testing loss functions...")
    
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    loss_fn = RestorationLoss(config, device=device)
    
    # Dummy data
    B, C, H, W = 2, 3, 256, 256
    restored = torch.randn(B, C, H, W).clamp(0, 1).to(device)
    clean = torch.randn(B, C, H, W).clamp(0, 1).to(device)
    deg_type = torch.randn(B, 3, device=device)
    deg_type_target = torch.randint(0, 3, (B,), device=device)
    severity = torch.randn(B, 5, device=device)
    severity_target = torch.randint(0, 5, (B,), device=device)
    
    total_loss, loss_dict = loss_fn(
        restored, clean, deg_type, deg_type_target,
        severity, severity_target
    )
    
    print(f"✓ Total loss: {total_loss.item():.6f}")
    for key, val in loss_dict.items():
        print(f"  - {key}: {val:.6f}")
    
    print()


def test_dataset_creation(data_dir="./data/DIV2K"):
    """Test dataset creation (requires DIV2K to be downloaded)."""
    print("Testing dataset creation...")
    
    import os
    from pathlib import Path
    
    config = Config()
    
    # Check if DIV2K exists
    train_dir = os.path.join(data_dir, "DIV2K_train_HR")
    
    if not os.path.exists(train_dir):
        print(f"✗ DIV2K dataset not found at {data_dir}")
        print(f"  Run: python download_dataset.py --dataset-dir {data_dir}")
        return
    
    try:
        from data.dataset import create_div2k_dataset
        
        dataset = create_div2k_dataset(data_dir, config, is_train=True)
        print(f"✓ Created training dataset with {len(dataset)} images")
        
        # Test one sample
        sample = dataset[0]
        print(f"✓ Sample shapes:")
        print(f"  - Degraded: {sample['degraded'].shape}")
        print(f"  - Clean: {sample['clean'].shape}")
        print(f"  - Deg type: {sample['deg_type'].item()} ({config.DEGRADATION_TYPES[sample['deg_type'].item()]})")
        print(f"  - Severity: {sample['severity'].item()}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()


def test_model_training_step():
    """Test a single training step."""
    print("Testing model training step...")
    
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DegradationAwareRestoration(config).to(device)
    loss_fn = RestorationLoss(config, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Dummy data
    B, C, H, W = 2, 3, 256, 256
    degraded = torch.randn(B, C, H, W).clamp(0, 1).to(device)
    clean = torch.randn(B, C, H, W).clamp(0, 1).to(device)
    deg_type_target = torch.randint(0, 3, (B,), device=device)
    severity_target = torch.randint(0, 5, (B,), device=device)
    
    # Forward pass
    restored, deg_type, severity, deg_embedding = model(degraded)
    
    # Compute loss
    total_loss, loss_dict = loss_fn(
        restored, clean, deg_type, deg_type_target,
        severity, severity_target, deg_embedding
    )
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print(f"✓ Training step completed")
    print(f"✓ Loss: {total_loss.item():.6f}")
    print()


def run_all_tests(test_data_dir="./data/DIV2K"):
    """Run all tests."""
    print("="*70)
    print("RUNNING COMPREHENSIVE TESTS")
    print("="*70)
    print()
    
    test_model_forward_pass()
    test_degradation_pipeline()
    test_loss_functions()
    test_model_training_step()
    test_dataset_creation(test_data_dir)
    
    print("="*70)
    print("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print()
    print("Next steps:")
    print("1. Download dataset: python download_dataset.py")
    print("2. Train model: python train.py --debug  (or full: python train.py)")
    print("3. Visualize: python visualize.py --degradation --architecture")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick start tests")
    parser.add_argument("--test", type=str, choices=[
        "model", "degradation", "loss", "dataset", "training", "all"
    ], default="all", help="Which test to run")
    parser.add_argument("--data-dir", type=str, default="./data/DIV2K",
                       help="DIV2K dataset directory")
    
    args = parser.parse_args()
    
    if args.test == "model" or args.test == "all":
        test_model_forward_pass()
    
    if args.test == "degradation" or args.test == "all":
        test_degradation_pipeline()
    
    if args.test == "loss" or args.test == "all":
        test_loss_functions()
    
    if args.test == "training" or args.test == "all":
        test_model_training_step()
    
    if args.test == "dataset" or args.test == "all":
        test_dataset_creation(args.data_dir)
    
    if args.test == "all":
        print("\n" + "="*70)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*70)
