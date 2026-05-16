"""Visualization and analysis utilities."""

import os
import sys
# Kaggle notebook path fix
sys.path.insert(0, '/kaggle/working/Degradation_Aware_Image_Restoration')

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import json

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config
from models.architecture import DegradationAwareRestoration
from data.dataset import DegradationPipeline


def visualize_degradation_types(output_dir="./visualizations"):
    """Visualize different degradation types at various severity levels."""
    os.makedirs(output_dir, exist_ok=True)
    
    config = Config()
    pipeline = DegradationPipeline(config)
    
    # Create a test image (natural scene-like)
    np.random.seed(42)
    test_image = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    
    # Add some pattern
    test_image[50:100, 50:100] = [100, 150, 200]
    test_image[100:200, 100:200] = [200, 100, 150]
    
    fig, axes = plt.subplots(
        len(config.DEGRADATION_TYPES) + 1,
        len(config.SEVERITY_LEVELS) + 1,
        figsize=(14, 10)
    )
    
    # Original image
    axes[0, 0].imshow(test_image)
    axes[0, 0].set_title("Original", fontweight='bold')
    axes[0, 0].axis('off')
    
    # Degradations
    for deg_idx, deg_type in enumerate(config.DEGRADATION_TYPES):
        axes[deg_idx + 1, 0].text(0.5, 0.5, deg_type.replace('_', ' ').title(),
                                   ha='center', va='center', fontsize=10, fontweight='bold')
        axes[deg_idx + 1, 0].axis('off')
        
        for sev_idx, severity in enumerate(config.SEVERITY_LEVELS):
            # Reset pipeline to apply specific degradation
            np.random.seed(42 + sev_idx)
            
            degraded = test_image.copy()
            if deg_type == "gaussian_noise":
                degraded, _ = pipeline.apply_gaussian_noise(degraded, severity)
            elif deg_type == "motion_blur":
                degraded, _ = pipeline.apply_motion_blur(degraded, severity)
            elif deg_type == "jpeg_compression":
                degraded, _ = pipeline.apply_jpeg_compression(degraded, severity)
            
            axes[deg_idx + 1, sev_idx + 1].imshow(degraded)
            axes[deg_idx + 1, sev_idx + 1].set_title(f"{severity:.1f}", fontsize=9)
            axes[deg_idx + 1, sev_idx + 1].axis('off')
    
    # Add severity labels to top row
    for sev_idx, severity in enumerate(config.SEVERITY_LEVELS):
        axes[0, sev_idx + 1].text(0.5, 0.5, f"Severity\n{severity:.1f}",
                                   ha='center', va='center', fontsize=9)
        axes[0, sev_idx + 1].axis('off')
    
    plt.suptitle("Degradation Types and Severity Levels", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "degradation_types.png")
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"✓ Degradation visualization saved to {output_path}")
    plt.close()


def plot_training_history(history_file, output_dir="./visualizations"):
    """Plot training history from JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Training History", fontsize=16, fontweight='bold')
    
    # Extract data
    train_epochs = [h['epoch'] for h in history['train']]
    train_loss = [h['loss'] for h in history['train']]
    train_l1 = [h['components']['l1'] for h in history['train']]
    train_perceptual = [h['components']['perceptual'] for h in history['train']]
    train_frequency = [h['components']['frequency'] for h in history['train']]
    
    val_epochs = [h['epoch'] for h in history['val']]
    val_loss = [h['loss'] for h in history['val']]
    val_l1 = [h['components']['l1'] for h in history['val']]
    val_perceptual = [h['components']['perceptual'] for h in history['val']]
    val_frequency = [h['components']['frequency'] for h in history['val']]
    
    # Total loss
    axes[0, 0].plot(train_epochs, train_loss, 'b-', label='Train', linewidth=2)
    if val_epochs:
        axes[0, 0].plot(val_epochs, val_loss, 'r-', label='Val', linewidth=2, marker='o')
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # L1 Loss
    axes[0, 1].plot(train_epochs, train_l1, 'b-', label='Train', linewidth=2)
    if val_epochs:
        axes[0, 1].plot(val_epochs, val_l1, 'r-', label='Val', linewidth=2, marker='o')
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("L1 Loss")
    axes[0, 1].set_title("L1 Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Perceptual Loss
    axes[0, 2].plot(train_epochs, train_perceptual, 'b-', label='Train', linewidth=2)
    if val_epochs:
        axes[0, 2].plot(val_epochs, val_perceptual, 'r-', label='Val', linewidth=2, marker='o')
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Perceptual Loss")
    axes[0, 2].set_title("Perceptual Loss (VGG)")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Frequency Loss
    axes[1, 0].plot(train_epochs, train_frequency, 'b-', label='Train', linewidth=2)
    if val_epochs:
        axes[1, 0].plot(val_epochs, val_frequency, 'r-', label='Val', linewidth=2, marker='o')
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Frequency Loss")
    axes[1, 0].set_title("Frequency Loss (FFT)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss landscape
    axes[1, 1].text(0.5, 0.5, 
                    f"Total epochs: {len(train_epochs)}\n" +
                    f"Val checkpoints: {len(val_epochs)}\n" +
                    f"Best train loss: {min(train_loss):.6f}\n" +
                    f"Best val loss: {min(val_loss):.6f}" if val_loss else "N/A",
                    ha='center', va='center', fontsize=11, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')
    axes[1, 1].set_title("Training Statistics")
    
    # All loss components together
    axes[1, 2].plot(train_epochs, train_loss, 'k-', label='Total', linewidth=2.5)
    axes[1, 2].plot(train_epochs, train_l1, 'b--', label='L1', alpha=0.7)
    axes[1, 2].plot(train_epochs, train_perceptual, 'g--', label='Perceptual', alpha=0.7)
    axes[1, 2].plot(train_epochs, train_frequency, 'r--', label='Frequency', alpha=0.7)
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("Loss")
    axes[1, 2].set_title("Loss Components (Train)")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"✓ Training history saved to {output_path}")
    plt.close()


def visualize_model_architecture(output_dir="./visualizations"):
    """Create ASCII visualization of model architecture."""
    os.makedirs(output_dir, exist_ok=True)
    
    architecture_text = """
╔═══════════════════════════════════════════════════════════════════════════╗
║           DEGRADATION-AWARE IMAGE RESTORATION ARCHITECTURE              ║
╚═══════════════════════════════════════════════════════════════════════════╝

INPUT (256×256×3)
    │
    ├─────────────────────────────────────┐
    │                                     │
    ▼                                     │ (Parallel)
┌───────────────────────────────────┐    │
│    ENCODER (Progressive Down)     │    │
│                                   │    │
│ Conv → (256×256×32)               │    │
│   ↓                               │    │
│ NAFNetBlock ×2                    │    │
│   ↓ Downsample (2×)               │    │
│ NAFNetBlock ×2 → (128×128×64)     │    │
│   ↓ Downsample (2×)               │    │
│ NAFNetBlock ×2 → (64×64×128)      │    │
│   ↓ Downsample (2×)               │    │
│ NAFNetBlock ×2 → (32×32×256)      │    ▼
│   ↓ [BOTTLENECK]                  │  ┌──────────────────────────────┐
└──────────┬──────────────────────┬─┘  │ DEGRADATION ESTIMATOR        │
           │                      │    │                              │
           │              ┌───────┘    │ Conv×3 (stride=2)            │
           │              │            │   ↓                          │
           │              │            │ Global Average Pool          │
           │              │            │   ↓                          │
           │              │            │ ┌─ Type (3-way)             │
           │              │            │ ├─ Severity (5-way)         │
           │              │            │ └─ Embedding (16D)          │
           │              │            │                              │
           │              │            └────────┬───────────────┬─────┘
           │              │                     │               │
           │              │              deg_type severity  deg_embedding
           │              │                     │               │
           ▼              │                     │               │
    ┌──────────────────────────────────────────┘               │
    │                                                           │
    │ ADD DEGRADATION EMBEDDING                                │
    │                                                           │
    ▼                                                           │
┌──────────────────────────────────┐                           │
│   DECODER (Progressive Up)       │                           │
│                                  │                           │
│ (32×32×256) [BOTTLENECK]         │                           │
│   ↓ NAFNetBlock ×2               │                           │
│   ↓ Upsample (2×) + Skip         │                           │
│ (64×64×128)                      │                           │
│   ↓ NAFNetBlock ×2               │                           │
│   ↓ Upsample (2×) + Skip         │                           │
│ (128×128×64)                     │                           │
│   ↓ NAFNetBlock ×2               │                           │
│   ↓ Upsample (2×) + Skip         │                           │
│ (256×256×32)                     │                           │
│   ↓ NAFNetBlock ×2               │                           │
│   ↓ Conv → (256×256×3)           │                           │
└──────────┬───────────────────────┘                           │
           │                                                    │
           ▼                                                    ▼
       RESTORED IMAGE (256×256×3)        deg_type, severity, deg_embedding


KEY COMPONENTS:

1. NAFNetBlock:
   ┌─────────────────────────┐
   │  GroupNorm              │
   │  Depthwise Conv (3×3)   │
   │  Pointwise Conv 1×2     │
   │  GELU                   │
   │  Pointwise Conv ×1      │
   │  Channel Attention      │
   │  + Residual Connection  │
   └─────────────────────────┘

2. Channel Attention:
   Input → AvgPool → FC → ReLU → FC → Sigmoid → Scale

3. Degradation Estimator:
   - Lightweight auxiliary network
   - Predicts degradation type and severity
   - Provides conditioning embedding to decoder

LOSS FUNCTION:

Total Loss = L1_Loss + 0.1×Perceptual_Loss + 0.05×Frequency_Loss 
             + 0.1×DegType_Loss + 0.05×Severity_Loss

Parameters: ~580K (lightweight)
Trainable: Yes

╔═══════════════════════════════════════════════════════════════════════════╗
║ Design Philosophy: Efficient, Interpretable, Degradation-Aware           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""
    
    output_path = os.path.join(output_dir, "architecture.txt")
    with open(output_path, 'w') as f:
        f.write(architecture_text)
    
    print(f"✓ Architecture visualization saved to {output_path}")
    print(architecture_text)


def compare_degradation_and_restoration(degraded_path, restored_path, output_dir="./visualizations"):
    """Visualize side-by-side degradation and restoration."""
    os.makedirs(output_dir, exist_ok=True)
    
    degraded = cv2.imread(str(degraded_path))
    degraded = cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB)
    
    restored = cv2.imread(str(restored_path))
    restored = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(degraded)
    axes[0].set_title("Degraded Input", fontweight='bold', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(restored)
    axes[1].set_title("Restored Output", fontweight='bold', fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "degradation_restoration_comparison.png")
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"✓ Comparison saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualization utilities")
    parser.add_argument(
        "--degradation",
        action="store_true",
        help="Visualize degradation types"
    )
    parser.add_argument(
        "--history",
        type=str,
        help="Path to training history JSON file"
    )
    parser.add_argument(
        "--architecture",
        action="store_true",
        help="Show model architecture"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./visualizations",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    if args.degradation:
        visualize_degradation_types(args.output_dir)
    
    if args.history:
        plot_training_history(args.history, args.output_dir)
    
    if args.architecture:
        visualize_model_architecture(args.output_dir)
    
    if not (args.degradation or args.history or args.architecture):
        # Run all by default
        print("Running all visualizations...\n")
        visualize_model_architecture(args.output_dir)
        print()
        visualize_degradation_types(args.output_dir)
