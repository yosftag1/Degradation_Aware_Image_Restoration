"""Training script for degradation-aware image restoration model."""

import os
import sys
# Kaggle notebook path fix
sys.path.insert(0, '/kaggle/working/Degradation_Aware_Image_Restoration')

import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import json
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config, DebugConfig
from models.architecture import DegradationAwareRestoration
from losses.loss_functions import RestorationLoss
from data.dataset import create_div2k_dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_training(config, debug=False):
    """Setup training environment."""
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = DegradationAwareRestoration(config).to(device)
    print(f"Model created with {count_parameters(model):,} parameters")
    
    # Create loss function
    loss_fn = RestorationLoss(config, device=device)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    warm_restart_period = max(1, config.EPOCHS - config.WARMUP_EPOCHS)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=warm_restart_period,
        T_mult=1,
        eta_min=1e-6,
    )
    
    return model, loss_fn, optimizer, scheduler, device


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_validation_samples(batch, restored, config, epoch):
    """Save a few validation samples (degraded | restored | clean)."""
    max_samples = max(1, config.VAL_SAMPLE_COUNT)
    degraded = batch['degraded'][:max_samples].detach().cpu()
    clean = batch['clean'][:max_samples].detach().cpu()
    restored = restored[:max_samples].detach().cpu()

    out_dir = os.path.join(config.OUTPUT_DIR, "val_samples", f"epoch_{epoch:03d}")
    os.makedirs(out_dir, exist_ok=True)

    for idx in range(degraded.size(0)):
        triplet = torch.stack([degraded[idx], restored[idx], clean[idx]], dim=0)
        grid = make_grid(triplet, nrow=3, padding=2)
        grid = grid.clamp(0, 1)
        grid_img = (grid * 255).byte().permute(1, 2, 0).numpy()

        pil_img = Image.fromarray(grid_img)
        draw = ImageDraw.Draw(pil_img)
        labels = ["degraded", "restored", "clean"]
        cell_w = degraded.shape[-1]
        padding = 2
        for col, label in enumerate(labels):
            x = padding + col * (cell_w + padding) + 4
            y = 4
            draw.text((x + 1, y + 1), label, fill=(0, 0, 0))
            draw.text((x, y), label, fill=(255, 255, 255))

        save_path = os.path.join(out_dir, f"sample_{idx:02d}.png")
        pil_img.save(save_path)


def train_epoch(model, train_loader, loss_fn, optimizer, device, config, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    loss_components = {
        'l1': 0.0, 'perceptual': 0.0, 'frequency': 0.0,
        'deg_type': 0.0, 'severity': 0.0
    }
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        degraded = batch['degraded'].to(device)
        clean = batch['clean'].to(device)
        deg_type_target = batch['deg_type'].to(device)
        severity_target = batch['severity'].to(device)
        
        # Forward pass
        restored, deg_type, severity, deg_embedding = model(degraded)
        
        # Compute loss
        total_batch_loss, loss_dict = loss_fn(
            restored, clean,
            deg_type, deg_type_target,
            severity, severity_target,
            deg_embedding
        )
        
        # Backward pass
        optimizer.zero_grad()
        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update metrics
        total_loss += total_batch_loss.item()
        for key in loss_components:
            loss_components[key] += loss_dict[key]
        
        # Log every LOG_INTERVAL batches
        if (batch_idx + 1) % config.LOG_INTERVAL == 0:
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    # Average metrics
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches
    
    return avg_loss, loss_components


@torch.no_grad()
def validate(model, val_loader, loss_fn, device, config, epoch=None):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    loss_components = {
        'l1': 0.0, 'perceptual': 0.0, 'frequency': 0.0,
        'deg_type': 0.0, 'severity': 0.0
    }

    save_samples = (
        epoch is not None
        and config.VAL_SAMPLE_INTERVAL > 0
        and epoch % config.VAL_SAMPLE_INTERVAL == 0
    )
    samples_saved = False
    
    for batch in tqdm(val_loader, desc="Validation"):
        degraded = batch['degraded'].to(device)
        clean = batch['clean'].to(device)
        deg_type_target = batch['deg_type'].to(device)
        severity_target = batch['severity'].to(device)
        
        # Forward pass
        restored, deg_type, severity, deg_embedding = model(degraded)
        
        # Compute loss
        total_batch_loss, loss_dict = loss_fn(
            restored, clean,
            deg_type, deg_type_target,
            severity, severity_target,
            deg_embedding
        )

        if save_samples and not samples_saved:
            save_validation_samples(batch, restored, config, epoch)
            samples_saved = True
        
        total_loss += total_batch_loss.item()
        for key in loss_components:
            loss_components[key] += loss_dict[key]
    
    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches
    
    return avg_loss, loss_components


def save_checkpoint(model, optimizer, scheduler, epoch, loss, config, is_best=False):
    """Save model checkpoint."""
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    if is_best:
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, "model_best.pt")
    else:
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"model_epoch_{epoch:03d}.pt")
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")


def train(config, debug=False, resume_from=None):
    """Main training loop."""
    set_seed(config.SEED)
    # Setup
    model, loss_fn, optimizer, scheduler, device = setup_training(config, debug)
    
    # Load datasets
    print("Loading datasets...")
    train_root = getattr(config, "TRAIN_DATASET_PATH", None) or config.DATASET_PATH
    val_root = getattr(config, "VAL_DATASET_PATH", None) or config.DATASET_PATH
    train_dataset = create_div2k_dataset(train_root, config, is_train=True)
    val_dataset = create_div2k_dataset(val_root, config, is_train=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    
    print(f"Train set: {len(train_dataset)} images")
    print(f"Val set: {len(val_dataset)} images")
    
    # Resume from checkpoint if provided
    start_epoch = 1
    best_loss = float('inf')
    
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('loss', float('inf'))
    
    # Training loop
    history = {'train': [], 'val': []}
    
    for epoch in range(start_epoch, config.EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config.EPOCHS}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_losses = train_epoch(
            model, train_loader, loss_fn, optimizer, device, config, epoch
        )
        
        # Validate every N epochs (configurable) or at epoch end
        if epoch % max(1, config.VAL_EPOCH_INTERVAL) == 0 or epoch == config.EPOCHS:
            val_loss, val_losses = validate(model, val_loader, loss_fn, device, config, epoch=epoch)
            
            print(f"\nTrain Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
            print("\nTrain loss components:")
            for key, val in train_losses.items():
                print(f"  {key}: {val:.6f}")
            print("\nVal loss components:")
            for key, val in val_losses.items():
                print(f"  {key}: {val:.6f}")
            
            history['val'].append({
                'epoch': epoch,
                'loss': val_loss,
                'components': val_losses
            })
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, epoch, val_loss, config, is_best=True)
        
        history['train'].append({
            'epoch': epoch,
            'loss': train_loss,
            'components': train_losses
        })
        
        # Save checkpoint
        if epoch % config.CHECKPOINT_EVERY_N_EPOCHS == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss, config, is_best=False)
        
        # Step scheduler
        scheduler.step()
    
    # Save training history
    os.makedirs(config.LOG_DIR, exist_ok=True)
    history_path = os.path.join(config.LOG_DIR, f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining completed! Best model saved.")
    print(f"History saved to {history_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train degradation-aware restoration model")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with smaller dataset")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--data-root", type=str, default=None, help="Override dataset root directory")
    parser.add_argument("--train-root", type=str, default=None, help="Training dataset root directory")
    parser.add_argument("--val-root", type=str, default=None, help="Validation dataset root directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Override checkpoint directory")
    parser.add_argument("--log-dir", type=str, default=None, help="Override log directory")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers")
    
    args = parser.parse_args()
    
    config = DebugConfig() if args.debug else Config()
    if args.data_root:
        config.DATASET_PATH = args.data_root
    if args.train_root:
        config.TRAIN_DATASET_PATH = args.train_root
    if args.val_root:
        config.VAL_DATASET_PATH = args.val_root
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    if args.checkpoint_dir:
        config.CHECKPOINT_DIR = args.checkpoint_dir
    if args.log_dir:
        config.LOG_DIR = args.log_dir
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.num_workers is not None:
        config.NUM_WORKERS = args.num_workers

    train(config, debug=args.debug, resume_from=args.resume)
