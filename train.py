"""Training script for degradation-aware image restoration model."""

import os
import sys
# Kaggle notebook path fix
sys.path.insert(0, '/kaggle/working/Degradation_Aware_Image_Restoration')

import random
import subprocess
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime
import numpy as np
import cv2

try:
    import pynvml
    pynvml.nvmlInit()
    _HAS_PYNVML = True
except Exception:
    _HAS_PYNVML = False

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config, DebugConfig
from models.architecture import DegradationAwareRestoration
from losses.loss_functions import RestorationLoss
from data.dataset import create_div2k_dataset, DegradationPipeline


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap_model(model):
    """Return underlying model if wrapped in DataParallel."""
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def get_gpu_stats(use_smi_fallback=False):
    """Collect compute utilization (%) and VRAM usage for all CUDA GPUs."""
    if not torch.cuda.is_available():
        return []

    num_gpus = torch.cuda.device_count()
    smi_map = {}

    if _HAS_PYNVML:
        for i in range(num_gpus):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                smi_map[i] = (int(util), mem_info.used / (1024 ** 3), mem_info.total / (1024 ** 3))
            except Exception:
                pass
    elif use_smi_fallback:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=2.0,
            )
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 4:
                    idx = int(parts[0])
                    smi_map[idx] = (int(parts[1]), float(parts[2]) / 1024.0, float(parts[3]) / 1024.0)
        except Exception:
            pass

    stats = []
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / (1024 ** 3)
        alloc_gb = torch.cuda.memory_allocated(i) / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved(i) / (1024 ** 3)
        peak_gb = torch.cuda.max_memory_allocated(i) / (1024 ** 3)
        util_pct, smi_used_gb, _ = smi_map.get(i, (None, reserved_gb, total_gb))
        used_gb = max(reserved_gb, smi_used_gb)
        mem_pct = (used_gb / total_gb * 100.0) if total_gb > 0 else 0.0
        stats.append({
            "index": i,
            "name": props.name,
            "util_pct": util_pct,
            "alloc_gb": alloc_gb,
            "used_gb": used_gb,
            "peak_gb": peak_gb,
            "total_gb": total_gb,
            "mem_pct": mem_pct,
        })
    return stats


def print_gpu_usage(stage_label="GPU Status"):
    """Print a detailed table of GPU compute utilization and VRAM usage across all GPUs."""
    stats = get_gpu_stats(use_smi_fallback=True)
    if not stats:
        print(f"[{stage_label}] Running on CPU (no CUDA GPUs detected).")
        return

    print(f"\n┌─ {stage_label} {'─' * max(1, 68 - len(stage_label))}┐")
    for s in stats:
        util_str = f"{s['util_pct']:3d}%" if s["util_pct"] is not None else "N/A "
        print(
            f"│ GPU {s['index']} ({s['name'][:18]:<18}) | "
            f"Compute Util: {util_str} | "
            f"VRAM: {s['used_gb']:.2f}/{s['total_gb']:.1f} GB ({s['mem_pct']:4.1f}%) | "
            f"PeakTorch: {s['peak_gb']:.2f} GB │"
        )
    print(f"└{'─' * 72}┘")


def format_gpu_postfix():
    """Compact string for live tqdm progress bar showing all GPUs."""
    stats = get_gpu_stats(use_smi_fallback=False)
    if not stats:
        return "CPU"
    parts = []
    for s in stats:
        util_str = f"{s['util_pct']}%" if s["util_pct"] is not None else f"{s['mem_pct']:.0f}%mem"
        parts.append(f"G{s['index']}:{util_str}({s['used_gb']:.1f}/{s['total_gb']:.0f}G)")
    return " ".join(parts)


def setup_training(config, debug=False):
    """Setup training environment with multi-GPU DataParallel and AMP support."""
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    use_multi_gpu = getattr(config, "MULTI_GPU", True) and num_gpus > 1
    use_amp = getattr(config, "USE_AMP", True) and (device.type == "cuda")

    # If only 1 GPU is present and user didn't explicitly override batch size, scale neatly
    if num_gpus == 1 and config.BATCH_SIZE >= 16 and not debug:
        config.BATCH_SIZE = 8

    print(f"Using device: {device} (detected {num_gpus} CUDA GPU(s))")
    
    # Create model
    model = DegradationAwareRestoration(config).to(device)
    print(f"Model created with {count_parameters(model):,} parameters")

    if use_multi_gpu:
        device_ids = list(range(num_gpus))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        print(
            f"✓ Multi-GPU DataParallel ENABLED across {num_gpus} GPUs {device_ids} "
            f"(total batch_size={config.BATCH_SIZE}, {config.BATCH_SIZE // num_gpus} per GPU)"
        )
    else:
        print(f"Single-device mode (batch_size={config.BATCH_SIZE})")

    print(
        f"Bridge mode: {config.BRIDGE_TYPE} | "
        f"steps={config.BRIDGE_STEPS} | "
        f"sigma_max={config.BRIDGE_SIGMA_MAX} | "
        f"ot_ode={config.BRIDGE_OT_ODE} | "
        f"AMP (FP16 Tensor Cores)={'ENABLED' if use_amp else 'DISABLED'}"
    )
    
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

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None
    print_gpu_usage("Pre-Training GPU & VRAM Check")
    
    return model, loss_fn, optimizer, scheduler, scaler, device


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _load_clean_image(image_path, config):
    """Load and center-crop an image for consistent evaluation."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    if h >= config.PATCH_SIZE and w >= config.PATCH_SIZE:
        y = (h - config.PATCH_SIZE) // 2
        x = (w - config.PATCH_SIZE) // 2
        return image[y:y + config.PATCH_SIZE, x:x + config.PATCH_SIZE]
    return cv2.resize(image, (config.PATCH_SIZE, config.PATCH_SIZE))


def _render_single_image_grid(clean_image, deg_rows, severities, title_text):
    """Build a labeled composite RGB grid for one validation image across all degradations and severities."""
    h, w = clean_image.shape[:2]
    num_rows = len(deg_rows)
    num_sev = len(severities)
    num_cols = 1 + 2 * num_sev  # Clean + (Degraded, Restored) per severity

    label_w = 180
    header_h = 64
    pad = 6
    group_gap = 10  # Extra spacing before each (Degraded, Restored) severity pair

    # Compute x offsets for each column
    col_x = []
    cur_x = label_w + pad
    for col_idx in range(num_cols):
        if col_idx > 0 and (col_idx - 1) % 2 == 0:
            cur_x += group_gap
        col_x.append(cur_x)
        cur_x += w + pad

    canvas_w = cur_x + pad
    canvas_h = header_h + num_rows * (h + pad) + pad
    canvas = np.full((canvas_h, canvas_w, 3), 24, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    # Title banner
    cv2.putText(canvas, title_text, (14, 24), font, 0.62, (255, 220, 120), 2, cv2.LINE_AA)

    # Column headers
    col_titles = ["Clean (GT)"]
    for s_idx, sev in enumerate(severities, start=1):
        col_titles.append(f"S{s_idx} ({sev:.2f}) Degraded")
        col_titles.append(f"S{s_idx} ({sev:.2f}) Restored")

    for col_idx, c_title in enumerate(col_titles):
        cx = col_x[col_idx]
        color = (160, 240, 160) if ("Restored" in c_title or col_idx == 0) else (230, 230, 230)
        cv2.putText(canvas, c_title, (cx + 6, header_h - 12), font, 0.48, color, 1, cv2.LINE_AA)

    # Populate rows
    for r_idx, (deg_name, pairs) in enumerate(deg_rows):
        ry = header_h + r_idx * (h + pad)
        # Row label on the left
        pretty_name = deg_name.replace("_", " ").title()
        cv2.putText(canvas, pretty_name, (12, ry + h // 2 + 5), font, 0.50, (240, 240, 240), 1, cv2.LINE_AA)

        # Column 0: Clean reference
        canvas[ry:ry + h, col_x[0]:col_x[0] + w] = clean_image

        # Columns 1..2*num_sev: Degraded and Restored pairs
        for s_idx, (deg_img, rest_img) in enumerate(pairs):
            c_deg = 1 + 2 * s_idx
            c_rest = 2 + 2 * s_idx
            canvas[ry:ry + h, col_x[c_deg]:col_x[c_deg] + w] = deg_img
            canvas[ry:ry + h, col_x[c_rest]:col_x[c_rest] + w] = rest_img

    return canvas


@torch.no_grad()
def save_validation_samples(model, device, config, val_dataset, eval_indices, epoch):
    """Save all validation degradations, severities, and restorations into one big composite figure per epoch."""
    out_dir = os.path.join(config.OUTPUT_DIR, "val_fixed")
    os.makedirs(out_dir, exist_ok=True)

    raw_model = unwrap_model(model)
    use_amp = getattr(config, "USE_AMP", True) and (device.type == "cuda")
    pipeline = DegradationPipeline(config)
    min_severity = getattr(config, "MIN_SEVERITY", 0.2)
    steps = max(1, int(config.EVAL_SEVERITY_STEPS))
    severities = np.linspace(min_severity, 1.0, steps)

    image_grids = []

    for local_idx, img_idx in enumerate(eval_indices):
        image_path = val_dataset.image_paths[img_idx]
        clean_image = _load_clean_image(image_path, config)
        deg_rows = []

        # 8 single degradations + 2 compound multi-degradation test rows
        eval_recipes = [(deg_name, {deg_name: 1.0}) for deg_name in config.DEGRADATION_TYPES] + [
            ("compound_blur+noise", {"motion_blur": 0.85, "gaussian_noise": 0.75}),
            ("compound_band+jpeg", {"color_banding": 0.85, "exposure_issue": 0.60, "jpeg_compression": 0.75}),
        ]

        for row_label, recipe in eval_recipes:
            degraded_list = []
            for severity in severities:
                scaled_recipe = {k: float(severity) * w for k, w in recipe.items()}
                degraded = pipeline.apply_compound_degradation(clean_image, scaled_recipe)
                degraded_list.append(degraded)

            degraded_batch = (
                torch.from_numpy(np.stack(degraded_list, axis=0).astype(np.float32) / 255.0)
                .permute(0, 3, 1, 2)
                .to(device)
            )
            with torch.amp.autocast("cuda", enabled=use_amp):
                restored_batch, _, _, _ = raw_model(degraded_batch)
            restored_batch_np = (
                restored_batch.float().permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy() * 255
            ).astype(np.uint8)

            pairs = list(zip(degraded_list, restored_batch_np))
            deg_rows.append((row_label, pairs))

        title = (
            f"Epoch {epoch:03d} | Sample {local_idx + 1}/{len(eval_indices)} "
            f"({os.path.basename(image_path)}) | Bridge: {config.BRIDGE_TYPE} ({config.BRIDGE_STEPS} steps)"
        )
        grid = _render_single_image_grid(clean_image, deg_rows, severities, title)
        image_grids.append(grid)

    if image_grids:
        # Stack all sample grids vertically into one big figure for the epoch
        sep_bar = np.full((16, image_grids[0].shape[1], 3), 80, dtype=np.uint8)
        combined_blocks = []
        for idx, g in enumerate(image_grids):
            if idx > 0:
                combined_blocks.append(sep_bar)
            combined_blocks.append(g)
        big_figure = np.concatenate(combined_blocks, axis=0)
        figure_path = os.path.join(out_dir, f"epoch_{epoch:03d}.png")
        cv2.imwrite(figure_path, cv2.cvtColor(big_figure, cv2.COLOR_RGB2BGR))
        print(f"Saved combined validation figure to {figure_path}")

        # Automatically render inline in Kaggle / Jupyter notebook cell output when run via %run
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                from IPython.display import display, Image as IPyImage
                display(IPyImage(filename=figure_path, width=1000))
        except Exception:
            pass


def train_epoch(model, train_loader, loss_fn, optimizer, scaler, device, config, epoch):
    """Train for one epoch with multi-GPU DataParallel, AMP, and live GPU usage reporting."""
    model.train()
    use_amp = getattr(config, "USE_AMP", True) and (device.type == "cuda")
    total_loss = 0.0
    loss_components = {
        'l1': 0.0, 'gradient': 0.0, 'perceptual': 0.0, 'frequency': 0.0,
        'deg_type': 0.0, 'severity': 0.0
    }
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        degraded = batch['degraded'].to(device, non_blocking=True)
        clean = batch['clean'].to(device, non_blocking=True)
        deg_type_target = batch['deg_type'].to(device, non_blocking=True)
        severity_target = batch['severity'].to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)

        # Forward pass under AMP (bridge training step: sample random t, construct x_t, predict clean x_0)
        with torch.amp.autocast("cuda", enabled=use_amp):
            restored, deg_type, severity, deg_embedding = model(degraded, clean)
            total_batch_loss, loss_dict = loss_fn(
                restored, clean,
                deg_type, deg_type_target,
                severity, severity_target,
                deg_embedding
            )
        
        # Backward pass with GradScaler
        if scaler is not None and use_amp:
            scaler.scale(total_batch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Print immediate GPU usage snapshot on the very first batch of Epoch 1
        if epoch == 1 and batch_idx == 0:
            print_gpu_usage("Live GPU Usage (Epoch 1, Batch 1 Active)")

        # Update metrics
        total_loss += total_batch_loss.item()
        for key in loss_components:
            loss_components[key] += loss_dict[key]
        
        # Log every LOG_INTERVAL batches (or on batch 1) with live GPU utilization
        if batch_idx == 0 or (batch_idx + 1) % config.LOG_INTERVAL == 0:
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'gpu': format_gpu_postfix(),
            })
    
    print_gpu_usage(f"End of Epoch {epoch} GPU Usage Summary")

    # Average metrics
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches
    
    return avg_loss, loss_components


@torch.no_grad()
def validate(model, val_loader, loss_fn, device, config, val_dataset, eval_indices, epoch=None):
    """Validate model."""
    model.eval()
    use_amp = getattr(config, "USE_AMP", True) and (device.type == "cuda")
    total_loss = 0.0
    loss_components = {
        'l1': 0.0, 'gradient': 0.0, 'perceptual': 0.0, 'frequency': 0.0,
        'deg_type': 0.0, 'severity': 0.0
    }

    save_samples = (
        epoch is not None
        and config.VAL_SAMPLE_INTERVAL > 0
        and epoch % config.VAL_SAMPLE_INTERVAL == 0
    )
    samples_saved = False
    
    for batch in tqdm(val_loader, desc="Validation"):
        degraded = batch['degraded'].to(device, non_blocking=True)
        clean = batch['clean'].to(device, non_blocking=True)
        deg_type_target = batch['deg_type'].to(device, non_blocking=True)
        severity_target = batch['severity'].to(device, non_blocking=True)
        
        # Forward pass (multi-step reverse bridge sampling from t=1 -> t=0)
        with torch.amp.autocast("cuda", enabled=use_amp):
            restored, deg_type, severity, deg_embedding = model(degraded)
            total_batch_loss, loss_dict = loss_fn(
                restored, clean,
                deg_type, deg_type_target,
                severity, severity_target,
                deg_embedding
            )

        if save_samples and not samples_saved:
            save_validation_samples(model, device, config, val_dataset, eval_indices, epoch)
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
    """Save model checkpoint (always unwrapped so single-GPU inference works seamlessly)."""
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    if is_best:
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, "model_best.pt")
    else:
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"model_epoch_{epoch:03d}.pt")
    
    raw_model = unwrap_model(model)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'bridge_config': {
            'bridge_type': config.BRIDGE_TYPE,
            'bridge_steps': config.BRIDGE_STEPS,
            'sigma_max': config.BRIDGE_SIGMA_MAX,
            'ot_ode': config.BRIDGE_OT_ODE,
        },
    }
    
    torch.save(checkpoint, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")


def train(config, debug=False, resume_from=None):
    """Main training loop."""
    set_seed(config.SEED)
    # Setup
    model, loss_fn, optimizer, scheduler, scaler, device = setup_training(config, debug)
    
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

    eval_count = min(len(val_dataset), max(1, int(config.EVAL_IMAGE_COUNT)))
    rng = random.Random(config.SEED)
    eval_indices = rng.sample(range(len(val_dataset)), eval_count)
    
    print(f"Train set: {len(train_dataset)} images")
    print(f"Val set: {len(val_dataset)} images")
    
    # Resume from checkpoint if provided
    start_epoch = 1
    best_loss = float('inf')
    
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        raw_state = checkpoint['model_state_dict']
        # Strip 'module.' prefix if loading an older DataParallel checkpoint
        if any(k.startswith("module.") for k in raw_state.keys()):
            raw_state = {k.replace("module.", "", 1): v for k, v in raw_state.items()}
        unwrap_model(model).load_state_dict(raw_state)
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
            model, train_loader, loss_fn, optimizer, scaler, device, config, epoch
        )
        
        # Validate every N epochs (configurable) or at epoch end
        if epoch % max(1, config.VAL_EPOCH_INTERVAL) == 0 or epoch == config.EPOCHS:
            val_loss, val_losses = validate(
                model, val_loader, loss_fn, device, config, val_dataset, eval_indices, epoch=epoch
            )
            
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
    parser.add_argument(
        "--bridge-type",
        type=str,
        choices=["i2sb", "brownian", "flow"],
        default=None,
        help="Bridge formulation to use (i2sb, brownian, or flow)",
    )
    parser.add_argument("--bridge-steps", type=int, default=None, help="Number of reverse bridge steps")
    parser.add_argument("--sigma-max", type=float, default=None, help="Peak noise bubble std for bridge")
    parser.add_argument("--ot-ode", action="store_true", help="Use deterministic OT-ODE bridge sampler")
    
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
    if args.bridge_type is not None:
        config.BRIDGE_TYPE = args.bridge_type
    if args.bridge_steps is not None:
        config.BRIDGE_STEPS = args.bridge_steps
    if args.sigma_max is not None:
        config.BRIDGE_SIGMA_MAX = args.sigma_max
    if args.ot_ode:
        config.BRIDGE_OT_ODE = True

    train(config, debug=args.debug, resume_from=args.resume)
