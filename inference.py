"""Inference and evaluation utilities."""

import os
import sys
# Kaggle notebook path fix
sys.path.insert(0, '/kaggle/working/Degradation_Aware_Image_Restoration')

import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config
from models.architecture import DegradationAwareRestoration


class Inferencer:
    """Inference wrapper for restoration model."""
    
    def __init__(self, model_path, config=None, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.config = config or Config()
        
        # Load checkpoint and sync bridge settings if stored in checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'bridge_config' in checkpoint:
            bcfg = checkpoint['bridge_config']
            self.config.BRIDGE_TYPE = bcfg.get('bridge_type', self.config.BRIDGE_TYPE)
            self.config.BRIDGE_STEPS = bcfg.get('bridge_steps', self.config.BRIDGE_STEPS)
            self.config.BRIDGE_SIGMA_MAX = bcfg.get('sigma_max', self.config.BRIDGE_SIGMA_MAX)
            self.config.BRIDGE_OT_ODE = bcfg.get('ot_ode', self.config.BRIDGE_OT_ODE)

        # Load model
        self.model = DegradationAwareRestoration(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {model_path} (bridge={self.config.BRIDGE_TYPE}, steps={self.config.BRIDGE_STEPS})")
        print(f"Using device: {self.device}")
    
    @torch.no_grad()
    def restore(self, image_path, return_degradation_info=False, steps=None, ot_ode=None, return_trajectory=False):
        """Restore a single image using the reverse bridge sampler.
        
        Args:
            image_path: Path to degraded image
            return_degradation_info: If True, also return degradation predictions
            steps: Optional override for number of reverse bridge steps
            ot_ode: Optional override for deterministic OT-ODE sampling
            return_trajectory: If True, also return list of intermediate bridge states (uint8 RGB)
        
        Returns:
            restored_image: Restored image (numpy array, 0-255, uint8)
            degradation_info: Dict with type and severity predictions (if requested)
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        # Inference across the bridge (t=1 -> t=0)
        if return_trajectory:
            restored, deg_type, severity, deg_embedding, trajectory = self.model.sample(
                image_tensor, steps=steps, ot_ode=ot_ode, return_trajectory=True
            )
            traj_np = [
                (frame[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                for frame in trajectory
            ]
        else:
            restored, deg_type, severity, deg_embedding = self.model(
                image_tensor, steps=steps, ot_ode=ot_ode
            )
            traj_np = None
        
        # Convert back to numpy
        restored_np = (restored[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        
        if return_degradation_info:
            deg_type_pred = deg_type[0].argmax().item()
            severity_pred = severity[0].argmax().item()
            
            degradation_info = {
                'type': self.config.DEGRADATION_TYPES[deg_type_pred],
                'severity_level': severity_pred,
                'severity_value': (severity_pred + 1) * 0.2,  # 0.2 to 1.0
                'bridge_type': self.model.bridge.bridge_type,
                'bridge_steps': steps or self.model.default_steps,
            }
            if return_trajectory:
                return restored_np, degradation_info, traj_np
            return restored_np, degradation_info
        
        if return_trajectory:
            return restored_np, traj_np
        return restored_np
    
    @torch.no_grad()
    def restore_batch(self, image_dir, output_dir, recursive=False):
        """Restore all images in a directory.
        
        Args:
            image_dir: Directory containing degraded images
            output_dir: Directory to save restored images
            recursive: If True, process subdirectories recursively
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all images
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
        image_paths = []
        
        if recursive:
            for ext in image_extensions:
                image_paths.extend(Path(image_dir).rglob(f"*{ext}"))
                image_paths.extend(Path(image_dir).rglob(f"*{ext.upper()}"))
        else:
            for ext in image_extensions:
                image_paths.extend(Path(image_dir).glob(f"*{ext}"))
                image_paths.extend(Path(image_dir).glob(f"*{ext.upper()}"))
        
        image_paths = sorted(list(set(image_paths)))
        
        if not image_paths:
            print(f"No images found in {image_dir}")
            return
        
        print(f"Found {len(image_paths)} images. Processing...")
        
        for img_path in tqdm(image_paths):
            try:
                restored = self.restore(str(img_path))
                
                # Save restored image
                output_path = os.path.join(output_dir, img_path.name)
                restored_bgr = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, restored_bgr)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        print(f"Restored images saved to {output_dir}")


def evaluate_restoration(clean_dir, restored_dir, output_file=None):
    """Evaluate restoration quality using PSNR and SSIM.
    
    Args:
        clean_dir: Directory with clean reference images
        restored_dir: Directory with restored images
        output_file: Optional file to save evaluation results
    
    Returns:
        metrics: Dict with average PSNR and SSIM
    """
    clean_paths = sorted(Path(clean_dir).glob("*.png"))
    
    psnr_values = []
    ssim_values = []
    
    print("Evaluating restoration quality...")
    
    for clean_path in tqdm(clean_paths):
        restored_path = Path(restored_dir) / clean_path.name
        
        if not restored_path.exists():
            print(f"Warning: Restored image not found for {clean_path.name}")
            continue
        
        # Load images
        clean = cv2.imread(str(clean_path))
        restored = cv2.imread(str(restored_path))
        
        if clean is None or restored is None:
            continue
        
        # Convert to grayscale for SSIM
        clean_gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
        restored_gray = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)
        
        # Compute metrics
        psnr = peak_signal_noise_ratio(clean_gray, restored_gray, data_range=255)
        ssim = structural_similarity(clean_gray, restored_gray, data_range=255)
        
        psnr_values.append(psnr)
        ssim_values.append(ssim)
    
    # Compute averages
    metrics = {
        'avg_psnr': np.mean(psnr_values),
        'std_psnr': np.std(psnr_values),
        'avg_ssim': np.mean(ssim_values),
        'std_ssim': np.std(ssim_values),
        'num_images': len(psnr_values),
    }
    
    print(f"\nEvaluation Results:")
    print(f"  PSNR: {metrics['avg_psnr']:.2f} ± {metrics['std_psnr']:.2f} dB")
    print(f"  SSIM: {metrics['avg_ssim']:.4f} ± {metrics['std_ssim']:.4f}")
    print(f"  Evaluated {metrics['num_images']} images")
    
    if output_file:
        import json
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Results saved to {output_file}")
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Inference and evaluation utilities")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, help="Single image to restore")
    parser.add_argument("--batch-dir", type=str, help="Directory of images to restore")
    parser.add_argument("--output-dir", type=str, default="./restored", help="Output directory")
    parser.add_argument("--eval", type=str, help="Evaluate restoration (provide clean image directory)")
    parser.add_argument("--eval-output", type=str, help="File to save evaluation results")
    parser.add_argument("--steps", type=int, default=None, help="Number of reverse bridge steps (default from Config)")
    parser.add_argument("--ot-ode", action="store_true", help="Use deterministic OT-ODE sampling instead of stochastic SDE")
    parser.add_argument("--save-trajectory", action="store_true", help="Save intermediate bridge states (t=1 -> t=0)")
    
    args = parser.parse_args()
    
    inferencer = Inferencer(args.model)
    ot_ode_flag = True if args.ot_ode else None
    
    if args.image:
        print(f"Restoring {args.image}...")
        if args.save_trajectory:
            restored, deg_info, traj = inferencer.restore(
                args.image,
                return_degradation_info=True,
                steps=args.steps,
                ot_ode=ot_ode_flag,
                return_trajectory=True,
            )
        else:
            restored, deg_info = inferencer.restore(
                args.image,
                return_degradation_info=True,
                steps=args.steps,
                ot_ode=ot_ode_flag,
            )
            traj = None
        
        output_path = os.path.join(args.output_dir, os.path.basename(args.image))
        os.makedirs(args.output_dir, exist_ok=True)
        restored_bgr = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, restored_bgr)
        
        if traj is not None:
            stem = os.path.splitext(os.path.basename(args.image))[0]
            traj_dir = os.path.join(args.output_dir, f"{stem}_bridge_trajectory")
            os.makedirs(traj_dir, exist_ok=True)
            for idx, frame in enumerate(traj):
                frame_path = os.path.join(traj_dir, f"step_{idx:02d}.png")
                cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f"Bridge trajectory ({len(traj)} states) saved to {traj_dir}")
        
        print(f"Restored image saved to {output_path}")
        print(f"Degradation info: {deg_info}")
    
    if args.batch_dir:
        inferencer.restore_batch(args.batch_dir, args.output_dir)
    
    if args.eval:
        evaluate_restoration(args.eval, args.output_dir, args.eval_output)
