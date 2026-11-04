"""Loss functions for degradation-aware image restoration."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss."""
    
    def __init__(self, layer="relu3_4", device="cuda", use_pretrained=True):
        super().__init__()
        
        try:
            weights = models.VGG19_Weights.DEFAULT if use_pretrained else None
            vgg = models.vgg19(weights=weights)
        except Exception:
            vgg = models.vgg19(weights=None)
        
        # Layer mapping
        layer_name_mapping = {
            "relu1_1": 1,
            "relu1_2": 3,
            "relu2_1": 6,
            "relu2_2": 8,
            "relu3_1": 11,
            "relu3_2": 13,
            "relu3_3": 15,
            "relu3_4": 17,
            "relu4_1": 20,
            "relu4_2": 22,
            "relu4_3": 24,
            "relu4_4": 26,
            "relu5_1": 29,
            "relu5_2": 31,
            "relu5_3": 33,
            "relu5_4": 35,
        }
        
        if layer not in layer_name_mapping:
            raise ValueError(f"Layer {layer} not in available layers")
        
        layer_index = layer_name_mapping[layer]
        self.features = nn.Sequential(*list(vgg.features.children())[:layer_index+1])
        self.features.eval()
        
        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        # ImageNet normalization
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        
        self.loss_fn = nn.L1Loss()

        # Ensure buffers and VGG features live on the requested device.
        self.to(device)
    
    def normalize(self, x):
        """Normalize using ImageNet statistics."""
        return (x - self.mean) / self.std
    
    def forward(self, x, y):
        """Compute perceptual loss between x and y."""
        x_feat = self.features(self.normalize(x))
        y_feat = self.features(self.normalize(y))
        return self.loss_fn(x_feat, y_feat)


class CharbonnierLoss(nn.Module):
    """Smooth L1 / Charbonnier loss: sqrt((x - y)^2 + eps^2), standard in NAFNet/Restormer."""

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, x, y):
        diff = x.float() - y.float()
        return torch.mean(torch.sqrt(diff * diff + self.eps2))


class SpatialGradientLoss(nn.Module):
    """First-order spatial derivative (dx, dy) L1 loss to erase color banding steps and JPEG block boundaries."""

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x_f = x.float()
        y_f = y.float()
        dx_x = x_f[:, :, :, 1:] - x_f[:, :, :, :-1]
        dx_y = y_f[:, :, :, 1:] - y_f[:, :, :, :-1]
        dy_x = x_f[:, :, 1:, :] - x_f[:, :, :-1, :]
        dy_y = y_f[:, :, 1:, :] - y_f[:, :, :-1, :]
        return F.l1_loss(dx_x, dx_y) + F.l1_loss(dy_x, dy_y)


class FrequencyLoss(nn.Module):
    """3-channel RGB frequency domain loss supervising both amplitude spectrum and spatial phase."""
    
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.L1Loss()
    
    def forward(self, x, y):
        """Compute RGB 2D FFT magnitude + complex phase loss (cast to float32 for cuFFT stability under AMP)."""
        x_f = x.float()
        y_f = y.float()
        
        # Orthonormal 2D FFT across all 3 RGB channels [B, 3, H, W//2+1]
        x_fft = torch.fft.rfft2(x_f, norm="ortho")
        y_fft = torch.fft.rfft2(y_f, norm="ortho")
        
        # Magnitude spectrum L1 + complex (real & imaginary phase-preserving) L1
        mag_loss = self.loss_fn(torch.abs(x_fft), torch.abs(y_fft))
        complex_loss = torch.view_as_real(x_fft - y_fft).abs().mean()
        return 0.5 * mag_loss + 0.5 * complex_loss


class RestorationLoss(nn.Module):
    """Combined loss: Charbonnier + SpatialGradient + Perceptual + 3-Channel Frequency + Positive-Weighted 8D Estimator."""
    
    def __init__(self, config, device="cuda"):
        super().__init__()
        self.config = config
        
        self.l1_loss = CharbonnierLoss(eps=1e-3)
        self.gradient_loss = SpatialGradientLoss()
        self.perceptual_loss = PerceptualLoss(
            layer=config.PERCEPTUAL_LAYERS[0],
            device=device,
            use_pretrained=True,
        )
        self.frequency_loss = FrequencyLoss()
        
        # Positive-weighted multi-label BCE (pos_weight=3.0 balances ~22% active vs ~78% inactive slots)
        self.register_buffer("pos_weight", torch.tensor(3.0, device=device))
    
    def forward(self, restored, clean, deg_type, deg_type_target, 
                severity, severity_target, deg_embedding=None):
        """
        Args:
            restored: Restored image [B, 3, H, W]
            clean: Clean ground truth [B, 3, H, W]
            deg_type: Predicted multi-label degradation presence logits [B, num_types]
            deg_type_target: Target binary mask [B, num_types] (or 1D class index [B])
            severity: Predicted 8D severity vector s_hat in [0, 1]^K [B, num_types]
            severity_target: Target 8D severity vector s in [0, 1]^K [B, num_types]
            deg_embedding: Degradation embedding (unused, for future extensions)
        """
        # Restoration losses
        loss_l1 = self.l1_loss(restored, clean)
        loss_gradient = self.gradient_loss(restored, clean)
        loss_perceptual = self.perceptual_loss(restored, clean)
        loss_frequency = self.frequency_loss(restored, clean)
        
        # Support both 8D vector targets [B, K] and legacy 1D class indices [B]
        num_types = deg_type.shape[-1]
        if deg_type_target.dim() == 1:
            deg_type_target = F.one_hot(deg_type_target.long(), num_classes=num_types).float()
        else:
            deg_type_target = deg_type_target.float()

        if severity_target.dim() == 1:
            sev_scalar = (severity_target.float() + 1.0) / float(self.config.SEVERITY_LEVELS)
            severity_target = deg_type_target * sev_scalar.unsqueeze(1)
        else:
            severity_target = severity_target.float()

        # Positive-weighted multi-degradation presence loss
        pos_w = self.pos_weight.to(deg_type.device)
        loss_deg_type = F.binary_cross_entropy_with_logits(
            deg_type.float(), deg_type_target, pos_weight=pos_w
        )

        # Active-weighted severity MSE (weights active degradations 4x so inactive zeros don't dominate)
        sev_weights = 1.0 + 3.0 * (deg_type_target > 0.5).float()
        loss_severity = torch.mean(sev_weights * ((severity.float() - severity_target) ** 2))
        
        grad_weight = getattr(self.config, "LOSS_GRADIENT_WEIGHT", 0.15)

        # Weighted sum
        total_loss = (
            self.config.LOSS_L1_WEIGHT * loss_l1 +
            grad_weight * loss_gradient +
            self.config.LOSS_PERCEPTUAL_WEIGHT * loss_perceptual +
            self.config.LOSS_FREQUENCY_WEIGHT * loss_frequency +
            0.05 * loss_deg_type +
            0.10 * loss_severity
        )
        
        loss_dict = {
            'l1': loss_l1.item(),
            'gradient': loss_gradient.item(),
            'perceptual': loss_perceptual.item(),
            'frequency': loss_frequency.item(),
            'deg_type': loss_deg_type.item(),
            'severity': loss_severity.item(),
            'total': total_loss.item(),
        }
        
        return total_loss, loss_dict


if __name__ == "__main__":
    from config import Config
    
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test losses
    loss_fn = RestorationLoss(config, device=device)
    
    B, C, H, W = 2, 3, 256, 256
    restored = torch.randn(B, C, H, W, device=device)
    clean = torch.randn(B, C, H, W, device=device)
    deg_type = torch.randn(B, 3, device=device)
    deg_type_target = torch.randint(0, 3, (B,), device=device)
    severity = torch.randn(B, 5, device=device)
    severity_target = torch.randint(0, 5, (B,), device=device)
    
    total_loss, loss_dict = loss_fn(
        restored, clean, deg_type, deg_type_target,
        severity, severity_target
    )
    
    print("Loss components:")
    for key, val in loss_dict.items():
        print(f"  {key}: {val:.6f}")
