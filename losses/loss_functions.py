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


class FrequencyLoss(nn.Module):
    """Frequency domain loss using FFT magnitude."""
    
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.L1Loss()
    
    def forward(self, x, y):
        """Compute frequency loss between x and y."""
        # Convert to grayscale for frequency analysis
        x_gray = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        y_gray = y.mean(dim=1, keepdim=True)
        
        # Compute FFT
        x_fft = torch.fft.rfft2(x_gray.squeeze(1))  # [B, H, W//2+1]
        y_fft = torch.fft.rfft2(y_gray.squeeze(1))
        
        # Magnitude spectrum
        x_mag = torch.abs(x_fft)
        y_mag = torch.abs(y_fft)
        
        # L1 loss on magnitude
        return self.loss_fn(x_mag, y_mag)


class RestorationLoss(nn.Module):
    """Combined loss: L1 + Perceptual + Frequency."""
    
    def __init__(self, config, device="cuda"):
        super().__init__()
        self.config = config
        
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss(
            layer=config.PERCEPTUAL_LAYERS[0],
            device=device,
            use_pretrained=False,
        )
        self.frequency_loss = FrequencyLoss()
        
        # Degradation classification loss
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, restored, clean, deg_type, deg_type_target, 
                severity, severity_target, deg_embedding=None):
        """
        Args:
            restored: Restored image [B, 3, H, W]
            clean: Clean ground truth [B, 3, H, W]
            deg_type: Predicted degradation type logits [B, num_types]
            deg_type_target: Target degradation type [B]
            severity: Predicted severity logits [B, severity_levels]
            severity_target: Target severity level [B]
            deg_embedding: Degradation embedding (unused, for future extensions)
        
        Returns:
            loss_dict: Dictionary with individual loss components
            total_loss: Weighted sum of losses
        """
        # Restoration losses
        loss_l1 = self.l1_loss(restored, clean)
        loss_perceptual = self.perceptual_loss(restored, clean)
        loss_frequency = self.frequency_loss(restored, clean)
        
        # Degradation estimation losses
        loss_deg_type = self.ce_loss(deg_type, deg_type_target)
        loss_severity = self.ce_loss(severity, severity_target)
        
        # Weighted sum
        total_loss = (
            self.config.LOSS_L1_WEIGHT * loss_l1 +
            self.config.LOSS_PERCEPTUAL_WEIGHT * loss_perceptual +
            self.config.LOSS_FREQUENCY_WEIGHT * loss_frequency +
            0.1 * loss_deg_type +  # Smaller weight for auxiliary task
            0.05 * loss_severity
        )
        
        loss_dict = {
            'l1': loss_l1.item(),
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
