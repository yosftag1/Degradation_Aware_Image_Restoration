"""Lightweight U-Net with channel attention and degradation estimator."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel attention module inspired by NAFNet."""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
    
    def forward(self, x):
        # x: [B, C, H, W]
        b, c, _, _ = x.size()
        se = self.avg_pool(x).view(b, c)  # [B, C]
        se = self.fc(se).view(b, c, 1, 1)  # [B, C, 1, 1]
        return x * se.sigmoid()


class NAFNetBlock(nn.Module):
    """Lightweight NAFNet-style block with channel attention."""
    
    def __init__(self, channels, reduction=16, dw_size=3):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        # Depthwise convolution
        self.dw = nn.Conv2d(
            channels, channels, dw_size, 
            padding=dw_size // 2, groups=channels, bias=False
        )
        self.pw1 = nn.Conv2d(channels, channels * 2, 1, bias=True)
        self.pw2 = nn.Conv2d(channels * 2, channels, 1, bias=True)
        self.attention = ChannelAttention(channels, reduction)
    
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.dw(x)
        x = self.pw1(x)
        x = F.gelu(x)
        x = self.pw2(x)
        x = self.attention(x)
        return x + residual


class Encoder(nn.Module):
    """Lightweight encoder with progressive downsampling."""
    
    def __init__(self, in_channels=3, base_channels=32, num_blocks=4):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        self.blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        
        for i in range(num_blocks):
            ch = base_channels * (2 ** i)
            self.blocks.append(
                nn.Sequential(
                    NAFNetBlock(ch),
                    NAFNetBlock(ch),
                )
            )
            if i < num_blocks - 1:
                self.downsample.append(
                    nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1)
                )
        
        self.num_blocks = num_blocks
    
    def forward(self, x):
        features = []
        x = self.conv_in(x)
        features.append(x)
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            features.append(x)
            if i < len(self.downsample):
                x = self.downsample[i](x)
        
        return features


class Decoder(nn.Module):
    """Decoder with progressive upsampling and skip connections."""
    
    def __init__(self, base_channels=32, num_blocks=4, out_channels=3):
        super().__init__()
        self.degradation_proj = nn.LazyConv2d(base_channels * (2 ** (num_blocks - 1)), 1)
        self.upsample = nn.ModuleList()
        self.fuse = nn.ModuleList()
        self.blocks = nn.ModuleList()

        for i in range(num_blocks - 1, -1, -1):
            ch = base_channels * (2 ** i)
            self.blocks.append(nn.Sequential(NAFNetBlock(ch), NAFNetBlock(ch)))
            if i > 0:
                self.upsample.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                        nn.Conv2d(ch, ch // 2, 3, padding=1),
                    )
                )
                self.fuse.append(nn.Conv2d(ch, ch // 2, 1))

        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)
    
    def forward(self, features, degradation_embedding=None):
        """
        Args:
            features: List of feature maps from encoder (largest to smallest)
            degradation_embedding: [B, E] degradation embedding to condition decoder
        """
        # features are [skip0, skip1, ..., bottleneck]
        x = features[-1]

        if degradation_embedding is not None:
            B, _, H, W = x.shape
            deg_map = degradation_embedding.view(B, -1, 1, 1).expand(B, -1, H, W)
            x = torch.cat([x, deg_map], dim=1)
            x = self.degradation_proj(x)

        x = self.blocks[0](x)

        for level in range(1, len(self.blocks)):
            x = self.upsample[level - 1](x)
            skip = features[-(level + 1)]
            x = torch.cat([x, skip], dim=1)
            x = self.fuse[level - 1](x)
            x = self.blocks[level](x)

        return self.conv_out(x)


class DegradationEstimator(nn.Module):
    """Predicts degradation type and severity from input image."""
    
    def __init__(self, num_degradations=3, severity_levels=5, embedding_dim=16):
        super().__init__()
        self.num_degradations = num_degradations
        self.severity_levels = severity_levels
        self.embedding_dim = embedding_dim
        
        # Lightweight feature extraction
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Branches for each degradation type + severity
        self.fc_type = nn.Linear(64, num_degradations)
        self.fc_severity = nn.Linear(64, severity_levels)
        self.fc_embedding = nn.Linear(64, embedding_dim)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.global_pool(x).view(x.shape[0], -1)
        
        deg_type = self.fc_type(x)  # [B, num_degradations]
        severity = self.fc_severity(x)  # [B, severity_levels]
        embedding = self.fc_embedding(x)  # [B, embedding_dim]
        
        return deg_type, severity, embedding


class DegradationAwareRestoration(nn.Module):
    """Complete degradation-aware image restoration model."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.encoder = Encoder(
            in_channels=config.INPUT_CHANNELS,
            base_channels=config.BASE_CHANNELS,
            num_blocks=config.NUM_BLOCKS,
        )
        
        self.decoder = Decoder(
            base_channels=config.BASE_CHANNELS,
            num_blocks=config.NUM_BLOCKS,
            out_channels=config.OUTPUT_CHANNELS,
        )
        
        self.degradation_estimator = DegradationEstimator(
            num_degradations=len(config.DEGRADATION_TYPES),
            severity_levels=config.SEVERITY_LEVELS,
            embedding_dim=config.DEGRADATION_EMBEDDING_DIM,
        )
    
    def forward(self, x):
        """
        Args:
            x: Input degraded image [B, 3, H, W]
        
        Returns:
            restored: Restored image [B, 3, H, W]
            deg_type: Degradation type logits [B, num_types]
            severity: Severity level logits [B, severity_levels]
        """
        # Estimate degradation
        deg_type, severity, deg_embedding = self.degradation_estimator(x)
        
        # Encode
        features = self.encoder(x)
        
        # Decode with degradation conditioning
        restored = self.decoder(features, deg_embedding)
        
        return restored, deg_type, severity, deg_embedding


if __name__ == "__main__":
    from config import Config
    
    model = DegradationAwareRestoration(Config())
    x = torch.randn(2, 3, 256, 256)
    restored, deg_type, severity, deg_emb = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Restored shape: {restored.shape}")
    print(f"Degradation type logits: {deg_type.shape}")
    print(f"Severity logits: {severity.shape}")
    print(f"Degradation embedding: {deg_emb.shape}")
