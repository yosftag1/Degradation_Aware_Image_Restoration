"""NAFNet-based degradation-aware restoration model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm for channels-first tensors."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


class SimpleGate(nn.Module):
    """Simple gating used in NAFNet blocks."""

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """NAFNet block with depthwise conv and simple gate."""

    def __init__(self, channels, dw_size=3, ffn_expand=2):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, channels * ffn_expand, kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(
            channels * ffn_expand,
            channels * ffn_expand,
            kernel_size=dw_size,
            padding=dw_size // 2,
            groups=channels * ffn_expand,
            bias=True,
        )
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        self.pw2 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        self.norm2 = LayerNorm2d(channels)
        self.pw3 = nn.Conv2d(channels, channels * ffn_expand, kernel_size=1, bias=True)
        self.sg2 = SimpleGate()
        self.pw4 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        residual = x

        x = self.norm1(x)
        x = self.pw1(x)
        x = self.dwconv(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.pw2(x)
        x = residual + x * self.beta

        y = self.norm2(x)
        y = self.pw3(y)
        y = self.sg2(y)
        y = self.pw4(y)
        return x + y * self.gamma


class NAFNetBackbone(nn.Module):
    """NAFNet encoder-decoder backbone."""

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        width=64,
        enc_blocks=None,
        dec_blocks=None,
        middle_blocks=12,
        embedding_dim=16,
    ):
        super().__init__()
        enc_blocks = enc_blocks or [2, 2, 4, 8]
        dec_blocks = dec_blocks or [2, 2, 2, 2]

        self.embedding_dim = embedding_dim
        self.intro = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        channels = width
        for num_blocks in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(channels) for _ in range(num_blocks)]))
            self.downs.append(nn.Conv2d(channels, channels * 2, kernel_size=2, stride=2))
            channels *= 2

        self.embedding_proj = nn.Conv2d(channels + embedding_dim, channels, kernel_size=1)
        self.middle = nn.Sequential(*[NAFBlock(channels) for _ in range(middle_blocks)])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for num_blocks in dec_blocks:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels * 2, kernel_size=1, bias=True),
                    nn.PixelShuffle(2),
                )
            )
            channels //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(channels) for _ in range(num_blocks)]))

        self.ending = nn.Conv2d(width, out_channels, kernel_size=3, padding=1)

    def forward(self, x, degradation_embedding=None):
        x = self.intro(x)
        skips = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        if degradation_embedding is not None:
            if degradation_embedding.shape[1] != self.embedding_dim:
                raise ValueError(
                    f"Expected degradation embedding dim {self.embedding_dim}, "
                    f"got {degradation_embedding.shape[1]}"
                )
            b, _, h, w = x.shape
            deg_map = degradation_embedding.view(b, -1, 1, 1).expand(b, -1, h, w)
            x = torch.cat([x, deg_map], dim=1)
            x = self.embedding_proj(x)

        x = self.middle(x)

        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = x + skip
            x = decoder(x)

        return self.ending(x)


class DegradationEstimator(nn.Module):
    """Predicts degradation type and severity from input image."""

    def __init__(self, num_degradations=3, severity_levels=5, embedding_dim=16):
        super().__init__()
        self.num_degradations = num_degradations
        self.severity_levels = severity_levels
        self.embedding_dim = embedding_dim

        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc_type = nn.Linear(64, num_degradations)
        self.fc_severity = nn.Linear(64, severity_levels)
        self.fc_embedding = nn.Linear(64, embedding_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.global_pool(x).view(x.shape[0], -1)

        deg_type = self.fc_type(x)
        severity = self.fc_severity(x)
        embedding = self.fc_embedding(x)

        return deg_type, severity, embedding


class DegradationAwareRestoration(nn.Module):
    """Complete degradation-aware image restoration model."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.backbone = NAFNetBackbone(
            in_channels=config.INPUT_CHANNELS,
            out_channels=config.OUTPUT_CHANNELS,
            width=config.NAFNET_WIDTH,
            enc_blocks=config.NAFNET_ENC_BLOCKS,
            dec_blocks=config.NAFNET_DEC_BLOCKS,
            middle_blocks=config.NAFNET_MIDDLE_BLOCKS,
            embedding_dim=config.DEGRADATION_EMBEDDING_DIM,
        )

        self.degradation_estimator = DegradationEstimator(
            num_degradations=len(config.DEGRADATION_TYPES),
            severity_levels=config.SEVERITY_LEVELS,
            embedding_dim=config.DEGRADATION_EMBEDDING_DIM,
        )

    def forward(self, x):
        deg_type, severity, deg_embedding = self.degradation_estimator(x)
        restored = self.backbone(x, deg_embedding)
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
