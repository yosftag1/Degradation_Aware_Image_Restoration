"""NAFNet-based degradation-aware restoration model with I2SB / Brownian / Flow Bridge."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for continuous bridge time t in [0, 1]."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t):
        """
        Args:
            t: Tensor of shape [B] or [B, 1] with values in [0, 1].
        Returns:
            Time embedding of shape [B, dim].
        """
        if t.dim() > 1:
            t = t.view(-1)
        t = t.float()
        half_dim = self.dim // 2
        emb_scale = math.log(10000.0) / max(1, half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb_scale)
        # Scale t in [0, 1] by 1000 for rich frequency coverage
        args = (t * 1000.0).unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class SchrodingerBridge(nn.Module):
    """Unified Image-to-Image Bridge supporting I2SB, Brownian Bridge, and Rectified Flow.

    Connects clean images x_0 (at t=0) and degraded images x_1 (at t=1):
        x_t = (1 - alpha_t) * x_0 + alpha_t * x_1 + sigma_t * eps
    """

    def __init__(self, bridge_type="i2sb", sigma_max=0.25):
        super().__init__()
        bridge_type = bridge_type.lower()
        if bridge_type not in ("i2sb", "brownian", "flow"):
            raise ValueError(f"Unsupported bridge_type '{bridge_type}'. Choose from 'i2sb', 'brownian', 'flow'.")
        self.bridge_type = bridge_type
        self.sigma_max = 0.0 if bridge_type == "flow" else float(sigma_max)

    def schedule(self, t):
        """Compute interpolation weight alpha_t and noise std sigma_t for t in [0, 1]."""
        t = t.clamp(0.0, 1.0)
        if self.bridge_type == "i2sb":
            # Symmetric quadratic schedule derived from a triangular beta_t diffusion rate
            # Smoothly departs x_0 (t=0) and x_1 (t=1), peaking at t=0.5
            alpha_t = torch.where(
                t < 0.5,
                2.0 * (t ** 2),
                1.0 - 2.0 * ((1.0 - t) ** 2),
            )
        else:
            # Linear interpolation for standard Brownian Bridge and Rectified Flow
            alpha_t = t

        sigma_t = self.sigma_max * torch.sqrt((alpha_t * (1.0 - alpha_t)).clamp(min=0.0))
        return alpha_t, sigma_t

    def q_sample(self, x_0, x_1, t, noise=None):
        """Sample intermediate bridge state x_t given clean x_0 and degraded x_1."""
        if noise is None:
            noise = torch.randn_like(x_0)
        alpha_t, sigma_t = self.schedule(t)
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        sigma_t = sigma_t.view(-1, 1, 1, 1)
        x_t = (1.0 - alpha_t) * x_0 + alpha_t * x_1 + sigma_t * noise
        return x_t

    def p_posterior_step(self, hat_x_0, x_t, t, u, ot_ode=False):
        """Step backward along the bridge from time t to earlier time u (0 <= u < t <= 1).

        Exact Markov bridge posterior p(x_u | hat_x_0, x_t):
            mean = (1 - alpha_u / alpha_t) * hat_x_0 + (alpha_u / alpha_t) * x_t
            std  = sigma_max * sqrt(alpha_u * (alpha_t - alpha_u) / alpha_t)
        """
        alpha_t, _ = self.schedule(t)
        alpha_u, _ = self.schedule(u)

        ratio = (alpha_u / alpha_t.clamp(min=1e-7)).clamp(0.0, 1.0)
        ratio_4d = ratio.view(-1, 1, 1, 1)

        mean_u = (1.0 - ratio_4d) * hat_x_0 + ratio_4d * x_t

        if ot_ode or self.sigma_max <= 0.0:
            return mean_u

        # Noise vanishes automatically when u == 0 (since alpha_u == 0)
        var_step = (self.sigma_max ** 2) * alpha_u * (1.0 - ratio)
        std_step = torch.sqrt(var_step.clamp(min=0.0)).view(-1, 1, 1, 1)
        noise = torch.randn_like(x_t)
        # Mask out noise wherever u == 0
        non_terminal = (u > 1e-6).float().view(-1, 1, 1, 1)
        return mean_u + non_terminal * std_step * noise


class LayerNorm2d(nn.Module):
    """LayerNorm for channels-first tensors."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        orig_dtype = x.dtype
        x_f = x.float()
        mean = x_f.mean(dim=1, keepdim=True)
        var = (x_f - mean).pow(2).mean(dim=1, keepdim=True)
        x_norm = ((x_f - mean) / torch.sqrt(var + self.eps)).to(orig_dtype)
        return x_norm * self.weight + self.bias


class SimpleGate(nn.Module):
    """Simple gating used in NAFNet blocks (clamped to prevent FP16 x1*x2 overflow)."""

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1.clamp(-64.0, 64.0) * x2.clamp(-64.0, 64.0)


class NAFBlock(nn.Module):
    """NAFNet block with depthwise conv, simple gate, and bounded time+degradation FiLM conditioning."""

    def __init__(self, channels, dw_size=3, ffn_expand=2, cond_dim=0):
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

        if cond_dim > 0:
            proj_linear = nn.Linear(cond_dim, channels * 4)
            nn.init.zeros_(proj_linear.weight)
            nn.init.zeros_(proj_linear.bias)
            self.cond_proj = nn.Sequential(nn.SiLU(), proj_linear)
        else:
            self.cond_proj = None

    def forward(self, x, cond=None):
        residual = x

        x = self.norm1(x)
        if self.cond_proj is not None and cond is not None:
            scale1, shift1, scale2, shift2 = (
                self.cond_proj(cond).unsqueeze(-1).unsqueeze(-1).chunk(4, dim=1)
            )
            # Bound FiLM scale and shift with tanh so 36 stacked blocks cannot overflow FP16
            x = x * (1.0 + 0.25 * torch.tanh(scale1)) + 2.0 * torch.tanh(shift1)
        else:
            scale2, shift2 = None, None

        x = self.pw1(x)
        x = self.dwconv(x)
        x = self.sg(x)
        x = x * self.sca(x).clamp(-16.0, 16.0)
        x = self.pw2(x)
        x = residual + x * self.beta

        y = self.norm2(x)
        if scale2 is not None:
            y = y * (1.0 + 0.25 * torch.tanh(scale2)) + 2.0 * torch.tanh(shift2)
        y = self.pw3(y)
        y = self.sg2(y)
        y = self.pw4(y)
        return x + y * self.gamma


class NAFStage(nn.Module):
    """Sequential wrapper that passes conditioning vector through multiple NAFBlocks."""

    def __init__(self, channels, num_blocks, cond_dim=0):
        super().__init__()
        self.blocks = nn.ModuleList([NAFBlock(channels, cond_dim=cond_dim) for _ in range(num_blocks)])

    def forward(self, x, cond=None):
        for block in self.blocks:
            x = block(x, cond=cond)
        return x


class NAFNetBackbone(nn.Module):
    """NAFNet encoder-decoder backbone conditioned on bridge time t and degradation embedding."""

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        width=64,
        enc_blocks=None,
        dec_blocks=None,
        middle_blocks=12,
        embedding_dim=16,
        time_embedding_dim=64,
        concat_degraded=True,
    ):
        super().__init__()
        enc_blocks = enc_blocks or [2, 2, 4, 8]
        dec_blocks = dec_blocks or [2, 2, 2, 2]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_dim = embedding_dim
        self.time_embedding_dim = time_embedding_dim
        self.concat_degraded = concat_degraded

        intro_in_channels = in_channels * 2 if concat_degraded else in_channels
        self.intro = nn.Conv2d(intro_in_channels, width, kernel_size=3, padding=1)

        self.time_embed = SinusoidalTimeEmbedding(time_embedding_dim)
        cond_dim = time_embedding_dim + embedding_dim

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        channels = width
        for num_blocks in enc_blocks:
            self.encoders.append(NAFStage(channels, num_blocks, cond_dim=cond_dim))
            self.downs.append(nn.Conv2d(channels, channels * 2, kernel_size=2, stride=2))
            channels *= 2

        self.embedding_proj = nn.Conv2d(channels + cond_dim, channels, kernel_size=1)
        self.middle = NAFStage(channels, middle_blocks, cond_dim=cond_dim)

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
            self.decoders.append(NAFStage(channels, num_blocks, cond_dim=cond_dim))

        self.ending = nn.Conv2d(width, out_channels, kernel_size=3, padding=1)

    def forward(self, x_t, degradation_embedding=None, t=None, x_degraded=None):
        """Predict clean image hat_x_0 from current bridge state x_t at time t."""
        b, _, h, w = x_t.shape

        if x_degraded is None:
            x_degraded = x_t

        if t is None:
            t = torch.ones(b, device=x_t.device, dtype=x_t.dtype)
        elif not torch.is_tensor(t):
            t = torch.full((b,), float(t), device=x_t.device, dtype=x_t.dtype)
        elif t.dim() == 0:
            t = t.expand(b)

        t_emb = self.time_embed(t)

        if degradation_embedding is None:
            degradation_embedding = torch.zeros(
                b, self.embedding_dim, device=x_t.device, dtype=x_t.dtype
            )
        elif degradation_embedding.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected degradation embedding dim {self.embedding_dim}, "
                f"got {degradation_embedding.shape[1]}"
            )

        cond = torch.cat([t_emb, degradation_embedding], dim=1)

        if self.concat_degraded:
            net_in = torch.cat([x_t, x_degraded], dim=1)
        else:
            net_in = x_t

        x = self.intro(net_in)
        skips = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x, cond=cond)
            skips.append(x)
            x = down(x)

        _, _, bh, bw = x.shape
        cond_map = cond.view(b, -1, 1, 1).expand(b, -1, bh, bw)
        x = torch.cat([x, cond_map], dim=1)
        x = self.embedding_proj(x)

        x = self.middle(x, cond=cond)

        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = x + skip
            x = decoder(x, cond=cond)

        # Predict clean image hat_x_0 via residual over current bridge state x_t
        return x_t + self.ending(x)


class DegradationEstimator(nn.Module):
    """Predicts multi-degradation presence logits, 8D severity vector s in [0,1]^8, and conditioning embedding.

    Uses Global Mean + Global Standard Deviation pooling so the estimator can
    simultaneously detect multiple active degradations and their individual severities.
    """

    def __init__(self, num_degradations=8, severity_levels=5, embedding_dim=32):
        super().__init__()
        self.num_degradations = num_degradations
        self.severity_levels = severity_levels
        self.embedding_dim = embedding_dim

        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)

        # Mean (128) + Std (128) = 256 diagnostic features (normalized for AMP stability)
        self.stats_norm = nn.LayerNorm(256)
        self.head_mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
        )

        # Multi-label presence logits [B, num_degradations]
        self.fc_type = nn.Linear(128, num_degradations)
        # Continuous per-degradation severity vector s in [0, 1]^K [B, num_degradations]
        self.fc_severity = nn.Linear(128, num_degradations)
        # Conditioning embedding combines hidden features + predicted presence + severity vector
        self.fc_embedding = nn.Linear(128 + 2 * num_degradations, embedding_dim)
        self.emb_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        orig_dtype = x.dtype
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        x = F.silu(self.conv3(x))

        # Compute spatial mean, std, and MLP heads in float32 so stats never overflow FP16
        with torch.amp.autocast("cuda", enabled=False):
            x_f = x.float()
            feat_mean = x_f.mean(dim=[2, 3])
            feat_std = torch.sqrt(x_f.var(dim=[2, 3], unbiased=False) + 1e-5)
            stats = self.stats_norm(torch.cat([feat_mean, feat_std], dim=1))

            hidden = self.head_mlp(stats)
            deg_type = self.fc_type(hidden).clamp(-20.0, 20.0)
            deg_probs = torch.sigmoid(deg_type)
            raw_severity = torch.sigmoid(self.fc_severity(hidden).clamp(-20.0, 20.0))
            # Couple predicted 8D severity vector with presence probability so inactive slots naturally vanish
            severity = deg_probs * raw_severity

            emb_in = torch.cat([hidden, deg_probs, severity], dim=1)
            embedding = self.emb_norm(self.fc_embedding(emb_in))

        return deg_type, severity, embedding.to(orig_dtype)


class DegradationAwareRestoration(nn.Module):
    """Degradation-aware image restoration using a Physics/Stochastic Bridge (I2SB / Brownian / Flow)."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        bridge_type = getattr(config, "BRIDGE_TYPE", "i2sb")
        sigma_max = getattr(config, "BRIDGE_SIGMA_MAX", 0.25)
        self.default_steps = getattr(config, "BRIDGE_STEPS", 4)
        self.default_ot_ode = getattr(config, "BRIDGE_OT_ODE", False)
        time_emb_dim = getattr(config, "TIME_EMBEDDING_DIM", 64)
        concat_degraded = getattr(config, "CONCAT_DEGRADED_INPUT", True)

        self.bridge = SchrodingerBridge(bridge_type=bridge_type, sigma_max=sigma_max)

        self.backbone = NAFNetBackbone(
            in_channels=config.INPUT_CHANNELS,
            out_channels=config.OUTPUT_CHANNELS,
            width=config.NAFNET_WIDTH,
            enc_blocks=config.NAFNET_ENC_BLOCKS,
            dec_blocks=config.NAFNET_DEC_BLOCKS,
            middle_blocks=config.NAFNET_MIDDLE_BLOCKS,
            embedding_dim=config.DEGRADATION_EMBEDDING_DIM,
            time_embedding_dim=time_emb_dim,
            concat_degraded=concat_degraded,
        )

        self.degradation_estimator = DegradationEstimator(
            num_degradations=len(config.DEGRADATION_TYPES),
            severity_levels=config.SEVERITY_LEVELS,
            embedding_dim=config.DEGRADATION_EMBEDDING_DIM,
        )

    def forward_bridge_train(self, degraded, clean, t=None):
        """Single-pass endpoint-anchored I2SB bridge training step.

        Because the backbone receives ONLY x_t (and NOT x_1 alongside x_t), it cannot
        algebraically cancel out x_1. Sampling t=1.0 on 50% of the batch and t in [0.15, 0.95]
        on the other 50% ensures strong restoration at t=1.0 and smooth refinement across the bridge.
        """
        b = degraded.shape[0]
        device = degraded.device
        dtype = degraded.dtype

        if t is None:
            mid = b // 2
            t = torch.ones(b, device=device, dtype=dtype)
            if mid > 0:
                t[mid:] = torch.empty(b - mid, device=device, dtype=dtype).uniform_(0.15, 0.95)

        x_t = self.bridge.q_sample(x_0=clean, x_1=degraded, t=t)
        deg_type, severity, deg_embedding = self.degradation_estimator(degraded)
        hat_x_0 = self.backbone(x_t, degradation_embedding=deg_embedding, t=t, x_degraded=degraded)
        return hat_x_0, deg_type, severity, deg_embedding

    def sample(self, degraded, steps=None, ot_ode=None, return_trajectory=False):
        """Walk backward along the bridge from t=1 (degraded x_1) to t=0 (clean x_0)."""
        steps = max(1, int(steps if steps is not None else self.default_steps))
        ot_ode = self.default_ot_ode if ot_ode is None else bool(ot_ode)

        b = degraded.shape[0]
        device = degraded.device
        dtype = degraded.dtype

        deg_type, severity, deg_embedding = self.degradation_estimator(degraded)

        # Time schedule from t=1.0 down to t=0.0
        time_grid = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
        x_t = degraded
        trajectory = [x_t] if return_trajectory else None

        for i in range(steps):
            t_cur = torch.full((b,), time_grid[i].item(), device=device, dtype=dtype)
            t_next = torch.full((b,), time_grid[i + 1].item(), device=device, dtype=dtype)

            hat_x_0 = self.backbone(
                x_t,
                degradation_embedding=deg_embedding,
                t=t_cur,
                x_degraded=degraded,
            )
            # Clamp predicted x_0 in valid image range during multi-step walk for stability
            if i < steps - 1:
                hat_x_0 = hat_x_0.clamp(0.0, 1.0)

            x_t = self.bridge.p_posterior_step(
                hat_x_0=hat_x_0,
                x_t=x_t,
                t=t_cur,
                u=t_next,
                ot_ode=ot_ode,
            )
            if return_trajectory:
                trajectory.append(x_t)

        if return_trajectory:
            return x_t, deg_type, severity, deg_embedding, trajectory
        return x_t, deg_type, severity, deg_embedding

    def forward(self, x, clean=None, t=None, steps=None, ot_ode=None):
        """Unified forward pass:
        - If `clean` is provided: runs a stochastic bridge training step at random/given `t`.
        - Otherwise: runs multi-step bridge inference from `x` (t=1 -> t=0).
        """
        if clean is not None:
            return self.forward_bridge_train(degraded=x, clean=clean, t=t)
        return self.sample(degraded=x, steps=steps, ot_ode=ot_ode, return_trajectory=False)


if __name__ == "__main__":
    from config import Config

    cfg = Config()
    model = DegradationAwareRestoration(cfg)
    x_deg = torch.rand(2, 3, 256, 256)
    x_clean = torch.rand(2, 3, 256, 256)

    # Test bridge training step
    hat_x0, deg_type, severity, deg_emb = model(x_deg, clean=x_clean)
    # Test multi-step bridge inference
    restored, _, _, _ = model(x_deg, steps=4)

    print(f"Bridge type: {cfg.BRIDGE_TYPE} (steps={cfg.BRIDGE_STEPS}, sigma_max={cfg.BRIDGE_SIGMA_MAX})")
    print(f"Input shape: {x_deg.shape}")
    print(f"Train hat_x0 shape: {hat_x0.shape}")
    print(f"Sampled restored shape: {restored.shape}")
    print(f"Degradation type logits: {deg_type.shape}")
    print(f"Severity logits: {severity.shape}")
    print(f"Degradation embedding: {deg_emb.shape}")
