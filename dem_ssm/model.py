"""Inspectable pure-PyTorch *reconstruction* of the three DeM-SSM components.

This implements the manuscript's architectural *description*, not an archival copy
of the unavailable original training source. The selective SSM is a transparent,
sequential diagonal-state implementation, not the official fused CUDA Mamba kernel.
"""
from __future__ import annotations
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class SelectiveSSM(nn.Module):
    """Input-dependent, diagonal selective state-space recurrence with ZOH discretization.

    Complexity is O(B*length*channels*state_size); the Python loop is intentionally
    explicit/auditable, not throughput-equivalent to optimized Mamba kernels.
    """
    def __init__(self, dim: int, state_size: int = 16, kernel: int = 3):
        super().__init__()
        self.dim = dim
        self.state_size = state_size
        self.in_proj = nn.Linear(dim, 2 * dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel, padding=kernel - 1, groups=dim)
        self.param_proj = nn.Linear(dim, 1 + 2 * state_size)
        self.log_a = nn.Parameter(torch.zeros(dim, state_size))
        self.skip = nn.Parameter(torch.ones(dim))
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, return_gate: bool = False):
        batch, length, dim = x.shape
        u, z = self.in_proj(x).chunk(2, dim=-1)
        u = F.silu(self.dwconv(u.transpose(1, 2))[..., :length].transpose(1, 2))
        dtbc = self.param_proj(u)
        delta = F.softplus(dtbc[..., :1]) + 1e-4
        bcoef, ccoef = dtbc[..., 1:].chunk(2, dim=-1)
        a = -torch.exp(self.log_a).clamp(max=100.0)
        state = x.new_zeros((batch, dim, self.state_size))
        outputs = []
        for step in range(length):
            dt = delta[:, step].unsqueeze(-1)
            da = dt * a.unsqueeze(0)
            abar = torch.exp(da)
            # Exact zero-order-hold integral for diagonal negative A.
            integral = torch.expm1(da) / a.unsqueeze(0)
            input_term = integral * bcoef[:, step].unsqueeze(1) * u[:, step].unsqueeze(-1)
            state = abar * state + input_term
            y = (state * ccoef[:, step].unsqueeze(1)).sum(dim=-1)
            y = y + self.skip * u[:, step]
            outputs.append(y)
        gate = torch.sigmoid(z)
        y = torch.stack(outputs, dim=1) * gate
        result = self.norm(x + self.out_proj(y))
        return (result, gate.mean(dim=-1)) if return_gate else result


class BidirectionalRasterSSM(nn.Module):
    """Two independent, reversed row-major selective scans (manuscript Section 3.3)."""
    def __init__(self, dim: int, state_size: int):
        super().__init__()
        self.forward_scan = SelectiveSSM(dim, state_size)
        self.backward_scan = SelectiveSSM(dim, state_size)
        self.fusion = nn.Linear(2 * dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: Tensor) -> Tensor:
        left = self.forward_scan(tokens)
        right = self.backward_scan(tokens.flip(1)).flip(1)
        return self.norm(tokens + self.fusion(torch.cat([left, right], dim=-1)))


class VisionPatchEncoder(nn.Module):
    def __init__(self, dim: int, patch: int, state_size: int, backbone: str):
        super().__init__()
        self.backbone = backbone
        if backbone == "vim":
            self.patch = nn.Conv2d(3, dim, patch, stride=patch)
            self.scan = BidirectionalRasterSSM(dim, state_size)
        elif backbone == "cnn":
            self.patch = nn.Sequential(
                nn.Conv2d(3, dim // 2, 3, padding=1), nn.GELU(),
                nn.Conv2d(dim // 2, dim, patch, stride=patch), nn.GELU())
        else:
            raise ValueError("backbone must be vim or cnn")

    def forward(self, frames: Tensor) -> Tensor:
        tokens = self.patch(frames).flatten(2).transpose(1, 2)
        return self.scan(tokens) if self.backbone == "vim" else tokens


class DeformableFacialPrior(nn.Module):
    """5 normalized XY anchors, M heads, K learned bilinear samples per head."""
    def __init__(self, dim: int, patch: int, n_heads: int, n_points: int,
                 n_landmarks: int = 5, max_offset_px: float = 24.0,
                 prior: str = "deformable"):
        super().__init__()
        if dim % n_heads:
            raise ValueError("dim must be divisible by deformable heads")
        self.heads, self.points, self.n_landmarks = n_heads, n_points, n_landmarks
        self.dim = dim
        self.patch = patch
        self.max_offset_px, self.prior_type = max_offset_px, prior
        self.stem = nn.Conv2d(3, dim, patch, stride=patch)
        self.value_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.offset_proj = nn.Linear(dim, n_heads * n_points * 2)
        self.weight_proj = nn.Linear(dim, n_heads * n_points)
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        nn.init.zeros_(self.offset_proj.weight)
        nn.init.zeros_(self.offset_proj.bias)

    def forward(self, frame: Tensor, xy: Tensor) -> tuple[Tensor, Tensor]:
        bt, channels, height, width = frame.shape
        if xy.shape != (bt, self.n_landmarks, 2):
            raise ValueError(f"Expected landmarks [BT,{self.n_landmarks},2], got {xy.shape}")
        if not bool(torch.isfinite(xy).all()):
            raise ValueError("Non-finite landmarks")
        feature = self.stem(frame)
        qgrid = xy.clamp(0, 1).mul(2).sub(1).unsqueeze(2)  # [BT,L,1,2]
        reference = F.grid_sample(feature, qgrid, align_corners=True,
                                  padding_mode="border").squeeze(-1).transpose(1, 2)
        value = self.value_proj(feature)
        if self.prior_type == "static":
            # Actual fixed Gaussian ROI pooling around reference anchors, not a
            # learned-offset substitute. Width is an explicit approximation,
            # because the paper does not disclose the Gaussian sigma.
            fh, fw = value.shape[-2:]
            y_grid, x_grid = torch.meshgrid(
                torch.linspace(-1, 1, fh, device=value.device, dtype=value.dtype),
                torch.linspace(-1, 1, fw, device=value.device, dtype=value.dtype),
                indexing="ij")
            grid = torch.stack((x_grid, y_grid), dim=-1).reshape(1, 1, fh * fw, 2)
            sigma = max(1.0, 0.75 * self.patch)
            scale = value.new_tensor([2 * sigma / max(width - 1, 1),
                                      2 * sigma / max(height - 1, 1)])
            dist2 = (((grid - qgrid) / scale) ** 2).sum(-1)  # [BT,L,F]
            weights = torch.softmax(-0.5 * dist2, dim=-1)
            h_values = value.reshape(bt, self.heads, self.dim // self.heads, fh * fw)
            pooled = torch.einsum("bls,bhds->blhd", weights, h_values)
            mean_offset_l2 = value.new_zeros(())
        elif self.prior_type == "deformable":
            offsets = self.offset_proj(reference).reshape(
                bt, self.n_landmarks, self.heads, self.points, 2).tanh() * self.max_offset_px
            delta_grid = offsets * offsets.new_tensor([2.0 / max(width - 1, 1),
                                                        2.0 / max(height - 1, 1)])
            sample_grid = (qgrid[:, :, None, :, :] + delta_grid).clamp(-1, 1)
            sample_grid = sample_grid.reshape(bt, self.n_landmarks * self.heads * self.points, 1, 2)
            sampled = F.grid_sample(value, sample_grid, align_corners=True,
                                    padding_mode="border")
            sampled = sampled.squeeze(-1).transpose(1, 2).reshape(
                bt, self.n_landmarks, self.heads, self.points,
                self.heads, value.shape[1] // self.heads)
            # Select only the value-channel slice belonging to each head.
            samples = sampled.diagonal(dim1=2, dim2=4).permute(0, 1, 4, 2, 3)
            weights = self.weight_proj(reference).reshape(
                bt, self.n_landmarks, self.heads, self.points).softmax(dim=-1)
            pooled = (samples * weights.unsqueeze(-1)).sum(dim=-2)
            mean_offset_l2 = offsets.norm(dim=-1).mean()
        else:
            raise ValueError("prior must be static or deformable")
        prior = self.out(pooled.flatten(start_dim=2))
        return prior, mean_offset_l2


class DeMSSM(nn.Module):
    def __init__(self, image_size: int = 224, frames: int = 64, patch: int = 16,
                 dim: int = 256, state_size: int = 16, n_classes: int = 3,
                 landmarks: int = 5, deform_heads: int = 4, deform_points: int = 4,
                 cross_heads: int = 4, max_offset_px: float = 24.0,
                 backbone: str = "vim", prior: str = "deformable", temporal: str = "ssm",
                 frame_chunk: int = 4, checkpoint_frames: bool = True,
                 variant: str = "dem_ssm"):
        super().__init__()
        if image_size % patch:
            raise ValueError("image_size must be divisible by patch size")
        if dim % cross_heads:
            raise ValueError("dim must be divisible by cross heads")
        self.image_size, self.frames = image_size, frames
        self.frame_chunk, self.checkpoint_frames = frame_chunk, checkpoint_frames
        self.backbone_name, self.temporal_name = backbone, temporal
        self.spatial = VisionPatchEncoder(dim, patch, state_size, backbone)
        self.prior = DeformableFacialPrior(dim, patch, deform_heads, deform_points,
                                          landmarks, max_offset_px, prior)
        self.cross_attn = nn.MultiheadAttention(dim, cross_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.position = nn.Parameter(torch.zeros(1, frames, dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        if temporal == "ssm":
            self.temporal_ssm = SelectiveSSM(dim, state_size)
        elif temporal == "ema":
            self.ema_alpha = nn.Parameter(torch.tensor(0.0))
        else:
            raise ValueError("temporal must be ssm or ema")
        self.pool = nn.Linear(dim, 1)
        self.classifier = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim),
                                        nn.GELU(), nn.Linear(dim, n_classes))

    def _one_chunk(self, frames: Tensor, landmarks: Tensor) -> tuple[Tensor, Tensor]:
        spatial = self.spatial(frames)
        anchors, penalty = self.prior(frames, landmarks)
        attended, _ = self.cross_attn(spatial, anchors, anchors, need_weights=False)
        fused = self.norm(spatial + attended)
        return fused.mean(dim=1), penalty

    def forward(self, video: Tensor, landmarks: Tensor, return_details: bool = False):
        if video.ndim != 5:
            raise ValueError("video must be [B,T,3,H,W]")
        b, t, c, h, w = video.shape
        if c != 3 or h != self.image_size or w != self.image_size or t > self.frames:
            raise ValueError("Unexpected clip shape or number of frames")
        if landmarks.shape != (b, t, self.prior.n_landmarks, 2):
            raise ValueError("landmarks must be [B,T,5,2]")
        flat = video.reshape(b * t, c, h, w)
        anchor = landmarks.reshape(b * t, self.prior.n_landmarks, 2)
        features, penalties = [], []
        for start in range(0, b * t, self.frame_chunk):
            f = flat[start:start + self.frame_chunk]
            lm = anchor[start:start + self.frame_chunk]
            if self.training and self.checkpoint_frames:
                encoded, penalty = checkpoint(self._one_chunk, f, lm, use_reentrant=False)
            else:
                encoded, penalty = self._one_chunk(f, lm)
            features.append(encoded)
            penalties.append((penalty, f.shape[0]))
        frame_tokens = torch.cat(features, 0).reshape(b, t, -1) + self.position[:, :t]
        if self.temporal_name == "ssm":
            output, memory_gate = self.temporal_ssm(frame_tokens, return_gate=True)
        else:
            alpha = torch.sigmoid(self.ema_alpha)
            state = frame_tokens[:, 0]
            out = [state]
            for i in range(1, t):
                state = alpha * state + (1 - alpha) * frame_tokens[:, i]
                out.append(state)
            output = torch.stack(out, 1)
            memory_gate = output.new_full((b, t), float(alpha.detach()))
        temporal_weights = torch.softmax(self.pool(output).squeeze(-1), dim=1)
        aggregated = (temporal_weights.unsqueeze(-1) * output).sum(dim=1)
        logits = self.classifier(aggregated)
        regularizer = sum(p * n for p, n in penalties) / (b * t)
        if return_details:
            return {"logits": logits, "offset_l2_px": regularizer,
                    "temporal_pool_weights": temporal_weights,
                    "internal_ssm_gate": memory_gate}
        return logits, regularizer
