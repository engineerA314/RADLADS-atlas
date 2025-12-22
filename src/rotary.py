import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math

from typing import Tuple

def generate_rotary_embedding(max_seqlen:int, dim:int, theta:float = 10000.0, scale:float = 1):
    angular_velocity = theta ** -(torch.arange(0, dim, 2, dtype=torch.float) / dim) / scale # frequencies from 1.0 ... 1/theta
    angles = torch.outer(torch.arange(max_seqlen), angular_velocity)
    return torch.polar(torch.ones_like(angles), angles)

def generate_rotary_cos_sin(max_seqlen:int, dim:int, theta:float = 10000.0, scale:float = 1):
    """
    Real-valued RoPE tables (cos/sin), avoids complex buffers.
    Shape: [max_seqlen, dim/2]
    """
    angular_velocity = theta ** -(torch.arange(0, dim, 2, dtype=torch.float32) / dim) / scale
    angles = torch.outer(torch.arange(max_seqlen, dtype=torch.float32), angular_velocity)
    return angles.cos(), angles.sin()

def generate_binary_rotary_embedding(max_seqlen:int, dim:int, scale:float=1):
    arange = torch.arange(dim // 2)
    angular_velocity = math.pi * (2.0 ** -arange) / scale # fastest velocity will rotate fully in two steps
    angular_velocity[int(math.log2(max_seqlen)):] = 0.0 # don't supply velocities slower than the one that will get a single full rotation across the seqlen
    #angular_velocity[20:] = 0.0 # don't supply velocities slower than the one that will get a single full rotation across 1024k ctxlen
    angles = torch.outer(torch.arange(max_seqlen), angular_velocity)
    return torch.polar(torch.ones_like(angles), angles)

def apply_rotary_embedding(q, k, cos: Tensor, sin: Tensor, seq_dim:int = -2) -> Tuple[Tensor, Tensor]:
    """
    Apply RoPE using precomputed real cos/sin tables.
    Supports q/k shaped either [B, T, D] or [B, H, T, D].
    """
    if cos.numel() == 0:
        return q, k

    q_dtype, k_dtype = q.dtype, k.dtype

    def _apply(x: Tensor, cos_t: Tensor, sin_t: Tensor) -> Tensor:
        if x.ndim == 3:
            B, T, D = x.shape
            L = T
            cos_l = cos_t[-L:].view(1, L, -1)  # [1, L, D/2]
            sin_l = sin_t[-L:].view(1, L, -1)
            x_ = x.float().reshape(B, L, -1, 2)  # [B, L, D/2, 2]
            x0 = x_[..., 0]
            x1 = x_[..., 1]
            out0 = x0 * cos_l - x1 * sin_l
            out1 = x0 * sin_l + x1 * cos_l
            out = torch.stack([out0, out1], dim=-1).reshape(B, L, D)
            return out.to(x.dtype)
        else:
            B, H, T, D = x.shape
            L = T
            cos_l = cos_t[-L:].view(1, 1, L, -1)  # [1, 1, L, D/2]
            sin_l = sin_t[-L:].view(1, 1, L, -1)
            x_ = x.float().reshape(B, H, L, -1, 2)  # [B, H, L, D/2, 2]
            x0 = x_[..., 0]
            x1 = x_[..., 1]
            out0 = x0 * cos_l - x1 * sin_l
            out1 = x0 * sin_l + x1 * cos_l
            out = torch.stack([out0, out1], dim=-1).reshape(B, H, L, D)
            return out.to(x.dtype)

    q = _apply(q, cos, sin).to(q_dtype)
    k = _apply(k, cos, sin).to(k_dtype)
    return q, k

class RotaryEmbedding(nn.Module):
    def __init__(self, max_sequence_length:int, dim:int, seq_dim:int = -2, theta:float = 10000):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.dim = dim
        self.theta = theta
        self.seq_dim = seq_dim

        # Cache (NOT buffers): this avoids `to_empty()` wiping RoPE tables.
        # We rebuild on demand (original RADLADS-paper style).
        self.cos: Tensor | None = None  # [L, dim/2] float32 on target device
        self.sin: Tensor | None = None  # [L, dim/2] float32 on target device

    def _ensure_tables_(self, *, device: torch.device, seq_len: int) -> Tuple[Tensor, Tensor]:
        if seq_len > self.max_sequence_length:
            raise ValueError(
                f"RoPE seq_len {seq_len} exceeds max_sequence_length {self.max_sequence_length}"
            )

        # Round up like the original code often does (helps avoid frequent rebuilds).
        need_len = min(self.max_sequence_length, ((seq_len + 15) // 16) * 16)

        def _cache_valid(cos: Tensor, sin: Tensor) -> bool:
            # RoPE invariant at pos0: angles=0 => cos=1, sin=0 for all dims.
            if cos.numel() == 0 or sin.numel() == 0:
                return False
            cos0 = cos[0]
            sin0 = sin[0]
            if (not torch.isfinite(cos0).all()) or (not torch.isfinite(sin0).all()):
                return False
            if not torch.allclose(cos0, torch.ones_like(cos0), atol=1e-3, rtol=0.0):
                return False
            if not torch.allclose(sin0, torch.zeros_like(sin0), atol=1e-3, rtol=0.0):
                return False
            return True

        # If cache is missing, on wrong device, too short, or invalid → rebuild.
        if (
            self.cos is None
            or self.sin is None
            or self.cos.device != device
            or self.sin.device != device
            or self.cos.size(0) < need_len
            or self.sin.size(0) < need_len
            or (not _cache_valid(self.cos, self.sin))
        ):
            # Build directly on target device in fp32.
            angular_velocity = self.theta ** -(
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim
            )
            angles = torch.outer(
                torch.arange(need_len, device=device, dtype=torch.float32),
                angular_velocity,
            )
            cos = angles.cos()
            sin = angles.sin()

            # Sanity: RoPE invariant at pos0: cos=1, sin=0 (helps catch accidental corruption).
            cos0 = cos[0]
            sin0 = sin[0]
            if not torch.allclose(cos0, torch.ones_like(cos0), atol=1e-3, rtol=0.0) or not torch.allclose(
                sin0, torch.zeros_like(sin0), atol=1e-3, rtol=0.0
            ):
                raise RuntimeError("RoPE table build invariant failed (cos/sin at position 0).")

            self.cos = cos
            self.sin = sin

        return self.cos, self.sin

    def forward(self, q, k):
        # q/k shape can be [B, T, D] or [B, H, T, D]; seq_len is always dim=-2.
        seq_len = q.shape[-2]
        cos, sin = self._ensure_tables_(device=q.device, seq_len=seq_len)
        return apply_rotary_embedding(q, k, cos, sin, self.seq_dim)
