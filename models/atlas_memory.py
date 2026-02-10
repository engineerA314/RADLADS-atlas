"""
Atlas RNN Memory: Explicit RNN-form implementation for RADLADS-atlas.

Ported from atlas-rnn/atlas_pytorch/rnn_memory.py with minimal dependencies.
Uses assoc_scan library for efficient parallel computation.

Key equations (per-token semantics):
- Surprise: δ_t = S_{t-1} φ_t - v_t
- Gradient: g_t = δ_t φ_t^T
- Momentum: Z_t = β_t Z_{t-1} + g_t
- Update: S_t = α_t S_{t-1} - η_t Z_t
- Retrieval: y_t = S_{t-1} ψ_t

Parallelization via efficient scalar scan (using assoc_scan library):
- Compute gradients using fixed S_0 (chunk-start state) for all tokens in parallel
- Apply scalar-gated recurrence: S_t = α_t * S_{t-1} + δ_t
- Same algorithm as original Titans paper
"""

from __future__ import annotations
from typing import NamedTuple
from functools import partial
import os
import json

import math
import warnings
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Module, Parameter, Linear, Conv1d

from assoc_scan import AssocScan
from einops import rearrange, repeat


# ============================================================================
# Helper functions
# ============================================================================

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

LinearNoBias = partial(Linear, bias=False)

# ============================================================================
# NaN debug helpers (omega CUDA I/O capture)
# ============================================================================

_NAN_DEBUG_OMEGA_DUMPED: set[tuple[int, int, int]] = set()


def _nan_debug_omega_steps() -> set[int]:
    raw = str(os.environ.get("NAN_DEBUG_OMEGA_STEPS", "") or "").strip()
    if raw == "":
        return set()
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if part == "":
            continue
        try:
            out.add(int(part))
        except Exception:
            continue
    return out


def _nan_debug_omega_layers() -> set[int] | None:
    raw = str(os.environ.get("NAN_DEBUG_OMEGA_LAYERS", "") or "").strip()
    if raw == "" or raw.lower() == "all":
        return None
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if part == "":
            continue
        try:
            out.add(int(part))
        except Exception:
            continue
    return out


def _nan_debug_omega_ranks() -> set[int] | None:
    raw = str(os.environ.get("NAN_DEBUG_OMEGA_RANKS", "") or "").strip()
    if raw == "" or raw.lower() == "all":
        return None
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if part == "":
            continue
        try:
            out.add(int(part))
        except Exception:
            continue
    return out


def _nan_debug_omega_dir() -> str:
    return str(os.environ.get("NAN_DEBUG_OMEGA_DIR", "/tmp/radlads_nan_debug_omega"))


def _nan_debug_omega_once() -> bool:
    raw = str(os.environ.get("NAN_DEBUG_OMEGA_ONCE", "1") or "").strip().lower()
    return raw in ("1", "true", "yes")


def _nan_debug_omega_inputs_enabled() -> bool:
    raw = str(os.environ.get("NAN_DEBUG_OMEGA_DUMP_INPUTS", "1") or "").strip().lower()
    return raw in ("1", "true", "yes")


def _nan_debug_omega_outputs_enabled() -> bool:
    raw = str(os.environ.get("NAN_DEBUG_OMEGA_DUMP_OUTPUTS", "1") or "").strip().lower()
    return raw in ("1", "true", "yes")


def _nan_debug_omega_dump_x_enabled() -> bool:
    raw = str(os.environ.get("NAN_DEBUG_OMEGA_DUMP_X", "0") or "").strip().lower()
    return raw in ("1", "true", "yes")


def _nan_debug_omega_context() -> dict:
    try:
        from atlasattn import get_nan_debug_context
        return get_nan_debug_context()
    except Exception:
        return {}


def _nan_debug_should_dump_omega(layer_idx: int) -> tuple[bool, dict, int | None, int]:
    steps = _nan_debug_omega_steps()
    if not steps:
        return False, {}, None, 0
    ctx = _nan_debug_omega_context()
    batch_idx = ctx.get("batch_idx", None)
    if batch_idx is None or int(batch_idx) not in steps:
        return False, ctx, None, 0
    try:
        import torch.distributed as dist
        is_dist = dist.is_available() and dist.is_initialized()
        rank = int(dist.get_rank()) if is_dist else 0
    except Exception:
        rank = 0
    ranks = _nan_debug_omega_ranks()
    if ranks is not None and rank not in ranks:
        return False, ctx, int(batch_idx), rank
    layers = _nan_debug_omega_layers()
    if layers is not None and layer_idx not in layers:
        return False, ctx, int(batch_idx), rank
    key = (int(batch_idx), int(layer_idx), int(rank))
    if _nan_debug_omega_once() and key in _NAN_DEBUG_OMEGA_DUMPED:
        return False, ctx, int(batch_idx), rank
    _NAN_DEBUG_OMEGA_DUMPED.add(key)
    return True, ctx, int(batch_idx), rank


def _nan_debug_omega_dump_tensor_dict(path: str, payload: dict) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass
    try:
        torch.save(payload, path)
    except Exception:
        pass


def _nan_debug_omega_dump_meta(path: str, meta: dict) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass
    try:
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass


# ============================================================================
# RNN Memory State
# ============================================================================

class RNNMemState(NamedTuple):
    """State for RNN memory.
    
    Attributes:
        seq_index: Current sequence position
        S: Memory state [batch*heads, d_v, d_phi]
        Z: Momentum state (optional, None if momentum disabled)
        omega_buffer: Rolling buffer for Omega window (optional)
    """
    seq_index: int              
    S: Tensor                   
    Z: Tensor | None            
    omega_buffer: Tensor | None 


def state_detach(state: RNNMemState) -> RNNMemState:
    """Detach all tensors in state from computation graph."""
    return RNNMemState(
        seq_index=state.seq_index,
        S=state.S.detach() if exists(state.S) else None,
        Z=state.Z.detach() if exists(state.Z) else None,
        omega_buffer=state.omega_buffer.detach() if exists(state.omega_buffer) else None,
    )


def _sliding_sum_along_time(x: Tensor, window: int) -> Tensor:
    """
    Sliding window sum along dim=1 (time), inclusive.
    x: [B, T, ...] -> y_t = sum_{k=max(0,t-window+1)}^t x_k
    """
    if window <= 1:
        return x
    T = x.shape[1]
    window = min(window, T)
    c = x.cumsum(dim=1)
    shifted = torch.cat([c.new_zeros((*c.shape[:1], window, *c.shape[2:])), c[:, :-window]], dim=1)
    return c - shifted


# ============================================================================
# Multi-head RMS Norm
# ============================================================================

class MultiheadRMSNorm(Module):
    """Per-head RMS normalization."""
    
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** -0.5
        self.gamma = Parameter(torch.ones(heads, 1, dim))
    
    def forward(self, x: Tensor) -> Tensor:
        rms = x.norm(dim=-1, keepdim=True) * self.scale
        return (x / rms.clamp(min=1e-8)) * self.gamma


# ============================================================================
# Polynomial Feature Maps
# ============================================================================

class PolynomialFeatureMap(Module):
    """
    Feature map for attention kernels.

    When degree=1, automatically uses identity mode regardless of requested mode.

    Modes:
    - 'identity' (auto when degree=1): φ(x) = x (no transformation)
    - 'off': φ(x) = x
    - 'elementwise': φ(x) = Σ_{i=1}^{degree} x^i
    - 'tensor': φ(x) = RandomProj([x, vec(x ⊗ x)])
    - 'polysketch': φ(x) = RandomProj([1, x, TensorSketch(x, x)])  (degree=2 only)
    """

    def __init__(self, dim: int, degree: int = 1, mode: str = 'off', sketch_size: int | None = None):
        super().__init__()
        self.dim = dim
        self.degree = degree

        # Force identity mode for degree=1 (regardless of requested mode)
        if degree == 1:
            self.mode = 'identity'
            if mode != 'identity' and mode != 'off':
                warnings.warn(
                    f"degree=1 forces identity mode (requested mode='{mode}' ignored)"
                )
        elif mode == 'tensor':
            self.mode = 'tensor'
            feat_dim = dim + dim * dim
            self.register_buffer(
                'proj',
                torch.randn(feat_dim, dim) / math.sqrt(feat_dim)
            )
        elif mode == 'polysketch':
            self.mode = 'polysketch'
            if degree != 2:
                warnings.warn(
                    f"PolynomialFeatureMap mode='polysketch' only supports degree=2; "
                    f"got degree={degree}. Forcing degree=2."
                )
                self.degree = 2
            self.sketch_size = default(sketch_size, dim)
            feat_dim = dim + self.sketch_size + 1
            self.proj = LinearNoBias(feat_dim, dim)
            # Trainable Taylor-series coefficients for degree-0/1/2 terms.
            # Initialize to exp(.) Taylor coefficients: 1, 1, 1/2.
            self.poly_coeff = Parameter(torch.tensor([1.0, 1.0, 0.5]))
            self.register_buffer(
                'sketch_h1',
                torch.randint(self.sketch_size, (dim,), dtype=torch.int64)
            )
            self.register_buffer(
                'sketch_h2',
                torch.randint(self.sketch_size, (dim,), dtype=torch.int64)
            )
            self.register_buffer(
                'sketch_s1',
                (torch.randint(0, 2, (dim,), dtype=torch.int8) * 2 - 1)
            )
            self.register_buffer(
                'sketch_s2',
                (torch.randint(0, 2, (dim,), dtype=torch.int8) * 2 - 1)
            )
        else:
            # 'off', 'elementwise', or explicit 'identity' with degree > 1
            if mode == 'identity' and degree > 1:
                warnings.warn(
                    f"mode='identity' only valid for degree=1, got degree={degree}. "
                    f"Using 'elementwise' instead."
                )
                self.mode = 'elementwise'
            else:
                self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        if self.mode == 'identity':
            return x

        if self.mode == 'off':
            return x

        if self.mode == 'elementwise':
            if self.degree <= 1:
                return x
            out = x.clone()
            power = x
            for _ in range(2, self.degree + 1):
                power = power * x
                out = out + power
            return out

        if self.mode == 'tensor':
            if self.degree <= 1:
                return x
            d = x.shape[-1]
            outer = torch.einsum('...i,...j->...ij', x, x)
            outer = outer.reshape(*x.shape[:-1], d * d)
            feats = torch.cat([x, outer], dim=-1)
            return feats @ self.proj.to(x.device, x.dtype)

        if self.mode == 'polysketch':
            return self._forward_polysketch(x)

        return x

    def _count_sketch(self, x: Tensor, h: Tensor, s: Tensor) -> Tensor:
        d = x.shape[-1]
        x_flat = x.reshape(-1, d)
        h = h.to(x_flat.device)
        s = s.to(x_flat.device)
        if s.dtype != x_flat.dtype:
            s = s.to(x_flat.dtype)
        out = x_flat.new_zeros(x_flat.shape[0], self.sketch_size)
        out.scatter_add_(1, h.expand(x_flat.shape[0], -1), x_flat * s)
        return out.view(*x.shape[:-1], self.sketch_size)

    def _forward_polysketch(self, x: Tensor) -> Tensor:
        sketch1 = self._count_sketch(x, self.sketch_h1, self.sketch_s1)
        sketch2 = self._count_sketch(x, self.sketch_h2, self.sketch_s2)
        if sketch1.dtype not in (torch.float32, torch.float64):
            sketch1_f = sketch1.float()
            sketch2_f = sketch2.float()
        else:
            sketch1_f = sketch1
            sketch2_f = sketch2
        fft1 = torch.fft.rfft(sketch1_f, n=self.sketch_size)
        fft2 = torch.fft.rfft(sketch2_f, n=self.sketch_size)
        sketch = torch.fft.irfft(fft1 * fft2, n=self.sketch_size)
        if sketch.dtype != x.dtype:
            sketch = sketch.to(x.dtype)
        coeff = self.poly_coeff.to(dtype=x.dtype, device=x.device)
        const_scaled = x.new_ones(*x.shape[:-1], 1) * coeff[0]
        x_scaled = x * coeff[1]
        sketch_scaled = sketch * coeff[2]
        feats = torch.cat([const_scaled, x_scaled, sketch_scaled], dim=-1)
        return self.proj(feats)


# ============================================================================
# einops-free head reshape utilities
# ============================================================================

def split_heads(x: Tensor, heads: int) -> Tensor:
    """[B, N, H*D] -> [B, H, N, D]"""
    B, N, HD = x.shape
    D = HD // heads
    return x.view(B, N, heads, D).transpose(1, 2)

def merge_heads(x: Tensor) -> Tensor:
    """[B, H, N, D] -> [B, N, H*D]"""
    B, H, N, D = x.shape
    return x.transpose(1, 2).reshape(B, N, H * D)

def rearrange_bhn_to_bh(x: Tensor, batch: int, heads: int, seq_len: int) -> Tensor:
    """[B, H, N, D] -> [(B*H), N, D]"""
    return x.reshape(batch * heads, seq_len, -1)


# ============================================================================
# RNN Memory Cell (Titans-RNN) - Per-token semantics with scalar scan
# ============================================================================

class RNNMemoryCell(Module):
    """
    Per-token RNN memory update with scalar scan parallelization.
    
    Implements (per-token):
        g_t = (S_0 φ_t - v_t) φ_t^T              (gradient using fixed S_0)
        Z_t = β_t Z_{t-1} + g_t                  (momentum)
        S_t = α_t S_{t-1} - η_t Z_t              (memory update)
        y_t = S_{t-1} ψ_t                        (retrieval)
    
    Parallelized via scalar scan using assoc_scan library.
    """
    
    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 1,
        use_momentum: bool = True,
        poly_degree: int = 2,  # Default: 2
        poly_mode: str = 'elementwise',  # Default: elementwise
        qk_norm: bool = False,  # Default: False
        qkv_conv_kernel: int | None = None,  # Default: None
        use_accelerated_scan: bool = False,
        scan_chunk_len: int | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.use_momentum = use_momentum
        self.qkv_conv_kernel = qkv_conv_kernel
        self.scan_chunk_len = scan_chunk_len
        
        # Associative scan for parallel computation (use_accelerated=True prefers Triton).
        # IMPORTANT: assoc-scan imports `accelerated_scan.triton` lazily inside forward(),
        # so we proactively validate availability here to avoid runtime crashes.
        if use_accelerated_scan:
            try:
                import accelerated_scan.triton  # noqa: F401
            except Exception as e:
                warnings.warn(
                    f"accelerated_scan.triton unavailable; disabling accelerated scan. Error: {e}"
                )
                use_accelerated_scan = False
        self.assoc_scan = AssocScan(use_accelerated=use_accelerated_scan)
        
        dim_inner = dim_head * heads
        
        # Projections
        self.activation = nn.Identity()
        self.to_q = LinearNoBias(dim, dim_inner)
        self.to_k = LinearNoBias(dim, dim_inner)
        self.to_v = LinearNoBias(dim, dim_inner)
        self.to_out = LinearNoBias(dim_inner, dim)

        # Optional depthwise conv for Q/K/V
        if exists(qkv_conv_kernel) and qkv_conv_kernel > 1:
            padding = qkv_conv_kernel // 2
            self.q_conv = Conv1d(dim, dim, qkv_conv_kernel, padding=padding, groups=dim, bias=False)
            self.k_conv = Conv1d(dim, dim, qkv_conv_kernel, padding=padding, groups=dim, bias=False)
            self.v_conv = Conv1d(dim, dim, qkv_conv_kernel, padding=padding, groups=dim, bias=False)
        else:
            self.q_conv = None
            self.k_conv = None
            self.v_conv = None

        # Learned hyperparameters (per-head)
        self.to_lr = nn.Sequential(Linear(dim, heads), nn.Sigmoid())
        nn.init.zeros_(self.to_lr[0].weight)
        nn.init.constant_(self.to_lr[0].bias, -4.0)  # sigmoid(-4) ≈ 0.018
        
        self.to_decay = nn.Sequential(Linear(dim, heads), nn.Sigmoid())
        nn.init.zeros_(self.to_decay[0].weight)
        nn.init.constant_(self.to_decay[0].bias, 4.0)  # sigmoid(4) ≈ 0.982
        
        self.to_momentum = nn.Sequential(Linear(dim, heads), nn.Sigmoid()) if use_momentum else None
        if use_momentum:
            nn.init.zeros_(self.to_momentum[0].weight)
            nn.init.constant_(self.to_momentum[0].bias, 2.0)  # sigmoid(2) ≈ 0.88
        
        # Norms
        self.pre_norm = nn.RMSNorm(dim)
        self.q_norm = MultiheadRMSNorm(dim_head, heads) if qk_norm else nn.Identity()
        self.k_norm = MultiheadRMSNorm(dim_head, heads) if qk_norm else nn.Identity()
        
        # Polynomial feature map
        self.phi = PolynomialFeatureMap(dim_head, poly_degree, poly_mode)
        
        self.register_buffer('_dummy', torch.zeros(1))
    
    def init_state(self, batch: int, device=None, dtype=None) -> RNNMemState:
        """Initialize memory state."""
        device = default(device, self._dummy.device)
        dtype = default(dtype, self._dummy.dtype)
        
        S = torch.zeros(batch * self.heads, self.dim_head, self.dim_head, 
                       device=device, dtype=dtype)
        Z = torch.zeros_like(S) if self.use_momentum else None
        
        return RNNMemState(seq_index=0, S=S, Z=Z, omega_buffer=None)

    def forward(
        self,
        x: Tensor,
        state: RNNMemState | None = None,
    ) -> tuple[Tensor, RNNMemState]:
        """
        Forward pass with per-token semantics.
        
        Uses scalar scan for O(log T) parallelization while maintaining
        equivalent update semantics (mini-batch SGD style).
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            state: Optional previous state
            
        Returns:
            retrieved: Output tensor [batch, seq_len, dim]
            next_state: Updated memory state
        """
        batch, seq_len, _ = x.shape
        
        if not exists(state):
            state = self.init_state(batch, x.device, x.dtype)
        
        # Chunked scan: reduces peak memory by computing [BH, chunk, d, d] tensors per chunk
        # instead of materializing them for the entire sequence at once.
        chunk_len = self.scan_chunk_len
        if exists(chunk_len) and seq_len > chunk_len:
            if exists(self.q_conv) or exists(self.k_conv) or exists(self.v_conv):
                raise NotImplementedError(
                    "scan_chunk_len is not supported with qkv_conv_kernel > 1 (conv mixes across chunk boundaries)."
                )
            outs: list[Tensor] = []
            cur_state = state
            # Preserve current (unchunked) semantics for g: use a fixed reference state from the start of the call.
            S_ref = cur_state.S
            for start in range(0, seq_len, chunk_len):
                end = min(seq_len, start + chunk_len)
                out, cur_state = self._forward_impl(x[:, start:end], cur_state, S_ref=S_ref)
                outs.append(out)
            return torch.cat(outs, dim=1), cur_state

        return self._forward_impl(x, state, S_ref=state.S)

    def _forward_impl(
        self,
        x: Tensor,
        state: RNNMemState,
        *,
        S_ref: Tensor,
    ) -> tuple[Tensor, RNNMemState]:
        batch, seq_len, _ = x.shape

        d = self.dim_head
        BH = batch * self.heads
        
        # Pre-norm + activation
        x = self.activation(self.pre_norm(x))

        q_in, k_in, v_in = x, x, x

        # Optional depthwise conv
        if exists(self.q_conv):
            q_in = self.q_conv(q_in.transpose(1, 2)).transpose(1, 2)
        if exists(self.k_conv):
            k_in = self.k_conv(k_in.transpose(1, 2)).transpose(1, 2)
        if exists(self.v_conv):
            v_in = self.v_conv(v_in.transpose(1, 2)).transpose(1, 2)
        
        # Project to Q, K, V and split heads: [batch, heads, seq, dim_head]
        q = split_heads(self.to_q(q_in), self.heads)
        k = split_heads(self.to_k(k_in), self.heads)
        v = split_heads(self.to_v(v_in), self.heads)
        
        # Truncate to original seq_len in case conv changed length
        q = q[:, :, :seq_len]
        k = k[:, :, :seq_len]
        v = v[:, :, :seq_len]
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Flatten batch*heads for processing
        # Apply polynomial feature map
        k_flat = k.reshape(batch * self.heads * seq_len, d)
        q_flat = q.reshape(batch * self.heads * seq_len, d)
        
        phi_k = self.phi(k_flat).view(BH, seq_len, d)
        phi_q = self.phi(q_flat).view(BH, seq_len, d)
        v_bh = v.reshape(BH, seq_len, d)
        
        # Learned hyperparameters: [BH, T]
        lr = self.to_lr(x).transpose(1, 2).reshape(BH, seq_len)
        decay = self.to_decay(x).transpose(1, 2).reshape(BH, seq_len)
        
        if self.use_momentum:
            momentum = self.to_momentum(x).transpose(1, 2).reshape(BH, seq_len)
        
        # -----------------------------------------------------------------
        # Efficient Scalar Scan Implementation (Exact Refactor)
        # -----------------------------------------------------------------
        # Compute per-token gradient: g_t = (S_ref @ φ_t - v_t) ⊗ φ_t^T
        # This avoids materializing G and B tensors (memory optimization)
        
        # Get initial states (for recurrence)
        S0 = state.S  # [BH, d, d]
        Z0 = state.Z if exists(state.Z) else torch.zeros_like(S0)
        
        # Step 1: Compute prediction error (δ_t = S_ref @ φ_t - v_t)
        pred = torch.einsum('bde,bte->btd', S_ref, phi_k)  # [BH, T, d]
        delta_vec = pred - v_bh                             # [BH, T, d]
        
        # Step 2: Compute gradient (g_t = δ_t ⊗ φ_t^T)
        g = torch.einsum('bti,btj->btij', delta_vec, phi_k)  # [BH, T, d, d]
        
        # Expand scalars for broadcasting: [BH, T, 1, 1]
        lr_e = lr[..., None, None]
        
        if self.use_momentum:
            # Step 3: Momentum via scalar scan: Z_t = β_t * Z_{t-1} + g_t
            Z_all = self.assoc_scan(momentum, g, prev=Z0)  # [BH, T, d, d]
            
            # Step 4: Compute delta: δ_t = -η_t * Z_t
            delta = -lr_e * Z_all  # [BH, T, d, d]
            
            # Step 5: State update via scalar scan: S_t = α_t * S_{t-1} + δ_t
            S_all = self.assoc_scan(decay, delta, prev=S0)  # [BH, T, d, d]
            
            # Final states
            S_end = S_all[:, -1].clamp(-100, 100)
            Z_end = Z_all[:, -1]
        else:
            # No momentum: δ_t = -η_t * g_t
            delta = -lr_e * g  # [BH, T, d, d]
            
            # State update via scalar scan: S_t = α_t * S_{t-1} + δ_t
            S_all = self.assoc_scan(decay, delta, prev=S0)  # [BH, T, d, d]
            
            S_end = S_all[:, -1].clamp(-100, 100)
            Z_end = None
        
        # -----------------------------------------------------------------
        # Retrieval: y_t = S_{t-1} @ ψ_t
        # -----------------------------------------------------------------
        y0 = torch.einsum('bde,be->bd', S0, phi_q[:, 0])  # [BH, d]
        if seq_len > 1:
            y_rest = torch.einsum('btdp,btp->btd', S_all[:, :-1], phi_q[:, 1:])  # [BH, T-1, d]
            retrieved = torch.cat([y0.unsqueeze(1), y_rest], dim=1)  # [BH, T, d]
        else:
            retrieved = y0.unsqueeze(1)  # [BH, 1, d]
        
        # Reshape to [batch, heads, seq, dim_head] and merge
        retrieved = retrieved.view(batch, self.heads, seq_len, d)
        retrieved = merge_heads(retrieved)
        retrieved = self.to_out(retrieved)
        
        next_state = RNNMemState(
            seq_index=state.seq_index + seq_len,
            S=S_end,
            Z=Z_end,
            omega_buffer=None
        )
        
        return retrieved, next_state


# ============================================================================
# Omega RNN Memory Cell (OmegaNet-RNN) - Sliding window context
# ============================================================================

class OmegaRNNMemoryCell(Module):
    """
    RNN memory with Omega rule (sliding window context).
    
    For omega_window > 1:
        G_t = Σ_{p∈W_t} U_t^p φ_p φ_p^T   (Gram matrix over window)
        B_t = Σ_{p∈W_t} U_t^p v_p φ_p^T   (Cross term)
    
    Then same scalar scan as Titans-RNN.
    
    New features:
    - GQA (Grouped Query Attention): K/V use fewer heads than Q
    - ROPE (Rotary Position Embedding): Optional positional encoding
    - GroupNorm: Optional per-head normalization
    """
    
    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 1,
        num_key_value_heads: int | None = None,  # GQA: None means same as heads
        omega_window: int = 1,
        use_omega_gate: bool = True,
        use_momentum: bool = True,
        poly_degree: int = 2,  # Default: 2
        poly_mode: str = 'elementwise',  # Default: elementwise
        qk_norm: bool = False,  # Default: False
        qkv_conv_kernel: int | None = None,  # Default: None
        use_rope: bool = False,  # ROPE option
        rope_theta: float = 10000.0,  # ROPE base frequency
        max_position_embeddings: int = 2048,  # ROPE max positions
        use_groupnorm: bool = False,  # GroupNorm option
        use_accelerated_scan: bool = False,
        scan_chunk_len: int | None = None,
        use_cuda: bool = False,  # CUDA backend for 50-100x speedup
    ):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.omega_window = omega_window
        self.use_omega_gate = use_omega_gate
        self.use_momentum = use_momentum
        self.qkv_conv_kernel = qkv_conv_kernel
        self.scan_chunk_len = scan_chunk_len
        self.use_cuda = use_cuda
        
        # CUDA backend setup (only if omega_window == 16 for now)
        self.cuda_available = False
        self.cuda_backend_requested = use_cuda
        if use_cuda:
            if omega_window != 16:
                warnings.warn(f"CUDA backend only supports omega_window=16, got {omega_window}. Falling back to PyTorch.")
                self.use_cuda = False
            else:
                try:
                    # Ensure CUDA kernel matches the memory head dimension.
                    os.environ.setdefault("ATLAS_CUDA_DIM_HEAD", str(self.dim_head))
                    from .cuda.atlas_omega_autograd import (
                        check_cuda_availability, AtlasOmegaFunction, get_cuda_status
                    )
                    available, msg = check_cuda_availability()
                    if not available:
                        warnings.warn(f"CUDA not available: {msg}. Falling back to PyTorch.")
                        self.use_cuda = False
                    else:
                        self.AtlasOmegaFunction = AtlasOmegaFunction
                        self.cuda_available = True
                        status = get_cuda_status()
                        if str(os.environ.get("ATLAS_CUDA_VERBOSE", "0")).strip().lower() in ("1", "true", "yes"):
                            warnings.warn(
                                f"OmegaRNNMemoryCell using CUDA (omega_window=16, loaded_via={status['loaded_via']})"
                            )
                except ImportError as e:
                    warnings.warn(f"CUDA extension not compiled: {e}. Falling back to PyTorch.")
                    self.use_cuda = False
        
        # GQA setup
        self.num_key_value_heads = default(num_key_value_heads, heads)
        self.num_key_value_groups = heads // self.num_key_value_heads
        assert heads % self.num_key_value_heads == 0, \
            f"heads ({heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})"
        
        # ROPE setup
        self.use_rope = use_rope
        if use_rope:
            from src.rotary import RotaryEmbedding
            self.rotary_emb = RotaryEmbedding(
                max_sequence_length=max_position_embeddings,
                dim=dim_head,
                theta=rope_theta
            )
        else:
            self.rotary_emb = None
        
        # GroupNorm setup
        self.use_groupnorm = use_groupnorm
        if use_groupnorm:
            dim_inner = dim_head * heads
            self.groupnorm = nn.GroupNorm(
                num_groups=heads,
                num_channels=dim_inner,
                eps=64e-5
            )
        else:
            self.groupnorm = None
        
        # Associative scan for parallel computation (PyTorch backend only).
        #
        # If the CUDA backend is ACTIVE (omega_window=16 + compiled extension), we never call
        # `AssocScan` in the hot path (state update is done by the custom CUDA kernel), so
        # we should NOT warn about `accelerated_scan.triton` even if it's missing.
        #
        # If CUDA is NOT active (or not available), we may rely on AssocScan; in that case,
        # prefer the accelerated (Triton) path when installed, otherwise fall back silently.
        if use_accelerated_scan and not (self.use_cuda and self.cuda_available):
            try:
                import accelerated_scan.triton  # noqa: F401
            except Exception as e:
                warnings.warn(
                    f"accelerated_scan.triton unavailable; disabling accelerated scan. Error: {e}"
                )
                use_accelerated_scan = False
        self.assoc_scan = AssocScan(use_accelerated=use_accelerated_scan)
        
        dim_inner = dim_head * heads
        dim_kv = dim_head * self.num_key_value_heads  # GQA: reduced dimension for K/V
        
        # Projections (GQA: K/V use fewer heads)
        self.activation = nn.Identity()
        self.to_q = LinearNoBias(dim, dim_inner)
        self.to_k = LinearNoBias(dim, dim_kv)  # GQA: reduced
        self.to_v = LinearNoBias(dim, dim_kv)  # GQA: reduced
        self.to_out = LinearNoBias(dim_inner, dim)

        # Optional depthwise conv
        if exists(qkv_conv_kernel) and qkv_conv_kernel > 1:
            padding = qkv_conv_kernel // 2
            self.q_conv = Conv1d(dim, dim, qkv_conv_kernel, padding=padding, groups=dim, bias=False)
            self.k_conv = Conv1d(dim, dim, qkv_conv_kernel, padding=padding, groups=dim, bias=False)
            self.v_conv = Conv1d(dim, dim, qkv_conv_kernel, padding=padding, groups=dim, bias=False)
        else:
            self.q_conv = None
            self.k_conv = None
            self.v_conv = None
        
        # Learned hyperparameters
        self.to_lr = nn.Sequential(Linear(dim, heads), nn.Sigmoid())
        nn.init.zeros_(self.to_lr[0].weight)
        nn.init.constant_(self.to_lr[0].bias, -4.0)
        
        self.to_decay = nn.Sequential(Linear(dim, heads), nn.Sigmoid())
        nn.init.zeros_(self.to_decay[0].weight)
        nn.init.constant_(self.to_decay[0].bias, 4.0)
        
        self.to_momentum = nn.Sequential(Linear(dim, heads), nn.Sigmoid()) if use_momentum else None
        if use_momentum:
            nn.init.zeros_(self.to_momentum[0].weight)
            nn.init.constant_(self.to_momentum[0].bias, 2.0)
        
        # Omega gate
        self.to_omega_gate = nn.Sequential(Linear(dim, heads), nn.Sigmoid()) if use_omega_gate else None
        if use_omega_gate:
            nn.init.zeros_(self.to_omega_gate[0].weight)
            nn.init.constant_(self.to_omega_gate[0].bias, 0.0)
        
        # Norms (K norm uses num_kv_heads for GQA)
        self.pre_norm = nn.RMSNorm(dim)
        self.q_norm = MultiheadRMSNorm(dim_head, heads) if qk_norm else nn.Identity()
        self.k_norm = MultiheadRMSNorm(dim_head, self.num_key_value_heads) if qk_norm else nn.Identity()
        
        # Polynomial feature map
        self.phi = PolynomialFeatureMap(dim_head, poly_degree, poly_mode)
        
        self.register_buffer('_dummy', torch.zeros(1))
    
    def init_state(self, batch: int, device=None, dtype=None) -> RNNMemState:
        """Initialize memory state with omega buffer."""
        device = default(device, self._dummy.device)
        dtype = default(dtype, self._dummy.dtype)
        
        S = torch.zeros(batch * self.heads, self.dim_head, self.dim_head,
                       device=device, dtype=dtype)
        Z = torch.zeros_like(S) if self.use_momentum else None
        
        # Buffer for sliding window: stores g for last (e-1) tokens (Exact Refactor: 50% smaller!)
        if self.omega_window > 1:
            omega_buffer = torch.zeros(
                batch * self.heads, self.omega_window - 1, self.dim_head, self.dim_head,
                device=device, dtype=dtype
            )
        else:
            omega_buffer = None
        
        return RNNMemState(seq_index=0, S=S, Z=Z, omega_buffer=omega_buffer)

    def forward(
        self,
        x: Tensor,
        state: RNNMemState | None = None,
    ) -> tuple[Tensor, RNNMemState]:
        """Forward pass with Omega rule (sliding window)."""
        batch, seq_len, _ = x.shape

        if not exists(state):
            state = self.init_state(batch, x.device, x.dtype)

        chunk_len = self.scan_chunk_len
        if exists(chunk_len) and seq_len > chunk_len:
            if exists(self.q_conv) or exists(self.k_conv) or exists(self.v_conv):
                raise NotImplementedError(
                    "scan_chunk_len is not supported with qkv_conv_kernel > 1 (conv mixes across chunk boundaries)."
                )
            outs: list[Tensor] = []
            cur_state = state
            # Preserve current (unchunked) semantics for g: use a fixed reference state from the start of the call.
            S_ref = cur_state.S
            for start in range(0, seq_len, chunk_len):
                end = min(seq_len, start + chunk_len)
                out, cur_state = self._forward_impl(x[:, start:end], cur_state, S_ref=S_ref)
                outs.append(out)
            return torch.cat(outs, dim=1), cur_state

        return self._forward_impl(x, state, S_ref=state.S)


    def _forward_impl(
        self,
        x: Tensor,
        state: RNNMemState,
        *,
        S_ref: Tensor,
    ) -> tuple[Tensor, RNNMemState]:
        batch, seq_len, _ = x.shape
        e = self.omega_window
        d = self.dim_head
        BH = batch * self.heads
        layer_idx = int(getattr(self, "layer_idx", -1))
        
        # CUDA backend (50-100x faster, 100x less memory)
        if self.use_cuda and self.cuda_available and x.is_cuda:
            return self._forward_cuda(x, state, S_ref=S_ref)
        
        # Pre-norm and project
        x_normed = self.activation(self.pre_norm(x))
        layer_idx = int(getattr(self, "layer_idx", -1))
        
        q_in, k_in, v_in = x_normed, x_normed, x_normed
        
        if exists(self.q_conv):
            q_in = self.q_conv(q_in.transpose(1, 2)).transpose(1, 2)
        if exists(self.k_conv):
            k_in = self.k_conv(k_in.transpose(1, 2)).transpose(1, 2)
        if exists(self.v_conv):
            v_in = self.v_conv(v_in.transpose(1, 2)).transpose(1, 2)
        
        q = split_heads(self.to_q(q_in), self.heads)
        k = split_heads(self.to_k(k_in), self.num_key_value_heads)  # GQA: fewer heads
        v = split_heads(self.to_v(v_in), self.num_key_value_heads)  # GQA: fewer heads
        
        # Truncate to original seq_len in case conv changed length
        q = q[:, :, :seq_len]
        k = k[:, :, :seq_len]
        v = v[:, :, :seq_len]
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Apply ROPE if enabled
        if self.use_rope and self.rotary_emb is not None:
            # ROPE expects [B, H, T, D] shape
            q, k = self.rotary_emb(q, k)
        
        # GQA: Expand K/V to match Q heads
        if self.num_key_value_groups > 1:
            # k: [batch, num_kv_heads, seq, dim_head] -> [batch, heads, seq, dim_head]
            k = k.unsqueeze(2).expand(
                batch, self.num_key_value_heads, self.num_key_value_groups, seq_len, d
            ).reshape(batch, self.heads, seq_len, d)
            
            # v: [batch, num_kv_heads, seq, dim_head] -> [batch, heads, seq, dim_head]
            v = v.unsqueeze(2).expand(
                batch, self.num_key_value_heads, self.num_key_value_groups, seq_len, d
            ).reshape(batch, self.heads, seq_len, d)
        
        # Apply polynomial features
        k_flat = k.reshape(batch * self.heads * seq_len, d)
        q_flat = q.reshape(batch * self.heads * seq_len, d)
        
        phi_k = self.phi(k_flat).view(BH, seq_len, d)
        phi_q = self.phi(q_flat).view(BH, seq_len, d)
        v_bh = v.reshape(BH, seq_len, d)
        
        # Learned hyperparameters (use original x_normed which has original seq_len)
        lr = self.to_lr(x_normed).transpose(1, 2).reshape(BH, seq_len)
        decay = self.to_decay(x_normed).transpose(1, 2).reshape(BH, seq_len)
        
        if self.use_momentum:
            momentum = self.to_momentum(x_normed).transpose(1, 2).reshape(BH, seq_len)
        
        omega_gate = None
        if self.use_omega_gate:
            omega_gate = self.to_omega_gate(x_normed).transpose(1, 2).reshape(BH, seq_len)
        
        # -----------------------------------------------------------------
        # Per-token gradient (Exact Refactor: avoids G, B materialization)
        # -----------------------------------------------------------------
        # δ_t = S_ref @ φ_t - v_t (prediction error)
        pred = torch.einsum('bde,bte->btd', S_ref, phi_k)  # [BH, T, d]
        delta_vec = pred - v_bh                             # [BH, T, d]
        
        # g_raw_t = δ_t ⊗ φ_t^T (gradient before omega window)
        g_raw = torch.einsum('bti,btj->btij', delta_vec, phi_k)  # [BH, T, d, d]
        
        # Apply omega gate to gradient (not to G/B separately)
        if exists(omega_gate):
            gate_e = omega_gate[..., None, None]  # [BH, T, 1, 1]
            g_raw = g_raw * gate_e
        
        # -----------------------------------------------------------------
        # Omega window: sliding sum of g (single buffer, not G+B)
        # -----------------------------------------------------------------
        if e > 1:
            omega_buffer = state.omega_buffer
            if exists(omega_buffer):
                # Backward compatibility: handle old 5D format [BH, e-1, d, d, 2]
                if omega_buffer.ndim == 5:
                    # Old format: extract G (index 0) - will be converted to new format
                    prev_g = omega_buffer[..., 0]  # [BH, e-1, d, d]
                else:
                    # New format (Exact Refactor): single tensor [BH, e-1, d, d]
                    prev_g = omega_buffer
            else:
                prev_g = g_raw.new_zeros((BH, e - 1, d, d))
            
            g_ext = torch.cat([prev_g, g_raw], dim=1)  # [BH, e-1+T, d, d]
            g = _sliding_sum_along_time(g_ext, e)[:, -(seq_len):]  # [BH, T, d, d]
            
            # Update buffer for next call (single tensor, 50% memory saving)
            new_omega_buffer = g_ext[:, -(e - 1):]  # [BH, e-1, d, d]
        else:
            g = g_raw
            new_omega_buffer = None
        
        # -----------------------------------------------------------------
        # Efficient Scalar Scan Implementation (same as Titans-RNN)
        # -----------------------------------------------------------------
        # Get initial states (for recurrence)
        S0 = state.S  # [BH, d, d]
        Z0 = state.Z if exists(state.Z) else torch.zeros_like(S0)
        
        # Note: g is already computed above via Exact Refactor
        # g = (S_ref @ φ - v) ⊗ φ^T (mathematically equivalent to S_ref @ G - B)
        
        # Expand scalars for broadcasting: [BH, T, 1, 1]
        lr_e = lr[..., None, None]
        
        if self.use_momentum:
            # Momentum via scalar scan: Z_t = β_t * Z_{t-1} + g_t
            Z_all = self.assoc_scan(momentum, g, prev=Z0)  # [BH, T, d, d]
            
            # Compute delta: δ_t = -η_t * Z_t
            delta = -lr_e * Z_all  # [BH, T, d, d]
            
            # State update via scalar scan: S_t = α_t * S_{t-1} + δ_t
            S_all = self.assoc_scan(decay, delta, prev=S0)  # [BH, T, d, d]
            
            # Final states
            S_end = S_all[:, -1].clamp(-100, 100)
            Z_end = Z_all[:, -1]
        else:
            # No momentum: δ_t = -η_t * g_t
            delta = -lr_e * g  # [BH, T, d, d]
            
            # State update via scalar scan: S_t = α_t * S_{t-1} + δ_t
            S_all = self.assoc_scan(decay, delta, prev=S0)  # [BH, T, d, d]
            
            S_end = S_all[:, -1].clamp(-100, 100)
            Z_end = None
        
        # -----------------------------------------------------------------
        # Retrieval: y_t = S_{t-1} @ ψ_t
        # -----------------------------------------------------------------
        # Memory optimization: avoid materializing S_start ([BH, T, d, d]).
        y0 = torch.einsum('bde,be->bd', S0, phi_q[:, 0])  # [BH, d]
        if seq_len > 1:
            y_rest = torch.einsum('btdp,btp->btd', S_all[:, :-1], phi_q[:, 1:])  # [BH, T-1, d]
            y_bh = torch.cat([y0.unsqueeze(1), y_rest], dim=1)  # [BH, T, d]
        else:
            y_bh = y0.unsqueeze(1)  # [BH, 1, d]
        
        retrieved = y_bh.view(batch, self.heads, seq_len, d)
        retrieved = merge_heads(retrieved)
        
        # Apply GroupNorm if enabled (before output projection)
        if self.use_groupnorm and self.groupnorm is not None:
            # GroupNorm expects [B*T, C] shape
            retrieved = self.groupnorm(retrieved.reshape(batch * seq_len, -1)).reshape(batch, seq_len, -1)
        
        retrieved = self.to_out(retrieved)
        
        next_state = RNNMemState(
            seq_index=state.seq_index + seq_len,
            S=S_end,
            Z=Z_end,
            omega_buffer=new_omega_buffer
        )
        
        return retrieved, next_state

    def _forward_cuda(
        self,
        x: Tensor,
        state: RNNMemState,
        *,
        S_ref: Tensor,
    ) -> tuple[Tensor, RNNMemState]:
        """
        CUDA-accelerated forward pass for omega_window=16.
        
        This is ~50-100x faster than PyTorch and uses ~100x less memory.
        Automatically handles preprocessing (norm, conv, projections).
        """
        batch, seq_len, _ = x.shape
        d = self.dim_head
        BH = batch * self.heads
        layer_idx = int(getattr(self, "layer_idx", -1))
        
        # Pre-norm and project
        x_normed = self.activation(self.pre_norm(x))
        
        q_in, k_in, v_in = x_normed, x_normed, x_normed
        
        if exists(self.q_conv):
            q_in = self.q_conv(q_in.transpose(1, 2)).transpose(1, 2)
        if exists(self.k_conv):
            k_in = self.k_conv(k_in.transpose(1, 2)).transpose(1, 2)
        if exists(self.v_conv):
            v_in = self.v_conv(v_in.transpose(1, 2)).transpose(1, 2)
        
        q = split_heads(self.to_q(q_in), self.heads)
        k = split_heads(self.to_k(k_in), self.num_key_value_heads)
        v = split_heads(self.to_v(v_in), self.num_key_value_heads)
        
        # Truncate to original seq_len in case conv changed length
        q = q[:, :, :seq_len]
        k = k[:, :, :seq_len]
        v = v[:, :, :seq_len]
        
        # Apply ROPE if enabled
        if self.use_rope and self.rotary_emb is not None:
            # Keep API consistent with the PyTorch path: RotaryEmbedding applies RoPE directly.
            q, k = self.rotary_emb(q, k)
        
        # Normalization
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # GQA: Expand K,V to match Q's number of heads
        if self.num_key_value_groups > 1:
            k = repeat(k, 'b h n d -> b (h g) n d', g=self.num_key_value_groups)
            v = repeat(v, 'b h n d -> b (h g) n d', g=self.num_key_value_groups)
        
        # Apply polynomial features
        k_flat = rearrange(k, 'b h n d -> (b h n) d')
        q_flat = rearrange(q, 'b h n d -> (b h n) d')
        
        phi_k = rearrange(self.phi(k_flat), '(b h n) d -> (b h) n d', b=batch, h=self.heads, n=seq_len)
        phi_q = rearrange(self.phi(q_flat), '(b h n) d -> (b h) n d', b=batch, h=self.heads, n=seq_len)
        v_bh = rearrange(v, 'b h n d -> (b h) n d')
        
        # Learned hyperparameters
        lr = rearrange(self.to_lr(x_normed), 'b n h -> (b h) n')
        decay = rearrange(self.to_decay(x_normed), 'b n h -> (b h) n')
        beta = rearrange(self.to_momentum(x_normed), 'b n h -> (b h) n') if self.use_momentum else torch.ones_like(lr)
        gate = rearrange(self.to_omega_gate(x_normed), 'b n h -> (b h) n') if self.use_omega_gate else torch.ones_like(lr)
        
        # Get initial states
        S0 = state.S if exists(state.S) else phi_k.new_zeros((BH, d, d))
        Z0 = state.Z if exists(state.Z) else phi_k.new_zeros((BH, d, d))
        
        # Ensure bfloat16 for CUDA kernel
        phi_k = phi_k.to(torch.bfloat16)
        phi_q = phi_q.to(torch.bfloat16)
        v_bh = v_bh.to(torch.bfloat16)
        S_ref = S_ref.to(torch.bfloat16)
        lr = lr.to(torch.bfloat16)
        decay = decay.to(torch.bfloat16)
        beta = beta.to(torch.bfloat16)
        gate = gate.to(torch.bfloat16)
        S0 = S0.to(torch.bfloat16)
        Z0 = Z0.to(torch.bfloat16)

        # Debug-only: optional dumps of omega inputs/outputs.
        # This does NOT change the kernel inputs/outputs; it only records tensors when enabled.
        should_dump, ctx, batch_idx, rank = _nan_debug_should_dump_omega(layer_idx)
        if should_dump:
            out_dir = _nan_debug_omega_dir()
            step_tag = f"step{batch_idx}" if batch_idx is not None else "stepNA"
            layer_tag = f"layer{layer_idx}"
            rank_tag = f"rank{rank}"
            if _nan_debug_omega_inputs_enabled():
                inputs_path = os.path.join(out_dir, f"omega_inputs_{rank_tag}_{step_tag}_{layer_tag}.pt")
                payload = {
                    "phi_k": phi_k.detach().cpu(),
                    "phi_q": phi_q.detach().cpu(),
                    "v": v_bh.detach().cpu(),
                    "S_ref": S_ref.detach().cpu(),
                    "lr": lr.detach().cpu(),
                    "decay": decay.detach().cpu(),
                    "beta": beta.detach().cpu(),
                    "gate": gate.detach().cpu(),
                    "S0": S0.detach().cpu(),
                    "Z0": Z0.detach().cpu(),
                }
                if _nan_debug_omega_dump_x_enabled():
                    payload["x_normed"] = x_normed.detach().cpu()
                    payload["x"] = x.detach().cpu()
                _nan_debug_omega_dump_tensor_dict(
                    inputs_path,
                    payload,
                )
            meta_path = os.path.join(out_dir, f"omega_meta_{rank_tag}_{step_tag}_{layer_tag}.json")
            _nan_debug_omega_dump_meta(
                meta_path,
                {
                    "batch_idx": batch_idx,
                    "global_step": ctx.get("global_step", None),
                    "rank": rank,
                    "layer_idx": layer_idx,
                    "shape": {"BH": int(BH), "T": int(seq_len), "d": int(d)},
                    "dtype": str(phi_k.dtype),
                    "device": str(phi_k.device),
                    "use_cuda": True,
                },
            )
            try:
                print(f"[NaN-DEBUG] omega_inputs_dump_dir={out_dir} {rank_tag} {step_tag} {layer_tag}")
            except Exception:
                pass

        # Call CUDA kernel (attach layer idx for debug context only).
        # This preserves the original kernel call behavior.
        prev_layer = os.environ.get("NAN_DEBUG_LAYER_IDX")
        try:
            layer_env = getattr(self, "layer_idx", None)
            if layer_env is not None:
                os.environ["NAN_DEBUG_LAYER_IDX"] = str(int(layer_env))
            y_bh, S_T, Z_T = self.AtlasOmegaFunction.apply(
                phi_k, phi_q, v_bh, S_ref, lr, decay, beta, gate, S0, Z0
            )
        finally:
            if prev_layer is None:
                try:
                    del os.environ["NAN_DEBUG_LAYER_IDX"]
                except Exception:
                    pass
            else:
                os.environ["NAN_DEBUG_LAYER_IDX"] = prev_layer

        if should_dump and _nan_debug_omega_outputs_enabled():
            out_dir = _nan_debug_omega_dir()
            step_tag = f"step{batch_idx}" if batch_idx is not None else "stepNA"
            layer_tag = f"layer{layer_idx}"
            rank_tag = f"rank{rank}"
            outputs_path = os.path.join(out_dir, f"omega_outputs_{rank_tag}_{step_tag}_{layer_tag}.pt")
            _nan_debug_omega_dump_tensor_dict(
                outputs_path,
                {
                    "y": y_bh.detach().cpu(),
                    "S_T": S_T.detach().cpu(),
                    "Z_T": Z_T.detach().cpu(),
                },
            )
        
        # Reshape output
        y_bh = y_bh.to(x.dtype)  # Convert back to original dtype
        retrieved = rearrange(y_bh, '(b h) t d -> b h t d', b=batch, h=self.heads)
        retrieved = merge_heads(retrieved)
        
        # Apply GroupNorm if enabled (before output projection)
        if self.use_groupnorm and self.groupnorm is not None:
            retrieved = self.groupnorm(retrieved.reshape(batch * seq_len, -1)).reshape(batch, seq_len, -1)
        
        retrieved = self.to_out(retrieved)
        
        # Update state (no omega_buffer needed for CUDA, it's managed internally)
        next_state = RNNMemState(
            seq_index=state.seq_index + seq_len,
            S=S_T.to(x.dtype),
            Z=Z_T.to(x.dtype) if self.use_momentum else None,
            omega_buffer=None  # CUDA manages ring buffer internally
        )
        
        return retrieved, next_state


# ============================================================================
# Convenience factory
# ============================================================================

class RNNMemory(Module):
    """
    Factory wrapper that selects the appropriate RNN memory cell.
    
    NOTE: omega_window=1 should never be used in practice. Always use omega_window >= 2.
    RNNMemoryCell is deprecated and only kept for backwards compatibility.
    
    New features:
    - GQA (Grouped Query Attention)
    - ROPE (Rotary Position Embedding)
    - GroupNorm (per-head normalization)
    """
    
    def __init__(
        self,
        dim: int,
        dim_head: int | None = None,
        heads: int = 1,
        num_key_value_heads: int | None = None,  # GQA
        use_momentum: bool = True,
        poly_degree: int = 2,  # Default: 2
        poly_mode: str = 'elementwise',  # Default: elementwise
        omega_window: int = 4,  # Changed default from 1 to 4
        use_omega_gate: bool = True,  # Changed default from False to True
        qk_norm: bool = False,  # Default: False
        qkv_conv_kernel: int | None = None,  # Default: None
        use_rope: bool = False,  # ROPE option
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 2048,
        use_groupnorm: bool = False,  # GroupNorm option
        use_accelerated_scan: bool = False,
        scan_chunk_len: int | None = None,
        use_cuda: bool = False,  # CUDA backend
    ):
        super().__init__()
        
        dim_head = default(dim_head, dim)
        
        # Always use OmegaRNNMemoryCell (omega_window=1 is deprecated)
        # RNNMemoryCell is kept only for backwards compatibility but should not be used
        self.cell = OmegaRNNMemoryCell(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            num_key_value_heads=num_key_value_heads,  # GQA
            omega_window=omega_window,
            use_omega_gate=use_omega_gate,
            use_momentum=use_momentum,
            poly_degree=poly_degree,
            poly_mode=poly_mode,
            qk_norm=qk_norm,
            qkv_conv_kernel=qkv_conv_kernel,
            use_rope=use_rope,  # ROPE
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
            use_groupnorm=use_groupnorm,  # GroupNorm
            use_accelerated_scan=use_accelerated_scan,
            scan_chunk_len=scan_chunk_len,
            use_cuda=use_cuda,  # CUDA backend
        )
    
    @property
    def heads(self):
        return self.cell.heads
    
    @property
    def dim_head(self):
        return self.cell.dim_head
    
    @property
    def use_momentum(self):
        return self.cell.use_momentum
    
    @property
    def use_cuda(self):
        return self.cell.use_cuda
    
    def init_state(self, batch: int, device=None, dtype=None) -> RNNMemState:
        return self.cell.init_state(batch, device, dtype)
    
    def forward(
        self,
        seq: Tensor,
        state: RNNMemState | None = None,
    ) -> tuple[Tensor, RNNMemState]:
        return self.cell(seq, state)


# Alias for OmegaNet-RNN
OmegaRNNMemory = partial(RNNMemory, omega_window=4, use_omega_gate=True)
