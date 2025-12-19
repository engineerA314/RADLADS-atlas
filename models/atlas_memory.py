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

import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Module, Parameter, Linear, Conv1d

from assoc_scan import AssocScan


# ============================================================================
# Helper functions
# ============================================================================

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

LinearNoBias = partial(Linear, bias=False)


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
    Polynomial feature expansion for keys and queries.
    
    Modes:
    - 'off': φ(x) = x
    - 'elementwise': φ(x) = Σ_{i=1}^{g} x^i
    - 'tensor': φ(x) = RandomProj([x, vec(x ⊗ x)])
    """
    
    def __init__(self, dim: int, degree: int = 1, mode: str = 'off'):
        super().__init__()
        self.dim = dim
        self.degree = degree
        self.mode = mode
        
        if mode == 'tensor':
            feat_dim = dim + dim * dim
            self.register_buffer(
                'proj',
                torch.randn(feat_dim, dim) / math.sqrt(feat_dim)
            )
    
    def forward(self, x: Tensor) -> Tensor:
        if self.degree <= 1 or self.mode == 'off':
            return x
        
        if self.mode == 'elementwise':
            out = x.clone()
            power = x
            for _ in range(2, self.degree + 1):
                power = power * x
                out = out + power
            return out
        
        if self.mode == 'tensor':
            d = x.shape[-1]
            outer = torch.einsum('...i,...j->...ij', x, x)
            outer = outer.reshape(*x.shape[:-1], d * d)
            feats = torch.cat([x, outer], dim=-1)
            return feats @ self.proj.to(x.device, x.dtype)
        
        return x


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
        poly_degree: int = 1,
        poly_mode: str = 'off',
        qk_norm: bool = True,
        qkv_conv_kernel: int | None = 4,
        use_accelerated_scan: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.use_momentum = use_momentum
        self.qkv_conv_kernel = qkv_conv_kernel
        
        # Associative scan for parallel computation (use_accelerated=True requires Triton)
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
        # Efficient Scalar Scan Implementation
        # -----------------------------------------------------------------
        # Step 1: Compute per-token outer products (all parallel)
        # G_t = φ_t ⊗ φ_t^T: [BH, T, d, d]
        G = torch.einsum('bti,btj->btij', phi_k, phi_k)
        # B_t = v_t ⊗ φ_t^T: [BH, T, d, d]
        B = torch.einsum('bti,btj->btij', v_bh, phi_k)
        
        # Get initial states
        S0 = state.S  # [BH, d, d]
        Z0 = state.Z if exists(state.Z) else torch.zeros_like(S0)
        
        # Step 2: Compute gradients using FIXED S_0 (one batch matmul!)
        # g_t = S_0 @ G_t - B_t
        # S0: [BH, d, d], G: [BH, T, d, d]
        g = torch.einsum('bde,btef->btdf', S0, G) - B  # [BH, T, d, d]
        
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
        # S_start = [S_0, S_1, ..., S_{T-1}] (shifted by 1)
        S_start = torch.cat([S0.unsqueeze(1), S_all[:, :-1]], dim=1)  # [BH, T, d, d]
        
        # S_start: [BH, T, d, d], phi_q: [BH, T, d]
        retrieved = torch.einsum('btdp,btp->btd', S_start, phi_q)  # [BH, T, d]
        
        # Reshape to [batch, heads, seq, dim_head] and merge
        retrieved = retrieved.view(batch, self.heads, seq_len, d)
        retrieved = merge_heads(retrieved)
        retrieved = self.to_out(retrieved)
        
        # Build next state
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
    """
    
    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 1,
        omega_window: int = 1,
        use_omega_gate: bool = True,
        use_momentum: bool = True,
        poly_degree: int = 1,
        poly_mode: str = 'off',
        qk_norm: bool = True,
        qkv_conv_kernel: int | None = 4,
        use_accelerated_scan: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.omega_window = omega_window
        self.use_omega_gate = use_omega_gate
        self.use_momentum = use_momentum
        self.qkv_conv_kernel = qkv_conv_kernel
        
        # Associative scan for parallel computation (use_accelerated=True requires Triton)
        self.assoc_scan = AssocScan(use_accelerated=use_accelerated_scan)
        
        dim_inner = dim_head * heads
        
        # Projections
        self.activation = nn.Identity()
        self.to_q = LinearNoBias(dim, dim_inner)
        self.to_k = LinearNoBias(dim, dim_inner)
        self.to_v = LinearNoBias(dim, dim_inner)
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
        
        # Norms
        self.pre_norm = nn.RMSNorm(dim)
        self.q_norm = MultiheadRMSNorm(dim_head, heads) if qk_norm else nn.Identity()
        self.k_norm = MultiheadRMSNorm(dim_head, heads) if qk_norm else nn.Identity()
        
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
        
        # Buffer for sliding window: stores (G, B) for last (e-1) tokens
        if self.omega_window > 1:
            omega_buffer = torch.zeros(
                batch * self.heads, self.omega_window - 1, self.dim_head, self.dim_head, 2,
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
        e = self.omega_window
        
        if not exists(state):
            state = self.init_state(batch, x.device, x.dtype)
        
        d = self.dim_head
        BH = batch * self.heads
        
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
        k = split_heads(self.to_k(k_in), self.heads)
        v = split_heads(self.to_v(v_in), self.heads)
        
        # Truncate to original seq_len in case conv changed length
        q = q[:, :, :seq_len]
        k = k[:, :, :seq_len]
        v = v[:, :, :seq_len]
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
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
        # Per-token G_t and B_t (before windowing)
        # -----------------------------------------------------------------
        # Raw per-token outer products
        G_raw = torch.einsum('bti,btj->btij', phi_k, phi_k)  # [BH, T, d, d]
        B_raw = torch.einsum('bti,btj->btij', v_bh, phi_k)    # [BH, T, d, d]
        
        # Apply omega gate if enabled
        if exists(omega_gate):
            gate_e = omega_gate[..., None, None]  # [BH, T, 1, 1]
            G_raw = G_raw * gate_e
            B_raw = B_raw * gate_e
        
        # -----------------------------------------------------------------
        # Omega window: sliding sum of G and B
        # -----------------------------------------------------------------
        if e > 1:
            # Prepend buffer from state
            omega_buffer = state.omega_buffer
            if exists(omega_buffer):
                prev_G = omega_buffer[..., 0]  # [BH, e-1, d, d]
                prev_B = omega_buffer[..., 1]
            else:
                prev_G = G_raw.new_zeros((BH, e - 1, d, d))
                prev_B = B_raw.new_zeros((BH, e - 1, d, d))
            
            G_ext = torch.cat([prev_G, G_raw], dim=1)  # [BH, e-1+T, d, d]
            B_ext = torch.cat([prev_B, B_raw], dim=1)
            
            # Sliding sum over window
            G = _sliding_sum_along_time(G_ext, e)[:, -(seq_len):]  # [BH, T, d, d]
            B = _sliding_sum_along_time(B_ext, e)[:, -(seq_len):]
            
            # Update buffer for next call
            new_omega_buffer = torch.stack([
                G_ext[:, -(e - 1):],
                B_ext[:, -(e - 1):]
            ], dim=-1)  # [BH, e-1, d, d, 2]
        else:
            G = G_raw
            B = B_raw
            new_omega_buffer = None
        
        # -----------------------------------------------------------------
        # Efficient Scalar Scan Implementation (same as Titans-RNN)
        # -----------------------------------------------------------------
        # Get initial states
        S0 = state.S  # [BH, d, d]
        Z0 = state.Z if exists(state.Z) else torch.zeros_like(S0)
        
        # Compute gradients using FIXED S_0 (one batch matmul!)
        # g_t = S_0 @ G_t - B_t
        g = torch.einsum('bde,btef->btdf', S0, G) - B  # [BH, T, d, d]
        
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
        
        # S_start = [S_0, S_1, ..., S_{T-1}] (shifted by 1)
        S_start = torch.cat([S0.unsqueeze(1), S_all[:, :-1]], dim=1)  # [BH, T, d, d]
        
        # -----------------------------------------------------------------
        # Retrieval: y_t = S_{t-1} @ ψ_t
        # -----------------------------------------------------------------
        retrieved = torch.einsum('btdp,btp->btd', S_start, phi_q)
        
        retrieved = retrieved.view(batch, self.heads, seq_len, d)
        retrieved = merge_heads(retrieved)
        retrieved = self.to_out(retrieved)
        
        next_state = RNNMemState(
            seq_index=state.seq_index + seq_len,
            S=S_end,
            Z=Z_end,
            omega_buffer=new_omega_buffer
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
    """
    
    def __init__(
        self,
        dim: int,
        dim_head: int | None = None,
        heads: int = 1,
        use_momentum: bool = True,
        poly_degree: int = 1,
        poly_mode: str = 'off',
        omega_window: int = 4,  # Changed default from 1 to 4
        use_omega_gate: bool = True,  # Changed default from False to True
        qk_norm: bool = True,
        qkv_conv_kernel: int | None = 4,
    ):
        super().__init__()
        
        dim_head = default(dim_head, dim)
        
        # Always use OmegaRNNMemoryCell (omega_window=1 is deprecated)
        # RNNMemoryCell is kept only for backwards compatibility but should not be used
        self.cell = OmegaRNNMemoryCell(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            omega_window=omega_window,
            use_omega_gate=use_omega_gate,
            use_momentum=use_momentum,
            poly_degree=poly_degree,
            poly_mode=poly_mode,
            qk_norm=qk_norm,
            qkv_conv_kernel=qkv_conv_kernel,
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
