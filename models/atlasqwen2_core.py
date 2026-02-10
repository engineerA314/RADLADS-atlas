"""
Atlas-Qwen2 Core: Single canonical implementation of Atlas-LMM blocks.

This module is imported by both:
- Training entrypoint (models/atlasqwen2.py)
- HuggingFace wrapper (atlasqwen2/modeling_atlasqwen2.py)

Architecture (LMM - Long-term Memory Model):
- No self-attention
- Each layer: RNNMemory + MLP with residual connections
- Compatible with Qwen2-style embedding/norm/head

state_dict key naming (locked):
- model.embed_tokens.*
- model.layers.{i}.input_layernorm.*
- model.layers.{i}.memory.*
- model.layers.{i}.post_memory_layernorm.*  
- model.layers.{i}.mlp.*
- model.norm.*
- lm_head.*
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, NamedTuple
import os
import json

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch import Tensor

from models.atlas_memory import RNNMemory, RNNMemState, state_detach


# ============================================================================
# NaN debug helpers (backward hook capture)
# ============================================================================

_NAN_DEBUG_BWD_DUMPED: set[tuple[int, int, int, str]] = set()


def _nan_debug_bwd_steps() -> set[int]:
    raw = str(os.environ.get("NAN_DEBUG_BWD_STEPS", "") or "").strip()
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


def _nan_debug_bwd_layers() -> set[int] | None:
    raw = str(os.environ.get("NAN_DEBUG_BWD_LAYERS", "") or "").strip()
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


def _nan_debug_bwd_ranks() -> set[int] | None:
    raw = str(os.environ.get("NAN_DEBUG_BWD_RANKS", "") or "").strip()
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


def _nan_debug_bwd_tags() -> set[str] | None:
    raw = str(os.environ.get("NAN_DEBUG_BWD_TAGS", "") or "").strip()
    if raw == "" or raw.lower() == "all":
        return None
    return {p.strip() for p in raw.split(",") if p.strip()}


def _nan_debug_bwd_dir() -> str:
    return str(os.environ.get("NAN_DEBUG_BWD_DIR", "/tmp/radlads_nan_debug_bwd"))


def _nan_debug_bwd_dump_tensors_enabled() -> bool:
    raw = str(os.environ.get("NAN_DEBUG_BWD_DUMP_TENSORS", "0") or "").strip().lower()
    return raw in ("1", "true", "yes")


def _nan_debug_bwd_max_elems() -> int:
    raw = str(os.environ.get("NAN_DEBUG_BWD_MAX_ELEMS", "2000000") or "").strip()
    try:
        return max(1, int(raw))
    except Exception:
        return 2000000


def _nan_debug_bwd_once() -> bool:
    raw = str(os.environ.get("NAN_DEBUG_BWD_ONCE", "1") or "").strip().lower()
    return raw in ("1", "true", "yes")


def _nan_debug_bwd_context() -> dict:
    try:
        from atlasattn import get_nan_debug_context
        return get_nan_debug_context()
    except Exception:
        return {}


def _nan_debug_bwd_should_attach(layer_idx: int, tag: str) -> tuple[bool, int | None, int, tuple[int, int, int, str]]:
    steps = _nan_debug_bwd_steps()
    if not steps:
        return False, None, 0, (0, 0, 0, tag)
    ctx = _nan_debug_bwd_context()
    batch_idx = ctx.get("batch_idx", None)
    if batch_idx is None or int(batch_idx) not in steps:
        return False, None, 0, (0, 0, 0, tag)
    try:
        import torch.distributed as dist
        is_dist = dist.is_available() and dist.is_initialized()
        rank = int(dist.get_rank()) if is_dist else 0
    except Exception:
        rank = 0
    ranks = _nan_debug_bwd_ranks()
    if ranks is not None and rank not in ranks:
        return False, int(batch_idx), rank, (int(batch_idx), int(layer_idx), rank, tag)
    layers = _nan_debug_bwd_layers()
    if layers is not None and layer_idx not in layers:
        return False, int(batch_idx), rank, (int(batch_idx), int(layer_idx), rank, tag)
    tags = _nan_debug_bwd_tags()
    if tags is not None and tag not in tags:
        return False, int(batch_idx), rank, (int(batch_idx), int(layer_idx), rank, tag)
    key = (int(batch_idx), int(layer_idx), int(rank), tag)
    if _nan_debug_bwd_once() and key in _NAN_DEBUG_BWD_DUMPED:
        return False, int(batch_idx), rank, key
    return True, int(batch_idx), rank, key


def _nan_debug_bwd_dump(tag: str, layer_idx: int, grad: Tensor, batch_idx: int, rank: int, key: tuple[int, int, int, str]) -> None:
    if _nan_debug_bwd_once():
        _NAN_DEBUG_BWD_DUMPED.add(key)
    try:
        flat = grad.detach().reshape(-1)
        stats = {
            "shape": list(grad.shape),
            "dtype": str(grad.dtype),
            "device": str(grad.device),
            "numel": int(flat.numel()),
            "nan": int(torch.isnan(flat).sum().item()),
            "inf": int(torch.isinf(flat).sum().item()),
        }
        finite = torch.isfinite(flat)
        if finite.any():
            f = flat[finite].float()
            stats.update(
                absmax=float(f.abs().max().item()),
                mean=float(f.mean().item()),
                std=float(f.std(unbiased=False).item()) if f.numel() > 1 else 0.0,
            )
    except Exception:
        stats = {"error": "stats_failed"}

    payload = {
        "batch_idx": int(batch_idx),
        "layer_idx": int(layer_idx),
        "rank": int(rank),
        "tag": tag,
        "grad": stats,
    }
    out_dir = _nan_debug_bwd_dir()
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        out_dir = "/tmp"
    out_path = os.path.join(out_dir, f"bwd_grad_rank{rank}_step{int(batch_idx)}_layer{int(layer_idx)}_{tag}.json")
    try:
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass

    if _nan_debug_bwd_dump_tensors_enabled() and grad.numel() <= _nan_debug_bwd_max_elems():
        try:
            tensor_path = os.path.join(out_dir, f"bwd_grad_rank{rank}_step{int(batch_idx)}_layer{int(layer_idx)}_{tag}.pt")
            torch.save(grad.detach().cpu(), tensor_path)
        except Exception:
            pass


def _nan_debug_bwd_hook(tensor: Tensor, *, tag: str, layer_idx: int) -> None:
    if tensor is None or (not torch.is_tensor(tensor)):
        return
    if not tensor.is_floating_point():
        return
    ok, batch_idx, rank, key = _nan_debug_bwd_should_attach(layer_idx, tag)
    if not ok:
        return

    def _hook(grad: Tensor) -> None:
        if grad is None:
            return
        _nan_debug_bwd_dump(tag, layer_idx, grad, int(batch_idx), int(rank), key)

    try:
        tensor.register_hook(_hook)
    except Exception:
        pass


# ============================================================================
# Atlas State Types
# ============================================================================

class AtlasLayerState(NamedTuple):
    """Per-layer state for Atlas model."""
    memory_state: RNNMemState


class AtlasModelState(NamedTuple):
    """Full model state across all layers."""
    layer_states: List[AtlasLayerState]
    seen_tokens: int


def init_model_state(
    n_layers: int, 
    batch: int, 
    heads: int,
    dim_head: int, 
    use_momentum: bool,
    omega_window: int = 4,
    device=None, 
    dtype=None
) -> AtlasModelState:
    """Initialize empty model state for all layers."""
    layer_states = []
    for _ in range(n_layers):
        S = torch.zeros(batch * heads, dim_head, dim_head, device=device, dtype=dtype)
        Z = torch.zeros_like(S) if use_momentum else None
        
        # Initialize omega_buffer if omega_window > 1
        if omega_window > 1:
            omega_buffer = torch.zeros(
                batch * heads, omega_window - 1, dim_head, dim_head, 2,
                device=device, dtype=dtype
            )
        else:
            omega_buffer = None
        
        mem_state = RNNMemState(seq_index=0, S=S, Z=Z, omega_buffer=omega_buffer)
        layer_states.append(AtlasLayerState(memory_state=mem_state))
    return AtlasModelState(layer_states=layer_states, seen_tokens=0)


def detach_model_state(state: AtlasModelState) -> AtlasModelState:
    """Detach all tensors in model state from computation graph."""
    new_layer_states = []
    for ls in state.layer_states:
        new_mem_state = state_detach(ls.memory_state)
        new_layer_states.append(AtlasLayerState(memory_state=new_mem_state))
    return AtlasModelState(layer_states=new_layer_states, seen_tokens=state.seen_tokens)


# ============================================================================
# Model Configuration (dataclass for easy serialization)
# ============================================================================

@dataclass
class AtlasConfig:
    """Configuration for Atlas-LMM/MAL model."""
    # Model architecture
    vocab_size: int = 151936
    n_embd: int = 896
    n_layer: int = 24
    dim_ffn: int = 4864
    rms_norm_eps: float = 1e-6
    
    # Memory configuration
    memory_heads: int = 14
    memory_dim_head: int = 64
    num_key_value_heads: int | None = None  # GQA: None means same as memory_heads
    use_momentum: bool = True
    omega_window: int = 4  # Omega window size (must be >= 2)
    use_omega_gate: bool = True  # Enable learnable omega gate
    use_cuda: bool = False  # Enable CUDA backend for 50-100x speedup (requires omega_window=16)
    poly_degree: int = 1
    poly_mode: str = 'off'
    qk_norm: bool = True
    qkv_conv_kernel: Optional[int] = None
    
    # ROPE configuration
    use_rope: bool = False  # Enable Rotary Position Embedding
    rope_theta: float = 10000.0  # ROPE base frequency
    max_position_embeddings: int = 2048  # ROPE max positions
    
    # GroupNorm configuration
    use_groupnorm: bool = False  # Enable per-head GroupNorm

    # AssocScan backend
    # If True, uses the accelerated (Triton) scan path in `assoc_scan` when available.
    use_accelerated_scan: bool = True

    # Chunked scan (memory)
    # If set, compute scan in chunks of this length to reduce peak memory.
    # This does NOT require a custom CUDA kernel, but may be slower due to Python-level looping.
    memory_scan_chunk_len: Optional[int] = None
    
    # Architecture variant: 'lmm' (memory only) or 'mal' (memory + attention)
    atlas_variant: str = 'lmm'
    
    # For MAL: sliding window attention config
    sliding_window: int = 512
    
    # Other
    ctx_len: int = 2048
    vocab_padding_idx: Optional[int] = None
    tie_word_embeddings: bool = True

    # Training-time knobs (kept here so the core can apply them without depending on TrainerCLI_Config)
    # grad_cp=1 enables activation checkpointing for Atlas blocks (reduces activation memory, recomputes in backward).
    grad_cp: int = 0
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'vocab_size': self.vocab_size,
            'n_embd': self.n_embd,
            'n_layer': self.n_layer,
            'dim_ffn': self.dim_ffn,
            'rms_norm_eps': self.rms_norm_eps,
            'memory_heads': self.memory_heads,
            'memory_dim_head': self.memory_dim_head,
            'num_key_value_heads': self.num_key_value_heads,
            'use_momentum': self.use_momentum,
            'omega_window': self.omega_window,
            'use_omega_gate': self.use_omega_gate,
            'poly_degree': self.poly_degree,
            'poly_mode': self.poly_mode,
            'qk_norm': self.qk_norm,
            'qkv_conv_kernel': self.qkv_conv_kernel,
            'use_rope': self.use_rope,
            'rope_theta': self.rope_theta,
            'max_position_embeddings': self.max_position_embeddings,
            'use_groupnorm': self.use_groupnorm,
            'use_accelerated_scan': self.use_accelerated_scan,
            'memory_scan_chunk_len': self.memory_scan_chunk_len,
            'ctx_len': self.ctx_len,
            'vocab_padding_idx': self.vocab_padding_idx,
            'tie_word_embeddings': self.tie_word_embeddings,
            'grad_cp': self.grad_cp,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'AtlasConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ============================================================================
# RMS Norm (Qwen2-compatible)
# ============================================================================

class AtlasRMSNorm(nn.Module):
    """RMS Normalization, compatible with Qwen2RMSNorm."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        layer_idx = int(getattr(self, "layer_idx", -1))
        tag_prefix = getattr(self, "debug_tag", None)
        if tag_prefix:
            _nan_debug_bwd_hook(hidden_states, tag=f"{tag_prefix}_in", layer_idx=layer_idx)
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        out = self.weight * hidden_states.to(input_dtype)
        if tag_prefix:
            _nan_debug_bwd_hook(out, tag=f"{tag_prefix}_out", layer_idx=layer_idx)
        return out


# ============================================================================
# MLP (Qwen2-compatible)
# ============================================================================

class AtlasMLP(nn.Module):
    """SiLU-gated MLP, compatible with Qwen2MLP."""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()
    
    def forward(self, x: Tensor) -> Tensor:
        layer_idx = int(getattr(self, "layer_idx", -1))
        _nan_debug_bwd_hook(x, tag="mlp_in", layer_idx=layer_idx)
        gate = self.gate_proj(x)
        _nan_debug_bwd_hook(gate, tag="mlp_gate", layer_idx=layer_idx)
        up = self.up_proj(x)
        _nan_debug_bwd_hook(up, tag="mlp_up", layer_idx=layer_idx)
        act = self.act_fn(gate)
        _nan_debug_bwd_hook(act, tag="mlp_act", layer_idx=layer_idx)
        out = self.down_proj(act * up)
        _nan_debug_bwd_hook(out, tag="mlp_out", layer_idx=layer_idx)
        return out


# ============================================================================
# Sliding Window Attention (for MAL/MAG variants)
# ============================================================================

class SlidingWindowAttention(nn.Module):
    """
    Sliding window self-attention for MAL/MAG blocks.
    
    Uses PyTorch's scaled_dot_product_attention with causal masking.
    For simplicity, we use full causal attention during training and 
    sliding window semantics are enforced via position-based masking.
    """
    
    def __init__(
        self,
        dim: int,
        window_size: int = 512,
        dim_head: int = 64,
        heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        dim_inner = dim_head * heads
        
        self.q_proj = nn.Linear(dim, dim_inner, bias=True)
        self.k_proj = nn.Linear(dim, dim_inner, bias=True)
        self.v_proj = nn.Linear(dim, dim_inner, bias=True)
        self.o_proj = nn.Linear(dim_inner, dim, bias=False)
        
        self.dropout = dropout
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for sliding window attention.
        
        Args:
            x: [batch, seq_len, dim]
            
        Returns:
            output: [batch, seq_len, dim]
        """
        B, T, _ = x.shape
        
        q = self.q_proj(x).view(B, T, self.heads, self.dim_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.heads, self.dim_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.heads, self.dim_head).transpose(1, 2)
        
        # Use SDPA with causal=True for simplicity
        # For true sliding window, would need custom mask
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )
        
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out)


# ============================================================================
# Atlas LMM Block (Memory + MLP, no attention)
# ============================================================================

class AtlasLMMBlock(nn.Module):
    """
    Atlas LMM decoder block: Memory + MLP with residual connections.
    
    Structure:
        x -> input_layernorm -> memory -> + -> post_memory_layernorm -> mlp -> + -> out
             |                           |    |                                |
             +---------------------------+    +--------------------------------+
    """
    
    def __init__(self, config: AtlasConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Layer norms
        self.input_layernorm = AtlasRMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.post_memory_layernorm = AtlasRMSNorm(config.n_embd, eps=config.rms_norm_eps)
        try:
            self.input_layernorm.layer_idx = int(layer_idx)
            self.input_layernorm.debug_tag = "input_ln"
            self.post_memory_layernorm.layer_idx = int(layer_idx)
            self.post_memory_layernorm.debug_tag = "post_mem_ln"
        except Exception:
            pass
        
        # Memory module (replaces attention)
        self.memory = RNNMemory(
            dim=config.n_embd,
            dim_head=config.memory_dim_head,
            heads=config.memory_heads,
            num_key_value_heads=config.num_key_value_heads,
            use_momentum=config.use_momentum,
            omega_window=config.omega_window,
            use_omega_gate=config.use_omega_gate,
            use_cuda=config.use_cuda,
            poly_degree=config.poly_degree,
            poly_mode=config.poly_mode,
            qk_norm=config.qk_norm,
            qkv_conv_kernel=config.qkv_conv_kernel,
            use_rope=config.use_rope,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            use_groupnorm=config.use_groupnorm,
            use_accelerated_scan=getattr(config, "use_accelerated_scan", True),
            scan_chunk_len=getattr(config, "memory_scan_chunk_len", None),
        )

        # Expose layer index for omega debug hooks.
        try:
            self.memory.layer_idx = int(layer_idx)
            if hasattr(self.memory, "cell"):
                self.memory.cell.layer_idx = int(layer_idx)
        except Exception:
            pass
        
        # MLP
        self.mlp = AtlasMLP(config.n_embd, config.dim_ffn)
        try:
            self.mlp.layer_idx = int(layer_idx)
        except Exception:
            pass
        try:
            self.mlp.layer_idx = int(layer_idx)
        except Exception:
            pass
    
    def forward(
        self,
        hidden_states: Tensor,
        layer_state: Optional[AtlasLayerState] = None,
    ) -> Tuple[Tensor, AtlasLayerState]:
        """
        Forward pass for one block.
        
        Args:
            hidden_states: [batch, seq_len, n_embd]
            layer_state: Optional previous state for this layer
            
        Returns:
            output: [batch, seq_len, n_embd]
            new_layer_state: Updated state for this layer
        """
        # Memory path with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        _nan_debug_bwd_hook(hidden_states, tag="post_input_ln", layer_idx=self.layer_idx)
        
        mem_state = layer_state.memory_state if layer_state is not None else None
        hidden_states, new_mem_state = self.memory(hidden_states, mem_state)
        _nan_debug_bwd_hook(hidden_states, tag="mem_out", layer_idx=self.layer_idx)
        hidden_states = residual + hidden_states
        _nan_debug_bwd_hook(hidden_states, tag="post_mem_resid", layer_idx=self.layer_idx)
        
        # MLP path with residual
        residual = hidden_states
        hidden_states = self.post_memory_layernorm(hidden_states)
        _nan_debug_bwd_hook(hidden_states, tag="post_mem_ln", layer_idx=self.layer_idx)
        hidden_states = self.mlp(hidden_states)
        _nan_debug_bwd_hook(hidden_states, tag="mlp_out", layer_idx=self.layer_idx)
        hidden_states = residual + hidden_states
        _nan_debug_bwd_hook(hidden_states, tag="post_mlp_resid", layer_idx=self.layer_idx)
        
        new_layer_state = AtlasLayerState(memory_state=new_mem_state)
        return hidden_states, new_layer_state


# ============================================================================
# Atlas MAL Block (Memory As Layer: Memory + Attention + MLP)
# ============================================================================

class AtlasMALBlock(nn.Module):
    """
    Atlas MAL (Memory-As-Layer) decoder block: Memory + Attention + MLP.
    
    Structure:
        x -> input_layernorm -> memory -> + -> post_memory_layernorm -> attention -> + -> post_attn_layernorm -> mlp -> + -> out
             |                            |    |                                     |    |                             |
             +----------------------------+    +-------------------------------------+    +-----------------------------+
    
    This keeps attention (sliding window) while adding memory as an additional sublayer.
    """
    
    def __init__(self, config: AtlasConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Layer norms
        self.input_layernorm = AtlasRMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.post_memory_layernorm = AtlasRMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.post_attn_layernorm = AtlasRMSNorm(config.n_embd, eps=config.rms_norm_eps)
        try:
            self.input_layernorm.layer_idx = int(layer_idx)
            self.input_layernorm.debug_tag = "input_ln"
            self.post_memory_layernorm.layer_idx = int(layer_idx)
            self.post_memory_layernorm.debug_tag = "post_mem_ln"
            self.post_attn_layernorm.layer_idx = int(layer_idx)
            self.post_attn_layernorm.debug_tag = "post_attn_ln"
        except Exception:
            pass
        
        # Memory module
        self.memory = RNNMemory(
            dim=config.n_embd,
            dim_head=config.memory_dim_head,
            heads=config.memory_heads,
            num_key_value_heads=config.num_key_value_heads,
            use_momentum=config.use_momentum,
            omega_window=config.omega_window,
            use_omega_gate=config.use_omega_gate,
            use_cuda=config.use_cuda,
            poly_degree=config.poly_degree,
            poly_mode=config.poly_mode,
            qk_norm=config.qk_norm,
            qkv_conv_kernel=config.qkv_conv_kernel,
            use_rope=config.use_rope,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            use_groupnorm=config.use_groupnorm,
            use_accelerated_scan=getattr(config, "use_accelerated_scan", True),
            scan_chunk_len=getattr(config, "memory_scan_chunk_len", None),
        )

        # Expose layer index for omega debug hooks.
        try:
            self.memory.layer_idx = int(layer_idx)
            if hasattr(self.memory, "cell"):
                self.memory.cell.layer_idx = int(layer_idx)
        except Exception:
            pass
        
        # Sliding window attention
        sliding_window = getattr(config, 'sliding_window', 512)
        self.attention = SlidingWindowAttention(
            dim=config.n_embd,
            window_size=sliding_window,
            dim_head=config.memory_dim_head,
            heads=config.memory_heads,
        )
        
        # MLP
        self.mlp = AtlasMLP(config.n_embd, config.dim_ffn)
    
    def forward(
        self,
        hidden_states: Tensor,
        layer_state: Optional[AtlasLayerState] = None,
    ) -> Tuple[Tensor, AtlasLayerState]:
        """
        Forward pass for MAL block.
        
        Args:
            hidden_states: [batch, seq_len, n_embd]
            layer_state: Optional previous state for this layer
            
        Returns:
            output: [batch, seq_len, n_embd]
            new_layer_state: Updated state for this layer
        """
        # Memory path with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        _nan_debug_bwd_hook(hidden_states, tag="post_input_ln", layer_idx=self.layer_idx)
        
        mem_state = layer_state.memory_state if layer_state is not None else None
        hidden_states, new_mem_state = self.memory(hidden_states, mem_state)
        _nan_debug_bwd_hook(hidden_states, tag="mem_out", layer_idx=self.layer_idx)
        hidden_states = residual + hidden_states
        _nan_debug_bwd_hook(hidden_states, tag="post_mem_resid", layer_idx=self.layer_idx)
        
        # Attention path with residual
        residual = hidden_states
        hidden_states = self.post_memory_layernorm(hidden_states)
        _nan_debug_bwd_hook(hidden_states, tag="post_mem_ln", layer_idx=self.layer_idx)
        hidden_states = self.attention(hidden_states)
        _nan_debug_bwd_hook(hidden_states, tag="attn_out", layer_idx=self.layer_idx)
        hidden_states = residual + hidden_states
        _nan_debug_bwd_hook(hidden_states, tag="post_attn_resid", layer_idx=self.layer_idx)
        
        # MLP path with residual
        residual = hidden_states
        hidden_states = self.post_attn_layernorm(hidden_states)
        _nan_debug_bwd_hook(hidden_states, tag="post_attn_ln", layer_idx=self.layer_idx)
        hidden_states = self.mlp(hidden_states)
        _nan_debug_bwd_hook(hidden_states, tag="mlp_out", layer_idx=self.layer_idx)
        hidden_states = residual + hidden_states
        _nan_debug_bwd_hook(hidden_states, tag="post_mlp_resid", layer_idx=self.layer_idx)
        
        new_layer_state = AtlasLayerState(memory_state=new_mem_state)
        return hidden_states, new_layer_state


# ============================================================================
# Block Factory
# ============================================================================

def create_atlas_block(config: AtlasConfig, layer_idx: int) -> nn.Module:
    """Factory function to create the appropriate block type based on config."""
    variant = getattr(config, 'atlas_variant', 'lmm').lower()
    if variant == 'mal':
        return AtlasMALBlock(config, layer_idx)
    else:  # default to 'lmm'
        return AtlasLMMBlock(config, layer_idx)


# ============================================================================
# Atlas Model (Decoder)
# ============================================================================

class AtlasQwen2Core(nn.Module):
    """
    Core Atlas decoder model (no lm_head).
    
    This is the canonical implementation shared by training and HF inference.
    Supports both LMM (memory only) and MAL (memory + attention) variants.
    """
    
    def __init__(self, config: AtlasConfig):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(
            config.vocab_size, 
            config.n_embd, 
            padding_idx=config.vocab_padding_idx
        )
        
        self.layers = nn.ModuleList([
            create_atlas_block(config, layer_idx=i) 
            for i in range(config.n_layer)
        ])
        
        self.norm = AtlasRMSNorm(config.n_embd, eps=config.rms_norm_eps)
    
    def forward(
        self,
        input_ids: Tensor,
        model_state: Optional[AtlasModelState] = None,
        output_hidden_states: bool = False,
    ) -> Tuple[Tensor, AtlasModelState, Optional[Tuple[Tensor, ...]]]:
        """
        Forward pass through the decoder.
        
        Args:
            input_ids: [batch, seq_len] token IDs
            model_state: Optional previous state
            output_hidden_states: Whether to return all hidden states
            
        Returns:
            hidden_states: [batch, seq_len, n_embd] final hidden states
            new_model_state: Updated model state
            all_hidden_states: Tuple of hidden states per layer (if requested)
        """
        batch, seq_len = input_ids.shape
        
        # Initialize state if not provided
        if model_state is None:
            model_state = init_model_state(
                n_layers=self.config.n_layer,
                batch=batch,
                heads=self.config.memory_heads,
                dim_head=self.config.memory_dim_head,
                use_momentum=self.config.use_momentum,
                omega_window=self.config.omega_window,
                device=input_ids.device,
                dtype=self.embed_tokens.weight.dtype,
            )
        
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        
        # Track hidden states if requested
        all_hidden_states = () if output_hidden_states else None
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Process through layers
        new_layer_states = []
        seen_tokens = model_state.seen_tokens
        
        for layer_idx, layer in enumerate(self.layers):
            layer_state = model_state.layer_states[layer_idx]
            # Activation checkpointing (memory saver).
            #
            # IMPORTANT: torch.utils.checkpoint requires all *inputs* to be Tensors, and outputs should be
            # Tensors (or nested structures of Tensors). Our layer_state is a NamedTuple, so we checkpoint
            # a small wrapper that takes only state tensors and reconstructs the state inside.
            if self.training and getattr(self.config, "grad_cp", 0) == 1:
                mem0 = layer_state.memory_state
                seq_index0 = mem0.seq_index
                use_momentum = bool(self.config.use_momentum)
                omega_enabled = int(self.config.omega_window) > 1

                S0 = mem0.S
                Z0 = mem0.Z if mem0.Z is not None else S0.new_zeros(S0.shape)
                omega0 = mem0.omega_buffer if mem0.omega_buffer is not None else S0.new_empty((0,))

                def _ckpt_layer(h: Tensor, S: Tensor, Z: Tensor, omega_buf: Tensor):
                    mem_state = RNNMemState(
                        seq_index=seq_index0,
                        S=S,
                        Z=(Z if use_momentum else None),
                        omega_buffer=(omega_buf if omega_enabled and omega_buf.numel() != 0 else None),
                    )
                    out_h, out_layer_state = layer(h, AtlasLayerState(memory_state=mem_state))
                    out_mem = out_layer_state.memory_state
                    out_S = out_mem.S
                    out_Z = out_mem.Z if out_mem.Z is not None else Z.new_zeros(out_S.shape)
                    out_omega = out_mem.omega_buffer if out_mem.omega_buffer is not None else omega_buf.new_empty((0,))
                    return out_h, out_S, out_Z, out_omega

                out_h, out_S, out_Z, out_omega = torch.utils.checkpoint.checkpoint(
                    _ckpt_layer,
                    hidden_states,
                    S0,
                    Z0,
                    omega0,
                    use_reentrant=False,
                )

                hidden_states = out_h
                new_mem_state = RNNMemState(
                    seq_index=seq_index0 + seq_len,
                    S=out_S,
                    Z=(out_Z if use_momentum else None),
                    omega_buffer=(out_omega if omega_enabled and out_omega.numel() != 0 else None),
                )
                new_layer_state = AtlasLayerState(memory_state=new_mem_state)
            else:
                hidden_states, new_layer_state = layer(hidden_states, layer_state)
            new_layer_states.append(new_layer_state)
            
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        # Build new model state
        new_model_state = AtlasModelState(
            layer_states=new_layer_states,
            seen_tokens=seen_tokens + seq_len,
        )
        
        return hidden_states, new_model_state, all_hidden_states


# ============================================================================
# Atlas LMM Causal LM (with lm_head)
# ============================================================================

class AtlasQwen2ForCausalLM(nn.Module):
    """
    Atlas-LMM Causal Language Model.
    
    Combines the core decoder with an LM head for next-token prediction.
    """
    
    def __init__(self, config: AtlasConfig):
        super().__init__()
        self.config = config
        
        self.model = AtlasQwen2Core(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Optionally tie embeddings
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
    
    def forward(
        self,
        input_ids: Tensor,
        model_state: Optional[AtlasModelState] = None,
        output_hidden_states: bool = False,
    ) -> Tuple[Tensor, AtlasModelState, Optional[Tuple[Tensor, ...]]]:
        """
        Forward pass for causal LM.
        
        Args:
            input_ids: [batch, seq_len] token IDs
            model_state: Optional previous state
            output_hidden_states: Whether to return all hidden states
            
        Returns:
            logits: [batch, seq_len, vocab_size] logit predictions
            new_model_state: Updated model state
            all_hidden_states: Tuple of hidden states (if requested)
        """
        hidden_states, new_model_state, all_hidden_states = self.model(
            input_ids, model_state, output_hidden_states
        )
        
        logits = self.lm_head(hidden_states)
        
        return logits, new_model_state, all_hidden_states
    
    def init_state(self, batch: int, device=None, dtype=None) -> AtlasModelState:
        """Initialize empty model state."""
        device = device or self.model.embed_tokens.weight.device
        dtype = dtype or self.model.embed_tokens.weight.dtype
        return init_model_state(
            n_layers=self.config.n_layer,
            batch=batch,
            heads=self.config.memory_heads,
            dim_head=self.config.memory_dim_head,
            use_momentum=self.config.use_momentum,
            omega_window=self.config.omega_window,
            device=device,
            dtype=dtype,
        )
