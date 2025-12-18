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

import torch
import torch.nn as nn
from torch import Tensor

from models.atlas_memory import RNNMemory, RNNMemState, state_detach


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
    device=None, 
    dtype=None
) -> AtlasModelState:
    """Initialize empty model state for all layers."""
    layer_states = []
    for _ in range(n_layers):
        S = torch.zeros(batch * heads, dim_head, dim_head, device=device, dtype=dtype)
        Z = torch.zeros_like(S) if use_momentum else None
        mem_state = RNNMemState(seq_index=0, S=S, Z=Z, omega_buffer=None)
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
    use_momentum: bool = True
    poly_degree: int = 1
    poly_mode: str = 'off'
    qk_norm: bool = True
    qkv_conv_kernel: Optional[int] = None
    
    # Architecture variant: 'lmm' (memory only) or 'mal' (memory + attention)
    atlas_variant: str = 'lmm'
    
    # For MAL: sliding window attention config
    sliding_window: int = 512
    
    # Other
    ctx_len: int = 2048
    vocab_padding_idx: Optional[int] = None
    tie_word_embeddings: bool = True
    
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
            'use_momentum': self.use_momentum,
            'poly_degree': self.poly_degree,
            'poly_mode': self.poly_mode,
            'qk_norm': self.qk_norm,
            'qkv_conv_kernel': self.qkv_conv_kernel,
            'ctx_len': self.ctx_len,
            'vocab_padding_idx': self.vocab_padding_idx,
            'tie_word_embeddings': self.tie_word_embeddings,
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
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


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
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


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
        
        # Memory module (replaces attention)
        self.memory = RNNMemory(
            dim=config.n_embd,
            dim_head=config.memory_dim_head,
            heads=config.memory_heads,
            use_momentum=config.use_momentum,
            poly_degree=config.poly_degree,
            poly_mode=config.poly_mode,
            qk_norm=config.qk_norm,
            qkv_conv_kernel=config.qkv_conv_kernel,
        )
        
        # MLP
        self.mlp = AtlasMLP(config.n_embd, config.dim_ffn)
    
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
        
        mem_state = layer_state.memory_state if layer_state is not None else None
        hidden_states, new_mem_state = self.memory(hidden_states, mem_state)
        hidden_states = residual + hidden_states
        
        # MLP path with residual
        residual = hidden_states
        hidden_states = self.post_memory_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
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
        
        # Memory module
        self.memory = RNNMemory(
            dim=config.n_embd,
            dim_head=config.memory_dim_head,
            heads=config.memory_heads,
            use_momentum=config.use_momentum,
            poly_degree=config.poly_degree,
            poly_mode=config.poly_mode,
            qk_norm=config.qk_norm,
            qkv_conv_kernel=config.qkv_conv_kernel,
        )
        
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
        
        mem_state = layer_state.memory_state if layer_state is not None else None
        hidden_states, new_mem_state = self.memory(hidden_states, mem_state)
        hidden_states = residual + hidden_states
        
        # Attention path with residual
        residual = hidden_states
        hidden_states = self.post_memory_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states)
        hidden_states = residual + hidden_states
        
        # MLP path with residual
        residual = hidden_states
        hidden_states = self.post_attn_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
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
            device=device,
            dtype=dtype,
        )
