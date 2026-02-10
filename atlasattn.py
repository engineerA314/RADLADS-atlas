# coding=utf-8
"""
Atlas Attention Replacement for RADLADS Stage 1 Distillation.

This module provides:
1. AtlasSelfAttention: Wraps Atlas memory to match HuggingFace self-attention interface
2. AttentionDistillationWrapper: Runs teacher (softmax) and student (Atlas) in parallel
3. load_and_patch_model_with_attention_replacement: Patches HF model for Stage 1

Usage:
    python train.py -c configs/atlasqwen0b5.yaml -c configs/distill1.yaml
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple, Callable

from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers import AutoConfig, AutoModelForCausalLM
from pydoc import locate

from transformers.utils import logging
logger = logging.get_logger(__name__)

from models.atlas_memory import RNNMemory, RNNMemState

_NAN_DEBUG_CTX = {"batch_idx": None, "global_step": None}


def set_nan_debug_context(*, batch_idx: int | None = None, global_step: int | None = None) -> None:
    _NAN_DEBUG_CTX["batch_idx"] = batch_idx
    _NAN_DEBUG_CTX["global_step"] = global_step
    try:
        if batch_idx is not None:
            os.environ["NAN_DEBUG_BATCH_IDX"] = str(int(batch_idx))
    except Exception:
        pass


def get_nan_debug_context() -> dict:
    # Return a shallow copy so callers can't mutate global state.
    return dict(_NAN_DEBUG_CTX)


class AtlasSelfAttention(nn.Module):
    """
    Atlas Memory wrapped to match HuggingFace self-attention interface.
    
    This allows Atlas memory to be used as a drop-in replacement for
    softmax attention in Qwen2 models during Stage 1 distillation.
    
    RNNMemory factory handles its own Q/K/V projections, so this is just a wrapper.
    """
    
    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Atlas / ablation configuration from config if available
        # NOTE: For HF patching, these fields are injected into the HF config
        # in load_and_patch_model_with_attention_replacement(...).
        omega_window = getattr(config, 'omega_window', 4)
        use_omega_gate = getattr(config, 'use_omega_gate', True)
        use_momentum = getattr(config, 'use_momentum', True)
        use_cuda = getattr(config, 'use_cuda', False)  # CUDA backend for 50-100x speedup
        use_accelerated_scan = getattr(config, 'use_accelerated_scan', True)
        memory_scan_chunk_len = getattr(config, 'memory_scan_chunk_len', None)
        poly_degree = getattr(config, 'poly_degree', 2)
        poly_mode = getattr(config, 'poly_mode', 'elementwise')
        qk_norm = getattr(config, 'qk_norm', False)
        qkv_conv_kernel = getattr(config, 'qkv_conv_kernel', None)
        use_rope = getattr(config, 'use_rope', False)
        use_groupnorm = getattr(config, 'use_groupnorm', False)

        # GQA / RoPE params (if present in HF config)
        # Note: num_key_value_heads must be <= heads and must divide heads.
        # Some unit-test MockConfig values violate this (e.g. heads=2, num_key_value_heads=4),
        # so we fall back to "no GQA" in that case.
        num_key_value_heads = getattr(config, 'num_key_value_heads', None)
        if not isinstance(num_key_value_heads, int) or num_key_value_heads <= 0:
            num_key_value_heads = None
        elif num_key_value_heads > self.num_heads or (self.num_heads % num_key_value_heads) != 0:
            num_key_value_heads = None
        rope_theta = getattr(config, 'rope_theta', 10000.0)
        max_position_embeddings = getattr(config, 'max_position_embeddings', 2048)
        
        # Atlas memory (uses OmegaRNNMemoryCell internally, handles its own projections)
        self.memory = RNNMemory(
            dim=self.hidden_size,
            dim_head=self.head_dim,
            heads=self.num_heads,
            num_key_value_heads=num_key_value_heads,
            use_momentum=use_momentum,
            omega_window=omega_window,
            use_omega_gate=use_omega_gate,
            use_cuda=use_cuda,
            use_accelerated_scan=use_accelerated_scan,
            scan_chunk_len=memory_scan_chunk_len,
            poly_degree=poly_degree,
            poly_mode=poly_mode,
            qk_norm=qk_norm,
            qkv_conv_kernel=qkv_conv_kernel,
            use_rope=use_rope,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
            use_groupnorm=use_groupnorm,
        )

        # Expose layer index for deep debugging (used by omega debug hooks).
        try:
            self.memory.layer_idx = int(layer_idx)
            if hasattr(self.memory, "cell"):
                self.memory.cell.layer_idx = int(layer_idx)
        except Exception:
            pass
        
        # Layer-specific state storage (for streaming)
        self._memory_state = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass matching HuggingFace attention interface.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Ignored (Atlas memory is causal by nature)
            position_ids: Ignored (Atlas uses implicit position)
            past_key_value: Ignored (Atlas has its own state management)
            output_attentions: Ignored (Atlas doesn't produce attention weights)
            use_cache: If True, maintains memory state
            
        Returns:
            (output, None, None) matching HF attention return format
        """
        # During training, always start fresh to avoid gradient issues
        # For inference with use_cache, we can maintain state
        if self.training or self._memory_state is None or not use_cache:
            state = None  # Let RNNMemoryCell create fresh state
        else:
            # Detach state to avoid gradient accumulation across calls
            state = self._memory_state
            if state is not None and hasattr(state, 'S'):
                from models.atlas_memory import RNNMemState
                state = RNNMemState(
                    seq_index=state.seq_index,
                    S=state.S.detach() if state.S is not None else None,
                    Z=state.Z.detach() if state.Z is not None else None,
                    omega_buffer=state.omega_buffer.detach() if state.omega_buffer is not None else None,
                )
        
        # Apply Atlas memory (handles Q/K/V projections internally)
        output, new_state = self.memory(
            seq=hidden_states,
            state=state,
        )
        
        # Store state for potential streaming inference
        if use_cache and not self.training:
            self._memory_state = new_state
        
        # Return format matching HF: (hidden_states, attn_weights, past_key_value)
        # Return: (hidden_states, attention_weights) matching HF Qwen2DecoderLayer expectations
        # attention_weights is None since we don't use attention
        return output, None


class AttentionDistillationWrapper(nn.Module):
    """
    Wraps teacher (softmax) and student (Atlas) attention for Stage 1 distillation.
    
    During forward:
    1. Runs both teacher and student attention on same input
    2. Returns teacher output (so model behavior unchanged)
    3. Computes and stores attention loss for training
    """
    
    def __init__(
        self,
        original_self_attn: nn.Module,
        ReplacementSelfAttentionType: Callable,
        model_config: Any,
        attention_distillation_stage: int,
    ):
        super().__init__()
        self.teacher_attn = original_self_attn
        self.student_attn = ReplacementSelfAttentionType(model_config, original_self_attn.layer_idx)
        assert attention_distillation_stage == 1, "AttentionDistillationWrapper only supports stage 1"
        self.attention_distillation_stage = attention_distillation_stage
        
        # Store last computed alignment loss for collection after forward
        self._last_alignment_loss = None
        
        # Copy teacher's starting parameter values into student where names match
        student_params_dict = dict(self.student_attn.named_parameters())
        for n, p in self.teacher_attn.named_parameters():
            if n in student_params_dict:
                student_params_dict[n].requires_grad_(False)
                student_params_dict[n].copy_(p)
                student_params_dict[n].requires_grad_(p.requires_grad)

        # Explicit Q/K/V/O transfer for Atlas memory (names do not match HF attention)
        # This mirrors the RADLADS paper's "attention weight transfer" setup.
        self._copy_teacher_qkvo_if_possible()

    def _copy_teacher_qkvo_if_possible(self) -> None:
        teacher = self.teacher_attn
        student = self.student_attn

        # AtlasSelfAttention owns an RNNMemory wrapper; projections live on its cell.
        memory = getattr(student, "memory", None)
        cell = getattr(memory, "cell", memory) if memory is not None else None
        if cell is None:
            return

        def _copy_weight(src: nn.Module | None, dst: nn.Module | None) -> None:
            if src is None or dst is None:
                return
            if not (hasattr(src, "weight") and hasattr(dst, "weight")):
                return
            if src.weight.shape != dst.weight.shape:
                return
            with torch.no_grad():
                dst.weight.copy_(src.weight)

        _copy_weight(getattr(teacher, "q_proj", None), getattr(cell, "to_q", None))
        _copy_weight(getattr(teacher, "k_proj", None), getattr(cell, "to_k", None))
        _copy_weight(getattr(teacher, "v_proj", None), getattr(cell, "to_v", None))
        _copy_weight(getattr(teacher, "o_proj", None), getattr(cell, "to_out", None))
    
    def forward(self, hidden_states, *args, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass running both teacher and student.
        
        Returns TEACHER output to preserve forward behavior.
        Student gradients come from the alignment loss only.
        HuggingFace Qwen2 expects (hidden_states, attention_weights) tuple.
        """
        # Don't need actual attention weights from teacher
        kwargs['output_attentions'] = False
        
        # Run student (with gradients) on original hidden_states
        student_outputs = self.student_attn(hidden_states, *args, **kwargs)
        
        # Run teacher (no gradients - it's frozen) on detached hidden_states
        with torch.no_grad():
            teacher_outputs = self.teacher_attn(hidden_states.detach(), *args, **kwargs)
        
        # Compute alignment loss: ||teacher - student||
        student_hidden = student_outputs[0]
        teacher_hidden = teacher_outputs[0]  # Already detached via no_grad
        alignment_loss = torch.linalg.vector_norm(
            teacher_hidden - student_hidden, dim=-1
        ).mean() * (teacher_hidden.size(-1) ** -0.5)
        
        # Store for collection
        self._last_alignment_loss = alignment_loss
        
        # Return (hidden_states, attention_weights) as expected by HF Qwen2
        # attention_weights is the alignment_loss for collection
        return (teacher_outputs[0], alignment_loss) + teacher_outputs[2:]



def load_and_patch_model_with_attention_replacement(
    model_path: str,
    attn_classes_path: str,
    ReplacementSelfAttentionType: Callable,
    attention_distillation_stage: int,
    atlas_model_config: Any = None,
):
    """
    Load HuggingFace model and patch attention layers for distillation.
    
    Args:
        model_path: Path to HuggingFace model (e.g., "Qwen/Qwen2-0.5B")
        attn_classes_path: Ignored (kept for compatibility)
        ReplacementSelfAttentionType: Class to use as student (e.g., AtlasSelfAttention)
        attention_distillation_stage: 1 for Stage 1, 2 for Stage 2
        
    Returns:
        Patched HuggingFace model
    """
    model_config = AutoConfig.from_pretrained(model_path)

    # Inject Atlas-specific knobs into the HF config so AtlasSelfAttention can see them.
    # This is required for WITH_ALL_ABLATION runs: Stage 1 must create the same optional
    # submodules (qk_norm, conv, groupnorm, rope) as Stage 2 expects.
    if atlas_model_config is not None:
        for k in [
            # Omega / core Atlas
            'omega_window', 'use_omega_gate', 'use_momentum', 'use_cuda',
            'use_accelerated_scan', 'memory_scan_chunk_len',
            # Ablation knobs
            'poly_degree', 'poly_mode', 'qk_norm', 'qkv_conv_kernel', 'use_rope', 'use_groupnorm',
            # Optional RoPE params
            'rope_theta', 'max_position_embeddings',
        ]:
            if hasattr(atlas_model_config, k):
                try:
                    setattr(model_config, k, getattr(atlas_model_config, k))
                except Exception:
                    # HF configs should allow arbitrary fields, but keep this robust.
                    pass
    
    # Handle Qwen-specific config
    if 'Qwen' in model_path:
        model_config.attention_bias = True
        model_config.attention_output_bias = False
    
    # Load model normally
    logger.info(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, config=model_config)
    
    # Get the layers (works for Qwen2, Llama, etc.)
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
    else:
        raise ValueError(f"Unknown model architecture: {type(model)}")
    
    logger.info(f"Found {len(layers)} layers to patch")
    
    # Patch model for Stage 1
    if attention_distillation_stage == 1:
        # Freeze entire model (teacher)
        model.requires_grad_(False)
        
        # Wrap each attention layer with distillation wrapper
        for i, layer in enumerate(layers):
            original_attn = layer.self_attn
            layer.self_attn = AttentionDistillationWrapper(
                original_attn,
                ReplacementSelfAttentionType,
                model_config,
                attention_distillation_stage,
            )
            logger.info(f"Patched layer {i} with AttentionDistillationWrapper")
        
        # Enable gradients for student attention
        for layer in layers:
            for n, p in layer.self_attn.student_attn.named_parameters():
                p.requires_grad_(True)
    
    elif attention_distillation_stage >= 2:
        # Stage 2+: Replace attention entirely with student
        for i, layer in enumerate(layers):
            original_attn = layer.self_attn
            layer.self_attn = ReplacementSelfAttentionType(model_config, i)
            logger.info(f"Replaced layer {i} attention with {ReplacementSelfAttentionType.__name__}")
        
        model.requires_grad_(True)
    
    return model


# Default attention replacement type
ATLAS_ATTENTION_CLASSES = {
    "eager": AtlasSelfAttention,
    "flash_attention_2": AtlasSelfAttention,
    "sdpa": AtlasSelfAttention,
}

