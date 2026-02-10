"""
Atlas-Qwen2 Training Model: Training-facing wrapper for Atlas-LMM.

This module provides the Model_atlasqwen2 class that integrates with
RADLADS-atlas train.py via the config.model.classname mechanism.

Usage in config:
    model:
        classname: atlasqwen2
        n_embd: 896
        n_layer: 24
        ...
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from models.atlasqwen2_core import (
    AtlasConfig,
    AtlasQwen2Core,
    AtlasQwen2ForCausalLM,
    AtlasModelState,
    AtlasLayerState,
    init_model_state,
    detach_model_state,
)

from src.state import ModelState, BlockState, TimeMixState, ChannelMixState


# ============================================================================
# Output dataclass (compatible with existing code)
# ============================================================================

@dataclass
class AtlasLLMOutput:
    """Output structure compatible with existing RADLADS training code."""
    logits: torch.FloatTensor = None
    model_state: ModelState = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    post_attention_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    student_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    student_post_attention_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


# ============================================================================
# Config adapter
# ============================================================================

def config_to_atlas_config(config) -> AtlasConfig:
    """Convert RADLADS TrainerCLI_Config.model to AtlasConfig."""
    model_cfg = config.model if hasattr(config, 'model') else config
    train_cfg = getattr(config, 'train', None)
    
    # Extract atlas-specific settings (with defaults)
    memory_heads = getattr(model_cfg, 'memory_heads', None)
    if memory_heads is None:
        # Default: use dim_att / head_size or n_embd / 64
        head_size = getattr(model_cfg, 'head_size', 64)
        dim_att = getattr(model_cfg, 'dim_att', model_cfg.n_embd)
        memory_heads = dim_att // head_size
    
    memory_dim_head = getattr(model_cfg, 'memory_dim_head', getattr(model_cfg, 'head_size', 64))
    
    dim_ffn = getattr(model_cfg, 'dim_ffn', 0)
    if dim_ffn <= 0:
        dim_ffn = int((model_cfg.n_embd * 3.5) // 32 * 32)
    
    return AtlasConfig(
        vocab_size=model_cfg.vocab_size,
        n_embd=model_cfg.n_embd,
        n_layer=model_cfg.n_layer,
        dim_ffn=dim_ffn,
        rms_norm_eps=getattr(model_cfg, 'rms_norm_eps', 1e-6),
        memory_heads=memory_heads,
        memory_dim_head=memory_dim_head,
        num_key_value_heads=getattr(model_cfg, 'num_key_value_heads', None),
        use_momentum=getattr(model_cfg, 'use_momentum', True),
        omega_window=getattr(model_cfg, 'omega_window', 4),
        use_omega_gate=getattr(model_cfg, 'use_omega_gate', True),
        use_cuda=getattr(model_cfg, 'use_cuda', False),
        poly_degree=getattr(model_cfg, 'poly_degree', 1),
        poly_mode=getattr(model_cfg, 'poly_mode', 'off'),
        qk_norm=getattr(model_cfg, 'qk_norm', True),
        qkv_conv_kernel=getattr(model_cfg, 'qkv_conv_kernel', None),
        use_rope=getattr(model_cfg, 'use_rope', False),
        rope_theta=getattr(model_cfg, 'rope_theta', 10000.0),
        max_position_embeddings=getattr(model_cfg, 'max_position_embeddings', 2048),
        use_groupnorm=getattr(model_cfg, 'use_groupnorm', False),
        use_accelerated_scan=getattr(model_cfg, 'use_accelerated_scan', True),
        memory_scan_chunk_len=getattr(model_cfg, 'memory_scan_chunk_len', None),
        atlas_variant=getattr(model_cfg, 'atlas_variant', 'lmm'),
        sliding_window=getattr(model_cfg, 'sliding_window', 512),
        ctx_len=model_cfg.ctx_len,
        vocab_padding_idx=getattr(model_cfg, 'vocab_padding_idx', None),
        tie_word_embeddings=getattr(model_cfg, 'tie_word_embeddings', True),
        grad_cp=getattr(train_cfg, 'grad_cp', 0) if train_cfg is not None else 0,
    )


# ============================================================================
# State conversion utilities
# ============================================================================

def atlas_state_to_model_state(atlas_state: AtlasModelState, n_layer: int) -> ModelState:
    """
    Convert AtlasModelState to RADLADS ModelState format.
    
    This allows compatibility with existing training infrastructure.
    """
    model_state = ModelState()
    
    for i in range(n_layer):
        layer_state = atlas_state.layer_states[i]
        mem_state = layer_state.memory_state
        
        # Pack memory state into TimeMixState.wkv_state format
        # We store S, Z, and seq_index in a structured way
        if mem_state.Z is not None:
            wkv_state = torch.stack([mem_state.S, mem_state.Z], dim=0)
        else:
            wkv_state = mem_state.S.unsqueeze(0)
        
        time_mix_state = TimeMixState(wkv_state=wkv_state, shift_state=torch.tensor([]))
        channel_mix_state = ChannelMixState()
        
        model_state.block_states.append(BlockState(time_mix_state, channel_mix_state))
    
    model_state.seq_pos = atlas_state.seen_tokens
    return model_state


def model_state_to_atlas_state(
    model_state: Optional[ModelState], 
    n_layer: int,
    batch: int,
    heads: int,
    dim_head: int,
    use_momentum: bool,
    omega_window: int = 4,
    device=None,
    dtype=None,
) -> AtlasModelState:
    """
    Convert RADLADS ModelState back to AtlasModelState.
    """
    if model_state is None or len(model_state.block_states) == 0:
        return init_model_state(n_layer, batch, heads, dim_head, use_momentum, omega_window, device, dtype)
    
    layer_states = []
    for i in range(n_layer):
        block_state = model_state.block_states[i]
        wkv_state = block_state.time_mix_state.wkv_state
        
        if use_momentum and wkv_state.shape[0] >= 2:
            S = wkv_state[0]
            Z = wkv_state[1]
        else:
            S = wkv_state[0] if wkv_state.dim() > 3 else wkv_state
            Z = None
        
        from models.atlas_memory import RNNMemState
        mem_state = RNNMemState(
            seq_index=model_state.seq_pos,
            S=S,
            Z=Z,
            omega_buffer=None,
        )
        layer_states.append(AtlasLayerState(memory_state=mem_state))
    
    return AtlasModelState(layer_states=layer_states, seen_tokens=model_state.seq_pos)


# ============================================================================
# Training Model (compatible with train.py's classname mechanism)
# ============================================================================

class Model_atlasqwen2(nn.Module):
    """
    Atlas-LMM training model compatible with RADLADS train.py.
    
    This is the entry point when config.model.classname = 'atlasqwen2'.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.atlas_config = config_to_atlas_config(config)
        
        # Model will be created in configure_model (delayed initialization)
        self.model = None
        self.lm_head = None
    
    def configure_model(self):
        """Initialize model weights (called after distributed setup)."""
        if self.model is not None:
            return
        
        # Create the core model
        self.model = AtlasQwen2Core(self.atlas_config)
        
        # Create LM head
        self.lm_head = nn.Linear(
            self.atlas_config.n_embd, 
            self.atlas_config.vocab_size, 
            bias=False
        )
        
        # Optionally tie embeddings
        if self.atlas_config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
    
    def forward(
        self,
        token_ids: Tensor,
        last_model_state: Optional[ModelState] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,  # Ignored for LMM (no attention)
        output_post_attention_hidden_states: bool = False,  # Ignored for LMM
    ) -> AtlasLLMOutput:
        """
        Forward pass compatible with RADLADS training code.
        
        Args:
            token_ids: [batch, seq_len] input token IDs
            last_model_state: Optional previous ModelState
            output_hidden_states: Whether to return hidden states
            output_attentions: Ignored (no attention in LMM)
            output_post_attention_hidden_states: Ignored
            
        Returns:
            AtlasLLMOutput with logits and state
        """
        batch = token_ids.shape[0]
        
        # Convert state format if provided
        atlas_state = model_state_to_atlas_state(
            last_model_state,
            n_layer=self.atlas_config.n_layer,
            batch=batch,
            heads=self.atlas_config.memory_heads,
            dim_head=self.atlas_config.memory_dim_head,
            use_momentum=self.atlas_config.use_momentum,
            omega_window=self.atlas_config.omega_window,
            device=token_ids.device,
            dtype=self.model.embed_tokens.weight.dtype,
        )
        
        # Run forward
        hidden_states, new_atlas_state, all_hidden_states = self.model(
            token_ids, atlas_state, output_hidden_states
        )
        # Apply LM head
        logits = self.lm_head(hidden_states)
        
        # Convert state back to ModelState format
        new_model_state = atlas_state_to_model_state(
            new_atlas_state, self.atlas_config.n_layer
        )
        
        return AtlasLLMOutput(
            logits=logits,
            model_state=new_model_state,
            hidden_states=all_hidden_states,
            attentions=None,  # No attention in LMM
            post_attention_hidden_states=None,
            student_attentions=None,
            student_post_attention_hidden_states=None,
        )
    
    def get_optim_groups(self):
        """
        Return optimizer parameter groups with weight decay settings.
        
        Compatible with RADLADS training infrastructure.
        """
        train_config = self.config.train
        
        lr_decay = set()
        lr_1x = set()
        
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            
            # Apply weight decay to 2D+ params (not biases, not norms)
            if train_config.weight_decay > 0 and '.bias' not in n and 'norm' not in n and 'ln' not in n:
                lr_decay.add(n)
            else:
                lr_1x.add(n)
        
        param_dict = {n: p for n, p in self.named_parameters()}
        
        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        
        print('1x', len(lr_1x))
        print('decay', len(lr_decay))
        
        optim_groups = [
            {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0, 'name': 'lr_1x'},
        ]
        if len(lr_decay) > 0:
            optim_groups += [
                {"params": [param_dict[n] for n in lr_decay], "weight_decay": train_config.weight_decay, "my_lr_scale": 1.0, 'name': 'lr_decay'}
            ]
        
        return optim_groups
    
    def init_all_weights(self):
        """Initialize model weights (optional, usually loaded from checkpoint)."""
        pass
    
    def set_grads(self):
        """
        Set which parameters require gradients based on training stage.
        
        Paper Stage 1 (attention_distillation_stage=1): Train only Atlas memory parameters
            - Freeze: embeddings, lm_head, MLP layers
            - Train: All OmegaRNNMemoryCell parameters (Q/K/V, gates, conv, norms)
        
        Paper Stage 2/3 (attention_distillation_stage != 1): Train everything
            - Paper Stage 2: All params with teacher KL loss
            - Paper Stage 3: All params with CE loss only
        """
        train_config = self.config.train
        if train_config is None:
            return
        
        if train_config.attention_distillation_stage == 1:
            # Paper Stage 1: Only train memory parameters
            print("Paper Stage 1 (Attention Alignment): Freezing all parameters except Atlas memory cells")
            self.requires_grad_(False)
            
            # Unfreeze all Atlas memory parameters in each layer
            for layer_idx, layer in enumerate(self.model.layers):
                # RNNMemory is a wrapper around the actual cell
                memory_cell = layer.memory.cell if hasattr(layer.memory, 'cell') else layer.memory
                print(f"  Layer {layer_idx}: Unfreezing Atlas memory cell")
                
                # Core projections (Q/K/V + output)
                memory_cell.to_q.requires_grad_(True)
                memory_cell.to_k.requires_grad_(True)
                memory_cell.to_v.requires_grad_(True)
                memory_cell.to_out.requires_grad_(True)
                
                # Learned gates (lr, decay, momentum, omega_gate)
                memory_cell.to_lr.requires_grad_(True)
                memory_cell.to_decay.requires_grad_(True)
                if memory_cell.use_momentum and hasattr(memory_cell, 'to_momentum'):
                    memory_cell.to_momentum.requires_grad_(True)
                if hasattr(memory_cell, 'use_omega_gate') and memory_cell.use_omega_gate and hasattr(memory_cell, 'to_omega_gate'):
                    memory_cell.to_omega_gate.requires_grad_(True)
                
                # Normalizations (pre_norm, q_norm, k_norm, groupnorm)
                memory_cell.pre_norm.requires_grad_(True)
                if hasattr(memory_cell, 'q_norm') and isinstance(memory_cell.q_norm, nn.Module):
                    memory_cell.q_norm.requires_grad_(True)
                if hasattr(memory_cell, 'k_norm') and isinstance(memory_cell.k_norm, nn.Module):
                    memory_cell.k_norm.requires_grad_(True)
                if hasattr(memory_cell, 'groupnorm') and memory_cell.groupnorm is not None:
                    memory_cell.groupnorm.requires_grad_(True)
                
                # Optional conv layers (q_conv, k_conv, v_conv)
                if hasattr(memory_cell, 'q_conv') and memory_cell.q_conv is not None:
                    memory_cell.q_conv.requires_grad_(True)
                if hasattr(memory_cell, 'k_conv') and memory_cell.k_conv is not None:
                    memory_cell.k_conv.requires_grad_(True)
                if hasattr(memory_cell, 'v_conv') and memory_cell.v_conv is not None:
                    memory_cell.v_conv.requires_grad_(True)
                
                # Polynomial feature map (if it has learnable params)
                if hasattr(memory_cell, 'phi') and hasattr(memory_cell.phi, 'proj'):
                    # proj is a buffer, not a parameter, so skip
                    pass
                
                # ROPE (rotary_emb) - usually has no learnable params, but check
                if hasattr(memory_cell, 'rotary_emb') and memory_cell.rotary_emb is not None:
                    if hasattr(memory_cell.rotary_emb, 'parameters'):
                        for p in memory_cell.rotary_emb.parameters():
                            p.requires_grad_(True)
            
            print("  Paper Stage 1 gradient setup complete")
        
        else:
            # Paper Stage 2/3: Train everything
            stage_name = "Paper Stage 2 (KL Divergence)" if train_config.attention_distillation_stage == 2 else "Paper Stage 3 (Long Context CE)"
            print(f"{stage_name}: Training all parameters")
            self.requires_grad_(True)
