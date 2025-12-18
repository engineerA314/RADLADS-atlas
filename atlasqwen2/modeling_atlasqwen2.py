"""
HuggingFace-compatible Atlas-Qwen2 (LMM) model.

This module provides HuggingFace PreTrainedModel wrappers around the core
Atlas-LMM implementation, enabling:
- AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)
- model.generate() with greedy/sampling
- save_pretrained / load_pretrained
"""

from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
from torch import Tensor

# Note: Not inheriting from Cache due to API changes in transformers 4.47+
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_atlasqwen2 import AtlasQwen2Config

# Import core implementation
import sys
import os
# Ensure models directory is in path for import
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from models.atlasqwen2_core import (
    AtlasQwen2Core as CoreModel,
    AtlasQwen2ForCausalLM as CoreCausalLM,
    AtlasConfig as CoreConfig,
    AtlasModelState,
    AtlasLayerState,
    init_model_state,
)
from models.atlas_memory import RNNMemState


logger = logging.get_logger(__name__)


# ============================================================================
# Atlas State (Cache) - HuggingFace compatible
# ============================================================================

class AtlasState:
    """
    HuggingFace-compatible cache for Atlas memory state.
    
    Stores per-layer memory state (S, Z, omega_buffer) and tracks seen tokens.
    Used as `past_key_values` in HF generation.
    
    Per-layer payload (fixed tuple length; stable contract):
    - S: Memory matrix tensor [batch*heads, dim_head, dim_head]
    - Z: Momentum tensor (placeholder if disabled)
    - omega_buffer: Omega window buffer (placeholder if disabled)
    
    Note: We don't inherit from transformers.cache_utils.Cache due to API changes
    in transformers 4.47+. We implement the required interface manually.
    """
    
    def __init__(self) -> None:
        self._seen_tokens = 0
        self.layer_memory_states: List[RNNMemState] = []
    
    def __getitem__(self, layer_idx: int) -> RNNMemState:
        if layer_idx < len(self):
            return self.layer_memory_states[layer_idx]
        raise KeyError(f"Cache only has {len(self)} layers, attempted to access {layer_idx}")
    
    def __iter__(self):
        for state in self.layer_memory_states:
            yield state
    
    def __len__(self):
        return len(self.layer_memory_states)
    
    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """RNN memory has no maximum length."""
        return new_seq_length
    
    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Beam search reordering - not yet implemented."""
        raise NotImplementedError('Beam search reordering not supported for Atlas memory state')
    
    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns total tokens seen so far."""
        return self._seen_tokens
    
    def get_max_cache_shape(self) -> Optional[int]:
        """No maximum length for RNN memory."""
        return None
    
    def get_max_length(self) -> Optional[int]:
        """No maximum length for RNN memory."""
        return None
    
    def crop(self, max_length: int):
        """Cannot crop RNN state."""
        return
    
    @torch.no_grad()
    def update(
        self,
        memory_state: RNNMemState,
        token_count: int,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> RNNMemState:
        """Update cache with new layer state."""
        if layer_idx == 0:
            self._seen_tokens += token_count
        
        # Extend list if needed
        while len(self.layer_memory_states) <= layer_idx:
            self.layer_memory_states.append(None)
        
        self.layer_memory_states[layer_idx] = memory_state
        return memory_state
    
    @classmethod
    def from_atlas_model_state(cls, model_state: AtlasModelState) -> 'AtlasState':
        """Convert internal AtlasModelState to HF-compatible AtlasState."""
        cache = cls()
        cache._seen_tokens = model_state.seen_tokens
        for layer_state in model_state.layer_states:
            cache.layer_memory_states.append(layer_state.memory_state)
        return cache
    
    def to_atlas_model_state(self) -> AtlasModelState:
        """Convert back to internal AtlasModelState."""
        layer_states = [
            AtlasLayerState(memory_state=mem_state)
            for mem_state in self.layer_memory_states
        ]
        return AtlasModelState(layer_states=layer_states, seen_tokens=self._seen_tokens)


# ============================================================================
# HuggingFace PreTrainedModel Base
# ============================================================================

class AtlasQwen2PreTrainedModel(PreTrainedModel):
    """Base class for Atlas-Qwen2 models."""
    
    config_class = AtlasQwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["AtlasLMMBlock"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_cache_class = True
    _supports_static_cache = False  # RNN state is different from static KV cache
    
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# ============================================================================
# Atlas-Qwen2 Model (decoder only, no lm_head)
# ============================================================================

class AtlasQwen2Model(AtlasQwen2PreTrainedModel):
    """
    Atlas-Qwen2 decoder model (without LM head).
    
    This wraps the core AtlasQwen2Core and provides HF-compatible I/O.
    """
    
    def __init__(self, config: AtlasQwen2Config):
        super().__init__(config)
        
        core_config = config.to_atlas_core_config()
        
        # Build layers directly (no _core wrapper to avoid duplicate keys)
        self.embed_tokens = nn.Embedding(
            core_config.vocab_size, 
            core_config.n_embd, 
            padding_idx=core_config.vocab_padding_idx
        )
        
        from models.atlasqwen2_core import create_atlas_block, AtlasRMSNorm
        self.layers = nn.ModuleList([
            create_atlas_block(core_config, layer_idx=i) 
            for i in range(core_config.n_layer)
        ])
        
        self.norm = AtlasRMSNorm(core_config.n_embd, eps=core_config.rms_norm_eps)
        
        self._core_config = core_config
        
        self.post_init()
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,  # Ignored for LMM
        position_ids: Optional[torch.LongTensor] = None,  # Ignored
        past_key_values: Optional[AtlasState] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,  # Ignored (no attention)
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,  # Ignored
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if inputs_embeds is not None:
            raise NotImplementedError("inputs_embeds not yet supported for Atlas")
        
        batch, seq_len = input_ids.shape
        
        # Initialize state if not provided or if it's an incompatible cache type
        if past_key_values is None or len(past_key_values) == 0:
            layer_states = [None] * len(self.layers)
            seen_tokens = 0
        elif isinstance(past_key_values, AtlasState):
            layer_states = list(past_key_values.layer_memory_states)
            seen_tokens = past_key_values._seen_tokens
            # Pad if needed
            while len(layer_states) < len(self.layers):
                layer_states.append(None)
        else:
            # HF generate may pass a DynamicCache - we ignore it and use fresh state
            # This happens on first call before we've returned our AtlasState
            layer_states = [None] * len(self.layers)
            seen_tokens = 0
        
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        
        # Track hidden states if requested
        all_hidden_states = () if output_hidden_states else None
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Process through layers
        new_layer_states = []
        for layer_idx, layer in enumerate(self.layers):
            layer_state = layer_states[layer_idx]
            # Wrap in AtlasLayerState if we have a memory state
            if layer_state is not None:
                from models.atlasqwen2_core import AtlasLayerState
                layer_state = AtlasLayerState(memory_state=layer_state)
            
            hidden_states, new_layer_state = layer(hidden_states, layer_state)
            new_layer_states.append(new_layer_state.memory_state)
            
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        # Build new cache
        new_cache = None
        if use_cache:
            new_cache = AtlasState()
            new_cache._seen_tokens = seen_tokens + seq_len
            new_cache.layer_memory_states = new_layer_states
        
        if not return_dict:
            return (hidden_states, new_cache, all_hidden_states)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=new_cache,
            hidden_states=all_hidden_states,
            attentions=None,  # No attention in LMM
        )


# ============================================================================
# Atlas-Qwen2 For Causal LM
# ============================================================================

class AtlasQwen2ForCausalLM(AtlasQwen2PreTrainedModel, GenerationMixin):
    """
    Atlas-Qwen2 Causal Language Model with LM head.
    
    Supports HuggingFace generate() for text generation.
    """
    
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config: AtlasQwen2Config):
        super().__init__(config)
        
        self.model = AtlasQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def get_decoder(self):
        return self.model
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[AtlasState] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Forward through decoder
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        
        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        
        # Only compute logits for tokens we need (memory optimization for generation)
        if logits_to_keep > 0:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if return_dict else outputs[1],
            hidden_states=outputs.hidden_states if return_dict else outputs[2],
            attentions=None,
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[AtlasState] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """Prepare inputs for generation step."""
        
        # If we have past state, only feed the last token
        if past_key_values is not None and len(past_key_values) > 0:
            input_ids = input_ids[:, -1:]
        
        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
            "attention_mask": attention_mask,
        }
        
        return model_inputs
    
    @staticmethod
    def _reorder_cache(past_key_values: AtlasState, beam_idx: torch.LongTensor):
        """Reorder cache for beam search - raises error (not supported)."""
        raise NotImplementedError("Beam search not supported for Atlas memory state")
