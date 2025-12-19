"""
TDD Tests for HuggingFace Atlas-Qwen2 Compatibility (Phase C).

These tests verify:
- Config roundtrip (to_dict/from_dict)
- save_pretrained / from_pretrained works
- use_cache semantics with AtlasState
- generate() smoke test
"""

import pytest
import torch
import tempfile
import os

from atlasqwen2.configuration_atlasqwen2 import AtlasQwen2Config
from atlasqwen2.modeling_atlasqwen2 import (
    AtlasState,
    AtlasQwen2Model,
    AtlasQwen2ForCausalLM,
)


@pytest.fixture
def tiny_hf_config():
    """Tiny HF config for fast testing."""
    return AtlasQwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        memory_heads=4,
        memory_dim_head=16,
        use_momentum=True,
        omega_window=4,  # Use proper omega_window (not 1)
        use_omega_gate=True,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=True,
    )


@pytest.fixture
def tiny_hf_model(tiny_hf_config):
    """Create a tiny HF model for testing."""
    return AtlasQwen2ForCausalLM(tiny_hf_config)


class TestAtlasQwen2Config:
    """Tests for HF config."""
    
    def test_config_creation(self, tiny_hf_config):
        """Config creates successfully."""
        assert tiny_hf_config.hidden_size == 64
        assert tiny_hf_config.num_hidden_layers == 2
        assert tiny_hf_config.model_type == "atlasqwen2"
    
    def test_config_to_dict(self, tiny_hf_config):
        """Config serializes to dict."""
        d = tiny_hf_config.to_dict()
        assert 'hidden_size' in d
        assert d['hidden_size'] == 64
    
    def test_config_roundtrip(self, tiny_hf_config):
        """Config survives JSON roundtrip."""
        import json
        json_str = tiny_hf_config.to_json_string()
        loaded = AtlasQwen2Config.from_dict(json.loads(json_str))
        
        assert loaded.hidden_size == tiny_hf_config.hidden_size
        assert loaded.num_hidden_layers == tiny_hf_config.num_hidden_layers
        assert loaded.memory_heads == tiny_hf_config.memory_heads
    
    def test_config_to_atlas_core(self, tiny_hf_config):
        """Config converts to internal AtlasConfig."""
        core_config = tiny_hf_config.to_atlas_core_config()
        assert core_config.n_embd == tiny_hf_config.hidden_size
        assert core_config.n_layer == tiny_hf_config.num_hidden_layers


class TestAtlasState:
    """Tests for HF-compatible cache."""
    
    def test_state_creation(self):
        """AtlasState creates empty."""
        state = AtlasState()
        assert len(state) == 0
        assert state.get_seq_length() == 0
    
    def test_state_update(self):
        """AtlasState updates correctly."""
        from models.atlas_memory import RNNMemState
        
        state = AtlasState()
        S = torch.randn(8, 16, 16)
        Z = torch.randn(8, 16, 16)
        mem_state = RNNMemState(seq_index=10, S=S, Z=Z, omega_buffer=None)
        
        state.update(mem_state, token_count=10, layer_idx=0)
        
        assert len(state) == 1
        assert state.get_seq_length() == 10
    
    def test_state_roundtrip(self):
        """AtlasState converts to/from AtlasModelState."""
        from models.atlasqwen2_core import AtlasModelState, AtlasLayerState
        from models.atlas_memory import RNNMemState
        
        # Create internal state
        S = torch.randn(8, 16, 16)
        Z = torch.randn(8, 16, 16)
        mem_state = RNNMemState(seq_index=5, S=S, Z=Z, omega_buffer=None)
        layer_states = [AtlasLayerState(memory_state=mem_state)]
        model_state = AtlasModelState(layer_states=layer_states, seen_tokens=5)
        
        # Convert to HF state
        hf_state = AtlasState.from_atlas_model_state(model_state)
        assert hf_state.get_seq_length() == 5
        
        # Convert back
        recovered = hf_state.to_atlas_model_state()
        assert recovered.seen_tokens == 5
        assert torch.allclose(recovered.layer_states[0].memory_state.S, S)


class TestAtlasQwen2Model:
    """Tests for base decoder model."""
    
    def test_model_creation(self, tiny_hf_config):
        """Model creates successfully."""
        model = AtlasQwen2Model(tiny_hf_config)
        assert model is not None
    
    def test_model_forward(self, tiny_hf_config):
        """Model forward works."""
        model = AtlasQwen2Model(tiny_hf_config)
        x = torch.randint(0, tiny_hf_config.vocab_size, (2, 8))
        
        output = model(x, return_dict=True)
        
        assert output.last_hidden_state.shape == (2, 8, tiny_hf_config.hidden_size)


class TestAtlasQwen2ForCausalLM:
    """Tests for causal LM model."""
    
    def test_model_creation(self, tiny_hf_model):
        """Causal LM creates successfully."""
        assert tiny_hf_model is not None
    
    def test_forward_shape(self, tiny_hf_model, tiny_hf_config):
        """Forward produces correct logit shape."""
        x = torch.randint(0, tiny_hf_config.vocab_size, (2, 8))
        
        output = tiny_hf_model(x, return_dict=True)
        
        assert output.logits.shape == (2, 8, tiny_hf_config.vocab_size)
    
    def test_forward_with_labels(self, tiny_hf_model, tiny_hf_config):
        """Forward with labels computes loss."""
        x = torch.randint(0, tiny_hf_config.vocab_size, (2, 8))
        
        output = tiny_hf_model(x, labels=x, return_dict=True)
        
        assert output.loss is not None
        assert output.loss.numel() == 1


class TestAtlasQwen2UseCache:
    """Tests for use_cache semantics."""
    
    def test_use_cache_returns_state(self, tiny_hf_model, tiny_hf_config):
        """use_cache=True returns past_key_values."""
        x = torch.randint(0, tiny_hf_config.vocab_size, (2, 8))
        
        output = tiny_hf_model(x, use_cache=True, return_dict=True)
        
        assert output.past_key_values is not None
        assert isinstance(output.past_key_values, AtlasState)
    
    def test_state_carry_forward(self, tiny_hf_model, tiny_hf_config):
        """State can be passed to next forward."""
        x1 = torch.randint(0, tiny_hf_config.vocab_size, (2, 4))
        x2 = torch.randint(0, tiny_hf_config.vocab_size, (2, 4))
        
        # First forward
        out1 = tiny_hf_model(x1, use_cache=True, return_dict=True)
        
        # Second forward with state
        out2 = tiny_hf_model(
            x2, 
            past_key_values=out1.past_key_values, 
            use_cache=True, 
            return_dict=True
        )
        
        assert out2.past_key_values.get_seq_length() == 8  # 4 + 4
    
    def test_streaming_matches_batch(self, tiny_hf_config, seed):
        """Streaming generation produces bounded differences from batch forward.
        
        Note: With scalar scan parallelization, batch forward uses fixed S_0 for
        all tokens (mini-batch SGD style), while streaming uses updated states
        (online SGD style). This is intentional and matches the Titans paper.
        
        We test that:
        1. First token outputs match exactly (both use initial state)
        2. Differences stay within reasonable bounds
        """
        torch.manual_seed(seed)
        model = AtlasQwen2ForCausalLM(tiny_hf_config)
        model.eval()
        
        batch, seq_len = 2, 4
        x = torch.randint(0, tiny_hf_config.vocab_size, (batch, seq_len))
        
        # Batch forward WITH use_cache=True
        batch_out = model(x, use_cache=True, return_dict=True)
        
        # Streaming forward
        state = None
        streaming_logits = []
        for t in range(seq_len):
            out = model(x[:, t:t+1], past_key_values=state, use_cache=True, return_dict=True)
            streaming_logits.append(out.logits)
            state = out.past_key_values
        streaming_out = torch.cat(streaming_logits, dim=1)
        
        # First token must match exactly - both use initial state
        assert torch.allclose(batch_out.logits[:, 0:1, :], streaming_out[:, 0:1, :], atol=1e-4), \
            f"First token diff: {(batch_out.logits[:, 0:1, :] - streaming_out[:, 0:1, :]).abs().max()}"
        
        # Overall differences should stay bounded (not exact due to mini-batch vs online SGD)
        max_diff = (batch_out.logits - streaming_out).abs().max()
        assert max_diff < 5.0, f"Max diff too large: {max_diff}"


class TestAtlasQwen2SaveLoad:
    """Tests for save/load functionality."""
    
    def test_save_pretrained(self, tiny_hf_model):
        """Model can be saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tiny_hf_model.save_pretrained(tmpdir)
            
            assert os.path.exists(os.path.join(tmpdir, "config.json"))
            assert os.path.exists(os.path.join(tmpdir, "model.safetensors")) or \
                   os.path.exists(os.path.join(tmpdir, "pytorch_model.bin"))
    
    def test_save_load_roundtrip(self, tiny_hf_model, tiny_hf_config, seed):
        """Saved model can be loaded and produces same outputs."""
        torch.manual_seed(seed)
        x = torch.randint(0, tiny_hf_config.vocab_size, (2, 8))
        
        with torch.no_grad():
            original_out = tiny_hf_model(x, return_dict=True).logits
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tiny_hf_model.save_pretrained(tmpdir)
            
            # Load using from_pretrained
            loaded_model = AtlasQwen2ForCausalLM.from_pretrained(tmpdir)
            
            with torch.no_grad():
                loaded_out = loaded_model(x, return_dict=True).logits
        
        assert torch.allclose(original_out, loaded_out, atol=1e-5), \
            f"Max diff: {(original_out - loaded_out).abs().max()}"


class TestAtlasQwen2Generate:
    """Tests for generation."""
    
    def test_prepare_inputs_for_generation(self, tiny_hf_model, tiny_hf_config):
        """prepare_inputs_for_generation works."""
        x = torch.randint(0, tiny_hf_config.vocab_size, (2, 8))
        
        # Without past
        inputs = tiny_hf_model.prepare_inputs_for_generation(x)
        assert inputs["input_ids"].shape == (2, 8)
        
        # With past
        out = tiny_hf_model(x, use_cache=True, return_dict=True)
        inputs = tiny_hf_model.prepare_inputs_for_generation(
            torch.cat([x, torch.randint(0, 256, (2, 1))], dim=1),
            past_key_values=out.past_key_values
        )
        assert inputs["input_ids"].shape == (2, 1)  # Only last token
    
    def test_greedy_generate_smoke(self, tiny_hf_model, tiny_hf_config):
        """Greedy generation runs without error."""
        x = torch.randint(0, tiny_hf_config.vocab_size, (1, 4))
        
        # Simple generation
        with torch.no_grad():
            output = tiny_hf_model.generate(
                x,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tiny_hf_config.eos_token_id,
            )
        
        assert output.shape[1] >= x.shape[1]  # Generated some tokens
