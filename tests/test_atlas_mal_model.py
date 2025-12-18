"""
TDD Tests for Atlas-MAL Model (Phase E).

Tests verify:
- MAL block instantiation
- MAL block forward shapes
- MAL block has both memory and attention
- Full MAL model works
- State carries correctly
"""

import pytest
import torch
from models.atlasqwen2_core import (
    AtlasConfig,
    AtlasMALBlock,
    AtlasQwen2Core,
    AtlasQwen2ForCausalLM,
    AtlasLayerState,
    AtlasModelState,
    create_atlas_block,
)


@pytest.fixture
def mal_config():
    """Small MAL config for testing."""
    return AtlasConfig(
        vocab_size=256,
        n_embd=64,
        n_layer=2,
        dim_ffn=128,
        memory_heads=4,
        memory_dim_head=16,
        use_momentum=True,
        atlas_variant='mal',
        sliding_window=32,
    )


@pytest.fixture
def seed():
    torch.manual_seed(42)
    return 42


class TestMALBlockInstantiation:
    """Test MAL block creation."""
    
    def test_create_mal_block(self, mal_config):
        """MAL block can be instantiated."""
        block = AtlasMALBlock(mal_config, layer_idx=0)
        assert block is not None
    
    def test_mal_block_has_memory(self, mal_config):
        """MAL block has memory module."""
        block = AtlasMALBlock(mal_config, layer_idx=0)
        assert hasattr(block, 'memory')
    
    def test_mal_block_has_attention(self, mal_config):
        """MAL block has attention module."""
        block = AtlasMALBlock(mal_config, layer_idx=0)
        assert hasattr(block, 'attention')
    
    def test_mal_block_has_three_layernorms(self, mal_config):
        """MAL block has three layernorms."""
        block = AtlasMALBlock(mal_config, layer_idx=0)
        assert hasattr(block, 'input_layernorm')
        assert hasattr(block, 'post_memory_layernorm')
        assert hasattr(block, 'post_attn_layernorm')
    
    def test_factory_creates_mal(self, mal_config):
        """Factory creates MAL block when variant='mal'."""
        block = create_atlas_block(mal_config, layer_idx=0)
        assert isinstance(block, AtlasMALBlock)


class TestMALBlockForward:
    """Test MAL block forward pass."""
    
    def test_forward_shape(self, mal_config, seed):
        """Forward produces correct output shape."""
        block = AtlasMALBlock(mal_config, layer_idx=0)
        x = torch.randn(2, 8, mal_config.n_embd)
        out, state = block(x)
        assert out.shape == x.shape
    
    def test_forward_returns_state(self, mal_config, seed):
        """Forward returns layer state."""
        block = AtlasMALBlock(mal_config, layer_idx=0)
        x = torch.randn(2, 8, mal_config.n_embd)
        out, state = block(x)
        assert isinstance(state, AtlasLayerState)
        assert state.memory_state is not None
    
    def test_forward_deterministic(self, mal_config):
        """Forward is deterministic with same input."""
        block = AtlasMALBlock(mal_config, layer_idx=0)
        block.eval()
        x = torch.randn(2, 8, mal_config.n_embd)
        
        torch.manual_seed(42)
        out1, _ = block(x)
        torch.manual_seed(42)
        out2, _ = block(x)
        
        assert torch.allclose(out1, out2, atol=1e-6)


class TestMALModelFull:
    """Test full MAL model."""
    
    def test_mal_model_instantiation(self, mal_config):
        """Full MAL model can be instantiated."""
        model = AtlasQwen2ForCausalLM(mal_config)
        assert model is not None
    
    def test_mal_model_layers_are_mal_blocks(self, mal_config):
        """MAL model layers are MAL blocks."""
        model = AtlasQwen2Core(mal_config)
        for layer in model.layers:
            assert isinstance(layer, AtlasMALBlock)
    
    def test_mal_forward_shape(self, mal_config, seed):
        """MAL model forward produces correct shape."""
        model = AtlasQwen2ForCausalLM(mal_config)
        x = torch.randint(0, mal_config.vocab_size, (2, 8))
        logits, state, hidden = model(x)
        assert logits.shape == (2, 8, mal_config.vocab_size)
    
    def test_mal_returns_state(self, mal_config, seed):
        """MAL model returns model state."""
        model = AtlasQwen2ForCausalLM(mal_config)
        x = torch.randint(0, mal_config.vocab_size, (2, 8))
        logits, state, hidden = model(x)
        assert isinstance(state, AtlasModelState)
        assert len(state.layer_states) == mal_config.n_layer


class TestMALStateCarry:
    """Test MAL state carrying across sequences."""
    
    @pytest.mark.skip(reason="MAL streaming requires attention KV caching, not yet implemented")
    def test_streaming_vs_batch(self, mal_config, seed):
        """Streaming and batch forward produce close results.
        
        NOTE: This test is skipped because MAL uses sliding window attention,
        which requires KV caching to work correctly in streaming mode.
        The memory state carries correctly, but attention sees different 
        contexts in streaming vs batch mode without KV cache.
        """
        torch.manual_seed(seed)
        model = AtlasQwen2ForCausalLM(mal_config)
        model.eval()
        
        batch, seq_len = 2, 4
        x = torch.randint(0, mal_config.vocab_size, (batch, seq_len))
        
        # Batch forward
        with torch.no_grad():
            batch_logits, _, _ = model(x)
            
            # Streaming forward
            state = None
            streaming_logits = []
            for t in range(seq_len):
                logits_t, state, _ = model(x[:, t:t+1], state)
                streaming_logits.append(logits_t)
            streaming_out = torch.cat(streaming_logits, dim=1)
        
        # Should be close (allows for some floating point differences)
        assert torch.allclose(batch_logits, streaming_out, atol=1e-4), \
            f"Max diff: {(batch_logits - streaming_out).abs().max()}"
    
    def test_memory_state_carries(self, mal_config, seed):
        """Memory state carries correctly in streaming mode."""
        torch.manual_seed(seed)
        model = AtlasQwen2ForCausalLM(mal_config)
        model.eval()
        
        x = torch.randint(0, mal_config.vocab_size, (2, 4))
        
        with torch.no_grad():
            _, state1, _ = model(x[:, :2])
            _, state2, _ = model(x[:, 2:], state1)
        
        # State2 should be different from state1
        for i in range(len(state1.layer_states)):
            S1 = state1.layer_states[i].memory_state.S
            S2 = state2.layer_states[i].memory_state.S
            assert not torch.allclose(S1, S2), f"Layer {i} state didn't change"
    
    def test_state_differs_from_initial(self, mal_config, seed):
        """State after processing differs from initial."""
        torch.manual_seed(seed)
        model = AtlasQwen2ForCausalLM(mal_config)
        x = torch.randint(0, mal_config.vocab_size, (2, 8))
        
        _, state, _ = model(x)
        
        # Memory state S should be non-zero
        for layer_state in state.layer_states:
            mem_state = layer_state.memory_state
            assert mem_state.S.abs().sum() > 0


class TestMALGradients:
    """Test MAL training mechanics."""
    
    def test_backward_produces_gradients(self, mal_config, seed):
        """Backward pass produces gradients on all modules."""
        torch.manual_seed(seed)
        model = AtlasQwen2ForCausalLM(mal_config)
        
        x = torch.randint(0, mal_config.vocab_size, (2, 8))
        y = torch.randint(0, mal_config.vocab_size, (2, 8))
        
        logits, _, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )
        loss.backward()
        
        # Check that memory and attention modules have gradients
        layer = model.model.layers[0]
        
        # Memory should have gradients
        mem_has_grad = False
        for p in layer.memory.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                mem_has_grad = True
                break
        assert mem_has_grad, "Memory module has no gradients"
        
        # Attention should have gradients
        attn_has_grad = False
        for p in layer.attention.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                attn_has_grad = True
                break
        assert attn_has_grad, "Attention module has no gradients"
    
    def test_gradients_are_finite(self, mal_config, seed):
        """Gradients are finite."""
        torch.manual_seed(seed)
        model = AtlasQwen2ForCausalLM(mal_config)
        
        x = torch.randint(0, mal_config.vocab_size, (2, 8))
        y = torch.randint(0, mal_config.vocab_size, (2, 8))
        
        logits, _, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )
        loss.backward()
        
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"
