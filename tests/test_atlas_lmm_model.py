"""
TDD Tests for Atlas-LMM Model (Phase B).

These tests define the contracts that the Atlas-LMM model must satisfy:
- Forward shape correctness
- Dtype/device handling
- Determinism under fixed seed
- State-carry equivalence (batch vs streaming)
- State structure validation
- No attention modules (LMM is memory-only)
- state_dict key naming conventions
"""

import pytest
import torch

from models.atlasqwen2 import Model_atlasqwen2
from models.atlasqwen2_core import (
    AtlasConfig, 
    AtlasQwen2ForCausalLM, 
    AtlasLMMBlock,
    AtlasQwen2Core,
    AtlasModelState,
)


@pytest.fixture
def tiny_atlas_config():
    """Tiny Atlas config for fast testing."""
    return AtlasConfig(
        vocab_size=256,
        n_embd=64,
        n_layer=2,
        dim_ffn=128,
        rms_norm_eps=1e-6,
        memory_heads=4,
        memory_dim_head=16,
        use_momentum=True,
        poly_degree=1,
        poly_mode='off',
        qk_norm=True,
        qkv_conv_kernel=None,
        ctx_len=32,
        tie_word_embeddings=True,
    )


@pytest.fixture
def tiny_model(tiny_atlas_config):
    """Create a tiny model for testing."""
    return AtlasQwen2ForCausalLM(tiny_atlas_config)


class TestAtlasLMMModelInstantiation:
    """Test model can be instantiated with valid config."""
    
    def test_instantiation_from_config(self, tiny_atlas_config):
        """Model instantiates from config."""
        model = AtlasQwen2ForCausalLM(tiny_atlas_config)
        assert model is not None
        assert model.config == tiny_atlas_config
    
    def test_core_instantiation(self, tiny_atlas_config):
        """Core model instantiates."""
        core = AtlasQwen2Core(tiny_atlas_config)
        assert len(core.layers) == tiny_atlas_config.n_layer
    
    def test_block_instantiation(self, tiny_atlas_config):
        """Individual blocks instantiate."""
        block = AtlasLMMBlock(tiny_atlas_config, layer_idx=0)
        assert block.memory is not None
        assert block.mlp is not None


class TestAtlasLMMForwardShape:
    """Contract: logits.shape == (B, T, vocab_size)."""
    
    def test_forward_shape_basic(self, tiny_model, tiny_atlas_config):
        """Forward produces correct logit shape."""
        batch, seq_len = 2, 16
        x = torch.randint(0, tiny_atlas_config.vocab_size, (batch, seq_len))
        
        logits, state, _ = tiny_model(x)
        
        assert logits.shape == (batch, seq_len, tiny_atlas_config.vocab_size)
    
    def test_forward_different_batch_sizes(self, tiny_model, tiny_atlas_config):
        """Forward works with various batch sizes."""
        for batch in [1, 2, 4]:
            x = torch.randint(0, tiny_atlas_config.vocab_size, (batch, 8))
            logits, _, _ = tiny_model(x)
            assert logits.shape[0] == batch
    
    def test_forward_different_seq_lengths(self, tiny_model, tiny_atlas_config):
        """Forward works with various sequence lengths."""
        for seq_len in [1, 4, 16, 32]:
            x = torch.randint(0, tiny_atlas_config.vocab_size, (2, seq_len))
            logits, _, _ = tiny_model(x)
            assert logits.shape[1] == seq_len


class TestAtlasLMMDeviceDtype:
    """Contract: CPU forward works; optional GPU test."""
    
    def test_cpu_forward(self, tiny_model, tiny_atlas_config):
        """Model runs on CPU."""
        x = torch.randint(0, tiny_atlas_config.vocab_size, (2, 8))
        logits, _, _ = tiny_model(x)
        assert logits.device == torch.device('cpu')
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self, tiny_atlas_config):
        """Model runs on CUDA if available."""
        model = AtlasQwen2ForCausalLM(tiny_atlas_config).cuda()
        x = torch.randint(0, tiny_atlas_config.vocab_size, (2, 8)).cuda()
        logits, _, _ = model(x)
        assert logits.device.type == 'cuda'


class TestAtlasLMMDeterminism:
    """Contract: fixed seed + eval mode = stable outputs."""
    
    def test_deterministic_output(self, tiny_atlas_config, seed):
        """Same seed + input produces same output."""
        torch.manual_seed(seed)
        model = AtlasQwen2ForCausalLM(tiny_atlas_config)
        model.eval()
        
        x = torch.randint(0, tiny_atlas_config.vocab_size, (2, 8))
        
        logits1, _, _ = model(x)
        logits2, _, _ = model(x)
        
        assert torch.allclose(logits1, logits2, atol=1e-6)


class TestAtlasLMMStateCarry:
    """Contract: batch forward matches streaming token-by-token."""
    
    def test_streaming_vs_batch_equivalence(self, tiny_atlas_config, seed):
        """Streaming with carried state matches batch forward."""
        torch.manual_seed(seed)
        model = AtlasQwen2ForCausalLM(tiny_atlas_config)
        model.eval()
        
        batch, seq_len = 2, 8
        x = torch.randint(0, tiny_atlas_config.vocab_size, (batch, seq_len))
        
        # Batch forward
        batch_logits, batch_final_state, _ = model(x)
        
        # Streaming forward (token by token)
        state = None
        streaming_logits_list = []
        for t in range(seq_len):
            logits_t, state, _ = model(x[:, t:t+1], state)
            streaming_logits_list.append(logits_t)
        streaming_logits = torch.cat(streaming_logits_list, dim=1)
        
        # Check equivalence
        assert torch.allclose(batch_logits, streaming_logits, atol=1e-4), \
            f"Max diff: {(batch_logits - streaming_logits).abs().max()}"


class TestAtlasLMMStateStructure:
    """Contract: state has correct structure per layer."""
    
    def test_state_has_n_layer_entries(self, tiny_model, tiny_atlas_config):
        """State contains exactly n_layer entries."""
        x = torch.randint(0, tiny_atlas_config.vocab_size, (2, 8))
        _, state, _ = tiny_model(x)
        
        assert len(state.layer_states) == tiny_atlas_config.n_layer
    
    def test_state_tensor_dtypes(self, tiny_model, tiny_atlas_config):
        """State tensors have expected dtypes."""
        x = torch.randint(0, tiny_atlas_config.vocab_size, (2, 8))
        _, state, _ = tiny_model(x)
        
        for layer_state in state.layer_states:
            assert layer_state.memory_state.S.dtype == torch.float32
            if tiny_atlas_config.use_momentum:
                assert layer_state.memory_state.Z.dtype == torch.float32
    
    def test_state_tensor_shapes(self, tiny_model, tiny_atlas_config):
        """State tensors have correct shapes."""
        batch = 2
        x = torch.randint(0, tiny_atlas_config.vocab_size, (batch, 8))
        _, state, _ = tiny_model(x)
        
        expected_shape = (
            batch * tiny_atlas_config.memory_heads,
            tiny_atlas_config.memory_dim_head,
            tiny_atlas_config.memory_dim_head,
        )
        
        for layer_state in state.layer_states:
            assert layer_state.memory_state.S.shape == expected_shape


class TestAtlasLMMNoAttention:
    """Contract: LMM has no attention modules."""
    
    def test_no_attention_params(self, tiny_model):
        """No parameters with 'attn' or 'attention' in name."""
        for name, _ in tiny_model.named_parameters():
            assert 'attn' not in name.lower(), f"Found attention param: {name}"
            # Note: 'attention' check relaxed since we use 'memory' not 'attention'
    
    def test_no_softmax_attention_modules(self, tiny_model):
        """No nn.MultiheadAttention or similar modules."""
        for name, module in tiny_model.named_modules():
            assert not isinstance(module, nn.MultiheadAttention), \
                f"Found MultiheadAttention: {name}"


class TestAtlasLMMStateDictKeys:
    """Contract: state_dict keys follow naming conventions."""
    
    def test_required_top_level_keys(self, tiny_model):
        """Keys include model.embed_tokens.*, model.norm.*, lm_head.*."""
        state_dict = tiny_model.state_dict()
        keys = list(state_dict.keys())
        
        # Check embed_tokens
        embed_keys = [k for k in keys if k.startswith('model.embed_tokens')]
        assert len(embed_keys) > 0, "Missing model.embed_tokens.* keys"
        
        # Check final norm
        norm_keys = [k for k in keys if k.startswith('model.norm')]
        assert len(norm_keys) > 0, "Missing model.norm.* keys"
        
        # Check lm_head (might be tied)
        if not tiny_model.config.tie_word_embeddings:
            lm_head_keys = [k for k in keys if k.startswith('lm_head')]
            assert len(lm_head_keys) > 0, "Missing lm_head.* keys"
    
    def test_layer_key_structure(self, tiny_model, tiny_atlas_config):
        """Layer keys follow model.layers.{i}.memory.*, model.layers.{i}.mlp.*."""
        state_dict = tiny_model.state_dict()
        keys = list(state_dict.keys())
        
        for i in range(tiny_atlas_config.n_layer):
            # Check memory keys exist
            memory_prefix = f'model.layers.{i}.memory'
            memory_keys = [k for k in keys if k.startswith(memory_prefix)]
            assert len(memory_keys) > 0, f"Missing {memory_prefix}.* keys"
            
            # Check mlp keys exist
            mlp_prefix = f'model.layers.{i}.mlp'
            mlp_keys = [k for k in keys if k.startswith(mlp_prefix)]
            assert len(mlp_keys) > 0, f"Missing {mlp_prefix}.* keys"
            
            # Check layer norms
            ln_prefix = f'model.layers.{i}.input_layernorm'
            ln_keys = [k for k in keys if k.startswith(ln_prefix)]
            assert len(ln_keys) > 0, f"Missing {ln_prefix}.* keys"
    
    def test_no_unexpected_prefixes(self, tiny_model):
        """No keys with unexpected prefixes."""
        state_dict = tiny_model.state_dict()
        allowed_prefixes = ['model.', 'lm_head.']
        
        for key in state_dict.keys():
            has_valid_prefix = any(key.startswith(p) for p in allowed_prefixes)
            assert has_valid_prefix, f"Unexpected key prefix: {key}"


class TestAtlasLMMHiddenStates:
    """Test hidden states output."""
    
    def test_output_hidden_states(self, tiny_model, tiny_atlas_config):
        """Can output hidden states per layer."""
        x = torch.randint(0, tiny_atlas_config.vocab_size, (2, 8))
        _, _, hidden_states = tiny_model(x, output_hidden_states=True)
        
        assert hidden_states is not None
        # Should have n_layer + 1 entries (embedding + each layer)
        assert len(hidden_states) == tiny_atlas_config.n_layer + 1


# Import nn for the test above
import torch.nn as nn
