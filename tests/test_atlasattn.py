"""
TDD Tests for atlasattn.py - Atlas Attention for RADLADS Stage 1 Distillation.

These tests verify:
- AtlasSelfAttention: HuggingFace attention interface compatibility
- AttentionDistillationWrapper: Teacher-student parallel execution
- Alignment loss computation and gradient flow
"""

import pytest
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from atlasattn import (
    AtlasSelfAttention,
    AttentionDistillationWrapper,
    ATLAS_ATTENTION_CLASSES,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@dataclass
class MockConfig:
    """Mock HuggingFace model config for testing."""
    hidden_size: int = 64
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    intermediate_size: int = 128
    rms_norm_eps: float = 1e-6
    vocab_size: int = 256
    attention_bias: bool = True
    attention_output_bias: bool = False


class MockSoftmaxAttention(nn.Module):
    """Mock softmax attention for testing AttentionDistillationWrapper."""
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Simple linear projection (no actual attention)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
    
    def forward(self, hidden_states, *args, **kwargs):
        output = self.proj(hidden_states)
        return (output, None, None)


@pytest.fixture
def mock_config():
    """Create mock config for testing."""
    return MockConfig()


@pytest.fixture
def tiny_config():
    """Create tiny config for fast testing."""
    return MockConfig(
        hidden_size=32,
        num_attention_heads=2,
        intermediate_size=64,
    )


@pytest.fixture
def sample_input(mock_config):
    """Create sample input tensor."""
    batch_size = 2
    seq_len = 8
    return torch.randn(batch_size, seq_len, mock_config.hidden_size)


# =============================================================================
# AtlasSelfAttention Tests
# =============================================================================

class TestAtlasSelfAttention:
    """Tests for AtlasSelfAttention class."""
    
    def test_init(self, mock_config):
        """Test AtlasSelfAttention initialization."""
        attn = AtlasSelfAttention(mock_config, layer_idx=0)
        
        assert attn.layer_idx == 0
        assert attn.hidden_size == mock_config.hidden_size
        assert attn.num_heads == mock_config.num_attention_heads
        assert attn.head_dim == mock_config.hidden_size // mock_config.num_attention_heads
        assert attn.memory is not None
        assert attn._memory_state is None
    
    def test_forward_shape(self, mock_config, sample_input):
        """Test forward pass output shape."""
        attn = AtlasSelfAttention(mock_config, layer_idx=0)
        
        output, attn_weights, past_kv = attn(sample_input)
        
        assert output.shape == sample_input.shape
        assert attn_weights is None
        assert past_kv is None
    
    def test_forward_hf_interface(self, mock_config, sample_input):
        """Test forward pass matches HuggingFace attention interface."""
        attn = AtlasSelfAttention(mock_config, layer_idx=0)
        
        # HuggingFace attention call with all optional args
        output, attn_weights, past_kv = attn(
            hidden_states=sample_input,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=None,
        )
        
        assert output.shape == sample_input.shape
        assert attn_weights is None
        assert past_kv is None
    
    def test_forward_training_mode(self, mock_config, sample_input):
        """Test forward in training mode creates fresh state each time."""
        attn = AtlasSelfAttention(mock_config, layer_idx=0)
        attn.train()
        
        # First forward
        output1, _, _ = attn(sample_input)
        state1 = attn._memory_state
        
        # Second forward (should not reuse state in training)
        output2, _, _ = attn(sample_input)
        
        # In training mode, outputs should be same (fresh state each time)
        assert torch.allclose(output1, output2, atol=1e-5)
    
    def test_forward_inference_use_cache(self, mock_config):
        """Test forward in inference mode with use_cache."""
        attn = AtlasSelfAttention(mock_config, layer_idx=0)
        attn.eval()
        
        batch_size = 1
        hidden_size = mock_config.hidden_size
        
        # First forward with full sequence
        x1 = torch.randn(batch_size, 4, hidden_size)
        output1, _, _ = attn(x1, use_cache=True)
        
        # State should be stored
        assert attn._memory_state is not None
        
        # Second forward with new token
        x2 = torch.randn(batch_size, 1, hidden_size)
        output2, _, _ = attn(x2, use_cache=True)
        
        # Output should have correct shape
        assert output2.shape == x2.shape
    
    def test_gradient_flow(self, mock_config, sample_input):
        """Test gradients flow through AtlasSelfAttention."""
        attn = AtlasSelfAttention(mock_config, layer_idx=0)
        attn.train()
        
        sample_input.requires_grad_(True)
        output, _, _ = attn(sample_input)
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert sample_input.grad is not None
        assert not torch.all(sample_input.grad == 0)
    
    def test_different_sequence_lengths(self, mock_config):
        """Test with different sequence lengths."""
        attn = AtlasSelfAttention(mock_config, layer_idx=0)
        
        for seq_len in [1, 4, 16, 32]:
            x = torch.randn(2, seq_len, mock_config.hidden_size)
            output, _, _ = attn(x)
            assert output.shape == x.shape
    
    def test_multiple_layers(self, mock_config, sample_input):
        """Test multiple AtlasSelfAttention layers with different layer_idx."""
        layers = [AtlasSelfAttention(mock_config, layer_idx=i) for i in range(4)]
        
        x = sample_input
        for i, layer in enumerate(layers):
            output, _, _ = layer(x)
            assert output.shape == x.shape
            x = output


# =============================================================================
# AttentionDistillationWrapper Tests
# =============================================================================

class TestAttentionDistillationWrapper:
    """Tests for AttentionDistillationWrapper class."""
    
    def test_init(self, mock_config):
        """Test AttentionDistillationWrapper initialization."""
        teacher = MockSoftmaxAttention(mock_config, layer_idx=0)
        
        wrapper = AttentionDistillationWrapper(
            original_self_attn=teacher,
            ReplacementSelfAttentionType=AtlasSelfAttention,
            model_config=mock_config,
            attention_distillation_stage=1,
        )
        
        assert wrapper.teacher_attn is teacher
        assert isinstance(wrapper.student_attn, AtlasSelfAttention)
        assert wrapper.attention_distillation_stage == 1
        assert wrapper._last_alignment_loss is None
    
    def test_init_stage_assertion(self, mock_config):
        """Test that only stage 1 is supported."""
        teacher = MockSoftmaxAttention(mock_config, layer_idx=0)
        
        with pytest.raises(AssertionError):
            AttentionDistillationWrapper(
                original_self_attn=teacher,
                ReplacementSelfAttentionType=AtlasSelfAttention,
                model_config=mock_config,
                attention_distillation_stage=2,  # Should fail
            )
    
    def test_forward_output_shape(self, mock_config, sample_input):
        """Test forward pass output shape."""
        teacher = MockSoftmaxAttention(mock_config, layer_idx=0)
        wrapper = AttentionDistillationWrapper(
            original_self_attn=teacher,
            ReplacementSelfAttentionType=AtlasSelfAttention,
            model_config=mock_config,
            attention_distillation_stage=1,
        )
        
        output, alignment_loss = wrapper(sample_input)
        
        assert output.shape == sample_input.shape
        assert alignment_loss.dim() == 0  # Scalar
    
    def test_forward_returns_student_output(self, mock_config, sample_input):
        """Test that forward returns student output (not teacher)."""
        teacher = MockSoftmaxAttention(mock_config, layer_idx=0)
        wrapper = AttentionDistillationWrapper(
            original_self_attn=teacher,
            ReplacementSelfAttentionType=AtlasSelfAttention,
            model_config=mock_config,
            attention_distillation_stage=1,
        )
        
        # Get wrapper output
        wrapper_output, _ = wrapper(sample_input)
        
        # Get student output directly
        student_output, _, _ = wrapper.student_attn(sample_input)
        
        # They should match
        assert torch.allclose(wrapper_output, student_output, atol=1e-5)
    
    def test_alignment_loss_computation(self, mock_config, sample_input):
        """Test alignment loss is computed correctly."""
        teacher = MockSoftmaxAttention(mock_config, layer_idx=0)
        wrapper = AttentionDistillationWrapper(
            original_self_attn=teacher,
            ReplacementSelfAttentionType=AtlasSelfAttention,
            model_config=mock_config,
            attention_distillation_stage=1,
        )
        
        output, alignment_loss = wrapper(sample_input)
        
        # Alignment loss should be positive
        assert alignment_loss.item() > 0
        
        # Should be stored
        assert wrapper._last_alignment_loss is not None
        assert wrapper._last_alignment_loss.item() == alignment_loss.item()
    
    def test_alignment_loss_is_scalar(self, mock_config, sample_input):
        """Test alignment loss is a scalar."""
        teacher = MockSoftmaxAttention(mock_config, layer_idx=0)
        wrapper = AttentionDistillationWrapper(
            original_self_attn=teacher,
            ReplacementSelfAttentionType=AtlasSelfAttention,
            model_config=mock_config,
            attention_distillation_stage=1,
        )
        
        _, alignment_loss = wrapper(sample_input)
        
        assert alignment_loss.dim() == 0
        assert alignment_loss.numel() == 1
    
    def test_teacher_no_gradient(self, mock_config, sample_input):
        """Test teacher attention doesn't compute gradients."""
        teacher = MockSoftmaxAttention(mock_config, layer_idx=0)
        wrapper = AttentionDistillationWrapper(
            original_self_attn=teacher,
            ReplacementSelfAttentionType=AtlasSelfAttention,
            model_config=mock_config,
            attention_distillation_stage=1,
        )
        
        sample_input.requires_grad_(True)
        output, alignment_loss = wrapper(sample_input)
        
        # Backward through alignment loss
        alignment_loss.backward()
        
        # Teacher parameters should not have gradients
        for p in wrapper.teacher_attn.parameters():
            assert p.grad is None or torch.all(p.grad == 0)
    
    def test_student_has_gradient(self, mock_config, sample_input):
        """Test student attention computes gradients."""
        teacher = MockSoftmaxAttention(mock_config, layer_idx=0)
        wrapper = AttentionDistillationWrapper(
            original_self_attn=teacher,
            ReplacementSelfAttentionType=AtlasSelfAttention,
            model_config=mock_config,
            attention_distillation_stage=1,
        )
        
        output, alignment_loss = wrapper(sample_input)
        
        # Backward through alignment loss
        alignment_loss.backward()
        
        # Student parameters should have gradients
        has_grad = False
        for p in wrapper.student_attn.parameters():
            if p.grad is not None and not torch.all(p.grad == 0):
                has_grad = True
                break
        
        assert has_grad, "Student should have gradients"
    
    def test_multiple_forward_backward(self, mock_config):
        """Test multiple forward/backward passes don't cause graph issues."""
        teacher = MockSoftmaxAttention(mock_config, layer_idx=0)
        wrapper = AttentionDistillationWrapper(
            original_self_attn=teacher,
            ReplacementSelfAttentionType=AtlasSelfAttention,
            model_config=mock_config,
            attention_distillation_stage=1,
        )
        wrapper.train()
        
        optimizer = torch.optim.Adam(wrapper.student_attn.parameters(), lr=0.001)
        
        for _ in range(3):
            x = torch.randn(2, 8, mock_config.hidden_size)
            
            optimizer.zero_grad()
            output, alignment_loss = wrapper(x)
            alignment_loss.backward()
            optimizer.step()
        
        # Should complete without errors


# =============================================================================
# ATLAS_ATTENTION_CLASSES Tests
# =============================================================================

class TestAtlasAttentionClasses:
    """Tests for ATLAS_ATTENTION_CLASSES dict."""
    
    def test_all_keys_exist(self):
        """Test all expected keys exist."""
        expected_keys = ["eager", "flash_attention_2", "sdpa"]
        for key in expected_keys:
            assert key in ATLAS_ATTENTION_CLASSES
    
    def test_all_values_are_atlas_self_attention(self):
        """Test all values are AtlasSelfAttention class."""
        for key, cls in ATLAS_ATTENTION_CLASSES.items():
            assert cls is AtlasSelfAttention


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for atlasattn module."""
    
    def test_stacked_wrappers(self, mock_config):
        """Test multiple wrappers in sequence (simulating multiple layers)."""
        num_layers = 4
        wrappers = []
        
        for i in range(num_layers):
            teacher = MockSoftmaxAttention(mock_config, layer_idx=i)
            wrapper = AttentionDistillationWrapper(
                original_self_attn=teacher,
                ReplacementSelfAttentionType=AtlasSelfAttention,
                model_config=mock_config,
                attention_distillation_stage=1,
            )
            wrappers.append(wrapper)
        
        # Forward through all layers
        x = torch.randn(2, 8, mock_config.hidden_size)
        total_loss = 0.0
        
        for wrapper in wrappers:
            x, loss = wrapper(x)
            total_loss = total_loss + loss
        
        # Should be able to backward through all
        total_loss.backward()
    
    def test_collect_alignment_losses(self, mock_config):
        """Test collecting alignment losses from multiple layers."""
        num_layers = 4
        wrappers = []
        
        for i in range(num_layers):
            teacher = MockSoftmaxAttention(mock_config, layer_idx=i)
            wrapper = AttentionDistillationWrapper(
                original_self_attn=teacher,
                ReplacementSelfAttentionType=AtlasSelfAttention,
                model_config=mock_config,
                attention_distillation_stage=1,
            )
            wrappers.append(wrapper)
        
        # Forward through all layers
        x = torch.randn(2, 8, mock_config.hidden_size)
        for wrapper in wrappers:
            x, _ = wrapper(x)
        
        # Collect losses
        losses = [w._last_alignment_loss for w in wrappers]
        
        assert len(losses) == num_layers
        assert all(loss is not None for loss in losses)
        assert all(loss.item() > 0 for loss in losses)
        
        # Stack and mean (how training loop uses them)
        mean_loss = torch.stack(losses).mean()
        assert mean_loss.dim() == 0
    
    def test_training_simulation(self, mock_config):
        """Simulate a mini training loop."""
        # Create "model" with multiple layers
        layers = nn.ModuleList()
        for i in range(3):
            teacher = MockSoftmaxAttention(mock_config, layer_idx=i)
            wrapper = AttentionDistillationWrapper(
                original_self_attn=teacher,
                ReplacementSelfAttentionType=AtlasSelfAttention,
                model_config=mock_config,
                attention_distillation_stage=1,
            )
            layers.append(wrapper)
        
        # Get all student parameters
        student_params = []
        for layer in layers:
            student_params.extend(layer.student_attn.parameters())
        
        optimizer = torch.optim.Adam(student_params, lr=0.001)
        
        # Training loop
        initial_loss = None
        for step in range(5):
            x = torch.randn(2, 8, mock_config.hidden_size)
            
            optimizer.zero_grad()
            
            # Forward through all layers
            for layer in layers:
                x, _ = layer(x)
            
            # Collect and average losses
            losses = [layer._last_alignment_loss for layer in layers]
            loss = torch.stack(losses).mean()
            
            if initial_loss is None:
                initial_loss = loss.item()
            
            loss.backward()
            optimizer.step()
        
        # Loss should have changed (not necessarily decreased with random data)
        final_loss = loss.item()
        assert initial_loss != final_loss or step == 0


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_batch_size_one(self, mock_config):
        """Test with batch size 1."""
        attn = AtlasSelfAttention(mock_config, layer_idx=0)
        x = torch.randn(1, 8, mock_config.hidden_size)
        output, _, _ = attn(x)
        assert output.shape == x.shape
    
    def test_sequence_length_one(self, mock_config):
        """Test with sequence length 1."""
        attn = AtlasSelfAttention(mock_config, layer_idx=0)
        x = torch.randn(2, 1, mock_config.hidden_size)
        output, _, _ = attn(x)
        assert output.shape == x.shape
    
    def test_large_batch(self, tiny_config):
        """Test with large batch size."""
        attn = AtlasSelfAttention(tiny_config, layer_idx=0)
        x = torch.randn(64, 4, tiny_config.hidden_size)
        output, _, _ = attn(x)
        assert output.shape == x.shape
    
    def test_long_sequence(self, tiny_config):
        """Test with longer sequence."""
        attn = AtlasSelfAttention(tiny_config, layer_idx=0)
        x = torch.randn(2, 128, tiny_config.hidden_size)
        output, _, _ = attn(x)
        assert output.shape == x.shape
    
    def test_bfloat16(self, tiny_config):
        """Test with bfloat16 precision."""
        attn = AtlasSelfAttention(tiny_config, layer_idx=0).to(torch.bfloat16)
        x = torch.randn(2, 8, tiny_config.hidden_size, dtype=torch.bfloat16)
        output, _, _ = attn(x)
        assert output.shape == x.shape
        assert output.dtype == torch.bfloat16
    
    def test_float16(self, tiny_config):
        """Test with float16 precision."""
        attn = AtlasSelfAttention(tiny_config, layer_idx=0).to(torch.float16)
        x = torch.randn(2, 8, tiny_config.hidden_size, dtype=torch.float16)
        output, _, _ = attn(x)
        assert output.shape == x.shape
        assert output.dtype == torch.float16
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda(self, tiny_config):
        """Test on CUDA device."""
        device = torch.device("cuda")
        attn = AtlasSelfAttention(tiny_config, layer_idx=0).to(device)
        x = torch.randn(2, 8, tiny_config.hidden_size, device=device)
        output, _, _ = attn(x)
        assert output.shape == x.shape
        assert output.device.type == device.type  # Compare device type, not exact device object
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_wrapper_cuda(self, tiny_config):
        """Test wrapper on CUDA device."""
        device = torch.device("cuda")
        teacher = MockSoftmaxAttention(tiny_config, layer_idx=0).to(device)
        wrapper = AttentionDistillationWrapper(
            original_self_attn=teacher,
            ReplacementSelfAttentionType=AtlasSelfAttention,
            model_config=tiny_config,
            attention_distillation_stage=1,
        ).to(device)
        
        x = torch.randn(2, 8, tiny_config.hidden_size, device=device)
        output, loss = wrapper(x)
        
        assert output.shape == x.shape
        assert output.device.type == device.type  # Compare device type, not exact device object
        assert loss.device.type == device.type

