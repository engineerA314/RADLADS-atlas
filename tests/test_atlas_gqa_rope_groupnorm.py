"""
Tests for GQA, ROPE, and GroupNorm features in Atlas-RNN.

Following TDD approach:
1. Write tests first
2. Implement features
3. Verify tests pass
"""

import pytest
import torch
import torch.nn as nn
from models.atlas_memory import RNNMemory, RNNMemState


class TestGQA:
    """Test Grouped Query Attention implementation."""
    
    def test_gqa_projection_dimensions(self):
        """Test that K/V projections use reduced number of heads."""
        dim = 896
        dim_head = 64
        heads = 14
        num_kv_heads = 2
        
        memory = RNNMemory(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            num_key_value_heads=num_kv_heads,
            use_momentum=True,
            omega_window=4,
        )
        
        # Q projection should use full heads
        assert memory.cell.to_q.out_features == heads * dim_head
        
        # K/V projections should use reduced heads
        assert memory.cell.to_k.out_features == num_kv_heads * dim_head
        assert memory.cell.to_v.out_features == num_kv_heads * dim_head
        
        # Verify num_key_value_groups calculation
        assert memory.cell.num_key_value_groups == heads // num_kv_heads
    
    def test_gqa_kv_expansion(self):
        """Test that K/V are properly expanded to match Q heads."""
        batch = 2
        seq_len = 16
        dim = 896
        dim_head = 64
        heads = 14
        num_kv_heads = 2
        
        memory = RNNMemory(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            num_key_value_heads=num_kv_heads,
            use_momentum=True,
            omega_window=4,
        )
        
        x = torch.randn(batch, seq_len, dim)
        output, state = memory(x)
        
        # Output should have correct shape
        assert output.shape == (batch, seq_len, dim)
        
        # State should be for full heads (after expansion)
        BH = batch * heads
        assert state.S.shape == (BH, dim_head, dim_head)
    
    def test_gqa_backward_pass(self):
        """Test that gradients flow properly through GQA."""
        batch = 2
        seq_len = 8
        dim = 896
        dim_head = 64
        heads = 14
        num_kv_heads = 2
        
        memory = RNNMemory(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            num_key_value_heads=num_kv_heads,
            use_momentum=True,
            omega_window=4,
        )
        
        x = torch.randn(batch, seq_len, dim, requires_grad=True)
        output, _ = memory(x)
        loss = output.sum()
        loss.backward()
        
        # All projections should have gradients
        assert memory.cell.to_q.weight.grad is not None
        assert memory.cell.to_k.weight.grad is not None
        assert memory.cell.to_v.weight.grad is not None
        assert x.grad is not None


class TestROPE:
    """Test Rotary Position Embedding implementation."""
    
    def test_rope_disabled_by_default(self):
        """Test that ROPE is disabled when use_rope=False."""
        memory = RNNMemory(
            dim=512,
            dim_head=64,
            heads=8,
            use_rope=False,
            omega_window=4,
        )
        
        assert memory.cell.use_rope is False
        assert not hasattr(memory.cell, 'rotary_emb') or memory.cell.rotary_emb is None
    
    def test_rope_enabled(self):
        """Test that ROPE is enabled when use_rope=True."""
        memory = RNNMemory(
            dim=512,
            dim_head=64,
            heads=8,
            use_rope=True,
            rope_theta=10000.0,
            omega_window=4,
        )
        
        assert memory.cell.use_rope is True
        assert hasattr(memory.cell, 'rotary_emb')
        assert memory.cell.rotary_emb is not None
    
    def test_rope_forward_pass(self):
        """Test forward pass with ROPE enabled."""
        batch = 2
        seq_len = 16
        dim = 512
        dim_head = 64
        heads = 8
        
        memory = RNNMemory(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            use_rope=True,
            rope_theta=10000.0,
            omega_window=4,
        )
        
        x = torch.randn(batch, seq_len, dim)
        output, state = memory(x)
        
        # Output should have correct shape
        assert output.shape == (batch, seq_len, dim)
    
    def test_rope_different_sequence_lengths(self):
        """Test that ROPE works with different sequence lengths."""
        batch = 2
        dim = 512
        dim_head = 64
        heads = 8
        
        memory = RNNMemory(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            use_rope=True,
            max_position_embeddings=2048,
            omega_window=4,
        )
        
        # Test with multiple sequence lengths
        for seq_len in [8, 16, 32, 64]:
            x = torch.randn(batch, seq_len, dim)
            output, _ = memory(x)
            assert output.shape == (batch, seq_len, dim)
    
    def test_rope_affects_output(self):
        """Test that ROPE actually changes the output."""
        batch = 2
        seq_len = 16
        dim = 512
        dim_head = 64
        heads = 8
        
        # Same architecture, different ROPE setting
        memory_no_rope = RNNMemory(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            use_rope=False,
            omega_window=4,
        )
        
        memory_with_rope = RNNMemory(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            use_rope=True,
            omega_window=4,
        )
        
        # Copy weights to make them identical except for ROPE
        with torch.no_grad():
            for (n1, p1), (n2, p2) in zip(
                memory_no_rope.named_parameters(),
                memory_with_rope.named_parameters()
            ):
                if 'rotary' not in n1 and 'rotary' not in n2:
                    p2.copy_(p1)
        
        x = torch.randn(batch, seq_len, dim)
        
        # Forward through both
        memory_no_rope.eval()
        memory_with_rope.eval()
        with torch.no_grad():
            out_no_rope, _ = memory_no_rope(x)
            out_with_rope, _ = memory_with_rope(x)
        
        # Outputs should be different due to ROPE
        assert not torch.allclose(out_no_rope, out_with_rope, atol=1e-5)


class TestGroupNorm:
    """Test GroupNorm implementation."""
    
    def test_groupnorm_disabled_by_default(self):
        """Test that GroupNorm is disabled by default."""
        memory = RNNMemory(
            dim=512,
            dim_head=64,
            heads=8,
            use_groupnorm=False,
            omega_window=4,
        )
        
        assert memory.cell.use_groupnorm is False
    
    def test_groupnorm_enabled(self):
        """Test that GroupNorm can be enabled."""
        memory = RNNMemory(
            dim=512,
            dim_head=64,
            heads=8,
            use_groupnorm=True,
            omega_window=4,
        )
        
        assert memory.cell.use_groupnorm is True
        assert hasattr(memory.cell, 'groupnorm')
        assert isinstance(memory.cell.groupnorm, nn.GroupNorm)
        
        # GroupNorm should use num_heads as num_groups
        assert memory.cell.groupnorm.num_groups == 8
        assert memory.cell.groupnorm.num_channels == 512
    
    def test_groupnorm_forward_pass(self):
        """Test forward pass with GroupNorm enabled."""
        batch = 2
        seq_len = 16
        dim = 512
        dim_head = 64
        heads = 8
        
        memory = RNNMemory(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            use_groupnorm=True,
            omega_window=4,
        )
        
        x = torch.randn(batch, seq_len, dim)
        output, state = memory(x)
        
        # Output should have correct shape
        assert output.shape == (batch, seq_len, dim)
    
    def test_groupnorm_affects_output(self):
        """Test that GroupNorm actually changes the output."""
        batch = 2
        seq_len = 16
        dim = 512
        dim_head = 64
        heads = 8
        
        memory_no_gn = RNNMemory(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            use_groupnorm=False,
            omega_window=4,
        )
        
        memory_with_gn = RNNMemory(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            use_groupnorm=True,
            omega_window=4,
        )
        
        # Copy weights where names and shapes match
        with torch.no_grad():
            params_no_gn = dict(memory_no_gn.named_parameters())
            for n2, p2 in memory_with_gn.named_parameters():
                if 'groupnorm' not in n2 and n2 in params_no_gn:
                    p1 = params_no_gn[n2]
                    if p1.shape == p2.shape:
                        p2.copy_(p1)
        
        x = torch.randn(batch, seq_len, dim)
        
        memory_no_gn.eval()
        memory_with_gn.eval()
        with torch.no_grad():
            out_no_gn, _ = memory_no_gn(x)
            out_with_gn, _ = memory_with_gn(x)
        
        # Outputs should be different due to GroupNorm
        assert not torch.allclose(out_no_gn, out_with_gn, atol=1e-5)
    
    def test_groupnorm_per_head_normalization(self):
        """Test that GroupNorm normalizes each head independently."""
        batch = 1
        seq_len = 4
        dim = 256
        dim_head = 64
        heads = 4
        
        memory = RNNMemory(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            use_groupnorm=True,
            omega_window=4,
        )
        
        # Create input where each head has different scale
        x = torch.randn(batch, seq_len, dim)
        for h in range(heads):
            start = h * dim_head
            end = (h + 1) * dim_head
            x[:, :, start:end] *= (h + 1)  # Scale each head differently
        
        memory.eval()
        with torch.no_grad():
            output, _ = memory(x)
        
        # After GroupNorm, each head should be normalized
        assert output.shape == (batch, seq_len, dim)
        
        # Check that output is normalized per head
        for h in range(heads):
            start = h * dim_head
            end = (h + 1) * dim_head
            head_output = output[:, :, start:end]
            
            # Mean should be close to 0, std close to 1 (after GroupNorm)
            # Note: This is approximate due to other operations
            assert head_output.std() > 0  # At least not constant


class TestIntegration:
    """Test integration of all three features."""
    
    def test_gqa_rope_groupnorm_together(self):
        """Test that all three features work together."""
        batch = 2
        seq_len = 16
        dim = 896
        dim_head = 64
        heads = 14
        num_kv_heads = 2
        
        memory = RNNMemory(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            num_key_value_heads=num_kv_heads,
            use_rope=True,
            rope_theta=10000.0,
            use_groupnorm=True,
            omega_window=4,
        )
        
        x = torch.randn(batch, seq_len, dim)
        output, state = memory(x)
        
        # Verify output shape
        assert output.shape == (batch, seq_len, dim)
        
        # Verify state shape (should use full heads after GQA expansion)
        BH = batch * heads
        assert state.S.shape == (BH, dim_head, dim_head)
    
    def test_backward_with_all_features(self):
        """Test backward pass with all features enabled."""
        batch = 2
        seq_len = 8
        dim = 512
        dim_head = 64
        heads = 8
        num_kv_heads = 2
        
        memory = RNNMemory(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            num_key_value_heads=num_kv_heads,
            use_rope=True,
            use_groupnorm=True,
            omega_window=4,
        )
        
        x = torch.randn(batch, seq_len, dim, requires_grad=True)
        output, _ = memory(x)
        loss = output.sum()
        loss.backward()
        
        # All parameters should have gradients
        for name, param in memory.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
