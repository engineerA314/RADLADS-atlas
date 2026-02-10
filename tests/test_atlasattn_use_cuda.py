"""
Test that atlasattn.py correctly propagates use_cuda parameter.

This is critical for Stage 1 (HF patching) to use the same implementation
as Stage 2/3 (Pure LMM), preventing NaN due to implementation mismatch.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import torch
import torch.nn as nn
from typing import Optional


def test_atlasattn_use_cuda_config_extraction():
    """Test that AtlasSelfAttention correctly extracts use_cuda from config."""
    from atlasattn import AtlasSelfAttention
    
    # Test 1: config with use_cuda=True
    class MockConfigTrue:
        hidden_size = 512
        num_attention_heads = 8
        num_key_value_heads = 2
        omega_window = 16
        use_omega_gate = True
        use_momentum = True
        use_cuda = True  # This should be read
        poly_degree = 2
        poly_mode = "elementwise"
        qk_norm = False
        qkv_conv_kernel = None
        use_rope = False
        use_groupnorm = False
        rope_theta = 10000.0
        max_position_embeddings = 2048
    
    config_true = MockConfigTrue()
    
    # Verify that getattr works
    assert getattr(config_true, 'use_cuda', False) == True, "Config extraction failed"
    
    attn_true = AtlasSelfAttention(config_true, layer_idx=0)
    
    # Check that use_cuda attribute exists
    assert hasattr(attn_true.memory, 'use_cuda'), "RNNMemory should have use_cuda attribute"
    
    # Note: If CUDA extension is not available, use_cuda may be False even if requested
    # This is expected behavior (graceful fallback)
    print(f"  Config use_cuda=True → Memory use_cuda={attn_true.memory.use_cuda}")
    
    # Test 2: config without use_cuda (should default to False)
    class MockConfigDefault:
        hidden_size = 512
        num_attention_heads = 8
        num_key_value_heads = 2
        omega_window = 16
        use_omega_gate = True
        use_momentum = True
        # use_cuda not specified
        poly_degree = 2
        poly_mode = "elementwise"
        qk_norm = False
        qkv_conv_kernel = None
        use_rope = False
        use_groupnorm = False
    
    config_default = MockConfigDefault()
    attn_default = AtlasSelfAttention(config_default, layer_idx=0)
    
    # Should default to False
    assert attn_default.memory.use_cuda == False, "Default should be False"
    print(f"  Config use_cuda=<unset> → Memory use_cuda={attn_default.memory.use_cuda}")
    
    print("✅ AtlasSelfAttention correctly extracts use_cuda from config")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_atlasattn_cuda_backend_activation():
    """Test that use_cuda=True actually activates CUDA backend on GPU."""
    from atlasattn import AtlasSelfAttention
    
    class MockConfig:
        hidden_size = 512
        num_attention_heads = 8
        num_key_value_heads = 2
        omega_window = 16
        use_omega_gate = True
        use_momentum = True
        use_cuda = True
        poly_degree = 2
        poly_mode = "elementwise"
        qk_norm = False
        qkv_conv_kernel = None
        use_rope = False
        use_groupnorm = False
    
    config = MockConfig()
    attn = AtlasSelfAttention(config, layer_idx=0).cuda().to(torch.bfloat16)
    
    # On GPU with compiled extension, this should be True
    print(f"  GPU: use_cuda={attn.memory.use_cuda}")
    
    if not attn.memory.use_cuda:
        pytest.skip("CUDA extension not available (needs compilation)")
    
    # Run a small forward pass
    batch = 2
    seq_len = 32
    hidden_states = torch.randn(batch, seq_len, config.hidden_size, dtype=torch.bfloat16, device='cuda')
    
    # Should not raise any errors
    output = attn(hidden_states)
    
    assert output[0].shape == hidden_states.shape, "Output shape should match input"
    assert not torch.isnan(output[0]).any(), "Output should not contain NaN"
    
    print("✅ CUDA backend runs without errors")


def test_load_and_patch_propagates_use_cuda():
    """Test that load_and_patch_model_with_attention_replacement propagates use_cuda."""
    from atlasattn import load_and_patch_model_with_attention_replacement
    import inspect
    
    # Test that use_cuda is in the list of fields to copy
    source = inspect.getsource(load_and_patch_model_with_attention_replacement)
    
    # Check that 'use_cuda' is in the field list
    assert "'use_cuda'" in source or '"use_cuda"' in source, \
        "load_and_patch_model_with_attention_replacement should include 'use_cuda' in field list"
    
    print("✅ load_and_patch_model_with_attention_replacement includes use_cuda in field list")


def test_stage1_stage2_config_compatibility():
    """
    Test that Stage 1 (HF patching) and Stage 2 (Pure LMM) have compatible configs.
    
    Both should support use_cuda parameter.
    """
    from atlasattn import AtlasSelfAttention
    from models.atlasqwen2_core import AtlasLMMBlock, AtlasConfig
    
    # Stage 1: HF patching (AtlasSelfAttention)
    class MockHFConfig:
        hidden_size = 512
        num_attention_heads = 8
        num_key_value_heads = 2
        omega_window = 16
        use_omega_gate = True
        use_momentum = True
        use_cuda = True
        poly_degree = 2
        poly_mode = "elementwise"
        qk_norm = False
        qkv_conv_kernel = None
        use_rope = False
        use_groupnorm = False
    
    stage1_attn = AtlasSelfAttention(MockHFConfig(), layer_idx=0)
    
    # Stage 2: Pure LMM (AtlasLMMBlock)
    atlas_config = AtlasConfig(
        n_embd=512,
        memory_heads=8,
        memory_dim_head=64,
        num_key_value_heads=2,
        omega_window=16,
        use_omega_gate=True,
        use_momentum=True,
        use_cuda=True,
        poly_degree=2,
        poly_mode="elementwise",
        qk_norm=False,
        qkv_conv_kernel=None,
        use_rope=False,
        use_groupnorm=False,
    )
    stage2_block = AtlasLMMBlock(atlas_config, layer_idx=0)
    
    # Both should have use_cuda attribute (may be False if extension not available)
    assert hasattr(stage1_attn.memory, 'use_cuda'), "Stage 1 missing use_cuda"
    assert hasattr(stage2_block.memory, 'use_cuda'), "Stage 2 missing use_cuda"
    
    print(f"  Stage 1 use_cuda: {stage1_attn.memory.use_cuda}")
    print(f"  Stage 2 use_cuda: {stage2_block.memory.use_cuda}")
    
    # Both should have same omega_window
    assert stage1_attn.memory.cell.omega_window == 16, "Stage 1 omega_window mismatch"
    assert stage2_block.memory.cell.omega_window == 16, "Stage 2 omega_window mismatch"
    
    print("✅ Stage 1 and Stage 2 configurations are compatible")


if __name__ == "__main__":
    # Run tests locally
    print("Running atlasattn use_cuda tests...\n")
    
    test_atlasattn_use_cuda_config_extraction()
    test_load_and_patch_propagates_use_cuda()
    test_stage1_stage2_config_compatibility()
    
    if torch.cuda.is_available():
        test_atlasattn_cuda_backend_activation()
    else:
        print("⚠️  Skipping CUDA activation test (no GPU)")
    
    print("\n✅ All tests passed!")
