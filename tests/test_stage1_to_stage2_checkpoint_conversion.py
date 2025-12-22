"""
Test Stage 1 (HF Qwen2 + Atlas patch) to Stage 2 (Pure Atlas-LMM) checkpoint conversion.

This test ensures that checkpoints saved from Stage 1 can be successfully loaded into Stage 2
models, regardless of hyperparameter settings.
"""

import pytest
import torch
import sys
import os
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import Atlas_Config
from models.atlasqwen2 import Model_atlasqwen2


def apply_stage1_remapping(state_dict):
    """
    Apply the remapping logic from src/lit.py for Stage 1 checkpoints.
    This should convert Stage 1 keys to Stage 2 compatible keys.
    """
    new_state_dict = {}
    
    for k in list(state_dict.keys()):
        # Skip teacher attention weights
        if '.teacher_attn.' in k:
            continue
        
        k_new = k
            
        # Remove student_attn wrapper (from AttentionDistillationWrapper)
        if '.student_attn.' in k_new:
            k_new = k_new.replace('.student_attn.', '.')
            
        # Convert HF patching format to Pure LMM format
        # self_attn.memory.* → memory.*
        if '.self_attn.memory.' in k_new:
            k_new = k_new.replace('.self_attn.memory.', '.memory.')
            
        # post_attention_layernorm → post_memory_layernorm
        if 'post_attention_layernorm' in k_new:
            k_new = k_new.replace('post_attention_layernorm', 'post_memory_layernorm')
            
        new_state_dict[k_new] = state_dict[k]
    
    return new_state_dict


# Test configurations with different hyperparameters
test_configs = [
    # Default hyperparameters
    {
        "name": "default",
        "poly_degree": 2,
        "poly_mode": "elementwise",
        "qk_norm": False,
        "qkv_conv_kernel": None,
        "use_rope": False,
        "use_groupnorm": False,
    },
    # Ablation: poly_degree=3
    {
        "name": "poly_degree_3",
        "poly_degree": 3,
        "poly_mode": "elementwise",
        "qk_norm": False,
        "qkv_conv_kernel": None,
        "use_rope": False,
        "use_groupnorm": False,
    },
    # Ablation: qk_norm=True
    {
        "name": "qk_norm_true",
        "poly_degree": 2,
        "poly_mode": "elementwise",
        "qk_norm": True,
        "qkv_conv_kernel": None,
        "use_rope": False,
        "use_groupnorm": False,
    },
    # Ablation: qkv_conv_kernel=4
    {
        "name": "qkv_conv_4",
        "poly_degree": 2,
        "poly_mode": "elementwise",
        "qk_norm": False,
        "qkv_conv_kernel": 4,
        "use_rope": False,
        "use_groupnorm": False,
    },
    # Ablation: use_rope=True
    {
        "name": "use_rope",
        "poly_degree": 2,
        "poly_mode": "elementwise",
        "qk_norm": False,
        "qkv_conv_kernel": None,
        "use_rope": True,
        "use_groupnorm": False,
    },
    # Ablation: use_groupnorm=True
    {
        "name": "use_groupnorm",
        "poly_degree": 2,
        "poly_mode": "elementwise",
        "qk_norm": False,
        "qkv_conv_kernel": None,
        "use_rope": False,
        "use_groupnorm": True,
    },
    # Ablation: All features enabled
    {
        "name": "all_features",
        "poly_degree": 3,
        "poly_mode": "elementwise",
        "qk_norm": True,
        "qkv_conv_kernel": 4,
        "use_rope": True,
        "use_groupnorm": True,
    },
]


@pytest.mark.parametrize("hyperparams", test_configs)
def test_stage1_to_stage2_checkpoint_conversion(hyperparams):
    """
    Test that Stage 1 checkpoint can be converted to Stage 2 format.
    
    Strategy:
    1. Create two Pure Atlas-LMM models (they're architecturally identical)
    2. Save one with Stage 1-style keys (.self_attn.memory.*)
    3. Load into the other with Stage 2-style keys (.memory.*)
    4. Verify all memory keys are successfully converted
    """
    config_name = hyperparams.pop("name")
    
    print(f"\n{'='*60}")
    print(f"Testing: {config_name}")
    print(f"Hyperparameters: {hyperparams}")
    print(f"{'='*60}")
    
    # Create config
    config = Atlas_Config(
        hf_path="Qwen/Qwen2.5-0.5B-Instruct",
        classname="atlasqwen2",
        vocab_size=151936,
        n_embd=896,
        n_layer=2,  # Use only 2 layers for faster testing
        dim_ffn=4864,
        dim_att=896,
        head_size=64,
        ctx_len=128,
        memory_heads=14,
        memory_dim_head=64,
        num_key_value_heads=14,
        use_momentum=True,
        use_omega_gate=True,
        omega_window=4,
        atlas_variant="lmm",
        tie_word_embeddings=True,
        **hyperparams
    )
    
    # Create a "Stage 1" model (actually Pure Atlas-LMM, but we'll rename its keys)
    print("\n[1] Creating source model (simulating Stage 1 checkpoint)...")
    stage1_model = Model_atlasqwen2(config)
    stage1_model.configure_model()  # Initialize model weights
    stage1_model.eval()
    
    # Get state dict and simulate Stage 1 key structure
    stage1_state_dict = stage1_model.state_dict()
    
    # Rename keys to simulate Stage 1 structure (.memory.* → .self_attn.student_attn.memory.*)
    stage1_sim_state_dict = {}
    for k, v in stage1_state_dict.items():
        if '.memory.' in k:
            # Convert Stage 2 keys to Stage 1 format (with .student_attn.)
            k_stage1 = k.replace('.memory.', '.self_attn.student_attn.memory.')
            stage1_sim_state_dict[k_stage1] = v
        elif 'post_memory_layernorm' in k:
            # Convert Stage 2 keys to Stage 1 format
            k_stage1 = k.replace('post_memory_layernorm', 'post_attention_layernorm')
            stage1_sim_state_dict[k_stage1] = v
        else:
            stage1_sim_state_dict[k] = v
    
    # Print some Stage 1-style keys
    print("\n[2] Stage 1-style keys (sample):")
    stage1_memory_keys = [k for k in stage1_sim_state_dict.keys() if '.self_attn.student_attn.memory.' in k]
    for k in stage1_memory_keys[:10]:
        print(f"  {k}")
    if len(stage1_memory_keys) > 10:
        print(f"  ... and {len(stage1_memory_keys) - 10} more")
    print(f"  Total Stage 1-style memory keys: {len(stage1_memory_keys)}")
    
    # Apply remapping (Stage 1 → Stage 2)
    print("\n[3] Applying remapping (Stage 1 → Stage 2)...")
    remapped_state_dict = apply_stage1_remapping(stage1_sim_state_dict)
    
    # Print some remapped keys
    print("\n[4] Remapped keys (sample):")
    remapped_memory_keys = [k for k in remapped_state_dict.keys() if '.memory.' in k and '.self_attn.memory.' not in k and '.student_attn.memory.' not in k]
    for k in remapped_memory_keys[:10]:
        print(f"  {k}")
    if len(remapped_memory_keys) > 10:
        print(f"  ... and {len(remapped_memory_keys) - 10} more")
    print(f"  Total remapped memory keys: {len(remapped_memory_keys)}")
    
    # Check for any keys that still have Stage 1 structure
    stage1_remaining_keys = [k for k in remapped_state_dict.keys() if '.student_attn.' in k or '.self_attn.memory.' in k]
    if len(stage1_remaining_keys) > 0:
        print(f"\n⚠️  WARNING: {len(stage1_remaining_keys)} keys still have Stage 1 structure:")
        for k in stage1_remaining_keys[:10]:
            print(f"  {k}")
        if len(stage1_remaining_keys) > 10:
            print(f"  ... and {len(stage1_remaining_keys) - 10} more")
    
    # Create Stage 2 model (Pure Atlas-LMM)
    print("\n[5] Creating target model (Stage 2)...")
    stage2_model = Model_atlasqwen2(config)
    stage2_model.configure_model()  # Initialize model weights
    stage2_model.eval()
    
    # Get Stage 2 expected keys
    stage2_state_dict = stage2_model.state_dict()
    
    # Print some Stage 2 keys
    print("\n[6] Stage 2 expected keys (sample):")
    stage2_memory_keys = [k for k in stage2_state_dict.keys() if '.memory.' in k]
    for k in stage2_memory_keys[:5]:
        print(f"  {k}")
    if len(stage2_memory_keys) > 5:
        print(f"  ... and {len(stage2_memory_keys) - 5} more")
    
    # Try to load remapped state dict into Stage 2 model
    print("\n[7] Loading remapped checkpoint into Stage 2 model...")
    
    try:
        result = stage2_model.load_state_dict(remapped_state_dict, strict=False)
        
        print(f"\n[8] Load result:")
        print(f"  Missing keys: {len(result.missing_keys)}")
        print(f"  Unexpected keys: {len(result.unexpected_keys)}")
        
        # Check for memory-related missing keys
        memory_missing_keys = [k for k in result.missing_keys if '.memory.' in k]
        memory_unexpected_keys = [k for k in result.unexpected_keys if '.memory.' in k or '.self_attn.memory.' in k]
        
        if len(memory_missing_keys) > 0:
            print(f"\n❌ FAILED: {len(memory_missing_keys)} memory-related keys are MISSING:")
            for k in memory_missing_keys[:20]:
                print(f"  - {k}")
            if len(memory_missing_keys) > 20:
                print(f"  ... and {len(memory_missing_keys) - 20} more")
            pytest.fail(f"Stage 1 to Stage 2 conversion failed: {len(memory_missing_keys)} memory keys missing")
        
        if len(memory_unexpected_keys) > 0:
            print(f"\n❌ FAILED: {len(memory_unexpected_keys)} memory-related keys are UNEXPECTED (not remapped properly):")
            for k in memory_unexpected_keys[:20]:
                print(f"  - {k}")
            if len(memory_unexpected_keys) > 20:
                print(f"  ... and {len(memory_unexpected_keys) - 20} more")
            pytest.fail(f"Stage 1 to Stage 2 conversion failed: {len(memory_unexpected_keys)} memory keys not remapped")
        
        print(f"\n✅ PASSED: All memory keys successfully converted for {config_name}!")
        
    except Exception as e:
        print(f"\n❌ FAILED with exception: {e}")
        pytest.fail(f"Stage 1 to Stage 2 conversion failed with exception: {e}")


if __name__ == "__main__":
    # Run all test cases for debugging
    for i, test_config in enumerate(test_configs):
        print(f"\n\n{'#'*70}")
        print(f"# Test {i+1}/{len(test_configs)}: {test_config['name']}")
        print(f"{'#'*70}")
        try:
            test_stage1_to_stage2_checkpoint_conversion(test_config.copy())
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            break

