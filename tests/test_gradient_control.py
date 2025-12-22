"""
Test gradient control (freeze/unfreeze) for ablation study experiments.

Stage mapping (Paper vs Code):
- Paper Stage 1 (Attention Alignment): attention_distillation_stage=1
  - Only Atlas memory parameters are trainable
  - Other layers (embeddings, MLP, lm_head) are frozen
  - No separate teacher model needed (parallel teacher/student in model)
  
- Paper Stage 2/3 (KL Divergence / Long Context): attention_distillation_stage != 1
  - All parameters are trainable
  - Stage 2: Uses separate teacher model for KL loss
  - Stage 3: Uses CE loss only (no teacher)

This test module ensures that:
1. Paper Stage 1 (attention_distillation_stage=1): Only Atlas memory parameters are trainable
2. Omega gate parameters are trainable when use_omega_gate=True
3. Conv parameters are trainable when qkv_conv_kernel is set
4. Other layers (embeddings, MLP, lm_head) are frozen in Stage 1
5. Full training mode (attention_distillation_stage != 1): All parameters are trainable
"""

import torch
import torch.nn as nn
from models.atlasqwen2 import Model_atlasqwen2, config_to_atlas_config


def create_test_config(
    poly_degree=2,
    use_omega_gate=True,
    omega_window=16,
    qkv_conv_kernel=None,
    use_rope=False,
    use_groupnorm=False,
    attention_distillation_stage=1,
):
    """Create a test config with specified ablation parameters."""
    from dataclasses import dataclass, field
    from typing import Optional
    
    # Create config objects with values
    @dataclass
    class Config:
        pass
    
    config = Config()
    config.model = Config()
    config.train = Config()
    
    # Model config
    config.model.classname = 'atlasqwen2'
    config.model.vocab_size = 256
    config.model.n_embd = 128
    config.model.n_layer = 2
    config.model.dim_ffn = 256
    config.model.ctx_len = 64
    config.model.head_size = 64
    config.model.memory_heads = 2
    config.model.memory_dim_head = 64
    config.model.use_momentum = True
    config.model.use_omega_gate = use_omega_gate
    config.model.omega_window = omega_window
    config.model.poly_degree = poly_degree
    config.model.poly_mode = 'elementwise'
    config.model.qk_norm = True
    config.model.qkv_conv_kernel = qkv_conv_kernel
    config.model.use_rope = use_rope
    config.model.use_groupnorm = use_groupnorm
    config.model.tie_word_embeddings = True
    config.model.rms_norm_eps = 1e-6
    config.model.vocab_padding_idx = None
    
    # Train config
    config.train.attention_distillation_stage = attention_distillation_stage
    config.train.train_stage = 1
    config.train.weight_decay = 0.1
    
    return config


def count_trainable_params(model):
    """Count number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_trainable_param_names(model):
    """Get names of all trainable parameters."""
    return [name for name, p in model.named_parameters() if p.requires_grad]


def test_paper_stage1_baseline_freeze():
    """Test Paper Stage 1 (Attention Alignment) with baseline config.
    
    Paper Stage 1 = attention_distillation_stage=1
    Only Atlas memory parameters should be trainable.
    """
    config = create_test_config(
        poly_degree=2,
        use_omega_gate=True,  # Fixed in ablation study
        omega_window=16,       # Fixed in ablation study
        qkv_conv_kernel=None,
        attention_distillation_stage=1,
    )
    
    model = Model_atlasqwen2(config)
    model.configure_model()
    model.set_grads()
    
    trainable_names = get_trainable_param_names(model)
    
    # Debug: print parameter names to see structure
    print(f"Sample trainable parameter names:")
    for name in trainable_names[:10]:
        print(f"  {name}")
    
    # Check that embeddings are frozen
    assert not any('embed_tokens' in name for name in trainable_names), \
        "Embeddings should be frozen in Stage 1"
    
    # Check that lm_head is frozen
    assert not any('lm_head' in name for name in trainable_names), \
        "LM head should be frozen in Stage 1"
    
    # Check that MLP layers are frozen
    assert not any('mlp' in name for name in trainable_names), \
        "MLP layers should be frozen in Stage 1"
    
    # Check that memory projections are trainable (allow for .cell. in path)
    assert any('memory' in name and 'to_q' in name for name in trainable_names), \
        "Memory Q projection should be trainable in Stage 1"
    assert any('memory' in name and 'to_k' in name for name in trainable_names), \
        "Memory K projection should be trainable in Stage 1"
    assert any('memory' in name and 'to_v' in name for name in trainable_names), \
        "Memory V projection should be trainable in Stage 1"
    assert any('memory' in name and 'to_out' in name for name in trainable_names), \
        "Memory output projection should be trainable in Stage 1"
    
    # Check that gates are trainable
    assert any('memory' in name and 'to_lr' in name for name in trainable_names), \
        "Learning rate gate should be trainable in Stage 1"
    assert any('memory' in name and 'to_decay' in name for name in trainable_names), \
        "Decay gate should be trainable in Stage 1"
    assert any('memory' in name and 'to_momentum' in name for name in trainable_names), \
        "Momentum gate should be trainable in Stage 1"
    
    # Check that omega_gate IS trainable (use_omega_gate=True in ablation study)
    assert any('memory' in name and 'to_omega_gate' in name for name in trainable_names), \
        "Omega gate should be trainable when use_omega_gate=True"
    
    # Check that conv is NOT present (qkv_conv_kernel=None)
    conv_params = [name for name in trainable_names if 'conv' in name]
    assert len(conv_params) == 0, \
        f"Conv layers should not be trainable when qkv_conv_kernel=None (found: {conv_params})"
    
    print(f"✅ Paper Stage 1 baseline: {count_trainable_params(model):,} trainable parameters")


def test_paper_stage1_with_omega_gate():
    """Test Paper Stage 1 (Attention Alignment) with omega_gate enabled."""
    config = create_test_config(
        poly_degree=2,
        use_omega_gate=True,
        omega_window=16,
        qkv_conv_kernel=None,
        attention_distillation_stage=1,
    )
    
    model = Model_atlasqwen2(config)
    model.configure_model()
    model.set_grads()
    
    trainable_names = get_trainable_param_names(model)
    
    # Check that omega_gate is trainable
    omega_gate_found = any('memory' in name and 'to_omega_gate' in name for name in trainable_names)
    
    if not omega_gate_found:
        print(f"DEBUG: Omega gate not found in trainable params. All params:")
        for name, p in model.named_parameters():
            if 'omega_gate' in name:
                print(f"  {name}: requires_grad={p.requires_grad}")
    
    assert omega_gate_found, \
        "Omega gate should be trainable when use_omega_gate=True"
    
    # Verify it's actually in the model
    has_omega_gate = False
    for layer in model.model.layers:
        memory_cell = layer.memory.cell if hasattr(layer.memory, 'cell') else layer.memory
        if hasattr(memory_cell, 'to_omega_gate') and memory_cell.to_omega_gate is not None:
            has_omega_gate = True
            # Check that it's trainable
            for p in memory_cell.to_omega_gate.parameters():
                if not p.requires_grad:
                    print(f"WARNING: Omega gate parameter has requires_grad=False")
                assert p.requires_grad, "Omega gate parameters should require gradients"
    
    assert has_omega_gate, "Model should have omega_gate when use_omega_gate=True"
    
    print(f"✅ Paper Stage 1 with omega_gate: {count_trainable_params(model):,} trainable parameters")


def test_paper_stage1_with_conv():
    """Test Paper Stage 1 (Attention Alignment) with conv layers enabled."""
    config = create_test_config(
        poly_degree=2,
        use_omega_gate=True,
        qkv_conv_kernel=4,
        attention_distillation_stage=1,
    )
    
    model = Model_atlasqwen2(config)
    model.configure_model()
    model.set_grads()
    
    trainable_names = get_trainable_param_names(model)
    
    # Check that conv layers are trainable
    assert any('memory' in name and 'q_conv' in name for name in trainable_names), \
        "Q conv should be trainable when qkv_conv_kernel is set"
    assert any('memory' in name and 'k_conv' in name for name in trainable_names), \
        "K conv should be trainable when qkv_conv_kernel is set"
    assert any('memory' in name and 'v_conv' in name for name in trainable_names), \
        "V conv should be trainable when qkv_conv_kernel is set"
    
    # Verify conv layers exist and are trainable
    for layer in model.model.layers:
        memory_cell = layer.memory.cell if hasattr(layer.memory, 'cell') else layer.memory
        assert memory_cell.q_conv is not None, "Q conv should exist"
        assert memory_cell.k_conv is not None, "K conv should exist"
        assert memory_cell.v_conv is not None, "V conv should exist"
        
        for p in memory_cell.q_conv.parameters():
            assert p.requires_grad, "Q conv parameters should require gradients"
        for p in memory_cell.k_conv.parameters():
            assert p.requires_grad, "K conv parameters should require gradients"
        for p in memory_cell.v_conv.parameters():
            assert p.requires_grad, "V conv parameters should require gradients"
    
    print(f"✅ Paper Stage 1 with conv: {count_trainable_params(model):,} trainable parameters")


def test_paper_stage1_with_groupnorm():
    """Test Paper Stage 1 (Attention Alignment) with groupnorm enabled."""
    config = create_test_config(
        poly_degree=2,
        use_omega_gate=True,
        use_groupnorm=True,
        attention_distillation_stage=1,
    )
    
    model = Model_atlasqwen2(config)
    model.configure_model()
    model.set_grads()
    
    trainable_names = get_trainable_param_names(model)
    
    # Debug: Check if groupnorm exists
    print(f"DEBUG: Looking for groupnorm parameters...")
    has_groupnorm = False
    for layer in model.model.layers:
        memory_cell = layer.memory.cell if hasattr(layer.memory, 'cell') else layer.memory
        if hasattr(memory_cell, 'groupnorm'):
            print(f"  Found groupnorm attribute: {memory_cell.groupnorm}")
            has_groupnorm = memory_cell.groupnorm is not None
            if has_groupnorm:
                print(f"  GroupNorm exists! Checking parameters...")
                for name, p in memory_cell.groupnorm.named_parameters():
                    print(f"    {name}: requires_grad={p.requires_grad}")
    
    # Check that groupnorm parameters are trainable
    groupnorm_trainable = any('memory' in name and 'groupnorm' in name for name in trainable_names)
    if not groupnorm_trainable:
        print(f"  WARNING: GroupNorm not found in trainable params")
        print(f"  All memory params: {[n for n in trainable_names if 'memory' in n][:20]}")
    
    if has_groupnorm:
        assert groupnorm_trainable, \
            "GroupNorm should be trainable when use_groupnorm=True"
    
    # Verify groupnorm exists and is trainable
    for layer in model.model.layers:
        memory_cell = layer.memory.cell if hasattr(layer.memory, 'cell') else layer.memory
        if hasattr(memory_cell, 'groupnorm') and memory_cell.groupnorm is not None:
            for p in memory_cell.groupnorm.parameters():
                assert p.requires_grad, "GroupNorm parameters should require gradients"
    
    print(f"✅ Paper Stage 1 with groupnorm: {count_trainable_params(model):,} trainable parameters")


def test_full_training_all_trainable():
    """Test full training mode (Stage 2/3): All parameters should be trainable.
    
    When attention_distillation_stage != 1, all parameters are trainable.
    This applies to:
    - Paper Stage 2 (KL Divergence with Teacher): attention_distillation_stage=2
    - Paper Stage 3 (Long Context CE): attention_distillation_stage=-1
    
    Both cases train all parameters, but Stage 2 uses teacher KL loss while
    Stage 3 uses CE loss only.
    """
    config = create_test_config(
        poly_degree=2,
        use_omega_gate=True,
        qkv_conv_kernel=4,
        attention_distillation_stage=-1,  # Full training mode (not Stage 1 alignment)
    )
    
    model = Model_atlasqwen2(config)
    model.configure_model()
    model.set_grads()
    
    trainable_names = get_trainable_param_names(model)
    
    # Check that embeddings are trainable
    assert any('embed_tokens' in name for name in trainable_names), \
        "Embeddings should be trainable in Stage 2"
    
    # Check that lm_head is trainable (may be tied to embeddings)
    # If tied, it won't show up as separate param
    lm_head_or_embeddings = any('lm_head' in name or 'embed_tokens' in name for name in trainable_names)
    assert lm_head_or_embeddings, \
        "LM head or embeddings should be trainable in Stage 2"
    
    # Check that MLP layers are trainable
    assert any('mlp' in name for name in trainable_names), \
        "MLP layers should be trainable in Stage 2"
    
    # Check that memory is also trainable
    assert any('memory' in name and 'to_q' in name for name in trainable_names), \
        "Memory should also be trainable in Stage 2"
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_trainable_params(model)
    
    assert total_params == trainable_params, \
        f"All parameters should be trainable in Stage 2 (got {trainable_params}/{total_params})"
    
    print(f"✅ Full training mode (Paper Stage 2/3): All {trainable_params:,} parameters trainable")


def test_forward_backward_paper_stage1():
    """Test that forward and backward passes work correctly in Paper Stage 1 (Attention Alignment)."""
    config = create_test_config(
        poly_degree=2,
        use_omega_gate=True,
        omega_window=16,
        qkv_conv_kernel=4,
        attention_distillation_stage=1,
    )
    
    model = Model_atlasqwen2(config)
    model.configure_model()
    model.set_grads()
    
    # Create dummy input
    batch, seq_len = 2, 16
    token_ids = torch.randint(0, 256, (batch, seq_len))
    
    # Forward pass
    output = model(token_ids)
    logits = output.logits
    
    assert logits.shape == (batch, seq_len, 256), \
        f"Expected logits shape {(batch, seq_len, 256)}, got {logits.shape}"
    
    # Backward pass
    loss = logits.sum()
    loss.backward()
    
    # Check that only memory parameters have gradients
    for name, p in model.named_parameters():
        if 'memory' in name:
            if p.requires_grad:
                assert p.grad is not None, \
                    f"Memory parameter {name} should have gradients"
        else:
            if p.requires_grad:
                raise AssertionError(f"Non-memory parameter {name} should not be trainable in Stage 1")
            assert p.grad is None, \
                f"Frozen parameter {name} should not have gradients"
    
    print("✅ Forward/backward pass works correctly in Paper Stage 1")


def test_ablation_configs():
    """Test all ablation study configurations."""
    ablation_configs = [
        ("Exp1: Baseline", {
            'poly_degree': 2,
            'use_omega_gate': True,
            'qkv_conv_kernel': None,
            'use_rope': False,
            'use_groupnorm': False,
        }),
        ("Exp2: poly_degree=3", {
            'poly_degree': 3,
            'use_omega_gate': True,
            'qkv_conv_kernel': None,
            'use_rope': False,
            'use_groupnorm': False,
        }),
        ("Exp4a: qk_norm", {
            'poly_degree': 2,
            'use_omega_gate': True,
            'qkv_conv_kernel': None,
            'use_rope': False,
            'use_groupnorm': False,
        }),
        ("Exp4b: groupnorm", {
            'poly_degree': 2,
            'use_omega_gate': True,
            'qkv_conv_kernel': None,
            'use_rope': False,
            'use_groupnorm': True,
        }),
        ("Exp5: qkv_conv", {
            'poly_degree': 2,
            'use_omega_gate': True,
            'qkv_conv_kernel': 4,
            'use_rope': False,
            'use_groupnorm': False,
        }),
        ("Exp6: rope", {
            'poly_degree': 2,
            'use_omega_gate': True,
            'qkv_conv_kernel': None,
            'use_rope': True,
            'use_groupnorm': False,
        }),
    ]
    
    print("\n" + "="*60)
    print("Testing all ablation study configurations")
    print("="*60)
    
    for exp_name, params in ablation_configs:
        config = create_test_config(**params, attention_distillation_stage=1)
        model = Model_atlasqwen2(config)
        model.configure_model()
        model.set_grads()
        
        trainable_params = count_trainable_params(model)
        trainable_names = get_trainable_param_names(model)
        
        # Verify basic freeze/unfreeze logic
        assert not any('embed_tokens' in name for name in trainable_names), \
            f"{exp_name}: Embeddings should be frozen"
        assert any('memory' in name and 'to_q' in name for name in trainable_names), \
            f"{exp_name}: Memory should be trainable"
        
        # Verify specific features
        if params['qkv_conv_kernel'] is not None:
            assert any('memory' in name and 'conv' in name for name in trainable_names), \
                f"{exp_name}: Conv should be trainable"
        
        if params['use_groupnorm']:
            assert any('memory' in name and 'groupnorm' in name for name in trainable_names), \
                f"{exp_name}: GroupNorm should be trainable"
        
        print(f"✅ {exp_name}: {trainable_params:,} trainable parameters")
    
    print("="*60)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Gradient Control Tests for Ablation Study")
    print("Paper Stage 1 = Attention Alignment (attention_distillation_stage=1)")
    print("Paper Stage 2/3 = Full Training (attention_distillation_stage != 1)")
    print("="*60 + "\n")
    
    test_paper_stage1_baseline_freeze()
    test_paper_stage1_with_omega_gate()
    test_paper_stage1_with_conv()
    test_paper_stage1_with_groupnorm()
    test_full_training_all_trainable()
    test_forward_backward_paper_stage1()
    test_ablation_configs()
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60 + "\n")
