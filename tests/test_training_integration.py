"""
TDD Tests for Atlas Training Integration (Phase D).

Tests verify:
- Model instantiates via classname mechanism
- Forward returns expected output structure
- get_optim_groups works correctly
- Training step produces finite loss and gradients
"""

import pytest
import torch
import torch.nn as nn

# Mock config classes for testing
from dataclasses import dataclass
from typing import Any


@dataclass
class MockTrainConfig:
    weight_decay: float = 0.01
    lr_init: float = 1e-4
    lr_final: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.99
    adam_eps: float = 1e-8
    attention_distillation_stage: int = -1


@dataclass 
class MockModelConfig:
    classname: str = 'atlasqwen2'
    vocab_size: int = 256
    n_embd: int = 64
    n_layer: int = 2
    dim_ffn: int = 128
    dim_att: int = 64
    head_size: int = 16
    ctx_len: int = 32
    rms_norm_eps: float = 1e-6
    hf_path: str = ''
    memory_heads: int = 4
    memory_dim_head: int = 16
    use_momentum: bool = True
    poly_degree: int = 1
    poly_mode: str = 'off'
    qk_norm: bool = True
    qkv_conv_kernel: int = None
    tie_word_embeddings: bool = True


@dataclass
class MockConfig:
    model: MockModelConfig
    train: MockTrainConfig


@pytest.fixture
def mock_config():
    return MockConfig(
        model=MockModelConfig(),
        train=MockTrainConfig()
    )


class TestAtlasModelClassname:
    """Test model instantiation via classname mechanism."""
    
    def test_import_model_class(self):
        """Model class can be imported via pydoc.locate pattern."""
        from pydoc import locate
        model_classpath = 'models.atlasqwen2.Model_atlasqwen2'
        model_factory = locate(model_classpath)
        assert model_factory is not None
    
    def test_instantiate_from_config(self, mock_config):
        """Model instantiates from mock config."""
        from models.atlasqwen2 import Model_atlasqwen2
        model = Model_atlasqwen2(mock_config)
        assert model is not None
    
    def test_configure_model(self, mock_config):
        """configure_model creates model internals."""
        from models.atlasqwen2 import Model_atlasqwen2
        model = Model_atlasqwen2(mock_config)
        model.configure_model()
        
        assert model.model is not None
        assert model.lm_head is not None
        assert len(model.model.layers) == mock_config.model.n_layer


class TestAtlasForwardOutput:
    """Test forward returns expected structure."""
    
    def test_forward_returns_output_with_logits(self, mock_config):
        """Forward returns object with .logits attribute."""
        from models.atlasqwen2 import Model_atlasqwen2
        model = Model_atlasqwen2(mock_config)
        model.configure_model()
        
        x = torch.randint(0, mock_config.model.vocab_size, (2, 8))
        result = model(x)
        
        assert hasattr(result, 'logits')
        assert result.logits.shape == (2, 8, mock_config.model.vocab_size)
    
    def test_forward_returns_model_state(self, mock_config):
        """Forward returns model_state."""
        from models.atlasqwen2 import Model_atlasqwen2
        model = Model_atlasqwen2(mock_config)
        model.configure_model()
        
        x = torch.randint(0, mock_config.model.vocab_size, (2, 8))
        result = model(x)
        
        assert hasattr(result, 'model_state')
        assert result.model_state is not None


class TestAtlasOptimGroups:
    """Test optimizer group configuration."""
    
    def test_get_optim_groups(self, mock_config):
        """get_optim_groups returns valid groups."""
        from models.atlasqwen2 import Model_atlasqwen2
        model = Model_atlasqwen2(mock_config)
        model.configure_model()
        
        groups = model.get_optim_groups()
        
        assert isinstance(groups, list)
        assert len(groups) > 0
        for g in groups:
            assert 'params' in g
            assert 'weight_decay' in g
    
    def test_optim_groups_all_params_trainable(self, mock_config):
        """All parameters appear in optimizer groups."""
        from models.atlasqwen2 import Model_atlasqwen2
        model = Model_atlasqwen2(mock_config)
        model.configure_model()
        
        groups = model.get_optim_groups()
        
        # Count parameters in groups
        group_params = set()
        for g in groups:
            for p in g['params']:
                group_params.add(id(p))
        
        # Count trainable parameters in model
        model_params = set()
        for p in model.parameters():
            if p.requires_grad:
                model_params.add(id(p))
        
        assert group_params == model_params


class TestAtlasTrainingStep:
    """Test training step mechanics."""
    
    def test_loss_computation(self, mock_config):
        """Cross-entropy loss can be computed from outputs."""
        from models.atlasqwen2 import Model_atlasqwen2
        model = Model_atlasqwen2(mock_config)
        model.configure_model()
        
        x = torch.randint(0, mock_config.model.vocab_size, (2, 8))
        y = torch.randint(0, mock_config.model.vocab_size, (2, 8))
        
        result = model(x)
        logits = result.logits
        
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )
        
        assert loss.isfinite()
        assert loss > 0
    
    def test_backward_produces_gradients(self, mock_config):
        """Backward pass produces gradients on parameters."""
        from models.atlasqwen2 import Model_atlasqwen2
        model = Model_atlasqwen2(mock_config)
        model.configure_model()
        
        x = torch.randint(0, mock_config.model.vocab_size, (2, 8))
        y = torch.randint(0, mock_config.model.vocab_size, (2, 8))
        
        result = model(x)
        logits = result.logits
        
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )
        loss.backward()
        
        # Check that some parameters have gradients
        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        
        assert has_grad, "No parameter has non-zero gradients"
    
    def test_gradients_are_finite(self, mock_config):
        """Gradients are finite (no NaN/Inf)."""
        from models.atlasqwen2 import Model_atlasqwen2
        model = Model_atlasqwen2(mock_config)
        model.configure_model()
        
        x = torch.randint(0, mock_config.model.vocab_size, (2, 8))
        y = torch.randint(0, mock_config.model.vocab_size, (2, 8))
        
        result = model(x)
        logits = result.logits
        
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )
        loss.backward()
        
        # Check all gradients are finite
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"


class TestAtlasDistillation:
    """Test distillation-related functionality."""
    
    def test_hidden_states_output(self, mock_config):
        """Model can output hidden states for distillation."""
        from models.atlasqwen2 import Model_atlasqwen2
        model = Model_atlasqwen2(mock_config)
        model.configure_model()
        
        x = torch.randint(0, mock_config.model.vocab_size, (2, 8))
        result = model(x, output_hidden_states=True)
        
        assert hasattr(result, 'hidden_states')
        assert result.hidden_states is not None
        # Should have n_layer + 1 hidden states (embedding + each layer)
        assert len(result.hidden_states) == mock_config.model.n_layer + 1
