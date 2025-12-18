"""
Pytest configuration and fixtures for RADLADS-atlas tests.

Provides:
- Tiny model factories for fast CPU-based testing
- Common test utilities
"""

import pytest
import torch


# ============================================================================
# Tiny model configuration for fast tests
# ============================================================================

@pytest.fixture
def tiny_config():
    """Return a minimal config dict for fast testing."""
    return {
        'dim': 64,
        'dim_head': 16,
        'heads': 4,
        'n_layer': 2,
        'vocab_size': 256,
        'ctx_len': 32,
    }


@pytest.fixture
def tiny_memory_config():
    """Config for testing memory cell in isolation."""
    return {
        'dim': 64,
        'dim_head': 16,
        'heads': 4,
        'use_momentum': True,
        'poly_degree': 1,
        'poly_mode': 'off',
        'qk_norm': True,
        'qkv_conv_kernel': None,  # Disable conv for simpler testing
    }


# ============================================================================
# Device fixtures
# ============================================================================

@pytest.fixture
def device():
    """Return CPU device for testing."""
    return torch.device('cpu')


@pytest.fixture
def dtype():
    """Return default dtype for testing."""
    return torch.float32


# ============================================================================
# Test data generators
# ============================================================================

@pytest.fixture
def make_input():
    """Factory to create random input tensors."""
    def _make(batch=2, seq_len=16, dim=64, device='cpu', dtype=torch.float32):
        return torch.randn(batch, seq_len, dim, device=device, dtype=dtype)
    return _make


@pytest.fixture
def make_token_ids():
    """Factory to create random token ID tensors."""
    def _make(batch=2, seq_len=16, vocab_size=256, device='cpu'):
        return torch.randint(0, vocab_size, (batch, seq_len), device=device)
    return _make


# ============================================================================
# Seeding for reproducibility
# ============================================================================

@pytest.fixture
def seed():
    """Set and return a fixed seed for reproducibility."""
    seed_val = 42
    torch.manual_seed(seed_val)
    return seed_val


# ============================================================================
# Skip conditions
# ============================================================================

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)
