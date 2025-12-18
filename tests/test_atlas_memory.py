"""
Tests for Atlas RNN Memory module.

Tests cover:
- Basic import and instantiation
- Forward pass shape correctness
- State initialization and carry
- Determinism under fixed seed
- Streaming vs batch equivalence
"""

import pytest
import torch

from models.atlas_memory import (
    RNNMemoryCell,
    RNNMemory,
    OmegaRNNMemoryCell,
    RNNMemState,
    state_detach,
)


class TestImportSmoke:
    """Ensure imports don't trigger GPU/CUDA compilation."""
    
    def test_import_no_cuda_compile(self):
        """Importing atlas_memory should not require CUDA."""
        # If we got here, the import in the module header succeeded
        assert RNNMemoryCell is not None
        assert RNNMemory is not None
        assert RNNMemState is not None


class TestRNNMemoryCellBasic:
    """Basic tests for RNNMemoryCell."""
    
    def test_instantiation(self, tiny_memory_config):
        """Cell can be instantiated with tiny config."""
        cell = RNNMemoryCell(**tiny_memory_config)
        assert cell is not None
        assert cell.dim == 64
        assert cell.heads == 4
        assert cell.dim_head == 16
    
    def test_forward_shape(self, tiny_memory_config, make_input, device, dtype):
        """Forward pass produces correct output shape."""
        cell = RNNMemoryCell(**tiny_memory_config).to(device, dtype)
        x = make_input(batch=2, seq_len=16, dim=64, device=device, dtype=dtype)
        
        out, state = cell(x)
        
        assert out.shape == x.shape
        assert isinstance(state, RNNMemState)
        assert state.S.shape == (2 * 4, 16, 16)  # batch*heads, dim_head, dim_head
    
    def test_state_initialization(self, tiny_memory_config, device, dtype):
        """init_state produces correct state shapes."""
        cell = RNNMemoryCell(**tiny_memory_config).to(device, dtype)
        
        state = cell.init_state(batch=3, device=device, dtype=dtype)
        
        assert state.seq_index == 0
        assert state.S.shape == (3 * 4, 16, 16)  # batch*heads, dim_head, dim_head
        if cell.use_momentum:
            assert state.Z.shape == state.S.shape
        else:
            assert state.Z is None
        assert state.omega_buffer is None


class TestRNNMemoryCellDeterminism:
    """Test determinism and reproducibility."""
    
    def test_deterministic_output(self, tiny_memory_config, make_input, seed, device, dtype):
        """Same seed + input produces same output."""
        cell = RNNMemoryCell(**tiny_memory_config).to(device, dtype)
        cell.eval()
        
        torch.manual_seed(seed)
        x = make_input(batch=2, seq_len=16, dim=64, device=device, dtype=dtype)
        out1, _ = cell(x)
        
        torch.manual_seed(seed)
        x = make_input(batch=2, seq_len=16, dim=64, device=device, dtype=dtype)
        out2, _ = cell(x)
        
        assert torch.allclose(out1, out2, atol=1e-6)


class TestRNNMemoryCellStateCarry:
    """Test state carry semantics (streaming vs batch equivalence)."""
    
    def test_streaming_vs_batch(self, tiny_memory_config, make_input, seed, device, dtype):
        """Streaming token-by-token matches batch forward."""
        cell = RNNMemoryCell(**tiny_memory_config).to(device, dtype)
        cell.eval()
        
        torch.manual_seed(seed)
        batch, seq_len = 2, 8
        x = make_input(batch=batch, seq_len=seq_len, dim=64, device=device, dtype=dtype)
        
        # Batch forward
        batch_out, batch_final_state = cell(x)
        
        # Streaming forward (token by token)
        state = None
        streaming_outs = []
        for t in range(seq_len):
            out_t, state = cell(x[:, t:t+1, :], state)
            streaming_outs.append(out_t)
        streaming_out = torch.cat(streaming_outs, dim=1)
        
        # Check equivalence
        assert torch.allclose(batch_out, streaming_out, atol=1e-5), \
            f"Max diff: {(batch_out - streaming_out).abs().max()}"
        assert torch.allclose(batch_final_state.S, state.S, atol=1e-5)
        if cell.use_momentum:
            assert torch.allclose(batch_final_state.Z, state.Z, atol=1e-5)


class TestRNNMemoryFactory:
    """Tests for the RNNMemory factory class."""
    
    def test_factory_basic_cell(self, tiny_memory_config):
        """Factory creates basic cell when omega not requested."""
        mem = RNNMemory(**tiny_memory_config)
        assert isinstance(mem.cell, RNNMemoryCell)
    
    def test_factory_omega_cell(self, tiny_memory_config):
        """Factory creates omega cell when omega_window > 1."""
        config = {**tiny_memory_config, 'omega_window': 4, 'use_omega_gate': True}
        mem = RNNMemory(**config)
        assert isinstance(mem.cell, OmegaRNNMemoryCell)
    
    def test_factory_forward(self, tiny_memory_config, make_input, device, dtype):
        """Factory forward works correctly."""
        mem = RNNMemory(**tiny_memory_config).to(device, dtype)
        x = make_input(batch=2, seq_len=16, dim=64, device=device, dtype=dtype)
        
        out, state = mem(x)
        
        assert out.shape == x.shape
        assert isinstance(state, RNNMemState)


class TestOmegaRNNMemoryCell:
    """Tests for the Omega (sliding window) variant."""
    
    def test_instantiation(self, tiny_memory_config):
        """Omega cell can be instantiated."""
        config = {**tiny_memory_config, 'omega_window': 4, 'use_omega_gate': True}
        cell = OmegaRNNMemoryCell(**config)
        assert cell.omega_window == 4
        assert cell.use_omega_gate is True
    
    def test_omega_buffer_initialization(self, tiny_memory_config, device, dtype):
        """Omega buffer is initialized with correct shape."""
        config = {**tiny_memory_config, 'omega_window': 4, 'use_omega_gate': True}
        cell = OmegaRNNMemoryCell(**config).to(device, dtype)
        
        state = cell.init_state(batch=2, device=device, dtype=dtype)
        
        assert state.omega_buffer is not None
        # Shape: [batch*heads, omega_window-1, dim_head, dim_head, 2]
        assert state.omega_buffer.shape == (2 * 4, 3, 16, 16, 2)
    
    def test_forward_shape(self, tiny_memory_config, make_input, device, dtype):
        """Omega cell forward produces correct shapes."""
        config = {**tiny_memory_config, 'omega_window': 4, 'use_omega_gate': True}
        cell = OmegaRNNMemoryCell(**config).to(device, dtype)
        x = make_input(batch=2, seq_len=16, dim=64, device=device, dtype=dtype)
        
        out, state = cell(x)
        
        assert out.shape == x.shape
        assert state.omega_buffer is not None


class TestStateDetach:
    """Test state_detach utility."""
    
    def test_detach_removes_grad(self, tiny_memory_config, make_input, device):
        """state_detach removes gradient tracking."""
        cell = RNNMemoryCell(**tiny_memory_config).to(device, torch.float32)
        x = make_input(batch=2, seq_len=8, dim=64, device=device, dtype=torch.float32)
        x.requires_grad_(True)
        
        out, state = cell(x)
        
        # State tensors might require grad through computation
        detached = state_detach(state)
        
        assert not detached.S.requires_grad
        if detached.Z is not None:
            assert not detached.Z.requires_grad


class TestNoMomentum:
    """Test cell behavior when momentum is disabled."""
    
    def test_no_momentum_instantiation(self, tiny_memory_config):
        """Cell works without momentum."""
        config = {**tiny_memory_config, 'use_momentum': False}
        cell = RNNMemoryCell(**config)
        assert cell.use_momentum is False
        assert cell.to_momentum is None
    
    def test_no_momentum_state(self, tiny_memory_config, device, dtype):
        """No-momentum cell has Z=None in state."""
        config = {**tiny_memory_config, 'use_momentum': False}
        cell = RNNMemoryCell(**config).to(device, dtype)
        
        state = cell.init_state(batch=2, device=device, dtype=dtype)
        
        assert state.Z is None
    
    def test_no_momentum_forward(self, tiny_memory_config, make_input, device, dtype):
        """No-momentum forward works correctly."""
        config = {**tiny_memory_config, 'use_momentum': False}
        cell = RNNMemoryCell(**config).to(device, dtype)
        x = make_input(batch=2, seq_len=8, dim=64, device=device, dtype=dtype)
        
        out, state = cell(x)
        
        assert out.shape == x.shape
        assert state.Z is None
