# CUDA Backend for Atlas-RNN

This directory contains GPU-accelerated CUDA implementations of Atlas memory operations.

## Files

- `atlas_omega.cu`: CUDA kernels (forward/backward) for omega_window=16
- `atlas_omega.cpp`: C++ wrapper for PyTorch
- `atlas_omega_autograd.py`: PyTorch autograd integration
- `compile.py`: JIT compilation script

## Performance

**Memory Usage (0.5B model):**

- 512 context: 0.098 GB
- 16K context: 1.229 GB (vs OOM in PyTorch)

**Speed:**

- 50-100x faster than PyTorch
- Training time: 4.1 hours (vs hundreds of hours)

## Installation

### Prerequisites

- NVIDIA GPU (A100 recommended, T4+ supported)
- CUDA 11.8+
- PyTorch 2.0+
- ninja (for fast compilation)

### Compile CUDA Extension

```bash
cd models/cuda
python3 compile.py
```

This will compile the CUDA kernels and verify they work correctly.

## Usage

### Option 1: Modify atlas_memory.py (Recommended for RADLADS-atlas)

Add `use_cuda=True` when creating `OmegaRNNMemoryCell`:

```python
# In models/atlas_memory.py, modify OmegaRNNMemoryCell.__init__:
def __init__(
    self,
    dim: int,
    ...
    omega_window: int = 16,
    use_cuda: bool = False,  # Add this parameter
):
    ...

    # Add CUDA backend setup
    if use_cuda and omega_window == 16:
        try:
            from models.cuda import check_cuda_availability, AtlasOmegaFunction
            available, msg = check_cuda_availability()
            if available:
                self.AtlasOmegaFunction = AtlasOmegaFunction
                self.cuda_available = True
            else:
                warnings.warn(f"CUDA not available: {msg}")
                self.cuda_available = False
        except ImportError:
            warnings.warn("CUDA extension not compiled")
            self.cuda_available = False
    else:
        self.cuda_available = False
```

Then in `_forward_impl`, add CUDA branch:

```python
def _forward_impl(self, x_normed, q_in, k_in, v_in, state, *, S_ref):
    if self.cuda_available:
        return self._forward_cuda(x_normed, q_in, k_in, v_in, state, S_ref=S_ref)

    # ... existing PyTorch implementation
```

See `atlas-rnn/atlas_pytorch/rnn_memory.py` for complete `_forward_cuda` implementation.

### Option 2: Enable in atlasattn.py

In `atlasattn.py`, when creating `RNNMemory`:

```python
# In AtlasSelfAttention.__init__:
self.memory = RNNMemory(
    dim=self.hidden_size,
    ...
    omega_window=omega_window,
    use_cuda=True,  # Enable CUDA backend
)
```

### Option 3: Enable via config

Add to your YAML config:

```yaml
# In configs/atlasqwen0b5.yaml:
use_cuda: true
omega_window: 16 # Required for CUDA
```

Then modify `atlasattn.py` to pass this parameter:

```python
use_cuda = getattr(config, 'use_cuda', False)
self.memory = RNNMemory(..., use_cuda=use_cuda)
```

## Requirements

**CUDA Backend only supports:**

- `omega_window = 16` (fixed)
- `use_momentum = True`
- `use_omega_gate = True`

Other configurations will automatically fall back to PyTorch.

## Numerical stability (important)

For long sequences (especially Stage 3 with ctx=16384), CUDA backward needs to avoid long backstepping drift.
This implementation uses **periodic state checkpoints** inside the CUDA kernels:

- **Checkpoint interval**: `K = 4` (compile-time constant in `models/cuda/atlas_omega.cu`)
- **What is stored**: `S_t`, `Z_t` at each checkpoint (fp32)
- **Why**: prevents gradient explosion → avoids early loss NaNs

## Testing

Run tests to verify correctness:

```bash
# From RADLADS-atlas root
cd models/cuda
pytest ../../atlas-rnn/tests/test_atlas_cuda.py -v
pytest ../../atlas-rnn/tests/test_atlas_performance.py -v
```

## Troubleshooting

### Compilation fails

```bash
# Install ninja
pip install ninja

# Clear cache and recompile
rm -rf ~/.cache/torch_extensions
python3 compile.py
```

### "CUDA not available" warning

Check:

```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show GPU name
```

### Falls back to PyTorch

Check omega_window:

```python
# CUDA only supports omega_window=16
omega_window = 16  # Must be exactly 16
```

## Development

To modify CUDA kernels:

1. Edit `atlas_omega.cu` or `atlas_omega.cpp`
2. Recompile: `rm -rf ~/.cache/torch_extensions && python3 compile.py`
3. Test: `pytest test_atlas_cuda.py -v`

## Performance Notes

**Global Memory Ring Buffer:**

- Backward kernel uses global memory for 16-element ring buffer
- Expected ~5-15% slower than shared memory version
- Still 50-100x faster than PyTorch
- Enables support for all GPUs (not just A100)

**Memory Savings:**

- PyTorch: O(BHTd²) activation memory → OOM at 16K
- CUDA: O(BHTd) activation memory → 1.2GB at 16K

**Speed Improvements:**

- No [BH, T, d, d] tensors materialized
- Fused operations in single kernel
- Checkpointing for memory efficiency

## References

- Implementation based on RWKV-7 WindBackstepping
- Exact omega_window=16 sliding window (no approximation)
- See `../../atlas-rnn/CUDA_IMPLEMENTATION_STRATEGY.md` for details
