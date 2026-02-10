"""
CUDA backend for Atlas-RNN memory cells.

This module provides GPU-accelerated implementations of Atlas memory operations.
"""

from .atlas_omega_autograd import AtlasOmegaFunction, check_cuda_availability

__all__ = ['AtlasOmegaFunction', 'check_cuda_availability']
