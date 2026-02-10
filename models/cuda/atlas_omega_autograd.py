"""
Atlas Omega RNN CUDA Autograd Function
Exact omega_window=16 implementation with checkpoint-based backward
"""

import json
import os
import sys
import importlib
import torch
import torch.nn.functional as F

# Import compiled CUDA extension (original simple version for testing)
_module_path = os.path.dirname(__file__)
_CUDA_AVAILABLE = False
_CUDA_ERROR = ""
_CUDA_LOADED_VIA = "none"
_CUDA_DEBUG_COUNTER = 0


def _cuda_debug_enabled() -> bool:
    raw = str(os.environ.get("NAN_DEBUG_CUDA_GRADS", "") or "").strip().lower()
    return raw in ("1", "true", "yes")


def _cuda_debug_max() -> int:
    raw = str(os.environ.get("NAN_DEBUG_CUDA_GRADS_MAX", "1") or "").strip()
    try:
        return max(1, int(raw))
    except Exception:
        return 1


def _cuda_debug_dir() -> str:
    return str(os.environ.get("NAN_DEBUG_CUDA_GRADS_DIR", "/tmp/radlads_nan_debug_cuda_grads"))


def _cuda_debug_steps() -> set[int]:
    raw = str(os.environ.get("NAN_DEBUG_CUDA_GRADS_STEPS", "") or "").strip()
    if raw == "":
        return set()
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if part == "":
            continue
        try:
            out.add(int(part))
        except Exception:
            continue
    return out


def _cuda_debug_ranks() -> set[int] | None:
    raw = str(os.environ.get("NAN_DEBUG_CUDA_GRADS_RANKS", "") or "").strip()
    if raw == "" or raw.lower() == "all":
        return None
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if part == "":
            continue
        try:
            out.add(int(part))
        except Exception:
            continue
    return out


def _cuda_debug_dump_tensors_enabled() -> bool:
    raw = str(os.environ.get("NAN_DEBUG_CUDA_GRADS_DUMP_TENSORS", "0") or "").strip().lower()
    return raw in ("1", "true", "yes")


def _cuda_debug_layers() -> set[int] | None:
    raw = str(os.environ.get("NAN_DEBUG_CUDA_GRADS_LAYERS", "") or "").strip()
    if raw == "" or raw.lower() == "all":
        return None
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if part == "":
            continue
        try:
            out.add(int(part))
        except Exception:
            continue
    return out


def _cuda_debug_layer() -> int | None:
    raw = str(os.environ.get("NAN_DEBUG_LAYER_IDX", "") or "").strip()
    if raw == "":
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _cuda_debug_step() -> int | None:
    try:
        raw = str(os.environ.get("NAN_DEBUG_BATCH_IDX", "") or "").strip()
        if raw != "":
            return int(raw)
    except Exception:
        pass
    try:
        from atlasattn import get_nan_debug_context
        ctx = get_nan_debug_context()
        step = ctx.get("batch_idx", None)
        if step is None:
            return None
        return int(step)
    except Exception:
        return None


def _tensor_stats(t: torch.Tensor) -> dict:
    flat = t.detach().reshape(-1)
    out = {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "numel": int(flat.numel()),
        "nan": int(torch.isnan(flat).sum().item()),
        "inf": int(torch.isinf(flat).sum().item()),
    }
    finite = torch.isfinite(flat)
    if finite.any():
        f = flat[finite].float()
        out.update(
            absmax=float(f.abs().max().item()),
            mean=float(f.mean().item()),
            std=float(f.std(unbiased=False).item()) if f.numel() > 1 else 0.0,
        )
    else:
        out.update(absmax=None, mean=None, std=None)
    return out


def _maybe_dump_cuda_grads(tag: str, payload: dict) -> None:
    global _CUDA_DEBUG_COUNTER
    if not _cuda_debug_enabled():
        return
    try:
        import torch.distributed as dist
        is_dist = dist.is_available() and dist.is_initialized()
        rank = int(dist.get_rank()) if is_dist else 0
    except Exception:
        rank = 0

    step = _cuda_debug_step()
    steps = _cuda_debug_steps()
    if steps and (step is None or int(step) not in steps):
        return
    ranks = _cuda_debug_ranks()
    if ranks is not None and rank not in ranks:
        return
    layer = payload.get("layer_idx", None)
    if layer is None:
        layer = _cuda_debug_layer()
    layers = _cuda_debug_layers()
    if layers is not None and (layer is None or layer not in layers):
        return
    if _CUDA_DEBUG_COUNTER >= _cuda_debug_max():
        return
    _CUDA_DEBUG_COUNTER += 1

    out_dir = _cuda_debug_dir()
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        out_dir = "/tmp"
    step_suffix = f"_step{step}" if step is not None else ""
    layer_suffix = f"_layer{layer}" if layer is not None else ""
    out_path = os.path.join(out_dir, f"cuda_grad_rank{rank}_call{_CUDA_DEBUG_COUNTER}_{tag}{step_suffix}{layer_suffix}.json")
    if step is not None:
        payload["batch_idx"] = int(step)
    if layer is not None:
        payload["layer_idx"] = int(layer)
    try:
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[NaN-DEBUG] cuda_grad_dump={out_path} tag={tag}")
    except Exception:
        pass

    if _cuda_debug_dump_tensors_enabled():
        try:
            tensor_path = os.path.join(out_dir, f"cuda_grad_tensors_rank{rank}_call{_CUDA_DEBUG_COUNTER}_{tag}{step_suffix}.pt")
            torch.save(payload.get("_tensors", {}), tensor_path)
            print(f"[NaN-DEBUG] cuda_grad_tensors_dump={tensor_path} tag={tag}")
        except Exception:
            pass

def get_cuda_status():
    """Return CUDA backend status for logging/debugging"""
    return {
        "available": _CUDA_AVAILABLE,
        "loaded_via": _CUDA_LOADED_VIA,
        "error": _CUDA_ERROR if not _CUDA_AVAILABLE else ""
    }

# JIT compilation with DeepSpeed-safe locking
atlas_omega_ext = None

def _cuda_dim_head() -> int:
    raw = str(os.environ.get("ATLAS_CUDA_DIM_HEAD", "64") or "").strip()
    try:
        return max(1, int(raw))
    except Exception:
        return 64

try:
    from torch.utils.cpp_extension import load
    import torch.distributed as dist
    
    # In DeepSpeed multi-GPU training, let rank 0 compile first, others wait
    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    
    if is_distributed and rank > 0:
        # Non-rank-0 processes: wait for rank 0 to finish compilation
        import time
        compile_lock = os.path.join(_module_path, ".compile_lock")
        
        # Wait up to 300 seconds for rank 0 to compile
        for _ in range(300):
            if os.path.exists(compile_lock):
                time.sleep(0.5)
            else:
                break
        time.sleep(1)  # Extra safety margin
    
    # Load (compile if needed)
    dim_head = _cuda_dim_head()
    ext_name = f"atlas_omega_ext_c{dim_head}"
    extra_cuda_cflags = [
        "-O3", "-lineinfo",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        f"-D_C_={dim_head}", "-D_E_=16",
        "--use_fast_math",
    ]

    atlas_omega_ext = load(
        name=ext_name,
        sources=[
            os.path.join(_module_path, "atlas_omega.cpp"),
            os.path.join(_module_path, "atlas_omega.cu"),
        ],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False
    )
    _CUDA_AVAILABLE = True
    _CUDA_LOADED_VIA = "jit"
    
    # Rank 0: remove lock file after successful compilation
    if is_distributed and rank == 0:
        compile_lock = os.path.join(_module_path, ".compile_lock")
        if os.path.exists(compile_lock):
            os.remove(compile_lock)
            
except Exception as e:
    atlas_omega_ext = None
    _CUDA_AVAILABLE = False
    _CUDA_ERROR = str(e)
    _CUDA_LOADED_VIA = "failed"



class AtlasOmegaFunction(torch.autograd.Function):
    """
    Atlas Omega RNN with exact sliding window (omega=16) and checkpoint-based backward
    
    Forward: O(BHTd + BH(T/16)d²) memory
    Backward: Recompute from checkpoints, O(d²) shared memory per block
    
    Requires: A100 GPU (shared memory >= 163KB)
    """
    
    @staticmethod
    def forward(ctx, phi_k, phi_q, v, S_ref, lr, decay, beta, gate, S0, Z0):
        """
        Args:
            phi_k: [BH, T, d] - Key features (polynomial-mapped)
            phi_q: [BH, T, d] - Query features (polynomial-mapped)
            v: [BH, T, d] - Values
            S_ref: [BH, d, d] - Reference state for delta computation
            lr: [BH, T] - Learning rates
            decay: [BH, T] - Decay factors (alpha)
            beta: [BH, T] - Momentum factors
            gate: [BH, T] - Omega gates (u_t)
            S0: [BH, d, d] - Initial state S
            Z0: [BH, d, d] - Initial state Z
        
        Returns:
            y: [BH, T, d] - Output
            S_T: [BH, d, d] - Final state S
            Z_T: [BH, d, d] - Final state Z
        """
        BH, T, d = phi_k.shape
        
        # Compute delta: pred = S_ref @ phi_k - v
        # Shape: [BH, T, d] = [BH, d, d] @ [BH, T, d]
        pred = torch.einsum('bde,bte->btd', S_ref, phi_k)
        delta = pred - v
        
        # Prepare outputs
        y = torch.empty((BH, T, d), device=phi_k.device, dtype=torch.bfloat16)
        S_T = torch.empty((BH, d, d), device=phi_k.device, dtype=torch.bfloat16)
        Z_T = torch.empty((BH, d, d), device=phi_k.device, dtype=torch.bfloat16)

        # Checkpoints for numerically stable backward (store S_t, Z_t every K steps)
        # K is fixed to 4 for numerical stability.
        K = 4
        n_ckpt = (T + K - 1) // K
        S_ckpt = torch.empty((BH, n_ckpt, d, d), device=phi_k.device, dtype=torch.float32)
        Z_ckpt = torch.empty((BH, n_ckpt, d, d), device=phi_k.device, dtype=torch.float32)
        
        # Call CUDA kernel
        atlas_omega_ext.forward_exact(
            phi_k.contiguous(), 
            phi_q.contiguous(), 
            delta.contiguous(),
            lr.contiguous(), 
            decay.contiguous(), 
            beta.contiguous(), 
            gate.contiguous(),
            S0.contiguous(), 
            Z0.contiguous(),
            y, S_T, Z_T,
            S_ckpt, Z_ckpt
        )
        
        # Save for backward
        ctx.save_for_backward(phi_k, phi_q, delta, lr, decay, beta, gate, S_T, Z_T, S_ref, S_ckpt, Z_ckpt, S0, Z0)
        # Capture layer index for debug (if provided via env)
        try:
            ctx._nan_layer_idx = _cuda_debug_layer()
        except Exception:
            ctx._nan_layer_idx = None

        return y, S_T, Z_T
    
    @staticmethod
    def backward(ctx, dy, dS_T, dZ_T):
        """
        Backward pass with checkpoint recomputation
        
        Strategy:
        - Load saved states (S_T, Z_T) and inputs
        - Recompute forward for each chunk (16 steps)
        - Compute gradients using ring buffer for "future 16 sum"
        """
        phi_k, phi_q, delta, lr, decay, beta, gate, S_T, Z_T, S_ref, S_ckpt, Z_ckpt, S0, Z0 = ctx.saved_tensors
        BH, T, d = phi_k.shape

        # Unused outputs may produce None grad outputs; treat as zeros.
        if dS_T is None:
            dS_T = torch.zeros_like(S_T)
        if dZ_T is None:
            dZ_T = torch.zeros_like(Z_T)
        
        # Prepare gradient outputs
        dphi_k = torch.empty_like(phi_k)
        dphi_q = torch.empty_like(phi_q)
        ddelta = torch.empty_like(delta)
        dlr = torch.empty((BH, T), device=phi_k.device, dtype=torch.bfloat16)
        ddecay = torch.empty((BH, T), device=phi_k.device, dtype=torch.bfloat16)
        dbeta = torch.empty((BH, T), device=phi_k.device, dtype=torch.bfloat16)
        dgate = torch.empty((BH, T), device=phi_k.device, dtype=torch.bfloat16)
        dS0 = torch.empty((BH, d, d), device=phi_k.device, dtype=torch.bfloat16)
        dZ0 = torch.empty((BH, d, d), device=phi_k.device, dtype=torch.bfloat16)

        # Call CUDA backward kernel
        atlas_omega_ext.backward_exact(
            phi_k, phi_q, delta, lr, decay, beta, gate, S_T, Z_T, S_ckpt, Z_ckpt, S0, Z0,
            dy.contiguous(), dS_T.contiguous(), dZ_T.contiguous(),
            dphi_k, dphi_q, ddelta, dlr, ddecay, dbeta, dgate, dS0, dZ0
        )

        debug_enabled = _cuda_debug_enabled()
        payload = None
        if debug_enabled:
            payload = {
                "tag": "backward",
                "shapes": {"BH": int(BH), "T": int(T), "d": int(d)},
                "inputs": {
                    "phi_k": _tensor_stats(phi_k),
                    "phi_q": _tensor_stats(phi_q),
                    "delta": _tensor_stats(delta),
                    "dy": _tensor_stats(dy),
                },
                "post_kernel": {
                    "dphi_k": _tensor_stats(dphi_k),
                    "dphi_q": _tensor_stats(dphi_q),
                    "ddelta": _tensor_stats(ddelta),
                    "dlr": _tensor_stats(dlr),
                    "ddecay": _tensor_stats(ddecay),
                    "dbeta": _tensor_stats(dbeta),
                    "dgate": _tensor_stats(dgate),
                    "dS0": _tensor_stats(dS0),
                    "dZ0": _tensor_stats(dZ0),
                },
            }
            try:
                layer_idx = getattr(ctx, "_nan_layer_idx", None)
                if layer_idx is not None:
                    payload["layer_idx"] = int(layer_idx)
            except Exception:
                pass
            if _cuda_debug_dump_tensors_enabled():
                payload["_tensors"] = {
                    "phi_k": phi_k.detach().cpu(),
                    "phi_q": phi_q.detach().cpu(),
                    "delta": delta.detach().cpu(),
                    "dy": dy.detach().cpu(),
                    "dS_T": dS_T.detach().cpu(),
                    "dZ_T": dZ_T.detach().cpu(),
                    "dphi_k": dphi_k.detach().cpu(),
                    "dphi_q": dphi_q.detach().cpu(),
                    "ddelta": ddelta.detach().cpu(),
                    "dlr": dlr.detach().cpu(),
                    "ddecay": ddecay.detach().cpu(),
                    "dbeta": dbeta.detach().cpu(),
                    "dgate": dgate.detach().cpu(),
                    "dS0": dS0.detach().cpu(),
                    "dZ0": dZ0.detach().cpu(),
                }
        
        # Propagate ddelta to dv, dphi_k (from delta = pred - v)
        dv = -ddelta
        
        # dphi_k from delta computation: delta = S_ref @ phi_k - v
        # → ddelta/dphi_k = S_ref^T
        dphi_k_from_delta = torch.einsum('bde,btd->bte', S_ref.transpose(1, 2), ddelta)
        dphi_k += dphi_k_from_delta
        
        # dS_ref from delta computation
        dS_ref = torch.einsum('btd,bte->bde', ddelta, phi_k)

        if debug_enabled and payload is not None:
            payload["post_delta"] = {
                "dphi_k": _tensor_stats(dphi_k),
                "dphi_k_from_delta": _tensor_stats(dphi_k_from_delta),
                "ddelta": _tensor_stats(ddelta),
                "dv": _tensor_stats(dv),
            }
            _maybe_dump_cuda_grads("backward", payload)
        
        # Return gradients in order: phi_k, phi_q, v, S_ref, lr, decay, beta, gate, S0, Z0
        return dphi_k, dphi_q, dv, dS_ref, dlr, ddecay, dbeta, dgate, dS0, dZ0


def atlas_omega_forward(phi_k, phi_q, v, S_ref, lr, decay, beta, gate, S0, Z0):
    """
    Convenience function for forward pass (no autograd)
    """
    return AtlasOmegaFunction.apply(phi_k, phi_q, v, S_ref, lr, decay, beta, gate, S0, Z0)


def check_cuda_availability():
    """
    Check if CUDA extension is available and shared memory is sufficient
    """
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    
    if not _CUDA_AVAILABLE:
        return False, f"CUDA extension not loaded: {_CUDA_ERROR}. Run: cd models/cuda && python3 compile.py"
    
    props = torch.cuda.get_device_properties(0)
    shared_mem = props.shared_memory_per_block
    
    # Note: Global memory ring buffer is used, so no strict shared memory requirement
    return True, f"Atlas Omega CUDA ready (GPU: {props.name}, loaded_via={_CUDA_LOADED_VIA}, using global memory ring buffer)"


if __name__ == "__main__":
    # Test
    available, msg = check_cuda_availability()
    print(f"Status: {msg}")
    
    if available:
        print("\nRunning quick test...")
        BH, T, d = 2, 32, 64
        
        phi_k = torch.randn(BH, T, d, device='cuda', dtype=torch.bfloat16)
        phi_q = torch.randn(BH, T, d, device='cuda', dtype=torch.bfloat16)
        v = torch.randn(BH, T, d, device='cuda', dtype=torch.bfloat16)
        S_ref = torch.randn(BH, d, d, device='cuda', dtype=torch.bfloat16)
        
        lr = torch.full((BH, T), 0.01, device='cuda', dtype=torch.bfloat16)
        decay = torch.full((BH, T), 0.99, device='cuda', dtype=torch.bfloat16)
        beta = torch.full((BH, T), 0.9, device='cuda', dtype=torch.bfloat16)
        gate = torch.full((BH, T), 1.0, device='cuda', dtype=torch.bfloat16)
        
        S0 = torch.randn(BH, d, d, device='cuda', dtype=torch.bfloat16)
        Z0 = torch.randn(BH, d, d, device='cuda', dtype=torch.bfloat16)
        
        y, S_T, Z_T = atlas_omega_forward(phi_k, phi_q, v, S_ref, lr, decay, beta, gate, S0, Z0)
        
        print(f"Output shape: {y.shape}")
        print(f"Final states: S_T={S_T.shape}, Z_T={Z_T.shape}")
        print("Test passed!")

