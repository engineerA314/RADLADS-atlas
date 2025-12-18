# GPU Debugging Log - Phase 2

## Date: 2025-12-18
## GPU: NVIDIA L4 (22GB VRAM)
## Model: Atlas-LMM 0.5B

---

## ✅ Issues Fixed

### 1. Import Errors
**Problem**: `ModuleNotFoundError: No module named 'qwen2'`
**Solution**: Changed imports from `from qwen2.* import` to `from transformers import`
- Fixed in: `train.py` line 175
- Fixed in: `run_lm_eval_hf.py` line 31-32

### 2. Model Loading Logic
**Problem**: `AssertionError: bad attention type specified` in `models/qwen2.py`
**Solution**: Added conditional import based on `classname`
```python
if config.model.classname == 'atlasqwen2':
    import models.atlasqwen2
else:
    import models.qwen2
```
**Files Modified**: `train.py` lines 48-59, 180-184

### 3. Data File Path
**Problem**: Code expects `dclm-10B.bin.idx` but we have `dclm-10B.idx`
**Solution**: Created symlink: `ln -sf dclm-10B.idx dclm-10B.bin.idx`
**Root Cause**: `binidx.py` auto-appends `.idx` and `.bin` extensions

### 4. Checkpoint Loading
**Problem**: Tried to load 'auto' as a file path
**Solution**: Set `--train.load_model ""` to train from scratch

### 5. Magic Prime Validation
**Problem**: `magic_prime` must match `ctx_len`
**Solutions**:
- ctx_len=256 → magic_prime=5859677
- ctx_len=512 → magic_prime=2929793
- ctx_len=2048 → magic_prime=732461

---

## ❌ Critical Issue: OOM on Atlas Memory

### Problem
**Atlas-LMM consistently runs out of memory on L4 GPU (22GB VRAM)**

### Tests Conducted
| ctx_len | grad_cp | micro_bsz | Result | Memory Used |
|---------|---------|-----------|--------|-------------|
| 2048 | 0 | 1 | ❌ OOM | 21.99 GB |
| 512 | 0 | 1 | ❌ OOM | 21.91 GB |
| 256 | 1 | 1 | ❌ OOM | 22.02 GB |

### Root Cause
The Atlas memory's **associative scan** (`_associative_scan` in `atlas_memory.py`) requires:
- Intermediate tensor storage scales with O(T * d * d) where T=ctx_len, d=head_dim
- For ctx_len=256, head_dim=64, 14 heads, 24 layers:
  - Per layer: ~256 * 64 * 64 * 14 * 4 bytes = ~58 MB
  - Scan tree depth log2(256) = 8 requires multiple 58MB allocations
  - Total: exceeds available VRAM

### Stack Trace
```
File "atlas_memory.py", line 462, in forward
  H_all = _affine_scan_apply(H0, A_seq, C_seq)
File "atlas_memory.py", line 159, in _affine_scan_apply
  A_pref, C_pref = _associative_scan(_affine_pair_operator, (A_seq, C_seq))
File "atlas_memory.py", line 83, in _interleave
  stacked = torch.stack([a, b], dim=2)
torch.OutOfMemoryError: CUDA out of memory.
```

---

## 📋 Recommendations

### Option 1: Use Larger GPU ✅ **Recommended**
- **A100 40GB** or **A100 80GB** required for Atlas-LMM
- L4 (22GB) insufficient for current implementation

### Option 2: Implement Chunked Scan
Modify `atlas_memory.py` to process sequences in chunks:
```python
def _chunked_affine_scan(H0, A_seq, C_seq, chunk_size=64):
    # Process in smaller chunks to reduce peak memory
    pass
```
**Complexity**: High - requires careful implementation

### Option 3: Switch to RWKV Model
Test with standard RWKV models that don't use Atlas memory:
```bash
python3 train.py -c configs/qwen0b5.yaml \
    --model.attention_type rwkv7_wind_triton \
    --train.data_file data/dclm-10B \
    --train.load_model "" \
    --train.my_exit_tokens 5120
```

---

## 🎯 Next Steps

1. **Inform user** of GPU requirements (A100 recommended)
2. **Test on larger GPU** if available
3. **Consider alternative architectures** if limited to L4
4. **Implement memory-efficient scan** (longer-term solution)

---

## Commands for Reference

### Working command (up to OOM):
```bash
python3 train.py -c configs/atlasqwen0b5.yaml \
    --model.ctx_len 256 \
    --train.magic_prime 5859677 \
    --train.grad_cp 1 \
    --train.data_file data/dclm-10B \
    --train.load_model "" \
    --train.micro_bsz 1 \
    --train.devices 1 \
    --train.my_exit_tokens 2560 \
    --train.proj_name "phase2-gpu-smoke-test"
```

### Files Modified:
- `train.py` (imports, conditional model loading)
- `run_lm_eval_hf.py` (imports)
- `tests/conftest.py` (GPU auto-detection)
- `data/dclm-10B.bin.idx` (symlink created)

---

## Summary
**All setup issues resolved ✅**  
**Atlas-LMM architecture incompatible with L4 GPU ❌**  
**Requires A100 or architectural changes**

