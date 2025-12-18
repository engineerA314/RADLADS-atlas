# RADLADS-Atlas

## Rapid Attention Distillation to Linear Attention Decoders at Scale + Atlas-RNN

This is a fork of [RADLADS](https://arxiv.org/abs/2505.03005) extended with **Atlas-RNN** memory architectures.

### Related Work

- **RADLADS Paper**: [arXiv:2505.03005](https://arxiv.org/abs/2505.03005)
- **Atlas Paper**: [ATLAS: Learning to Optimally Memorize the Context at Test Time (arXiv:2505.23735)](https://arxiv.org/abs/2505.23735)
- **Atlas-RNN Implementation**: [github.com/engineerA314/atlas-rnn](https://github.com/engineerA314/atlas-rnn)

---

## Overview

### Original RADLADS

RADLADS converts traditional softmax attention transformers to use linear attention variants that feature constant-time inference per token. This is accomplished via a three stage distillation process that maintains quality close to the original teacher model.

<div align="center">
    <img src="assets/radlads_process.png" height=63 alt="RADLADS Conversion Process" /> 
</div>

### Atlas-RNN Extension

This fork adds **Atlas-RNN** memory-based architectures as alternative target models:

| Variant       | Description                       | Use Case                           |
| ------------- | --------------------------------- | ---------------------------------- |
| **Atlas-LMM** | Pure RNN memory, no attention     | Maximum efficiency, O(1) per token |
| **Atlas-MAL** | Memory + Sliding Window Attention | Better quality, hybrid approach    |

Both variants support:

- Full HuggingFace `transformers` compatibility (`save_pretrained`, `from_pretrained`, `generate`)
- Recurrent memory state caching via `past_key_values`
- Integration with RADLADS distillation pipeline
- Safetensors export for deployment

---

## What's Pre-configured (DCLM-10B Ready)

If you're using the **DCLM-10B dataset**, everything is already set up:

| Component            | Status   | Notes                                             |
| -------------------- | -------- | ------------------------------------------------- |
| `magic_prime` values | ✅ Ready | Pre-computed for all ctx_len in `distill*.yaml`   |
| Teacher configs      | ✅ Ready | `qwen*teacher.yaml` and `qwen*hfteacher.yaml`     |
| Atlas model configs  | ✅ Ready | `atlasqwen0b5.yaml`                               |
| Distillation stages  | ✅ Ready | `distill1.yaml`, `distill2.yaml`, `distill3.yaml` |

**Minimal command to start training:**

```bash
python train.py \
    -c configs/atlasqwen0b5.yaml \
    -c configs/qwen0b5hfteacher.yaml \
    -c configs/distill2.yaml \
    --train.data_file data/dclm-10B.bin
```

---

## Current Implementation Status

> ⚠️ **Note**: This implementation has been tested with CPU-based unit tests (89 tests pass).  
> **GPU end-to-end training has NOT been validated yet.**

| Component                    | Status          | Notes                                 |
| ---------------------------- | --------------- | ------------------------------------- |
| Atlas Memory (RNNMemoryCell) | ✅ Implemented  | Unit tests pass                       |
| Atlas-LMM Model              | ✅ Implemented  | Forward/backward works on CPU         |
| Atlas-MAL Model              | ✅ Implemented  | Streaming requires KV cache (skipped) |
| HuggingFace Wrapper          | ✅ Implemented  | save/load/generate works              |
| Training Integration         | ✅ Implemented  | Connects to train.py                  |
| GPU Training                 | ⚠️ **Untested** | Needs validation                      |
| Distillation Pipeline        | ⚠️ **Untested** | Needs validation with teacher         |

### First GPU Test Checklist

If you're debugging this on GPU for the first time:

```bash
# 1. Verify CPU tests pass
python -m pytest tests/ -v
# Expected: 89 passed, 2 skipped

# 2. Quick GPU smoke test (few steps only)
python train.py -c configs/atlasqwen0b5.yaml \
    --train.data_file data/dclm-10B.bin \
    --train.micro_bsz 1 \
    --train.devices 1 \
    --train.max_steps 10

# 3. If step 2 works, try with teacher
python train.py \
    -c configs/atlasqwen0b5.yaml \
    -c configs/qwen0b5hfteacher.yaml \
    -c configs/distill2.yaml \
    --train.data_file data/dclm-10B.bin \
    --train.micro_bsz 1 \
    --train.devices 1 \
    --train.max_steps 10

# 4. If both work, run full training
```

### Known Potential Issues

Things that might need fixing during GPU testing:

1. **CUDA kernel compilation** — Triton/flash-linear-attention may need specific versions
2. **Memory issues** — Batch size may need tuning for your GPU
3. **DeepSpeed compatibility** — Some strategies may not work with Atlas memory state
4. **Teacher loading** — Large teacher models may OOM during loading

---

## GPU Requirements

### Minimum (0.5B model, debugging)

| GPU      | VRAM | Batch Size | Notes              |
| -------- | ---- | ---------- | ------------------ |
| L4       | 24GB | 1-2        | Good for debugging |
| A10G     | 24GB | 2-4        | Faster than L4     |
| RTX 3090 | 24GB | 1-2        | Consumer option    |

### Recommended (0.5B model, training)

| GPU       | VRAM | Batch Size | Notes         |
| --------- | ---- | ---------- | ------------- |
| A100 40GB | 40GB | 4-8        | 2-3x faster   |
| A100 80GB | 80GB | 8-16       | Large batches |

### Training Time Estimates (0.5B model, 100M tokens)

| GPU         | Approximate Time |
| ----------- | ---------------- |
| L4 (24GB)   | ~8-12 hours      |
| A100 (40GB) | ~3-5 hours       |

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/RADLADS-atlas.git
cd RADLADS-atlas

# Install dependencies (Option A: requirements.txt)
pip install -r requirements.txt

# Install dependencies (Option B: manual)
pip install lightning torch flash-linear-attention triton deepspeed wandb ninja transformers safetensors --upgrade

# Verify installation with tests
python -m pytest tests/ -v
# Expected: 89 passed, 2 skipped
```

### 2. Download Training Data

```bash
mkdir -p data
wget --continue -O data/dclm-10B.idx https://huggingface.co/datasets/recursal/DCLM-10B-Qwen2-binidx/resolve/main/dclm-10B.idx?download=true
wget --continue -O data/dclm-10B.bin https://huggingface.co/datasets/recursal/DCLM-10B-Qwen2-binidx/resolve/main/dclm-10B.bin?download=true
```

### 3. Train Atlas-LMM (Simplest Example)

```bash
python train.py -c configs/atlasqwen0b5.yaml \
    --train.data_file data/dclm-10B.bin \
    --train.devices 1
```

---

## Practical Tutorial

### Changing the Target Model Size

The model architecture is defined in config files. Key fields to modify:

| Field                   | Description            | Example Values           |
| ----------------------- | ---------------------- | ------------------------ |
| `model.vocab_size`      | Vocabulary size        | 151936 (Qwen)            |
| `model.n_embd`          | Hidden dimension       | 896, 1536, 2048, 3584    |
| `model.n_layer`         | Number of layers       | 24, 28, 32               |
| `model.dim_ffn`         | FFN intermediate size  | Usually 2.5-4x n_embd    |
| `model.memory_heads`    | Number of memory heads | n_embd / memory_dim_head |
| `model.memory_dim_head` | Dimension per head     | 64 (typical)             |

**Example: Creating a 1.5B config**

```bash
cp configs/atlasqwen0b5.yaml configs/atlasqwen1b5.yaml
```

Edit `configs/atlasqwen1b5.yaml`:

```yaml
model:
  __type__: configs.Atlas_Config
  classname: atlasqwen2
  vocab_size: 151936
  n_embd: 1536 # Larger hidden size
  n_layer: 28 # More layers
  dim_ffn: 8960 # Larger FFN
  dim_att: 1536
  head_size: 64
  ctx_len: 2048
  memory_heads: 24 # 1536 / 64
  memory_dim_head: 64
  use_momentum: true
  atlas_variant: "lmm"
```

### Switching Between LMM and MAL Variants

**LMM (Pure Memory, No Attention)**:

```yaml
model:
  atlas_variant: "lmm"
```

**MAL (Memory + Sliding Window Attention)**:

```yaml
model:
  atlas_variant: "mal"
  sliding_window: 512 # Attention window size
```

Command line override:

```bash
python train.py -c configs/atlasqwen0b5.yaml \
    --model.atlas_variant mal \
    --model.sliding_window 256
```

---

## Training Configurations

### Training Modes

| Mode              | `attention_distillation_stage` | Description                     |
| ----------------- | ------------------------------ | ------------------------------- |
| CE Only           | `-1`                           | Standard cross-entropy training |
| KL Distillation   | `2`                            | Logit distillation from teacher |
| Hidden State + KL | `23`                           | Hidden state matching + KL      |

### Example: Training with Distillation

**Option A: Direct HuggingFace Loading (Recommended, No Conversion Needed)**

```bash
# Uses hf_path to load teacher directly from HuggingFace
python train.py \
    -c configs/atlasqwen0b5.yaml \
    -c configs/qwen0b5hfteacher.yaml \
    -c configs/distill2.yaml \
    --train.data_file data/dclm-10B.bin \
    --train.devices 1
```

The `qwen0b5hfteacher.yaml` config contains:

```yaml
train:
  teacher:
    model:
      hf_path: Qwen/Qwen2.5-0.5B-Instruct # Auto-downloads from HF!
```

**Option B: Manual PTH Conversion**

```bash
# Step 1: Download and convert Qwen teacher
huggingface-cli download Qwen/Qwen2.5-0.5B

# Find the snapshot path
QWEN_PATH=~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/*/

# Convert to PTH format
python convert_hf_to_pth.py $QWEN_PATH out/Qwen2.5-0.5B.pth

# Step 2: Train Atlas with KL distillation
python train.py \
    -c configs/atlasqwen0b5.yaml \
    -c configs/qwen0b5teacher.yaml \
    --train.attention_distillation_stage 2 \
    --train.data_file data/dclm-10B.bin \
    --train.devices 1
```

### Understanding `magic_prime`

`magic_prime` is used for pseudo-random data shuffling. It must be:

- A prime number where `magic_prime % 3 == 2`
- Approximately equal to `data_size / ctx_len`

**Good news: Pre-computed values for DCLM-10B are already in the configs!**

| ctx_len | magic_prime | Config File                  |
| ------- | ----------- | ---------------------------- |
| 512     | 2929793     | distill1.yaml, distill2.yaml |
| 1024    | 1464917     | distill3-1024.yaml           |
| 2048    | 732461      | distill3-2048.yaml           |
| 4096    | 366227      | distill3-4096.yaml           |
| 8192    | 183089      | distill3-8192.yaml           |

**For custom datasets**, calculate magic_prime:

```bash
python make_data_hf.py --examine your_data.bin
```

### Optimizer and Learning Rate

```yaml
train:
  optimizer: adamw # adamw, adam8bit, lion
  lr_init: 3e-4 # Initial learning rate
  lr_final: 1e-5 # Final learning rate
  warmup_steps: 100 # Warmup steps
  weight_decay: 0.1 # Weight decay
  beta1: 0.9
  beta2: 0.95
  gradient_clip_val: 1.0
```

Command line:

```bash
python train.py -c configs/atlasqwen0b5.yaml \
    --train.lr_init 1e-4 \
    --train.warmup_steps 200
```

---

## WandB Logging Configuration

WandB logging is enabled by default. Configure via:

### Option 1: Environment Variables

```bash
export WANDB_API_KEY="your-api-key"
export WANDB_PROJECT="atlas-training"
export WANDB_ENTITY="your-username-or-team"
```

### Option 2: Config File

Edit your config YAML:

```yaml
train:
  proj_dir: out # Local output directory
  proj_name: atlas-lmm-0b5 # WandB run name prefix
  log_every_n_steps: 10 # Logging frequency
```

### Option 3: Command Line

```bash
python train.py -c configs/atlasqwen0b5.yaml \
    --train.proj_name "my-experiment" \
    --train.log_every_n_steps 5
```

### Disable WandB

```bash
WANDB_MODE=disabled python train.py -c configs/atlasqwen0b5.yaml ...
```

---

## Multi-GPU Training

### Single Node, Multiple GPUs

```bash
python train.py -c configs/atlasqwen0b5.yaml \
    --train.devices 4 \
    --train.strategy deepspeed_stage_2
```

### Multi-Node Training

```bash
python train.py -c configs/atlasqwen0b5.yaml \
    --train.devices 8 \
    --train.num_nodes 2 \
    --train.strategy deepspeed_stage_3
```

### Available Strategies

| Strategy            | Memory Usage | Speed   | Use Case                   |
| ------------------- | ------------ | ------- | -------------------------- |
| `auto`              | High         | Fastest | Small models, single GPU   |
| `deepspeed_stage_2` | Medium       | Fast    | Multi-GPU, moderate models |
| `deepspeed_stage_3` | Low          | Slower  | Large models, limited VRAM |
| `fsdp`              | Low          | Medium  | Alternative to DeepSpeed   |

---

## HuggingFace Integration

### Saving Models for HuggingFace Hub

After training, convert the checkpoint:

```bash
# Convert PTH to safetensors
python convert_to_safetensors.py \
    out/your-run-name/rwkv-final.pth \
    hf-model/model.safetensors
```

Then copy the HF model files:

```bash
# Copy the Atlas HF package files
cp atlasqwen2/*.py hf-model/

# Create a proper config.json (or save via code)
python -c "
from atlasqwen2 import AtlasQwen2Config
config = AtlasQwen2Config(
    vocab_size=151936,
    hidden_size=896,
    num_hidden_layers=24,
    intermediate_size=4864,
    atlas_variant='lmm',
)
config.save_pretrained('hf-model')
"
```

### Loading and Using the Model

```python
from atlasqwen2 import AtlasQwen2ForCausalLM, AtlasQwen2Config
from transformers import AutoTokenizer

# Load model
model = AtlasQwen2ForCausalLM.from_pretrained("hf-model")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# Generate
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

### Modifying HuggingFace Config

Key files:

- `atlasqwen2/configuration_atlasqwen2.py` — Model hyperparameters
- `atlasqwen2/modeling_atlasqwen2.py` — Model architecture

Important config fields:

```python
AtlasQwen2Config(
    vocab_size=151936,
    hidden_size=896,           # n_embd
    intermediate_size=4864,    # dim_ffn
    num_hidden_layers=24,      # n_layer
    memory_heads=14,           # Number of memory heads
    memory_dim_head=64,        # Per-head dimension
    use_momentum=True,         # Memory momentum
    atlas_variant='lmm',       # 'lmm' or 'mal'
    sliding_window=512,        # For MAL only
)
```

---

## Evaluation & Distillation Analysis

### Understanding Distillation Stages

RADLADS uses a multi-stage distillation process. Each stage has different goals and metrics:

| Stage     | Config Value                            | What It Does               | Key Metric                |
| --------- | --------------------------------------- | -------------------------- | ------------------------- |
| Stage 0/1 | `attention_distillation_stage=0` or `1` | Attention output alignment | MSE loss (lower = better) |
| Stage 2   | `attention_distillation_stage=2`        | KL divergence on logits    | KL loss (lower = better)  |
| Stage 2+3 | `attention_distillation_stage=23`       | Hidden state + KL combined | Combined loss             |

### Monitoring Training (WandB)

During training, watch these metrics in WandB:

**Stage 0/1 (Attention Alignment)**:

- `train/loss` — Should decrease steadily
- Target: Loss should converge to a low value (model-dependent, but < 0.1 is good)

**Stage 2 (KL Distillation)**:

- `train/loss` — KL divergence loss
- `train/acc` — Token prediction accuracy
- Target: Accuracy should approach teacher's accuracy

**Stage 2+3 (Hidden State + KL)**:

- `train/loss` — Combined loss
- Watch for both KL and hidden state components decreasing

### Quick Sanity Check: Dragon Test

After training, run a quick generation test:

```bash
# Test generation quality
python dragon_test.py -c configs/atlasqwen0b5.yaml \
    --train.load_model out/your-run/rwkv-final.pth

# Compare with teacher (optional)
python dragon_test.py -c configs/qwen0b5.yaml \
    --train.load_model out/Qwen2.5-0.5B.pth
```

**What to look for**:

- Generated text should be coherent and grammatical
- Style should roughly match the teacher model
- No repetitive loops or garbage output

### Quantitative Evaluation: LM Eval Harness

Run standardized benchmarks to measure quality:

```bash
# Basic evaluation suite
python run_lm_eval.py -c configs/atlasqwen0b5.yaml \
    --train.load_model out/your-run/rwkv-final.pth \
    --eval.tasks hellaswag,winogrande,arc_easy,arc_challenge,piqa

# Compare with teacher baseline
python run_lm_eval.py -c configs/qwen0b5.yaml \
    --train.load_model out/Qwen2.5-0.5B.pth \
    --eval.tasks hellaswag,winogrande,arc_easy,arc_challenge,piqa
```

**Benchmark Interpretation**:

| Benchmark     | What It Measures       | Good Student Score   |
| ------------- | ---------------------- | -------------------- |
| HellaSwag     | Commonsense reasoning  | Within 5% of teacher |
| WinoGrande    | Coreference resolution | Within 5% of teacher |
| ARC-Easy      | Science QA (easy)      | Within 3% of teacher |
| ARC-Challenge | Science QA (hard)      | Within 5% of teacher |
| PIQA          | Physical intuition     | Within 3% of teacher |

### Perplexity Evaluation

For language modeling quality:

```bash
# Evaluate perplexity on validation set
python run_lm_eval.py -c configs/atlasqwen0b5.yaml \
    --train.load_model out/your-run/rwkv-final.pth \
    --eval.tasks wikitext
```

**Interpreting Perplexity**:

- Lower is better
- Student perplexity should be close to teacher (within 10-20% is acceptable)
- Large gap indicates distillation needs more training or tuning

### Complete Evaluation Workflow

```bash
# 1. After Stage 1 (attention alignment) completes
STAGE1_CKPT="out/atlas-stage1/rwkv-final.pth"

# Quick sanity check
python dragon_test.py -c configs/atlasqwen0b5.yaml \
    --train.load_model $STAGE1_CKPT

# 2. After Stage 2 (KL distillation) completes
STAGE2_CKPT="out/atlas-stage2/rwkv-final.pth"

# Full evaluation
python run_lm_eval.py -c configs/atlasqwen0b5.yaml \
    --train.load_model $STAGE2_CKPT \
    --eval.tasks hellaswag,winogrande,arc_easy,arc_challenge,piqa,wikitext

# 3. Compare with teacher
TEACHER_CKPT="out/Qwen2.5-0.5B.pth"
python run_lm_eval.py -c configs/qwen0b5.yaml \
    --train.load_model $TEACHER_CKPT \
    --eval.tasks hellaswag,winogrande,arc_easy,arc_challenge,piqa,wikitext
```

### Troubleshooting Poor Distillation

| Symptom                       | Likely Cause                 | Solution                            |
| ----------------------------- | ---------------------------- | ----------------------------------- |
| Stage 1 loss not decreasing   | Learning rate too low/high   | Try `lr_init` in range 1e-4 to 1e-3 |
| Stage 2 accuracy plateaus low | Stage 1 wasn't good enough   | Retrain Stage 1 longer              |
| Large teacher-student gap     | Not enough distillation data | Train longer, use more data         |
| Student generates garbage     | Distillation didn't converge | Check Stage 1 alignment first       |
| High perplexity               | Model capacity issue         | Try MAL instead of LMM              |

### Saving Evaluation Results

```bash
# Save results to file
python run_lm_eval.py -c configs/atlasqwen0b5.yaml \
    --train.load_model out/your-run/rwkv-final.pth \
    --eval.tasks hellaswag,winogrande \
    --eval.output_path results/eval_results.json
```

---

## File Structure

```
RADLADS-atlas/
├── atlasqwen2/                    # HuggingFace-compatible package
│   ├── __init__.py
│   ├── configuration_atlasqwen2.py
│   └── modeling_atlasqwen2.py
├── models/
│   ├── atlas_memory.py            # Core RNN memory implementation
│   ├── atlasqwen2_core.py         # Atlas decoder blocks
│   └── atlasqwen2.py              # Training wrapper
├── configs/
│   ├── atlasqwen0b5.yaml          # 0.5B Atlas config
│   └── ...                        # Other configs
├── tests/                         # Test suite (89 tests)
├── train.py                       # Main training script
├── convert_hf_to_pth.py           # HF → PTH converter
├── convert_to_safetensors.py      # PTH → safetensors converter
├── dragon_test.py                 # Quick inference test
├── run_lm_eval.py                 # Evaluation script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Troubleshooting

### Common Issues

**CUDA Out of Memory**

```bash
# Reduce batch size
--train.micro_bsz 2

# Use gradient accumulation instead
--train.accumulate_grad_batches 8

# Use DeepSpeed ZeRO-3
--train.strategy deepspeed_stage_3
```

**WandB Login Issues**

```bash
wandb login
# Or disable wandb
WANDB_MODE=disabled python train.py ...
```

**Checkpoint Loading Errors**

```bash
# Check checkpoint contents
python examine_ckpt.py out/your-checkpoint.pth

# Ensure config matches checkpoint
--train.load_partial true
```

---

## Configuration Reference

### Full Atlas Config Fields

```yaml
model:
  __type__: configs.Atlas_Config
  classname: atlasqwen2

  # Architecture
  vocab_size: 151936
  n_embd: 896
  n_layer: 24
  dim_ffn: 4864
  dim_att: 896
  head_size: 64
  ctx_len: 2048
  rms_norm_eps: 1e-6

  # Atlas Memory
  memory_heads: 14
  memory_dim_head: 64
  use_momentum: true
  poly_degree: 1
  poly_mode: "off" # "off", "elementwise", "tensor"
  qk_norm: true
  qkv_conv_kernel: null

  # Variant
  atlas_variant: "lmm" # "lmm" or "mal"
  sliding_window: 512 # MAL only
  tie_word_embeddings: true

train:
  # Data
  data_type: binidx
  data_file: data/dclm-10B.bin
  ctx_len: 2048

  # Batch
  micro_bsz: 4
  accumulate_grad_batches: 4

  # Optimizer
  optimizer: adamw
  lr_init: 3e-4
  lr_final: 1e-5
  warmup_steps: 100
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  adam_eps: 1e-8
  gradient_clip_val: 1.0

  # Training mode
  attention_distillation_stage: -1 # -1=CE, 2=KL

  # Output
  proj_dir: out
  proj_name: atlas-lmm
  epoch_save: 1
  log_every_n_steps: 10

  # Hardware
  accelerator: gpu
  devices: 1
  num_nodes: 1
  precision: bf16
  strategy: auto
```

---

## Citation

If you use this code, please cite:

```bibtex
@misc{goldstein2025radlads,
    title={RADLADS: Rapid Attention Distillation to Linear Attention Decoders at Scale},
    author={Daniel Goldstein and Eric Alcaide and Janna Lu and Eugene Cheah},
    year={2025},
    eprint={2505.03005},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2505.03005},
}

@misc{behrouz2025atlas,
    title={ATLAS: Learning to Optimally Memorize the Context at Test Time},
    author={Ali Behrouz and Zeman Li and Praneeth Kacham and Majid Daliri and Yuan Deng and Peilin Zhong and Meisam Razaviyayn and Vahab Mirrokni},
    year={2025},
    eprint={2505.23735},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2505.23735},
}
```

---

## License

This project is licensed under the same terms as the original RADLADS repository.
