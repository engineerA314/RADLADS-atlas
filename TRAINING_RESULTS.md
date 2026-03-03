# RADLADS Training Results Comparison

## Overview

This document summarizes the training results from three different model variants trained through the RADLADS distillation process:

1. **Atlas-LMM 0.5B (Polysketch)** (from RADLADS-atlas): Pure memory-based RNN using Atlas-LMM architecture with Polysketch (trainable Taylor-series) kernel approximation
2. **RWKV 0.5B** (from RADLADS-paper): RWKV6/7 architecture variant
3. **RWKV 7B** (from RADLADS-paper): Larger RWKV6/7 architecture variant

All models are derived from **Qwen2.5 Instruct** base models through a 3-stage distillation process.

All three model variants have been successfully trained through all 3 stages.

---

## Performance Evaluation

All metrics are reported as lm-eval-harness scores (same scale as the Hugging Face benchmark tables).

### Baseline Models

| Model                     | lmbda  | mmlu   | arc_c  | arc_e  | hella  | piqa   | winog  |
| ------------------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| **Qwen2.5-0.5B-Instruct** | 0.4968 | 0.4582 | 0.3020 | 0.6566 | 0.4062 | 0.7051 | 0.5541 |
| **Qwen2.5-7B-Instruct**   | 0.6949 | 0.7174 | 0.5282 | 0.8131 | 0.6195 | 0.7938 | 0.7119 |

Source: BF16 scores from Hugging Face eval tables for `OPEA/Qwen2.5-0.5B-Instruct-int4-sym-inc` and `OPEA/Qwen2.5-7B-Instruct-int4-sym-inc`.

### Atlas-LMM 0.5B (Polysketch) Results

| Checkpoint                    | lmbda  | mmlu   | arc_c  | arc_e  | hella  | piqa   | winog  |
| ----------------------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| **Stage 1 (Attention Align)** | 0.0576 | 0.2478 | 0.2261 | 0.4028 | 0.2833 | 0.5658 | 0.4980 |
| **Stage 2 (KL Distill)**      | 0.4207 | 0.2649 | 0.2713 | 0.6178 | 0.3718 | 0.6882 | 0.5375 |
| **Stage 3 (Long Context)**    | 0.4244 | 0.2621 | 0.2739 | 0.6124 | 0.3725 | 0.6877 | 0.5343 |

### RWKV 0.5B Results

| Checkpoint                    | lmbda  | mmlu   | arc_c  | arc_e  | hella  | piqa   | winog  |
| ----------------------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| **Stage 1 (Attention Align)** | 0.0631 | 0.3034 | 0.1988 | 0.2917 | 0.3081 | 0.5441 | 0.5051 |
| **Stage 2 (KL Distill)**      | 0.5973 | 0.3748 | 0.3089 | 0.6397 | 0.3879 | 0.6991 | 0.5620 |
| **Stage 3 (Long Context)**    | 0.5828 | 0.3577 | 0.2969 | 0.6435 | 0.3798 | 0.6942 | 0.5533 |

### RWKV 7B Results

| Checkpoint                    | lmbda  | mmlu   | arc_c  | arc_e  | hella  | piqa   | winog  |
| ----------------------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| **Stage 1 (Attention Align)** | 0.6175 | 0.6543 | 0.5128 | 0.8068 | 0.6033 | 0.7894 | 0.6559 |
| **Stage 2 (KL Distill)**      | 0.6905 | 0.6821 | 0.5597 | 0.8346 | 0.6076 | 0.7992 | 0.7356 |
| **Stage 3 (Long Context)**    | 0.7058 | 0.6825 | 0.5606 | 0.8350 | 0.5964 | 0.8101 | 0.7372 |

### Atlas-LMM 0.5B vs RWKV 0.5B Comparison (Stage 2/3)

| Benchmark | RWKV 0.5B (S2) | Polysketch (S3) | Recovery Rate |
| --------- | -------------- | --------------- | ------------- |
| lmbda     | 0.5973         | 0.4244          | 71%           |
| mmlu      | 0.3748         | 0.2621          | 70%           |
| arc_c     | 0.3089         | 0.2739          | 89%           |
| arc_e     | 0.6397         | 0.6124          | 96%           |
| hella     | 0.3879         | 0.3725          | 96%           |
| piqa      | 0.6991         | 0.6877          | 98%           |
| winog     | 0.5620         | 0.5343          | 95%           |

**Visualization**:

![Eval Comparison](../wandb_plots/eval_comparison.png)

### Benchmark Descriptions

- **lmbda (lambada_openai)**: Language modeling accuracy on the LAMBADA dataset
- **mmlu**: Massive Multitask Language Understanding - measures general knowledge and reasoning
- **arc_c (arc_challenge)**: AI2 Reasoning Challenge (difficult questions)
- **arc_e (arc_easy)**: AI2 Reasoning Challenge (easier questions)
- **hella (hellaswag)**: Commonsense reasoning through sentence completion
- **piqa**: Physical Interaction QA - tests physical commonsense reasoning
- **winog (winogrande)**: Coreference resolution and reasoning

---

## Training Target vs Achieved

### Stage 1 Targets

| Metric          | Target | Achieved                | Status             |
| --------------- | ------ | ----------------------- | ------------------ |
| Loss            | ~0.025 | 0.0549 (last 100 avg)   | **미달 (2.2x)**    |
| Token Accuracy  | ~0.25  | 0.2299 (Stage 2 초기값) | **근접 (92%)**     |
| KL Divergence   | ~1.7   | 1.8108 (Stage 2 초기값) | **근접 (6% 초과)** |
| Eval benchmarks | N/A    | Random 수준             | Expected           |

**Notes**: Token accuracy와 KL divergence는 Stage 2 시작 시점의 첫 step 값으로 측정 (Stage 1 학습 결과를 반영). Token accuracy 0.23은 목표 0.25에 근접. KL divergence 1.81은 목표 1.7보다 소폭 높으나 근접. Loss는 목표(0.025)보다 높은 0.055에서 수렴.

### Stage 2 Targets

| Metric         | Target | Achieved              | Status           |
| -------------- | ------ | --------------------- | ---------------- |
| Loss           | ~0.06  | 0.2191 (last 100 avg) | **미달 (3.7x)**  |
| Token Accuracy | ~0.37  | 0.3697 (last 100 avg) | **근접 (99.9%)** |
| lmbda          | 0.50+  | 0.4207                | **미달 (84%)**   |
| mmlu           | 0.35+  | 0.2649                | **미달 (76%)**   |

**Notes**: Token accuracy는 목표에 근접 달성 (0.37 vs 0.3697). 그러나 loss는 목표(0.06)의 3.7배인 0.22에 수렴하여 큰 gap. lmbda와 mmlu 모두 미달이며, 특히 mmlu는 RWKV 0.5B baseline(0.3748) 대비 70% 수준. Stage 2 실제 학습은 `polysketch-stage2-gradcp0-20260202-135934Z` run에서 수행됨 (10174 rows, 500M tokens).

### Stage 3 (Long Context Extension)

| Metric | Stage 2 → Stage 3 | Delta |
| ------ | ----------------- | ----- |
| lmbda  | 0.4207 → 0.4244   | +0.4% |
| arc_c  | 0.2713 → 0.2739   | +0.3% |
| hella  | 0.3718 → 0.3725   | +0.2% |
| piqa   | 0.6882 → 0.6877   | -0.1% |
| arc_e  | 0.6178 → 0.6124   | -0.9% |

**Notes**: Stage 3 (ctx16384 long-context extension)은 대부분의 지표에서 미세한 변화만 있었으며, catastrophic forgetting 없이 안정적으로 학습됨.

---

## Running Evaluations

### Prerequisites

```bash
# Install evaluation dependencies
pip install lm_eval --upgrade

# Ensure you're in the RADLADS-paper directory
cd /path/to/RADLADS-paper
```

### Evaluation Commands

#### For Atlas-LMM 0.5B (Polysketch) Models

**Using the custom evaluation script:**

```bash
# Evaluate a checkpoint
python run_lm_eval.py \
    -c configs/atlasqwen0b5.yaml \
    --path /path/to/checkpoint.pth \
    --tasks lambada_openai,mmlu,arc_challenge,arc_easy,hellaswag,piqa,winogrande \
    --bsz 48 \
    --precision bf16
```

**Example for each stage:**

```bash
# Stage 1 checkpoint
python run_lm_eval.py \
    -c configs/atlasqwen0b5.yaml \
    --path out/atlas-lmm-0b5-stage1/rwkv-final.pth \
    --tasks lambada_openai,mmlu,arc_challenge,arc_easy,hellaswag,piqa,winogrande \
    --bsz 48

# Stage 2 checkpoint
python run_lm_eval.py \
    -c configs/atlasqwen0b5.yaml \
    --path out/atlas-lmm-0b5-stage2/rwkv-final.pth \
    --tasks lambada_openai,mmlu,arc_challenge,arc_easy,hellaswag,piqa,winogrande \
    --bsz 48

# Stage 3 checkpoint
python run_lm_eval.py \
    -c configs/atlasqwen0b5.yaml \
    --path out/atlas-lmm-0b5-stage3/rwkv-final.pth \
    --tasks lambada_openai,mmlu,arc_challenge,arc_easy,hellaswag,piqa,winogrande \
    --bsz 48
```

#### For RWKV 0.5B Models

```bash
# Evaluate RWKV 0.5B checkpoint
python run_lm_eval.py \
    -c configs/qwerky6.yaml \
    --path /path/to/checkpoint.pth \
    --tasks lambada_openai,mmlu,arc_challenge,arc_easy,hellaswag,piqa,winogrande \
    --bsz 48 \
    --precision bf16
```

**Example for each stage:**

```bash
# Stage 1 checkpoint
python run_lm_eval.py \
    -c configs/qwerky6.yaml \
    --path out/qwerky6-stage1/rwkv-final.pth \
    --tasks lambada_openai,mmlu,arc_challenge,arc_easy,hellaswag,piqa,winogrande \
    --bsz 48

# Stage 2 checkpoint
python run_lm_eval.py \
    -c configs/qwerky6.yaml \
    --path out/qwerky6-stage2/rwkv-final.pth \
    --tasks lambada_openai,mmlu,arc_challenge,arc_easy,hellaswag,piqa,winogrande \
    --bsz 48

# Stage 3 checkpoint
python run_lm_eval.py \
    -c configs/qwerky6.yaml \
    --path out/qwerky6-stage3/rwkv-final.pth \
    --tasks lambada_openai,mmlu,arc_challenge,arc_easy,hellaswag,piqa,winogrande \
    --bsz 48
```

#### For RWKV 7B Models

```bash
# Evaluate RWKV 7B checkpoint
python run_lm_eval.py \
    -c configs/qwerky7.yaml \
    --path /path/to/checkpoint.pth \
    --tasks lambada_openai,mmlu,arc_challenge,arc_easy,hellaswag,piqa,winogrande \
    --bsz 24 \
    --precision bf16
```

**Example for each stage:**

```bash
# Stage 1 checkpoint
python run_lm_eval.py \
    -c configs/qwerky7.yaml \
    --path out/qwerky7-stage1/rwkv-final.pth \
    --tasks lambada_openai,mmlu,arc_challenge,arc_easy,hellaswag,piqa,winogrande \
    --bsz 24

# Stage 2 checkpoint
python run_lm_eval.py \
    -c configs/qwerky7.yaml \
    --path out/qwerky7-stage2/rwkv-final.pth \
    --tasks lambada_openai,mmlu,arc_challenge,arc_easy,hellaswag,piqa,winogrande \
    --bsz 24

# Stage 3 checkpoint
python run_lm_eval.py \
    -c configs/qwerky7.yaml \
    --path out/qwerky7-stage3/rwkv-final.pth \
    --tasks lambada_openai,mmlu,arc_challenge,arc_easy,hellaswag,piqa,winogrande \
    --bsz 24
```

#### For Baseline Qwen2.5 Models

**Option 1: Using HuggingFace evaluation (recommended for baseline):**

```bash
# Qwen2.5-0.5B-Instruct baseline
python run_lm_eval_hf.py \
    -c configs/qwen0b5hf.yaml \
    --path Qwen/Qwen2.5-0.5B-Instruct \
    --tasks lambada_openai,mmlu,arc_challenge,arc_easy,hellaswag,piqa,winogrande \
    --bsz 48

# Qwen2.5-7B-Instruct baseline
python run_lm_eval_hf.py \
    -c configs/qwen7bhf.yaml \
    --path Qwen/Qwen2.5-7B-Instruct \
    --tasks lambada_openai,mmlu,arc_challenge,arc_easy,hellaswag,piqa,winogrande \
    --bsz 24
```

**Option 2: Using converted PTH checkpoints:**

```bash
# First convert HF model to PTH if needed
python convert_hf_to_pth.py \
    ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*/ \
    out/Qwen2.5-0.5B-Instruct.pth

# Then evaluate
python run_lm_eval.py \
    -c configs/qwen0b5.yaml \
    --path out/Qwen2.5-0.5B-Instruct.pth \
    --tasks lambada_openai,mmlu,arc_challenge,arc_easy,hellaswag,piqa,winogrande \
    --bsz 48
```

### Notes on Evaluation

- **Batch size (`--bsz`)**: Larger values speed up evaluation but require more GPU memory. Adjust based on your GPU capacity.
  - 0.5B models: `--bsz 48` or higher on 24GB+ GPUs
  - 7B models: `--bsz 24` or lower depending on GPU memory
- **Precision**: Use `--precision bf16` for best performance on modern GPUs
- **Tasks**: You can evaluate individual tasks by specifying just one, e.g., `--tasks lambada_openai`
- **Results**: The script outputs detailed results in JSON format at the end of evaluation

### Saving Results

To save evaluation results to a file for later analysis:

```bash
python run_lm_eval.py \
    -c configs/atlasqwen0b5.yaml \
    --path out/atlas-lmm-0b5-atlas-stage2/rwkv-final.pth \
    --tasks lambada_openai,mmlu,arc_challenge,arc_easy,hellaswag,piqa,winogrande \
    --bsz 48 \
    > results/atlas-0b5-stage2-eval.json
```

---

## Training Curves Visualization

### Stage 1: Attention Alignment

**Description**: Compares the attention alignment loss curves across all three models during Stage 1 training. This stage aligns the student model's attention outputs with the teacher's softmax attention.

**Expected Trends**:

- Loss should decrease steadily
- Atlas-LMM may show different convergence patterns due to its pure memory-based architecture
- RWKV models should show similar patterns but potentially faster convergence

**Visualization**:

![Stage 1 Attention Alignment Loss](../wandb_plots/stage1_attention_loss.png)

- X-axis: Training tokens (billions)
- Y-axis: Attention alignment loss
- Lines: Atlas-LMM 0.5B Polysketch (blue), RWKV 0.5B (orange), RWKV 7B (green)
- Atlas run used: `2026-01-30-03-49-58 | ablation-detach-poly2-polymodepolysketch-qknorm0-qkvconvnone-rope0-gn0-stage1/20260130-034819Z`

### Stage 2: KL Divergence Loss

**Description**: Compares KL divergence loss between teacher and student logits during Stage 2 training.

**Expected Trends**:

- KL loss should decrease as the student learns to match teacher predictions
- Larger models (7B) typically achieve lower final KL loss
- Convergence speed varies by architecture

**Visualization**:

![Stage 2 KL Divergence Loss](../wandb_plots/stage2_kl_loss.png)

- X-axis: Training tokens (billions)
- Y-axis: KL divergence loss
- Lines: Atlas-LMM 0.5B Polysketch (blue), RWKV 0.5B (orange), RWKV 7B (green)
- Atlas run used: `2026-02-02-13-59-43 | polysketch-stage2-gradcp0-20260202-135934Z`
- Note: 사용자가 지정한 Stage 2 WandB run (`2026-01-30-05-09-10`)은 1 row만 기록됨 (첫 스텝에서 중단). 실제 학습 데이터는 위 run에서 가져옴.

### Stage 2: Token Prediction Accuracy

**Description**: Token-level prediction accuracy during Stage 2 training (matches teacher's top-1 prediction).

**Expected Trends**:

- Accuracy should increase and plateau near teacher performance
- Architecture differences may lead to different final accuracy levels

**Visualization**:

![Stage 2 Token Prediction Accuracy](../wandb_plots/stage2_token_accuracy.png)

- X-axis: Training tokens (billions)
- Y-axis: Token accuracy
- Lines: Atlas-LMM 0.5B Polysketch (blue), RWKV 0.5B (orange), RWKV 7B (green)
- Atlas run used: `2026-02-02-13-59-43 | polysketch-stage2-gradcp0-20260202-135934Z`

### Stage 3: Cross-Entropy Loss

**Description**: Standard language modeling loss during Stage 3 long-context training (ctx16384).

**Expected Trends**:

- Loss decreases as models adapt to longer context lengths
- May show step increases when context length is increased (if using progressive training)
- Final loss indicates model's language modeling capability

**Visualization**:

![Stage 3 Cross-Entropy Loss](../wandb_plots/stage3_ce_loss.png)

- X-axis: Training steps (models trained with different token counts)
- Y-axis: Cross-entropy loss
- Lines: Atlas-LMM 0.5B Polysketch (blue), RWKV 0.5B (orange), RWKV 7B (green)
- Atlas run used: `2026-02-03-06-56-37 | atlas-stage3-ctx16384`
- Note: Atlas Stage 3은 `train/tokens`를 micro-step 단위로 기록하므로, 플롯에서 acc_grad=6을 곱하여 실제 처리 토큰 수로 보정함. 65 optimizer steps × 1.57M tokens/step ≈ 100M tokens.

### Accessing Training Curves from WandB

To generate these comparison plots:

1. **Export data from WandB**:

   ```bash
   # Using wandb CLI
   wandb export --entity YOUR_ENTITY --project YOUR_PROJECT --run RUN_NAME
   ```

2. **Or use the provided script**:

   ```bash
   # From the RADLADS-atlas directory
   python scripts/update_wandb_plots.py
   ```

   This script fetches data from W&B and generates all training curve plots automatically.

3. **Key metrics to track**:
   - **Stage 1**: `train/loss` (attention alignment loss)
   - **Stage 2**: `train/loss` (KL divergence), `train/acc` (token accuracy)
   - **Stage 3**: `train/loss` (cross-entropy loss)

---

## Analysis and Observations

### Stage 1 (Attention Alignment)

- 대부분의 벤치마크에서 random 수준 성능 (lambada: 5.8%)
- MMLU는 random baseline (25%) 수준
- Stage 1은 attention output alignment만 수행하므로, eval 벤치마크 수치는 의미 없음
- 이 단계에서는 loss convergence와 kl divergence 추이가 중요

### Stage 2 → Stage 3 변화

| Metric | Stage 2 | Stage 3 | Delta |
| ------ | ------- | ------- | ----- |
| lmbda  | 0.4207  | 0.4244  | +0.4% |
| arc_c  | 0.2713  | 0.2739  | +0.3% |
| hella  | 0.3718  | 0.3725  | +0.2% |
| piqa   | 0.6882  | 0.6877  | -0.1% |
| arc_e  | 0.6178  | 0.6124  | -0.9% |

대부분 지표에서 미세한 개선 또는 유지. Long-context extension이 short-context 성능을 해치지 않음을 확인.

### Polysketch vs RWKV 0.5B Baseline

Polysketch Stage 3 모델은 RWKV 0.5B baseline (Stage 2) 대비:

- **piqa**: 98% 수준 (0.6877 vs 0.6991) — 거의 동등
- **arc_easy**: 96% 수준 (0.6124 vs 0.6397) — 근접
- **hellaswag**: 96% 수준 (0.3725 vs 0.3879) — 근접
- **arc_challenge**: 89% 수준 (0.2739 vs 0.3089) — 다소 gap
- **lambada**: 71% 수준 (0.4244 vs 0.5973) — 큰 gap
- **mmlu**: 70% 수준 (0.2621 vs 0.3748) — 큰 gap

### Key Findings

1. **Training stability**: Polysketch 모델은 3단계 모두 안정적으로 학습 완료. 이전 실험에서 Stage 3에서 gradient explosion이 발생했던 문제가 해결됨.
2. **Convergence**: Stage 2에서 KL divergence가 안정적으로 수렴하며, token accuracy도 일정 수준에 도달.
3. **Final performance**: RWKV baseline 대비 piqa/arc_easy/hellaswag에서 95%+ 성능 달성. Lambada와 MMLU가 가장 큰 gap (각각 71%, 70%).
4. **Architecture-specific observations**: Polysketch kernel approximation은 commonsense reasoning (piqa, hellaswag) 작업에서 softmax attention에 가깝게 근접하지만, 장거리 문맥 의존성(lambada)과 지식 회상(mmlu)에서 gap이 큼. 이는 Taylor-series polynomial approximation의 expressiveness 한계를 시사.
5. **Long-context extension**: Stage 3 (ctx 512→16384) 확장 시 catastrophic forgetting 없이 안정적. Short-context 벤치마크 성능이 거의 유지됨.

---

## Appendix

### Model Configurations

**Atlas-LMM 0.5B (Polysketch)**:

- Configuration file: `configs/atlasqwen0b5.yaml`
- Architecture: Pure memory-based (no traditional attention), Polysketch kernel
- Hidden size: 896
- Layers: 24
- Memory heads: 14
- Omega window: 16
- Poly degree: 2

**RWKV 0.5B**:

- Configuration file: `configs/qwerky6.yaml`
- Architecture: RWKV6 or RWKV7 variant
- Hidden size: 896
- Layers: 24

**RWKV 7B**:

- Configuration file: `configs/qwerky7.yaml`
- Architecture: RWKV6 or RWKV7 variant
- Hidden size: 3584
- Layers: 28

### Training Hyperparameters

**Stage 1**:

- Tokens trained: `100M` (gcp launch defaults)
- Sequence length: `512`
- Optimizer: AdamW (dsfusedadamw in RADLADS-paper)
- Learning rate: `1e-3` → `1e-5` (cosine decay)
- Adam betas / eps: `0.9, 0.95` / `1e-8`
- Batch size (global): `32` = `micro_bsz 4` × `acc_grad 1` × `devices 8` (RWKV launch defaults)

**Stage 2**:

- Tokens trained: `500M` (gcp launch defaults)
- Sequence length: `512`
- Optimizer: AdamW (dsfusedadamw in RADLADS-paper)
- Learning rate: `1e-5` → `1e-5` (decay: `none`)
- Adam betas / eps: `0.9, 0.95` / `1e-8`
- Batch size (global): `96` = `micro_bsz 12` × `acc_grad 1` × `devices 8`

**Stage 3**:

- Tokens trained: `100M`
- Context length: `16384`
- Optimizer: AdamW (dsfusedadamw in RADLADS-paper)
- Learning rate: `1e-5` → `1e-5` (decay: `none`)
- Adam betas / eps: `0.9, 0.95` / `1e-8`
- Batch size (global): `96` in RADLADS-paper (`micro_bsz 12` × `acc_grad 1` × `devices 8`); RADLADS-atlas Stage 3 uses `micro_bsz 2`, `acc_grad 6`, `devices 8` → `96`
