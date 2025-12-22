#!/bin/bash
# Run RADLADS Paper Stage 1: Attention Alignment (HF patching)
#
# Paper Stage 1 uses attention_distillation_stage=1 (no separate teacher model).
# This script uses atlas_distill1.yaml which patches HF Qwen2 with Atlas attention (AtlasSelfAttention).
# 
# Key knobs (optional, via env):
# - Dataset: DATA_PREFIX (default: data/dclm-10B)
# - Context: CTX_LEN (default: 512)
# - Training length: TOKENS (default: 100000000)
# - Parallelism: DEVICES (default: 1), NUM_NODES (default: 1)
# - Batch: MICRO_BSZ (auto default), ACC_GRAD (default: 1)
# - Strategy: STRATEGY (default: auto), GRAD_CP (default: 0), DS_BUCKET_MB (default: 200)
# - W&B: WANDB_PROJECT (empty disables), WANDB_MODE (defaults to disabled when WANDB_PROJECT empty)
# - Architecture overrides (optional): OMEGA_WINDOW, USE_OMEGA_GATE, USE_MOMENTUM, POLY_MODE, ATLAS_VARIANT, SLIDING_WINDOW
# - Ablations (optional): WITH_ALL_ABLATION=1 or individual overrides (POLY_DEGREE, QK_NORM, QKV_CONV_KERNEL, USE_ROPE, USE_GROUPNORM)

set -euo pipefail

echo "=============================================="
echo "RADLADS Paper Stage 1: Attention Alignment"
echo "=============================================="

usage() {
    echo "Usage:"
    echo "  bash scripts/run_stage1.sh [tokens] [devices] [ctx_len]"
    echo ""
    echo "Examples:"
    echo "  bash scripts/run_stage1.sh 100000000 1 512"
    echo "  WANDB_PROJECT=radlads-atlas bash scripts/run_stage1.sh 100000000 8 512"
    echo "  WITH_ALL_ABLATION=1 bash scripts/run_stage1.sh 100000000 8 512"
    echo ""
    echo "Env overrides:"
    echo "  DATA_PREFIX, CTX_LEN, TOKENS, DEVICES, NUM_NODES, MICRO_BSZ, ACC_GRAD, STRATEGY, GRAD_CP, DS_BUCKET_MB,"
    echo "  WANDB_PROJECT, WANDB_MODE,"
    echo "  OMEGA_WINDOW, USE_OMEGA_GATE, USE_MOMENTUM, POLY_MODE, ATLAS_VARIANT, SLIDING_WINDOW,"
    echo "  WITH_ALL_ABLATION, POLY_DEGREE, QK_NORM, QKV_CONV_KERNEL, USE_ROPE, USE_GROUPNORM"
}

TOKENS="${TOKENS:-${1:-100000000}}"
DEVICES="${DEVICES:-${2:-1}}"
CTX_LEN="${CTX_LEN:-${3:-512}}"
NUM_NODES="${NUM_NODES:-1}"
ACC_GRAD="${ACC_GRAD:-1}"
STRATEGY="${STRATEGY:-auto}"
GRAD_CP="${GRAD_CP:-0}"
DS_BUCKET_MB="${DS_BUCKET_MB:-200}"
DATA_PREFIX="${DATA_PREFIX:-data/dclm-10B}"
export DATA_PREFIX
export CTX_LEN

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

# Auto defaults for batch sizing (safe-ish; override explicitly for production)
if [ -z "${MICRO_BSZ:-}" ]; then
    if [ "${DEVICES}" -ge 8 ]; then
        MICRO_BSZ=2
    else
        MICRO_BSZ=1
    fi
else
    MICRO_BSZ="${MICRO_BSZ}"
fi

# Check dataset exists
if [ ! -f "${DATA_PREFIX}.idx" ] || [ ! -f "${DATA_PREFIX}.bin" ]; then
    echo "❌ Data not found: ${DATA_PREFIX}.idx/.bin"
    echo "   If you want DCLM-10B, run: bash scripts/download_dclm.sh"
    exit 1
fi

# Compute data_size + magic_prime for (DATA_PREFIX, CTX_LEN)
read -r DATA_SIZE DATASET_SLOT MAGIC_PRIME MAGIC_RATIO < <(
python3 - <<'PY'
import os, sys, math, random
random.seed(1234)
sys.path.insert(0, os.getcwd())
from src.binidx import MMapIndexedDataset
from src.utils import MaybeIsPrime

prefix = os.environ["DATA_PREFIX"]
ctx_len = int(os.environ["CTX_LEN"])

data = MMapIndexedDataset(prefix)
data_size = len(data._bin_buffer) // data._index._dtype_size
dataset_slot = data_size // ctx_len

if dataset_slot < 10:
    raise SystemExit(f"dataset_slot too small ({dataset_slot}); use a larger dataset or smaller ctx_len")

low = int(math.floor(0.99 * dataset_slot)) + 1
p = dataset_slot - 1  # safer than ==dataset_slot (avoid read-past-end)
while p >= low:
    if (p % 3 == 2) and MaybeIsPrime(p):
        break
    p -= 1
if p < low:
    raise SystemExit(f"could not find valid magic_prime in [{low}, {dataset_slot-1}] for ctx_len={ctx_len}, data_size={data_size}")

# Safety: max offset read in MyDataset must not exceed token buffer.
max_offset = (p - 1) * ctx_len
req_len = ctx_len + 1
if max_offset + req_len > data_size:
    raise SystemExit(f"magic_prime={p} invalid: max_offset({max_offset})+req_len({req_len}) > data_size({data_size})")

ratio = p / dataset_slot
print(data_size, dataset_slot, p, f"{ratio:.6f}")
PY
)

if [ "${TOKENS}" -gt "${DATA_SIZE}" ]; then
    echo "❌ TOKENS (${TOKENS}) must be <= data_size (${DATA_SIZE}) for DATA_PREFIX=${DATA_PREFIX}"
    exit 1
fi

# W&B control: enable only when WANDB_PROJECT is non-empty
WANDB_PROJECT="${WANDB_PROJECT:-}"
if [ -z "${WANDB_PROJECT}" ]; then
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    TRAIN_WANDB_ARG="--train.wandb"
    TRAIN_WANDB_VAL=""
else
    export WANDB_MODE="${WANDB_MODE:-online}"
    TRAIN_WANDB_ARG="--train.wandb"
    TRAIN_WANDB_VAL="${WANDB_PROJECT}"
fi

# Ablation knobs (must match across Stage 1/2/3 for checkpoint compatibility)
WITH_ALL_ABLATION="${WITH_ALL_ABLATION:-0}"
ABLATION_ARGS=()
if [ "${WITH_ALL_ABLATION}" = "1" ]; then
    ABLATION_ARGS+=(--model.poly_degree 3 --model.qk_norm true --model.qkv_conv_kernel 4 --model.use_rope true --model.use_groupnorm true)
else
    if [ -n "${POLY_DEGREE:-}" ]; then ABLATION_ARGS+=(--model.poly_degree "${POLY_DEGREE}"); fi
    if [ -n "${QK_NORM:-}" ]; then ABLATION_ARGS+=(--model.qk_norm "${QK_NORM}"); fi
    if [ -n "${QKV_CONV_KERNEL:-}" ]; then ABLATION_ARGS+=(--model.qkv_conv_kernel "${QKV_CONV_KERNEL}"); fi
    if [ -n "${USE_ROPE:-}" ]; then ABLATION_ARGS+=(--model.use_rope "${USE_ROPE}"); fi
    if [ -n "${USE_GROUPNORM:-}" ]; then ABLATION_ARGS+=(--model.use_groupnorm "${USE_GROUPNORM}"); fi
fi

# Additional architecture overrides (keep consistent across Stage 1/2/3)
MODEL_ARGS=()
if [ -n "${OMEGA_WINDOW:-}" ]; then MODEL_ARGS+=(--model.omega_window "${OMEGA_WINDOW}"); fi
if [ -n "${USE_OMEGA_GATE:-}" ]; then MODEL_ARGS+=(--model.use_omega_gate "${USE_OMEGA_GATE}"); fi
if [ -n "${USE_MOMENTUM:-}" ]; then MODEL_ARGS+=(--model.use_momentum "${USE_MOMENTUM}"); fi
if [ -n "${POLY_MODE:-}" ]; then MODEL_ARGS+=(--model.poly_mode "${POLY_MODE}"); fi
if [ -n "${ATLAS_VARIANT:-}" ]; then MODEL_ARGS+=(--model.atlas_variant "${ATLAS_VARIANT}"); fi
if [ -n "${SLIDING_WINDOW:-}" ]; then MODEL_ARGS+=(--model.sliding_window "${SLIDING_WINDOW}"); fi

echo "Configuration:"
echo "  Config: configs/atlas_distill1.yaml (HF Qwen2 + Atlas patch)"
echo "  DATA_PREFIX: ${DATA_PREFIX} (size=${DATA_SIZE} tokens, slot=${DATASET_SLOT}, magic_prime=${MAGIC_PRIME}, ratio=${MAGIC_RATIO})"
echo "  CTX_LEN: ${CTX_LEN}"
echo "  TOKENS (my_exit_tokens): ${TOKENS}"
echo "  DEVICES x NUM_NODES: ${DEVICES} x ${NUM_NODES}"
echo "  MICRO_BSZ: ${MICRO_BSZ}   ACC_GRAD: ${ACC_GRAD}"
echo "  STRATEGY: ${STRATEGY}   GRAD_CP: ${GRAD_CP}   DS_BUCKET_MB: ${DS_BUCKET_MB}"
echo "  WANDB_MODE: ${WANDB_MODE}   WANDB_PROJECT: ${WANDB_PROJECT:-<disabled>}"
if [ "${WITH_ALL_ABLATION}" = "1" ]; then
    echo "  Ablation: WITH_ALL_ABLATION=1 (poly_degree=3, qk_norm=true, qkv_conv_kernel=4, use_rope=true, use_groupnorm=true)"
else
    echo "  Ablation: custom overrides only (if provided)"
fi
echo ""

# Run training
python3 train.py \
    -c configs/atlas_distill1.yaml \
    --train.data_file "${DATA_PREFIX}" \
    --train.my_exit_tokens "${TOKENS}" \
    --train.magic_prime "${MAGIC_PRIME}" \
    --train.micro_bsz "${MICRO_BSZ}" \
    --train.accumulate_grad_batches "${ACC_GRAD}" \
    --train.devices "${DEVICES}" \
    --train.num_nodes "${NUM_NODES}" \
    --train.train_stage 2 \
    --train.attention_distillation_stage 1 \
    --train.load_model "" \
    --train.strategy "${STRATEGY}" \
    --train.grad_cp "${GRAD_CP}" \
    --train.ds_bucket_mb "${DS_BUCKET_MB}" \
    "${TRAIN_WANDB_ARG}" "${TRAIN_WANDB_VAL}" \
    --train.proj_suffix atlas-stage1 \
    --model.ctx_len "${CTX_LEN}" \
    "${MODEL_ARGS[@]}" \
    "${ABLATION_ARGS[@]}"

echo ""
echo "✅ Paper Stage 1 complete!"
echo ""
echo "Next step (Paper Stage 2):"
echo "  bash scripts/run_stage2.sh <stage1_checkpoint.pth>"
