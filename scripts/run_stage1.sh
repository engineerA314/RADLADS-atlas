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
# - Memory overrides (optional): USE_ACCELERATED_SCAN, MEMORY_SCAN_CHUNK_LEN
# - Ablations (optional): WITH_ALL_ABLATION=1 or individual overrides (POLY_DEGREE, QK_NORM, QKV_CONV_KERNEL, USE_ROPE, USE_GROUPNORM)
# - Stage1 config selection: STAGE1_CONFIG (default: configs/atlas_distill1.yaml)
# - Early stopping (optional, via env → CLI):
#   EARLY_STOP=1
#   EARLY_STOP_MIN_STEPS=0
#   EARLY_STOP_PATIENCE_STEPS=0
#   EARLY_STOP_MIN_DELTA=0.0
#   EARLY_STOP_EMA_ALPHA=0.05
#   EARLY_STOP_LOSS_CLIP=10.0 (or empty to disable clipping)
#   EARLY_STOP_TARGET_LOSS=0.1 (or empty to disable target-stop)
#   EARLY_STOP_TARGET_PATIENCE_STEPS=200

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
    echo "  USE_ACCELERATED_SCAN, MEMORY_SCAN_CHUNK_LEN,"
    echo "  WITH_ALL_ABLATION, POLY_DEGREE, QK_NORM, QKV_CONV_KERNEL, USE_ROPE, USE_GROUPNORM,"
    echo "  STAGE1_CONFIG,"
    echo ""
    echo "Paper-style init checkpoint (recommended for reproducibility):"
    echo "  AUTO_PREPARE_STAGE1_WEIGHTS=1   (default: 1; downloads HF model and saves .pth if missing)"
    echo "  STAGE1_HF_ID=<hf_id_or_local_dir> (default: extracted from STAGE1_CONFIG hf_path, else Qwen/Qwen2.5-0.5B-Instruct)"
    echo "  STAGE1_INIT_CKPT=<path.pth>     (default: out/qwen2-0B5.pth)"
    echo "  EARLY_STOP, EARLY_STOP_MIN_STEPS, EARLY_STOP_PATIENCE_STEPS, EARLY_STOP_MIN_DELTA, EARLY_STOP_EMA_ALPHA,"
    echo "  EARLY_STOP_LOSS_CLIP, EARLY_STOP_TARGET_LOSS, EARLY_STOP_TARGET_PATIENCE_STEPS"
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

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
STAGE1_CONFIG="${STAGE1_CONFIG:-configs/atlas_distill1.yaml}"

# Full auto-resume (FULL checkpoint: optimizer + step counters) from a single "latest" checkpoint in GCS.
FULL_AUTO_RESUME="${FULL_AUTO_RESUME:-1}"
FULL_RESUME_LOCAL_DIR="${FULL_RESUME_LOCAL_DIR:-out/_full_resume/${EXP_ID:-unknown}/stage1}"
FULL_RESUME_LOCAL_CKPT="${FULL_RESUME_LOCAL_DIR}/full-resume.ckpt"
FULL_RESUME_LOCAL_META="${FULL_RESUME_LOCAL_DIR}/full-resume.meta.json"

FORCE_FRESH="${FORCE_FRESH:-0}"
LOAD_MODEL_ARG=(--train.load_model "")

# 1) Prefer FULL resume checkpoint if present
if [ "${FULL_AUTO_RESUME}" = "1" ] && [ "${FORCE_FRESH}" != "1" ]; then
    if [ -n "${EXP_ID:-}" ] && [ -n "${GCS_BUCKET:-}" ] && [ -n "${GCS_PREFIX:-}" ]; then
        FULL_GCS_DIR="${GCS_BUCKET%/}/${GCS_PREFIX#/}/${EXP_ID}/_latest/full_resume/stage1"
        FULL_GCS_CKPT="${FULL_GCS_DIR}/full-resume.ckpt"
        FULL_GCS_META="${FULL_GCS_DIR}/full-resume.meta.json"

        if gsutil ls "${FULL_GCS_META}" >/dev/null 2>&1; then
            mkdir -p "${FULL_RESUME_LOCAL_DIR}"
            echo "🔁 Found FULL resume checkpoint in GCS. Downloading..."
            # Download meta first to decide whether ckpt is a directory.
            gsutil cp -q "${FULL_GCS_META}" "${FULL_RESUME_LOCAL_META}" || true
            CKPT_IS_DIR="$(
            FULL_RESUME_LOCAL_META="${FULL_RESUME_LOCAL_META}" python3 - <<'PY'
import json, os
try:
    with open(os.environ["FULL_RESUME_LOCAL_META"], "r") as f:
        meta = json.load(f)
    print("1" if bool(meta.get("ckpt_is_dir", False)) else "0")
except Exception:
    print("0")
PY
            )"
            # Download checkpoint (file or directory) based on meta.
            if [ "${CKPT_IS_DIR}" = "1" ]; then
                gsutil -m rsync -r "${FULL_GCS_CKPT}" "${FULL_RESUME_LOCAL_CKPT}" || true
            else
                gsutil cp -q "${FULL_GCS_CKPT}" "${FULL_RESUME_LOCAL_CKPT}" || true
            fi

            if [ -e "${FULL_RESUME_LOCAL_CKPT}" ] && [ -f "${FULL_RESUME_LOCAL_META}" ]; then
                FULL_META_CTX_LEN="$(
                FULL_RESUME_LOCAL_META="${FULL_RESUME_LOCAL_META}" python3 - <<'PY'
import json, os
meta_path = os.environ["FULL_RESUME_LOCAL_META"]
try:
    with open(meta_path, "r") as f:
        meta = json.load(f)
    print(int(meta.get("ctx_len", 0)))
except Exception:
    print(0)
PY
                )"
                if [ "${FULL_META_CTX_LEN}" -gt 0 ] && [ "${FULL_META_CTX_LEN}" -ne "${CTX_LEN}" ]; then
                    echo "⚠️ FULL resume checkpoint ctx_len=${FULL_META_CTX_LEN} != requested CTX_LEN=${CTX_LEN}; ignoring full resume."
                else
                    export FULL_RESUME_CKPT_PATH="${FULL_RESUME_LOCAL_CKPT}"
                    export FULL_RESUME=1
                    echo "✅ FULL auto-resume enabled: ckpt_path=${FULL_RESUME_CKPT_PATH}"
                fi
            else
                echo "⚠️ FULL resume files failed to download; falling back."
            fi
        fi
    fi
fi

DRY_RUN="${DRY_RUN:-0}"
if [ "${DRY_RUN}" = "1" ]; then
    echo "DRY_RUN=1: skipping dataset check and training."
    # NOTE: bash 3.2 + `set -u` treats empty arrays as "unbound" on expansion.
    # Use `-` default expansion to avoid errors when arrays are empty.
    echo "LOAD_MODEL_ARG=${LOAD_MODEL_ARG[*]-}"
    echo "FULL_RESUME=${FULL_RESUME:-0}"
    echo "FULL_RESUME_CKPT_PATH=${FULL_RESUME_CKPT_PATH:-}"
    exit 0
fi

# -------------------------------------------------------------------
# Paper-style: prepare a base HF checkpoint as .pth and pass it via --train.load_model
# This makes Stage 1 starts reproducible and avoids relying on implicit HF download behavior.
# If FULL_RESUME is active, we do NOT override (trainer.fit will resume from ckpt_path).
# -------------------------------------------------------------------
AUTO_PREPARE_STAGE1_WEIGHTS="${AUTO_PREPARE_STAGE1_WEIGHTS:-1}"
STAGE1_INIT_CKPT="${STAGE1_INIT_CKPT:-out/qwen2-0B5.pth}"
# Default HF id from config if not provided
if [ -z "${STAGE1_HF_ID:-}" ]; then
  STAGE1_HF_ID="$(grep -E '^\s*hf_path:\s*' "${STAGE1_CONFIG}" | head -n 1 | sed -E 's/^\s*hf_path:\s*//')"
fi
STAGE1_HF_ID="${STAGE1_HF_ID:-Qwen/Qwen2.5-0.5B-Instruct}"

if [ "${FULL_RESUME:-0}" != "1" ]; then
  if [ "${AUTO_PREPARE_STAGE1_WEIGHTS}" = "1" ] && [ ! -f "${STAGE1_INIT_CKPT}" ]; then
      echo ""
      echo "=============================================="
      echo "Preparing Stage 1 init checkpoint (HF → .pth)"
      echo "HF:  ${STAGE1_HF_ID}"
      echo "OUT: ${STAGE1_INIT_CKPT}"
      echo "=============================================="
      mkdir -p "$(dirname "${STAGE1_INIT_CKPT}")"
      python3 convert_hf_to_pth.py "${STAGE1_HF_ID}" "${STAGE1_INIT_CKPT}"
      echo "✅ Stage 1 init checkpoint ready: ${STAGE1_INIT_CKPT}"
  fi
  # Use the init checkpoint if it exists (or AUTO_PREPARE was disabled but user provided the file)
  if [ -f "${STAGE1_INIT_CKPT}" ]; then
    LOAD_MODEL_ARG=(--train.load_model "${STAGE1_INIT_CKPT}")
  fi
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
if [ -n "${USE_ACCELERATED_SCAN:-}" ]; then MODEL_ARGS+=(--model.use_accelerated_scan "${USE_ACCELERATED_SCAN}"); fi
if [ -n "${MEMORY_SCAN_CHUNK_LEN:-}" ]; then MODEL_ARGS+=(--model.memory_scan_chunk_len "${MEMORY_SCAN_CHUNK_LEN}"); fi

echo "Configuration:"
echo "  Config: ${STAGE1_CONFIG} (HF Qwen2 + Atlas patch)"
echo "  Init CKPT: ${STAGE1_INIT_CKPT} (AUTO_PREPARE_STAGE1_WEIGHTS=${AUTO_PREPARE_STAGE1_WEIGHTS}, FULL_RESUME=${FULL_RESUME:-0})"
echo "  DATA_PREFIX: ${DATA_PREFIX} (size=${DATA_SIZE} tokens, slot=${DATASET_SLOT}, magic_prime=${MAGIC_PRIME}, ratio=${MAGIC_RATIO})"
echo "  CTX_LEN: ${CTX_LEN}"
echo "  TOKENS (my_exit_tokens): ${TOKENS}"
echo "  DEVICES x NUM_NODES: ${DEVICES} x ${NUM_NODES}"
echo "  MICRO_BSZ: ${MICRO_BSZ}   ACC_GRAD: ${ACC_GRAD}"
echo "  STRATEGY: ${STRATEGY}   GRAD_CP: ${GRAD_CP}   DS_BUCKET_MB: ${DS_BUCKET_MB}"
echo "  WANDB_MODE: ${WANDB_MODE}   WANDB_PROJECT: ${WANDB_PROJECT:-<disabled>}"
echo "  Memory: USE_ACCELERATED_SCAN=${USE_ACCELERATED_SCAN:-<default>}   MEMORY_SCAN_CHUNK_LEN=${MEMORY_SCAN_CHUNK_LEN:-<default>}"
if [ "${WITH_ALL_ABLATION}" = "1" ]; then
    echo "  Ablation: WITH_ALL_ABLATION=1 (poly_degree=3, qk_norm=true, qkv_conv_kernel=4, use_rope=true, use_groupnorm=true)"
else
    echo "  Ablation: custom overrides only (if provided)"
fi
echo ""

# Early stopping CLI args (optional)
EARLY_STOP_ARGS=()
if [ "${EARLY_STOP:-0}" = "1" ]; then
    EARLY_STOP_ARGS+=(--train.early_stop true)
    if [ -n "${EARLY_STOP_MIN_STEPS:-}" ]; then EARLY_STOP_ARGS+=(--train.early_stop_min_steps "${EARLY_STOP_MIN_STEPS}"); fi
    if [ -n "${EARLY_STOP_PATIENCE_STEPS:-}" ]; then EARLY_STOP_ARGS+=(--train.early_stop_patience_steps "${EARLY_STOP_PATIENCE_STEPS}"); fi
    if [ -n "${EARLY_STOP_MIN_DELTA:-}" ]; then EARLY_STOP_ARGS+=(--train.early_stop_min_delta "${EARLY_STOP_MIN_DELTA}"); fi
    if [ -n "${EARLY_STOP_EMA_ALPHA:-}" ]; then EARLY_STOP_ARGS+=(--train.early_stop_ema_alpha "${EARLY_STOP_EMA_ALPHA}"); fi
    # clip can be explicitly disabled by setting empty string
    if [ -n "${EARLY_STOP_LOSS_CLIP+x}" ]; then
        if [ -n "${EARLY_STOP_LOSS_CLIP}" ]; then
            EARLY_STOP_ARGS+=(--train.early_stop_loss_clip "${EARLY_STOP_LOSS_CLIP}")
        else
            EARLY_STOP_ARGS+=(--train.early_stop_loss_clip null)
        fi
    fi
    if [ -n "${EARLY_STOP_TARGET_LOSS:-}" ]; then EARLY_STOP_ARGS+=(--train.early_stop_target_loss "${EARLY_STOP_TARGET_LOSS}"); fi
    if [ -n "${EARLY_STOP_TARGET_PATIENCE_STEPS:-}" ]; then EARLY_STOP_ARGS+=(--train.early_stop_target_patience_steps "${EARLY_STOP_TARGET_PATIENCE_STEPS}"); fi
fi

# Run training
python3 train.py \
    -c "${STAGE1_CONFIG}" \
    --train.data_file "${DATA_PREFIX}" \
    --train.my_exit_tokens "${TOKENS}" \
    --train.magic_prime "${MAGIC_PRIME}" \
    --train.micro_bsz "${MICRO_BSZ}" \
    --train.accumulate_grad_batches "${ACC_GRAD}" \
    --train.devices "${DEVICES}" \
    --train.num_nodes "${NUM_NODES}" \
    --train.train_stage 2 \
    --train.attention_distillation_stage 1 \
    "${LOAD_MODEL_ARG[@]}" \
    --train.strategy "${STRATEGY}" \
    --train.grad_cp "${GRAD_CP}" \
    --train.ds_bucket_mb "${DS_BUCKET_MB}" \
    "${TRAIN_WANDB_ARG}" "${TRAIN_WANDB_VAL}" \
    --train.proj_suffix atlas-stage1 \
    --model.ctx_len "${CTX_LEN}" \
    "${MODEL_ARGS[@]}" \
    "${ABLATION_ARGS[@]}" \
    "${EARLY_STOP_ARGS[@]}"

echo ""
echo "✅ Paper Stage 1 complete!"
echo ""
echo "Next step (Paper Stage 2):"
echo "  bash scripts/run_stage2.sh <stage1_checkpoint.pth>"
