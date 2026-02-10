#!/bin/bash
# Run RADLADS Paper Stage 2: Logit KL Divergence
#
# Paper Stage 2 uses attention_distillation_stage=2 with a separate teacher model.

set -euo pipefail

echo "=============================================="
echo "RADLADS Paper Stage 2: Logit KL Divergence"
echo "=============================================="

usage() {
    echo "Usage:"
    echo "  bash scripts/run_stage2.sh <paper_stage1_checkpoint> [tokens] [devices] [ctx_len]"
    echo ""
    echo "Examples:"
    echo "  bash scripts/run_stage2.sh out/.../rwkv-final.pth 500000000 8 512"
    echo "  WANDB_PROJECT=radlads-atlas bash scripts/run_stage2.sh out/.../rwkv-final.pth 500000000 8 512"
    echo "  WITH_ALL_ABLATION=1 bash scripts/run_stage2.sh out/.../rwkv-final.pth 500000000 8 512"
    echo ""
    echo "Env overrides:"
    echo "  DATA_PREFIX, CTX_LEN, TOKENS, DEVICES, NUM_NODES, MICRO_BSZ, ACC_GRAD, STRATEGY, GRAD_CP, DS_BUCKET_MB,"
    echo "  WANDB_PROJECT, WANDB_MODE,"
    echo "  OMEGA_WINDOW, USE_OMEGA_GATE, USE_MOMENTUM, POLY_MODE, ATLAS_VARIANT, SLIDING_WINDOW,"
    echo "  USE_ACCELERATED_SCAN, MEMORY_SCAN_CHUNK_LEN,"
    echo "  WITH_ALL_ABLATION, POLY_DEGREE, QK_NORM, QKV_CONV_KERNEL, USE_ROPE, USE_GROUPNORM"
    echo ""
    echo "Optional:"
    echo "  TEACHER_CONFIG (default: configs/qwen0b5teacher.yaml; passed as an extra -c <yaml>)"
    echo ""
    echo "Teacher preparation (recommended):"
    echo "  AUTO_PREPARE_TEACHER=1 (default) will create teacher checkpoint if missing."
    echo "  TEACHER_HF_ID (default: Qwen/Qwen2.5-0.5B-Instruct)"
    echo "  TEACHER_CKPT (default: out/qwen2-0B5.safetensors)"
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] || [ -z "${1:-}" ]; then
    usage
    exit 0
fi

PAPER_STAGE1_CKPT=$1
TOKENS="${TOKENS:-${2:-500000000}}"  # Default: 500M tokens
DEVICES="${DEVICES:-${3:-1}}"
CTX_LEN="${CTX_LEN:-${4:-512}}"
NUM_NODES="${NUM_NODES:-1}"
ACC_GRAD="${ACC_GRAD:-1}"
STRATEGY="${STRATEGY:-auto}"
GRAD_CP="${GRAD_CP:-0}"
DS_BUCKET_MB="${DS_BUCKET_MB:-200}"
DATA_PREFIX="${DATA_PREFIX:-data/dclm-10B}"
export DATA_PREFIX
export CTX_LEN

# Full auto-resume (proper resume: optimizer + step counters)
FULL_AUTO_RESUME="${FULL_AUTO_RESUME:-1}"
FORCE_FRESH="${FORCE_FRESH:-0}"
FULL_RESUME_ALLOW_NO_META="${FULL_RESUME_ALLOW_NO_META:-0}"

FULL_RESUME_STAGE_LABEL="stage2"
FULL_RESUME_LOCAL_DIR="${FULL_RESUME_LOCAL_DIR:-out/_full_resume/${EXP_ID:-unknown}/${FULL_RESUME_STAGE_LABEL}}"
FULL_RESUME_LOCAL_CKPT="${FULL_RESUME_LOCAL_DIR}/full-resume.ckpt"
FULL_RESUME_LOCAL_META="${FULL_RESUME_LOCAL_DIR}/full-resume.meta.json"

if [ "${FULL_AUTO_RESUME}" = "1" ] && [ "${FORCE_FRESH}" != "1" ]; then
    if [ -n "${EXP_ID:-}" ] && [ -n "${GCS_BUCKET:-}" ] && [ -n "${GCS_PREFIX:-}" ]; then
        FULL_GCS_DIR="${GCS_BUCKET%/}/${GCS_PREFIX#/}/${EXP_ID}/_latest/full_resume/${FULL_RESUME_STAGE_LABEL}"
        FULL_GCS_CKPT="${FULL_GCS_DIR}/full-resume.ckpt"
        FULL_GCS_META="${FULL_GCS_DIR}/full-resume.meta.json"

        if gsutil ls "${FULL_GCS_META}" >/dev/null 2>&1; then
            mkdir -p "${FULL_RESUME_LOCAL_DIR}"
            echo "🔁 Found FULL resume checkpoint in GCS. Downloading..."
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
        elif [ "${FULL_RESUME_ALLOW_NO_META}" = "1" ]; then
            FULL_GCS_CKPT_DIR="${FULL_GCS_CKPT%/}/"
            if gsutil ls "${FULL_GCS_CKPT_DIR}" >/dev/null 2>&1; then
                mkdir -p "${FULL_RESUME_LOCAL_CKPT}"
                echo "⚠️ FULL resume meta missing. Forcing resume from checkpoint dir (no ctx_len check)."
                gsutil -m rsync -r "${FULL_GCS_CKPT_DIR}" "${FULL_RESUME_LOCAL_CKPT}/" || true
                if [ -e "${FULL_RESUME_LOCAL_CKPT}" ]; then
                    export FULL_RESUME_CKPT_PATH="${FULL_RESUME_LOCAL_CKPT}"
                    export FULL_RESUME=1
                    echo "✅ FULL auto-resume enabled (no meta): ckpt_path=${FULL_RESUME_CKPT_PATH}"
                else
                    echo "⚠️ FULL resume checkpoint dir failed to download; falling back."
                fi
            fi
        fi
    fi
fi

DRY_RUN="${DRY_RUN:-0}"
if [ "${DRY_RUN}" = "1" ]; then
    echo "DRY_RUN=1: skipping dataset check and training."
    echo "FULL_RESUME=${FULL_RESUME:-0}"
    echo "FULL_RESUME_CKPT_PATH=${FULL_RESUME_CKPT_PATH:-}"
    echo "LOAD_MODEL=${PAPER_STAGE1_CKPT}"
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

# Check if checkpoint exists
if [ ! -f "$PAPER_STAGE1_CKPT" ]; then
    echo "❌ Checkpoint not found: $PAPER_STAGE1_CKPT"
    exit 1
fi

# Check if data exists
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

low = int(math.floor(0.99 * dataset_slot)) + 1
p = dataset_slot - 1
while p >= low:
    if (p % 3 == 2) and MaybeIsPrime(p):
        break
    p -= 1
if p < low:
    raise SystemExit(f"could not find valid magic_prime in [{low}, {dataset_slot-1}] for ctx_len={ctx_len}, data_size={data_size}")

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

# Optional extra teacher config (paper-style; provides teacher.model + teacher.path)
TEACHER_CONFIG="${TEACHER_CONFIG:-configs/qwen0b5teacher.yaml}"
TEACHER_CONFIG_ARGS=()
if [ -n "${TEACHER_CONFIG}" ]; then
    if [ ! -f "${TEACHER_CONFIG}" ]; then
        echo "❌ TEACHER_CONFIG not found: ${TEACHER_CONFIG}"
        exit 1
    fi
    TEACHER_CONFIG_ARGS=(-c "${TEACHER_CONFIG}")
fi

# --- Ensure teacher checkpoint exists (Stage 2 requires a teacher) ---
AUTO_PREPARE_TEACHER="${AUTO_PREPARE_TEACHER:-1}"
TEACHER_HF_ID="${TEACHER_HF_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
TEACHER_CKPT="${TEACHER_CKPT:-out/qwen2-0B5.pth}"

if [ "${AUTO_PREPARE_TEACHER}" = "1" ]; then
    if [ ! -f "${TEACHER_CKPT}" ]; then
        echo ""
        echo "=============================================="
        echo "Preparing teacher checkpoint for Stage 2"
        echo "HF:   ${TEACHER_HF_ID}"
        echo "OUT:  ${TEACHER_CKPT}"
        echo "=============================================="
        mkdir -p "$(dirname "${TEACHER_CKPT}")"
        # Save as .pth directly (Paper README workflow). This avoids safetensors issues with tied weights.
        python3 convert_hf_to_pth.py "${TEACHER_HF_ID}" "${TEACHER_CKPT}"
        echo "✅ Teacher checkpoint ready: ${TEACHER_CKPT}"
    fi
fi

echo "Configuration:"
echo "  Paper Stage 1 Checkpoint: $PAPER_STAGE1_CKPT"
echo "  Configs: configs/atlasqwen0b5.yaml + configs/distill2.yaml + ${TEACHER_CONFIG:-<none>}"
echo "  Teacher CKPT: ${TEACHER_CKPT}"
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

# Run training
python3 train.py \
    -c configs/atlasqwen0b5.yaml \
    "${TEACHER_CONFIG_ARGS[@]}" \
    -c configs/distill2.yaml \
    --train.data_file "${DATA_PREFIX}" \
    --train.my_exit_tokens "${TOKENS}" \
    --train.magic_prime "${MAGIC_PRIME}" \
    --train.micro_bsz "${MICRO_BSZ}" \
    --train.accumulate_grad_batches "${ACC_GRAD}" \
    --train.devices "${DEVICES}" \
    --train.num_nodes "${NUM_NODES}" \
    --train.strategy "${STRATEGY}" \
    --train.grad_cp "${GRAD_CP}" \
    --train.ds_bucket_mb "${DS_BUCKET_MB}" \
    "${TRAIN_WANDB_ARG}" "${TRAIN_WANDB_VAL}" \
    --train.proj_suffix atlas-stage2 \
    --train.load_model "${PAPER_STAGE1_CKPT}" \
    --train.attention_distillation_stage 2 \
    --train.teacher.path "${TEACHER_CKPT}" \
    --model.ctx_len "${CTX_LEN}" \
    "${MODEL_ARGS[@]}" \
    "${ABLATION_ARGS[@]}"

echo ""
echo "✅ Paper Stage 2 complete!"
echo ""
echo "Next step (Paper Stage 3 - Long Context):"
echo "  bash scripts/run_stage3.sh <stage2_checkpoint.pth> 512,1024,2048 250000000 8"
