#!/bin/bash
# Run RADLADS Paper Stage 3: Long Context Mid-Training (progressive ctx)
# Paper Stage 3: CE loss only, NO teacher model required.
#
# This script runs a progressive context schedule (e.g., 512→1024→2048),
# where each step loads the checkpoint from the previous step.

set -euo pipefail

echo "=============================================="
echo "RADLADS Paper Stage 3: Long Context Mid-Training"
echo "=============================================="
echo "Paper Stage 3: CE loss only (no teacher model)"
echo "attention_distillation_stage=-1"
echo "=============================================="

usage() {
    echo "Usage:"
    echo "  bash scripts/run_stage3.sh <stage2_checkpoint> [ctx_schedule] [tokens] [devices]"
    echo ""
    echo "ctx_schedule:"
    echo "  - comma-separated list (recommended): 512,1024,2048"
    echo "  - single ctx (backward compatible): 2048"
    echo ""
    echo "Examples:"
    echo "  bash scripts/run_stage3.sh out/.../rwkv-final.pth 512,1024,2048 250000000 8"
    echo "  bash scripts/run_stage3.sh out/.../rwkv-final.pth 2048 250000000 8"
    echo "  WANDB_PROJECT=radlads-atlas bash scripts/run_stage3.sh out/.../rwkv-final.pth 512,1024 250000000 8"
    echo "  WITH_ALL_ABLATION=1 bash scripts/run_stage3.sh out/.../rwkv-final.pth 512,1024,2048 250000000 8"
    echo ""
    echo "Env overrides:"
    echo "  DATA_PREFIX, CTX_SCHEDULE, TOKENS, TOKENS_SCHEDULE, DEVICES, NUM_NODES, MICRO_BSZ, ACC_GRAD,"
    echo "  STRATEGY, GRAD_CP, DS_BUCKET_MB, WANDB_PROJECT, WANDB_MODE,"
    echo "  OMEGA_WINDOW, USE_OMEGA_GATE, USE_MOMENTUM, POLY_MODE, ATLAS_VARIANT, SLIDING_WINDOW,"
    echo "  USE_ACCELERATED_SCAN, MEMORY_SCAN_CHUNK_LEN,"
    echo "  WITH_ALL_ABLATION, POLY_DEGREE, QK_NORM, QKV_CONV_KERNEL, USE_ROPE, USE_GROUPNORM"
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] || [ -z "${1:-}" ]; then
    usage
    exit 0
fi

STAGE2_CKPT="$1"

# Backward-compatible positional parsing:
# - If $2 contains a comma, treat it as a schedule.
# - Else treat it as a single ctx_len (old interface).
CTX_SCHEDULE="${CTX_SCHEDULE:-${2:-512,1024,2048}}"
if [ -n "${2:-}" ] && [[ "${2:-}" != *","* ]]; then
    CTX_SCHEDULE="${2}"
fi

TOKENS="${TOKENS:-${3:-250000000}}"
DEVICES="${DEVICES:-${4:-1}}"
NUM_NODES="${NUM_NODES:-1}"
ACC_GRAD="${ACC_GRAD:-1}"
STRATEGY="${STRATEGY:-auto}"
GRAD_CP="${GRAD_CP:-0}"
DS_BUCKET_MB="${DS_BUCKET_MB:-200}"
DATA_PREFIX="${DATA_PREFIX:-data/dclm-10B}"
export DATA_PREFIX

# Optional per-stage tokens list, e.g. TOKENS_SCHEDULE="50000000,50000000,250000000"
TOKENS_SCHEDULE="${TOKENS_SCHEDULE:-}"

# Full auto-resume (proper resume: optimizer + step counters)
FULL_AUTO_RESUME="${FULL_AUTO_RESUME:-1}"
FORCE_FRESH="${FORCE_FRESH:-0}"
DRY_RUN="${DRY_RUN:-0}"

# Check if checkpoint exists
if [ ! -f "$STAGE2_CKPT" ]; then
    echo "❌ Checkpoint not found: $STAGE2_CKPT"
    exit 1
fi

if [ "${DRY_RUN}" != "1" ]; then
    # Check if data exists
    if [ ! -f "${DATA_PREFIX}.idx" ] || [ ! -f "${DATA_PREFIX}.bin" ]; then
        echo "❌ Data not found: ${DATA_PREFIX}.idx/.bin"
        echo "   If you want DCLM-10B, run: bash scripts/download_dclm.sh"
        exit 1
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
echo "  Stage 2 Checkpoint: $STAGE2_CKPT"
echo "  DATA_PREFIX: ${DATA_PREFIX}"
echo "  CTX_SCHEDULE: ${CTX_SCHEDULE}"
echo "  TOKENS (default my_exit_tokens per stage): ${TOKENS}"
echo "  TOKENS_SCHEDULE: ${TOKENS_SCHEDULE:-<none>}"
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

IFS=',' read -r -a CTX_LIST <<< "${CTX_SCHEDULE}"
TOKENS_LIST=()
if [ -n "${TOKENS_SCHEDULE}" ]; then
    IFS=',' read -r -a TOKENS_LIST <<< "${TOKENS_SCHEDULE}"
fi

PREV_CKPT="${STAGE2_CKPT}"
FINAL_CKPT=""

for idx in "${!CTX_LIST[@]}"; do
    ctx="${CTX_LIST[$idx]}"
    ctx="$(echo "${ctx}" | tr -d '[:space:]')"
    if [ -z "${ctx}" ]; then
        continue
    fi

    # Determine tokens for this step
    step_tokens="${TOKENS}"
    if [ "${#TOKENS_LIST[@]}" -gt 0 ]; then
        if [ "${#TOKENS_LIST[@]}" -eq 1 ]; then
            step_tokens="${TOKENS_LIST[0]}"
        else
            step_tokens="${TOKENS_LIST[$idx]}"
        fi
    fi

    # Determine distill3 config for this ctx
    if [ "${ctx}" = "512" ]; then
        CONFIG_FILE="configs/distill3.yaml"
    else
        CONFIG_FILE="configs/distill3-${ctx}.yaml"
    fi
    if [ ! -f "${CONFIG_FILE}" ]; then
        echo "❌ Config not found for ctx=${ctx}: ${CONFIG_FILE}"
        echo "   Available examples: distill3.yaml (512), distill3-1024.yaml, distill3-2048.yaml, distill3-4096.yaml, distill3-8192.yaml, distill3-16384.yaml"
        exit 1
    fi

    if [ "${DRY_RUN}" != "1" ]; then
        # Compute data_size + magic_prime for (DATA_PREFIX, ctx)
        export CTX_LEN="${ctx}"
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

        if [ "${step_tokens}" -gt "${DATA_SIZE}" ]; then
            echo "❌ TOKENS (${step_tokens}) must be <= data_size (${DATA_SIZE}) for ctx=${ctx} (DATA_PREFIX=${DATA_PREFIX})"
            exit 1
        fi
    else
        DATA_SIZE="0"
        DATASET_SLOT="0"
        MAGIC_PRIME="0"
        MAGIC_RATIO="0"
    fi

    PROJ_SUFFIX="atlas-stage3-ctx${ctx}"
    OUT_DIR="out/atlas-lmm-0b5-${PROJ_SUFFIX}"

    # If already finished, skip step.
    if [ "${DRY_RUN}" != "1" ] && [ -f "${OUT_DIR}/rwkv-final.pth" ] && [ "${FORCE_FRESH}" != "1" ]; then
        echo "✅ Found existing final checkpoint for ctx=${ctx}. Skipping training step."
        PREV_CKPT="${OUT_DIR}/rwkv-final.pth"
        FINAL_CKPT="${PREV_CKPT}"
        continue
    fi

    # Full auto-resume for this ctx step (best-effort)
    unset FULL_RESUME_CKPT_PATH || true
    export FULL_RESUME=0
    if [ "${FULL_AUTO_RESUME}" = "1" ] && [ "${FORCE_FRESH}" != "1" ]; then
        if [ -n "${EXP_ID:-}" ] && [ -n "${GCS_BUCKET:-}" ] && [ -n "${GCS_PREFIX:-}" ]; then
            STEP_LABEL="stage3-ctx${ctx}"
            FULL_RESUME_LOCAL_DIR="${FULL_RESUME_LOCAL_DIR:-out/_full_resume/${EXP_ID:-unknown}/${STEP_LABEL}}"
            FULL_RESUME_LOCAL_CKPT="${FULL_RESUME_LOCAL_DIR}/full-resume.ckpt"
            FULL_RESUME_LOCAL_META="${FULL_RESUME_LOCAL_DIR}/full-resume.meta.json"

            FULL_GCS_DIR="${GCS_BUCKET%/}/${GCS_PREFIX#/}/${EXP_ID}/_latest/full_resume/${STEP_LABEL}"
            FULL_GCS_CKPT="${FULL_GCS_DIR}/full-resume.ckpt"
            FULL_GCS_META="${FULL_GCS_DIR}/full-resume.meta.json"

            if gsutil ls "${FULL_GCS_META}" >/dev/null 2>&1; then
                mkdir -p "${FULL_RESUME_LOCAL_DIR}"
                echo "🔁 Found FULL resume checkpoint in GCS for ${STEP_LABEL}. Downloading..."
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
                    export FULL_RESUME_CKPT_PATH="${FULL_RESUME_LOCAL_CKPT}"
                    export FULL_RESUME=1
                    echo "✅ FULL auto-resume enabled for ${STEP_LABEL}: ckpt_path=${FULL_RESUME_CKPT_PATH}"
                else
                    echo "⚠️ FULL resume files failed to download for ${STEP_LABEL}; starting fresh."
                fi
            fi
        fi
    fi

    echo "----------------------------------------------"
    echo "Stage 3 step $((idx + 1))/${#CTX_LIST[@]}: ctx=${ctx}"
    echo "  Config: ${CONFIG_FILE}"
    echo "  Load: ${PREV_CKPT}"
    echo "  DATA_PREFIX: ${DATA_PREFIX} (size=${DATA_SIZE} tokens, slot=${DATASET_SLOT}, magic_prime=${MAGIC_PRIME}, ratio=${MAGIC_RATIO})"
    echo "  TOKENS (my_exit_tokens): ${step_tokens}"
    echo "  Output dir: ${OUT_DIR}"
    echo "----------------------------------------------"

    if [ "${DRY_RUN}" = "1" ]; then
        echo "DRY_RUN: would run ctx=${ctx} with PROJ_SUFFIX=${PROJ_SUFFIX}"
        echo "FULL_RESUME=${FULL_RESUME}"
        echo "FULL_RESUME_CKPT_PATH=${FULL_RESUME_CKPT_PATH:-}"
        FINAL_CKPT="${PREV_CKPT}"
        continue
    fi

    # Run training (CE only, no teacher)
    python3 train.py \
        -c configs/atlasqwen0b5.yaml \
        -c "${CONFIG_FILE}" \
        --train.data_file "${DATA_PREFIX}" \
        --train.my_exit_tokens "${step_tokens}" \
        --train.magic_prime "${MAGIC_PRIME}" \
        --train.micro_bsz "${MICRO_BSZ}" \
        --train.accumulate_grad_batches "${ACC_GRAD}" \
        --train.devices "${DEVICES}" \
        --train.num_nodes "${NUM_NODES}" \
        --train.train_stage 2 \
        --train.attention_distillation_stage -1 \
        --train.load_model "${PREV_CKPT}" \
        --train.strategy "${STRATEGY}" \
        --train.grad_cp "${GRAD_CP}" \
        --train.ds_bucket_mb "${DS_BUCKET_MB}" \
        "${TRAIN_WANDB_ARG}" "${TRAIN_WANDB_VAL}" \
        --train.proj_suffix "${PROJ_SUFFIX}" \
        --model.ctx_len "${ctx}" \
        "${MODEL_ARGS[@]}" \
        "${ABLATION_ARGS[@]}"

    # Resolve checkpoint for next step
    if [ -f "${OUT_DIR}/rwkv-final.pth" ]; then
        PREV_CKPT="${OUT_DIR}/rwkv-final.pth"
    else
        # Fallback: pick newest checkpoint
        newest_ckpt="$(ls -t "${OUT_DIR}"/rwkv-*.pth 2>/dev/null | head -n 1 || true)"
        if [ -z "${newest_ckpt}" ]; then
            echo "❌ Could not find checkpoint in ${OUT_DIR} (expected rwkv-final.pth or rwkv-*.pth)"
            exit 1
        fi
        PREV_CKPT="${newest_ckpt}"
    fi
    FINAL_CKPT="${PREV_CKPT}"
done

echo ""
echo "✅ Paper Stage 3 complete!"
echo "Final checkpoint: ${FINAL_CKPT:-<unknown>}"
echo ""
echo "To continue on longer context:"
echo "  bash scripts/run_stage3.sh \"${FINAL_CKPT:-<stage3_ckpt.pth>}\" 4096 250000000 ${DEVICES}"
