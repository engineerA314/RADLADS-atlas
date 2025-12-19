#!/bin/bash
# Run RADLADS Stage 2: Logit KL Divergence

set -e

echo "=============================================="
echo "RADLADS Stage 2: Logit KL Divergence"
echo "=============================================="

# Check arguments
if [ -z "$1" ]; then
    echo "Usage: bash scripts/run_stage2.sh <stage1_checkpoint> [tokens] [devices]"
    echo "Example: bash scripts/run_stage2.sh out/atlas-stage1/rwkv-final.pth 500000000 1"
    exit 1
fi

STAGE1_CKPT=$1
TOKENS=${2:-500000000}  # Default: 500M tokens
DEVICES=${3:-1}         # Default: 1 GPU

# Check if checkpoint exists
if [ ! -f "$STAGE1_CKPT" ]; then
    echo "❌ Checkpoint not found: $STAGE1_CKPT"
    exit 1
fi

# Check if data exists
if [ ! -f "data/dclm-10B.idx" ] || [ ! -f "data/dclm-10B.bin" ]; then
    echo "❌ Data not found. Please run: bash scripts/download_dclm.sh"
    exit 1
fi

echo "Configuration:"
echo "  Stage 1 Checkpoint: $STAGE1_CKPT"
echo "  Tokens: $TOKENS"
echo "  Devices: $DEVICES"
echo ""

# Run training
python3 train.py \
    -c configs/atlasqwen0b5.yaml \
    -c configs/qwen0b5hfteacher.yaml \
    -c configs/distill2.yaml \
    --train.data_file data/dclm-10B \
    --train.my_exit_tokens $TOKENS \
    --train.devices $DEVICES \
    --train.load_model "$STAGE1_CKPT"

echo ""
echo "✅ Stage 2 complete!"
echo "Output saved to: out/L6-D512-x060-2/"
echo ""
echo "Next step (optional, for long context):"
echo "  bash scripts/run_stage3.sh out/L6-D512-x060-2/rwkv-final.pth 2048"
