#!/bin/bash
# Run RADLADS Stage 3: Long Context Mid-Training

set -e

echo "=============================================="
echo "RADLADS Stage 3: Long Context Mid-Training"
echo "=============================================="

# Check arguments
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: bash scripts/run_stage3.sh <stage2_checkpoint> <ctx_len> [tokens] [devices]"
    echo "Example: bash scripts/run_stage3.sh out/atlas-stage2/rwkv-final.pth 2048 250000000 1"
    echo ""
    echo "Available context lengths: 512, 1024, 2048, 4096, 8192, 16384"
    exit 1
fi

STAGE2_CKPT=$1
CTX_LEN=$2
TOKENS=${3:-250000000}  # Default: 250M tokens
DEVICES=${4:-1}         # Default: 1 GPU

# Check if checkpoint exists
if [ ! -f "$STAGE2_CKPT" ]; then
    echo "❌ Checkpoint not found: $STAGE2_CKPT"
    exit 1
fi

# Check if data exists
if [ ! -f "data/dclm-10B.idx" ] || [ ! -f "data/dclm-10B.bin" ]; then
    echo "❌ Data not found. Please run: bash scripts/download_dclm.sh"
    exit 1
fi

# Determine config file
CONFIG_FILE="configs/distill3-${CTX_LEN}.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config not found: $CONFIG_FILE"
    echo "Available: 512, 1024, 2048, 4096, 8192, 16384"
    exit 1
fi

echo "Configuration:"
echo "  Stage 2 Checkpoint: $STAGE2_CKPT"
echo "  Context Length: $CTX_LEN"
echo "  Config: $CONFIG_FILE"
echo "  Tokens: $TOKENS"
echo "  Devices: $DEVICES"
echo ""

# Run training
python3 train.py \
    -c configs/atlasqwen0b5.yaml \
    -c configs/qwen0b5hfteacher.yaml \
    -c "$CONFIG_FILE" \
    --train.data_file data/dclm-10B \
    --train.my_exit_tokens $TOKENS \
    --train.devices $DEVICES \
    --train.load_model "$STAGE2_CKPT" \
    --model.ctx_len $CTX_LEN

echo ""
echo "✅ Stage 3 (ctx=$CTX_LEN) complete!"
echo "Output saved to: out/L6-D512-x060-4/"
echo ""
echo "To train on longer context:"
echo "  bash scripts/run_stage3.sh out/L6-D512-x060-4/rwkv-final.pth 4096"
