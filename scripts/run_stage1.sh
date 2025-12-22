#!/bin/bash
# Run RADLADS Paper Stage 1: Attention Alignment
#
# Paper Stage 1 uses attention_distillation_stage=1 (no separate teacher model).
# This script uses atlas_distill1.yaml which patches HF Qwen2 with Atlas attention.
# 
# Alternative: Use atlasqwen0b5.yaml + distill1.yaml for pure Atlas model
# (see gcp_test/ scripts for that approach)

set -e

echo "=============================================="
echo "RADLADS Paper Stage 1: Attention Alignment"
echo "=============================================="

# Check if data exists
if [ ! -f "data/dclm-10B.idx" ] || [ ! -f "data/dclm-10B.bin" ]; then
    echo "❌ Data not found. Please run: bash scripts/download_dclm.sh"
    exit 1
fi

# Configuration
TOKENS=${1:-100000000}  # Default: 100M tokens
DEVICES=${2:-1}         # Default: 1 GPU

echo "Configuration:"
echo "  Tokens: $TOKENS"
echo "  Devices: $DEVICES"
echo "  Config: configs/atlas_distill1.yaml (HF Qwen2 + Atlas patch)"
echo ""

# Run training
python3 train.py \
    -c configs/atlas_distill1.yaml \
    --train.data_file data/dclm-10B \
    --train.my_exit_tokens $TOKENS \
    --train.devices $DEVICES

echo ""
echo "✅ Paper Stage 1 complete!"
echo "Output saved to: out/L6-D512-x060-atlas-stage1/"
echo ""
echo "Next step (Paper Stage 2):"
echo "  bash scripts/run_stage2.sh out/L6-D512-x060-atlas-stage1/rwkv-final.pth"
