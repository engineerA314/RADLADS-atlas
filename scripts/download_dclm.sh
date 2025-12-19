#!/bin/bash
# Download DCLM-10B dataset for RADLADS training

set -e

echo "=============================================="
echo "Downloading DCLM-10B Dataset"
echo "=============================================="

# Create data directory
mkdir -p data

# Download .idx file
echo "Downloading dclm-10B.idx..."
wget --continue -O data/dclm-10B.idx \
    "https://huggingface.co/datasets/recursal/DCLM-10B-Qwen2-binidx/resolve/main/dclm-10B.idx?download=true"

# Download .bin file
echo "Downloading dclm-10B.bin..."
wget --continue -O data/dclm-10B.bin \
    "https://huggingface.co/datasets/recursal/DCLM-10B-Qwen2-binidx/resolve/main/dclm-10B.bin?download=true"

# Verify files exist
if [ -f "data/dclm-10B.idx" ] && [ -f "data/dclm-10B.bin" ]; then
    echo ""
    echo "✅ Download complete!"
    echo ""
    echo "Files:"
    ls -lh data/dclm-10B.*
    echo ""
    echo "You can now run training with:"
    echo "  bash scripts/run_stage1.sh"
else
    echo ""
    echo "❌ Download failed. Please check your connection and try again."
    exit 1
fi
