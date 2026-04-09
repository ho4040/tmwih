#!/bin/bash
# RunPod pod setup script for tmwih-train

set -e

echo "=== Setting up tmwih training environment ==="

# Clone repo
if [ ! -d "/workspace/tmwih" ]; then
    git clone https://github.com/ho4040/tmwih.git /workspace/tmwih
fi
cd /workspace/tmwih

# Install dependencies
pip install -e ".[dev]"

# Login to HuggingFace (needed for dataset downloads)
# Token should be set as HUGGING_FACE_HUB_TOKEN env var in RunPod
if [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then
    huggingface-cli login --token "$HUGGING_FACE_HUB_TOKEN"
fi

echo "=== Setup complete ==="
echo "To train baseline:  cd /workspace/tmwih/src && python train_baseline.py"
echo "To run boost loop:  cd /workspace/tmwih/src && python run_loop.py"
