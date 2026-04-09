# Tell Me Where It Hurts

Cost-Effective Encoder Improvement via Attribution-Guided LLM Augmentation

## Quick Start

```bash
# Install
pip install -e .

# 1. Train baseline BERT on SNLI
cd src && python train_baseline.py

# 2. Run post-hoc boosting loop
python run_loop.py
```

## Project Structure

```
src/
  data.py               # Dataset loading (SNLI, HANS, Kaushik CAD)
  train_baseline.py     # Step 0: Train baseline encoder
  evaluate_failures.py  # Step 1: Collect failures + attention/gradient
  diagnose_generate.py  # Step 2-3: LLM diagnosis + CF pair generation
  finetune_boost.py     # Step 4: Finetune with CE + contrastive loss
  run_loop.py           # Main loop: iterate until convergence
configs/
  default.yaml          # Default configuration
scripts/
  setup_pod.sh          # RunPod environment setup
```

## Environment Variables

- `OPENROUTER_API_KEY` — Required for LLM calls (diagnosis + generation)
- `HUGGING_FACE_HUB_TOKEN` — Optional, for gated datasets
