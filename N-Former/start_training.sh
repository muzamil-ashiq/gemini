#!/bin/bash

# Mathematical nanoGPT Training Script
# Phase 1: Pre-training on TinyStories (Days 1-3)

echo "🧠 MATHEMATICAL NANOGPT TRAINING - PHASE 1"
echo "=========================================="
echo "Dataset: TinyStories (~4M tokens)"
echo "Model: Enhanced with Hebbian/Hopfield/Pi-Delta/Infinity systems"
echo "Hardware: Optimized for 4GB VRAM"
echo "Duration: Long-running pre-training"
echo ""

cd "/home/muzamil/plus gpt/nanoGPT"

# Start training with proper parameters
"/home/muzamil/plus gpt/.venv/bin/python" train.py \
    --dataset=tinystories \
    --max_iters=50000 \
    --eval_interval=1000 \
    --log_interval=200 \
    --eval_iters=200 \
    --compile=False \
    --always_save_checkpoint=True \
    --wandb_log=False

echo ""
echo "✅ Phase 1 Pre-training Complete!"
echo "Next: Run test_model.sh to evaluate the pre-trained model"
