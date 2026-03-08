#!/bin/bash
# Phase 2: PSMA fine-tuning on 8x B200
# Usage: bash scripts/run_finetune.sh

set -e
source .venv/bin/activate

N_GPUS=${N_GPUS:-8}
CONFIG=${CONFIG:-configs/finetune_b200.yaml}

echo "=== Phase 2: PSMA Fine-tuning ==="
echo "GPUs: $N_GPUS | Config: $CONFIG"

torchrun \
  --nproc_per_node=$N_GPUS \
  --master_port=29501 \
  -m training.finetune_psma \
  --config $CONFIG
