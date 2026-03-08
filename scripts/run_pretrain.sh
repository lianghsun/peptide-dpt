#!/bin/bash
# Phase 1: Pretrain gemma-3-1b from scratch on 8x B200
# Usage: bash scripts/run_pretrain.sh [--resume CHECKPOINT]

set -e
source .venv/bin/activate

N_GPUS=${N_GPUS:-8}
CONFIG=${CONFIG:-configs/pretrain_b200.yaml}
RESUME=${1}

echo "=== Phase 1: Pretraining ==="
echo "GPUs: $N_GPUS | Config: $CONFIG"

torchrun \
  --nproc_per_node=$N_GPUS \
  --master_port=29500 \
  -m training.pretrain \
  --config $CONFIG \
  ${RESUME:+--resume $RESUME}
