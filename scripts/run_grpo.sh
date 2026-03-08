#!/bin/bash
# Phase 3: GRPO with docking reward on 8x B200
# Usage: bash scripts/run_grpo.sh
#
# NOTE: Gnina must be installed and in PATH.
# Download: https://github.com/gnina/gnina/releases

set -e
source .venv/bin/activate

N_GPUS=${N_GPUS:-8}
CONFIG=${CONFIG:-configs/grpo_b200.yaml}

echo "=== Phase 3: GRPO Training ==="
echo "GPUs: $N_GPUS | Config: $CONFIG"

# Verify gnina exists
if ! command -v gnina &>/dev/null; then
  echo "ERROR: gnina not found in PATH"
  echo "Install from: https://github.com/gnina/gnina/releases"
  exit 1
fi

# Verify receptor is prepared
if [ ! -f docking/8BOW_receptor.pdb ]; then
  echo "Preparing receptor..."
  python -m docking.prepare_receptor
fi

torchrun \
  --nproc_per_node=$N_GPUS \
  --master_port=29502 \
  -m training.grpo \
  --config $CONFIG
