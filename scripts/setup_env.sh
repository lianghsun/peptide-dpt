#!/bin/bash
# Setup environment on B200 training server
# Run once: bash scripts/setup_env.sh

set -euo pipefail

echo "=== peptide-dpt environment setup ==="
echo "Python: $(python3 --version)"
echo "CUDA:   $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not in PATH')"
echo "GPUs:   $(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 0)"

# Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# Upgrade packaging toolchain first
python -m pip install --upgrade pip setuptools wheel build
python -m pip install ninja packaging

# Install PyTorch first (CUDA 12.8 wheels)
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

# Install the rest (exclude flash-attn from requirements.txt)
python -m pip install -r requirements.txt

# Install project in editable mode
python -m pip install -e .

# Flash Attention 2
# Official install pattern recommends --no-build-isolation.
MAX_JOBS=${MAX_JOBS:-8} \
python -m pip install --no-build-isolation flash-attn || \
  echo "⚠ flash-attn install failed — continuing without it"

echo ""
echo "=== Verifying ==="
python - <<'PY'
import torch
print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}")
print(f"cuda available: {torch.cuda.is_available()}")
PY

python - <<'PY'
import selfies, rdkit, transformers, trl, deepspeed
print("Core packages OK")
PY

python - <<'PY'
try:
    import flash_attn
    print(f"flash-attn {flash_attn.__version__}")
except Exception as e:
    print(f"flash-attn not available: {e}")
PY

echo ""
echo "✓ Setup complete. Activate with: source .venv/bin/activate"
