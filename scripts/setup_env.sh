#!/bin/bash
# Setup environment on B200 training server
# Run once: bash scripts/setup_env.sh

set -e

echo "=== peptide-dpt environment setup ==="
echo "Python: $(python3 --version)"
echo "CUDA:   $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not in PATH')"
echo "GPUs:   $(nvidia-smi --list-gpus | wc -l)"

# Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch first (CUDA 12.8 for B200)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install the rest
pip install -r requirements.txt

# Flash Attention 2 (needs CUDA toolkit + ninja)
pip install flash-attn --no-build-isolation || echo "⚠ flash-attn install failed — continuing without it"

# Install project in editable mode
pip install -e .

echo ""
echo "=== Verifying ==="
python -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')"
python -c "import selfies, rdkit, transformers, trl, deepspeed; print('All packages OK')"
python -c "import flash_attn; print(f'flash-attn {flash_attn.__version__}')" || echo "flash-attn not available"

echo ""
echo "✓ Setup complete. Activate with: source .venv/bin/activate"
