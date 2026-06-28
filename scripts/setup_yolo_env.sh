#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${1:-$ROOT/.conda-yolo}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required for this setup script" >&2
  exit 1
fi

if [ ! -d "$ENV_PREFIX" ]; then
  conda create -y -p "$ENV_PREFIX" python=3.11 pip
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision
python -m pip install -r "$ROOT/sources/yolov5/requirements.txt"
python -m pip install -r "$ROOT/sources/yolov9/requirements.txt"
python -m pip install -e "$ROOT/sources/yolov8"

python - <<'PY_CHECK'
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu0', torch.cuda.get_device_name(0))
from ultralytics import YOLO
print('ultralytics import ok')
PY_CHECK

echo "Environment ready: $ENV_PREFIX"
echo "Activate with: conda activate $ENV_PREFIX"
