#!/usr/bin/env bash
#
# setup.sh — recreate the `lr_env` Python environment on a fresh PC.
#
# Rebuilds the exact env used for the banana-in-pot experiments:
#   Python 3.12, torch 2.11.0+cu128, torchvision 0.26.0+cu128, torchcodec 0.11.1,
#   diffusers 0.35.2, lerobot 0.6.1 (editable, pinned clone).
#
# Measured ground truth (this box, 2026-07-09): uv 0.11.8, Python 3.12.3,
# 119-package freeze; requirements-lock.txt is that freeze minus the editable
# lerobot line (installed separately in step 3 so the clone exists first).
#
# CUDA / driver assumption: the pinned wheels are cu128, so the host needs an
# NVIDIA driver new enough for CUDA 12.8 (>= 570 series; this box: 590.48.01 on
# an RTX 3060 12 GB). ffmpeg must be on PATH (torchcodec / av need it). VRAM:
# batch 8 fp32 uses ~9.7 GB; on smaller GPUs reduce --batch_size.
#
# CPU / other-GPU fallback: swap the --extra-index-url below for
#   https://download.pytorch.org/whl/cpu   (or the matching cuXXX)
# AND strip the "+cu128" local-version suffixes from requirements-lock.txt
# (torch==2.11.0, torchvision==0.26.0). CPU is viable only for
# eval_offline.py --smoke and the dataset converters, NOT for training.
#
# Does NOT touch the GPU or download datasets — that is fetch_data.sh.

set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

# 0. prereqs: uv (any >= 0.11). Install via the official Astral script if missing.
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # the installer drops uv in ~/.local/bin; make it visible for the rest of this run
    export PATH="$HOME/.local/bin:$PATH"
fi

# 1. venv
uv venv lr_env --python 3.12

# 2. locked deps — the +cu128 wheels resolve via the pytorch extra index.
uv pip install --python ./lr_env/bin/python \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r requirements-lock.txt

# 3. lerobot, pinned + editable (installed last so the clone exists; --no-deps
#    because every dependency is already pinned by requirements-lock.txt).
LEROBOT_PIN=8a74e0ac6d01706d67fddfed682a09d694d9c8c0   # lerobot 0.6.1
[ -d lerobot ] || git clone https://github.com/huggingface/lerobot.git lerobot
git -C lerobot checkout "$LEROBOT_PIN"
uv pip install --python ./lr_env/bin/python --no-deps -e ./lerobot

# 4. smoke test — prints versions and CUDA availability.
#    Expect: 2.11.0+cu128 True 0.6.1 0.35.2   (CUDA "True" only on a cu128 host)
./lr_env/bin/python - <<'PY'
import torch, lerobot, diffusers
print("torch     ", torch.__version__)
print("cuda avail", torch.cuda.is_available())
print("lerobot   ", lerobot.__version__)
print("diffusers ", diffusers.__version__)
PY

echo "setup.sh: done. Environment is in ./lr_env (activate with: source lr_env/bin/activate)."
