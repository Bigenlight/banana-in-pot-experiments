#!/usr/bin/env bash
# Resume ACT training for banana_in_pot from the 50k checkpoint to 80k steps.
# Uses lerobot's native --resume: loads optimizer/rng/step from checkpoints/last
# (step=50000) and continues to --steps=80000. save_freq/img-transforms/etc. all
# come from the saved train_config.json, so we only override steps.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

SC="${SCRATCH_DIR:-$PWD/.cache}"
mkdir -p "$SC/torch_home" "$SC/hf_lerobot_home"
export TORCH_HOME="$SC/torch_home"          # cached pretrained ResNet18 (home is read-only)
export HF_LEROBOT_HOME="$SC/hf_lerobot_home"
export HF_HUB_OFFLINE=1

exec ./lr_env/bin/lerobot-train \
  --config_path=outputs/train/act_banana_in_pot/checkpoints/last/pretrained_model/train_config.json \
  --resume=true \
  --steps=80000
