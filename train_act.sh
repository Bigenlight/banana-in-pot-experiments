#!/usr/bin/env bash
# Standard ACT training for the banana_in_pot LeRobot dataset (RTX 3060 12GB).
# Images resized on-the-fly to 360x640 (aspect-preserving half-720p); no dataset re-encode.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

SC="${SCRATCH_DIR:-$PWD/.cache}"
mkdir -p "$SC/torch_home" "$SC/hf_lerobot_home"
export TORCH_HOME="$SC/torch_home"          # cached pretrained ResNet18 (home is read-only)
export HF_LEROBOT_HOME="$SC/hf_lerobot_home"
export HF_HUB_OFFLINE=1

exec ./lr_env/bin/lerobot-train \
  --dataset.repo_id=theo/banana_in_pot \
  --dataset.root=./banana_in_pot_lerobot \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=1 \
  --dataset.image_transforms.tfs='{"resize":{"weight":1.0,"type":"Resize","kwargs":{"size":[360,640]}}}' \
  --batch_size=8 \
  --steps=50000 \
  --save_freq=10000 \
  --log_freq=200 \
  --num_workers=4 \
  --seed=1000 \
  --wandb.enable=false \
  --job_name=act_banana_in_pot \
  --output_dir=outputs/train/act_banana_in_pot
