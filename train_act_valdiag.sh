#!/usr/bin/env bash
# OVERFIT DIAGNOSTIC run for ACT banana_in_pot.
# A FRESH training (NOT the deploy model) whose only purpose is to measure the
# train-vs-held-out gap. lerobot holds out the LAST ceil(51*0.117)=6 episodes
# (eps 45..50, matching eval_offline convention), trains on the other 45, and
# logs held-out eval loss every --eval_steps so we can see if val loss stalls
# while train loss keeps dropping (= overfit).
#
# Separate output_dir so it never touches outputs/train/act_banana_in_pot.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

SC="${SCRATCH_DIR:-$PWD/.cache}"
mkdir -p "$SC/torch_home" "$SC/hf_lerobot_home"
export TORCH_HOME="$SC/torch_home"
export HF_LEROBOT_HOME="$SC/hf_lerobot_home"
export HF_HUB_OFFLINE=1

exec ./lr_env/bin/lerobot-train \
  --dataset.repo_id=theo/banana_in_pot \
  --dataset.root=./banana_in_pot_lerobot \
  --dataset.eval_split=0.117 \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=1 \
  --dataset.image_transforms.tfs='{"resize":{"weight":1.0,"type":"Resize","kwargs":{"size":[360,640]}}}' \
  --batch_size=8 \
  --steps=80000 \
  --save_freq=10000 \
  --eval_steps=2000 \
  --log_freq=200 \
  --num_workers=4 \
  --seed=1000 \
  --wandb.enable=false \
  --job_name=act_banana_val_diag \
  --output_dir=outputs/train/act_banana_val_diag
