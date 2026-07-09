#!/usr/bin/env bash
# OVERFIT DIAGNOSTIC run for Diffusion Policy (JOINT action space) on banana_in_pot.
# Mirrors train_act_valdiag.sh: holds out the LAST ceil(51*0.117)=6 episodes (45..50),
# trains on the other 45, logs held-out eval loss every --eval_steps.
# NOTE: diffusion held-out loss is STOCHASTIC (random noise+timestep per forward);
# interpret via smoothed trend, not single points.
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
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.drop_n_last_frames=31 \
  --policy.resize_shape='[360,640]' \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=1 \
  --dataset.image_transforms.tfs='{"resize":{"weight":1.0,"type":"Resize","kwargs":{"size":[360,640]}}}' \
  --batch_size=8 \
  --steps=100000 \
  --save_freq=10000 \
  --eval_steps=2000 \
  --log_freq=200 \
  --num_workers=4 \
  --seed=1000 \
  --wandb.enable=false \
  --job_name=diffusion_joint_val_diag \
  --output_dir=outputs/train/diffusion_joint_val_diag
