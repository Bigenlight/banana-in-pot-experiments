#!/usr/bin/env bash
# FAIR-COMPARISON run: ACT in EEF (EE 10-dim action space) on banana_in_pot.
#
# This is the EEF counterpart of train_act_valdiag.sh (JOINT). It is a byte-for-
# byte mirror of that script EXCEPT for three things:
#   1. dataset   -> theo/banana_in_pot_ee_action (10-D: xyz + 6D-rot + gripper)
#                   root ./banana_in_pot_ee_action_lerobot  (local-only, rebuilt
#                   by fetch_data.sh all)
#   2. steps     -> 50000  (matches the JOINT ACT baseline: valdiag settings, 50k)
#   3. job_name / output_dir -> act_ee_val_diag
#
# Everything else is IDENTICAL to the JOINT run for an apples-to-apples compare:
#   --dataset.eval_split=0.117  -> lerobot holds out the LAST ceil(51*0.117)=6
#     episodes (eps 45..50), the same held-out set as eval_offline.py; trains on
#     the other 45; logs held-out eval loss every --eval_steps.
#   batch 8, seed 1000, on-the-fly 360x640 resize, ACT policy defaults.
#
# ACT needs NO code change to switch JOINT->EEF: state/action dim (10) is auto-
# derived from the dataset features, and ACT's MEAN_STD normalization is per-dim
# so it handles the mixed xyz(m) + 6D-rotation + gripper scales fine.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

SC="${SCRATCH_DIR:-$PWD/.cache}"
mkdir -p "$SC/torch_home" "$SC/hf_lerobot_home"
export TORCH_HOME="$SC/torch_home"          # cached pretrained ResNet18 (home is read-only)
export HF_LEROBOT_HOME="$SC/hf_lerobot_home"
export HF_HUB_OFFLINE=1

exec ./lr_env/bin/lerobot-train \
  --dataset.repo_id=theo/banana_in_pot_ee_action \
  --dataset.root=./banana_in_pot_ee_action_lerobot \
  --dataset.eval_split=0.117 \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=1 \
  --dataset.image_transforms.tfs='{"resize":{"weight":1.0,"type":"Resize","kwargs":{"size":[360,640]}}}' \
  --batch_size=8 \
  --steps=50000 \
  --save_freq=10000 \
  --eval_steps=2000 \
  --log_freq=200 \
  --num_workers=4 \
  --seed=1000 \
  --wandb.enable=false \
  --job_name=act_ee_val_diag \
  --output_dir=outputs/train/act_ee_val_diag
