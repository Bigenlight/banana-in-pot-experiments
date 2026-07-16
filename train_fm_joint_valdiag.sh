#!/usr/bin/env bash
# OVERFIT DIAGNOSTIC run for FLOW MATCHING (JOINT action space) on banana_in_pot.
# Policy: lerobot `multi_task_dit` with objective=flow_matching  (rectified-flow /
# conditional-flow-matching action head, CLIP ViT-B/16 vision+text conditioning,
# small DiT transformer). This is the only non-VLA, train-from-scratch flow-matching
# policy in lerobot 0.6.1 (pin 8a74e0a) — pi0/smolvla/evo1 all need heavy pretrained
# VLM backbones. See EXPERIMENT_LOG for the investigation.
#
# Mirrors train_diffusion_joint_valdiag.sh methodology: holds out the LAST
# ceil(51*0.117)=6 episodes (45..50), trains on the other 45, logs held-out eval loss
# every --eval_steps. Uses multi_task_dit DEFAULT architecture/horizon (n_obs_steps=2,
# horizon=32, n_action_steps=24; drop_n_last_frames auto = 32-24-2+1 = 7).
#
# IMAGE HANDLING: CLIP ViT-B/16 requires exactly 224x224 input (fixed position
# embeddings). The policy default (image_resize_shape=None, crop=224) would RandomCrop a
# 224x224 patch out of the full 720x1280 frame -> sees only a tiny sliver and could crop
# a banana out of view. We instead resize the WHOLE frame to 224x224 and make the crop a
# no-op (crop==resize) => resize-only, no crop augmentation. This matches the diffusion
# runs' deliberate crop-OFF choice and is safe for the banana-SELECTION task (never cuts
# a banana). Minor aspect squish 1280x720 -> 224x224 is acceptable for CLIP features.
#
# NOTE: like diffusion, the held-out flow-matching loss is STOCHASTIC (random noise +
# timestep per forward) -> interpret via smoothed trend, and select the deploy checkpoint
# by open-loop MAE (eval_offline.py), NOT by eval_loss.
#
# PREREQ: CLIP weights (openai/clip-vit-base-patch16) must be in the HF cache BEFORE this
# runs, because HF_HUB_OFFLINE=1 below blocks downloads. One-time prefetch:
#   HF_HUB_OFFLINE=0 ./lr_env/bin/python -c "from transformers import CLIPVisionModel, CLIPTextModel; \
#     CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch16'); \
#     CLIPTextModel.from_pretrained('openai/clip-vit-base-patch16')"
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

SC="${SCRATCH_DIR:-$PWD/.cache}"
mkdir -p "$SC/torch_home" "$SC/hf_lerobot_home"
export TORCH_HOME="$SC/torch_home"
export HF_LEROBOT_HOME="$SC/hf_lerobot_home"
export HF_HUB_OFFLINE=1

# STEPS overridable for smoke tests: STEPS=2 ./train_fm_joint_valdiag.sh
STEPS="${STEPS:-100000}"

exec ./lr_env/bin/lerobot-train \
  --dataset.repo_id=theo/banana_in_pot \
  --dataset.root=./banana_in_pot_lerobot \
  --dataset.eval_split=0.117 \
  --policy.type=multi_task_dit \
  --policy.objective=flow_matching \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.image_resize_shape='[224,224]' \
  --policy.image_crop_shape='[224,224]' \
  --batch_size=8 \
  --steps="$STEPS" \
  --save_freq=10000 \
  --eval_steps=2000 \
  --log_freq=200 \
  --num_workers=4 \
  --seed=1000 \
  --wandb.enable=false \
  --job_name=fm_joint_val_diag \
  --output_dir=outputs/train/fm_joint_val_diag
