#!/usr/bin/env bash
# fp16 (Accelerate mixed-precision) variant of train_diffusion_joint_valdiag.sh.
# Identical recipe to the fp32 JOINT diagnostic EXCEPT --policy.dtype=float16, which
# activates HF Accelerate mixed_precision=fp16 (auto GradScaler) in the train loop.
# Requires the locally-added `dtype` field on DiffusionConfig.
# Measured on A4000: fp16 ~4.4 step/s vs fp32 ~3.5 step/s (~1.25x); 80k ~= 5.1h.
# Stop point: 80k budget (repo's diffusion best-checkpoint / plateau >=60k); monitor
# train + held-out eval loss and cut early if plateaued. Select deploy ckpt by
# eval_offline.py open-loop MAE, NOT eval_loss (repo's established finding).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

SC="${SCRATCH_DIR:-$PWD/.cache}"
mkdir -p "$SC/torch_home" "$SC/hf_lerobot_home"
export TORCH_HOME="$SC/torch_home"
export HF_LEROBOT_HOME="$SC/hf_lerobot_home"
export HF_HUB_OFFLINE=1

STEPS="${STEPS:-80000}"

exec ./lr_env/bin/lerobot-train \
  --dataset.repo_id=theo/banana_in_pot \
  --dataset.root=./banana_in_pot_lerobot \
  --dataset.eval_split=0.117 \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.drop_n_last_frames=31 \
  --policy.resize_shape='[360,640]' \
  --policy.dtype=float16 \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=1 \
  --dataset.image_transforms.tfs='{"resize":{"weight":1.0,"type":"Resize","kwargs":{"size":[360,640]}}}' \
  --batch_size=8 \
  --steps="$STEPS" \
  --save_freq=20000 \
  --eval_steps=5000 \
  --max_eval_samples=480 \
  --num_workers=8 \
  --log_freq=200 \
  --seed=1000 \
  --wandb.enable=false \
  --job_name=diffusion_joint_fp16 \
  --output_dir=outputs/train/diffusion_joint_fp16
