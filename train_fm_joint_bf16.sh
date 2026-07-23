#!/usr/bin/env bash
# bf16 (Accelerate mixed-precision) variant of train_fm_joint_valdiag.sh.
# Flow matching (multi_task_dit, objective=flow_matching, CLIP ViT-B/16 + DiT).
# Identical recipe EXCEPT --policy.dtype=bfloat16, which activates HF Accelerate
# mixed_precision=bf16 in the train loop. bf16 (not fp16) is used deliberately: CLIP
# ViT + DiT attention have documented fp16 NaN/overflow history; bf16 keeps fp32's
# exponent range and needs no GradScaler. Requires the locally-added `dtype` field on
# MultiTaskDiTConfig. CLIP weights are pre-cached in .cache/hf/hub (HF_HUB_OFFLINE=1).
# Image handling identical to the fp32 run: whole frame resized to 224x224, crop no-op.
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
  --policy.type=multi_task_dit \
  --policy.objective=flow_matching \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.image_resize_shape='[224,224]' \
  --policy.image_crop_shape='[224,224]' \
  --policy.dtype=bfloat16 \
  --batch_size=8 \
  --steps="$STEPS" \
  --save_freq=20000 \
  --eval_steps=5000 \
  --max_eval_samples=480 \
  --num_workers=8 \
  --log_freq=200 \
  --seed=1000 \
  --wandb.enable=false \
  --job_name=fm_joint_bf16 \
  --output_dir=outputs/train/fm_joint_bf16
