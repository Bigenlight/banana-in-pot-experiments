#!/usr/bin/env bash
# 3-way smoke benchmark for Diffusion Policy JOINT: measure real A4000 step throughput and
# the data_s vs update_s split (bottleneck diagnosis) for fp32 vs fp16 and vs more workers.
# Runs a few hundred steps each, NO eval, NO kept checkpoints (deleted after each run).
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"

SC="${SCRATCH_DIR:-$PWD/.cache}"
mkdir -p "$SC/torch_home" "$SC/hf_lerobot_home"
export TORCH_HOME="$SC/torch_home"
export HF_LEROBOT_HOME="$SC/hf_lerobot_home"
export HF_HUB_OFFLINE=1

STEPS="${STEPS:-400}"
mkdir -p smoke_logs

run() {
  local tag="$1" dtype="$2" nw="$3"
  local out="outputs/train/smoke_${tag}"
  echo "======== SMOKE $tag : dtype=$dtype num_workers=$nw steps=$STEPS ========"
  rm -rf "$out"
  ./lr_env/bin/lerobot-train \
    --dataset.repo_id=theo/banana_in_pot \
    --dataset.root=./banana_in_pot_lerobot \
    --dataset.eval_split=0.117 \
    --policy.type=diffusion \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.drop_n_last_frames=31 \
    --policy.resize_shape='[360,640]' \
    --policy.dtype="$dtype" \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.max_num_transforms=1 \
    --dataset.image_transforms.tfs='{"resize":{"weight":1.0,"type":"Resize","kwargs":{"size":[360,640]}}}' \
    --batch_size=8 \
    --steps="$STEPS" \
    --save_freq=999999 \
    --eval_steps=0 \
    --log_freq=50 \
    --num_workers="$nw" \
    --seed=1000 \
    --wandb.enable=false \
    --job_name="smoke_${tag}" \
    --output_dir="$out" > "smoke_logs/${tag}.log" 2>&1
  local rc=$?
  echo "  exit=$rc  (last metric lines:)"
  grep -iE 'step:|dataloading|update_s|loss:' "smoke_logs/${tag}.log" | tail -4
  rm -rf "$out"   # reclaim disk immediately
  return $rc
}

run fp32_nw4  float32 4
run fp16_nw4  float16 4
run fp16_nw12 float16 12
echo "======== DONE ========"
