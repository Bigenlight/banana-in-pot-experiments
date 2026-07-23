#!/usr/bin/env bash
# Open-loop eval of the fp16 diffusion JOINT checkpoints (DDIM-10), mirroring the repo's
# fp32 methodology so poseMAE/gripAcc are comparable to the fp32 run (0.0845 rad / 0.953 @80k).
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
SC="${SCRATCH_DIR:-$PWD/.cache}"
export TORCH_HOME="$SC/torch_home" HF_LEROBOT_HOME="$SC/hf_lerobot_home" HF_HUB_OFFLINE=1
mkdir -p results

for STEP in 020000 040000 060000 080000; do
  echo "======== EVAL fp16 diffusion @ $STEP (DDIM-10) ========"
  ./lr_env/bin/python eval_offline.py \
    --checkpoint outputs/train/diffusion_joint_fp16/checkpoints/$STEP/pretrained_model \
    --root ./banana_in_pot_lerobot \
    --repo-id theo/banana_in_pot \
    --episodes 45,46,47,48,49,50 \
    --scheduler DDIM --num-inference-steps 10 \
    --device cuda \
    --out results/eval_diff_fp16_$STEP 2>&1 | tee results/eval_diff_fp16_$STEP.log | grep -iE 'poseMAE|gripAcc|overallL1|MAE|acc' | tail -6
done
echo "======== EVAL DONE ========"
