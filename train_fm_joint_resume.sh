#!/usr/bin/env bash
# RESUME the flow-matching JOINT run (multi_task_dit, objective=flow_matching) from its
# last complete checkpoint. The original run (train_fm_joint_valdiag.sh) crashed at step
# 20000 when the SSD filled up mid-checkpoint-save (optimizer_state serialize -> ENOSPC);
# the `checkpoints/last` symlink correctly still points to 010000 (the last checkpoint
# whose full training_state — model + optimizer + rng + scheduler — was written), so we
# resume from 10k and lose only ~10k steps.
#
# lerobot resume: --config_path=<last>/pretrained_model/train_config.json reloads the FULL
# original config (dataset, policy, batch, steps=100000, output_dir, image flags, ...),
# and --resume=true restores step/optimizer/lr_scheduler/rng from training_state and
# continues in the SAME output_dir. CLIP weights are read from the HF cache (offline).
#
# PREREQ: free SSD space first (old runs were moved to /home/theo_lab/workspace HDD).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

SC="${SCRATCH_DIR:-$PWD/.cache}"
mkdir -p "$SC/torch_home" "$SC/hf_lerobot_home"
export TORCH_HOME="$SC/torch_home"
export HF_LEROBOT_HOME="$SC/hf_lerobot_home"
export HF_HUB_OFFLINE=1

CKPT_CFG=outputs/train/fm_joint_val_diag/checkpoints/last/pretrained_model/train_config.json

exec ./lr_env/bin/lerobot-train \
  --config_path="$CKPT_CFG" \
  --resume=true
