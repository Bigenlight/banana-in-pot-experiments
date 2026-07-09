#!/usr/bin/env bash
# GPU gate for the Diffusion Policy runs (DIFFUSION_PLAN.md §1), REVISED 2026-07-09.
# Exits 0 ONLY when the GPU is free for a CUDA training job; exits 1 otherwise.
# Poll before ANY CUDA work (JOINT smoke/full, EEF smoke/full, eval_offline).
#
# Revision rationale: this box has a DESKTOP attached (Xorg/gnome/TeamViewer/Chrome/
# VSCode) that permanently holds ~0.8-1.2 GB of GPU memory as GRAPHICS (Type G), with
# NO compute process. The old "<500 MiB total" test could never pass here, and the old
# "ACT reached step 80000" test never passes when ACT is early-stopped. The correct test
# for "can I start a training run" is: no lerobot-train process AND no CUDA COMPUTE
# (Type C) process is using the GPU. Graphics/display memory is ignored.
set -uo pipefail

fail=0

# --- Condition 1: no lerobot-train process ---------------------------------
if procs=$(pgrep -af 'lr_env/bin/lerobot-train'); then
  echo "FAIL [1/2] lerobot-train process still alive:"
  echo "$procs" | sed 's/^/         /'
  fail=1
else
  echo "PASS [1/2] no lerobot-train process alive"
fi

# --- Condition 2: no CUDA COMPUTE process on the GPU -----------------------
# nvidia-smi compute-apps lists ONLY Type-C (CUDA) processes; graphics/display
# memory does not appear here, so this ignores the desktop's ~1 GB baseline.
if ! apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null); then
  echo "FAIL [2/2] nvidia-smi compute-apps query failed (cannot verify GPU is free)"
  fail=1
elif [ -z "$(echo "$apps" | tr -d '[:space:]')" ]; then
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')
  echo "PASS [2/2] no CUDA compute process (GPU compute-free; ${mem:-?} MiB graphics/display baseline)"
else
  echo "FAIL [2/2] a CUDA compute process is using the GPU:"
  echo "$apps" | sed 's/^/         /'
  fail=1
fi

echo "----------------------------------------------------------------------"
if [ "$fail" -eq 0 ]; then
  echo "GATE OPEN: GPU is free for a CUDA training job."
  exit 0
else
  echo "GATE CLOSED: at least one condition failed - do NOT start CUDA work."
  exit 1
fi
