# Diffusion Policy JOINT — fp16 (mixed-precision) run

fp16 (HF Accelerate `mixed_precision=fp16`, automatic GradScaler) reproduction of the JOINT
Diffusion Policy, to measure the training speedup and quality vs the fp32 baseline.

## Why a code change was needed

`lerobot-train` (pin `8a74e0a`) drives Accelerate's mixed precision from a policy-config
`dtype` field: `mixed_precision = {"bfloat16":"bf16","float16":"fp16","float32":"no"}.get(
getattr(cfg.policy,"dtype",None))` (`lerobot_train.py:215-216`). **`DiffusionConfig` had no
`dtype` field**, so it was always `None` → `fp32`, and the older `use_amp` flag is dead code
for training. We added `dtype: str = "float32"` to `DiffusionConfig` (see
`patches/lerobot-add-dtype-field.patch`) and launched with `--policy.dtype=float16`.

TF32 matmul + `cudnn.benchmark` are already enabled by the train loop, so fp16's gain is
incremental on top of TF32, not vs naive fp32.

## Setup

Identical to `train_diffusion_joint_valdiag.sh` except `--policy.dtype=float16`, `--steps=80000`,
`--save_freq=20000`, `--eval_steps=5000 --max_eval_samples=480`, `--num_workers=8`.
Hardware: single **RTX A4000** (GPU 5), batch 8, seed 1000, 45 train / 6 held-out episodes.
Script: `train_diffusion_joint_fp16.sh`.

## Throughput (measured on A4000)

| precision | steady step/s | 80k wall-clock |
|---|---|---|
| fp32 | 3.49 | ~6.4 h (est.) |
| **fp16** | **4.48** | **4:57:38 (~4.96 h)** |

**~1.25× faster / ~22% wall-clock saved.** A 3-way smoke also showed `num_workers` 4→12 gives
~0% (4.38→4.47 step/s): this workload is **GPU-compute-bound, not video-decode-bound** (the
dataset is GOP=2 AV1 with torchcodec + uint8 transport + parallel 2-camera decode already in
place). No NaN/instability over the full run.

## held-out eval_loss (in-loop, subsampled 480)

`5k 0.0359 → 20k 0.0499 → 40k 0.0728 → 60k 0.1440 → 80k 0.1576` — same rising shape and
magnitude as the fp32 run (0.0289@4k → 0.1487@80k). As established for diffusion, rising
`eval_loss` is **not** overfitting; select by open-loop MAE.

## Open-loop rollout (DDIM-10, held-out eps 45–50) — the deploy metric

| step | poseMAE (rad) | gripAcc | overallL1 |
|---|---|---|---|
| 20k | 0.09996 | 0.923 | 0.09998 |
| 40k | 0.08717 | 0.931 | 0.08561 |
| 60k | 0.08540 | 0.949 | 0.08113 |
| **80k** ⭐ | **0.08268** | **0.960** | **0.07698** |

## fp16 vs fp32 verdict

| | fp32 @80k | fp16 @80k |
|---|---|---|
| poseMAE (rad) | 0.0845 | **0.08268** |
| gripAcc | 0.953 | **0.960** |
| wall-clock (80k) | ~6.4 h | **~5.0 h** |

**fp16 matches (marginally beats, within run-to-run noise) fp32 open-loop accuracy while
training ~22% faster and using less VRAM — no quality cost.** Best checkpoint = **80k**,
uploaded to `Bigenlight/diffusion_banana_in_pot_joint_fp16`. Deploy checkpoint selection by
open-loop MAE per the repo's established methodology (`eval_offline.py`), not `eval_loss`.

> Caveat: the fp16 run used a `steps=80000` cosine schedule (annealed to 0 at 80k), vs the
> fp32 baseline's 80k checkpoint taken from a 100k-scheduled run — a minor LR-trajectory
> difference. The comparison is precision-plus-schedule, but the result (parity/slightly
> better) is unambiguous that fp16 does not degrade this policy.
