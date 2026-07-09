# TROUBLESHOOTING — banana-in-pot-experiments

Hard-won cautions from this experiment, written as **Symptom → Cause → Fix**. Most
of these cost real debugging time; the copy-pasteable command in each section is the
fix that actually worked. Read item 4 even if nothing is broken — it is the project's
headline methodological lesson.

All commands assume you are in the repo root and the env is set up (`./setup.sh`,
`./fetch_data.sh`). Scripts already bake in the non-obvious flags; the sections below
explain *why* they are there so you do not "clean them up" and reintroduce the bug.

## Table of contents

1. [Training dies when the terminal/session closes](#1-training-dies-when-the-terminalsession-closes)
2. [Diffusion crashes on the FIRST forward with a shape/size error](#2-diffusion-crashes-on-the-first-forward-with-a-shapesize-error)
3. [IndexError / weird sampling near episode ends (diffusion)](#3-indexerror--weird-sampling-near-episode-ends-diffusion)
4. [eval_loss is going UP — is it overfitting?](#4-eval_loss-is-going-up--is-it-overfitting)
5. [Permission denied / cache written to weird places](#5-permission-denied--cache-written-to-weird-places)
6. [gpu_gate.sh passes/fails unexpectedly](#6-gpu_gatesh-passesfails-unexpectedly)
7. [How do I resume an interrupted run?](#7-how-do-i-resume-an-interrupted-run)
8. [What exactly is the val split?](#8-what-exactly-is-the-val-split)
9. [OOM at batch 8](#9-oom-at-batch-8)
10. [EE-action dataset rebuild fails](#10-ee-action-dataset-rebuild-fails)

---

## 1. Training dies when the terminal/session closes

**Symptom.** You launch a multi-hour training run, walk away (or the SSH/agent
session ends), and the run is gone — no checkpoints past where it stopped, no live
process.

**Cause.** A plain `./script.sh &` stays in the launching session's process group.
When that session closes it gets SIGHUP/SIGTERM'd. Agent harnesses (Claude Code and
similar) are worse: they actively **reap** child processes at turn boundaries, so a
backgrounded job that is not fully detached from the session dies within one turn.

**Fix.** Detach into a new session with `setsid` + `nohup`, redirecting stdin from
`/dev/null` so nothing blocks on the closed TTY:

```bash
setsid nohup ./train_diffusion_joint_valdiag.sh </dev/null > train_dj.log 2>&1 &
```

**Wrapper-PID vs real-PID gotcha.** `setsid` (and the shell `&`) hand you back the PID
of a *wrapper*, not the actual `lerobot-train` process — so `kill`, `wait`, or
`nvidia-smi` PID matching against that number silently does nothing. Get the real main
process by pattern-matching the launched command and taking the lowest PID (the parent):

```bash
pgrep -f 'lr_env/bin/lerobot-train.*diffusion_joint' | sort -n | head -1
```

Use that PID to monitor or to stop the run. `tail -f train_dj.log` to watch progress.

---

## 2. Diffusion crashes on the FIRST forward with a shape/size error

**Symptom.** A diffusion run explodes on the very first training step (or the first
`eval_offline.py` forward) with a tensor shape / size mismatch deep inside the RGB
encoder — before any useful training happens. The exact same dataset trains ACT fine.

**Cause.** `DiffusionRgbEncoder`'s `SpatialSoftmax` layer is **input-shape-rigid**: its
learned spatial coordinate grid is sized from the configured input resolution at
construction, so the feature-map dimensions must match exactly. The dataset `meta`
declares camera frames as **720×1280**, but the training dataloader resizes images to
**360×640** (via the `ImageTransforms` Resize baked into the train scripts). Diffusion
sees 360×640 tensors through a 720×1280-shaped encoder and dies. ACT's encoder is not
shape-rigid in the same way, so it tolerates the mismatch — which is why this bites
only when you move from ACT to diffusion.

**Fix.** Tell the diffusion policy its true input resolution so the encoder is built at
360×640:

```bash
--policy.resize_shape='[360,640]'
```

This flag is already in `train_diffusion_joint_valdiag.sh` and
`train_diffusion_ee_valdiag.sh` (alongside the matching `ImageTransforms` Resize to
`[360,640]`). Do **not** remove it, and keep the two resolutions in sync.

---

## 3. IndexError / weird sampling near episode ends (diffusion)

**Symptom.** Diffusion training throws an `IndexError` or samples oddly-truncated
windows near the ends of episodes (worse on this dataset's shorter episodes).

**Cause.** lerobot's diffusion config defaults to `drop_n_last_frames=7`, tuned for the
default horizon. This experiment uses `n_obs_steps=2`, `horizon=64`,
`n_action_steps=32`; with that much lookahead, dropping only 7 trailing frames leaves
the sampler able to request an action window that runs off the end of an episode.

**Fix.** Drop enough trailing frames to cover the sampling window:

```bash
--policy.drop_n_last_frames=31
```

Already set in both diffusion train scripts. Rule of thumb if you change the horizon:
`drop_n_last_frames` must cover the last valid start index, i.e. roughly
`horizon - n_action_steps + n_obs_steps - 1` = `64 - 32 + 2 - 1 = 33`-ish; the tested,
known-good value on these episodes is **31**.

---

## 4. eval_loss is going UP — is it overfitting?

**Symptom.** Your held-out `eval_loss` (the val curve in the overfit-diagnostic plot)
bottoms out early and then climbs steadily. For the JOINT diffusion run it went from
~0.029 at step 4k to ~0.067 at step 44k — a textbook "overfitting from step ~6k"
picture. First instinct: early-stop.

**Cause (the headline lesson).** For a **diffusion** policy, held-out denoising
`eval_loss` is a **misleading** overfit / early-stop signal. It scores *noise
prediction at random diffusion timesteps*, which **decorrelates** from the quality of
the actual *sampled action*. On this run the open-loop rollout MAE — the
deployment-relevant metric — kept **improving** while eval_loss rose:

| checkpoint | poseMAE (rad) ↓ | gripAcc ↑ | overall L1 ↓ |
|---|---|---|---|
| 10k | 0.1193 | 0.729 | 0.1454 |
| 20k | 0.1037 | 0.888 | 0.1078 |
| 30k | 0.0921 | 0.919 | 0.0928 |
| **40k** | **0.0907** | **0.949** | **0.0862** |

poseMAE and gripper accuracy improve monotonically 10k→40k, directly contradicting the
denoising-loss "overfit" reading. (For **ACT** the two signals agreed — there, a rising
eval_loss really did mean overfitting.)

**Fix.** Judge diffusion checkpoints by **open-loop rollout MAE**, not eval_loss. Run
the offline eval on the held-out episodes (45–50) with fast DDIM sampling and pick the
checkpoint with the lowest poseMAE / highest gripAcc:

```bash
./lr_env/bin/python eval_offline.py \
  --checkpoint outputs/train/diffusion_joint_val_diag/checkpoints/040000/pretrained_model \
  --policy-type diffusion --scheduler DDIM --num-inference-steps 10 \
  --episodes 45,46,47,48,49,50 --device cuda --out eval_out_40k
```

Sweep it across the saved checkpoints (10k, 20k, …) and compare. Best available
joint-diffusion checkpoint on this run was **40k**; the open-loop trend had not yet
plateaued, so more steps could help — the opposite of what eval_loss suggested.

Full write-up and the two-signal comparison: [`docs/DIFFUSION_JOINT_OVERFIT.md`](docs/DIFFUSION_JOINT_OVERFIT.md).

---

## 5. Permission denied / cache written to weird places

**Symptom.** A run fails with "permission denied" writing a cache/checkpoint, or tries
to download a pretrained backbone (e.g. ResNet18) and stalls/fails offline, or writes
into someone else's home / a read-only or session-specific directory.

**Cause.** Torch and HF/lerobot default their caches to `$HOME`-relative or
process-environment paths that may be read-only, missing, or session-specific on
another machine. And the ImageNet-pretrained backbone tries to hit the network.

**Fix.** Point the three cache/offline knobs at writable dirs. The repo scripts already
do this, defaulting the cache to `<repo>/.cache` and honoring a `SCRATCH_DIR` override:

```bash
export SCRATCH_DIR="$PWD/.cache"          # or any writable dir with room
export TORCH_HOME="$SCRATCH_DIR/torch_home"
export HF_LEROBOT_HOME="$SCRATCH_DIR/hf_lerobot_home"
export HF_HUB_OFFLINE=1                    # never phone home; use local dataset/backbone
```

**Before the first run on any new machine**, guard against stale hardcoded paths
leaking in from the original session — this check must come back empty:

```bash
grep -rn '/home/theo_lab\|/tmp/claude' *.sh *.py
```

If it prints anything, edit those lines to use `SCRATCH_DIR` / `$(dirname "$0")`
relative paths before running.

---

## 6. gpu_gate.sh passes/fails unexpectedly

**Symptom.** `./gpu_gate.sh` reports the GPU busy when `nvidia-smi` shows only ~1 GB
used (so "surely it's free?"), or you expected it to gate on a memory threshold and it
doesn't.

**Cause.** `gpu_gate.sh` deliberately checks **processes, not memory**. It opens the
gate only when BOTH hold:

1. no `lerobot-train` process is alive (`pgrep -af 'lr_env/bin/lerobot-train'`), and
2. no CUDA **compute** (Type-C) process appears in
   `nvidia-smi --query-compute-apps`.

A desktop workstation session (Xorg / GNOME / Chrome / VSCode / TeamViewer) permanently
holds ~0.8–1.2 GB as **graphics** memory (Type-G), which does **not** show up in the
compute-apps query and is intentionally ignored. So ~1 GB "used" is expected and the
gate still opens. On a headless server there is no such baseline and the gate trivially
passes.

**Fix.** Nothing to fix — this is by design. Read the gate output: it tells you which of
the two conditions failed. If a real compute process is listed, that is a genuine
in-flight CUDA job; wait for it or kill it (see item 1 for finding the real PID). **Do
not** rewrite the gate into a memory-threshold check — a `<500 MiB total` test could
never pass on this workstation, which is exactly why the process-based logic exists.

---

## 7. How do I resume an interrupted run?

**Symptom.** A run was interrupted (crash, reboot, early-stop) and you want to continue
from the last checkpoint without restarting from step 0 or losing optimizer/RNG state.

**Cause.** N/A — this is a how-to. lerobot supports native resume that restores the
optimizer, RNG, global step, and data ordering from a checkpoint's saved config.

**Fix.** Point `--config_path` at the checkpoint's `train_config.json`, set
`--resume=true`, and (optionally) a new `--steps` target:

```bash
./lr_env/bin/lerobot-train \
  --config_path=outputs/train/diffusion_joint_val_diag/checkpoints/last/pretrained_model/train_config.json \
  --resume=true \
  --steps=100000
```

Everything else (save_freq, image transforms, batch size, seed, …) comes from the saved
`train_config.json`, so you normally override only `--steps`. This is exactly how
`train_act_resume_80k.sh` continues ACT from 50k → 80k.

**Note.** On resume, the tqdm progress bar shows the **remaining** steps, not the
absolute global step — a resume to 100k that starts at 45k shows a bar counting the
~55k left, which looks alarming but is correct.

---

## 8. What exactly is the val split?

**Symptom.** You want to know which episodes are training vs held-out, and whether
`eval_offline.py` evaluates the same ones the training-time eval loss used.

**Cause.** N/A — clarification. The split is a fraction that rounds up.

**Fix / facts.** `--dataset.eval_split=0.117` holds out the **last**
`ceil(51 * 0.117) = ceil(5.967) = 6` episodes — indices **45, 46, 47, 48, 49, 50** —
and trains on the other 45. `eval_offline.py` defaults to exactly those held-out
episodes (`DEFAULT_EPISODES = [45, 46, 47, 48, 49, 50]`), so the offline open-loop eval
and the training-time held-out eval loss look at the **same** unseen episodes. If you
change `eval_split`, update the `--episodes` you pass to `eval_offline.py` to match, or
you will be evaluating on episodes the model trained on (data leakage).

---

## 9. OOM at batch 8

**Symptom.** Training dies with a CUDA out-of-memory error at `--batch_size=8`.

**Cause.** Batch 8 in fp32 uses ~**9.7 GB** on this workload — fine on the 12 GB RTX
3060 it was tuned for, but over budget on smaller cards (and other processes eat into
the headroom).

**Fix.** Reduce the batch size (and/or enable AMP). The OOM ladder actually tried during
the experiment was **8 → 8+amp → 6+amp → 4+amp**:

```bash
# on a smaller GPU, e.g.:
--batch_size=4
# (add mixed precision if your lerobot build exposes it, to recover some throughput)
```

Expect **different loss curves** at a different batch size — the numbers here were
recorded at batch 8, so smaller batches will match in shape (curve trend, MAE ranking)
but not digit-for-digit. Bitwise reproduction is an explicit non-goal.

---

## 10. EE-action dataset rebuild fails

**Symptom.** `convert_to_lerobot_ee_action.py` errors out — missing files, can't find
videos, or nothing to copy — when building `banana_in_pot_ee_action_lerobot`.

**Cause.** The 10-dim EEF-action dataset is built by **video-reuse**: it copies the
already-AV1-encoded camera videos from the JOINT dataset (no re-encode) and only
regenerates the parquet + stats from the raw h5 pose data. So the rebuild **requires two
inputs present locally first**: the JOINT dataset (`banana_in_pot_lerobot`) for the
videos, and the raw h5 (`Put_right_banana_in_the_pot`) for the EEF poses. If either is
absent, the copy/convert step fails.

**Fix.** Fetch both inputs first (`./fetch_data.sh` handles this), verify the transforms
with the self-test (needs no data), then build and validate:

```bash
# 0. transform unit tests only — no data needed, run this FIRST
./lr_env/bin/python convert_to_lerobot_ee_action.py --selftest

# 1. rebuild (both inputs must already exist locally)
./lr_env/bin/python convert_to_lerobot_ee_action.py \
  --data ./Put_right_banana_in_the_pot \
  --source ./banana_in_pot_lerobot \
  --out ./banana_in_pot_ee_action_lerobot \
  --repo-id theo/banana_in_pot_ee_action

# 2. validate — expect 51 eps / 21524 frames, range checks pass
./lr_env/bin/python validate_ee_dataset.py
```

If `--out` already exists, the script refuses to overwrite — remove it or pass a fresh
`--out`. This dataset is **local-only** (not on HF at time of writing), so the rebuild
chain is the canonical way to obtain it.
