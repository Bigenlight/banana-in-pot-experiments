---
license: apache-2.0
library_name: lerobot
tags:
  - robotics
  - lerobot
  - flow-matching
  - multi-task-dit
  - imitation-learning
  - ur7e
  - manipulation
pipeline_tag: robotics
datasets:
  - Bigenlight/banana_in_pot_lerobot_v3
---

# Flow-Matching Policy — Put the right banana in the pot (UR7e, JOINT action space)

A **flow-matching visuomotor policy** (`multi_task_dit`, `objective=flow_matching`) trained
by imitation learning to perform the manipulation task *"put the right banana in the pot"*
on a **Universal Robots UR7e** arm with two RGB cameras. Actions are **7-D absolute joint
targets** (6 UR joints in radians + gripper).

- **Policy:** LeRobot `multi_task_dit` with `--policy.objective=flow_matching` — a
  conditional / rectified **flow-matching action head** (linear-interpolation probability
  path, velocity target, **Euler ODE integration** at inference) on a small **DiT
  transformer** denoiser (hidden_dim = 512, 6 layers, 8 heads, RoPE), conditioned on
  **CLIP ViT-B/16** vision **and text** features (shared encoder, text encoder frozen).
  Receding-horizon action generation: `horizon = 32`, `n_obs_steps = 2`,
  `n_action_steps = 24`.
- **Flow-matching process:** linear-interpolation path
  `x_t = t·data + (1 − (1 − σ)t)·noise`, velocity target `data − (1 − σ)·noise`, MSE on the
  predicted velocity field, `sigma_min = 0.0`, `timestep_sampling = beta`. Inference
  integrates the learned velocity field with **Euler** ODE steps (`num_integration_steps`,
  default 100; evals here used 10).
- **Trained on:** [`Bigenlight/banana_in_pot_lerobot_v3`](https://huggingface.co/datasets/Bigenlight/banana_in_pot_lerobot_v3)
  — 51 teleoperated episodes / 21,524 frames, UR7e follower + GELLO leader, 2 RGB cameras.
- **This checkpoint:** step **70,000** (best by open-loop held-out MAE; see
  [Results](#results--the-headline-finding)).
- **Framework:** [LeRobot](https://github.com/huggingface/lerobot) v0.6.1 (pin `8a74e0a`).

> **Why this policy?** `multi_task_dit` + `flow_matching` is the **only non-VLA,
> train-from-scratch flow-matching policy in LeRobot 0.6.1**. The other flow-matching /
> action-expert policies (pi0, pi0_fast, pi05, smolvla, evo1) all require heavy pretrained
> VLM backbones that do not fit a single RTX 3060 12 GB and/or need a language stack. This
> policy trains its DiT denoiser from scratch and only borrows a frozen CLIP ViT-B/16 for
> vision + text conditioning, so it fits the same 12 GB budget as the ACT and Diffusion
> siblings — making it a fair third data point in the same overfit-diagnostic study.

> **Headline finding (read this first):** on a held-out split the flow-matching **`eval_loss`
> ROSE ~2.6× (min ≈ 0.089 near 6k → 0.233 @ 100k)** over training, which naively screams
> "severe overfitting". But the deployment-relevant **open-loop rollout MAE kept IMPROVING**
> then plateaued, reaching its minimum at step 70k (0.0816 → 0.0735 rad). **For a
> flow-matching policy too, the held-out flow/denoising loss is a misleading
> overfit/early-stop signal — select checkpoints by open-loop MAE, not by `eval_loss`.**
> This now reproduces the diffusion sibling's lesson for a **third** policy family.

---

## Task & data

**"put the right banana in the pot."** The tabletop holds several distractor objects —
**two bananas, an apple, carrots/peppers, and a slice of watermelon** — plus a **silver
pot**. The operator must grasp the **RIGHT banana** (the target) and place it inside the
pot. Success = the right banana ends up inside the pot. Every demonstration is a success.

- **Dataset:** [`Bigenlight/banana_in_pot_lerobot_v3`](https://huggingface.co/datasets/Bigenlight/banana_in_pot_lerobot_v3)
  (LeRobot v3.0 format).
- **Scale:** **51 episodes / 21,524 frames / 30 fps / ~12 min.**
- **Action / state space:** 7-D absolute joint (6 UR joints in radians + gripper), i.e.
  `[cmd1..cmd6, grip_cmd]`. The gripper channel is effectively binary (open/close).
- **Cameras:** two RGB viewpoints (Intel RealSense D435 + D435if), captured at 1280×720
  (720p) @ 30 fps, **RGB only** (no depth / IR). `cam1 ↔ cam2` order is fixed and must be
  preserved at deploy time.
- **Text conditioning:** unlike the ACT and Diffusion siblings (which ignore the task
  string), this policy **is** text-conditioned through CLIP. The task string is fixed to
  `"put the right banana in the pot"` for both training and inference.

### Train / held-out split

Training used `--dataset.eval_split=0.117`, which holds out the **LAST
`ceil(51 × 0.117) = 6` episodes (indices 45–50)** as a true validation split and trains on
the other **45** episodes (0–44), i.e. **18,807 frames**. The held-out episodes 45–50 are
used both for the in-training flow-matching `eval_loss` probe and for all offline open-loop
evaluation below.

---

## Model architecture

LeRobot `multi_task_dit` with `objective = flow_matching`. All values below are quoted
directly from the checkpoint's `config.json` / `train_config.json`.

**Observation encoder (vision + text, CLIP ViT-B/16):**

| Item | Value |
|---|---|
| Vision + text backbone | **CLIP ViT-B/16** (`openai/clip-vit-base-patch16`) |
| Per-camera encoder | `use_separate_rgb_encoder_per_camera = false` (shared encoder across both views) |
| Text encoder | **frozen** (CLIP text tower; conditions on the task string) |
| Vision encoder LR | trained at reduced LR (`vision_encoder_lr_multiplier = 0.1`) |
| Cameras | 2 × RGB (`observation.images.cam1`, `observation.images.cam2`) |
| Network input resolution | **224 × 224** (`image_resize_shape = [224, 224]`; CLIP requires exactly 224 — see [why](#why-images-must-be-224224-and-must-not-be-pre-resized)) |
| Crop | **resize-only** — `image_crop_shape = [224, 224]` equals the resize, so the crop is a no-op (mirrors the diffusion runs' crop-OFF intent) |
| State input | `observation.state`, shape `(7,)` |

> Note: `config.json` records the raw dataset image feature shape as `[3, 720, 1280]`, but
> the policy resizes **internally** to `[224, 224]` before the CLIP encoder. Unlike the
> diffusion sibling (which expected an **external** 360×640 resize), this policy does its
> own 224×224 resize — feed it the **native/full-res** decoded frame and let it resize. See
> [why](#why-images-must-be-224224-and-must-not-be-pre-resized).

**Denoiser (DiT transformer):**

| Item | Value |
|---|---|
| Denoiser | **DiT** transformer (diffusion/flow transformer) |
| `hidden_dim` | `512` |
| `num_layers` | `6` |
| `num_heads` | `8` |
| Positional encoding | **RoPE** (rotary) |
| `dropout` | `0.1` |
| `horizon` | `32` (prediction horizon, in frames) |
| `n_obs_steps` | `2` (observation context length) |
| `n_action_steps` | `24` (actions executed before replanning) |

**Flow-matching process:**

| Item | Value |
|---|---|
| `objective` | `flow_matching` |
| Probability path | linear interpolation: `x_t = t·data + (1 − (1 − σ)t)·noise` |
| Velocity target | `data − (1 − σ)·noise` (MSE on predicted velocity field) |
| `sigma_min` | `0.0` |
| `timestep_sampling` | `beta` (Beta-distributed training timesteps) |
| Inference integrator | **Euler** ODE steps |
| `num_integration_steps` | `100` (default); evals here used **10** for ~10× faster rollouts |

**Normalization (`normalization_mapping`):**

| Feature group | Mode |
|---|---|
| `VISUAL` (images) | `MEAN_STD` (CLIP image stats) |
| `STATE` (observation.state) | `MIN_MAX` |
| `ACTION` (action) | `MIN_MAX` |

Normalizer statistics are baked into the pre/post-processor pipelines saved alongside the
checkpoint (`policy_preprocessor.json` / `policy_postprocessor.json`), not into `forward()`.
`select_action` returns a **normalized** action; the post-processor converts it back to
radians.

**I/O summary:**

| I/O | Spec |
|---|---|
| `observation.state` | `(7,)` — UR joints `q1..q6` (radians) + gripper position |
| `observation.images.cam1` / `cam2` | RGB, **native/full-res** in → policy resizes to 224 × 224 internally |
| `task` | `"put the right banana in the pot"` (text conditioning, via CLIP) |
| `action` | `(7,)` — `[cmd1..cmd6, grip_cmd]`, **absolute** joint targets (radians) + ~binary gripper |

**Parameter count:** ~**186 M learnable** parameters (~**249 M total** including the frozen
CLIP text tower).

### Why images must be 224×224 (and must NOT be pre-resized)

CLIP ViT-B/16 is built for a fixed **224 × 224** patch grid. This policy therefore sets
`image_resize_shape = [224, 224]` and performs the resize **inside** the policy's own image
transform (`image_crop_shape = [224, 224]` equals the resize, so the crop is a no-op). This
differs from the diffusion sibling, whose SpatialSoftmax encoder expected an **external**
360×640 resize.

**Practical consequence for deployment:** feed the decoded RGB frame at its **native / full
resolution** and let the policy resize to 224×224. **Do NOT pre-resize to 360×640** (that is
a diffusion-specific step and would double-resize / distort the input here). The same
internal 224×224 path is reproduced in `eval_offline.py`, so offline eval, training, and
deploy all feed CLIP identically.

---

## Training setup

Trained with `lerobot-train` (LeRobot 0.6.1, pin `8a74e0a`). Exact invocation:
`train_fm_joint_valdiag.sh`. Values below are from that script and the saved
`train_config.json`.

| Item | Value |
|---|---|
| Policy | `multi_task_dit` (`--policy.type=multi_task_dit`) |
| Objective | `flow_matching` (`--policy.objective=flow_matching`) |
| Dataset | `banana_in_pot_lerobot_v3`, `--dataset.eval_split=0.117` (holds out eps 45–50; trains on 45 eps / 18,807 frames) |
| Batch size | **8** |
| Steps | **100,000**; checkpoints saved every 10,000 |
| Optimizer | **AdamW**, `lr = 2e-5`, `betas = [0.95, 0.999]` |
| Vision encoder LR | `vision_encoder_lr_multiplier = 0.1` (CLIP vision trained at 0.1× LR; CLIP text **frozen**) |
| LR scheduler | **cosine**, `num_warmup_steps = 0` |
| Horizon config | `n_obs_steps = 2`, `horizon = 32`, `n_action_steps = 24` |
| `drop_n_last_frames` | **7** (auto = `horizon − n_action_steps − (n_obs_steps − 1) = 32 − 24 − 1`) |
| Seed | **1000** |
| Precision | **fp32** (`use_amp = false`) |
| Integration (inference) | **Euler**, `num_integration_steps = 100` default (`timestep_sampling = beta`, `sigma_min = 0.0`) |
| Eval probe | held-out flow-matching `eval_loss` during training |
| GPU | single **RTX 3060 12 GB**, ~**8 GB** used, ~**1.6 step/s** |
| Params | ~**186 M learnable** / ~**249 M total** (incl. frozen CLIP text) |
| W&B | disabled |

**Prerequisite — CLIP weights must be cached.** The policy loads
`openai/clip-vit-base-patch16` from the Hugging Face cache. Make sure those weights are
present (download once with network access) before training or offline eval, or
construction will fail.

### Repro note — crash & resume

The run **crashed at step 20k** when the SSD filled during a checkpoint save (ENOSPC). It was
**resumed from the 10k checkpoint** and completed 100k cleanly. To reproduce a resume:

```bash
# train_fm_joint_resume.sh
lerobot-train \
  --config_path=outputs/train/fm_joint_val_diag/checkpoints/010000/pretrained_model/train_config.json \
  --resume=true
```

`--config_path=<last_ckpt>/pretrained_model/train_config.json` plus `--resume=true` restores
the optimizer / scheduler / step counter. (Old runs were archived to an HDD to free space;
see `EXPERIMENT_LOG.md`.)

---

## Results & the headline finding

Offline **open-loop** evaluation on the held-out episodes **45–50** with `eval_offline.py`
(each logged observation is fed to `select_action`; the predicted action is compared to the
dataset ground truth). Sampling used **Euler with 10 integration steps**
(`--num-inference-steps 10`, which maps onto `num_integration_steps`) for ~10× faster
rollouts — the flow-matching analogue of the diffusion sibling's DDIM-10. `poseMAE` is the
mean absolute error over the 6 joint dims (radians); `gripAcc` is the binary
gripper-open/close accuracy (threshold 0.5); `overall L1` averages all 7 dims.

| checkpoint | poseMAE (rad) | gripAcc | overall L1 |
|---|---|---|---|
| 10k | 0.0816 | 0.953 | 0.0790 |
| 20k | 0.0850 | 0.953 | 0.0810 |
| 30k | 0.0763 | **0.960** | 0.0728 |
| 40k | 0.0775 | 0.954 | 0.0740 |
| 50k | 0.0753 | 0.949 | 0.0724 |
| 60k | 0.0773 | 0.953 | 0.0735 |
| **70k** ⭐ | **0.0735** (min) | 0.953 | **0.0703** (min) |
| 80k | 0.0745 | 0.952 | 0.0712 |
| 90k | 0.0748 | 0.952 | 0.0716 |
| 100k | 0.0746 | 0.952 | 0.0713 |

**Best checkpoint = 70k** (min poseMAE **and** min overall L1). Open-loop poseMAE improves
early (already ~0.076 by 30k) then settles onto a **~0.074 rad plateau from ~40k onward** (70k/80k/90k/100k =
0.0735/0.0745/0.0748/0.0746, within eval noise); gripper accuracy holds steady at
**~0.95–0.96**. There is **no destructive open-loop overfitting through 100k**.

### The misleading `eval_loss` (the lesson)

During training the held-out **flow-matching `eval_loss`** (LeRobot's in-training validation
probe, computed under `policy.eval()` on eps 45–50) did the opposite of the rollout metric:

| step | held-out eval_loss |
|---|---|
| ~6k | ~0.089 (min region) |
| 10k | 0.096 |
| 30k | 0.120 |
| 50k | 0.171 |
| 70k | 0.193 |
| 80k | 0.224 |
| 100k | 0.233 |

Read naively, the held-out `eval_loss` bottoms near step 6k and then rises **~2.6×**, so an
early-stop rule would pick **~step 6k** and declare "severe overfit". **That recommendation
is wrong for deployment:** the same held-out episodes, evaluated by open-loop rollout, get
*better* out to 70k and then hold flat.

**Why:** the flow-matching loss is a per-sample MSE on the predicted **velocity field**, at a
*randomly re-sampled noise vector and flow timestep at every forward pass* — it is (a)
high-variance/stochastic by construction and (b) only loosely coupled to closed-loop action
quality. As the model sharpens its learned action distribution, the average velocity-MSE on
unseen frames can rise even while the *sampled* (Euler-integrated) action trajectories become
more accurate. **Takeaway: for a flow-matching policy, select checkpoints and early-stop by
open-loop rollout MAE, not by held-out `eval_loss`.** This is now the **third** policy family
(after Diffusion) to show the divergence — the ACT sibling did not, so this is a pitfall of
denoising/flow-style generative action heads specifically.

### Comparison — Flow matching vs Diffusion vs ACT (JOINT)

Held-out eps 45–50, open-loop poseMAE (radians), best checkpoint of each family:

| policy | best poseMAE (rad) | @ checkpoint | best gripAcc |
|---|---|---|---|
| **Flow matching (`multi_task_dit`)** ⭐ | **0.0735** | 70k | ~0.96 |
| Diffusion Policy | 0.0845 | 80k | 0.953 |
| ACT | ~0.098 | 30k | ~0.94 |

**Flow matching is the best of the three on JOINT open-loop pose accuracy** — about **13%
lower poseMAE than Diffusion** and **~25% lower than ACT**, at comparable (excellent) gripper
accuracy. FM also **beats Diffusion at *every* checkpoint** (e.g. 10k: 0.0816 vs 0.1193; 30k:
0.0763 vs 0.0921) and converges fast (already on-plateau by ~30k).

> Caveat: eps 45–50 are the chronological tail (not an i.i.d. split); poseMAE is
> teacher-forced open-loop, not a closed-loop success rate; Euler-10 is a fast sampler (more
> integration steps could shift absolute numbers but not the ranking). Raw per-checkpoint
> metrics: `results/eval_fm_final_gpu.csv`.

---

## Usage / inference

### Load the policy (LeRobot 0.6.1)

Use the **generic loader** (`get_policy_class`) — it works for any policy type and resolves
to `MultiTaskDiTPolicy` here. Normalization is **not** baked into `forward()` in LeRobot
0.6.1; it lives in the pre/post-processor pipelines saved with the checkpoint.

> **Prerequisite:** `openai/clip-vit-base-patch16` must be in your Hugging Face cache
> (the CLIP vision + text towers), or construction fails.

```python
import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

CKPT = "Bigenlight/flow_matching_banana_in_pot_joint"
device = "cuda"

cfg = PreTrainedConfig.from_pretrained(CKPT)
cfg.pretrained_path = CKPT
cfg.device = device
cfg.num_integration_steps = 10        # optional: fewer Euler steps = faster (default 100)

policy = get_policy_class(cfg.type).from_pretrained(CKPT, config=cfg)  # -> MultiTaskDiTPolicy
policy.to(device)
policy.eval()

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=cfg,
    pretrained_path=CKPT,
    preprocessor_overrides={"device_processor": {"device": device}},
)
```

### Run the control loop

Build the observation dict exactly as training did: joint state `(7,)` plus **both** cameras
as RGB CHW tensors in `[0, 1]` at **native / full resolution** (the policy resizes to 224×224
internally — **do NOT pre-resize to 360×640**), plus the fixed **task string**. `cam1`/`cam2`
must map to the same physical viewpoints as at collection.

```python
policy.reset()          # once at the start of each episode/rollout
preprocessor.reset()
postprocessor.reset()

# obs = {
#   "observation.state":        state_7,          # (7,) float32, radians + gripper
#   "observation.images.cam1":  img1_chw,         # (3, H, W) float32 in [0,1], NATIVE res
#   "observation.images.cam2":  img2_chw,         # (3, H, W) float32 in [0,1], NATIVE res
#   "task": "put the right banana in the pot",    # text conditioning (CLIP) — required
# }

with torch.inference_mode():
    proc   = preprocessor(obs)             # rename -> add batch dim -> device -> normalize
    action = policy.select_action(proc)    # (1, 7) NORMALIZED
    action = postprocessor(action)         # (1, 7) radians, on cpu
q_target = action.squeeze(0).numpy()       # (7,) -> [cmd1..cmd6, grip_cmd]
```

`select_action` returns **one** action per call from an internal queue. Because
`n_action_steps = 24`, the policy integrates a fresh action sequence, executes 24 actions
from it, then replans (with `n_obs_steps = 2` frames of observation context) — i.e. it
**replans ~every 24 ticks (~0.8 s @ 30 Hz)**. Call `policy.reset()` at the start of every
episode to clear that queue. The gripper channel `grip_cmd` is ~binary — threshold at
`> 0.5 → close` and map to your gripper driver.

### Reproduce the offline evaluation

The repo's `eval_offline.py` runs the exact open-loop protocol used for the results table
(same internal 224×224 resize, same normalization via the saved processors). The
`--num-inference-steps` flag maps onto the flow-matching `num_integration_steps`:

```bash
python eval_offline.py \
  --checkpoint outputs/train/fm_joint_val_diag/checkpoints/070000/pretrained_model \
  --episodes 45,46,47,48,49,50 \
  --device cuda \
  --num-inference-steps 10 \
  --out eval_out_fm_70k
```

`--num-inference-steps 10` gives the ~10× rollout speedup; raise it (up to 100) to integrate
the velocity field more finely.

---

## Deployment on a real UR7e

Closed-loop deployment targets a real **UR7e** through the ROS 2 Humble stack in
[**Bigenlight/gello_software**](https://github.com/Bigenlight/gello_software), package
`gello_policy` — **exactly like the ACT and Diffusion JOINT models**. The 7-D joint action +
two-camera observation contract is **byte-identical** to the diffusion JOINT deploy, so the
**entire ROS / safety side is unchanged**:

- A **py3.10 ROS node** (`policy_leader_node`) acts as a *synthetic GELLO leader*, publishing
  `/gello/joint_states` at **30 Hz** and reusing the safety-tuned `gello_ur_bridge`
  **unmodified**.
- The node talks over a **localhost ZMQ REQ/REP** split to a **py3.12 policy server** that
  runs the torch / LeRobot inference. The split exists because Humble's `rclpy` is py3.10 but
  LeRobot needs py3.12.

**What's new for flow matching** — a dedicated py3.12 server **`fm_server.py`** (added in
`gello_policy` alongside `act_server.py` / `diffusion_server.py`), because:

1. it must load the policy **generically** (`get_policy_class`) rather than hardcoding
   `DiffusionPolicy`;
2. FM uses **`num_integration_steps` (Euler)** — not a DDIM scheduler; and
3. FM images **must NOT be externally resized to 360×640** — the policy does its own 224×224
   resize, so the server forwards the native frame; and
4. FM is **text-conditioned** (CLIP), so the server must send the real `task` string
   (`"put the right banana in the pot"`) into the obs each tick — unlike the single-task
   diffusion/ACT servers, which send an empty task.

The run script defaults to **Euler-10** integration (the sampler used for the results
above) to keep refill latency under the leader's fault timeout. Point `fm_server.py` at
this checkpoint (70k). See the deploy runbook
`gello_software/docs/ros2/GELLO_UR7E_FM_DEPLOY.md`.

Each control tick (target **30 Hz**) the pipeline:

1. reads the UR7e measured joints + gripper → `observation.state` `(7,)`;
2. grabs both camera frames, BGR→RGB, CHW `[0, 1]` at **native res** (no external resize) →
   `observation.images.cam1` / `cam2`, plus the fixed task string;
3. `preprocessor → policy.select_action → postprocessor` → `q_target` (7,);
4. streams `q_target[:6]` to the arm and drives the gripper from `grip_cmd`.

**Safety — actions are ABSOLUTE joint positions** (same guards as the diffusion card):

1. **Start near the dataset initial pose** before enabling the policy, or the first absolute
   command is a large jump.
2. **First-command jump guard:** if `max(|q_target − getActualQ()|)` exceeds a small
   threshold (~0.15 rad), **abort**.
3. **Clamp per-tick joint change** and clamp to UR software joint limits; run at reduced speed
   for first trials with a hand on the **E-stop**.
4. **`cam1`/`cam2` mapping is fixed** — swap the two views and the policy fails silently.
   Verify wiring every session.

---

## Limitations & intended use

- **Small, single-task lab dataset:** 51 demonstrations, one scene layout, one operator.
  Expect limited generalization to novel object arrangements, lighting, or camera placement.
- **Success-only demonstrations:** no failure/recovery data; not suited as-is for methods that
  need negative examples.
- **Offline metrics only:** the best checkpoint (70k) reaches **held-out poseMAE ≈ 0.0735
  rad** and gripper accuracy ≈ 0.95–0.96 in open-loop rollout. These are *not* closed-loop
  task success rates — real closed-loop success on hardware has not been measured here and
  must be validated on the arm.
- **Absolute-joint action space** demands the safety guards above; the policy was only ever
  conditioned on states near the data-collection start pose.
- **Not for production.** Intended for research in imitation learning / flow-matching policies
  for robot manipulation. Workspace-, robot-, and camera-specific.
- **Encoder provenance:** the CLIP ViT-B/16 vision + text towers are **web-pretrained** (not
  robotics-pretrained; text tower frozen, vision fine-tuned at 0.1× LR); the **DiT denoiser is
  trained from scratch** on this task.

---

## Links

- **This model:** [`Bigenlight/flow_matching_banana_in_pot_joint`](https://huggingface.co/Bigenlight/flow_matching_banana_in_pot_joint)
- **Dataset:** [`Bigenlight/banana_in_pot_lerobot_v3`](https://huggingface.co/datasets/Bigenlight/banana_in_pot_lerobot_v3)
- **Experiments repo:** [github.com/Bigenlight/banana-in-pot-experiments](https://github.com/Bigenlight/banana-in-pot-experiments)
- **Diffusion sibling model:** [`Bigenlight/diffusion_banana_in_pot_joint`](https://huggingface.co/Bigenlight/diffusion_banana_in_pot_joint)
- **ACT sibling model:** [`Bigenlight/act_banana_in_pot`](https://huggingface.co/Bigenlight/act_banana_in_pot)
- **Deployment stack (ROS 2 Humble):** [github.com/Bigenlight/gello_software](https://github.com/Bigenlight/gello_software)
- **Framework:** [LeRobot](https://github.com/huggingface/lerobot) v0.6.1 (pin `8a74e0a`)

## Citation

```bibtex
@misc{theo2026bananainpotflowmatching,
  title        = {Flow-Matching Policy for "put the right banana in the pot"
                  (UR7e, joint action space)},
  author       = {Theo and {Bigenlight}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/Bigenlight/flow_matching_banana_in_pot_joint}},
  note         = {LeRobot 0.6.1 multi_task_dit, objective=flow_matching,
                  trained on banana_in_pot_lerobot_v3}
}
```

License: **Apache-2.0**.
