# Deploying the ACT "banana-in-pot" policy on the real UR arm

This guide explains how to run the trained ACT policy on the **real Universal Robots
follower** that produced the dataset. GELLO was only the teleoperation *leader* during
data collection — it plays **no role at deployment**. At deploy time you drive the UR
follower directly and feed it the two cameras.

All API claims below are cited against the pinned repo:
`lerobot` **0.6.1** (see `lerobot/pyproject.toml:` `version = "0.6.1"`).

> lerobot has **no built-in UR robot class**. You therefore build the observation dict
> yourself and push joint targets with `ur_rtde`. The rest of the pipeline (checkpoint
> load, normalization, action chunking) uses the stock lerobot API exactly as `lerobot-eval`
> and the ACT tutorial do.

---

## 0. Dataset contract that MUST hold at deploy time

| Feature | Shape | Meaning |
|---|---|---|
| `observation.state` | `(7,)` | `[ur_q1..q6, grip_pos]`, joints in **radians** |
| `action` | `(7,)` | `[cmd1..cmd6, grip_cmd]`, **absolute** joint targets in **radians**, `grip_cmd` ~binary |
| `observation.images.cam1` | `(360, 640, 3)` | RGB |
| `observation.images.cam2` | `(360, 640, 3)` | RGB |

- **fps = 30** → control period **33.3 ms**.
- Images were trained at **360×640** (resized on-the-fly from 720p). **You MUST resize every
  live camera frame to 360×640** before inference. A different aspect ratio / interpolation
  than training will degrade the policy.
- Actions are **absolute joint positions**, not deltas. This is the single biggest safety
  driver (see §3).

---

## 1. Loading the checkpoint & how normalization is handled

### 1a. Load the policy

```python
from lerobot.policies.act import ACTPolicy
policy = ACTPolicy.from_pretrained("outputs/train/act_banana_in_pot/checkpoints/<step>/pretrained_model")
policy.eval()
```

- `ACTPolicy` is defined at `lerobot/src/lerobot/policies/act/modeling_act.py:42`.
- `from_pretrained` is inherited from `PreTrainedPolicy` at
  `lerobot/src/lerobot/policies/pretrained.py:162`. It reads `config.json` +
  `model.safetensors` from the `pretrained_model` directory.

### 1b. Load the pre/post processors from the SAME checkpoint (this is where normalization lives)

**Important 0.6.1 change:** normalization is **NOT baked into the policy `forward`
anymore.** In this refactor it lives in a separate *processor pipeline*. So
`policy.select_action(...)` returns a **NORMALIZED** action; you must run the
**post-processor** to get radians.

```python
from lerobot.policies import make_pre_post_processors

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path="outputs/train/act_banana_in_pot/checkpoints/<step>/pretrained_model",
)
```

- `make_pre_post_processors` is at `lerobot/src/lerobot/policies/factory.py:273`.
- When `pretrained_path` is given it **loads the pipelines saved in the checkpoint**
  (`policy_preprocessor.json` + `policy_postprocessor.json`, names from
  `lerobot/src/lerobot/utils/constants.py:58-59`) via
  `PolicyProcessorPipeline.from_pretrained` at `factory.py:323` (pre) and `factory.py:333`
  (post). The dataset stats used during training are **baked into these files**, so you do
  **not** need the dataset present at deploy time.

### 1c. What the pipelines do (confirmed from code)

`make_act_pre_post_processors` at
`lerobot/src/lerobot/policies/act/processor_act.py:35` builds:

**Pre-processor** (`processor_act.py:57-64`):
1. `RenameObservationsProcessorStep`
2. `AddBatchDimensionProcessorStep` — adds a batch dim only if missing
   (`lerobot/src/lerobot/processor/batch_processor.py:104-119`: state gets a dim if 1-D,
   images if 3-D → **idempotent**, safe to feed unbatched tensors).
3. `DeviceProcessorStep` — moves to `config.device`.
4. `NormalizerProcessorStep` — normalizes `observation.state` and the images.

**Post-processor** (`processor_act.py:66-72`):
1. `UnnormalizerProcessorStep` — **un-normalizes the action back to raw units (radians)**.
2. `DeviceProcessorStep(device="cpu")`.

Normalization mode for STATE / ACTION / VISUAL is `MEAN_STD`
(`lerobot/src/lerobot/policies/act/configuration_act.py:90-92`).

**Answer to "is raw `select_action` output already in radians?" → NO.**
`select_action` output is normalized (mean/std space). You get radians **only after
`postprocessor(...)`**. This mirrors the ACT tutorial
(`lerobot/examples/tutorial/act/act_using_example.py:51-52`):

```python
action = model.select_action(obs)   # normalized
action = postprocess(action)        # radians  <-- required
```

---

## 2. The 30 Hz control loop & ACT action chunking

### 2a. Action chunking — what `select_action` actually returns

`select_action` (`modeling_act.py:101-123`) returns **one action per call** from an internal
queue:

- On an empty queue it predicts a full chunk `predict_action_chunk(...)[:, :n_action_steps]`
  (`modeling_act.py:117-118`) and fills a `deque` of length `n_action_steps`
  (`modeling_act.py:98`).
- Each subsequent call just `popleft()`s the next precomputed action
  (`modeling_act.py:123`) — no network forward pass.

Defaults (`configuration_act.py:85-86`): `chunk_size = 100`, `n_action_steps = 100`. So with
defaults the policy predicts 100 steps, then replans after 100 executed actions. **Check your
own `config.json`** — if you trained with a smaller `n_action_steps`, it replans more often.

**Temporal ensembling** (`modeling_act.py:110-113`, ensembler class at `modeling_act.py:168`)
is active only if `temporal_ensemble_coeff is not None` (default `None`,
`configuration_act.py:119`). If enabled, `n_action_steps` must be 1
(`configuration_act.py:138`) and every step calls the network. Most ACT trainings ship with
it **off** — verify in `config.json`.

### 2b. Reset the queue before every episode

```python
policy.reset()   # modeling_act.py:93  -> clears the action queue / ensembler
```

Call this **once at the start of each rollout**, otherwise you execute stale actions from a
previous episode.

### 2c. Loop structure (per 33.3 ms tick)

1. Read UR joints `q1..q6` (rad) via `ur_rtde` `RTDEReceiveInterface.getActualQ()`.
2. Read gripper position → `grip_pos`.
3. Grab both camera frames, convert **BGR→RGB**, **resize to 360×640**.
4. Build the observation dict with keys `observation.state`,
   `observation.images.cam1`, `observation.images.cam2` (+ `"task"`).
5. `obs = preprocessor(obs)`.
6. `action = policy.select_action(obs)` → `action = postprocessor(action)` → numpy `(7,)`.
7. Split: `q_target = action[:6]` (rad), `grip_cmd = action[6]`.
8. `servoJ(q_target, ...)` to the UR + drive the gripper from `grip_cmd`.
9. Sleep to hold exactly 30 Hz (use `RTDEControlInterface.initPeriod()/waitPeriod()`).

Image tensor format expected downstream: RGB, **CHW, float32 in [0,1]** — this is what the
tutorial's `prepare_observation_for_inference` produces (uint8 HWC → `/255`, `permute(2,0,1)`
at `lerobot/src/lerobot/policies/utils.py:128-131`). The reference script below does this
conversion explicitly.

---

## 3. CRITICAL SAFETY (read before powering the arm)

Because actions are **absolute joint positions**:

1. **Start near the dataset's initial pose.** The policy was only ever conditioned on states
   near data-collection start. Move the UR to approximately the dataset state mean
   **`q1..q6 ≈ [2.84, -1.41, 1.78, -2.01, -1.66, -3.42]` rad** *before* enabling the policy.
   Starting far away → the very first absolute command is a large jump.
2. **First-command jump guard.** Before the first `servoJ`, compare `q_target` to the current
   `getActualQ()`. If `max(|Δq|)` exceeds a small threshold (e.g. **0.15 rad**), **abort** —
   do not let the arm snap. The reference script implements this.
3. **Velocity / step clamping.** Clamp per-tick joint change to a safe max (e.g. ≤0.05–0.1 rad
   per 33 ms early on). `servoJ` interpolates, but a bad prediction can still command a large
   move.
4. **Joint limits.** Clamp `q_target` to the UR software joint limits before sending.
5. **Reduced speed for first trials.** Keep `servoJ` gains low / lookahead high, or scale the
   commanded step, for the first runs. Increase only once behavior looks correct.
6. **Dead-man switch.** Keep a hand on the teach-pendant **E-stop** (or a keyboard kill in the
   script) for every run. Be ready to release.
7. **Gripper.** `grip_cmd` is ~binary; threshold it (e.g. `>0.5 → close`) and map to your
   gripper driver's open/close, never send it raw as a position unless you calibrated it.

---

## 4. Camera correspondence (cam1 vs cam2)

The policy learned a fixed mapping from **each physical viewpoint** to `cam1` / `cam2`.
**If you swap them, the policy fails silently** (no error, just bad actions).

Verify before every session:
- Determine which physical camera was recorded as `cam1` and which as `cam2` during data
  collection (check your recording config / a sample episode: open
  `observation.images.cam1` frames and match the viewpoint by eye).
- At deploy, print/preview each captured frame and confirm the **same physical camera →
  same key**. Fix the OS device indices (`/dev/video*` order is not stable across reboots —
  prefer by-id paths or serial-number matching).
- Confirm resolution/orientation match (360×640 after resize, not rotated/mirrored).

The stock lerobot inference path enforces the *names* but cannot know your *physical* wiring —
this is on you. (Feature-name consistency is checked at
`lerobot/src/lerobot/policies/utils.py:226` `validate_visual_features_consistency`.)

---

## 5. Which UR interface

lerobot ships robot classes for SO-100/SO-101, LeKiwi, Koch, etc., but **no UR class**. Two
options:

**(A) Direct `ur_rtde` in the loop (recommended, least code).** Use
[`ur_rtde`](https://sdurobotics.gitlab.io/ur_rtde/):
- `RTDEReceiveInterface(ip).getActualQ()` → 6 joint angles (rad) for the state.
- `RTDEControlInterface(ip).servoJ(q, vel, acc, dt, lookahead_time, gain)` → stream absolute
  joint targets at 30 Hz. Use `initPeriod()` / `waitPeriod(t_start)` around each tick to hold
  the rate; `servoStop()` on exit.
- `moveJ(q)` only for the slow, one-time move to the start pose (blocking).

**(B) Subclass lerobot's `Robot`.** Implement `get_observation()` / `send_action()` around
`ur_rtde` so you can reuse `build_inference_frame` / `make_robot_action`
(`lerobot/src/lerobot/policies/utils.py:141,175`) and the tutorial loop verbatim. More
idiomatic, more boilerplate.

**Prior art:** the community repo **`F-Fer/lerobot_ur5e_gello`** wires a UR5e + GELLO into
lerobot and is a good reference for exactly this (UR follower + GELLO leader). Consult it for
the `ur_rtde` servo parameters and gripper handling it uses.

The reference script below uses option **(A)**.

---

## 6. Reference script

See `deploy_ur_act.py` in the project root. It is a **skeleton** — every place that touches
your specific hardware (robot IP, gripper driver, camera indices, start pose, safety
thresholds) is marked `# TODO`. Do a **dry run with the motors disabled / speed scaled to
near-zero first**, verify the camera mapping and the first-command guard, then increase speed.
