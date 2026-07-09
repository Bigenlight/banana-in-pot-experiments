# HIL-SERL Online Training Runbook — "put right banana in the pot" (UR7e)

Everything offline is already built and frozen (see `HILSERL_PREP_RESULTS.md`). This runbook is
the exact sequence to start **online** HIL-SERL actor–learner training once the **UR7e** is
physically connected. All robot-only steps are marked **[ROBOT]**.

- **Shared train config (learner + actor):**
  `hilserl/config/train_hilserl_ur7e.json`
- **Offline demo buffer:** `theo/banana_in_pot_rl` at `hilserl/banana_rl_lerobot` (51 ep / 21 524 frames)
- **Reward classifier:** `hilserl/reward_classifier/checkpoint` (`success_threshold=0.7`)
- **UR7e URDF (pre-generated):** `hilserl/ur7e.urdf` (IK frame `tool0`)
- **venv:** `lr_env/bin/python`  ·  env vars: `HF_HUB_OFFLINE=1`, `TORCH_HOME=scratchpad/torch_home`,
  `HF_LEROBOT_HOME=<root>/hilserl`

---

## 0. Robot-PC installs (one-time) **[ROBOT]**

The offline env (`lr_env`) is missing the online/transport/hardware deps on purpose. On the machine
that will talk to the arm and run the actor+learner, install:

```bash
# gRPC transport (actor<->learner). NOTE: the offline uv cache only has a cp310 wheel; you
# need a wheel matching lr_env's Python (3.12). Requires network.
uv pip install --python lr_env/bin/python 'lerobot[grpcio-dep]'   # grpcio

# HIL-SERL extra: gym-hil + placo (IK) + transformers, etc.
uv pip install --python lr_env/bin/python 'lerobot[hilserl]'      # pulls placo, gym-hil

# placo (IK solver, required by lerobot/model/kinematics.py:59). If not pulled above:
uv pip install --python lr_env/bin/python placo

# ur_rtde — UR robot I/O. NOT installed anywhere yet; required for the UR robot driver.
uv pip install --python lr_env/bin/python ur_rtde
```

Verify: `lr_env/bin/python -c "import grpc, placo, rtde_control; print('ok')"`.

> Why these are not already installed: they are hardware/transport-only. The entire offline prep
> (buffer, classifier, EE conversion, config, dry-run) was completed without them. The learner
> module import fails fast at `lerobot/transport/__init__.py:27` (`require_package("grpcio")`) until
> grpcio is present — this is the only thing blocking a full end-to-end module launch offline.

---

## 1. (Already done, re-verify) UR7e URDF

`hilserl/ur7e.urdf` is already generated and referenced by the config
(`env.processor.inverse_kinematics.urdf_path`). To regenerate:

```bash
source /opt/ros/jazzy/setup.bash
xacro /opt/ros/jazzy/share/ur_description/urdf/ur.urdf.xacro \
      ur_type:=ur7e name:=ur7e > hilserl/ur7e.urdf
```

IK target frame = `tool0` (offset-free flange; Agent D validated recorded TCP == flange). If you
mount a gripper whose TCP differs, add a `gripper_frame_link` to the URDF and set
`env.processor.inverse_kinematics.target_frame_name` accordingly.

---

## 2. Fill the robot + teleop blocks in the config **[ROBOT]**

In `hilserl/config/train_hilserl_ur7e.json`, `env.robot` and `env.teleop` are currently `null`
(robot-free dry-run). Set them for the live arm, e.g.:

```jsonc
"robot":  { "type": "<ur_follower_type>", "ip": "<UR7e_IP>", /* motors, cameras cam1/cam2 */ },
"teleop": { "type": "gamepad", "use_gripper": true }
```

`motor_names = list(env.robot.bus.motors.keys())` feeds the kinematics solver
(gym_manipulator.py:414-420), so the robot config's motor order must match the 6 UR joints + gripper.

---

## 3. Tune the workspace-specific values against the real arm **[ROBOT]**

These are **TODO placeholders** in the config — draft values are in there, but they MUST be verified
with the arm before letting the policy move:

| Field | Config path | Current placeholder | How to tune |
|---|---|---|---|
| EE safety bounds | `env.processor.inverse_kinematics.end_effector_bounds` | `min[-0.6,-0.6,0.0] max[0.6,0.6,0.6]` (m, base frame) | Jog the arm to the reachable corners of the banana/pot workspace; set a tight box that contains the task but clamps runaways. |
| Reset pose | `env.processor.reset.fixed_reset_joint_positions` | `[3.05,-1.60,1.90,-1.85,-1.55,-3.30,0.02]` (rad, from dataset joint-range mid) | Set to a safe, repeatable pre-grasp home; verify the arm returns there each episode. |
| EE step sizes | `env.processor.inverse_kinematics.end_effector_step_sizes` | `{x:0.05,y:0.05,z:0.05}` (m) | **Keep at 0.05** — the offline action = TCP-delta ÷ 0.05, so changing this desyncs the demo actions. |
| Episode length | `env.processor.reset.control_time_s` | `20.0` s (→ `max_episode_steps = 20*30 = 600`) | Match a comfortable single-attempt duration. |

The reward classifier decision boundary around the release moment is uncalibrated (Agent B caveat);
`success_threshold=0.7` adds margin. Consider recording a few real **failure/near-miss** episodes
early to harden it.

---

## 4. (Optional) Crop ROI — only if you crop online **[ROBOT]**

The offline buffer images are **full-frame resized to 128×128, no crop**
(`image_preprocessing.resize_size=[128,128]`, `crop_params_dict=null`). If you decide to crop online
to focus on the workspace, you MUST keep offline and online identical:

```bash
# 1) find the ROI interactively on a recorded dataset
lr_env/bin/python -m lerobot.rl.crop_dataset_roi --repo-id theo/banana_in_pot_rl
# 2) put the returned crop_params_dict into env.processor.image_preprocessing.crop_params_dict
# 3) RE-RUN the crop on the OFFLINE buffer too (hilserl/banana_rl_lerobot) so the demo images
#    match the online cropped+resized size — otherwise the encoder sees two different distributions.
```

If you do not crop, skip this entirely (the default is consistent already).

---

## 5. Start the LEARNER (terminal 1) **[ROBOT]**

```bash
export HF_HUB_OFFLINE=1
export TORCH_HOME=<root>/scratchpad/torch_home
export HF_LEROBOT_HOME=<root>/hilserl
lr_env/bin/python -m lerobot.rl.learner \
    --config_path hilserl/config/train_hilserl_ur7e.json
```

The learner: builds the SAC policy + critics, loads the offline demo buffer via
`ReplayBuffer.from_lerobot_dataset`, opens a gRPC server on `127.0.0.1:50051`, then **idles at the
online-buffer gate** (`learner.py:412-413`) until the actor sends transitions. This idle state was
validated offline (see §7 / `HILSERL_PREP_RESULTS.md`).

> **RAM WARNING — full offline buffer ≈ 25 GB host RAM.**
> `from_lerobot_dataset` eagerly materializes **every** transition (decoded float32 state +
> next-state images) into a Python list before filling storage; the full 21 524-frame ×
> 2×(3×128×128) set peaks at **~25 GB** and was OOM-killed on the 31 GB box (Agent C). Storage
> itself at `offline_buffer_capacity=25000` adds ~9.8 GB (`optimize_memory=True` aliases next_state).
> **Mitigations** (pick one):
> - Run on a machine with **≥ 48 GB RAM** (or add swap).
> - Subset the demos: add `--dataset.episodes='[0,1,...,N]'` (fewer episodes → proportional RAM).
> - Lower `policy.offline_buffer_capacity` toward the frame count you actually load.

---

## 6. Start the ACTOR (terminal 2, same config) **[ROBOT]**

```bash
export HF_HUB_OFFLINE=1 TORCH_HOME=<root>/scratchpad/torch_home HF_LEROBOT_HOME=<root>/hilserl
lr_env/bin/python -m lerobot.rl.actor \
    --config_path hilserl/config/train_hilserl_ur7e.json
```

The actor connects to the learner over gRPC, opens cam1/cam2, builds the EE-delta → IK action
pipeline (`MapTensorToDeltaActionDict → MapDeltaActionToRobotAction → EEReferenceAndDelta →
EEBoundsAndSafety → GripperVelocityToJoint → InverseKinematicsRLStep`), runs the policy on the arm,
and streams transitions back. Once ≥ `online_step_before_learning` (100) online transitions arrive,
the learner starts SAC updates with a 50/50 online/offline RLPD mix (`online_ratio=0.5`).

---

## 7. Human-in-the-loop interventions **[ROBOT]**

- Press the **upper-right trigger** on the gamepad (or **`space`** on the keyboard) to take over and
  provide a corrective demonstration; release to hand control back to the policy.
- Intervene heavily at the start, then taper — a healthy run shows the intervention rate dropping as
  the policy improves (watch it in wandb if `wandb.enable=true`).
- The success reward comes from the vision classifier (`reward=1` when `prob>0.7`); with
  `terminate_on_success=true` the episode ends on the first success frame.

---

## 8. Key hyperparameters to tune (config: `hilserl/config/train_hilserl_ur7e.json`)

- `algorithm.temperature_init` (SAC entropy temp) — start `0.01`; too high makes interventions
  ineffective.
- `policy.actor_learner_config.policy_parameters_push_frequency` — seconds between weight pushes
  (default 4; drop to 1–2 for fresher actor weights).
- `policy.storage_device` — keep `"cpu"` here (12 GB GPU can't hold the offline image buffer). Set
  `"cuda"` only if you move to a big-VRAM box.
- `algorithm.utd_ratio` (2) / `algorithm.num_critics` (2) — raise UTD for more updates per step.

---

### Offline validation already passed (no robot)
`hilserl/config/dryrun_validate_learner.py` reproduced the learner setup path (config parse →
`make_policy` → `make_algorithm` (SAC) → offline buffer load+sample → gate). It reached the idle gate
`len(online_buffer)=0 < online_step_before_learning=100`. Full evidence in `HILSERL_PREP_RESULTS.md`.
