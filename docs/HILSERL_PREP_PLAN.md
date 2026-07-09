# HIL-SERL base preparation plan (offline, robot-free)

Goal: from the existing teleop dataset (`banana_in_pot_lerobot`), prepare everything for HIL-SERL
UP TO (but not including) the online RL step, so that when the robot is available the user can
launch the online actor-learner in one command. ACT(BC) deploy is the separate near-term test.

**ROBOT = UR7e** (NOT ur5e — the dataset's robot_type metadata string says "ur5e_gello" but the real
arm is a UR7e; that string is cosmetic and does not need fixing).

## Local environment already installed (recon'd 2026-07-08 — reuse, don't reinstall)
- **UR7e kinematics available**: `/opt/ros/jazzy/share/ur_description/config/ur7e/`
  (default_kinematics.yaml [DH], joint_limits.yaml, physical/visual params). Full URDF generable via
  `ur_description` xacro (`ur.urdf.xacro` with `ur_type:=ur7e`). Use this for FK/IK.
- Our raw h5 ALSO already records `tcp_pose` (x,y,z, quat) per frame — reuse it directly for the EE
  observation/action instead of recomputing FK where possible; DH/URDF is for IK (EE-delta→joint).
- ROS 2 Jazzy (`/opt/ros/jazzy`): `ur_description`, `ur_robot_driver`, `ur_moveit_config` present.
- conda envs: `il` (mujoco 3.3.1, robosuite 1.4.1, gymnasium 1.1.1, urdfdom-py — good for URDF parse /
  kinematics), `robocasa`/`robocasa_v02` (sim). LeRobot lives in project `lr_env` (0.6.1).
- `ur_rtde` is NOT installed anywhere — only needed for the ONLINE/robot phase (out of scope now);
  note it in the runbook as a robot-PC install step.
- Up to 10 Sonnet research agents may be spawned for extra info gathering if the Opus builders need it.

## What is offline-preparable (robot NOT required)
1. **Reward classifier** — `src/lerobot/rewards/classifier/`. Supervised, trainable offline from labeled frames.
   - PROBLEM: all 51 demos are successes → no negatives. Team must design a labeling scheme
     (e.g. last K frames of each episode = success(1), early/mid frames = not-success(0)),
     or flag that the user should record a few failure/approach episodes. Document the choice.
2. **Offline demonstration buffer** — feed our LeRobot dataset into HIL-SERL's offline replay buffer
   (`learner.py:349-400`, `OnlineOfflineMixer`, `buffer.py`, `data_sources/`). Build + verify transitions.
3. **UR + task config** — `rl/gym_manipulator.py` uses EE-delta actions + IK from a URDF
   (`EEReferenceAndDelta`, `MapDeltaActionToRobotActionStep`, `inverse_kinematics.urdf_path`).
   Our data is ABSOLUTE JOINT space → must convert to the EE representation HIL-SERL expects
   (needs UR5e URDF + FK to get TCP; our dataset already has tcp_pose in the raw h5 — reuse it!).
## What is NOT possible without the robot (decided — DO NOT attempt)
- **Offline SAC pretraining is NOT supported and is OUT OF SCOPE.** Evidence: `learner.py:412`
  `if len(replay_buffer) < online_step_before_learning: continue` — the learner gates ALL gradient
  steps on the ONLINE replay buffer (actor/robot transitions) reaching a threshold. With no actor the
  online buffer stays empty and no update ever runs. Forcing offline-only would need non-standard code
  surgery (bypass actor/gRPC, 100%-offline mixer, standalone loop) AND is prone to SAC Q-divergence on
  OOD actions. Skip it. The prepared offline demo buffer feeds the first ONLINE session (50-50 RLPD mix)
  where SAC learns fast — that is the intended path.
- Completing SAC policy training (online actor-learner needs real-robot interaction + human interventions).

## FINAL SCOPE (user-confirmed): items 1–4 above + online launch runbook. No offline SAC pretrain.

## Opus team task division (spawn when ACT training is DONE + cleaned up)
- A. Deep-read HIL-SERL code: `rl/train_rl.py`, `learner.py`, `learner_service.py`, `actor.py`,
     `buffer.py`, `data_sources/`, `algorithms/sac/*`, `gym_manipulator.py`, `configs.py`, docs
     `hilserl.mdx`/`hilserl_sim.mdx`. Output: exact required config schema + whether offline-only
     pretraining is runnable.
- B. Reward classifier: read `rewards/classifier/*`, design labeling scheme from our data, TRAIN it
     offline, report accuracy. Produce checkpoint.
- C. Offline demo buffer builder: convert `banana_in_pot_lerobot` → the transition/replay-buffer format
     the SAC learner ingests. Use raw h5 tcp_pose for EE. Verify shapes/reward/terminal flags.
- D. Joint→EE action conversion: use UR5e kinematics (URDF via `lerobot/model/kinematics.py` or
     ur_rtde FK) to express actions as EE deltas matching gym_manipulator's processor. Reuse
     dataset tcp_pose (x,y,z,quat) — already recorded.
- E. Config author: full SAC/HIL-SERL config for UR + this task (cameras 360x640, EE bounds, crop ROI,
     num_critics, utd_ratio, offline mix ratio). Sim smoke via `hilserl_sim` if a sim env exists.
- F. (DROPPED — offline SAC pretrain is out of scope, see above.)
- G. Integrator/validator: assemble outputs, dry-run the learner init end-to-end (no robot, expect it to
     wait at the online-buffer gate — that's correct), write HILSERL_RUNBOOK.md with the exact commands
     to start online training (actor cmd + learner cmd) when the robot is ready, with the offline demo
     buffer + reward classifier + config wired in.

Single GPU: the actual training steps (reward classifier, any offline SAC) run sequentially, not
concurrently. Agents parallelize research/build; heavy GPU jobs are serialized by the integrator.

## Constraints / safety
- No robot actuation, no irreversible actions — purely offline build + training + docs.
- Reuse the existing `lr_env`, scratchpad TORCH_HOME/HF_LEROBOT_HOME, HF_HUB_OFFLINE=1.
- Do not disturb the ACT outputs; write HIL-SERL artifacts under `hilserl/` in the project root.
