#!/usr/bin/env python
"""
Reference inference-loop skeleton for running the trained ACT "banana-in-pot" policy on the
REAL Universal Robots follower arm.

This is a SKELETON. Search for "TODO" and adapt every hardware-specific part on your robot PC.
GELLO is NOT used at deploy time -- it was only the teleop leader during data collection.

API is lerobot 0.6.1. Every non-obvious call is annotated with the source file:line it comes
from. See DEPLOY_UR.md for the full explanation and, importantly, the SAFETY section.

Requires (on the robot PC):
    pip install ur_rtde opencv-python torch
    lerobot (0.6.1) installed

Run:
    python deploy_ur_act.py
"""

import time

import cv2
import numpy as np
import torch

# --- lerobot API ---
# ACTPolicy: lerobot/src/lerobot/policies/act/modeling_act.py:42
# from_pretrained: inherited, lerobot/src/lerobot/policies/pretrained.py:162
from lerobot.policies.act import ACTPolicy

# make_pre_post_processors: lerobot/src/lerobot/policies/factory.py:273
# When pretrained_path is passed it loads the processor pipelines saved in the checkpoint
# (factory.py:323 / :333), with dataset normalization stats BAKED IN.
from lerobot.policies import make_pre_post_processors

# --- UR driver (option A in DEPLOY_UR.md) ---
# pip install ur_rtde
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface


# =============================================================================
# CONFIG -- TODO: adapt all of these on your robot PC
# =============================================================================
CHECKPOINT = "outputs/train/act_banana_in_pot/checkpoints/<step>/pretrained_model"  # TODO
DEVICE = "cuda"  # or "cpu"

ROBOT_IP = "192.168.1.100"  # TODO: your UR controller IP

CAM1_INDEX = 0  # TODO: physical camera recorded as observation.images.cam1 (VERIFY! see DEPLOY_UR.md sec 4)
CAM2_INDEX = 1  # TODO: physical camera recorded as observation.images.cam2

IMG_H, IMG_W = 360, 640  # training resolution -- MUST match
FPS = 30
DT = 1.0 / FPS  # 33.3 ms control period

TASK = ""  # ACT is single-task; empty string is fine (tutorial does the same)

# Dataset state mean joints (rad) -- start the arm NEAR this pose. See DEPLOY_UR.md sec 3.
START_Q = np.array([2.84, -1.41, 1.78, -2.01, -1.66, -3.42], dtype=np.float64)  # TODO: refine from your stats

# --- SAFETY thresholds (start conservative, loosen only when behavior is verified) ---
FIRST_JUMP_LIMIT = 0.15     # rad: abort if first commanded target is farther than this from current q
MAX_STEP_PER_TICK = 0.08    # rad: clamp per-joint change each 33 ms tick
GRIP_CLOSE_THRESHOLD = 0.5  # grip_cmd is ~binary
# UR software joint limits (rad). TODO: set to YOUR robot's configured limits.
Q_MIN = np.array([-2 * np.pi] * 6, dtype=np.float64)
Q_MAX = np.array([2 * np.pi] * 6, dtype=np.float64)

# servoJ params -- keep gentle for first trials. TODO: tune (see F-Fer/lerobot_ur5e_gello for reference values).
SERVO_VEL = 0.5
SERVO_ACC = 0.5
SERVO_LOOKAHEAD = 0.1   # larger = smoother/slower reaction
SERVO_GAIN = 300        # lower = softer


# =============================================================================
# Gripper -- TODO: plug in your gripper driver
# =============================================================================
def read_gripper_position() -> float:
    """Return current gripper opening as a scalar matching the dataset's grip_pos convention."""
    # TODO: read from your gripper (Robotiq / OnRobot / custom). Placeholder:
    return 0.0


def send_gripper_command(grip_cmd: float) -> None:
    """grip_cmd is ~binary. Threshold and drive open/close."""
    close = grip_cmd > GRIP_CLOSE_THRESHOLD
    # TODO: call your gripper driver, e.g. gripper.close() / gripper.open()
    _ = close


# =============================================================================
# Cameras
# =============================================================================
def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    # Capture at native res (e.g. 720p) then resize to 360x640; matches training pipeline.
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}")
    return cap


def grab_frame(cap: cv2.VideoCapture) -> np.ndarray:
    """Return an (IMG_H, IMG_W, 3) uint8 RGB frame, resized to the training resolution."""
    ok, frame_bgr = cap.read()
    if not ok:
        raise RuntimeError("Camera read failed")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Resize to 360x640 (H, W). cv2.resize takes (W, H).
    frame_rgb = cv2.resize(frame_rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return frame_rgb  # uint8 HWC


def to_image_tensor(frame_rgb: np.ndarray) -> torch.Tensor:
    """uint8 HWC [0,255] -> float32 CHW [0,1]. Mirrors prepare_observation_for_inference
    (lerobot/src/lerobot/policies/utils.py:128-131). The preprocessor's NormalizerProcessorStep
    then applies dataset mean/std on top."""
    t = torch.from_numpy(frame_rgb).float() / 255.0
    t = t.permute(2, 0, 1).contiguous()  # CHW
    return t  # (3, 360, 640), no batch dim -- AddBatchDimensionProcessorStep adds it (idempotent)


# =============================================================================
# Observation dict
# =============================================================================
def build_observation(ur_q: np.ndarray, grip_pos: float,
                      cam1: np.ndarray, cam2: np.ndarray) -> dict:
    """Keys MUST match the dataset feature names exactly."""
    state = np.concatenate([ur_q, [grip_pos]]).astype(np.float32)  # (7,) = [q1..q6, grip_pos]
    return {
        "observation.state": torch.from_numpy(state),             # (7,)
        "observation.images.cam1": to_image_tensor(cam1),         # (3,360,640)
        "observation.images.cam2": to_image_tensor(cam2),         # (3,360,640)
        "task": TASK,
    }


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    print("Loading policy + processors from checkpoint (stats baked in)...")
    policy = ACTPolicy.from_pretrained(CHECKPOINT)  # pretrained.py:162
    policy.to(DEVICE)
    policy.eval()

    # Loads the SAME normalization used in training from the checkpoint. factory.py:273
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=CHECKPOINT,
    )

    print(f"n_action_steps={policy.config.n_action_steps}, chunk_size={policy.config.chunk_size}, "
          f"temporal_ensemble_coeff={policy.config.temporal_ensemble_coeff}")

    # Connect to UR.
    rtde_r = RTDEReceiveInterface(ROBOT_IP)
    rtde_c = RTDEControlInterface(ROBOT_IP)

    # Move to start pose (slow, blocking) so the first ABSOLUTE action is not a huge jump.
    print("Moving to dataset start pose (slow)...")
    rtde_c.moveJ(START_Q.tolist(), 0.3, 0.3)  # TODO: confirm safe speed/acc

    # Reset the ACT action queue at episode start. modeling_act.py:93
    policy.reset()

    cap1 = open_camera(CAM1_INDEX)
    cap2 = open_camera(CAM2_INDEX)

    # ---- one-time camera sanity dump: VERIFY cam1/cam2 match data collection viewpoints ----
    cv2.imwrite("verify_cam1.png", cv2.cvtColor(grab_frame(cap1), cv2.COLOR_RGB2BGR))
    cv2.imwrite("verify_cam2.png", cv2.cvtColor(grab_frame(cap2), cv2.COLOR_RGB2BGR))
    print("Wrote verify_cam1.png / verify_cam2.png -- CONFIRM viewpoints match training before trusting output.")

    first_command = True
    prev_q = np.array(rtde_r.getActualQ(), dtype=np.float64)

    try:
        while True:
            t_start = rtde_c.initPeriod()  # start of 30 Hz period

            # 1-3) read state + cameras
            ur_q = np.array(rtde_r.getActualQ(), dtype=np.float64)  # 6 joints, rad
            grip_pos = read_gripper_position()
            cam1 = grab_frame(cap1)
            cam2 = grab_frame(cap2)

            # 4) observation dict
            obs = build_observation(ur_q, grip_pos, cam1, cam2)

            # 5) preprocess (rename -> batch -> device -> normalize)
            obs = preprocessor(obs)

            # 6) inference: select_action returns NORMALIZED action (one from the chunk queue).
            #    modeling_act.py:101
            with torch.no_grad():
                action = policy.select_action(obs)
            # postprocess: UnnormalizerProcessorStep -> radians, on CPU. processor_act.py:66
            action = postprocessor(action)
            action = action.squeeze(0).cpu().numpy().astype(np.float64)  # (7,)

            q_target = action[:6]
            grip_cmd = action[6]

            # 7) SAFETY: joint-limit clamp
            q_target = np.clip(q_target, Q_MIN, Q_MAX)

            # SAFETY: first-command jump guard
            if first_command:
                jump = np.max(np.abs(q_target - ur_q))
                if jump > FIRST_JUMP_LIMIT:
                    raise RuntimeError(
                        f"First command jump {jump:.3f} rad > {FIRST_JUMP_LIMIT}. "
                        f"Arm is not near the start pose, or cam mapping is wrong. Aborting.")
                first_command = False

            # SAFETY: per-tick velocity clamp
            dq = np.clip(q_target - prev_q, -MAX_STEP_PER_TICK, MAX_STEP_PER_TICK)
            q_send = prev_q + dq
            prev_q = q_send

            # 8) send to robot
            rtde_c.servoJ(q_send.tolist(), SERVO_VEL, SERVO_ACC, DT, SERVO_LOOKAHEAD, SERVO_GAIN)
            send_gripper_command(grip_cmd)

            # 9) hold 30 Hz
            rtde_c.waitPeriod(t_start)

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        rtde_c.servoStop()
        rtde_c.stopScript()
        cap1.release()
        cap2.release()
        print("Stopped cleanly.")


if __name__ == "__main__":
    main()
