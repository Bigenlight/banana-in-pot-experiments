#!/usr/bin/env python3
"""
Convert the "Put right banana in the pot" gello/UR teleop dataset (h5 + mp4 per take)
into a LeRobot v3.0 dataset.

Design (verified against lerobot 0.6.1 source):
  - fps = 30 (camera rate). Every stream is resampled onto the cam1 timestamp grid
    via nearest-timestamp lookup on each stream's own `t_rel_s`.
  - observation.state (7): ur_joint_states q1..q6 + gripper.grip_pos   (real robot only)
  - action           (7): command cmd1..cmd6 + gripper.grip_cmd        (absolute joint targets)
  - observation.images.cam1 / cam2: 720p RGB video (HWC uint8)
  - gello_* streams are intentionally ignored (not observable at inference time).

Alignment master clock = cam1_frames/t_rel_s. cam2 frame chosen by nearest cam2 timestamp
(handles the ±1-2 frame count mismatch between cameras seen in 21 takes).
"""
import argparse
import glob
import os
import sys

import cv2
import h5py
import numpy as np

TASK = "put the right banana in the pot"
FPS = 30

STATE_NAMES = ["ur_q1", "ur_q2", "ur_q3", "ur_q4", "ur_q5", "ur_q6", "grip_pos"]
ACTION_NAMES = ["cmd1", "cmd2", "cmd3", "cmd4", "cmd5", "cmd6", "grip_cmd"]


def nearest_idx(src_t: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    """For each query timestamp, index of the nearest src sample (src_t is sorted asc)."""
    j = np.searchsorted(src_t, query_t)
    j = np.clip(j, 1, len(src_t) - 1)
    left = src_t[j - 1]
    right = src_t[j]
    pick_left = (query_t - left) <= (right - query_t)
    out = np.where(pick_left, j - 1, j)
    return np.clip(out, 0, len(src_t) - 1)


def ffill_bfill(v: np.ndarray) -> np.ndarray:
    """Forward-fill then back-fill NaNs (for the one take with NaN grip_cmd)."""
    v = v.copy()
    n = len(v)
    # forward fill
    last = np.nan
    for i in range(n):
        if np.isnan(v[i]):
            v[i] = last
        else:
            last = v[i]
    # back fill leading NaNs
    nxt = np.nan
    for i in range(n - 1, -1, -1):
        if np.isnan(v[i]):
            v[i] = nxt
        else:
            nxt = v[i]
    return v


def load_take_arrays(h5_path: str):
    """Return (cam1_t, state[N,7], action[N,7]) resampled onto cam1 timeline."""
    with h5py.File(h5_path, "r") as f:
        cam1_t = f["cam1_frames"]["t_rel_s"][:]
        cam2_t = f["cam2_frames"]["t_rel_s"][:]
        N = len(cam1_t)

        # --- state: UR joints + grip_pos ---
        ur_t = f["ur_joint_states"]["t_rel_s"][:]
        ur_j = nearest_idx(ur_t, cam1_t)
        state = np.zeros((N, 7), dtype=np.float32)
        for k in range(6):
            state[:, k] = f["ur_joint_states"][f"q{k+1}"][:][ur_j]
        grip_t = f["gripper"]["t_rel_s"][:]
        grip_j = nearest_idx(grip_t, cam1_t)
        state[:, 6] = f["gripper"]["grip_pos"][:][grip_j]

        # --- action: commanded joints + grip_cmd ---
        cmd_t = f["command"]["t_rel_s"][:]
        cmd_j = nearest_idx(cmd_t, cam1_t)
        action = np.zeros((N, 7), dtype=np.float32)
        for k in range(6):
            action[:, k] = f["command"][f"cmd{k+1}"][:][cmd_j]
        grip_cmd_full = ffill_bfill(f["gripper"]["grip_cmd"][:])
        action[:, 6] = grip_cmd_full[grip_j]

    return cam1_t, cam2_t, state, action


def open_reader(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    return cap


def convert(data_root: str, out_root: str, repo_id: str, limit=None,
            image_writer_processes=4, image_writer_threads=2):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "action": {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES},
        "observation.state": {"dtype": "float32", "shape": (7,), "names": STATE_NAMES},
        "observation.images.cam1": {"dtype": "video", "shape": (720, 1280, 3),
                                     "names": ["height", "width", "channels"]},
        "observation.images.cam2": {"dtype": "video", "shape": (720, 1280, 3),
                                     "names": ["height", "width", "channels"]},
    }

    if os.path.exists(out_root):
        print(f"[!] output dir already exists: {out_root}", file=sys.stderr)
        print("    remove it first or pass a fresh --out", file=sys.stderr)
        sys.exit(1)

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=features,
        root=out_root,
        robot_type="ur5e_gello",
        use_videos=True,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )

    takes = sorted(glob.glob(os.path.join(data_root, "take_*")))
    if limit:
        takes = takes[:limit]

    total_frames = 0
    for ti, tk in enumerate(takes):
        h5_path = os.path.join(tk, "vectors.h5")
        cam1_mp4 = os.path.join(tk, "cam1.mp4")
        cam2_mp4 = os.path.join(tk, "cam2.mp4")
        cam1_t, cam2_t, state, action = load_take_arrays(h5_path)
        N = len(cam1_t)

        # cam2 frame index (in its own mp4) nearest to each master timestamp
        cam2_map = nearest_idx(cam2_t, cam1_t)

        cap1 = open_reader(cam1_mp4)
        cap2 = open_reader(cam2_mp4)

        # sequential decode of cam1 (master, frame k == master k)
        cam1_frames = []
        for k in range(N):
            ok, fr = cap1.read()
            if not ok:
                # cam1 ran short: pad with last frame
                fr = cam1_frames[-1][:, :, ::-1].copy() if cam1_frames else np.zeros((720, 1280, 3), np.uint8)
                cam1_frames.append(fr[:, :, ::-1].copy())
                continue
            cam1_frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap1.release()

        # sequential decode of cam2 fully, then index by cam2_map
        cam2_all = []
        while True:
            ok, fr = cap2.read()
            if not ok:
                break
            cam2_all.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap2.release()
        if len(cam2_all) == 0:
            raise RuntimeError(f"no frames decoded from {cam2_mp4}")
        cam2_map = np.clip(cam2_map, 0, len(cam2_all) - 1)

        for k in range(N):
            ds.add_frame({
                "action": action[k],
                "observation.state": state[k],
                "observation.images.cam1": cam1_frames[k],
                "observation.images.cam2": cam2_all[cam2_map[k]],
                "task": TASK,
            })
        ds.save_episode()
        total_frames += N
        print(f"[{ti+1:2d}/{len(takes)}] {os.path.basename(tk):32s} frames={N:4d}  (cam2 decoded {len(cam2_all)})", flush=True)

    ds.finalize()
    print(f"\nDONE: {len(takes)} episodes, {total_frames} frames -> {out_root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Put_right_banana_in_the_pot")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-id", default="theo/banana_in_pot")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    convert(args.data, args.out, args.repo_id, limit=args.limit,
            image_writer_processes=args.procs, image_writer_threads=args.threads)
