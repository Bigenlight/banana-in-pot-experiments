#!/usr/bin/env python3
"""
Build the EE-ACTION LeRobot v3.0 dataset for "Put right banana in the pot".

Feature spec (DIFFUSION_PLAN.md §3.1):
  - action              (10,) float32 = [tcp x,y,z @ k+1, 6D-rot @ k+1, grip_cmd @ k]
      names = ["x","y","z","r1","r2","r3","r4","r5","r6","grip"]
  - observation.state   (10,) float32 = [tcp x,y,z @ k, 6D-rot @ k, grip_pos @ k]
      names = ["x","y","z","r1","r2","r3","r4","r5","r6","grip_pos"]
  - observation.images.cam1 / cam2 : video (720,1280,3)  -- UNCHANGED (reused)

6D rotation (Zhou et al.): quaternion (scalar-last qx,qy,qz,qw) -> rotation matrix ->
first two COLUMNS, flattened col-major as [R00,R10,R20, R01,R11,R21].

Episode boundary = REPEAT last pose (action[N-1] pose = pose[N-1]); frame count preserved
(21,524). state carries grip_pos (@k), action carries grip_cmd (@k, ffill/bfill, NOT shifted).

EFFICIENCY: the camera videos are byte-identical to ./banana_in_pot_lerobot (same frames,
same cam1 master-clock resampling, same episode order). Rather than re-encode AV1 (CPU heavy,
would contend with the running ACT job), we REUSE the already-encoded videos: copy the source
dataset dir, then regenerate ONLY the data parquet action/state columns + meta (info.json,
stats.json, episodes-parquet per-episode stats). Frame alignment is proven at build time by a
bit-exact cross-check of the gripper channels against the source joint dataset (same
nearest-timestamp cam1 grid -> identical row order).

Resampling / NaN(ffill_bfill) / cam1-master-clock / episode-order logic is reused UNCHANGED
from convert_to_lerobot_ee.py (nearest_idx, ffill_bfill, sorted(glob('take_*'))).

CPU-only. Local-only (never push_to_hub).
"""
import argparse
import glob
import json
import os
import shutil
import sys

import h5py
import numpy as np

TASK = "put the right banana in the pot"
FPS = 30

STATE_NAMES = ["x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6", "grip_pos"]
ACTION_NAMES = ["x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6", "grip"]
QUAT_KEYS = ["qx", "qy", "qz", "qw"]  # scalar-last, as stored in tcp_pose group


# ---------------------------------------------------------------------------
# resampling / NaN helpers  (UNCHANGED from convert_to_lerobot_ee.py)
# ---------------------------------------------------------------------------
def nearest_idx(src_t: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    """For each query timestamp, index of the nearest src sample (src_t sorted asc)."""
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
    last = np.nan
    for i in range(n):
        if np.isnan(v[i]):
            v[i] = last
        else:
            last = v[i]
    nxt = np.nan
    for i in range(n - 1, -1, -1):
        if np.isnan(v[i]):
            v[i] = nxt
        else:
            nxt = v[i]
    return v


# ---------------------------------------------------------------------------
# 6D rotation (exact formula from PLAN §3.1)
# ---------------------------------------------------------------------------
def quat_to_rot6d(q):  # q = [qx, qy, qz, qw] scalar-last, shape (...,4)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
        2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(*q.shape[:-1], 3, 3)
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)  # (...,6) = col0,col1


def rot6d_to_matrix_gs(r6):
    """Gram-Schmidt recovery of a rotation matrix from a 6D rep (for validation)."""
    a1 = r6[..., 0:3]
    a2 = r6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)  # columns = b1,b2,b3


# ---------------------------------------------------------------------------
# per-take load + EE array construction
# ---------------------------------------------------------------------------
def load_take_pose(h5_path: str):
    """Resample tcp xyz+quat, grip_pos, grip_cmd onto the cam1 master clock (nearest)."""
    with h5py.File(h5_path, "r") as f:
        cam1_t = f["cam1_frames"]["t_rel_s"][:]
        N = len(cam1_t)

        tcp_t = f["tcp_pose"]["t_rel_s"][:]
        tcp_j = nearest_idx(tcp_t, cam1_t)
        xyz = np.zeros((N, 3), dtype=np.float64)
        for i, key in enumerate(["x", "y", "z"]):
            xyz[:, i] = f["tcp_pose"][key][:][tcp_j]
        quat = np.zeros((N, 4), dtype=np.float64)
        for i, key in enumerate(QUAT_KEYS):
            quat[:, i] = f["tcp_pose"][key][:][tcp_j]

        grip_t = f["gripper"]["t_rel_s"][:]
        grip_j = nearest_idx(grip_t, cam1_t)
        grip_pos = f["gripper"]["grip_pos"][:][grip_j].astype(np.float64)
        grip_cmd_full = ffill_bfill(f["gripper"]["grip_cmd"][:])
        grip_cmd = grip_cmd_full[grip_j].astype(np.float64)

    return cam1_t, xyz, quat, grip_pos, grip_cmd


def build_ee_arrays(xyz, quat, grip_pos, grip_cmd):
    """Return state(10) and action(10) plus the shared pose9 (for verification)."""
    rot6d = quat_to_rot6d(quat)                                   # (N,6)
    pose9 = np.concatenate([xyz, rot6d], axis=1)                  # (N,9)  [xyz, 6D]

    # state @k : pose9[k] + grip_pos[k]
    state = np.concatenate([pose9, grip_pos.reshape(-1, 1)], axis=1).astype(np.float32)

    # action pose @k+1 (boundary = repeat last), grip_cmd @k (NOT shifted)
    pose_shift = np.empty_like(pose9)
    pose_shift[:-1] = pose9[1:]
    pose_shift[-1] = pose9[-1]
    action = np.concatenate([pose_shift, grip_cmd.reshape(-1, 1)], axis=1).astype(np.float32)

    return state, action, pose9.astype(np.float32)


# ---------------------------------------------------------------------------
# self-test (Task 2) -- runs before every build; --selftest runs it standalone
# ---------------------------------------------------------------------------
def selftest():
    ok = True
    rng = np.random.default_rng(0)

    # (1) quat -> R -> 6D -> Gram-Schmidt -> R'  round-trip
    q = rng.standard_normal((2000, 4))
    r6 = quat_to_rot6d(q)
    R = rot6d_to_matrix_gs(r6)
    # rebuild the full quat-derived R (all 3 cols) to compare against GS recovery
    qn = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = qn[:, 0], qn[:, 1], qn[:, 2], qn[:, 3]
    Rfull = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
        2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(-1, 3, 3)
    err = np.abs(R - Rfull).max()
    dets = np.linalg.det(R)
    orth = np.abs(np.matmul(R.transpose(0, 2, 1), R) - np.eye(3)).max()
    c1 = err < 1e-5 and np.abs(dets - 1).max() < 1e-5 and orth < 1e-5
    ok &= c1
    print(f"[selftest] 6D round-trip: max|R-R'|={err:.2e}  det in [{dets.min():.6f},{dets.max():.6f}]  "
          f"max|RtR-I|={orth:.2e}  -> {'PASS' if c1 else 'FAIL'}")

    # (1b) q and -q give identical R (sign invariance -> no hemisphere fixing needed)
    err_sign = np.abs(quat_to_rot6d(q) - quat_to_rot6d(-q)).max()
    c1b = err_sign < 1e-12
    ok &= c1b
    print(f"[selftest] q/-q invariance: max|6d(q)-6d(-q)|={err_sign:.2e} -> {'PASS' if c1b else 'FAIL'}")

    # (2) +1 shift / boundary-repeat indexing + grip assignment
    N = 7
    xyz = np.arange(N * 3, dtype=np.float64).reshape(N, 3) + 100.0
    quat = rng.standard_normal((N, 4))
    grip_pos = np.linspace(0.0, 0.9, N)
    grip_cmd = (np.arange(N) % 2).astype(np.float64)
    state, action, pose9 = build_ee_arrays(xyz, quat, grip_pos, grip_cmd)
    shift_ok = True
    for k in range(N - 1):
        shift_ok &= np.array_equal(action[k, 0:9], pose9[k + 1])       # action pose = pose @k+1
    shift_ok &= np.array_equal(action[N - 1, 0:9], pose9[N - 1])       # boundary repeat
    state_ok = True
    for k in range(N):
        state_ok &= np.array_equal(state[k, 0:9], pose9[k])            # state pose = pose @k
    grip_state_ok = np.allclose(state[:, 9], grip_pos.astype(np.float32))
    grip_action_ok = np.allclose(action[:, 9], grip_cmd.astype(np.float32))
    c2 = shift_ok and state_ok and grip_state_ok and grip_action_ok
    ok &= c2
    print(f"[selftest] +1 shift={shift_ok} boundary-repeat included  state@k={state_ok}  "
          f"grip_pos->state={grip_state_ok}  grip_cmd->action={grip_action_ok} -> {'PASS' if c2 else 'FAIL'}")

    print(f"[selftest] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# build via video reuse
# ---------------------------------------------------------------------------
def build(data_root, out_root, repo_id, source_root, limit=None):
    from pathlib import Path

    import pandas as pd
    from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
    from lerobot.datasets.io_utils import load_stats, write_stats

    if os.path.exists(out_root):
        print(f"[!] output dir already exists: {out_root}", file=sys.stderr)
        print("    remove it first or pass a fresh --out", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(source_root):
        print(f"[!] source dataset dir missing: {source_root}", file=sys.stderr)
        sys.exit(1)

    takes = sorted(glob.glob(os.path.join(data_root, "take_*")))
    if limit:
        takes = takes[:limit]
    print(f"[build] {len(takes)} takes from {data_root}")

    # ---- 1. build per-take EE arrays (same order as the source joint dataset) ----
    states, actions, lens = [], [], []
    for ti, tk in enumerate(takes):
        cam1_t, xyz, quat, grip_pos, grip_cmd = load_take_pose(os.path.join(tk, "vectors.h5"))
        st, ac, _ = build_ee_arrays(xyz, quat, grip_pos, grip_cmd)
        states.append(st)
        actions.append(ac)
        lens.append(len(st))
        print(f"[{ti+1:2d}/{len(takes)}] {os.path.basename(tk):32s} frames={len(st):4d}", flush=True)
    state_all = np.concatenate(states, axis=0)
    action_all = np.concatenate(actions, axis=0)
    print(f"[build] total frames = {len(state_all)}")

    # ---- 2. load source data parquet, PROVE frame alignment ----
    src_data = os.path.join(source_root, "data", "chunk-000", "file-000.parquet")
    df = pd.read_parquet(src_data)
    assert len(df) == len(state_all), f"frame count mismatch {len(df)} vs {len(state_all)}"
    src_lens = df.groupby("episode_index").size().to_numpy()
    assert np.array_equal(src_lens, np.array(lens)), "per-episode length mismatch vs source"
    src_state = np.stack(df["observation.state"].to_numpy())   # (T,7) joints
    src_action = np.stack(df["action"].to_numpy())             # (T,7)
    # gripper channels are the ONLY overlap; bit-exact match proves identical cam1-grid row order
    assert np.allclose(src_state[:, 6], state_all[:, 9], atol=0, rtol=0), "grip_pos misaligned vs source"
    assert np.allclose(src_action[:, 6], action_all[:, 9], atol=0, rtol=0), "grip_cmd misaligned vs source"
    print("[build] alignment verified: per-episode lengths + grip_pos/grip_cmd bit-exact vs source")

    # ---- 3. copy the source dataset (reuse encoded videos + video/index meta) ----
    print(f"[build] copytree {source_root} -> {out_root} (reusing encoded videos)")
    shutil.copytree(source_root, out_root)

    # ---- 4. overwrite data parquet action + observation.state columns ----
    df = df.reset_index(drop=True)
    df["action"] = list(action_all.astype(np.float32))
    df["observation.state"] = list(state_all.astype(np.float32))
    out_data = os.path.join(out_root, "data", "chunk-000", "file-000.parquet")
    df.to_parquet(out_data, index=False)
    print(f"[build] rewrote {out_data}")

    # ---- 5. info.json: update feature shapes/names + repo_id ----
    info_path = os.path.join(out_root, "meta", "info.json")
    with open(info_path) as fh:
        info = json.load(fh)
    info["features"]["action"]["shape"] = [10]
    info["features"]["action"]["names"] = ACTION_NAMES
    info["features"]["observation.state"]["shape"] = [10]
    info["features"]["observation.state"]["names"] = STATE_NAMES
    with open(info_path, "w") as fh:
        json.dump(info, fh, indent=4)
    print(f"[build] rewrote {info_path} (action/state -> 10-dim)")

    # ---- 6. per-episode stats -> aggregate -> stats.json + episodes parquet ----
    feat_spec = {
        "action": {"dtype": "float32", "shape": (10,)},
        "observation.state": {"dtype": "float32", "shape": (10,)},
    }
    ep_stats_list = []
    off = 0
    for n in lens:
        sl = slice(off, off + n)
        ep_stats_list.append(compute_episode_stats(
            {"action": action_all[sl], "observation.state": state_all[sl]}, feat_spec))
        off += n
    agg = aggregate_stats(ep_stats_list)

    stats = load_stats(Path(out_root))  # numpy dict, includes image/timestamp/etc
    stats["action"] = agg["action"]
    stats["observation.state"] = agg["observation.state"]
    write_stats(stats, Path(out_root))
    print(f"[build] rewrote stats.json (action/state per-dim, images/index unchanged)")

    # episodes parquet: replace stale 7-dim stats/action|state columns (load_episodes drops
    # these at train time, but keep the file self-consistent)
    ep_path = os.path.join(out_root, "meta", "episodes", "chunk-000", "file-000.parquet")
    epdf = pd.read_parquet(ep_path)
    # assign whole columns (per-cell .at unwraps length-1 arrays to 0-d -> pyarrow error)
    new_cols = {}
    for es in ep_stats_list:
        for feat in ("action", "observation.state"):
            for stat, val in es[feat].items():
                new_cols.setdefault(f"stats/{feat}/{stat}", []).append(np.asarray(val).ravel())
    for col, vals in new_cols.items():
        epdf[col] = pd.Series(vals, index=epdf.index)
    epdf.to_parquet(ep_path, index=False)
    print(f"[build] rewrote {ep_path} (per-episode action/state stats)")

    print(f"\nDONE: {len(takes)} episodes, {len(state_all)} frames -> {out_root} (repo_id {repo_id})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Put_right_banana_in_the_pot")
    ap.add_argument("--out", default="./banana_in_pot_ee_action_lerobot")
    ap.add_argument("--repo-id", default="theo/banana_in_pot_ee_action")
    ap.add_argument("--source", default="./banana_in_pot_lerobot",
                    help="existing joint dataset to reuse encoded videos from")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--selftest", action="store_true", help="run transform unit tests and exit")
    args = ap.parse_args()

    print("=== self-test (transforms) ===")
    if not selftest():
        print("[FATAL] self-test failed; aborting.", file=sys.stderr)
        sys.exit(2)
    if args.selftest:
        sys.exit(0)

    print("\n=== build ===")
    build(args.data, args.out, args.repo_id, args.source, limit=args.limit)
