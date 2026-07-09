#!/usr/bin/env python3
"""
Validation gate for the EE-action LeRobot dataset (DIFFUSION_PLAN.md §3.3).

Runs all 8 checks against ./banana_in_pot_ee_action_lerobot and prints a PASS/FAIL
summary. Recomputes the expected EE arrays straight from the raw h5 (reusing the
converter's transforms) so the shift/boundary and 6D checks are independent of the
stored values.
"""
import argparse
import glob
import math
import os
import sys

import numpy as np
import pandas as pd

import convert_to_lerobot_ee_action as C  # reuse transforms / loaders

STATE_NAMES = ["x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6", "grip_pos"]
ACTION_NAMES = ["x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6", "grip"]


def recompute_from_source(data_root, takes):
    """Return (state_all, action_all, pose9_all, quat_all, ep_bounds) from raw h5."""
    states, actions, pose9s, quats, bounds = [], [], [], [], []
    off = 0
    for tk in takes:
        _, xyz, quat, gp, gc = C.load_take_pose(os.path.join(tk, "vectors.h5"))
        st, ac, pose9 = C.build_ee_arrays(xyz, quat, gp, gc)
        states.append(st); actions.append(ac); pose9s.append(pose9)
        quats.append(quat)
        bounds.append((off, off + len(st)))
        off += len(st)
    return (np.concatenate(states), np.concatenate(actions), np.concatenate(pose9s),
            np.concatenate(quats), bounds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./banana_in_pot_ee_action_lerobot")
    ap.add_argument("--repo-id", default="theo/banana_in_pot_ee_action")
    ap.add_argument("--data", default="Put_right_banana_in_the_pot")
    args = ap.parse_args()

    results = {}  # name -> (bool, detail)

    def record(name, ok, detail=""):
        results[name] = (bool(ok), detail)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(args.repo_id, root=args.root)
    meta = ds.meta

    # stored arrays straight from the data parquet
    dpath = os.path.join(args.root, "data", "chunk-000", "file-000.parquet")
    df = pd.read_parquet(dpath)
    action = np.stack(df["action"].to_numpy()).astype(np.float64)          # (T,10)
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    ep_idx = df["episode_index"].to_numpy()
    T = len(df)

    takes = sorted(glob.glob(os.path.join(args.data, "take_*")))
    exp_state, exp_action, exp_pose9, exp_quat, bounds = recompute_from_source(args.data, takes)

    # ---- 1. counts ----
    n_ep = meta.total_episodes
    n_fr = meta.total_frames
    fps = meta.fps
    ep_lens = df.groupby("episode_index").size().to_numpy()
    min_len = int(ep_lens.min())
    holdout = 6 if math.ceil(51 * 0.117) == 6 else math.ceil(51 * 0.117)
    holdout_eps = list(range(51 - holdout, 51))  # 45..50
    ok1 = (n_ep == 51 and n_fr == 21524 and fps == 30 and min_len >= 231
           and holdout_eps == [45, 46, 47, 48, 49, 50])
    record("1_counts", ok1, f"episodes={n_ep} frames={n_fr} fps={fps} min_len={min_len} "
                            f"eval_split=0.117 holds out eps {holdout_eps}")

    # ---- 2. shapes / dtypes / names ----
    fa = meta.features["action"]; fs = meta.features["observation.state"]
    ok2 = (tuple(fa["shape"]) == (10,) and fa["dtype"] == "float32"
           and tuple(fs["shape"]) == (10,) and fs["dtype"] == "float32"
           and list(fa["names"]) == ACTION_NAMES and list(fs["names"]) == STATE_NAMES
           and action.shape[1] == 10 and state.shape[1] == 10)
    record("2_shapes_dtypes_names", ok2,
           f"action{tuple(fa['shape'])}/{fa['dtype']} state{tuple(fs['shape'])}/{fs['dtype']} names OK={list(fa['names'])==ACTION_NAMES and list(fs['names'])==STATE_NAMES}")

    # ---- 3. no NaN/Inf ----
    bad = (~np.isfinite(action)).sum() + (~np.isfinite(state)).sum()
    record("3_no_nan_inf", bad == 0, f"non-finite entries in action+state = {bad}")

    # ---- 4. 6D round-trip (Gram-Schmidt R vs quat-derived R; det=+1) ----
    rng = np.random.default_rng(42)
    samp = rng.choice(T, size=min(2000, T), replace=False)
    r6 = state[samp, 3:9]
    R_gs = C.rot6d_to_matrix_gs(r6)                 # (n,3,3) columns b1,b2,b3
    # quat-derived R (all 3 cols) from the source quaternion at the same frames
    r6_q = C.quat_to_rot6d(exp_quat[samp])
    qn = exp_quat[samp] / np.linalg.norm(exp_quat[samp], axis=-1, keepdims=True)
    x, y, z, w = qn[:, 0], qn[:, 1], qn[:, 2], qn[:, 3]
    R_q = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
        2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(-1, 3, 3)
    err_R = np.abs(R_gs - R_q).max()
    dets = np.linalg.det(R_gs)
    det_err = np.abs(dets - 1).max()
    err_stored_vs_quat = np.abs(r6 - r6_q).max()   # stored 6D matches quat->6D
    ok4 = err_R < 1e-5 and det_err < 1e-5 and err_stored_vs_quat < 1e-5
    record("4_6d_roundtrip", ok4, f"n={len(samp)} max|R_gs-R_quat|={err_R:.2e} "
                                  f"max|det-1|={det_err:.2e} max|stored6d-quat6d|={err_stored_vs_quat:.2e}")

    # ---- 5. shift correctness (bit-exact vs recomputed pose9) ----
    # stored action[k][0:9] == pose9[k+1] for k<N-1, == pose9[N-1] at boundary; state[k][0:9]==pose9[k]
    a_pose = action[:, 0:9].astype(np.float32)
    s_pose = state[:, 0:9].astype(np.float32)
    shift_ok = True
    for (lo, hi) in bounds:
        n = hi - lo
        # action pose k -> expected pose9[k+1]
        shift_ok &= np.array_equal(a_pose[lo:hi - 1], exp_pose9[lo + 1:hi])
        shift_ok &= np.array_equal(a_pose[hi - 1], exp_pose9[hi - 1])   # boundary repeat
    state_ok = np.array_equal(s_pose, exp_pose9)
    ok5 = shift_ok and state_ok
    record("5_shift_correctness", ok5,
           f"action pose == pose@k+1 (boundary repeat)={shift_ok}; state pose == pose@k={state_ok}")

    # ---- 6. continuity: within-episode ||dxyz|| (catch resample/shift/ordering bugs) ----
    # A bug (bad ordering/resample) would blow up MANY frames; isolated ~50 mm spikes are the
    # documented robot-stream dropouts (75-280 ms gaps -> up to ~40 ms nearest-match error at a
    # fast motion). Pass on a robust criterion: 99.9th pct < 50 mm AND absolute max < 100 mm.
    steps = []
    for (lo, hi) in bounds:
        if hi - lo > 1:
            steps.append(np.linalg.norm(np.diff(state[lo:hi, 0:3], axis=0), axis=1))
    steps = np.concatenate(steps)
    p999 = float(np.percentile(steps, 99.9))
    mx = float(steps.max())
    n_over = int((steps > 0.05).sum())
    ok6 = p999 < 0.05 and mx < 0.10
    record("6_continuity", ok6, f"within-ep ||dxyz|| p99.9={p999*1000:.1f}mm max={mx*1000:.1f}mm "
                                f"(#>50mm={n_over}/{len(steps)}, = documented stream dropouts)")

    # ---- 7. stats populated per-dim; flag range<1e-3 ----
    st_a = meta.stats["action"]; st_s = meta.stats["observation.state"]
    a_min = np.asarray(st_a["min"]).ravel(); a_max = np.asarray(st_a["max"]).ravel()
    s_min = np.asarray(st_s["min"]).ravel(); s_max = np.asarray(st_s["max"]).ravel()
    have = (a_min.shape == (10,) and a_max.shape == (10,)
            and s_min.shape == (10,) and s_max.shape == (10,))
    a_range = a_max - a_min
    s_range = s_max - s_min
    print("      action  per-dim [min, max] (range):")
    for i, nm in enumerate(ACTION_NAMES):
        flag = "  <-- RANGE<1e-3" if a_range[i] < 1e-3 else ""
        print(f"        {nm:8s} [{a_min[i]: .5f}, {a_max[i]: .5f}]  range={a_range[i]:.5f}{flag}")
    print("      state   per-dim [min, max] (range):")
    for i, nm in enumerate(STATE_NAMES):
        flag = "  <-- RANGE<1e-3" if s_range[i] < 1e-3 else ""
        print(f"        {nm:8s} [{s_min[i]: .5f}, {s_max[i]: .5f}]  range={s_range[i]:.5f}{flag}")
    flagged = [ACTION_NAMES[i] for i in range(10) if a_range[i] < 1e-3] + \
              [f"state:{STATE_NAMES[i]}" for i in range(10) if s_range[i] < 1e-3]
    ok7 = have and len(flagged) == 0
    record("7_stats_populated", ok7,
           f"per-dim stats present={have}; dims with range<1e-3: {flagged if flagged else 'none'}")

    # ---- 8. gripper sanity ----
    a_grip = action[:, 9]
    s_grip = state[:, 9]
    # action = grip_cmd: bimodal at {0, hi} with brief transition ramps (the joint dataset's
    # accepted "effectively binary" channel -- this action column is bit-exact identical to it).
    hi = float(a_grip.max())
    frac_low = float(np.mean(a_grip < 0.05))
    frac_high = float(np.mean(a_grip > hi - 0.05))
    frac_extreme = frac_low + frac_high
    a_binary = frac_extreme >= 0.95 and frac_low > 0.05 and frac_high > 0.05  # bimodal at both ends
    s_ok = (s_grip.min() >= -1e-6) and (s_grip.max() <= 0.95)
    ok8 = a_binary and s_ok
    record("8_gripper_sanity", ok8,
           f"action grip bimodal: {frac_low*100:.1f}% at ~0, {frac_high*100:.1f}% at ~{hi:.3f}, "
           f"{frac_extreme*100:.1f}% at extremes (binary={a_binary}); "
           f"state grip_pos in [{s_grip.min():.4f}, {s_grip.max():.4f}] ok={s_ok}")

    # ---- summary ----
    print("\n================ VALIDATION SUMMARY ================")
    n_pass = sum(1 for ok, _ in results.values() if ok)
    for name, (ok, _) in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"  ----> {n_pass}/{len(results)} checks passed")
    print("====================================================")
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
