#!/usr/bin/env python
"""Offline open-loop evaluation harness for an ACT policy trained on the
banana_in_pot LeRobot dataset (lerobot 0.6.1).

Runs a trained ACT checkpoint OPEN-LOOP against ground-truth dataset episodes
(no robot, no simulator): at every timestep the real logged observation
(state + both cameras) is fed to `policy.select_action`, and the predicted
action is compared against the dataset's ground-truth action at that timestep.

The image path exactly mirrors training: cameras are resized to 360x640 with the
same `ImageTransforms` Resize the training used (see train_act.sh), and
normalization / batching / device placement are handled by the policy's saved
pre/post-processor pipelines (loaded from the checkpoint dir).

Usage (full run, once a checkpoint exists):
    python eval_offline.py \
        --checkpoint outputs/train/act_banana_in_pot/checkpoints/010000/pretrained_model \
        --episodes 45,46,47,48,49,50 --device cuda --out eval_out

Smoke/validation (no trained checkpoint, CPU only, does not touch GPU):
    python eval_offline.py --smoke --device cpu --episodes 0 --max-frames 8
"""

import argparse
import os
import sys
from pathlib import Path

# ---- Environment (offline, redirect caches to scratchpad) -------------------
_SCRATCH = os.environ.get("SCRATCH_DIR", os.path.join(os.getcwd(), ".cache"))
os.environ.setdefault("HF_LEROBOT_HOME", os.path.join(_SCRATCH, "hf_lerobot_home"))
os.environ.setdefault("TORCH_HOME", os.path.join(_SCRATCH, "torch_home"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")  # no display
import matplotlib.pyplot as plt

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import (
    get_policy_class,
    make_policy,
    make_policy_config,
    make_pre_post_processors,
)
from lerobot.transforms.transforms import (
    ImageTransformConfig,
    ImageTransforms,
    ImageTransformsConfig,
)
from lerobot.utils.constants import ACTION, OBS_STATE

# -----------------------------------------------------------------------------
DEFAULT_REPO_ID = "theo/banana_in_pot"
DEFAULT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "banana_in_pot_lerobot"
)
DEFAULT_EPISODES = [45, 46, 47, 48, 49, 50]  # last 6 episodes (dataset has 51: 0..50)
RESIZE_HW = [360, 640]
GRIPPER_THRESH = 0.5

# Fallback action-dim names if the dataset meta omits them (current 7-dim joint set).
FALLBACK_DIM_NAMES = ["cmd1", "cmd2", "cmd3", "cmd4", "cmd5", "cmd6", "grip_cmd"]


# ---- Dimension metadata (dim-agnostic) --------------------------------------
def resolve_dim_names(meta):
    """Action-dim names from ds.meta.features['action']['names'] (fallback 7-dim)."""
    try:
        names = list(meta.features["action"]["names"])
    except Exception:
        names = None
    return names if names else list(FALLBACK_DIM_NAMES)


def dim_units(names):
    """Per-dim physical units for the report.
      last dim            -> 'grip'
      x/y/z               -> 'm'      (EE translation, meters)
      r1..rN              -> '6d'     (6D rotation, unitless)
      otherwise           -> 'rad'    (joint angles)
    """
    n = len(names)
    units = []
    for j, nm in enumerate(names):
        low = str(nm).lower()
        if j == n - 1:
            units.append("grip")
        elif low in ("x", "y", "z"):
            units.append("m")
        elif len(low) >= 2 and low[0] == "r" and low[1:].isdigit():
            units.append("6d")
        else:
            units.append("rad")
    return units


def apply_scheduler_overrides(cfg, scheduler, num_inference_steps):
    """Mutate a loaded PreTrainedConfig BEFORE from_pretrained so the noise
    scheduler is (re)built from it at model init. No-op for policies without
    these fields (e.g. ACT), so ACT behavior is untouched."""
    if scheduler and scheduler != "asis" and hasattr(cfg, "noise_scheduler_type"):
        cfg.noise_scheduler_type = scheduler
    if num_inference_steps is not None and hasattr(cfg, "num_inference_steps"):
        cfg.num_inference_steps = int(num_inference_steps)


# ---- Image transform: identical Resize used at training time ----------------
def build_image_transforms() -> ImageTransforms:
    """Replicate the training-time transform:
    --dataset.image_transforms.tfs='{"resize":{"weight":1.0,"type":"Resize",
      "kwargs":{"size":[360,640]}}}'  with max_num_transforms=1, enable=true.
    With a single transform and n_subset=1 this is deterministic (always applied).
    """
    cfg = ImageTransformsConfig(
        enable=True,
        max_num_transforms=1,
        random_order=False,
        tfs={
            "resize": ImageTransformConfig(
                weight=1.0, type="Resize", kwargs={"size": RESIZE_HW}
            )
        },
    )
    return ImageTransforms(cfg)


# ---- Model / processor loading ----------------------------------------------
def load_policy_and_processors(checkpoint: str, device: str, dataset_stats,
                               scheduler="asis", num_inference_steps=None):
    """Load a policy (class resolved from cfg.type) + its pre/post-processor
    pipelines from a checkpoint dir.

    Verified against lerobot 0.6.1:
      - policy: get_policy_class(cfg.type).from_pretrained(dir)
        (PreTrainedPolicy.from_pretrained loads config.json + model.safetensors,
        calls .eval(), .to(device)). ACT -> ACTPolicy, diffusion -> DiffusionPolicy.
      - Fast diffusion sampling: scheduler/num_inference_steps are applied to the
        loaded config BEFORE from_pretrained (the noise scheduler is built from
        config at model init). DDIM is a valid sampler for a DDPM-trained epsilon
        model (same beta schedule). No-op for ACT.
      - processors: make_pre_post_processors(cfg, pretrained_path=dir) loads
        policy_preprocessor.json / policy_postprocessor.json (Normalizer stats
        baked in). The preprocessor does Rename->AddBatchDim->Device->Normalize;
        the postprocessor does Unnormalize->Device(cpu).
    """
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.pretrained_path = checkpoint
    cfg.device = device
    apply_scheduler_overrides(cfg, scheduler, num_inference_steps)
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(checkpoint, config=cfg)
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=checkpoint,
        dataset_stats=dataset_stats,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor


def build_smoke_policy(meta, device: str, policy_type: str = "act"):
    """Build a FRESH (random-weight) policy + processors from dataset stats.
    Used only for --smoke validation on CPU; never touches a real checkpoint or
    the GPU. Confirms the observation dict keys/shapes select_action expects and
    the dim-agnostic metric/plot path for the given policy type.
    """
    cfg = make_policy_config(policy_type)
    cfg.device = device
    if policy_type == "act":
        # keep it tiny/fast for smoke; shapes are what matter, not accuracy
        cfg.n_action_steps = 10
        cfg.chunk_size = 10
    elif policy_type == "diffusion":
        # tiny/fast CPU smoke; horizon 16 keeps default drop_n_last_frames=7 valid
        cfg.horizon = 16
        cfg.n_action_steps = 8
        cfg.num_inference_steps = 2
        # The diffusion RGB encoder's SpatialSoftmax is shape-rigid; build it for the
        # harness's resized 360x640 input so the fresh policy is self-consistent on CPU
        # (smoke only -- real runs use the trained checkpoint's own config).
        cfg.resize_shape = tuple(RESIZE_HW)
    policy = make_policy(cfg, ds_meta=meta)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        dataset_stats=meta.stats,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor


# ---- Core open-loop rollout over one dataset episode ------------------------
def rollout_episode(
    policy, preprocessor, postprocessor, ep_index, root, repo_id, transforms,
    device, camera_keys, max_frames=None,
):
    """Feed each real frame of one episode to select_action; return (pred, gt).

    Mirrors lerobot's SyncInference tick (rollout/inference/sync.py):
        obs -> preprocessor -> policy.select_action -> postprocessor.
    ACT's select_action manages an internal action-chunk queue, so we call it
    once per timestep after policy.reset() at the episode start.
    """
    ds = LeRobotDataset(
        repo_id,
        root=root,
        episodes=[ep_index],
        image_transforms=transforms,
        video_backend="pyav",
    )
    n = len(ds)
    if max_frames is not None:
        n = min(n, max_frames)

    policy.reset()  # clear the action-chunk queue for a fresh episode
    preprocessor.reset()
    postprocessor.reset()

    preds, gts = [], []
    for i in range(n):
        frame = ds[i]
        obs = {OBS_STATE: frame[OBS_STATE]}  # (7,) float32
        for cam in camera_keys:
            obs[cam] = frame[cam]  # (3,360,640) float32 in [0,1], already resized
        obs["task"] = frame.get("task", "")

        with torch.inference_mode():
            proc = preprocessor(obs)           # adds batch dim, device, normalizes
            action = policy.select_action(proc)  # (1,7) normalized
            action = postprocessor(action)       # (1,7) unnormalized, on cpu

        preds.append(action.squeeze(0).float().cpu().numpy())
        gts.append(frame[ACTION].float().cpu().numpy())

    return np.asarray(preds), np.asarray(gts)  # (T,7),(T,7)


# ---- Metrics (dim-agnostic) --------------------------------------------------
def compute_metrics(pred, gt):
    """Dim-agnostic: the LAST action dim is the gripper (binary-ish, threshold
    0.5), all PRECEDING dims form the pose/joints block used for MAE/RMSE."""
    err = pred - gt
    abs_err = np.abs(err)
    per_dim_mae = abs_err.mean(axis=0)                    # (D,)
    per_dim_rmse = np.sqrt((err ** 2).mean(axis=0))       # (D,)
    d = pred.shape[1]
    g = d - 1  # gripper index
    pose_mae = per_dim_mae[:g].mean() if g > 0 else 0.0
    pose_rmse = per_dim_rmse[:g].mean() if g > 0 else 0.0
    grip_mae = per_dim_mae[g]
    grip_rmse = per_dim_rmse[g]
    overall_l1 = abs_err.mean()
    grip_acc = float(
        ((pred[:, g] > GRIPPER_THRESH) == (gt[:, g] > GRIPPER_THRESH)).mean()
    )
    return {
        "T": int(pred.shape[0]),
        "per_dim_mae": per_dim_mae,
        "per_dim_rmse": per_dim_rmse,
        "pose_mae": float(pose_mae),
        "pose_rmse": float(pose_rmse),
        "grip_mae": float(grip_mae),
        "grip_rmse": float(grip_rmse),
        "overall_l1": float(overall_l1),
        "grip_acc": grip_acc,
    }


def print_table(per_ep, agg, dim_names):
    print("\n" + "=" * 78)
    print("OPEN-LOOP OFFLINE EVAL  (predicted action vs dataset ground-truth)")
    print("=" * 78)
    header = (
        f"{'episode':>8} {'T':>5} {'poseMAE':>10} {'poseRMSE':>11} "
        f"{'gripMAE':>9} {'gripAcc':>8} {'overallL1':>10}"
    )
    print(header)
    print("-" * len(header))
    for ep, m in per_ep:
        print(
            f"{ep:>8} {m['T']:>5} {m['pose_mae']:>10.5f} {m['pose_rmse']:>11.5f} "
            f"{m['grip_mae']:>9.5f} {m['grip_acc']:>8.3f} {m['overall_l1']:>10.5f}"
        )
    print("-" * len(header))
    print(
        f"{'MEAN':>8} {agg['T']:>5} {agg['pose_mae']:>10.5f} "
        f"{agg['pose_rmse']:>11.5f} {agg['grip_mae']:>9.5f} "
        f"{agg['grip_acc']:>8.3f} {agg['overall_l1']:>10.5f}"
    )
    print("=" * 78)
    print("\nPer-dimension MAE / RMSE (aggregated over all evaluated frames):")
    print(f"{'dim':>10} {'MAE':>12} {'RMSE':>12}   units")
    units_col = dim_units(dim_names)
    for j, name in enumerate(dim_names):
        print(
            f"{name:>10} {agg['per_dim_mae'][j]:>12.5f} "
            f"{agg['per_dim_rmse'][j]:>12.5f}   {units_col[j]}"
        )
    print()


# ---- Plotting ----------------------------------------------------------------
def save_traj_plot(pred, gt, ep_index, out_dir, dim_names):
    T = pred.shape[0]
    d = pred.shape[1]
    t = np.arange(T)
    fig, axes = plt.subplots(d, 1, figsize=(10, 2 * d), sharex=True)
    if d == 1:
        axes = [axes]
    for j, ax in enumerate(axes):
        ax.plot(t, gt[:, j], label="ground-truth", color="tab:blue", lw=1.5)
        ax.plot(t, pred[:, j], label="predicted", color="tab:red", lw=1.2, ls="--")
        ax.set_ylabel(dim_names[j])
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("timestep")
    fig.suptitle(f"Episode {ep_index}: predicted vs ground-truth action")
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    path = os.path.join(out_dir, f"eval_ep{ep_index}_traj.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ---- Main --------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        default="outputs/train/act_banana_in_pot/checkpoints/010000/pretrained_model",
        help="Path to a saved pretrained_model dir (config.json + model.safetensors + processor jsons).",
    )
    p.add_argument("--episodes", default=",".join(map(str, DEFAULT_EPISODES)),
                   help="Comma-separated held-out episode indices, e.g. 45,46,47,48,49,50")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--out", default="eval_out", help="Output dir for figures.")
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--plot-episodes", default=None,
                   help="Comma-separated subset to plot (default: first 2 evaluated).")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Cap frames per episode (smoke/debug).")
    p.add_argument("--smoke", action="store_true",
                   help="Validate data+resize+GT+shape path with a FRESH random policy on CPU "
                        "(no checkpoint, no GPU). Metrics are meaningless; shapes are checked.")
    p.add_argument("--policy-type", default="act", choices=["act", "diffusion"],
                   help="Policy type for --smoke (default act). Ignored otherwise; "
                        "the checkpoint's config.json determines the class for real runs.")
    p.add_argument("--scheduler", default="asis", choices=["asis", "DDIM"],
                   help="Diffusion noise scheduler override applied before load "
                        "(default asis = as trained). Use DDIM for fast eval sampling. No-op for ACT.")
    p.add_argument("--num-inference-steps", type=int, default=None,
                   help="Diffusion denoising steps at eval (default: as trained). "
                        "e.g. 10 with --scheduler DDIM for ~10x rollout speedup. No-op for ACT.")
    return p.parse_args()


def resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda requested but not available; falling back to cpu.")
        return "cpu"
    return device


def main():
    args = parse_args()
    device = resolve_device(args.device)
    episodes = [int(x) for x in args.episodes.split(",") if x.strip() != ""]
    os.makedirs(args.out, exist_ok=True)
    transforms = build_image_transforms()

    # metadata (stats + camera keys) via a lightweight dataset handle on ep 0
    meta_ds = LeRobotDataset(args.repo_id, root=args.root, episodes=[episodes[0]],
                             video_backend="pyav")
    meta = meta_ds.meta
    camera_keys = list(meta.camera_keys)
    dim_names = resolve_dim_names(meta)
    del meta_ds

    if args.smoke:
        print(f"[smoke] Building FRESH random {args.policy_type} policy on CPU "
              f"(no checkpoint, no GPU).")
        policy, pre, post = build_smoke_policy(meta, "cpu", args.policy_type)
        device = "cpu"
    else:
        if not Path(args.checkpoint).is_dir():
            sys.exit(
                f"[error] checkpoint dir not found: {args.checkpoint}\n"
                f"        (first checkpoint appears at step 010000). "
                f"Use --smoke to validate the pipeline without one."
            )
        print(f"[info] loading checkpoint: {args.checkpoint} on {device}")
        policy, pre, post = load_policy_and_processors(
            args.checkpoint, device, meta.stats,
            scheduler=args.scheduler, num_inference_steps=args.num_inference_steps,
        )

    print(f"[info] action dims ({len(dim_names)}): {dim_names}")
    print(f"[info] cameras: {camera_keys}")
    print(f"[info] evaluating episodes: {episodes}")

    per_ep = []
    all_pred, all_gt = [], []
    ep_arrays = {}
    for ep in episodes:
        pred, gt = rollout_episode(
            policy, pre, post, ep, args.root, args.repo_id, transforms,
            device, camera_keys, max_frames=args.max_frames,
        )
        m = compute_metrics(pred, gt)
        per_ep.append((ep, m))
        all_pred.append(pred)
        all_gt.append(gt)
        ep_arrays[ep] = (pred, gt)
        print(f"[info] episode {ep}: T={m['T']}  overallL1={m['overall_l1']:.5f}  "
              f"gripAcc={m['grip_acc']:.3f}")

    agg_pred = np.concatenate(all_pred, axis=0)
    agg_gt = np.concatenate(all_gt, axis=0)
    agg = compute_metrics(agg_pred, agg_gt)
    print_table(per_ep, agg, dim_names)

    plot_eps = (
        [int(x) for x in args.plot_episodes.split(",")]
        if args.plot_episodes
        else episodes[:2]
    )
    for ep in plot_eps:
        if ep in ep_arrays:
            pred, gt = ep_arrays[ep]
            path = save_traj_plot(pred, gt, ep, args.out, dim_names)
            print(f"[info] wrote {path}")

    if args.smoke:
        d = agg_pred.shape[1]
        print("\n[smoke] OK: observation dict fed to select_action (pre-batch) was:")
        print(f"[smoke]   '{OBS_STATE}': ({agg_gt.shape[1]},) ; each camera: (3,360,640) float32 in [0,1]")
        print(f"[smoke]   After preprocessor: state (1,{agg_gt.shape[1]}), images (1,3,360,640) on device, normalized.")
        print(f"[smoke]   select_action -> (1,{d}) normalized -> postprocessor -> (1,{d}) unnormalized on cpu.")
        print(f"[smoke]   policy_type={args.policy_type}; dim-agnostic metrics/plots over {d} action dims OK.")


if __name__ == "__main__":
    main()
