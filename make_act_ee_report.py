#!/usr/bin/env python3
"""
Plot the EEF ACT (act_ee_val_diag) results: held-out eval_loss curve and the
open-loop offline eval trend (pose MAE + gripper accuracy) vs checkpoint, with the
JOINT Run-B gripper-accuracy reference overlaid (the one directly-comparable metric).

Inputs (all already on disk):
  - outputs/train/act_ee_val_diag/train.log            (eval_loss lines)
  - results/eval_ee_act_<STEP>/metrics.txt             (MEAN row -> poseMAE, gripAcc)
Outputs:
  - results/report_assets/act_ee_loss_curve.png
  - results/report_assets/act_ee_eval_trend.png
Run inside the container (matplotlib is in lr_env):
  ./lr_env/bin/python make_act_ee_report.py
"""
import os
import re
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_LOG = os.path.join(ROOT, "outputs/train/act_ee_val_diag/train.log")
ASSETS = os.path.join(ROOT, "results/report_assets")
os.makedirs(ASSETS, exist_ok=True)

# JOINT Run-B (docs/ACT_OVERFIT_DIAGNOSIS.md) — held-out valdiag reference.
JOINT_GRIPACC = {10000: 0.918, 20000: 0.938, 30000: 0.937, 40000: 0.937}

# ---- 1. eval_loss vs step (held-out) --------------------------------------
steps, evals = [], []
with open(TRAIN_LOG, errors="ignore") as f:
    for m in re.finditer(r"step (\d+): eval_loss=([\d.]+)", f.read()):
        steps.append(int(m.group(1)))
        evals.append(float(m.group(2)))
pairs = sorted(set(zip(steps, evals)))
steps = [s for s, _ in pairs]
evals = [e for _, e in pairs]

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(steps, evals, "-o", color="#2a6f97", ms=4, lw=1.6)
best_i = min(range(len(evals)), key=lambda i: evals[i])
ax.scatter([steps[best_i]], [evals[best_i]], color="#e63946", zorder=5,
           label=f"min {evals[best_i]:.4f} @ {steps[best_i]//1000}k")
ax.set_xlabel("training step")
ax.set_ylabel("held-out eval_loss (ACT)")
ax.set_title("ACT · EEF (10-D) — held-out eval_loss (eps 45–50)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
p1 = os.path.join(ASSETS, "act_ee_loss_curve.png")
fig.savefig(p1, dpi=130)
print("wrote", p1)

# ---- 2. open-loop trend: poseMAE + gripAcc vs checkpoint -------------------
ck_steps, pose_mae, grip_acc = [], [], []
for d in sorted(glob.glob(os.path.join(ROOT, "results/eval_ee_act_*"))):
    mfile = os.path.join(d, "metrics.txt")
    if not os.path.isfile(mfile):
        continue
    step = int(re.search(r"eval_ee_act_(\d+)", d).group(1))
    with open(mfile, errors="ignore") as f:
        for line in f:
            t = line.split()
            if t and t[0] == "MEAN":
                pose_mae.append(float(t[2]))
                grip_acc.append(float(t[5]))
                ck_steps.append(step)
                break
order = sorted(range(len(ck_steps)), key=lambda i: ck_steps[i])
ck_steps = [ck_steps[i] for i in order]
pose_mae = [pose_mae[i] for i in order]
grip_acc = [grip_acc[i] for i in order]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
# left: EEF open-loop pose MAE (native units: m + 6D)
axL.plot(ck_steps, pose_mae, "-o", color="#2a6f97", ms=5, lw=1.8)
bi = min(range(len(pose_mae)), key=lambda i: pose_mae[i])
axL.scatter([ck_steps[bi]], [pose_mae[bi]], color="#e63946", zorder=5,
            label=f"best {pose_mae[bi]:.4f} @ {ck_steps[bi]//1000}k")
axL.set_xlabel("checkpoint step")
axL.set_ylabel("open-loop pose MAE  (m + 6D, mixed units)")
axL.set_title("EEF ACT — open-loop pose MAE ↓")
axL.grid(alpha=0.3)
axL.legend()
# right: gripper accuracy — the ONE directly comparable metric (EEF vs JOINT)
axR.plot(ck_steps, grip_acc, "-o", color="#2a9d8f", ms=5, lw=1.8, label="EEF (10-D)")
jx = sorted(JOINT_GRIPACC)
axR.plot(jx, [JOINT_GRIPACC[s] for s in jx], "--s", color="#8d6e63", ms=5, lw=1.6,
         label="JOINT (7-D, Run B)")
axR.set_xlabel("checkpoint step")
axR.set_ylabel("gripper accuracy ↑")
axR.set_title("Gripper accuracy — directly comparable")
axR.grid(alpha=0.3)
axR.legend()
fig.suptitle("ACT · EEF vs JOINT — open-loop offline eval (held-out eps 45–50)", y=1.02)
fig.tight_layout()
p2 = os.path.join(ASSETS, "act_ee_eval_trend.png")
fig.savefig(p2, dpi=130, bbox_inches="tight")
print("wrote", p2)

# ---- console summary -------------------------------------------------------
print("\nEEF eval summary:")
print(f"  eval_loss: min {min(evals):.4f} @ {steps[best_i]//1000}k (last {evals[-1]:.4f} @ 50k)")
for s, pm, ga in zip(ck_steps, pose_mae, grip_acc):
    jr = JOINT_GRIPACC.get(s)
    d = f"  Δgrip vs JOINT {ga - jr:+.3f}" if jr is not None else ""
    print(f"  {s//1000:>2}k: poseMAE={pm:.5f}  gripAcc={ga:.3f}{d}")
