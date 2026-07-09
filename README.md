# banana-in-pot-experiments

Imitation-learning experiments for a **"put the right banana in the pot"** manipulation
task on a **UR7e arm teleoperated with GELLO**. The dataset is **51 episodes** (21,524
frames, 30 fps, ~11.96 min) of dual-RGB-camera teleop, converted to **LeRobot v3.0**. Two
policy families are trained and diagnosed here — **ACT** (Action Chunking Transformer) first,
then the **lerobot Diffusion Policy** — under a deliberate **train/val OVERFIT-DIAGNOSTIC**
methodology: hold out the **last 6 of 51 episodes** (`--dataset.eval_split=0.117`, eps 45–50)
as a true validation split and watch held-out signals while train loss falls. Everything is run
in two **action spaces**: **JOINT** (7-dim: UR q1..q6 + gripper) and **EEF** (10-dim: xyz +
6D rotation + gripper).

> Robot note: the LeRobot dataset metadata tags `robot_type=ur5e_gello` and repo_id
> `theo/banana_in_pot`; the physical arm and all deploy docs are UR7e.

---

## Headline finding — for Diffusion, `eval_loss` is a misleading overfit signal

For the **Diffusion Policy**, the held-out **denoising `eval_loss` ROSE** while the
deployment-relevant **open-loop rollout MAE** (from `eval_offline.py`, DDIM-10, on held-out
eps 45–50) **kept IMPROVING**. The denoising loss scores noise prediction at random diffusion
timesteps — not sampled-action accuracy — so the two decorrelate. **Select diffusion
checkpoints by open-loop MAE, never by `eval_loss`.**

JOINT diffusion, held-out eps 45–50 (radians):

| checkpoint | held-out `eval_loss` | open-loop poseMAE | gripAcc |
|---|---|---|---|
| 10k | (rising) | 0.1193 | 0.729 |
| 20k | ↑ | 0.1037 | 0.888 |
| 30k | ↑ | 0.0921 | 0.919 |
| **40k** | 0.0673 (44k) vs 0.0289 min @4k | **0.0907** | **0.949** |

`eval_loss` bottomed at **0.0289 (step 4k)** and rose to **0.0673 (44k)** — textbook "overfit"
— yet open-loop poseMAE and gripper accuracy improved **monotonically** 10k→40k. For **ACT the
two signals agreed** (open-loop best ≈ 30k: poseMAE 0.0978, gripAcc 0.937; `eval_loss`
0.6505→0.5252, never turned up). Full analysis: **[docs/DIFFUSION_JOINT_OVERFIT.md](docs/DIFFUSION_JOINT_OVERFIT.md)**.

---

## Quickstart

```bash
# 1. Clone with the deploy submodule
git clone --recursive <experiment-repo-url> && cd banana-in-pot-experiments

# 2. Environment: uv venv + locked deps + lerobot pinned-editable (~10 min)
./setup.sh          # smoke line should print: 2.11.0+cu128 True 0.6.1 0.35.2

# 3. Fetch the JOINT dataset from HF (and, for the EEF leg, rebuild the EE-action set)
./fetch_data.sh

# 4. Gate the GPU before ANY training (no lerobot-train + no CUDA compute process)
./gpu_gate.sh

# 5. Train JOINT diffusion (val-diag) — detached so an agent harness can't reap it
setsid nohup ./train_diffusion_joint_valdiag.sh </dev/null > train_dj.log 2>&1 &
#    setsid returns a WRAPPER pid; find the REAL training pid with:
pgrep -f 'lr_env/bin/lerobot-train.*diffusion_joint' | sort -n | head -1

# 6. Offline eval + checkpoint selection (DDIM-10 open-loop on held-out eps 45–50)
./lr_env/bin/python eval_offline.py \
    --checkpoint outputs/train/diffusion_joint_val_diag/checkpoints/040000/pretrained_model \
    --episodes 45,46,47,48,49,50 --device cuda --out eval_out_40k
```

CPU smoke (no GPU): `./lr_env/bin/python eval_offline.py --smoke --device cpu --episodes 0 --max-frames 8`.
Every hard-won caution (detached launch, resume syntax, shape flags, cache paths) is in
**TROUBLESHOOTING.md**.

---

## Repo map

### Scripts

| Script | What it does |
|---|---|
| `setup.sh` | Build `lr_env` uv venv (Python 3.12), install `requirements-lock.txt`, clone+checkout lerobot @ `8a74e0a` editable, smoke-test. |
| `fetch_data.sh` | Download JOINT dataset (+ raw data) from HF and rebuild the local-only EE-action dataset. |
| `gpu_gate.sh` | Exit 0 only when GPU is free: no `lerobot-train` process AND no CUDA **compute** (Type-C) process; display/graphics memory ignored. |
| `train_act.sh` | Standard ACT training; images resized on-the-fly to 360×640 (no dataset re-encode). |
| `train_act_valdiag.sh` | ACT overfit-diagnostic: train on 45 eps, hold out last 6, log held-out eval loss. |
| `train_act_resume_80k.sh` | Resume ACT from the 50k checkpoint to 80k via lerobot native `--resume`. |
| `train_diffusion_joint_valdiag.sh` | Diffusion Policy, JOINT (7-dim) action space, val-diagnostic run. |
| `train_diffusion_ee_valdiag.sh` | Diffusion Policy, EEF (10-dim) action space, val-diagnostic run (needs rebuilt EE-action dataset). |
| `convert_to_lerobot.py` | Raw per-take h5+mp4 → LeRobot v3.0 JOINT dataset (state/action 7-dim, 2 cams, nearest-timestamp resample). |
| `convert_to_lerobot_ee.py` | JOINT converter + adds `observation.tcp_pose` (7) and `observation.wrench` (6). |
| `convert_to_lerobot_ee_action.py` | Build the 10-dim EE-ACTION dataset (xyz + 6D rot + grip) by **video-reuse** from the JOINT set; `--selftest` unit tests. |
| `validate_ee_dataset.py` | 8-check validation gate for the EE-action dataset (51 eps / 21,524 frames, range + 6D + shift checks). |
| `eval_offline.py` | Open-loop offline eval: feed logged obs to `select_action`, compare vs ground-truth action; DDIM-10 for diffusion; `--smoke` CPU path. |
| `make_overfit_report.py` | Parse a val-diag training log → train-vs-held-out `eval_loss` plot + Markdown report + auto verdict. |
| `make_act_report.py` | Generate `ACT_RESULTS.md` from the training log + per-checkpoint `eval_out_*k/` logs. |
| `deploy_ur_act.py` | Reference inference-loop **skeleton** for running the trained ACT policy on the real UR arm (GELLO not used at deploy). |

### Docs

| Doc | One-liner |
|---|---|
| [DATASET_REPORT.md](docs/DATASET_REPORT.md) | LeRobot v3.0 conversion + 6-validator QA (schema, coverage, video, alignment, sanity, train-readiness). |
| [ACT_RESULTS.md](docs/ACT_RESULTS.md) | ACT training results on the 51-episode dataset. |
| [ACT_OVERFIT_DIAGNOSIS.md](docs/ACT_OVERFIT_DIAGNOSIS.md) | ACT train/val diagnostic — no destructive overfit; loss and open-loop MAE agree (best ≈ 30k). |
| [DIFFUSION_PLAN.md](docs/DIFFUSION_PLAN.md) | Execution-ready spec for the diffusion JOINT→EEF diagnostic (feature spec, splits, flags). |
| [DIFFUSION_JOINT_OVERFIT.md](docs/DIFFUSION_JOINT_OVERFIT.md) | **The headline finding**: `eval_loss` rises while open-loop MAE improves; select by MAE. |
| [DEPLOY_REPO_DECISION.md](docs/DEPLOY_REPO_DECISION.md) | Repo-setup decision for real-robot ACT/HIL-SERL deploy on the UR7e. |
| [DEPLOY_UR.md](docs/DEPLOY_UR.md) | How to run the trained ACT policy on the real UR arm. |
| [HILSERL_PREP_PLAN.md](docs/HILSERL_PREP_PLAN.md) | Offline, robot-free plan to prepare everything up to HIL-SERL online RL. |
| [HILSERL_PREP_RESULTS.md](docs/HILSERL_PREP_RESULTS.md) | Results of the offline HIL-SERL preparation. |
| [HILSERL_RUNBOOK.md](docs/HILSERL_RUNBOOK.md) | HIL-SERL online training runbook (UR7e). |

### Data & models

Nothing large is committed (repo < 1 MB, no git-lfs). Assets are fetched from Hugging Face or rebuilt.

| Asset | Size | Source | Strategy |
|---|---|---|---|
| JOINT dataset (LeRobot v3) | 483 MB | [`Bigenlight/banana_in_pot_lerobot_v3`](https://huggingface.co/datasets/Bigenlight/banana_in_pot_lerobot_v3) | HF download |
| Raw h5+mp4 | 745 MB | [`Bigenlight/banana_in_pot_raw`](https://huggingface.co/datasets/Bigenlight/banana_in_pot_raw) | HF download (only to rebuild datasets) |
| EE-action dataset (10-dim) | 484 MB | not on HF (401) | **Rebuild** via `convert_to_lerobot_ee_action.py` (video-reuse; needs JOINT + raw) |
| EE-obs dataset | — | [`Bigenlight/banana_in_pot_ee_lerobot_v3`](https://huggingface.co/datasets/Bigenlight/banana_in_pot_ee_lerobot_v3) | HF download (optional) |
| Pretrained ACT model | 4.7 GB | [`Bigenlight/act_banana_in_pot`](https://huggingface.co/Bigenlight/act_banana_in_pot) | HF download (or re-train) |
| ACT / diffusion val-diag checkpoints | 2.4 / 19 GB | not on HF | **Re-train**; push only the best-by-open-loop-MAE checkpoint |
| `results/eval_out*`, `report_assets` | ~2 MB | committed | Reference outputs for qualitative comparison |

---

## Hardware / software assumptions

- **GPU:** RTX 3060 12GB (this box). Batch 8 fp32 uses ~9.7 GB — reduce `--batch_size` on smaller GPUs.
- **NVIDIA driver ≥ 570** (this box 590.48.01) — required by the cu128 wheels.
- **CUDA:** cu128 wheels (`torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`).
- **Python 3.12** (uv-managed venv, 119 packages; `lerobot==0.6.1` editable).
- **Ubuntu 22.04** — required **only** for the ROS2 Humble real-robot deploy step.
- **Disk:** ~2 GB for data + **20 GB+** for a full checkpoint run (`outputs/` can reach ~19–26 GB).

Bitwise reproduction is a non-goal — a different GPU changes nondeterminism, so numbers match in
shape (curve trends, MAE ranking), not in exact digits.

---

## Non-obvious flags (do NOT remove)

| Flag | Why |
|---|---|
| `--policy.resize_shape='[360,640]'` | Diffusion's SpatialSoftmax is input-shape-rigid; meta says 720×1280 but the loader resizes to 360×640 — without this the **first forward crashes**. (ACT tolerates it; diffusion does not.) |
| `--policy.drop_n_last_frames=31` | lerobot's diffusion default of 7 is wrong for `n_obs_steps=2 / horizon=64 / n_action_steps=32` on these short episodes. |
| `--dataset.eval_split=0.117` | Holds out the **last 6 of 51** episodes (eps 45–50), matching the `eval_offline.py` convention. |

Full flag list and symptom→fix table: **TROUBLESHOOTING.md**.

---

## Pointers

- **Deploy stack** (`gello_software/` submodule) — real-UR7e deploy on **ROS2 Humble**; see
  `deploy_ur_act.py`, [docs/DEPLOY_UR.md](docs/DEPLOY_UR.md),
  [docs/DEPLOY_REPO_DECISION.md](docs/DEPLOY_REPO_DECISION.md). HIL-SERL: [docs/HILSERL_RUNBOOK.md](docs/HILSERL_RUNBOOK.md).
- **lerobot pin** = `8a74e0a` (0.6.1), cloned editable and gitignored by `setup.sh` (not a submodule).
- **repo_id-mismatch caveat:** the dataset's `meta/info.json` carries `repo_id: theo/banana_in_pot`
  while the HF mirror is `Bigenlight/...`. Harmless under `--dataset.root=` + `HF_HUB_OFFLINE=1`, but
  do **not** "fix" it, and do not call `LeRobotDataset("theo/banana_in_pot")` without `root=`
  (§9-R5 of the plan).
- Full design rationale: **[REPRODUCIBILITY_PLAN.md](REPRODUCIBILITY_PLAN.md)**.
  Operational cautions: **TROUBLESHOOTING.md** · narrative timeline: **EXPERIMENT_LOG.md** (both to be written).
