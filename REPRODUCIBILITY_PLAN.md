# REPRODUCIBILITY_PLAN.md — turning this experiment into a clone-and-rerun GitHub repo

Status: DESIGN document. Nothing below has been executed — no `git init`, no pushes,
no data moved. A diffusion training run is live on the GPU as of writing.

All sizes/versions/HF statuses below were measured on this box on 2026-07-09.

---

## 1. Feasibility verdict + recommendation ("어떻게 생각해?")

**Verdict: YES — cleanly reproducible, with one honest caveat.**

The experiment is in unusually good shape for repro:

- Every training/eval entrypoint is already a small script (14 scripts, all < 25 KB).
- Both nested repos are **clean and pinned**: `gello_software` @ `0730fb42` on
  `feat/gello-ur7e-humble-22.04` (pushed, in sync with `origin`), `lerobot` @
  `8a74e0ac` (upstream huggingface, no local commits — verified `git status` clean).
- The JOINT dataset, the raw data, and the ACT model are **already public on HF**
  (verified HTTP 200): `Bigenlight/banana_in_pot_lerobot_v3`,
  `Bigenlight/banana_in_pot_raw`, `Bigenlight/act_banana_in_pot`. The EE-obs dataset
  is also public (`Bigenlight/banana_in_pot_ee` → redirects to
  `Bigenlight/banana_in_pot_ee_lerobot_v3`).
- The environment is a plain uv venv (Python 3.12.3, 119 packages) — trivially
  freezable.

The caveat: **every train script and `eval_offline.py` hardcodes two absolute
paths** — `cd /home/theo_lab/Downloads/Put_right_banana...` and a session-specific
`/tmp/claude-1002/...scratchpad` cache dir. These MUST be parameterized (a
~20-line diff) before the repo is created, or nothing runs on another PC.
Also, one dataset (`banana_in_pot_ee_action_lerobot`, the 10-dim EEF-action one)
and the val-diag checkpoints are **local-only** (HF returns 401) — handled in §4.

**Single best strategy:** a new small "experiment repo"
(`Bigenlight/banana-in-pot-experiments` or similar) that contains *only* scripts +
docs (~200 KB of git-tracked content), with:
- `gello_software` as a **git submodule** (we own it),
- `lerobot` as a **setup-script pinned clone** (third-party, gitignored),
- datasets/models **fetched from HF or rebuilt** by a `fetch_data.sh` /
  converter chain (never committed),
- a committed `requirements-lock.txt` from `uv pip freeze` + `setup.sh`.

Estimated fresh-PC time-to-first-training-step: ~30 min (mostly torch download
and the 483 MB dataset pull).

---

## 2. Recommended repo structure

```
banana-in-pot-experiments/            # NEW top-level git repo
├── README.md                         # NEW — see §8 skeleton
├── TROUBLESHOOTING.md                # NEW — the 10 hard-won cautions, see §8
├── EXPERIMENT_LOG.md                 # NEW — narrative timeline (optional but valuable)
├── REPRODUCIBILITY_PLAN.md           # this file
├── .gitignore                        # see §6
├── .gitmodules                       # gello_software only
├── setup.sh                          # NEW — env + lerobot pin (see §5)
├── fetch_data.sh                     # NEW — HF downloads + EE-action rebuild (see §4)
├── requirements-lock.txt             # NEW — from `uv pip freeze` (119 pkgs)
│
├── scripts/                          # (or keep flat at root — either works; tree
│   │                                 #  below assumes flat-at-root like today)
│   ├── train_act.sh
│   ├── train_act_valdiag.sh
│   ├── train_act_resume_80k.sh
│   ├── train_diffusion_joint_valdiag.sh
│   ├── train_diffusion_ee_valdiag.sh
│   ├── gpu_gate.sh
│   ├── convert_to_lerobot.py
│   ├── convert_to_lerobot_ee.py
│   ├── convert_to_lerobot_ee_action.py
│   ├── validate_ee_dataset.py
│   ├── eval_offline.py
│   ├── make_overfit_report.py
│   ├── make_act_report.py
│   └── deploy_ur_act.py
│
├── docs/
│   ├── DATASET_REPORT.md
│   ├── ACT_RESULTS.md
│   ├── ACT_OVERFIT_DIAGNOSIS.md
│   ├── DIFFUSION_PLAN.md
│   ├── DIFFUSION_JOINT_OVERFIT.md
│   ├── DEPLOY_REPO_DECISION.md
│   ├── DEPLOY_UR.md
│   ├── HILSERL_PREP_PLAN.md
│   ├── HILSERL_PREP_RESULTS.md
│   └── HILSERL_RUNBOOK.md
│
├── results/                          # small committed reference outputs (~2 MB)
│   ├── eval_out/ eval_out_10k/ ... eval_out_50k/
│   └── report_assets/
│
├── gello_software/                   # git SUBMODULE @ 0730fb42 (deploy stack)
│
│ # ---- everything below is GITIGNORED, created by setup.sh / fetch_data.sh ----
├── lerobot/                          # pinned clone @ 8a74e0a (setup.sh)
├── lr_env/                           # uv venv (setup.sh)
├── banana_in_pot_lerobot/            # HF download        (483 MB)
├── banana_in_pot_ee_action_lerobot/  # rebuilt locally     (484 MB)
├── Put_right_banana_in_the_pot/      # HF download, only for rebuild (745 MB)
└── outputs/                          # training checkpoints (26 GB here — never commit)
```

Note on `scripts/` vs flat: the current scripts `cd` to repo root and use `./`
relative paths, so if you move them into `scripts/`, change the `cd` line to
`cd "$(dirname "$0")/.."`. Flat-at-root is the zero-risk option; `scripts/` is
tidier. Either way the `cd`/scratchpad fix in §9-R1 is mandatory.

---

## 3. Repo-in-repo decision, argued

Two nested repos, two different answers.

### 3a. `gello_software` → **git submodule** (firm recommendation)

Facts: remote `git@github.com:Bigenlight/gello_software.git` (our org, write
access), branch `feat/gello-ur7e-humble-22.04` @ `0730fb4212b425fbbd49162d59ce08be6896cedc`,
working tree clean, branch pushed and in sync with origin. 279 MB working tree.
It contains the actual deploy package (`gello_policy` ACT real-robot deploy for UR7e).

Why submodule and not the alternatives:

- **We own the remote and will keep committing to it.** The deploy package
  co-evolves with the experiment (e.g. when the diffusion policy gets its own
  deploy node). A submodule lets you commit inside `gello_software/`, push to its
  own remote, then bump the pointer in the experiment repo — one atomic,
  reviewable "deploy stack moved to X" commit. Exactly the workflow subtrees make
  painful (subtree merges pollute history, splitting commits back out is fragile).
- **Pinning is structural, not documentary.** The superproject records the exact
  SHA; `git clone --recursive` can't silently drift the way a README instruction can.
- **Vendoring (subtree) would bloat the experiment repo** with 279 MB of ROS2
  workspace history that already lives in its own repo, and would fork the truth:
  robot-side deploys would still happen from the standalone repo.
- Cost: teammates need read access to the Bigenlight org repo (it's private-ish).
  That's acceptable — anyone deploying to the lab UR7e needs org access anyway —
  but document the HTTPS+PAT fallback for people without SSH keys (below).

Exact commands (when creating the repo — NOT now):

```bash
git submodule add -b feat/gello-ur7e-humble-22.04 \
    git@github.com:Bigenlight/gello_software.git gello_software
git -C gello_software checkout 0730fb4212b425fbbd49162d59ce08be6896cedc
git add .gitmodules gello_software
# fresh-PC side:
git clone --recursive <experiment-repo-url>
# or, after a plain clone:
git submodule update --init gello_software
# HTTPS fallback for users without SSH keys:
git config submodule.gello_software.url https://github.com/Bigenlight/gello_software.git
git submodule sync && git submodule update --init
```

### 3b. `lerobot` → **documented pinned clone in `setup.sh`, gitignored** (firm recommendation)

Facts: plain clone of `https://github.com/huggingface/lerobot.git`, on `main` @
`8a74e0ac6d01706d67fddfed682a09d694d9c8c0` ("Bump lerobot to 0.6.1"), **zero local
commits, clean working tree** (verified). Installed editable into `lr_env`.
177 MB working tree.

Why *not* a submodule here:

- **It's a pure third-party read-only pin.** We have no write access, will never
  push, and there is no local delta to preserve. A submodule's main benefit
  (structural pinning) is fully matched by a SHA hardcoded in a 3-line `setup.sh`
  block that lives in git — equally reproducible, reviewable, and drift-proof.
- **Submodules cost every teammate friction** (`--recursive` forgotten, detached
  HEADs, CI checkout config) — worth paying once for gello, not twice for a repo
  we treat as a frozen dependency.
- **Flexibility:** if lerobot 0.6.1 wheels ever suffice, the setup.sh block
  collapses to `uv pip install "lerobot @ git+https://github.com/huggingface/lerobot.git@8a74e0a..."`.
  We keep the local editable clone for now because being able to read/patch
  lerobot source in place was operationally useful (the `resize_shape` and
  `drop_n_last_frames` discoveries came from reading it).
- Why not subtree: same bloat argument as gello, worse — 177 MB of upstream
  history we'll never touch.

Exact commands (this is the `setup.sh` block):

```bash
LEROBOT_PIN=8a74e0ac6d01706d67fddfed682a09d694d9c8c0   # lerobot 0.6.1
git clone https://github.com/huggingface/lerobot.git lerobot
git -C lerobot checkout "$LEROBOT_PIN"
uv pip install --python ./lr_env/bin/python -e ./lerobot
```

---

## 4. Data & model strategy

Legend: COMMIT (in git) / HF (download) / REBUILD (script) / SKIP.

| Asset | Size (measured) | On HF? (verified) | Strategy |
|---|---|---|---|
| `banana_in_pot_lerobot` (JOINT, LeRobot v3) | 483 MB | PUBLIC — `Bigenlight/banana_in_pot_lerobot_v3` (200) | **HF download** |
| `Put_right_banana_in_the_pot` (raw h5+mp4) | 745 MB | PUBLIC — `Bigenlight/banana_in_pot_raw` (200) | **HF download**, only when rebuilding datasets |
| `banana_in_pot_ee_action_lerobot` (10-dim EEF action) | 484 MB | **NO** (401 — local-only) | **REBUILD** via converter (default), + recommend a one-time HF push later |
| EE-obs dataset (`hf_upload/banana_in_pot_ee`) | in 712 MB `hf_upload/` | PUBLIC — redirects to `Bigenlight/banana_in_pot_ee_lerobot_v3` (200) | **HF download** (optional; not needed for the two diffusion runs) |
| `outputs/train/act_banana_in_pot` | 4.7 GB | PUBLIC — `Bigenlight/act_banana_in_pot` (model, 200) | **HF download** (or re-train) |
| `outputs/train/act_banana_val_diag` | 2.4 GB | NO (401) | **RE-TRAIN** (it's a diagnostic artifact); optionally push selected ckpts later |
| `outputs/train/diffusion_joint_val_diag` | 19 GB (run live) | NO (401) | **RE-TRAIN**; after the run, push only the best-by-open-loop-MAE checkpoint (~a few GB), never the 19 GB tree |
| `eval_out*`, `report_assets` | ~2 MB total | — | **COMMIT** as reference results |
| `hf_upload/`, `hilserl/` (83 MB), `scratch_convert_ee.log` | 712 MB / 83 MB | — | **SKIP** (gitignore; `hf_upload` is a staging dir, `hilserl` is rebuildable per HILSERL_RUNBOOK.md) |

**No git-lfs needed** — nothing large is committed. This keeps the repo < 1 MB.

### Download commands (`fetch_data.sh`)

```bash
# huggingface_hub 1.22.0 is already in the lock; its CLI is `hf`
hf download Bigenlight/banana_in_pot_lerobot_v3 --repo-type dataset \
    --local-dir ./banana_in_pot_lerobot
# raw source — ONLY needed to rebuild the EE-action dataset (or from scratch):
hf download Bigenlight/banana_in_pot_raw --repo-type dataset \
    --local-dir ./Put_right_banana_in_the_pot
# pretrained ACT (optional, to skip re-training):
hf download Bigenlight/act_banana_in_pot --local-dir ./outputs/train/act_banana_in_pot/pretrained_from_hf
```

Caveat: the local dataset's `meta/info.json` carries `repo_id: theo/banana_in_pot`
while the HF mirror is under `Bigenlight/`. Harmless for our scripts (everything
uses `--dataset.root=` + `HF_HUB_OFFLINE=1`), but do not "fix" the repo_id — some
lerobot paths key caches off it. `fetch_data.sh` should sanity-check
`meta/info.json` exists after download.

### The local-only EE-action dataset — explicit handling

This is the one asset that would be silently lost. Two-pronged:

1. **Primary (works today): local rebuild.** It was built by **video-reuse**
   (AV1-encoded videos copied from the JOINT dataset; only parquet + stats
   regenerated), so the rebuild REQUIRES the JOINT dataset AND the raw h5 present
   first. Exact chain (CLI args verified against the script):

   ```bash
   ./lr_env/bin/python convert_to_lerobot_ee_action.py --selftest   # unit tests, no data needed
   ./lr_env/bin/python convert_to_lerobot_ee_action.py \
       --data ./Put_right_banana_in_the_pot \
       --source ./banana_in_pot_lerobot \
       --out ./banana_in_pot_ee_action_lerobot \
       --repo-id theo/banana_in_pot_ee_action
   ./lr_env/bin/python validate_ee_dataset.py    # 51 eps / 21524 frames, range checks
   ```

   CPU-only, no GPU needed; a few minutes (no video re-encode thanks to reuse).

2. **Recommended follow-up (not part of this design task, requires a push):**
   one-time `hf upload Bigenlight/banana_in_pot_ee_action_lerobot ./banana_in_pot_ee_action_lerobot --repo-type dataset`
   so future re-runners skip the raw download. Until then, the rebuild path is
   canonical and `fetch_data.sh` implements it.

---

## 5. Dependency reproduction (`lr_env` on a fresh PC)

Measured ground truth on this box:

- uv **0.11.8**, Python **3.12.3** (uv venv, `include-system-site-packages=false`)
- `torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`, `torchcodec==0.11.1`
- `lerobot==0.6.1` **editable** from `./lerobot` @ 8a74e0a
- `diffusers==0.35.2`, `transformers==5.13.0`, `datasets==4.8.5`,
  `huggingface-hub==1.22.0`, `av==15.1.0`, `h5py==3.16.0`, `numpy==2.2.6`,
  `opencv-python-headless==4.13.0.92`, `pyarrow==24.0.0`, `pandas==2.3.3`,
  `einops==0.8.2`, `draccus==0.10.0`, `gymnasium==1.3.0`, `wandb==0.27.2`,
  `matplotlib==3.11.0` — **119 packages total** in the freeze.

### Lock strategy

Commit **`requirements-lock.txt`** generated by:

```bash
uv pip freeze --python ./lr_env/bin/python > requirements-lock.txt
```

(Note: the venv has no `pip` module — it's uv-managed; `python -m pip freeze`
fails. Use `uv pip freeze`.) Then hand-edit exactly one line: the freeze emits
`-e /home/theo_lab/Downloads/.../lerobot` — replace with `-e ./lerobot` (or drop
the line and let setup.sh install it, which is cleaner because the clone must
exist first). Recommended: **drop the lerobot line from the lock; setup.sh
installs it last.**

### `setup.sh` (committed; the full recipe)

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# 0. prereqs: uv (any >=0.11), NVIDIA driver supporting CUDA 12.8 (>=570.x;
#    this box: 590.48.01 on RTX 3060 12GB), ffmpeg on PATH (torchcodec/av need it)
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. venv
uv venv lr_env --python 3.12

# 2. locked deps — the +cu128 wheels resolve via the pytorch extra index
uv pip install --python ./lr_env/bin/python \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r requirements-lock.txt

# 3. lerobot, pinned + editable
LEROBOT_PIN=8a74e0ac6d01706d67fddfed682a09d694d9c8c0
[ -d lerobot ] || git clone https://github.com/huggingface/lerobot.git lerobot
git -C lerobot checkout "$LEROBOT_PIN"
uv pip install --python ./lr_env/bin/python --no-deps -e ./lerobot

# 4. smoke
./lr_env/bin/python -c "import torch,lerobot,diffusers; \
  print(torch.__version__, torch.cuda.is_available(), lerobot.__version__, diffusers.__version__)"
```

CUDA/driver assumption, stated plainly: wheels are **cu128**; the host needs an
NVIDIA driver new enough for CUDA 12.8 (>=570 series). VRAM: batch 8 fp32 uses
~9.7 GB on the RTX 3060 12GB — on smaller GPUs reduce `--batch_size` (and expect
different loss curves); on bigger GPUs it just runs. **CPU/other-GPU fallback:**
swap the extra-index-url for `https://download.pytorch.org/whl/cpu` (or the
matching cuXXX) and strip `+cu128` local-version suffixes from the lock — CPU is
viable only for `eval_offline.py --smoke` and the dataset converters, not training.

---

## 6. `.gitignore`

```gitignore
# environment (rebuilt by setup.sh)
lr_env/
__pycache__/
*.pyc

# third-party pinned clone (setup.sh) — NOT a submodule
lerobot/

# datasets & raw data (fetch_data.sh / converters)
banana_in_pot_lerobot/
banana_in_pot_ee_action_lerobot/
Put_right_banana_in_the_pot/

# training outputs & staging (26 GB / 712 MB / 83 MB here)
outputs/
hf_upload/
hilserl/

# caches & logs
torch_home/
hf_lerobot_home/
wandb/
*.log
scratch_*

# NOTE: gello_software/ is intentionally NOT ignored — it's a submodule,
# git tracks it as a gitlink, not as a working tree.
```

(If `eval_out*`/`report_assets` are moved under `results/` as in §2, no rule is
needed; if kept at root, do NOT ignore them — they're the 2 MB reference results.)

---

## 7. Reproduction runbook — fresh PC, step by step

Prereqs: Linux x86_64 (Ubuntu 22.04 recommended — required only for the ROS2
deploy step), NVIDIA GPU >= 10 GB VRAM with driver >= 570, GitHub access to
`Bigenlight/gello_software`, ffmpeg, uv.

1. **Clone (with submodule).**
   ```bash
   git clone --recursive <experiment-repo-url> && cd banana-in-pot-experiments
   # forgot --recursive? → git submodule update --init gello_software
   ```
2. **Environment.** `./setup.sh` (≈10 min; §5). Verify the smoke line prints
   `2.11.0+cu128 True 0.6.1 0.35.2`.
3. **Fix the cache path (CRITICAL).** The train scripts and `eval_offline.py`
   were written with a session-specific scratchpad; the repo-ified versions must
   take `SC="${SCRATCH_DIR:-$PWD/.cache}"`. If any script still contains
   `/tmp/claude-1002/...` or `cd /home/theo_lab/...`, **edit before running** —
   these are the #1 cross-PC failure. The three env vars that must point at
   writable dirs: `TORCH_HOME`, `HF_LEROBOT_HOME`, and `HF_HUB_OFFLINE=1`.
4. **Fetch data.** `./fetch_data.sh` → downloads the JOINT dataset (483 MB) and,
   for the EEF leg, the raw data (745 MB) + rebuilds `banana_in_pot_ee_action_lerobot`
   (§4 chain, CPU-only). Validate: `./lr_env/bin/python validate_ee_dataset.py`.
5. **Smoke eval (CPU-ok).**
   `./lr_env/bin/python eval_offline.py --smoke` — exercises dataset + policy
   forward without GPU commitment.
6. **GPU gate.** `./gpu_gate.sh` before ANY training. Know what it checks: "no
   `lerobot-train` process AND no CUDA *compute* process" (this lab box has a
   desktop session holding ~1 GB as Type-**G** graphics, which is fine and
   deliberately not counted). On a headless PC it should trivially pass; do not
   convert it to a memory threshold.
7. **Train JOINT diffusion (val-diag).** Canonical detached launch — background
   jobs get reaped by agent harnesses unless detached:
   ```bash
   setsid nohup ./train_diffusion_joint_valdiag.sh </dev/null > train_dj.log 2>&1 &
   # setsid returns a WRAPPER pid; the real main process is:
   pgrep -f 'lr_env/bin/lerobot-train.*diffusion_joint' | sort -n | head -1
   ```
   Non-negotiable flags already baked into the script — do not remove:
   - `--policy.resize_shape='[360,640]'` — diffusion's SpatialSoftmax is
     input-shape-rigid; meta says 720x1280 but the loader resizes to 360x640, and
     without this flag the FIRST forward crashes. (ACT tolerates it; diffusion doesn't.)
   - `--policy.drop_n_last_frames=31` — lerobot's diffusion default of 7 is wrong
     for n_obs_steps=2 / horizon=64 / n_action_steps=32 on these episodes.
   - `--dataset.eval_split=0.117` — holds out the LAST 6 of 51 episodes (45–50).
8. **Train EEF diffusion.** Same launch with `./train_diffusion_ee_valdiag.sh`
   (requires step 4's rebuilt EE-action dataset). Resume syntax if interrupted:
   `--config_path=<ckpt>/pretrained_model/train_config.json --resume=true --steps=N`.
9. **Offline eval + checkpoint selection.** `eval_offline.py` on the held-out
   episodes (45–50), DDIM-10 open-loop rollout. **THE headline lesson: for
   diffusion, held-out denoising `eval_loss` is a MISLEADING overfit signal — it
   rose while open-loop rollout MAE (the deployment-relevant metric) kept
   improving. Select checkpoints by open-loop MAE from `eval_offline.py`, never
   by eval_loss.** (For ACT the two signals agreed; for diffusion they diverge.)
10. **Reports.** `make_overfit_report.py` / `make_act_report.py` → compare against
    the committed `results/` reference outputs.
11. **(Deploy pointer.)** Real-UR deploy lives in the `gello_software` submodule
    (ROS2 Humble / Ubuntu 22.04): see `docs/DEPLOY_UR.md`,
    `docs/DEPLOY_REPO_DECISION.md`, and `deploy_ur_act.py`. HIL-SERL prep:
    `docs/HILSERL_RUNBOOK.md`.

---

## 8. Docs to ship + gaps

**Ship as-is (all exist, all small) →** `docs/`:
DATASET_REPORT.md, ACT_RESULTS.md, ACT_OVERFIT_DIAGNOSIS.md, DIFFUSION_PLAN.md,
DIFFUSION_JOINT_OVERFIT.md, DEPLOY_REPO_DECISION.md, DEPLOY_UR.md,
HILSERL_PREP_PLAN.md, HILSERL_PREP_RESULTS.md, HILSERL_RUNBOOK.md.

**Gaps — must be written when the repo is created:**

1. **`README.md`** (top-level; doesn't exist). Skeleton:
   - What this is: put-banana-in-pot manipulation, UR7e + gello teleop, 51
     episodes; ACT → Diffusion Policy; train/val overfit-diagnostic methodology;
     JOINT and EEF action spaces.
   - **Headline finding** (one paragraph, up top): diffusion eval_loss vs
     open-loop MAE divergence — link DIFFUSION_JOINT_OVERFIT.md.
   - Quickstart: the §7 runbook compressed to 6 commands.
   - Repo map: scripts table (one line each), docs index, data table from §4
     with HF links.
   - Hardware/software assumptions: RTX 3060 12GB / driver >= 570 / cu128 /
     Ubuntu 22.04 (deploy only) / ~2 GB disk for data + 20+ GB for checkpoints.
   - Pointers: gello_software submodule = deploy; lerobot pin = 8a74e0a.
2. **`TROUBLESHOOTING.md`** (doesn't exist) — the 10 cautions as symptom→cause→fix:
   1. "My training died when the session ended" → harness reaps background jobs →
      `setsid nohup ... </dev/null >LOG 2>&1 &`; wrapper-pid vs real-pid `pgrep` recipe.
   2. "Diffusion crashes on the first forward with a shape error" → SpatialSoftmax
      rigid vs 720x1280 meta / 360x640 loader → `--policy.resize_shape='[360,640]'`.
   3. "IndexError / short-episode sampling weirdness" → `--policy.drop_n_last_frames=31`
      (diffusion default 7 wrong for horizon 64 / n_action_steps 32).
   4. "eval_loss is going UP, is it overfitting?" → for diffusion, eval_loss is
      misleading; judge by `eval_offline.py` open-loop MAE (DDIM-10). ACT: signals agree.
   5. "Permission denied / weird cache writes" → set `TORCH_HOME`,
      `HF_LEROBOT_HOME` writable + `HF_HUB_OFFLINE=1`; grep scripts for stale
      absolute paths before first run.
   6. "gpu_gate fails/passes unexpectedly" → it checks processes (lerobot-train +
      CUDA compute), NOT memory; desktop Type-G graphics allocation is expected on
      this lab box, absent on headless.
   7. "How do I resume?" → `--config_path=<ckpt>/pretrained_model/train_config.json --resume=true --steps=N`.
   8. "What's the val split?" → eval_split=0.117 = last 6 of 51 episodes (45–50);
      eval_offline evaluates exactly those.
   9. "OOM at batch 8" → 9.7 GB fp32 on 12 GB; reduce batch on smaller GPUs.
   10. "EE-action rebuild fails" → video-reuse: JOINT dataset must exist locally
       BEFORE running `convert_to_lerobot_ee_action.py` (plus raw h5).
3. **`EXPERIMENT_LOG.md`** (optional, recommended) — dated narrative: dataset
   conversion → ACT 50k/80k → overfit diagnosis → diffusion plan → JOINT run →
   the eval_loss-vs-MAE discovery. The existing MDs are per-topic; a timeline
   stitches them for a newcomer.
4. **`setup.sh` + `fetch_data.sh` + `requirements-lock.txt`** (§4/§5) — these are
   code-gaps, not doc-gaps, but they don't exist yet.

---

## 9. Risks & sharp edges for cross-PC repro

- **R1 — Absolute-path leakage (WILL bite, fix first).** Every train script has
  `cd /home/theo_lab/Downloads/Put_right_banana...` and
  `SC=/tmp/claude-1002/.../scratchpad`; `eval_offline.py` hardcodes both too
  (lines ~30–36, ~64). Repo-ification requires a pass replacing these with
  `cd "$(dirname "$0")"` and `SC="${SCRATCH_DIR:-$PWD/.cache}"`. Add a CI-ish
  guard: `grep -rn '/home/theo_lab\|/tmp/claude' *.sh *.py` must return empty.
- **R2 — Auth / private repos.** `gello_software` submodule needs Bigenlight org
  read access (SSH key or PAT; HTTPS fallback in §3a). The EE-action dataset and
  val-diag checkpoints are NOT on HF (401) — until someone pushes them, the
  rebuild path is the only route, which drags in the 745 MB raw download. HF
  token needed only if those get pushed private.
- **R3 — Hardware/driver mismatch.** cu128 wheels need driver >= 570 (this box:
  590.48.01). <12 GB VRAM breaks batch 8 (~9.7 GB fp32); different GPU → different
  nondeterminism, so numbers will match in shape (curves, MAE ranking), not in
  digits. Bitwise checkpoint repro is explicitly a non-goal; the committed
  `results/` references are for qualitative comparison.
- R4 — Dataset governance: robot lab data with visible workspace is already
  public on HF under Bigenlight; confirm the lab is fine with that exposure and
  slap an explicit license (e.g. CC-BY-NC) on the HF dataset cards and README.
- R5 — repo_id mismatch (`theo/...` in meta vs `Bigenlight/...` on HF) is benign
  under `--dataset.root` + `HF_HUB_OFFLINE=1` but will confuse anyone who tries
  `LeRobotDataset("theo/banana_in_pot")` without `root=` — document in README.
- R6 — The 19 GB diffusion checkpoint tree: never push wholesale; select
  by open-loop MAE, push one checkpoint. Also `outputs/` on a fresh PC needs
  ~20+ GB free disk for a full re-run.
