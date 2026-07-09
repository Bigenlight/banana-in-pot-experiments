# DIFFUSION_PLAN.md — Diffusion Policy train/val overfit diagnostic (JOINT → EEF)

**Status:** execution-ready spec for the Opus dev/review/train team.
**Author:** planner (fable5), 2026-07-08. Source of truth for facts:
`DIFFUSION_RESEARCH_BRIEF.md` (scratchpad), `train_act_valdiag.sh`, `DATASET_REPORT.md`,
`lerobot/src/lerobot/policies/diffusion/configuration_diffusion.py`, `lerobot/AGENT_GUIDE.md` §6–7.

**Goal:** reproduce the ACT overfit-diagnostic methodology (45 train eps / 6 held-out eps
45–50 via `--dataset.eval_split=0.117`, held-out `eval_loss` every 2000 steps,
`make_overfit_report.py` train-vs-val curve + verdict, `eval_offline.py` open-loop held-out MAE)
for **Diffusion Policy**, twice, sequentially:

1. **JOINT** action space (existing dataset `./banana_in_pot_lerobot`, action = 7-dim absolute joints+grip).
2. **EEF** action space (**new dataset must be built**: 10-dim absolute EE pose
   `[x,y,z, r1..r6 (6D rot), grip]`).

All decisions below are RESOLVED — execute as written; fallbacks are listed where a real risk exists.

Paths used throughout:

```
PROJECT = /home/theo_lab/Downloads/Put_right_banana_in_the_pot-20260707T103848Z-3-001
SC      = /tmp/claude-1002/-home-theo-lab-Downloads-Put-right-banana-in-the-pot-20260707T103848Z-3-001/cbc6765d-7359-4632-9edc-c7f3f13db195/scratchpad
```

---

## 1. Phase overview & GPU gating

| Phase | Work | GPU? | Can start |
|---|---|---|---|
| P0 | Environment (diffusers 0.35.2 installed in `./lr_env`) | – | ✅ DONE |
| P1 | CPU prep in parallel: (a) EEF dataset build + validation, (b) eval_offline/report patches, (c) write both train scripts | **No** | **Immediately** (while ACT diagnostic still runs) |
| P2 | GPU gate → JOINT smoke (300 steps) → JOINT full run (100k) | Yes | After ACT frees GPU |
| P3 | JOINT report (`DIFFUSION_JOINT_OVERFIT.md`) + `eval_offline` on JOINT checkpoints | Yes (eval only) | After P2 |
| P4 | EEF smoke → EEF full run (100k) | Yes | After P3's eval_offline finishes (GPU free) |
| P5 | EEF report (`DIFFUSION_EE_OVERFIT.md`) + `eval_offline` (EE units) | Yes (eval only) | After P4 |

### GPU gate (mandatory before ANY CUDA work)

The ACT val-diagnostic is running (as of 2026-07-08 23:09 it was at step 14,000/80,000,
~3.1 steps/s effective, GPU 7.4/12.3 GB → ETA roughly 05:00–06:00 on 2026-07-09).
Poll; only proceed when **all three** hold:

```bash
# 1) no lerobot-train process alive
! pgrep -af 'lerobot-train' 
# 2) GPU essentially free
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # must be < 500 MiB
# 3) ACT log confirms completion (reached step 80000 / "End of training", or process exited)
tail -c 4000 "$SC/act_valdiag.log" | grep -aE 'step:80K|step 80000|End of'
```

Poll every 5–10 min (background loop or Monitor). Do NOT start smoke runs "to be ready" —
the 3060 cannot fit both jobs. Note `train_act_resume_80k.sh` exists in the repo; if any
*other* ACT job is queued by another team, coordinate — this plan assumes the GPU is ours
once the val-diag exits.

---

## 2. JOINT diffusion run

### 2.1 Resolved decisions (with rationale)

| Decision | Value | Rationale (1 line) |
|---|---|---|
| `steps` | **100,000** | AGENT_GUIDE §7.3 says diffusion needs 80k–150k (converges slower than ACT); the diagnostic's job is to *find* the val minimum, so the longer tail is informative — and ACT comparability is preserved by simply reading the curve at step 80k. ≈45 epochs over the 45 train eps (17.6k usable samples / batch 8 ≈ 2.2k steps/epoch). |
| `batch_size` | **8** | Apples-to-apples with ACT val-diag (batch 8); AGENT_GUIDE profiling (batch 4 → 4.94 GB SGD) scaled to batch 8 + AdamW + 2×360×640 cams + n_obs=2 estimates ~9–11 GB, inside 12 GB. |
| OOM fallback ladder | 8(fp32) → 8+`--policy.use_amp=true` → 6+amp → 4+amp | Try AMP before shrinking batch (keeps gradient quality); each rung verified by the 300-step smoke, not mid-run. |
| `use_amp` initially | **false** | Exactness first: AMP adds loss-scale noise to an already-stochastic diffusion eval curve; enable only as OOM lever. |
| `n_obs_steps / horizon / n_action_steps` | **2 / 64 / 32** (defaults) | Standard Diffusion Policy settings; min episode length 231 ≫ 64 so windows fit; n_action_steps=32 @30 fps ≈ 1.07 s replan interval, comparable to the ACT chunk regime. |
| `drop_n_last_frames` | **31** (must pass explicitly) | Hardcoded default 7 is only correct for horizon 16; correct formula `horizon − n_action_steps − n_obs_steps + 1 = 64−32−2+1 = 31` (see comment at configuration_diffusion.py:116). |
| Crop augmentation | **OFF** (leave `resize_shape`/`crop_ratio` unset) | The diagnostic must be apples-to-apples with ACT, which trained with Resize-only; adding crop regularization would confound the overfit comparison. If the verdict is "OVERFIT early", crop-aug (`--policy.resize_shape='[360,640]' --policy.crop_ratio=0.9`) is the first follow-up lever — as a *new* run, not a mid-course change. |
| `image_transforms` | Resize 360×640 (identical to ACT) | Same input pipeline as ACT + matches `eval_offline.py`'s deterministic Resize. |
| `eval_split / eval_steps / save_freq / log_freq / seed / num_workers` | 0.117 / 2000 / 10000 / 200 / 1000 / 4 | Mirror ACT val-diag exactly; 0.117 holds out eps 45–50; 2000-step evals give 50 val points; keep `make_overfit_report.py` constants consistent. |
| Optimizer/scheduler/noise | lerobot diffusion presets (Adam 1e-4, cosine + 500 warmup, DDPM 100 train timesteps, epsilon) | Defaults are the published Diffusion Policy recipe; nothing in the brief argues for deviation. |
| `output_dir / job_name` | `outputs/train/diffusion_joint_val_diag` / `diffusion_joint_val_diag` | Never touches ACT dirs; joint vs EE separated. |

### 2.2 `train_diffusion_joint_valdiag.sh` (exact contents)

```bash
#!/usr/bin/env bash
# OVERFIT DIAGNOSTIC run for Diffusion Policy (JOINT action space) on banana_in_pot.
# Mirrors train_act_valdiag.sh: holds out the LAST ceil(51*0.117)=6 episodes (45..50),
# trains on the other 45, logs held-out eval loss every --eval_steps.
# NOTE: diffusion held-out loss is STOCHASTIC (random noise+timestep per forward);
# interpret via smoothed trend, not single points.
set -euo pipefail
cd "/home/theo_lab/Downloads/Put_right_banana_in_the_pot-20260707T103848Z-3-001"

SC=/tmp/claude-1002/-home-theo-lab-Downloads-Put-right-banana-in-the-pot-20260707T103848Z-3-001/cbc6765d-7359-4632-9edc-c7f3f13db195/scratchpad
export TORCH_HOME="$SC/torch_home"
export HF_LEROBOT_HOME="$SC/hf_lerobot_home"
export HF_HUB_OFFLINE=1

exec ./lr_env/bin/lerobot-train \
  --dataset.repo_id=theo/banana_in_pot \
  --dataset.root=./banana_in_pot_lerobot \
  --dataset.eval_split=0.117 \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.drop_n_last_frames=31 \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=1 \
  --dataset.image_transforms.tfs='{"resize":{"weight":1.0,"type":"Resize","kwargs":{"size":[360,640]}}}' \
  --batch_size=8 \
  --steps=100000 \
  --save_freq=10000 \
  --eval_steps=2000 \
  --log_freq=200 \
  --num_workers=4 \
  --seed=1000 \
  --wandb.enable=false \
  --job_name=diffusion_joint_val_diag \
  --output_dir=outputs/train/diffusion_joint_val_diag
```

(Defaults intentionally *not* overridden, therefore in effect: `n_obs_steps=2`,
`horizon=64`, `n_action_steps=32`, DDPM/100, epsilon, MIN_MAX action/state norm,
separate resnet18 per camera, no crop, `use_amp=false`.)

### 2.3 Smoke run (mandatory, ~3 min GPU) before the full launch

Run once the GPU gate passes. Same script with three overrides; **separate output dir; delete it afterwards** (never resume the full run from smoke output):

```bash
cd "$PROJECT" && SC=... ./lr_env/bin/lerobot-train  <same flags>  \
  --steps=300 --eval_steps=100 --save_freq=300 \
  --job_name=diffusion_joint_smoke --output_dir=outputs/train/diffusion_joint_smoke \
  2>&1 | tee "$SC/diffusion_joint_smoke.log"
```

Pass criteria (reviewer checks, then `rm -rf outputs/train/diffusion_joint_smoke`):
- no OOM; `nvidia-smi` peak ≤ 11.5 GB (else move down the fallback ladder and re-smoke);
- at least 2 `eval_loss=` lines appear and are finite;
- measured throughput ≥ 2.0 steps/s (if 1.0–2.0: keep batch 8 but cut `--steps` to 80000; if < 1.0: enable AMP and re-smoke).

### 2.4 Detached full launch (robust background-job pattern)

```bash
cd "$PROJECT"
chmod +x train_diffusion_joint_valdiag.sh
setsid nohup ./train_diffusion_joint_valdiag.sh \
  > "$SC/diffusion_joint_valdiag.log" 2>&1 &
echo $! > "$SC/diffusion_joint_valdiag.pid"
```

Heartbeat: `tail -c 2000 "$SC/diffusion_joint_valdiag.log"` — train lines every 200 steps
(`step:… loss:…`), eval lines every 2000 (`step N: eval_loss=…`); both regex-compatible with
`make_overfit_report.py` (verified against lerobot_train.py:606/637 formats in the ACT log).

---

## 3. EEF dataset build (new script: `convert_to_lerobot_ee_action.py`)

**No usable EE-action dataset exists** (`banana_in_pot_ee` added EE *observations* only;
`hilserl/banana_rl_lerobot` action is 4-dim position-only delta — too lossy: the task needs wrist
reorientation). FK is **NOT** needed: recorded `tcp_pose` is ground truth (validated 0.85 mm).

Start from a copy of `convert_to_lerobot_ee.py` (same resampling/NaN/video logic — keep
`nearest_idx`, `ffill_bfill`, the cam1-master-clock alignment, and the episode loop untouched).

### 3.1 Feature spec (exact)

| Feature | dtype/shape | Contents | `names` |
|---|---|---|---|
| `action` | float32 (10,) | `[tcp_x,y,z @ k+1, r1..r6 @ k+1, grip_cmd @ k]` | `["x","y","z","r1","r2","r3","r4","r5","r6","grip"]` |
| `observation.state` | float32 (10,) | `[tcp_x,y,z @ k, r1..r6 @ k, grip_pos @ k]` | `["x","y","z","r1","r2","r3","r4","r5","r6","grip_pos"]` |
| `observation.images.cam1/cam2` | video (720,1280,3) | unchanged | unchanged |

Do **not** carry `observation.tcp_pose`/`observation.wrench` as extra features — keep the
feature set minimal so the policy's auto-derived input features are unambiguous
(state + 2 cams only, exactly like the joint dataset).

Resolved choices:
- **Action = absolute EE pose** (not delta): deltas drift over a 64-step diffusion horizon; absolute matches the joint dataset's absolute-target convention.
- **Pose action target = achieved `tcp_pose` at frame k+1** (the recorded pose resampled onto the cam1 grid, shifted +1): no commanded EE pose exists in the logs; next achieved pose is the standard "next-state-as-action" conversion.
- **Gripper action = `grip_cmd` at frame k** (ffill/bfill'd, NOT shifted, NOT `grip_pos`): `grip_cmd` is the true actuation target and matches the joint dataset's gripper-action semantics; `grip_pos[k+1]` lags the command by many frames (slow gripper) and would teach a delayed ramp. *(This overrides the brief's `grip_pos` suggestion — state carries `grip_pos`, action carries `grip_cmd`, exactly as in the joint dataset.)*
- **Episode boundary: REPEAT** — `action[N-1] pose = tcp_pose[N-1]` (a hold). Keeps frame count identical to the joint dataset (21,524) so episode lengths, eval_split arithmetic, and cross-run comparisons stay aligned; dropping the last frame buys nothing.
- **6D rotation (Zhou et al.)**: quaternion (scalar-last `qx,qy,qz,qw`) → rotation matrix → **first two COLUMNS**, flattened as `[R00,R10,R20, R01,R11,R21]`. Note q and −q give the same R, so quaternion sign flips in the log cannot cause discontinuities — no hemisphere fixing needed.
- **Pure-numpy quat→matrix** (scipy is NOT installed in `lr_env`; do not install anything while ACT trains). Normalize the quaternion first, then:

```python
def quat_to_rot6d(q):            # q = [qx, qy, qz, qw] scalar-last, shape (...,4)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.stack([
        1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w),
        2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w),
        2*(x*z - y*w),      2*(y*z + x*w),      1 - 2*(x*x + y*y),
    ], axis=-1).reshape(*q.shape[:-1], 3, 3)
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)  # (...,6) = col0,col1
```

- **Normalization:** keep lerobot diffusion defaults (ACTION/STATE = MIN_MAX, per-dim — stats are computed per dimension automatically, which is exactly the "per-block scaling" the brief asks for: xyz (m), 6D (±1), grip each normalize independently to [−1,1], so position cannot dominate the loss). `clip_sample_range=1.0` is consistent with MIN_MAX. **Risk check in validation:** if any action dim has range (max−min) < 1e-3 (a near-constant 6D component would explode under MIN_MAX), flag it; fallback is `--policy.normalization_mapping` override to MEAN_STD for ACTION+STATE — do not apply preemptively.

### 3.2 Build command (CPU-only — safe to run while ACT is still training)

```bash
cd "$PROJECT"
./lr_env/bin/python convert_to_lerobot_ee_action.py \
  --data Put_right_banana_in_the_pot \
  --out  ./banana_in_pot_ee_action_lerobot \
  --repo-id theo/banana_in_pot_ee_action \
  2>&1 | tee "$SC/convert_ee_action.log"
```

New dataset dir `./banana_in_pot_ee_action_lerobot`, repo_id `theo/banana_in_pot_ee_action`
(never overwrite `banana_in_pot_lerobot`, `hf_upload/banana_in_pot_ee`, or the RL 4-dim set).
Expect ~30–60 min (AV1 re-encode of both cams dominates, same as the joint build).

### 3.3 Validation gate (script `validate_ee_dataset.py`; MUST pass before EEF training)

All checks against the built dataset (open with `LeRobotDataset(repo_id, root=…)`):
1. **Counts:** 51 episodes, 21,524 total frames, fps 30, min episode length ≥ 231 → `eval_split=0.117` still holds out exactly eps 45–50.
2. **Shapes/dtypes:** `action` (10,) float32, `observation.state` (10,) float32; `names` present as specced.
3. **No NaN/Inf** in action or state across all frames.
4. **6D round-trip:** for ≥1000 random frames, Gram-Schmidt the 6D back to R (`b1=norm(a1); b2=norm(a2−(b1·a2)b1); b3=b1×b2`), compare to the quat-derived R: max abs elementwise error < 1e-5; `det(R)=+1±1e-5`.
5. **Shift correctness:** for random (ep, k<N−1): `action[k][0:9] == state-side pose computed at k+1` (bit-exact, same source array); `action[N-1][0:9] == pose at N-1`.
6. **Continuity:** max per-frame ‖Δxyz‖ < 0.05 m within an episode (catches resample/shift bugs).
7. **Stats populated:** meta stats exist for action/state, per-dim; print per-dim min/max; **flag any action dim with range < 1e-3** (see §3.1 fallback).
8. **Gripper sanity:** action grip effectively binary (grip_cmd), state grip_pos ∈ [0, ~0.9].

Print a PASS/FAIL summary; the reviewer signs off on the log before P4.

---

## 4. EEF diffusion run

`train_diffusion_ee_valdiag.sh` = the JOINT script (§2.2) with **only these deltas**:

```
  --dataset.repo_id=theo/banana_in_pot_ee_action \
  --dataset.root=./banana_in_pot_ee_action_lerobot \
  --job_name=diffusion_ee_val_diag \
  --output_dir=outputs/train/diffusion_ee_val_diag
```

Log: `"$SC/diffusion_ee_valdiag.log"`. Same smoke protocol (§2.3, output dir
`diffusion_ee_smoke`), same detached launch (§2.4), same 100k/batch-8/no-crop/
`drop_n_last_frames=31` settings.

Config deltas that happen **automatically** (no CLI needed): action dim 10 and state dim
**10** (= xyz 3 + 6D 6 + grip 1 — note: 10, not 9) are derived from dataset features;
per-dim MIN_MAX stats come from the new dataset. VRAM/time ≈ identical to JOINT (dims 7→10
are negligible next to the vision encoders).

Only genuine new risk: MIN_MAX blow-up on a near-constant 6D dim — resolved by validation
check §3.3(7) *before* training.

---

## 5. Diagnostic & report tooling

### 5.1 `eval_offline.py` patch (owner: eval/report agent)

Minimal generalization, keeping ACT behavior as default:
1. **Policy class (the core one-liner):** replace the hardcoded `ACTPolicy` with
   `from lerobot.policies.factory import get_policy_class` (exists at factory.py:87) and
   `policy_cls = get_policy_class(cfg.type)`; `policy = policy_cls.from_pretrained(checkpoint, config=cfg)`.
2. **Fast diffusion sampling:** add `--scheduler {asis,DDIM}` (default asis) and
   `--num-inference-steps N` (default: as trained). Implement by setting
   `cfg.noise_scheduler_type="DDIM"` / `cfg.num_inference_steps=10` on the loaded
   `PreTrainedConfig` *before* `from_pretrained(config=cfg)` — the noise scheduler is built
   from config at model init, and DDIM is a valid sampler for a DDPM-trained epsilon model
   (same beta schedule). For diffusion evals always pass `--scheduler DDIM --num-inference-steps 10`
   (~10× rollout speedup; select_action still manages the obs-history + 32-step action queue internally — caller loop unchanged).
3. **Dim-agnostic metrics/plots:** derive `DIM_NAMES` from `ds.meta.features["action"]["names"]`
   (fallback to the current 7-dim list); treat the **last dim as gripper** (threshold 0.5 stays
   valid — grip action is binary-ish grip_cmd in both datasets) and all preceding dims as the
   "pose/joints" block for MAE/RMSE; subplot count = action dim. For the EE run report xyz MAE
   in meters and 6D MAE unitless (units column: `m`×3, `6d`×6, `grip`).
4. **Smoke gating:** `--smoke` builds its policy via `make_policy_config(args.policy_type)`
   with a new `--policy-type {act,diffusion}` (diffusion smoke overrides: `horizon=16`,
   `n_action_steps=8`, `num_inference_steps=2` for CPU speed; note horizon 16 keeps the
   default `drop_n_last_frames=7` consistent). CPU smoke of the diffusion path is part of P1 (no GPU).

Eval commands (after each run; GPU, a few minutes each):
```bash
./lr_env/bin/python eval_offline.py \
  --checkpoint outputs/train/diffusion_joint_val_diag/checkpoints/<STEP>/pretrained_model \
  --episodes 45,46,47,48,49,50 --device cuda \
  --scheduler DDIM --num-inference-steps 10 \
  --out eval_out_diffusion_joint_<STEP>
# EE run: same, with the ee checkpoint, --root ./banana_in_pot_ee_action_lerobot \
#   --repo-id theo/banana_in_pot_ee_action --out eval_out_diffusion_ee_<STEP>
```
Evaluate **two** checkpoints per run: the final (100k) and the saved checkpoint nearest the
val-loss minimum (save_freq 10k → round `s_min` to the nearest 10k).

### 5.2 `make_overfit_report.py` parameterization (owner: eval/report agent)

No parsing-logic change (regexes already match lerobot_train.py:606/637 lines). Add argparse
(defaults = current ACT behavior, so the ACT flow is untouched):
`--log`, `--md-out`, `--png-out`, `--title`, `--log-freq` (200), `--eval-freq` (2000),
`--total-steps` (100000 for these runs), and `--smooth N` (default 1 = off).

**Smoothing (required for diffusion):** with `--smooth 3`, plot the raw val points faintly
plus a centered rolling-mean(3) overlay, and compute the verdict (`v_min`, `s_min`, ratio) on
the **smoothed** series — diffusion eval loss re-noises every forward, so single points are
noisy by construction. Add one caveat bullet to the generated MD explaining this.

Report invocations (safe mid-training; rerun any time):
```bash
./lr_env/bin/python make_overfit_report.py --log "$SC/diffusion_joint_valdiag.log" \
  --md-out DIFFUSION_JOINT_OVERFIT.md --png-out report_assets/diffusion_joint_overfit_diag.png \
  --title "Diffusion (joint) overfit diagnostic (45 train / 6 held-out eps)" \
  --total-steps 100000 --smooth 3
./lr_env/bin/python make_overfit_report.py --log "$SC/diffusion_ee_valdiag.log" \
  --md-out DIFFUSION_EE_OVERFIT.md --png-out report_assets/diffusion_ee_overfit_diag.png \
  --title "Diffusion (EE 10-dim) overfit diagnostic (45 train / 6 held-out eps)" \
  --total-steps 100000 --smooth 3
```

**Deliverable reports:** `DIFFUSION_JOINT_OVERFIT.md` and `DIFFUSION_EE_OVERFIT.md` in the
project root, same graph-embedded style as `ACT_OVERFIT_DIAGNOSIS.md` (embedded PNG, per-eval
table, summary numbers, auto verdict, caveats), each appended with the `eval_offline`
held-out MAE table for the two evaluated checkpoints and a short comparison note vs
`ACT_OVERFIT_DIAGNOSIS.md` at step 80k.

---

## 6. Resource & time estimates (RTX 3060 12 GB)

| Item | Estimate | Basis |
|---|---|---|
| VRAM, diffusion batch 8, fp32, 2 cams 360×640, n_obs 2, AdamW | ~9–11 GB (smoke-verified) | AGENT_GUIDE: batch 4 → 4.94 GB SGD; AdamW + 2 cams adds; ACT batch 8 measured 7.4 GB on this GPU |
| Throughput | ~2–3 steps/s effective (train ~2.5–4, minus eval pauses every 2k) | Diffusion update ≈ 2× ACT (168.6 vs 83.9 ms @ batch 4); ACT val-diag measured 3.1 steps/s effective |
| JOINT run, 100k steps | **~9–14 h** | above |
| EEF run, 100k steps | **~9–14 h** | same footprint (action dim negligible) |
| EEF dataset build | ~30–60 min **CPU, runs during ACT/JOINT training** | joint conversion of same 51 eps / AV1 encode |
| eval_offline per checkpoint (6 eps ≈ 2.5k frames, DDIM-10) | ~5–15 min | UNet sampled once per 32 frames + per-frame resnet |
| Smokes + reports + validation | < 1 h total | — |
| **Total GPU wall-clock** | **~20–30 h** after ACT finishes (~05:30 on 07-09 → EEF report done roughly 07-10 morning) | serialized P2→P5 |

If measured smoke throughput < 2 steps/s at batch 8 fp32: cut both runs to `--steps=80000`
(still ≥ the AGENT_GUIDE floor, saves ~2–3 h each) and/or enable AMP — decide at the smoke
gate, identically for both runs so they stay comparable to each other.

---

## 7. Review / verification checkpoints (gates — do not burn compute past a failed gate)

| Gate | When | What must be verified (by the reviewer agent) |
|---|---|---|
| **V1 script review** | end of P1 | Both train scripts diff-reviewed against §2.2/§4 (flag-for-flag vs `train_act_valdiag.sh`; `drop_n_last_frames=31` present; correct dirs/logs). Converter + validator code-reviewed (shift indexing, boundary repeat, quat formula col-major order, grip_cmd vs grip_pos assignment). eval_offline/report patches reviewed; `eval_offline.py --smoke --policy-type diffusion --device cpu --episodes 0 --max-frames 8` passes on CPU. |
| **V2 dataset gate** | after §3.2 build | `validate_ee_dataset.py` all 8 checks PASS (log saved to `$SC/validate_ee_action.log`). Any MIN_MAX range flag resolved before V4-EE. |
| **V3 GPU gate** | before P2 and before P4 | §1 three-condition check passes; also confirm no other queued GPU job. |
| **V4 smoke gate** (×2) | before each full launch | §2.3 criteria: no OOM, ≤11.5 GB, eval lines present & finite, throughput decision applied; smoke output dir deleted. |
| **V5 early-curve sanity** (×2) | after first 3 eval points (~step 6k, ≈45 min in) | eval_loss finite, same order of magnitude as train loss, downward trend after smoothing; train loss decreasing; step rate as smoked. If val is flat at noise level or NaN → kill the run, investigate before re-launch. |
| **V6 report gate** (×2) | end of each run | `make_overfit_report.py` verdict computed on smoothed series; eval_offline run on final + val-min checkpoints; report MD committed with both. |

---

## 8. File ownership / task division (Opus dev/review/train team)

Three workers + one reviewer; zero file overlap, so P1 work is fully parallel.

| Agent | Owns (exclusive) | Tasks |
|---|---|---|
| **D1 dataset-builder** | `convert_to_lerobot_ee_action.py`, `validate_ee_dataset.py`, `banana_in_pot_ee_action_lerobot/`, `$SC/convert_ee_action.log`, `$SC/validate_ee_action.log` | §3 build + validation. **CPU-only — start immediately**, even while ACT trains. |
| **D2 train-runner** | `train_diffusion_joint_valdiag.sh`, `train_diffusion_ee_valdiag.sh`, `outputs/train/diffusion_*`, `$SC/diffusion_*_valdiag.log`, `$SC/diffusion_*_smoke.log`, PID files | Write scripts in P1; own the GPU gate (§1), smokes (§2.3), detached launches (§2.4), heartbeat monitoring, throughput/VRAM logging, OOM-ladder decisions. |
| **D3 eval-report** | `eval_offline.py`, `make_overfit_report.py`, `report_assets/diffusion_*`, `DIFFUSION_JOINT_OVERFIT.md`, `DIFFUSION_EE_OVERFIT.md`, `eval_out_diffusion_*` | §5 patches in P1 (incl. CPU smoke of the diffusion eval path); mid-run report refreshes; post-run eval_offline (GPU — coordinate with D2 so it never overlaps a training run) and final reports. |
| **R reviewer** (code-reviewer agent) | (read-only) | Gates V1–V6. Special attention: D1's +1-shift & 6D column order; D2's flag parity with ACT script; D3's normalization/unnormalization path for 10-dim actions and DDIM-at-eval config mutation. |

Sequencing summary: P1 (D1 ∥ D2 ∥ D3, all CPU, during ACT) → V1/V2 → V3 → D2 JOINT smoke
(V4) → JOINT 100k → D3 joint report + eval_offline (V5, V6) → V3 again → D2 EEF smoke (V4) →
EEF 100k → D3 EE report + eval_offline (V5, V6). GPU is used by exactly one agent at a time,
always D2's training or D3's eval, never concurrently.
