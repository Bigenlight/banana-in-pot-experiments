# EXPERIMENT_LOG.md

_A dated, narrative timeline of the "put the right banana in the pot" imitation-learning experiment (UR7e + GELLO teleop). It stitches the per-topic docs in `docs/` into one story for a newcomer: how the dataset was built, how ACT and Diffusion Policy were trained and diagnosed, and the one finding that changed how we pick checkpoints. Numbers here are copied from the source docs — where a doc reports a value two ways, both are shown. See each `docs/*.md` for the full detail behind a given entry._

---

## 2026-07-07 — Dataset conversion: raw teleop → LeRobot v3.0 (JOINT)

Raw teleoperation logs (per-episode HDF5 + two MP4 camera streams, UR7e arm driven by a GELLO leader) were converted to a **LeRobot v3.0** dataset (`convert_to_lerobot.py`, lerobot 0.6.1). Result: **51 episodes · 21,524 frames · 30 fps · ~11.96 min · 483 MB**, repo_id `theo/banana_in_pot`, local dir `banana_in_pot_lerobot/`. Schema is `observation.state` (7,) = UR joints q1..q6 + `grip_pos`, `action` (7,) = absolute joint targets cmd1..cmd6 + `grip_cmd`, plus two AV1-encoded 720×1280 camera videos. The GELLO leader streams were deliberately excluded (not observable at inference). Because the source is multi-rate (cameras 30 fps, robot ~56 Hz), everything was resampled onto the cam1 timestamp grid by nearest-timestamp match.

**Why / what was found:** the conversion was gated behind six independent Opus validators (schema, episode coverage, video integrity, alignment, value sanity, training-readiness), all PASS. Frame totals reconcile four ways (raw = data = meta = info = 21,524); a real `lerobot-train` ran two ACT steps at native 720p (loss 91.6 → 65.2) confirming trainability. **Surprises / caveats:** ~7 takes have raw robot-stream dropouts (75–280 ms) producing up to ~40 ms cam↔robot match error at those frames (mostly harmless for ACT); one take's 333-NaN `grip_cmd` was repaired; takes #9/23/30/35 are absent by design (re-takes fill the count back to 51). See `docs/DATASET_REPORT.md`.

## 2026-07-08 — Two EE-space datasets (observation, then 10-dim action)

Two end-effector variants were built later the same day. First an **EE-observation** dataset (`convert_to_lerobot_ee.py`, afternoon), then the **10-dim EE-ACTION** dataset (`convert_to_lerobot_ee_action.py`, late night) needed for the EEF diffusion leg. The EE-action set uses `action` (10,) = `[tcp_x,y,z @ k+1, r1..r6 @ k+1, grip_cmd @ k]` and a matching 10-dim `observation.state` (with `grip_pos`); rotation is encoded as **6D (Zhou et al.) from the recorded TCP quaternion** (scalar-last → rotation matrix → first two columns, pure-numpy). The action is the **absolute** achieved next-pose (deltas would drift over a 64-step diffusion horizon); the gripper action is `grip_cmd` at frame k (not the lagging `grip_pos`), matching the joint dataset's semantics.

**Why / what was found:** it was built by **video-reuse** — the AV1 camera videos were copied straight from the JOINT dataset and only the parquet + stats were regenerated, so no re-encode was needed (a few CPU-minutes rather than the ~30–60 min a full build takes). It keeps the same 51 episodes / 21,524 frames so `eval_split=0.117` still holds out exactly eps 45–50 and cross-run comparisons stay aligned. This dataset is **local-only** (484 MB, not on HF). See `docs/DIFFUSION_PLAN.md` §3.

## 2026-07-07/08 — ACT training: 50k, then resumed to 80k

An ACT policy (ResNet18 backbone, VAE, `chunk_size=100`) was trained on the JOINT dataset: `observation.state` (7) + two cameras resized on-the-fly to 360×640 (no re-encode), `action` (7) absolute joint targets. Config: batch 8, AdamW lr 1e-5 constant, ImageNet-pretrained backbone, RTX 3060 12 GB (~4.7 GB used, ~3.9 step/s). The first run went to **50,000 steps** (documented loss 0.065; best-checkpoint train L1 0.0205); it was then **resumed toward 80,000 steps** (`train_act_resume_80k.sh`) for the deploy model, reaching a final loss of ~0.052.

**Why / what was found:** offline open-loop eval of the 50k model on episodes 0,1,2,45–50 looked excellent — joints MAE ~0.0237 rad (≈1.36°), gripper accuracy 99.2%, overall L1 0.0235. **The surprise came from re-reading the setup:** this model was trained on *all 51 episodes*, so the "held-out" episodes 45–50 in that eval table had actually been in the training set — there was **no true train/val split**. That realization triggered a dedicated overfit diagnostic. See `docs/ACT_RESULTS.md`.

## 2026-07-09 — ACT overfit diagnosis: no destructive overfit

To get an honest generalization read, a separate ACT run trained on **45 episodes with eps 45–50 genuinely held out** (`--dataset.eval_split=0.117`), logging train loss every 200 steps and held-out `eval_loss` every 2000. Held-out `eval_loss` **fell** from 0.6505 (2k) to its minimum **0.5252 (44k)** and never turned back up — the marginal gain just flattened after ~30k. The clean open-loop held-out MAE (`eval_offline.py`, teacher-forced over eps 45–50) agreed: poseMAE 0.1055 (10k) → **0.0978 (30k, best ≈0.098 rad)** → 0.1000 (40k), gripper accuracy ~0.92–0.94.

**Why / what was found:** both signals — the val curve and the open-loop MAE — pointed the same direction and put the sweet spot at ~30k. **Verdict: NO destructive overfitting** for this recipe; the deploy model (all 51 eps, 80k) is not at risk of the "val rises while train falls" failure mode. The diagnostic was intentionally early-stopped at ~45k of a planned 80k once the answer was clear, and the GPU was handed to the diffusion experiments. Honest caveats noted in the doc: eps 45–50 are the chronological tail (not an i.i.d. split), and this characterizes the *recipe's* tendency via an independent 45-ep model, not the exact 80k deploy weights. See `docs/ACT_OVERFIT_DIAGNOSIS.md`.

## 2026-07-08 — Diffusion Policy plan (JOINT → EEF)

A plan (fable5, 2026-07-08) laid out reproducing the ACT overfit methodology for **Diffusion Policy**, run twice sequentially: **JOINT** first (existing 7-dim dataset), then **EEF** (the new 10-dim EE-action dataset). Same val-split methodology as ACT: `eval_split=0.117` (hold out eps 45–50), held-out eval every 2000 steps, `make_overfit_report.py` curves + `eval_offline.py` open-loop MAE.

Key resolved config decisions (chosen for ACT-comparability and correctness): **batch 8**, **100k steps** (diffusion converges slower; read the curve at 80k for the ACT comparison), `n_obs_steps/horizon/n_action_steps = 2/64/32`, and two non-obvious flags — **`drop_n_last_frames=31`** (the hardcoded default 7 is only right for horizon 16; correct value is 64−32−2+1=31) and **Resize 360×640** input with **crop-augmentation OFF** (ACT trained Resize-only; adding crop regularization would confound the overfit comparison). AMP left off initially (exactness first). See `docs/DIFFUSION_PLAN.md`.

## 2026-07-09 — Diffusion JOINT run + the headline discovery: the two overfit signals DIVERGE

This is the most instructive entry, and it involved a real course-correction — recorded honestly.

The JOINT diffusion run's held-out **denoising `eval_loss` ROSE** from its minimum ~0.0289 (4k) to ~0.0673 (44k) — a textbook-looking overfit curve (`val_final/val_min` ≈ 2.2, auto-verdict "OVERFIT from ~step 6000"). **On that misleading signal the run was early-stopped at ~45k.** Then `eval_offline.py` open-loop rollout MAE (DDIM-10, held-out eps 45–50) was computed on the saved checkpoints and told the **opposite** story — it improved **monotonically**:

| checkpoint | poseMAE (rad) | gripAcc | overall L1 |
|---|---|---|---|
| 10k | 0.1193 | 0.729 | 0.1454 |
| 20k | 0.1037 | 0.888 | 0.1078 |
| 30k | 0.0921 | 0.919 | 0.0928 |
| **40k** | **0.0907** | **0.949** | **0.0862** |

**What this means:** for a diffusion policy the held-out denoising MSE is **not** a reliable early-stop/overfit signal — it scores noise prediction at random timesteps, not sampled-action accuracy, so it can rise while the actual generated-action quality keeps improving. The **open-loop rollout MAE is the signal that matters**, and it says the model was **not** harmfully overfitting. This is the opposite of ACT, where the two signals agreed. Because the open-loop trend had **not yet plateaued** at the early-stop point, the run was **resumed (≈40k → 80k)** rather than abandoned; the best available JOINT-diffusion checkpoint by open-loop MAE is **40k** (poseMAE 0.091, gripAcc 0.95). Standing lesson: **select diffusion checkpoints by open-loop MAE, never by `eval_loss`.** See `docs/DIFFUSION_JOINT_OVERFIT.md`.

## 2026-07-08 — Deploy repo decision (real UR7e)

In parallel, the real-robot deploy architecture was decided (2026-07-08): **extend `gello_software` with a new ROS2 package `gello_policy/`** rather than fork LeRobot or start fresh. The policy acts as a *synthetic GELLO leader* (Option-Bridge) — it publishes ACT's actions to `/gello/joint_states` at 30 Hz and reuses the existing, safety-tuned `gello_ur_bridge` unmodified (250 Hz upsampling, One-Euro smoothing, slew clamp, staleness watchdog, move-to-start handshake). Because Humble's rclpy is Python 3.10 and lerobot needs ≥3.12, the two run as **two processes joined by a local ZMQ REQ/REP split** (also the future HIL-SERL actor/learner boundary; also crash-isolates CUDA). Action chunk consumed receding-horizon: re-query every 30 steps (1.0 s). Sharp edges captured there: the bridge bounds *velocity* not *position* (position clamps must live in the policy node), and the saved preprocessor has **no resize** — deploy must replicate `imdecode → BGR→RGB → Resize 360×640` verbatim or silently degrade. See `docs/DEPLOY_REPO_DECISION.md`.

## 2026-07-09 — Reproducibility packaging (this repo)

The experiment was packaged into this clone-and-rerun repo: scripts + docs only (~200 KB tracked), with **`gello_software` as a git submodule** (we own it, pinned @ `0730fb42`), **lerobot as a setup-script pinned clone** (@ `8a74e0a` = 0.6.1, gitignored), datasets/models **fetched from HF or rebuilt** (`fetch_data.sh`), and a committed **`requirements-lock.txt`** (uv venv, Python 3.12.3, 119 packages) driven by `setup.sh`. The JOINT dataset, raw data, EE-obs dataset, and ACT model are public on HF under `Bigenlight/*`; the 10-dim EE-action dataset and the val-diag checkpoints are local-only and rebuilt/re-trained. The one mandatory fix before cross-PC use: every train script and `eval_offline.py` hardcodes absolute `cd` and scratchpad paths that must be parameterized. See `REPRODUCIBILITY_PLAN.md`.

---

## Open threads

- **EEF diffusion run** — the 10-dim EE-action leg (`train_diffusion_ee_valdiag.sh`) is planned and scripted but not yet run/reported (`DIFFUSION_EE_OVERFIT.md` does not exist yet).
- **EE-action dataset not on HF** — `banana_in_pot_ee_action_lerobot` (484 MB) is local-only (HF 401); until a one-time push, the canonical route is rebuild-from-raw (needs the JOINT dataset + raw h5 present first, video-reuse).
- **Best-diffusion-checkpoint selection** — pick by **open-loop MAE** (`eval_offline.py`, DDIM-10), not `eval_loss`. JOINT best-so-far is 40k; the trend suggested more steps could still help, hence the resume toward 80k.
- **Deploy is a design, not yet executed** — `gello_policy` ROS2 package and `act_server.py` are specified in `DEPLOY_REPO_DECISION.md` but not yet built on the robot PC; closed-loop real-robot validation is the outstanding verdict for both ACT and diffusion.
