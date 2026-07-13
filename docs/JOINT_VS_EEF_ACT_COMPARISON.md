# JOINT vs EEF — ACT Comparison (scaffold)

**Status:** scaffold prepared while the EEF ACT run is training. Numbers get filled
in as (a) the EEF ACT run finishes + is eval'd here, and (b) the JOINT ACT numbers
are pulled in from the other PC. Do not treat empty cells as results.

Goal: decide whether training ACT on **end-effector (EEF) actions** helps or hurts
vs the **joint-space** baseline, under an otherwise *identical* recipe, on the same
held-out episodes — so the only variable is the action/observation representation.

---

## 1. Protocol — what is held identical (fair comparison)

Both runs use the **same** `lerobot-train --policy.type=act` recipe. The EEF script
(`train_act_ee_valdiag.sh`) is a byte-for-byte mirror of the JOINT script
(`train_act_valdiag.sh`) except for the three unavoidable differences:

| Knob | JOINT | EEF | same? |
|---|---|---|---|
| dataset | `theo/banana_in_pot` (7-D) | `theo/banana_in_pot_ee_action` (10-D) | ✗ (the variable under test) |
| `--steps` | 50000 | 50000 | ✓ |
| `job_name` / `output_dir` | `act_banana_val_diag` | `act_ee_val_diag` | ✗ (bookkeeping only) |
| `--dataset.eval_split` | 0.117 | 0.117 | ✓ |
| held-out episodes | eps **45–50** (last 6 of 51) | eps **45–50** | ✓ |
| `--batch_size` | 8 | 8 | ✓ |
| `--seed` | 1000 | 1000 | ✓ |
| image transforms | resize 360×640 | resize 360×640 | ✓ |
| `--save_freq` / `--eval_steps` | 10000 / 2000 | 10000 / 2000 | ✓ |
| policy | ACT (ResNet18+VAE, chunk 100) | ACT (identical) | ✓ |
| ACT normalization | MEAN_STD (state & action) | MEAN_STD (state & action) | ✓ |

Representation detail:
- **JOINT**: `observation.state`/`action` = 7-D `[q1..q6, gripper]` (absolute joint targets, rad).
- **EEF**: `observation.state`/`action` = 10-D `[x,y,z, r1..r6 (Zhou 6D rotation), gripper]`
  (absolute next-frame TCP pose; xyz in metres, gripper same channel as joint).

Both eval on the **same 6 held-out episodes (45–50)** via `eval_offline.py` open-loop
rollout — the same protocol the repo uses everywhere.

---

## 2. ⚠️ Metric comparability — read before comparing numbers

The two policies predict in **different spaces**, so their error metrics are **not in
the same units** and must not be compared as raw numbers:

- JOINT open-loop **pose MAE** is per-joint **radians**.
- EEF open-loop **pose MAE** mixes **metres** (x,y,z) and **unitless** 6D-rotation
  components (r1..r6). An "average MAE" over the 10 EEF dims is dimensionally
  meaningless and is **not** comparable to a radian MAE.
- The `eval_loss` values are also **not** directly comparable: ACT's loss is computed
  in each dataset's normalized action space, and the 7-D vs 10-D targets have
  different per-dim statistics, so equal loss ≠ equal quality.

**What IS directly comparable across the two runs:**
- **Gripper accuracy** — both runs carry the *same* gripper channel, so gripper-open/close
  accuracy is an apples-to-apples number.
- **Relative trends** within each run (does eval_loss/MAE keep improving; where is the
  sweet-spot checkpoint; is there overfitting) — comparable qualitatively, not in value.

---

## 3. Recommended fair-comparison methods (ranked)

To answer "which representation is better" *quantitatively*, pick one of these — they
put both policies on a common yardstick:

1. **Real-robot task success rate** (gold standard). Run each selected checkpoint on the
   UR7e for the banana-in-pot task, N trials, report success %. Deploy path differs:
   JOINT → `servoJ` (as in `deploy_ur_act.py`); EEF → would need `servoL`/IK. Ultimate
   metric but needs hardware time.
2. **Common-space open-loop error via FK** (offline, no hardware). Convert the JOINT
   policy's predicted joint actions → EEF pose via forward kinematics, then compare
   **EEF-space error (metres + rotation)** for *both* policies against the recorded
   `tcp_pose` ground truth. Needs the UR7e FK (the repo's FK lived under the gitignored
   `/hilserl/`; regenerate from `ur_description` `ur7e` if needed). This is the cleanest
   offline apples-to-apples.
3. **Gripper accuracy + per-run sweet-spot** (already available, weakest). Compare gripper
   accuracy directly and each run's best-checkpoint trend. Directional only.

Default recommendation: report **method 3 now** (free, from the evals we already run),
and set up **method 2** for a rigorous offline verdict; use **method 1** if/when the
robot is available.

> Checkpoint selection note (repo finding): pick the checkpoint by **open-loop MAE**
> from `eval_offline.py`, not by `eval_loss` — for diffusion the two diverge; for ACT
> they agree, but staying consistent keeps JOINT and EEF selection identical.

---

## 4. Results tables (fill as data arrives)

### JOINT reference — which numbers to use
The repo contains **two** JOINT ACT runs; pick the right baseline:
- **Run A — deploy model** (`docs/ACT_RESULTS.md`): trained on **all 51 episodes**, so its
  "held-out" eval is actually **in-sample** — 50k pose MAE **0.0237 rad**, gripper **99.2%**.
  *Not a fair generalization number; do not compare our held-out EEF to this.*
- **Run B — val-diagnostic** (`docs/ACT_OVERFIT_DIAGNOSIS.md`): genuine **45-train / 6-held-out**
  (eval_split=0.117, eps 45–50), seed 1000, batch 8 — **the same recipe as our EEF run**.
  Early-stopped ~45k (no 50k). This is the **apples-to-apples JOINT baseline** and is used
  in the tables below. eval_loss min **0.5252 @ 44k** (no overfit). Held-out open-loop MAE
  (radians): 10k=0.1055, 20k=0.1057, **30k=0.0978 (best)**, 40k=0.1000; gripAcc 0.918/0.938/0.937/0.937.

> The user is also running a **fresh JOINT ACT to 50k on another PC**. When those numbers
> arrive, put them in a "JOINT (fresh 50k)" column — they, not Run B, are the true
> like-for-like baseline (Run B stopped at 45k). Run B stays as the repo reference.

### Table A — held-out `eval_loss` per checkpoint (trend only; NOT cross-comparable in value)
| step | JOINT eval_loss | EEF eval_loss |
|---|---|---|
| 2000  | _<joint>_ | 0.6737 |
| 4000  | _<joint>_ | 0.6480 |
| 6000  | _<joint>_ | 0.6100 |
| 8000  | _<joint>_ | 0.5432 |
| 10000 | _<joint>_ | 0.5680 |
| 12000 | _<joint>_ | 0.5343 |
| 14000 | _<joint>_ | 0.5323 |
| 16000 | _<joint>_ | 0.5034 |
| 18000 | _<joint>_ | 0.5009 |
| 20000 | _<joint>_ | 0.5028 |
| 30000 | _<joint>_ | 0.4882 |
| 40000 | _<joint>_ | 0.4849 |
| 50000 | _<joint>_ | **0.4594** |

*(EEF values above are live from the current run's log. JOINT column = paste the fresh
other-PC 50k run's eval_loss. Repo Run B reference: eval_loss bottomed at **0.5252 @ 44k**,
no overfit — note our EEF is already ~0.50 by 18k, but eval_loss across 7-D vs 10-D action
spaces is NOT comparable in value, only in trend.)*

### Table B — open-loop rollout MAE per checkpoint (native units — compare WITHIN a column, not across)
JOINT column = Run B (valdiag, held-out eps 45–50, radians). ⚠️ JOINT rad vs EEF m/6D are **not** the same unit.
| checkpoint | JOINT pose MAE (rad) | JOINT grip acc | EEF pose MAE (m + 6D) | EEF grip acc |
|---|---|---|---|---|
| 10000 | 0.1055 | 0.918 | 0.06338 | 0.913 |
| 20000 | 0.1057 | 0.938 | 0.05939 | 0.906 |
| 30000 | **0.0978** | 0.937 | 0.05775 | 0.911 |
| 40000 | 0.1000 | 0.937 | **0.05564** | **0.914** |
| 50000 | _n/a (Run B stopped ~45k)_ | _n/a_ | 0.05564 | 0.911 |

### Table C — directly comparable: gripper accuracy per checkpoint
JOINT = Run B (valdiag). This is the one column pair you CAN compare in value.
| checkpoint | JOINT grip acc | EEF grip acc | Δ (EEF−JOINT) |
|---|---|---|---|
| 10000 | 0.918 | 0.913 | −0.005 |
| 20000 | 0.938 | 0.906 | −0.032 |
| 30000 | 0.937 | 0.911 | −0.026 |
| 40000 | 0.937 | 0.914 | −0.023 |
| 50000 | _n/a_ | 0.911 | _n/a_ |

---

## 5. Checklist — what to collect from the JOINT (other-PC) run

To fill the JOINT column, gather from the other PC (same repo/scripts):
- [ ] `eval_loss` at each logged step (from that run's training log) → Table A.
- [ ] `eval_offline.py` output per checkpoint (10k–50k) on eps 45–50 → Table B/C
      (pose MAE + gripper accuracy). Use the **same command/flags** as the EEF eval.
- [ ] The exact config it ran (confirm it was `train_act_valdiag.sh` @ 50k, seed 1000,
      batch 8, eval_split 0.117 — else note any difference; it changes fairness).
- [ ] Its train-loss/eval-loss curve PNG, if you want side-by-side plots.
- [ ] (For method 2) confirm access to UR7e FK to convert joint preds → EEF space.

Drop those numbers into the tables above (or hand them to me and I'll fill + regenerate
the side-by-side plots next to `results/report_assets/`).

---

## 6. Provenance & exact eval command
- EEF run: `train_act_ee_valdiag.sh`, container `banana-train-ee` (Docker, GPU 3),
  output `outputs/train/act_ee_val_diag/`, eval → `results/eval_ee_act_<step>/`.
- EEF dataset: `banana_in_pot_ee_action_lerobot` (10-D, validated 8/8: 51 eps / 21524
  frames, held-out eps 45–50).
- JOINT run: repo Run A (`docs/ACT_RESULTS.md`, in-sample), Run B (`docs/ACT_OVERFIT_DIAGNOSIS.md`,
  held-out valdiag) + a fresh 50k run on the user's other PC (to import).

**EEF eval — exact command** (two gotchas: `eval_offline.py` defaults `--root`/`--repo-id`
to the JOINT dataset, and it prints metrics to **stdout only** — no json/csv is written, so
capture it):
```
docker run --rm --gpus '"device=3"' --shm-size=16g --user 1001:1001 -e HOME=/workspace \
  -v /home/woonsang/theo/banana-in-pot-experiments:/workspace banana-eef:latest bash -lc \
  './lr_env/bin/python eval_offline.py \
     --checkpoint outputs/train/act_ee_val_diag/checkpoints/<STEP>/pretrained_model \
     --root ./banana_in_pot_ee_action_lerobot --repo-id theo/banana_in_pot_ee_action \
     --episodes 45,46,47,48,49,50 --device cuda \
     --out results/eval_ee_act_<STEP>' \
  2>&1 | tee results/eval_ee_act_<STEP>/metrics.txt
```
`eval_offline.py` auto-labels units from the dataset feature names (x/y/z→m, r1..r6→6d,
last→grip), so the EEF metrics come out correctly without code changes. `--policy-type` is
only used by `--smoke`; real runs infer the policy class from the checkpoint's config.json.
Metrics reported: `per_dim_mae/rmse`, `pose_mae/rmse`, `grip_mae/rmse`, `overall_l1`, `grip_acc`.
