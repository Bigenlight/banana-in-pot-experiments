# Flow-Matching (JOINT) results — `multi_task_dit`, objective=flow_matching

Third policy family on the "put the right banana in the pot" task (UR7e + GELLO, LeRobot
v3.0), after ACT and Diffusion Policy. Same **overfit-diagnostic methodology**: hold out
the last 6 of 51 episodes (`--dataset.eval_split=0.117`, eps 45–50) as a true validation
split; train on the other 45; watch held-out signals while train loss falls.

- **Policy:** lerobot `multi_task_dit` with `--policy.objective=flow_matching` — a
  conditional/rectified flow-matching action head (linear-interpolation path, velocity
  target, Euler ODE integration) on a small DiT transformer with CLIP ViT-B/16 vision+text
  conditioning. It is the **only** non-VLA, train-from-scratch flow-matching policy in
  lerobot 0.6.1 (pin `8a74e0a`); pi0/pi0_fast/pi05/smolvla/evo1 all need heavy pretrained
  VLM backbones that don't fit a 3060 / need language.
- **Config (multi_task_dit defaults):** n_obs_steps=2, horizon=32, n_action_steps=24
  (drop_n_last_frames auto=7), hidden_dim=512, 6 layers, lr 2e-5 cosine, batch 8, 100k
  steps. **Image handling override:** resize-only to 224×224 (CLIP requires exactly
  224×224; crop disabled to never cut a banana out of the wide FOV — mirrors the diffusion
  runs' crop-OFF choice). Train script: `train_fm_joint_valdiag.sh`.
- **Run note:** the SSD filled at step 20k (checkpoint-save ENOSPC); resumed from the 10k
  checkpoint (`train_fm_joint_resume.sh`), lost ~10k steps, completed 100k cleanly. Old
  runs were archived to an HDD to free space. See `EXPERIMENT_LOG.md`.

---

## Headline: the `eval_loss`-vs-open-loop divergence reproduces for a 3rd policy family

Just like Diffusion Policy, the held-out **flow-matching `eval_loss` ROSE** (a
textbook-looking overfit curve) while the deployment-relevant **open-loop rollout MAE
improved then plateaued** — no destructive overfit through 100k.

| checkpoint | held-out `eval_loss` (val loss) | open-loop poseMAE (rad) | gripAcc | overallL1 |
|---|---|---|---|---|
| 10k | 0.096 | 0.0816 | 0.953 | 0.0790 |
| 20k | 0.119 | 0.0850 | 0.953 | 0.0810 |
| 30k | 0.120 | 0.0763 | **0.960** | 0.0728 |
| 40k | 0.142 | 0.0775 | 0.954 | 0.0740 |
| 50k | 0.171 | 0.0753 | 0.949 | 0.0724 |
| 60k | 0.188 | 0.0773 | 0.953 | 0.0735 |
| **70k** | 0.193 | **0.0735** (min) | 0.953 | **0.0703** (min) |
| 80k | 0.224 | 0.0745 | 0.952 | 0.0712 |
| 90k | 0.240 | 0.0748 | 0.952 | 0.0716 |
| 100k | 0.233 | 0.0746 | 0.952 | 0.0713 |

- **`eval_loss`** rises ~2.6× (min 0.089 @ ~6k → 0.233 @ 100k). By this signal alone you'd
  "early-stop at ~6k" — **wrong**.
- **open-loop poseMAE** (all 6 held-out eps 45–50, GPU, Euler-10 integration — the analogue
  of diffusion's DDIM-10) improves to **0.0735 @ 70k** then holds flat (70k/80k/90k/100k =
  0.0735/0.0745/0.0748/0.0746, within eval noise). gripAcc steady ~0.95–0.96.
- **Verdict: NO destructive overfitting through 100k. Best checkpoint = 70k** (min pose,
  min overallL1); anything ≥30k is effectively on the plateau.

**Standing lesson, now confirmed on flow matching too: select by open-loop MAE, never by
`eval_loss`.** The flow/denoising loss scores velocity-field prediction at random
timesteps, not sampled-action accuracy, so the two decorrelate.

---

## Comparison — FM vs Diffusion vs ACT (JOINT, held-out eps 45–50, open-loop poseMAE, rad)

| policy | best poseMAE (rad) | @ checkpoint | best gripAcc |
|---|---|---|---|
| **Flow matching (`multi_task_dit`)** | **0.0735** | 70k | ~0.96 |
| Diffusion Policy | 0.0845 | 80k | 0.953 |
| ACT | ~0.098 | 30k | ~0.94 |

**Flow matching is the best of the three on JOINT open-loop pose accuracy** — ~13% lower
poseMAE than Diffusion and ~25% lower than ACT, at comparable (excellent) gripper accuracy.
FM also beats Diffusion at *every* checkpoint (e.g. 10k: 0.0816 vs 0.1193; 30k: 0.0763 vs
0.0921), and converges fast (already on-plateau by ~30k).

> Caveat: same-family caveats as the other legs — eps 45–50 are the chronological tail (not
> an i.i.d. split); poseMAE is teacher-forced open-loop, not a closed-loop success rate;
> Euler-10 is a fast sampler (more integration steps could shift absolute numbers but not
> the ranking). Raw per-checkpoint metrics: `results/eval_fm_final_gpu.csv`.

---

## Artifacts

- Train: `train_fm_joint_valdiag.sh` (+ `train_fm_joint_resume.sh` for the crash-resume).
- Eval: `eval_offline.py` (now maps `--num-inference-steps` onto FM's `num_integration_steps`).
- Final metrics: `results/eval_fm_final_gpu.csv` (10 checkpoints × 6 eps, GPU, Euler-10).
- Best deploy checkpoint: **70k** (`outputs/train/fm_joint_val_diag/checkpoints/070000`).
