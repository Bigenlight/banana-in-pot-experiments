# Diffusion Policy (joint) — overfit diagnosis

_Diffusion Policy, JOINT 7-D action space. Trained on 45 episodes, 6 episodes (eps 45–50) held out as a true validation split. Ran to 80k steps. Two held-out signals were tracked: the denoising `eval_loss` (every 2k steps) and the deployment-relevant open-loop rollout MAE (`eval_offline.py`, DDIM-10, every 10k)._

## TL;DR — verdict

> **NO open-loop overfitting through 80k. Best checkpoint = 80k. Select diffusion checkpoints by open-loop MAE, NOT by `eval_loss`.**

The held-out **denoising `eval_loss` rose ~5×** (0.0289 @ 4k → 0.1487 @ 80k), which *looks* like severe overfitting and is what a naive early-stop rule flags. But the signal that actually matters for deployment — the **open-loop rollout MAE on the same held-out episodes — IMPROVED monotonically and then PLATEAUED** through 80k (poseMAE 0.1193 → 0.0845 rad; gripper accuracy 0.729 → 0.953). The model kept getting better (or held) on unseen data the entire run.

The two signals **decorrelate**, and for a diffusion policy the loss is the misleading one. Deploy the **80k** checkpoint; anything ≥60k is effectively equivalent on pose.

## Setup / method

- **Policy:** Diffusion Policy, JOINT action space (7-D: 6 arm joints + gripper).
- **Data:** 51 episodes total. `eval_split = 0.117` holds out the **last 6 episodes (45–50)** as a true validation set; the policy is **trained on the first 45**.
- **Training:** ran to **80k** steps. Train loss logged every 200 steps; held-out denoising `eval_loss` every 2k steps.
- **Open-loop eval:** `eval_offline.py`, **DDIM-10** sampling, teacher-forced over held-out eps 45–50, errors in **radians**. This is the deployment-relevant metric — it scores the *sampled action*, not the noise-prediction loss.

(Same 45-train / 6-held-out protocol as the ACT sibling diagnosis, so the two are directly comparable.)

## The figure

![Diffusion joint: eval_loss vs open-loop MAE](../results/report_assets/diffusion_joint_overfit_diag.png)

**Blue = held-out denoising `eval_loss`** (raw points faint + centered rolling-mean(3) bold). It bottoms early (~0.029 around 4–6k) then climbs steadily to ~0.15 — the textbook "val turns up while train falls" overfit shape. **This is the misleading signal.** The open-loop MAE curve (table below) tells the *opposite* story on the very same held-out episodes: it goes down and stays down.

## Why the two signals disagree (diffusion-specific)

The denoising `eval_loss` and the open-loop MAE measure fundamentally different things, and only for diffusion do they decouple this hard:

- **`eval_loss` scores noise prediction at *random* diffusion timesteps.** Each held-out eval draws a fresh noise vector and a random timestep, then measures how well the network predicts that noise. As training sharpens the network to the 45 training trajectories, its per-timestep noise-prediction on unseen episodes degrades — a real generalization gap on the *auxiliary* denoising objective. It is also stochastic by construction (hence the rolling-mean smoothing).
- **Open-loop MAE scores the *sampled action* — the integral of the full reverse process.** The deployed quantity is the endpoint of the DDIM-10 reverse chain, not any single-timestep residual. Errors at individual timesteps partially cancel over the reverse integration, so the sampled action can keep improving even as the pointwise denoising loss worsens.

Net: a rising denoising `eval_loss` is **not** evidence that the sampled policy generalizes worse. The reverse-process output is what deploys, and that is what open-loop MAE measures.

## Results

**Open-loop rollout MAE** — `eval_offline.py`, DDIM-10, held-out eps 45–50, radians:

| checkpoint | poseMAE (rad) | gripAcc | overall L1 |
|---|---|---|---|
| 10k | 0.1193 | 0.729 | 0.1454 |
| 20k | 0.1037 | 0.888 | 0.1078 |
| 30k | 0.0921 | 0.919 | 0.0928 |
| 40k | 0.0907 | 0.949 | 0.0862 |
| 50k | 0.0865 | 0.942 | 0.0832 |
| 60k | 0.0849 | 0.944 | 0.0812 |
| 70k | 0.0855 | 0.951 | 0.0809 |
| **80k** | **0.0845** | **0.953** | **0.0796** |

- poseMAE improves monotonically then **plateaus at ~0.085 from 60k onward** (60k/70k/80k = 0.0849/0.0855/0.0845, within eval noise).
- Gripper accuracy climbs all the way to **0.953 @ 80k**.
- **No open-loop overfitting through 80k.**

**Denoising `eval_loss` endpoints** (smoothed, rolling-mean 3): minimum **0.0289 @ ~4k** → **0.1487 @ 80k** (ratio ≈ 4.93). This is the signal a naive early-stop rule would have followed to stop at ~6k — and it would have been **wrong**, discarding 74k steps of genuine open-loop improvement.

## Contrast with ACT

The two policies behave differently, and this is the whole lesson:

- **ACT — signals AGREE.** Held-out `eval_loss` fell and never turned up, *and* open-loop MAE improved to ~30k then flattened. Both said "no destructive overfit." Either signal alone would have led you to the right checkpoint.
- **Diffusion — signals DISAGREE.** Held-out denoising `eval_loss` screams "5× overfit," while open-loop MAE says "still improving / plateaued, deploy latest." Trusting `eval_loss` here would throw away the best model.

See [`ACT_OVERFIT_DIAGNOSIS.md`](ACT_OVERFIT_DIAGNOSIS.md) for the ACT side.

## Conclusion / recommendation

- **Deploy the 80k checkpoint** — tied-best poseMAE (0.0845 rad) and best gripAcc (0.953). Checkpoints **≥60k are ~equivalent on pose**, so 60k/70k/80k are all safe choices; 80k wins on gripper.
- **Operational lesson for future diffusion runs:** the held-out **denoising `eval_loss` is NOT a valid overfit / early-stop signal** for a diffusion policy. Its rise reflects the auxiliary noise-prediction objective, not deployed-action quality. **Always select diffusion checkpoints by open-loop rollout MAE** (or, ultimately, closed-loop success), and run the open-loop eval across the full training curve before picking a checkpoint.
- **Caveat:** held-out eps 45–50 are the last 6 by collection order (chronological tail), not an i.i.d. random split, so the reported gap conflates generalization with any session drift. The "no open-loop overfit, select by MAE" conclusion is robust to this.
</content>
</invoke>
