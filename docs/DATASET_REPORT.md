# banana_in_pot — LeRobot v3.0 dataset (conversion + QA report)

**Task:** "put the right banana in the pot" · UR5e + GELLO teleop · 2× RGB cam

## Result
- **Format:** LeRobot v3.0 (lerobot 0.6.1), `robot_type=ur5e_gello`
- **Location:** `./banana_in_pot_lerobot/` (repo_id `theo/banana_in_pot`, local-only, not pushed)
- **51 episodes · 21,524 frames · 30 fps · ~11.96 min · 483 MB**
- Converter: `./convert_to_lerobot.py` · venv: `./lr_env` (activate its python for anything lerobot)

## Schema
| key | dtype | shape | contents |
|---|---|---|---|
| `observation.state` | float32 | (7,) | UR q1..q6 + `grip_pos` |
| `action` | float32 | (7,) | command cmd1..cmd6 + `grip_cmd` (absolute joint targets) |
| `observation.images.cam1` | video (av1) | (720,1280,3) | viewpoint 1 |
| `observation.images.cam2` | video (av1) | (720,1280,3) | viewpoint 2 |

- **gello_* streams intentionally excluded** (not observable at inference).
- Multi-rate source (cam 30fps / robot ~56Hz) resampled onto the **cam1 timestamp grid via nearest-timestamp**. Normalization stats auto-computed.

## QA — 6 independent Opus validators, all PASS
1. **Schema/metadata** — v3.0 compliant, stats complete, no NaN in stats.
2. **Episode coverage** — 51 takes → 51 episodes, correct sorted order, frame totals reconcile 4 ways (raw=data=meta=info=21524). Only take #9/23/30/35 absent *by design* (re-takes fill to 51).
3. **Video integrity** — both cams 21,524 frames av1 720p30; ep0 vs raw mp4 corr 0.9995 (codec drift only); cameras not swapped; distinct viewpoints.
4. **Alignment** — independently re-derived (brute-force argmin) to 1e-7 for every frame; no gello leakage; ordering correct.
5. **Value sanity** — 0 NaN/Inf; joints in ±2π; grip_cmd effectively binary; no >0.5 rad wraparound jumps; cmd tracks state (correct labeling); the 333-NaN grip_cmd take fully repaired.
6. **Training-readiness** — real `lerobot-train` ran 2 ACT steps @ native 720p, loss 91.6→65.2, peak 9.5/12 GB.

### Known data characteristics (not conversion bugs)
- ~7 takes have raw robot-stream dropouts (75–280 ms gaps) → cam↔robot nearest-match error up to ~40 ms (≈2 control periods) at those frames. Mostly harmless for ACT.
- `grip_pos` peaks at 0.898 (6.3% of frames >0.6) = fully-open gripper, physical.
- Stray empty `images/` dir = temp scaffold (features are `video`), harmless.

## Working ACT train command (RTX 3060 12GB, native 720p)
```bash
cd /home/theo_lab/Downloads/Put_right_banana_in_the_pot-20260707T103848Z-3-001
export HF_LEROBOT_HOME=<writable_dir>          # dataset root is passed explicitly below
export TORCH_HOME=<writable_dir>               # backbone weights cache (home is read-only)
./lr_env/bin/lerobot-train \
  --dataset.repo_id=theo/banana_in_pot \
  --dataset.root=./banana_in_pot_lerobot \
  --policy.type=act --policy.device=cuda \
  --policy.push_to_hub=false \
  --batch_size=8 --steps=100000 \
  --wandb.enable=false --num_workers=4 \
  --output_dir=outputs/act_banana
```
Notes: at batch 2 native 720p peaked 9.5/12 GB — for batch 8 consider an image resize (`--dataset.image_transforms`) or drop to 480p. Keep `pretrained_backbone_weights` (needs net + writable `TORCH_HOME`) or set `=null` to train the ResNet from scratch offline.
