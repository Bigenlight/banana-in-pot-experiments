# DEPLOY_REPO_DECISION — Repo setup for real-robot ACT (and later HIL-SERL) on the UR7e

## TL;DR

**Extend `gello_software` (Option 3). Do not fork LeRobot; do not start a fresh repo.**
Add one new ROS2 package `ros2_ur_ws/src/gello_policy/` that acts as a **synthetic GELLO leader**
(Option-Bridge): it publishes ACT's actions to `/gello/joint_states` at 30 Hz and lets the existing
`gello_ur_bridge` do 250 Hz upsampling, One-Euro smoothing, slew clamping, staleness watchdog, and
the move-to-start handshake — all **unmodified**. LeRobot stays an unpatched pinned dependency
(0.6.1) living in a **Python 3.12 venv**, connected to the rclpy (Humble / Python 3.10) node by a
**local ZMQ REQ/REP split** — a single-process venv is impossible *on Humble* (`lerobot/pyproject.toml:32`
needs Python ≥3.12; Humble rclpy is CPython 3.10) and we deliberately stay on Humble to avoid
re-validating the safety stack on a distro migration; the split is also the HIL-SERL boundary and
crash-isolates CUDA.
Consume the action chunk **receding-horizon: re-query every 30 steps (1.0 s), publish at 30 Hz**.
The same setup extends to HIL-SERL: the ZMQ boundary becomes the gym env's transport, and the real
GELLO leader plus a small mux node gives human intervention for free.

---

## 1. Repo setup: extend `gello_software`

**Decision: Option 3 — a new ROS2 package inside `gello_software/ros2_ur_ws/src/`.**

| Criterion | Why Option 3 wins |
|---|---|
| **Code reuse / safety** | ALL low-level safety already lives in `ur_gello_bringup/gello_ur_bridge_node.py`: 250 Hz publisher with per-tick slew clamp (`max_step_rad`, 0.0025 rad/tick in `ur7e_gello.yaml` ⇒ 0.625 rad/s ceiling), One-Euro filtering tuned for a 30 Hz leader, 0.5 s monotonic-clock staleness watchdog, soft-start ramp, and a resume-alignment gate. `gello_move_to_start_node.py` provides the branch-cut-safe startup chase + STRICT controller switch + gated `~/resume`. A fresh repo or a lerobot fork would re-implement or wrap all of this — every line of duplicated safety code is a new way to crash the arm. |
| **Contract locality** | The train-time observation/action contract is *defined by this repo* (`gello_recorder/gello_ur_recorder_node.py`): `observation.state` = `/joint_states` (reordered) + `/robotiq_gripper/position_percent`; `action` = `/forward_position_controller/commands` + `/robotiq_gripper/command_percent`; cams = the two `.../color/image_raw/compressed` topics bound by RealSense serial. Deploy code that must byte-match this contract belongs next to the recorder that created it. |
| **Fork-lerobot rejected** | ACT inference needs only stable public API: `ACTPolicy.from_pretrained`, `make_pre_post_processors`, `policy.select_action` (already demonstrated in the root `deploy_ur_act.py` skeleton). Zero lerobot internals need patching. A fork buys nothing and costs permanent rebase burden against a fast-moving upstream — which matters for HIL-SERL, where you *want* to track upstream `rl/` fixes, not merge around them. lerobot's own robot-abstraction (`lerobot record`-style drivers) has no notion of this ROS2 bridge/handshake stack, so "adding UR7e inside lerobot" means porting the safety layer into a foreign framework. |
| **Fresh-repo rejected** | A standalone repo would still have to depend on `ur_gello_bringup` for the driver launch, controllers, gripper nodes, and calibration — so it isn't actually standalone; it's `gello_software` with extra clone steps and a split history. The deploy target is a separate robot PC: one repo = one thing to rsync/clone. |
| **ROS2-Humble reality** | The real stack is a colcon workspace built against Humble (`build_ur7e.sh`). A new `ament_python` package drops into the existing build with zero new infrastructure. |
| **HIL-SERL forward-compat** | See §5 — the heavy HIL-SERL machinery stays inside unpatched lerobot (py3.12 side); the ROS side only needs the same synthetic-leader interface plus a mux. Option 3 already gives that boundary. |

Precedent inside the repo: `ur_gello_bringup/fake_gello_node.py` already publishes a synthetic
`/gello/joint_states` + gripper width at 30 Hz for robotless testing — the policy node is exactly
that pattern with ACT as the signal source.

## 2. Integration mechanism: Option-Bridge (synthetic leader)

**Decision: publish synthetic `/gello/joint_states` (JointState, 6 joints, UR order) at 30 Hz and
reuse `gello_ur_bridge` unmodified. Do NOT publish to `/forward_position_controller/commands`
directly.**

- `forward_position_controller` does **not** interpolate (bridge docstring,
  `gello_ur_bridge_node.py:13`) — whatever you send is the 500 Hz servo setpoint. Option-Direct
  therefore means re-implementing slew clamp, soft-start, re-seed-on-stale, and the resume gate
  yourself, in new code, on a real arm. That is duplicated safety-critical code with no upside.
- The bridge is *specified* for exactly our input: "GELLO samples arrive at ~30 Hz while this
  bridge publishes at 250 Hz" (`gello_ur_bridge_node.py:81-84`). ACT at 30 Hz is a drop-in leader.
- The startup story also comes free: the policy node publishes the episode start pose as its
  "leader" value; `gello_move_to_start` chases it, STRICT-switches controllers, and gates
  `~/resume` — the same fail-safe path used for data collection.
- Known cost: the recorded `action[0:6]` are the bridge's *post-filter* outputs, so Option-Bridge
  filters an already-smooth signal twice → a few tens of ms extra lag and slight attenuation.
  Acceptable for this task; if tracking is sluggish, loosen `one_euro_min_cutoff` /
  `max_step_rad` in a deploy-specific YAML — a config change, not code.
- Gripper: publish `action[6]` straight to `/robotiq_gripper/command_percent` (Float32, 0..1
  continuous) — that is the exact topic the training action was recorded from, and the gripper
  has no slew-safety concern. Do not launch `gello_gripper_bridge` at deploy (avoid dual writers).

## 3. Layout and dependency strategy

### The hard constraint (verified)

`lerobot/pyproject.toml:32` → `requires-python = ">=3.12"`, and the working `lr_env` is
Python 3.12.3. ROS2 **Humble** ships rclpy built for CPython **3.10**. So lerobot+torch and rclpy
cannot share one interpreter **on Humble** without forking lerobot (rejected in §1) or building
rclpy from source for 3.12 (fragile, unsupported).

A single 3.12 env *is* technically possible on ROS2 **Jazzy** (rclpy on 3.12, with Jazzy builds of
`ur_robot_driver` and `realsense2_camera`). **We deliberately stay on Humble** because the entire
validated, field-tuned safety stack — bridge, move-to-start handshake, Robotiq Modbus, calibration
— is Humble-only, and a distro migration would re-validate safety-critical code on live hardware
for no functional gain. The ZMQ split is not a workaround we'd shed on Jazzy: it is *also* the
HIL-SERL actor/learner boundary (§5) and it crash-isolates the CUDA process from the ROS control
node. So the split stays regardless of distro; Humble just makes it additionally mandatory.

**Decision: two processes on the robot PC, joined by ZMQ REQ/REP on localhost.**

- **Process A — `policy_leader_node` (system Python 3.10, rclpy only):** subscribes
  `/joint_states`, `/robotiq_gripper/position_percent`, both `CompressedImage` topics; assembles
  the observation (raw JPEG bytes passed through — no cv2/torch needed); REQs the policy server at
  re-query time; runs a 30 Hz timer that pops the current action buffer and publishes
  `/gello/joint_states` + `/robotiq_gripper/command_percent`. Holds last target between chunks
  (bridge watchdog tolerates 0.5 s).

  **Position safety lives HERE, because the bridge has none.** `gello_ur_bridge`'s `_on_timer` is
  filter + per-tick velocity slew ONLY — it bounds *speed* (0.625 rad/s), never *position*: there
  is no joint-limit, workspace, or max-deviation check. A human leader physically can't drive the
  arm into the table; an OOD ACT output can, and the bridge will faithfully chase it at 0.625 rad/s
  until a contact protective-stop. So `policy_leader_node` MUST, before publishing each target:
  (a) clamp each joint to per-joint **dataset min/max ± margin**; (b) clamp to a **max deviation
  from live `/joint_states`** (e.g. 0.5 rad/joint). On any violation: hold current pose, log, and
  do not advance the chunk.

  **Fail-silent on server death, NOT hold-forever.** On ZMQ timeout or `act_server` crash,
  `policy_leader_node` MUST **stop publishing `/gello/joint_states`**, deliberately tripping the
  bridge's 0.5 s staleness watchdog (`gello_ur_bridge_node.py:355-369`). The bridge then halts
  streaming and, on recovery, re-seeds from the arm's *actual* pose with the soft-start ramp — the
  correct fail-safe. Do NOT keep re-publishing the last target through an outage (that would let
  the arm resume by closing a stale gap at full slew).
- **Process B — `act_server.py` (Python 3.12 venv, torch CUDA + lerobot 0.6.1 pinned):**
  `ACTPolicy.from_pretrained` + `make_pre_post_processors(pretrained_path=...)` (normalization
  stats baked in), runs the mandatory image preprocessing below, then the saved preprocessor
  pipeline exactly as in training, REPs a 30×7 action block per request. Sets
  `policy.config.n_action_steps=30` **before** the first `policy.reset()`; calls `policy.reset()`
  on episode reset.

  **Image preprocessing is NOT optional and NOT in the checkpoint.** The saved preprocessor has
  only 4 steps (rename → to_batch → device → normalizer) — there is **no resize**. The 360×640
  resize was a *dataloader* `image_transforms` applied to 100% of training samples
  (`train_act.sh:18-20`, `Resize [360,640]`, single deterministic transform); `config.json` still
  declares `3×720×1280` because `image_transforms` don't update declared features. So `act_server`
  MUST, per camera, per frame: `cv2.imdecode` the JPEG → **BGR→RGB** (the dataset was built RGB,
  `convert_to_lerobot.py:157,166`) → **torchvision `Resize([360,640])` bilinear + antialias**,
  byte-identical to the validated `eval_offline.py:68-85` `build_image_transforms()` → **then** the
  saved 4-step preprocessor. Skipping resize (feeding 720×1280) or leaving BGR silently degrades
  the policy — no error is raised. Reuse `eval_offline.py`'s transform verbatim.

Tradeoff vs single venv: ~1 ms IPC latency and one extra process to supervise — negligible at a
1 Hz query cadence, and it matches the repo's existing ZMQ idiom (`gello/zmq_core`). The split is
also precisely the boundary HIL-SERL's actor/learner needs later (§5). On the robot PC, install
Python 3.12 via `uv`/deadsnakes for the venv; ROS side needs nothing beyond apt Humble.

Do **not** ship the dev PC's editable lerobot clone; pin `lerobot==0.6.1` (or the exact commit)
in a `requirements.lock` and bundle the checkpoint directory.

### Tree (new items marked NEW)

```
gello_software/
├── ros2_ur_ws/src/
│   ├── ur_gello_bringup/                  # UNCHANGED — bridge, move_to_start, gripper nodes
│   ├── gello_recorder/                    # UNCHANGED — the train-time contract reference
│   └── gello_policy/                      # NEW ament_python package
│       ├── package.xml / setup.py
│       ├── gello_policy/
│       │   ├── policy_leader_node.py      # py3.10 rclpy: obs → zmq → /gello/joint_states @30Hz
│       │   ├── obs_assembler.py           # /joint_states reorder to UR order, grip, jpeg passthrough
│       │   └── policy_mux_node.py         # (HIL-SERL later) human-GELLO vs policy arbitration
│       ├── policy_server/                 # py3.12 side — run under venv, NOT imported by rclpy
│       │   ├── act_server.py              # ACTPolicy + processors + ZMQ REP loop
│       │   ├── hilserl_actor.py           # (later) lerobot actor wrapper
│       │   └── requirements.lock          # lerobot==0.6.1, torch cu12, pyzmq — pinned
│       ├── launch/
│       │   └── ur7e_act_real.launch.py    # ur7e_gello_real minus gello_publisher/gripper_bridge,
│       │                                  #   plus policy_leader_node
│       └── config/act_deploy.yaml         # topics, cam serials, start pose, k=30, checkpoint path
└── scripts/run_ur7e_act_real.sh           # NEW: colcon-sourced launch + venv policy server
```

## 4. Consuming the action chunk (chunk_size=100, n_action_steps=100, ensemble off)

**Decision: receding horizon — re-query the policy every k=30 executed steps (1.0 s), publish
actions into the bridge at 30 Hz.** Implemented by setting `policy.config.n_action_steps = 30`
after `from_pretrained` and **before the first `policy.reset()`** — `reset()` builds the action
deque with `maxlen=n_action_steps` (`modeling_act.py:93-98`), so changing it afterward has no
effect on the live queue. The checkpoint weights are untouched. Equivalently, `act_server` returns
the first 30 rows of each chunk.

Why:

- **30 Hz publish is non-negotiable.** The dataset timestep is 30 Hz (recorder cadence) and the
  bridge's filter tuning assumes a ~30 Hz leader. Publishing chunks faster/slower than trained
  changes the effective dynamics the policy learned.
- **Full-chunk open loop (k=100) = 3.3 s blind.** For a pick-and-place with a graspable object,
  3.3 s without feedback lets a slightly-off grasp become a fully-off episode. With
  `temporal_ensemble_coeff=None` there is no ensembling to blend chunks, so the *only* feedback
  channel is re-querying.
- **k=30 boundary jumps are already handled.** A new chunk conditioned on fresh obs can disagree
  with the tail of the old one; the bridge's One-Euro filter plus the 0.625 rad/s slew ceiling
  turns any boundary discontinuity into a bounded, smooth correction — this is exactly the class
  of input the bridge was built to sanitize.
- **Latency fits.** ACT forward on CUDA is ~10–50 ms; the 30 Hz publisher simply re-publishes the
  last target while the (asynchronous) re-query completes — far inside the 0.5 s staleness window.
- Escalation path, in order: **start at k=30**; if grasps miss / the policy reacts too late
  (reactive failure), **drop to k=10–15** — re-querying is cheap (ACT forward 10–50 ms sits well
  inside the 33 ms step budget); only if chunk-boundary artifacts visibly dominate **and**
  reactivity is already proven adequate, raise k toward 100.

## 5. HIL-SERL forward-compatibility

**Same repo, same package, same ZMQ boundary — one extra ROS node and a separate lockfile.**

- Everything heavy in HIL-SERL (`rl/gym_manipulator`, EE-delta processors, actor/learner, replay
  buffer) lives *inside* lerobot on the **py3.12 side** — untouched upstream code, which is the
  strongest argument against having forked it. Actor and learner run as py3.12 processes exactly
  like `act_server` does.
- Effort is more than "a `gym.Env` over ZMQ": lerobot's actor (`rl/actor.py`) and
  `make_robot_env` (`gym_manipulator.py:304,337`) construct a **registered lerobot `Robot` class**.
  So UR7e HIL-SERL needs a **custom `lerobot.Robot` subclass** (whose `send_action`/`get_observation`
  speak our ZMQ protocol to `policy_leader_node`) plus a **`Teleoperator` shim** exposing the GELLO
  leader as the intervention device. Both are **plugin-registered via draccus `ChoiceRegistry`** —
  no lerobot fork required; they live in `policy_server/` on the py3.12 side. The forward-compat
  conclusion (same repo, no fork) holds; the work is writing two registered adapter classes, not a
  thin env wrapper.
- The ROS side needs exactly one addition: `policy_mux_node.py` arbitrating between the real GELLO
  leader (`gello_publisher_node`, already in the repo — this is the human-intervention device
  HIL-SERL requires) and the policy stream, with intervention flags reported back over ZMQ.
- Joint↔EE conversion work already lives at project root (`hilserl/joint_to_ee.py`,
  `HILSERL_PREP_PLAN.md`); it slots in as py3.12-side processors, not ROS code.
- Keep envs separated by **lockfile, not by repo**: `requirements-act.lock` vs
  `requirements-hilserl.lock` (HIL-SERL may need a newer lerobot commit for `rl/` fixes; pin them
  independently so an RL upgrade can never break the working ACT deploy).

## 6. Risks and the single recommended path

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | Python 3.12 (lerobot) vs 3.10 (Humble rclpy) — someone "simplifies" to one venv with `--ignore-requires-python` | High | ZMQ split is the design, not an optimization. Document it; CI-check the lockfiles' python markers. |
| 2 | Camera identity swap (cam1 ↔ cam2) silently degrades the policy | High | Bind by serial in launch exactly as the recorder did (D435 `147122072740`=cam1, D435iF `243222072700`=cam2, `1280x720x30`); assert serials at node startup. |
| 2a | **Missing image preprocessing** — saved processor has NO resize (4 steps only); 360×640 resize + BGR→RGB were dataloader/build-time, not in the checkpoint. Feeding 720×1280 or BGR silently degrades the policy, no error | High | `act_server` explicitly does `imdecode → BGR→RGB (convert_to_lerobot.py:157,166) → torchvision Resize([360,640]) bilinear+antialias (verbatim from eval_offline.py:68-85) → saved preprocessor`. |
| 2b | **Bridge bounds velocity, never position** — `_on_timer` is filter+slew only; no joint-limit/workspace/max-deviation check. An OOD ACT target is chased at 0.625 rad/s into the table until a contact protective-stop | High | `policy_leader_node` clamps every target to per-joint dataset min/max ± margin AND to ≤0.5 rad deviation from live `/joint_states`; on violation hold pose + log + don't advance chunk. |
| 3 | Startup: `gello_move_to_start` chases the "leader" — policy node must already be publishing a sane start pose before the handshake | High | Policy node boots in HOLD, publishing the **dataset episode-start mean q = `[3.106, -1.817, 1.653, -1.618, -1.628, -3.195]` rad** (mean first-frame joints) — NOT the `deploy_ur_act.py:60` placeholder `START_Q` — until `~/resume` succeeds; only then begins chunk execution. |
| 4 | Double filtering (actions recorded post-One-Euro, bridge filters again) → lag/attenuation | Medium | Accept for v1; tune `one_euro_min_cutoff`/`max_step_rad` in `act_deploy.yaml` if tracking lags. Never bypass the bridge to "fix" this. |
| 5 | Chunk-boundary discontinuities at k=30 | Medium | Bounded by 0.625 rad/s slew ceiling; verify on first runs, raise k if jerky. |
| 6 | Stale state across episodes (action queue, One-Euro state, gripper) | Medium | Set `n_action_steps` BEFORE `policy.reset()` (deque `maxlen`, `modeling_act.py:93-98`); `policy.reset()` on every episode start; bridge re-seed covers its own filter on resume/stale. |
| 6a | **ACT-server crash / ZMQ timeout** — holding last target forever would let the arm later resume by closing a stale gap at full slew | High | `policy_leader_node` STOPS publishing `/gello/joint_states` on timeout, tripping the bridge's 0.5 s staleness watchdog (`gello_ur_bridge_node.py:355-369`) → bridge halts + soft-start re-seed on recovery. Never hold-last-forever. |
| 7 | Dual gripper writers if `gello_gripper_bridge` is left running | Medium | `ur7e_act_real.launch.py` excludes it; policy node is the sole `/robotiq_gripper/command_percent` publisher. |
| 8 | Dev-PC editable lerobot clone drifts from robot-PC install | Low | Pin `lerobot==0.6.1` + torch versions in `requirements.lock`; bundle checkpoint; no editable installs on the robot PC. |

**The path:** create `ros2_ur_ws/src/gello_policy/` in `gello_software` → implement
`policy_leader_node.py` (py3.10, modeled on `fake_gello_node.py`) and `act_server.py` (py3.12,
modeled on root `deploy_ur_act.py`, minus its ur_rtde path) → `ur7e_act_real.launch.py` reusing
`ur7e_gello_real.launch.py` with the GELLO leader nodes swapped for the policy node → colcon build
on the robot PC, py3.12 venv from `requirements.lock` → first runs at k=30, arm E-stop in hand →
add `policy_mux_node` + HIL-SERL lockfile when RL starts.
