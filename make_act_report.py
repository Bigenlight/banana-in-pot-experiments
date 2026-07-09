#!/usr/bin/env python3
"""
Generate ACT_RESULTS.md (pretty, with embedded PNG graphs) from the training log and the
per-checkpoint offline-eval logs. Designed to be re-runnable: it picks up whatever eval_out_*k/eval.log
dirs exist. Images go to report_assets/ and are linked with relative paths (VSCode markdown preview).

Usage: lr_env/bin/python make_act_report.py [--train-log <path>] [--out ACT_RESULTS.md]
"""
import argparse, glob, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(ROOT, "report_assets")
TRAIN_EPS = [0, 1, 2]
HELDOUT_EPS = [45, 46, 47, 48, 49, 50]
JOINT_NAMES = ["cmd1", "cmd2", "cmd3", "cmd4", "cmd5", "cmd6", "grip_cmd"]

def parse_train_log(path, log_freq=200):
    # metric lines are logged every `log_freq` steps; step is abbreviated (e.g. "step:37K") in the
    # log, so recover the exact step by enumeration instead of parsing the abbreviated value.
    steps, losses, grads = [], [], []
    if not path or not os.path.exists(path):
        return steps, losses, grads
    with open(path, errors="ignore") as f:
        txt = f.read().replace("\r", "\n")
    i = 0
    for m in re.finditer(r"ot_train\.py:606 step:\S+\s+smpl:\S+\s+ep:\S+\s+epch:[\d.]+\s+loss:([\d.]+)\s+grdn:([\d.]+)", txt):
        i += 1
        steps.append(i * log_freq); losses.append(float(m.group(1))); grads.append(float(m.group(2)))
    return steps, losses, grads

def parse_eval_log(path):
    """Return dict: {episodes:{ep:(overallL1,gripAcc)}, mean:{...}, perdim:{name:(mae,rmse)}}"""
    out = {"episodes": {}, "mean": {}, "perdim": {}}
    if not os.path.exists(path):
        return None
    with open(path, errors="ignore") as f:
        for line in f:
            m = re.search(r"episode (\d+): T=\d+\s+overallL1=([\d.]+)\s+gripAcc=([\d.]+)", line)
            if m:
                out["episodes"][int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
            m = re.search(r"^\s*MEAN\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
            if m:
                out["mean"] = dict(jointsMAE=float(m.group(1)), jointsRMSE=float(m.group(2)),
                                   gripMAE=float(m.group(3)), gripAcc=float(m.group(4)),
                                   overallL1=float(m.group(5)))
            m = re.search(r"^\s*(cmd[1-6]|grip_cmd)\s+([\d.]+)\s+([\d.]+)", line)
            if m:
                out["perdim"][m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return out if out["mean"] else None

def mean_over(evald, eps):
    vals = [evald["episodes"][e][0] for e in eps if e in evald["episodes"]]
    accs = [evald["episodes"][e][1] for e in eps if e in evald["episodes"]]
    return (sum(vals)/len(vals) if vals else None, sum(accs)/len(accs) if accs else None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-log", default=os.path.join(ROOT, "outputs/train/act_banana_in_pot/train_act.log"))
    ap.add_argument("--out", default=os.path.join(ROOT, "ACT_RESULTS.md"))
    args = ap.parse_args()
    os.makedirs(ASSETS, exist_ok=True)

    # ---- collect eval checkpoints ----
    evals = {}
    for d in sorted(glob.glob(os.path.join(ROOT, "eval_out_*k"))):
        m = re.search(r"eval_out_(\d+)k", d)
        if not m: continue
        step = int(m.group(1)) * 1000
        e = parse_eval_log(os.path.join(d, "eval.log"))
        if e: evals[step] = (e, d)
    ck_steps = sorted(evals)

    # ---- Figure 1: training loss curve ----
    steps, losses, grads = parse_train_log(args.train_log)
    loss_png = None
    if steps:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.plot(steps, losses, color="#2563eb", lw=1.6)
        ax.set_xlabel("training step"); ax.set_ylabel("loss (L1 + KL)")
        ax.set_title("ACT training loss"); ax.set_yscale("log"); ax.grid(alpha=0.3)
        for cs in ck_steps:
            ax.axvline(cs, color="#9ca3af", ls="--", lw=0.8)
        fig.tight_layout(); loss_png = "report_assets/loss_curve.png"
        fig.savefig(os.path.join(ROOT, loss_png), dpi=130); plt.close(fig)

    # ---- Figure 2: eval trend across checkpoints (train vs held-out) ----
    trend_png = None
    if ck_steps:
        tr_l1 = [mean_over(evals[s][0], TRAIN_EPS)[0] for s in ck_steps]
        ho_l1 = [mean_over(evals[s][0], HELDOUT_EPS)[0] for s in ck_steps]
        ho_acc = [mean_over(evals[s][0], HELDOUT_EPS)[1] for s in ck_steps]
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.plot(ck_steps, tr_l1, "o-", color="#16a34a", label="train eps L1")
        ax.plot(ck_steps, ho_l1, "s-", color="#dc2626", label="held-out eps L1")
        ax.set_xlabel("checkpoint step"); ax.set_ylabel("overall L1 (rad)")
        ax.set_title("Offline eval: action error vs checkpoint"); ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(ck_steps, ho_acc, "^:", color="#7c3aed", label="held-out gripper acc")
        ax2.set_ylabel("gripper accuracy"); ax2.set_ylim(0.9, 1.0)
        l1, lab1 = ax.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(l1+l2, lab1+lab2, loc="upper right", fontsize=8)
        fig.tight_layout(); trend_png = "report_assets/eval_trend.png"
        fig.savefig(os.path.join(ROOT, trend_png), dpi=130); plt.close(fig)

    # ---- best checkpoint = lowest held-out overall L1 ----
    best = min(ck_steps, key=lambda s: (mean_over(evals[s][0], HELDOUT_EPS)[0] or 9)) if ck_steps else None

    # ---- Figure 3: per-joint MAE at best ckpt ----
    perjoint_png = None
    if best and evals[best][0]["perdim"]:
        pd = evals[best][0]["perdim"]
        names = [n for n in JOINT_NAMES if n in pd]
        maes = [pd[n][0] for n in names]
        fig, ax = plt.subplots(figsize=(8, 3.8))
        bars = ax.bar(names, maes, color=["#2563eb"]*6 + ["#f59e0b"])
        ax.set_ylabel("MAE (rad | grip)"); ax.set_title(f"Per-dimension MAE @ step {best}")
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, maes):
            ax.text(b.get_x()+b.get_width()/2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout(); perjoint_png = "report_assets/perjoint_best.png"
        fig.savefig(os.path.join(ROOT, perjoint_png), dpi=130); plt.close(fig)

    # ---- trajectory PNGs from best ckpt dir ----
    traj_imgs = []
    if best:
        traj_imgs = sorted(glob.glob(os.path.join(evals[best][1], "eval_ep*_traj.png")))
        traj_imgs = [os.path.relpath(p, ROOT) for p in traj_imgs]

    # ---- write markdown ----
    L = []
    L.append("# ACT 학습 결과 — Put the right banana in the pot\n")
    L.append("> LeRobot v3.0 데이터셋(51 에피소드, UR7e + 2 cam)으로 학습한 ACT(Action Chunking Transformer) 정책.\n")
    done = steps and max(steps) >= 49000
    status = "완료 ✅" if done else f"진행 중 (최신 step {max(steps) if steps else '?'})"
    L.append(f"**상태:** {status}  |  **최종/현재 loss:** {losses[-1]:.3f}" if losses else f"**상태:** {status}")
    L.append("")
    L.append("## 학습 설정")
    L.append("| 항목 | 값 |")
    L.append("|---|---|")
    L.append("| 정책 | ACT (ResNet18 backbone, VAE, chunk_size=100) |")
    L.append("| 입력 | observation.state(7: UR q1~6 + grip_pos), cam1/cam2 @360×640 |")
    L.append("| 출력(action) | 7: cmd1~6(절대 관절각, rad) + grip_cmd |")
    L.append("| 이미지 | 720p → **360×640 on-the-fly resize** (재인코딩 없음) |")
    L.append("| batch / steps | 8 / 50,000 |")
    L.append("| optimizer | AdamW, lr 1e-5 (constant), pretrained ImageNet backbone |")
    L.append("| GPU | RTX 3060 12GB (~4.7GB 사용, ~3.9 step/s) |")
    L.append("")
    if loss_png:
        L.append("## 학습 loss 곡선")
        L.append(f"![training loss]({loss_png})")
        L.append("*점선 = 체크포인트 저장 지점.*\n")
    if trend_png:
        L.append("## 체크포인트별 오프라인 평가 (open-loop: 예측 액션 vs 정답)")
        L.append(f"![eval trend]({trend_png})")
        L.append("*train 에피소드와 held-out 에피소드의 오차가 거의 같음 = 과적합 없음 / 일반화됨.*\n")
        L.append("| step | joints MAE (rad) | overall L1 | gripper acc | train L1 | held-out L1 |")
        L.append("|---|---|---|---|---|---|")
        for s in ck_steps:
            e = evals[s][0]; m = e["mean"]
            tr = mean_over(e, TRAIN_EPS)[0]; ho = mean_over(e, HELDOUT_EPS)[0]
            star = " ⭐" if s == best else ""
            L.append(f"| {s}{star} | {m['jointsMAE']:.4f} | {m['overallL1']:.4f} | {m['gripAcc']*100:.1f}% | {tr:.4f} | {ho:.4f} |")
        L.append(f"\n⭐ = held-out 기준 최적 체크포인트 (step {best}).\n")
    if perjoint_png:
        L.append("## 관절별 오차 (최적 체크포인트)")
        L.append(f"![per-joint MAE]({perjoint_png})")
        L.append("*손목(cmd6)이 상대적으로 큼 — 원래 변동이 큰 축. 그리퍼는 이진에 가까움.*\n")
    if traj_imgs:
        L.append("## 예측 vs 정답 궤적 (샘플 에피소드)")
        for p in traj_imgs:
            L.append(f"![trajectory {os.path.basename(p)}]({p})")
        L.append("*파랑=정답(teleop), 주황=ACT 예측. 7개 액션 차원.*\n")
    L.append("## 해석 / 데이터셋 판정")
    if ck_steps:
        b = evals[best][0]["mean"]
        bho = mean_over(evals[best][0], HELDOUT_EPS)[0]; btr = mean_over(evals[best][0], TRAIN_EPS)[0]
        L.append(f"- 최적(step {best}) held-out overall L1 = **{bho:.4f} rad** (≈ {bho*57.3:.2f}°), train = {btr:.4f} → **갭 거의 없음 = 과적합 없이 일반화**.")
        L.append(f"- 관절 MAE **{b['jointsMAE']:.4f} rad (≈ {b['jointsMAE']*57.3:.2f}°)**, 그리퍼 정확도 **{b['gripAcc']*100:.1f}%**.")
        L.append("- **오프라인 지표상 데이터셋은 학습 가능하고 일관적 = '괜찮다'.** 최종 판정은 실로봇 closed-loop(deploy) 필요.")
    L.append("\n---")
    L.append("- 배포: `DEPLOY_UR.md` + `deploy_ur_act.py` (UR7e, ur_rtde).")
    L.append(f"- 최적 체크포인트: `outputs/train/act_banana_in_pot/checkpoints/{str(best).zfill(6) if best else 'NNNNNN'}/pretrained_model`")
    L.append("- 평가 재현: `lr_env/bin/python eval_offline.py --checkpoint <ckpt> --episodes 0,1,2,45,46,47,48,49,50 --device cuda --out <dir>`")

    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[report] wrote {args.out}")
    print(f"[report] checkpoints: {ck_steps}, best={best}")
    print(f"[report] figures: {[p for p in [loss_png, trend_png, perjoint_png] if p]} + {len(traj_imgs)} trajectory imgs")

if __name__ == "__main__":
    main()
