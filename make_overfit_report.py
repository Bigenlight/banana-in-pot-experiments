#!/usr/bin/env python3
"""
Reusable overfit-diagnostic report generator for a val-split training run.

Parses the val-diagnostic training stdout log into two series
(train_loss vs step, held-out/val eval_loss vs step), plots them, and
writes a Markdown report with a table + an auto verdict.

Safe to run mid-training or at completion (robust to a partial/growing log).
Does NOT touch the training process.

Default (no args) reproduces the ACT overfit report EXACTLY (same default log
path, same ACT_OVERFIT_DIAGNOSIS.md + report_assets/act_overfit_diag.png,
same verdict). The argparse options below only change behavior when supplied,
so the no-argument ACT monitoring invocation is unchanged.

Usage:
    python make_overfit_report.py                       # ACT default, unchanged
    python make_overfit_report.py --log path/to.log \\
        --md-out DIFFUSION_JOINT_OVERFIT.md \\
        --png-out report_assets/diffusion_joint_overfit_diag.png \\
        --title "Diffusion (joint) overfit diagnostic" --total-steps 100000 --smooth 3
"""

import argparse
import os
import re
import glob
import tempfile

# ---------------------------------------------------------------------------
# Defaults (current ACT behavior)
# ---------------------------------------------------------------------------
DEFAULT_LOG_FREQ = 200          # train loss logged every 200 steps
DEFAULT_EVAL_FREQ = 2000        # held-out eval logged every 2000 steps
DEFAULT_TOTAL_STEPS = 100000

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(PROJECT_ROOT, "report_assets")
PNG_REL = "report_assets/act_overfit_diag.png"
PNG_ABS = os.path.join(PROJECT_ROOT, PNG_REL)
MD_ABS = os.path.join(PROJECT_ROOT, "ACT_OVERFIT_DIAGNOSIS.md")

# Title used for the ACT plot (kept identical to the pre-argparse hardcoded one).
ACT_PLOT_TITLE = "ACT overfit diagnostic (45 train / 6 held-out eps)"

_SCRATCH = os.environ.get("SCRATCH_DIR", os.path.join(os.getcwd(), ".cache"))
DEFAULT_LOG_CANDIDATES = [
    os.path.join(_SCRATCH, "act_valdiag.log"),
]
# Fallback glob in case the session dir differs.
DEFAULT_LOG_GLOB = os.path.join(
    tempfile.gettempdir(), "claude-*", "**", "act_valdiag.log"
)


def find_log(explicit):
    if explicit:
        return explicit
    for c in DEFAULT_LOG_CANDIDATES:
        if os.path.exists(c):
            return c
    hits = glob.glob(DEFAULT_LOG_GLOB, recursive=True)
    if hits:
        # newest by mtime
        return max(hits, key=os.path.getmtime)
    return DEFAULT_LOG_CANDIDATES[0]


# ---------------------------------------------------------------------------
# Parsing  (regexes UNCHANGED)
# ---------------------------------------------------------------------------
# Train loss INFO lines, e.g.:
#   INFO ... ot_train.py:606 step:2K smpl:16K ep:38 epch:0.85 loss:1.329 grdn:...
# The step: token may be exact ("step:200") or K-abbreviated ("step:2K").
# Because K-abbreviation is imprecise, we prefer ordinal*LOG_FREQ for the step
# and only fall back to the parsed token if the ordinal is unavailable.
TRAIN_RE = re.compile(r"step:(\d+)(K?)\s+smpl:\S+\s+ep:\d+\s+epch:[\d.]+\s+loss:([\d.]+)")

# Eval lines, e.g.:  ot_train.py:637 step 2000: eval_loss=0.6520
EVAL_RE = re.compile(r"step\s+(\d+):\s*eval_loss=([\d.]+)")


def parse_token_step(num, ksuffix):
    n = int(num)
    return n * 1000 if ksuffix else n


def parse_log(path, log_freq=DEFAULT_LOG_FREQ):
    """Return (train_series, val_series).

    train_series: list of (step:int, loss:float)
    val_series:   list of (step:int, eval_loss:float)
    """
    if not os.path.exists(path):
        return [], []
    with open(path, "r", errors="replace") as f:
        text = f.read()

    # --- train series ---
    train = []
    for ordinal, m in enumerate(TRAIN_RE.finditer(text), start=1):
        token_step = parse_token_step(m.group(1), m.group(2))
        loss = float(m.group(3))
        # Exact step: ordinal * log_freq (robust to K-abbreviation).
        exact_step = ordinal * log_freq
        # Cross-check: if the parsed token (non-K) disagrees a lot with the
        # ordinal estimate, trust the parsed exact integer token instead.
        if m.group(2) == "" and abs(token_step - exact_step) > log_freq:
            exact_step = token_step
        train.append((exact_step, loss))

    # --- val series ---
    val = []
    seen = set()
    for m in EVAL_RE.finditer(text):
        step = int(m.group(1))
        loss = float(m.group(2))
        if step in seen:
            # keep last occurrence for a given step
            val = [(s, l) for (s, l) in val if s != step]
        seen.add(step)
        val.append((step, loss))
    val.sort(key=lambda x: x[0])
    return train, val


def nearest_train_loss(train, step):
    """Train loss closest to a given step (for the per-eval table)."""
    if not train:
        return None
    return min(train, key=lambda t: abs(t[0] - step))[1]


def smooth_series(series, window):
    """Centered rolling mean over the value axis; x (step) positions unchanged.
    Edges use a shrinking window. window<=1 returns the series unchanged."""
    if window <= 1 or not series:
        return list(series)
    xs = [s for s, _ in series]
    ys = [l for _, l in series]
    n = len(ys)
    half = window // 2
    out = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append((xs[i], sum(ys[lo:hi]) / (hi - lo)))
    return out


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(train, val, val_smooth, s_min, v_min, png_abs, title, smooth):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(png_abs) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    if train:
        xs = [s for s, _ in train]
        ys = [l for _, l in train]
        ax.plot(xs, ys, lw=0.9, color="#4C72B0", alpha=0.85,
                label="train loss (per 200 steps)")

    if val:
        vx = [s for s, _ in val]
        vy = [l for _, l in val]
        if smooth > 1:
            # raw points faint, smoothed overlay bold
            ax.plot(vx, vy, lw=1.0, color="#C44E52", alpha=0.30, marker="o",
                    ms=3, label="held-out (val) eval loss (raw)")
            sx = [s for s, _ in val_smooth]
            sy = [l for _, l in val_smooth]
            ax.plot(sx, sy, lw=2.6, color="#C44E52", marker="o", ms=6,
                    label=f"held-out (val) eval loss (rolling mean {smooth})")
        else:
            ax.plot(vx, vy, lw=2.6, color="#C44E52", marker="o", ms=6,
                    label="held-out (val) eval loss")

    if s_min is not None:
        ax.axvline(s_min, ls="--", lw=1.4, color="#555555")
        ax.annotate(f"val min @ step {s_min} (={v_min:.4f})",
                    xy=(s_min, v_min),
                    xytext=(8, 12), textcoords="offset points",
                    fontsize=9, color="#333333",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#FFF6CC", ec="#999", lw=0.7))

    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(png_abs, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
def compute_verdict(train, val):
    """Return dict with verdict fields. Handles <2 eval points.
    `val` is the series the verdict is computed on (smoothed when smooth>1)."""
    out = {
        "n_eval": len(val),
        "insufficient": len(val) < 2,
    }
    if not val:
        out["verdict"] = "INSUFFICIENT EVAL POINTS YET (no held-out eval logged)"
        return out

    s_min, v_min = min(val, key=lambda x: x[1])
    s_last, v_final = val[-1][0], val[-1][1]
    out.update(dict(s_min=s_min, v_min=v_min, s_last=s_last, v_final=v_final))

    # train-vs-val gap at the final eval step
    tr_at_final = nearest_train_loss(train, s_last)
    out["train_at_final"] = tr_at_final
    out["gap_final"] = (tr_at_final - v_final) if tr_at_final is not None else None

    if len(val) < 2:
        out["verdict"] = ("INSUFFICIENT EVAL POINTS YET (only 1 held-out eval); "
                          "trend cannot be assessed")
        return out

    # Classification
    ratio = v_final / v_min if v_min > 0 else 1.0
    frac = s_min / s_last if s_last > 0 else 1.0
    if s_min >= 0.9 * s_last and v_final <= 1.02 * v_min:
        verdict = "NO OVERFIT (val still improving)"
    elif ratio <= 1.10 and (ratio > 1.02 or (0.5 <= frac < 0.9)):
        verdict = "MILD OVERFIT"
    elif ratio > 1.10:
        verdict = f"OVERFIT from ~step {s_min}"
    else:
        # ratio <= 1.02 but s_min not near the end -> plateau -> mild
        verdict = "NO OVERFIT (val still improving)"
    out["verdict"] = verdict
    out["ratio"] = ratio
    out["frac"] = frac
    return out


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def write_md(train, val, v, md_abs, png_rel, title, smooth, log_freq, eval_freq):
    # `title is None` -> exact ACT default text (byte-identical to pre-argparse).
    is_act_default = title is None
    lines = []
    if is_act_default:
        lines.append("# ACT Overfit Diagnosis")
        lines.append("")
        lines.append("_ACT trained on 45 episodes, 6 episodes (eps 45-50) held out as a "
                     "true validation split. Train loss logged every 200 steps; held-out "
                     "eval loss every 2000 steps._")
    else:
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"_Trained on 45 episodes, 6 episodes (eps 45-50) held out as a "
                     f"true validation split. Train loss logged every {log_freq} steps; "
                     f"held-out eval loss every {eval_freq} steps._")
    lines.append("")
    alt = "ACT overfit diagnostic" if is_act_default else "overfit diagnostic"
    lines.append(f"![{alt}]({png_rel})")
    lines.append("")

    # --- table ---
    lines.append("## Held-out eval points")
    lines.append("")
    if val:
        lines.append("| step | train_loss (nearest) | val / held-out loss |")
        lines.append("|-----:|---------------------:|--------------------:|")
        for s, vl in val:
            tr = nearest_train_loss(train, s)
            tr_s = f"{tr:.4f}" if tr is not None else "n/a"
            lines.append(f"| {s} | {tr_s} | {vl:.4f} |")
    else:
        lines.append("_No held-out eval points logged yet._")
    lines.append("")

    # --- summary numbers ---
    smooth_note = f" (smoothed, rolling mean {smooth})" if smooth > 1 else ""
    lines.append("## Summary")
    lines.append("")
    if v.get("insufficient"):
        lines.append(f"- Held-out eval points so far: **{v['n_eval']}** "
                     "-> **insufficient eval points yet** to judge the val-curve "
                     "shape (need >= 2).")
        if val:
            s0, v0 = val[0]
            lines.append(f"- First (only) eval: step {s0}, val_loss = {v0:.4f}.")
    else:
        lines.append(f"- **Val-loss minimum:** {v['v_min']:.4f} @ step **{v['s_min']}**{smooth_note}")
        lines.append(f"- **Final val loss:** {v['v_final']:.4f} @ step {v['s_last']}{smooth_note}")
        lines.append(f"- **val_final / val_min ratio:** {v['ratio']:.4f}")
        if v.get("train_at_final") is not None:
            lines.append(f"- **Train vs val at final step ({v['s_last']}):** "
                         f"train {v['train_at_final']:.4f} vs val {v['v_final']:.4f} "
                         f"(gap = {v['gap_final']:+.4f})")
        lines.append(f"- **Recommended early-stop step:** {v['s_min']} "
                     "(step of the val-loss minimum)")
    lines.append("")

    # --- verdict ---
    lines.append("## Auto verdict")
    lines.append("")
    lines.append(f"> **{v['verdict']}**")
    lines.append("")
    if not v.get("insufficient"):
        s_min, v_min = v["s_min"], v["v_min"]
        vf = v["v_final"]
        if v["verdict"].startswith("NO OVERFIT"):
            lines.append(
                f"The held-out loss is still at (or near) its minimum at the most "
                f"recent eval: it bottoms at {v_min:.4f} (step {s_min}) and finishes at "
                f"{vf:.4f}, within 2% of the min. No sign of the val curve turning back "
                f"up, so training longer is not yet hurting generalization. Keep the "
                f"latest checkpoint; the natural early-stop point is currently the end "
                f"of the run.")
        elif v["verdict"].startswith("MILD"):
            lines.append(
                f"The held-out loss reached {v_min:.4f} at step {s_min} and has since "
                f"drifted to {vf:.4f} (ratio {v['ratio']:.3f}), or plateaued "
                f"(min at {v['frac']*100:.0f}% of the run). This is mild/borderline "
                f"overfitting: generalization is no longer clearly improving but has not "
                f"badly degraded. **Recommended early-stop step: {s_min}.**")
        else:  # OVERFIT
            lines.append(
                f"The held-out loss bottoms at {v_min:.4f} (step {s_min}) then rises to "
                f"{vf:.4f} at the end (ratio {v['ratio']:.3f} > 1.10). The model is "
                f"overfitting the 45 training episodes past ~step {s_min}: further steps "
                f"trade held-out generalization for lower train loss. "
                f"**Use the checkpoint at step {s_min} (recommended early-stop).**")
    else:
        lines.append("Re-run this script once more held-out eval points have been "
                     "logged (they appear every 2000 steps) to obtain a trend-based "
                     "verdict.")
    lines.append("")

    # --- caveats ---
    lines.append("## How to read this (caveats)")
    lines.append("")
    lines.append("- **Train and val magnitudes are NOT directly comparable.** The train "
                 "loss is logged with dropout active and is a running-average tracker "
                 "over recent batches; the held-out eval loss is computed under "
                 "`policy.eval()` (dropout off) on unseen episodes. A raw train-below-val "
                 "gap is expected and is *not* itself evidence of overfitting.")
    lines.append("- **Only the trends and the val-curve shape matter.** Overfitting shows "
                 "up as the held-out (red, bold) curve flattening and then turning "
                 "*upward* while train loss keeps dropping. The vertical dashed line marks "
                 "the val minimum = the recommended early-stop step.")
    if smooth > 1:
        lines.append("- **Held-out eval loss is stochastic for diffusion**: each held-out "
                     "eval re-samples a random noise vector and diffusion timestep per "
                     f"forward, so single points are noisy by construction. The plot shows "
                     f"the raw per-eval points faintly plus a **centered rolling-mean({smooth})** "
                     f"overlay (bold), and the verdict / val-minimum above are computed on "
                     f"that **smoothed** series. Read the smoothed curve, not individual raw points.")
    if is_act_default:
        lines.append("- ACT total loss = L1 reconstruction + KLD (VAE) term; both series use "
                     "the same total-loss definition.")
    lines.append("- Generated by `make_overfit_report.py`; safe to re-run on the growing "
                 "log at any time.")
    lines.append("")

    with open(md_abs, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", default=None,
                   help="Path to the training stdout log (default: ACT val-diag log).")
    p.add_argument("--md-out", default=MD_ABS,
                   help="Output Markdown path (default: ACT_OVERFIT_DIAGNOSIS.md).")
    p.add_argument("--png-out", default=PNG_ABS,
                   help="Output PNG path (default: report_assets/act_overfit_diag.png).")
    p.add_argument("--title", default=None,
                   help="Report/plot title (default: ACT report, unchanged).")
    p.add_argument("--log-freq", type=int, default=DEFAULT_LOG_FREQ,
                   help="Train-loss log frequency in steps (default 200).")
    p.add_argument("--eval-freq", type=int, default=DEFAULT_EVAL_FREQ,
                   help="Held-out eval frequency in steps (default 2000).")
    p.add_argument("--total-steps", type=int, default=DEFAULT_TOTAL_STEPS,
                   help="Planned total training steps (default 100000; informational).")
    p.add_argument("--smooth", type=int, default=1,
                   help="Centered rolling-mean window over the val series (default 1 = off). "
                        "When >1, the verdict is computed on the smoothed series.")
    return p.parse_args()


def main():
    args = parse_args()
    log = find_log(args.log)
    md_abs = os.path.abspath(args.md_out)
    png_abs = os.path.abspath(args.png_out)
    # PNG path embedded in the MD, relative to the MD's directory.
    png_rel = os.path.relpath(png_abs, start=os.path.dirname(md_abs))

    # Plot title: for the ACT default (title None), keep the exact old title.
    plot_title = ACT_PLOT_TITLE if args.title is None else args.title

    train, val = parse_log(log, log_freq=args.log_freq)
    val_smooth = smooth_series(val, args.smooth)
    # Verdict on the smoothed series when smoothing is on.
    v = compute_verdict(train, val_smooth if args.smooth > 1 else val)

    s_min = v.get("s_min")
    v_min = v.get("v_min")
    make_plot(train, val, val_smooth, s_min, v_min, png_abs, plot_title, args.smooth)
    write_md(train, val, v, md_abs, png_rel, args.title, args.smooth,
             args.log_freq, args.eval_freq)

    # Console summary (data for the caller).
    print(f"log:            {log}")
    print(f"train points:   {len(train)}"
          + (f"  (last step {train[-1][0]}, loss {train[-1][1]:.4f})" if train else ""))
    print(f"eval points:    {len(val)} -> {[(s, round(l,4)) for s, l in val]}")
    if args.smooth > 1:
        print(f"smoothed(win {args.smooth}) -> "
              f"{[(s, round(l,4)) for s, l in val_smooth]}")
    print(f"verdict:        {v['verdict']}")
    if not v.get("insufficient"):
        print(f"val min:        {v_min:.4f} @ step {s_min}")
        print(f"val final:      {v['v_final']:.4f} @ step {v['s_last']}")
    print(f"png:            {png_abs}")
    print(f"md:             {md_abs}")


if __name__ == "__main__":
    main()
