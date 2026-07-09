#!/usr/bin/env bash
# =============================================================================
# fetch_data.sh — pull / rebuild the banana-in-pot datasets on a fresh PC
# -----------------------------------------------------------------------------
# See REPRODUCIBILITY_PLAN.md §4 (Data & model strategy) for the full rationale.
#
# What this does:
#   MODE=joint (default)  Download the JOINT LeRobot v3 dataset (public HF).
#                         Optionally the pretrained ACT model.  Enough to run
#                         the ACT experiments and the JOINT diffusion run.
#   MODE=all              Also download the raw h5+mp4 source AND locally
#                         rebuild the EE-action dataset (NOT on HF — 401),
#                         then validate it.
#
# The EE-action dataset was built by *video-reuse* (AV1 videos copied from the
# JOINT dataset; only parquet + stats regenerated), so its rebuild REQUIRES both
# the JOINT dataset and the raw h5 present first. CPU-only, a few minutes.
#
# repo_id mismatch caveat (§4): the local dataset's meta/info.json carries
#   repo_id: theo/banana_in_pot(_ee_action)  while the HF mirror lives under
#   Bigenlight/... . This is HARMLESS for our scripts — everything runs with
#   --dataset.root=<dir> and HF_HUB_OFFLINE=1 — and some lerobot paths key
#   caches off repo_id, so DO NOT "fix"/rewrite it.
#
# Usage:
#   ./fetch_data.sh            # same as: ./fetch_data.sh joint
#   ./fetch_data.sh joint      # JOINT dataset (+ optional ACT model)
#   ./fetch_data.sh all        # JOINT + raw + rebuild EE-action + validate
#   ./fetch_data.sh --help
#
# Env toggles:
#   FETCH_ACT_MODEL=1   also download the 4.7 GB pretrained ACT model (default 0)
#   FETCH_EE_OBS=1      also download the optional EE-obs dataset (default 0)
#
# Requirements: the `hf` CLI (huggingface_hub >= 1.22.0, provided by lr_env /
# setup.sh) on PATH, and — for MODE=all — ./lr_env populated (see setup.sh).
# =============================================================================

set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

# ---- config ----------------------------------------------------------------
JOINT_DIR="./banana_in_pot_lerobot"
RAW_DIR="./Put_right_banana_in_the_pot"
EE_DIR="./banana_in_pot_ee_action_lerobot"
ACT_DIR="./outputs/train/act_banana_in_pot/pretrained_from_hf"
EE_OBS_DIR="./banana_in_pot_ee_lerobot"

PY="./lr_env/bin/python"

FETCH_ACT_MODEL="${FETCH_ACT_MODEL:-0}"
FETCH_EE_OBS="${FETCH_EE_OBS:-0}"

# ---- helpers ---------------------------------------------------------------
usage() {
    sed -n '2,45p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

log()  { printf '\n\033[1;36m[fetch_data] %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[fetch_data][warn] %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31m[fetch_data][error] %s\033[0m\n' "$*" >&2; exit 1; }

need_hf() {
    command -v hf >/dev/null 2>&1 \
        || die "the 'hf' CLI is not on PATH — activate lr_env or run setup.sh first"
}

# already_have <dir> : true if a LeRobot dataset already sits there (meta/info.json)
already_have() {
    [[ -f "$1/meta/info.json" ]]
}

# download a HF dataset repo into a local dir, idempotently
fetch_dataset() {
    local repo="$1" dir="$2"
    if already_have "$dir"; then
        log "SKIP $repo — already present at $dir (meta/info.json found)"
        return 0
    fi
    log "Downloading dataset $repo -> $dir"
    hf download "$repo" --repo-type dataset --local-dir "$dir"
}

# ---- arg parse -------------------------------------------------------------
MODE="joint"
case "${1:-joint}" in
    -h|--help|help) usage ;;
    joint)          MODE="joint" ;;
    all)            MODE="all" ;;
    "")             MODE="joint" ;;
    *)              die "unknown argument '$1' (expected: joint | all | --help)" ;;
esac

need_hf

# =============================================================================
# 1. JOINT dataset (public) — always
# =============================================================================
fetch_dataset "Bigenlight/banana_in_pot_lerobot_v3" "$JOINT_DIR"

# =============================================================================
# 2. Optional: pretrained ACT model (4.7 GB) — skip re-training
# =============================================================================
if [[ "$FETCH_ACT_MODEL" == "1" ]]; then
    if [[ -d "$ACT_DIR" && -n "$(ls -A "$ACT_DIR" 2>/dev/null)" ]]; then
        log "SKIP ACT model — already present at $ACT_DIR"
    else
        log "Downloading pretrained ACT model -> $ACT_DIR"
        hf download Bigenlight/act_banana_in_pot --local-dir "$ACT_DIR"
    fi
else
    log "SKIP pretrained ACT model (set FETCH_ACT_MODEL=1 to fetch, or re-train)"
fi

# =============================================================================
# 3. Optional: EE-obs dataset (public; NOT needed for the two diffusion runs)
# =============================================================================
if [[ "$FETCH_EE_OBS" == "1" ]]; then
    fetch_dataset "Bigenlight/banana_in_pot_ee_lerobot_v3" "$EE_OBS_DIR"
fi

# =============================================================================
# 4. MODE=all : raw source + rebuild EE-action + validate
# =============================================================================
if [[ "$MODE" == "all" ]]; then

    # 4a. raw h5+mp4 (745 MB) — needed only for the rebuild
    fetch_dataset "Bigenlight/banana_in_pot_raw" "$RAW_DIR"

    # 4b. rebuild the LOCAL-ONLY EE-action dataset (video-reuse from JOINT).
    #     Requires JOINT + raw present (checked above).  CLI flags verified
    #     against convert_to_lerobot_ee_action.py / validate_ee_dataset.py.
    [[ -x "$PY" ]] || die "expected venv python at $PY — run setup.sh to build lr_env"

    if already_have "$EE_DIR"; then
        log "SKIP EE-action rebuild — already present at $EE_DIR (meta/info.json found)"
    else
        log "EE-action rebuild: self-test (transforms only, no data)"
        "$PY" convert_to_lerobot_ee_action.py --selftest

        log "EE-action rebuild: building $EE_DIR (video-reuse, CPU-only)"
        "$PY" convert_to_lerobot_ee_action.py \
            --data "$RAW_DIR" \
            --source "$JOINT_DIR" \
            --out "$EE_DIR" \
            --repo-id theo/banana_in_pot_ee_action

        log "EE-action rebuild: validating (expect 51 eps / 21524 frames)"
        "$PY" validate_ee_dataset.py \
            --root "$EE_DIR" \
            --repo-id theo/banana_in_pot_ee_action \
            --data "$RAW_DIR"
    fi
fi

# =============================================================================
# 5. Post-download sanity check
# =============================================================================
log "Sanity check"
already_have "$JOINT_DIR" \
    || die "$JOINT_DIR/meta/info.json missing — JOINT download did not complete"
echo "  OK  $JOINT_DIR/meta/info.json"

if [[ "$MODE" == "all" ]]; then
    already_have "$EE_DIR" \
        || die "$EE_DIR/meta/info.json missing — EE-action rebuild did not complete"
    echo "  OK  $EE_DIR/meta/info.json"
fi

log "Done (mode=$MODE)."
