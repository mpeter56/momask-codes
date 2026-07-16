#!/usr/bin/env bash
# ============================================================================
# pipeline.sh — identity-preservation full pipeline, end-to-end.
#
# Given per-dance-value joint arrays that already exist under editing/run_d*/,
# runs the full chain:
#
#   1. blend    : MiniGainNet blend + stick-figure MP4 (Ted's blue skeleton)
#   2. bvh      : joint arrays -> BVH via IK (the Joint2BVHConvertor)
#   3. avatar   : BVH -> retarget onto Y Bot / X Bot -> Blender-rendered MP4
#
# Usage:
#   bash pipeline.sh                       # run all stages
#   bash pipeline.sh --stage blend         # only stage 1
#   bash pipeline.sh --stage avatar        # only stage 3 (assumes 1+2 done)
#   bash pipeline.sh --character "X Bot"   # swap avatar
#   bash pipeline.sh --dance 0.40,1.00     # only render these values
#   bash pipeline.sh --dry-run             # print what would run, do nothing
#
# Requires:
#   - conda env 'momask' (activated by this script)
#   - Blender at /Applications/Blender.app/Contents/MacOS/Blender
#   - MiniGainNet checkpoint at ~/identity_preservation_mini/outputs/checkpoints/
#   - Joint .npy files at ~/Downloads/momask-codes-main/editing/run_d{X.XX}/joints/0/
# ============================================================================
set -euo pipefail

# ---------- config ----------------------------------------------------------
PROJ_DIR="$HOME/Downloads/momask-codes-main/identity_preservation"
SCRIPTS="$PROJ_DIR/scripts"
MOMASK_REPO="$HOME/Downloads/momask-codes-main"
CHECKPOINT="$HOME/identity_preservation_mini/outputs/checkpoints/mini_gain_net_final.pt"
EDITING_DIR="$MOMASK_REPO/editing"
# For Rokoko retargeting you need Blender 4.2 LTS (Rokoko 1.4.3 doesn't
# support the Slotted Actions API in Blender 4.4+/5.x). Set BLENDER_ROKOKO
# to your 4.2 install; leave BLENDER as your default for the copyrot path.
BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"
BLENDER_ROKOKO="/Applications/Blender 4.2.app/Contents/MacOS/Blender"

OUT_ROOT="$PROJ_DIR/outputs/pipeline_output"
BVH_DIR="$OUT_ROOT/bvh"
AVATAR_DIR="$OUT_ROOT/avatar_mp4"
LOG_DIR="$OUT_ROOT/logs"

# defaults (override via flags)
STAGE="all"
CHARACTER_NAME="Y Bot"
DANCE_VALUES=(0.09 0.40 0.74 1.00)
DRY_RUN=0
RENDER_W=720
RENDER_H=720
FPS=20
VIEW="front"
ZOOM=3.5
RETARGET="rokoko"   # rokoko | copyrot

# ---------- args ------------------------------------------------------------
usage() {
    sed -n '/^# ===/,/^# ===/{/^#/p}' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)      STAGE="$2"; shift 2 ;;
        --character)  CHARACTER_NAME="$2"; shift 2 ;;
        --dance)      IFS=',' read -ra DANCE_VALUES <<< "$2"; shift 2 ;;
        --dry-run)    DRY_RUN=1; shift ;;
        --width)      RENDER_W="$2"; shift 2 ;;
        --height)     RENDER_H="$2"; shift 2 ;;
        --fps)        FPS="$2"; shift 2 ;;
        --view)       VIEW="$2"; shift 2 ;;
        --zoom)       ZOOM="$2"; shift 2 ;;
        --retarget)   RETARGET="$2"; shift 2 ;;
        -h|--help)    usage ;;
        *)            echo "Unknown flag: $1"; usage 1 ;;
    esac
done

CHARACTER_FBX="$HOME/Downloads/${CHARACTER_NAME}.fbx"

# ---------- pretty logging --------------------------------------------------
if [[ -t 1 ]]; then
    C_RESET='\033[0m'; C_BOLD='\033[1m'
    C_INFO='\033[36m'; C_OK='\033[32m'; C_WARN='\033[33m'; C_ERR='\033[31m'
else
    C_RESET=''; C_BOLD=''; C_INFO=''; C_OK=''; C_WARN=''; C_ERR=''
fi
log()  { printf "${C_INFO}[%s]${C_RESET} %s\n" "$(date +%H:%M:%S)" "$*"; }
ok()   { printf "${C_OK}[%s] OK${C_RESET} %s\n"   "$(date +%H:%M:%S)" "$*"; }
warn() { printf "${C_WARN}[%s] WARN${C_RESET} %s\n" "$(date +%H:%M:%S)" "$*"; }
err()  { printf "${C_ERR}[%s] ERROR${C_RESET} %s\n" "$(date +%H:%M:%S)" "$*" >&2; exit 1; }
banner() { printf "\n${C_BOLD}%s${C_RESET}\n" "==== $* ===="; }
run() {
    if (( DRY_RUN )); then log "DRY: $*"; else "$@"; fi
}

# ---------- preflight -------------------------------------------------------
preflight() {
    banner "PREFLIGHT"

    # conda env
    if [[ "${CONDA_DEFAULT_ENV:-}" != "momask" ]]; then
        warn "conda env is '${CONDA_DEFAULT_ENV:-none}', not 'momask'"
        # try to activate
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || err "conda not available"
        conda activate momask || err "could not activate conda env 'momask'"
    fi
    ok "conda env: $CONDA_DEFAULT_ENV  ($(which python))"

    # required paths
    [[ -f "$CHECKPOINT" ]]        || err "checkpoint missing: $CHECKPOINT"
    [[ -d "$MOMASK_REPO" ]]       || err "momask repo missing: $MOMASK_REPO"
    [[ -d "$EDITING_DIR" ]]       || err "editing dir missing: $EDITING_DIR"
    [[ -f "$CHARACTER_FBX" ]]     || err "character FBX missing: $CHARACTER_FBX"
    [[ -x "$BLENDER" ]]           || err "Blender not at $BLENDER"
    ok "checkpoint       : $CHECKPOINT"
    ok "character        : $CHARACTER_FBX"
    ok "blender          : $BLENDER"

    # joint files for each dance value
    for d in "${DANCE_VALUES[@]}"; do
        local npy="$EDITING_DIR/run_d${d}/joints/0/sample0_repeat0_len196.npy"
        if [[ -f "$npy" ]]; then
            ok "  d=${d}  joint npy present"
        else
            # allow loose glob (run_d1.00 might not exist in exact form)
            local hits
            hits=$(find "$EDITING_DIR" -type f -name "sample0_repeat0_len196.npy" \
                        -path "*${d}*" 2>/dev/null | head -1 || true)
            [[ -n "$hits" ]] && ok "  d=${d}  joint npy found: $hits" \
                             || err "no joint npy for d=$d anywhere under $EDITING_DIR"
        fi
    done

    mkdir -p "$OUT_ROOT" "$BVH_DIR" "$AVATAR_DIR" "$LOG_DIR"
    ok "output dirs prepared under $OUT_ROOT"
}

# ---------- stage 1 : blend + stick-figure render ---------------------------
stage_blend() {
    banner "STAGE 1: blend + stick-figure render"
    local logf="$LOG_DIR/1_blend_$(date +%H%M%S).log"
    run python "$SCRIPTS/run_blend_and_render.py" \
        --dance-values "$(IFS=,; echo "${DANCE_VALUES[*]}")" \
        --fps "$FPS" \
        2>&1 | tee "$logf"
    ok "stage 1 done  (log: $logf)"
}

# ---------- stage 2 : joints -> BVH -----------------------------------------
stage_bvh() {
    banner "STAGE 2: joints -> BVH (IK)"
    local logf="$LOG_DIR/2_bvh_$(date +%H%M%S).log"
    run python "$SCRIPTS/joints_to_bvh.py" 2>&1 | tee "$logf"
    ok "stage 2 done  (log: $logf)"
}

# ---------- rokoko plugin auto-download -------------------------------------
ensure_rokoko_zip() {
    local zip
    zip=$(ls "$HOME/Downloads"/rokoko-studio-live-blender*.zip 2>/dev/null | head -1)
    if [[ -n "$zip" ]]; then
        echo "$zip"
        return 0
    fi
    log "downloading Rokoko Studio Live plugin from GitHub releases..."
    local url
    url=$(curl -s https://api.github.com/repos/Rokoko/rokoko-studio-live-blender/releases/latest \
          | python -c 'import json,sys; d=json.load(sys.stdin); print(next(a["browser_download_url"] for a in d["assets"] if a["browser_download_url"].endswith(".zip")))' \
          2>/dev/null)
    if [[ -z "$url" ]]; then
        warn "could not resolve Rokoko download URL — please download manually to ~/Downloads/"
        return 1
    fi
    zip="$HOME/Downloads/$(basename "$url")"
    curl -sL -o "$zip" "$url" || { warn "download failed"; return 1; }
    ok "downloaded: $zip"
    echo "$zip"
}

# ---------- stage 3 : BVH -> Blender avatar render --------------------------
stage_avatar() {
    banner "STAGE 3: Blender avatar render  (retarget=$RETARGET)"

    local script rokoko_zip="" blender_bin
    if [[ "$RETARGET" == "rokoko" ]]; then
        script="$SCRIPTS/rokoko_render_avatar.py"
        rokoko_zip=$(ensure_rokoko_zip || true)
        if [[ -x "$BLENDER_ROKOKO" ]]; then
            blender_bin="$BLENDER_ROKOKO"
            log "using Blender 4.2 for Rokoko: $blender_bin"
        else
            err "Rokoko needs Blender 4.2 LTS. Install to '$BLENDER_ROKOKO' or run --retarget copyrot"
        fi
    else
        script="$SCRIPTS/blender_render_avatar.py"
        blender_bin="$BLENDER"
    fi

    for d in "${DANCE_VALUES[@]}"; do
        local bvh="$BVH_DIR/blended_d${d}.bvh"
        local out="$AVATAR_DIR/avatar_blended_d${d}.mp4"
        local logf="$LOG_DIR/3_avatar_d${d}_$(date +%H%M%S).log"

        if [[ ! -f "$bvh" ]]; then
            warn "d=$d  no BVH at $bvh — skip (run stage bvh first)"
            continue
        fi
        log "d=${d}  rendering  ->  $out"
        if [[ "$RETARGET" == "rokoko" ]]; then
            run "$blender_bin" -b \
                --python "$script" -- \
                --fbx "$CHARACTER_FBX" \
                --bvh "$bvh" \
                --out "$out" \
                --rokoko-zip "$rokoko_zip" \
                --fps "$FPS" \
                --width "$RENDER_W" --height "$RENDER_H" \
                --view "$VIEW" --zoom "$ZOOM" \
                > "$logf" 2>&1
        else
            run "$blender_bin" -b \
                --python "$script" -- \
                --fbx "$CHARACTER_FBX" \
                --bvh "$bvh" \
                --out "$out" \
                --fps "$FPS" \
                --width "$RENDER_W" --height "$RENDER_H" \
                --view "$VIEW" --zoom "$ZOOM" \
                > "$logf" 2>&1
        fi
        grep -E "\[rokoko\]|\[mapping\]|\[render\]|\[camera\]|\[done\]|resolved|MISSING|FATAL|Traceback|Error:|^  File " "$logf" | sed 's/^/    /' || true
        # Only claim OK if the target MP4 was actually produced
        if [[ -f "$out" ]] && [[ "$(stat -f %m "$out" 2>/dev/null || stat -c %Y "$out")" -gt "$(date +%s -d '5 minutes ago' 2>/dev/null || date -v -5M +%s)" ]]; then
            ok "d=${d} avatar rendered"
        else
            warn "d=${d} avatar NOT produced (file missing or stale) — see $logf"
        fi
    done
}

# ---------- final summary ---------------------------------------------------
summary() {
    banner "SUMMARY"
    printf "%-8s  %-10s  %s\n" "dance" "type" "path"
    printf "%-8s  %-10s  %s\n" "-----" "----" "----"
    for d in "${DANCE_VALUES[@]}"; do
        local stick="$OUT_ROOT/blended_d${d}.mp4"
        local bvh="$BVH_DIR/blended_d${d}.bvh"
        local avatar="$AVATAR_DIR/avatar_blended_d${d}.mp4"
        [[ -f "$stick"  ]] && printf "%-8s  ${C_OK}%-10s${C_RESET}  %s\n" "$d" "stick"  "$stick" \
                          || printf "%-8s  ${C_ERR}%-10s${C_RESET}  MISSING\n" "$d" "stick"
        [[ -f "$bvh"    ]] && printf "%-8s  ${C_OK}%-10s${C_RESET}  %s\n" "$d" "bvh"    "$bvh" \
                          || printf "%-8s  ${C_ERR}%-10s${C_RESET}  MISSING\n" "$d" "bvh"
        [[ -f "$avatar" ]] && printf "%-8s  ${C_OK}%-10s${C_RESET}  %s\n" "$d" "avatar" "$avatar" \
                          || printf "%-8s  ${C_ERR}%-10s${C_RESET}  MISSING\n" "$d" "avatar"
    done
    echo
    log "logs -> $LOG_DIR/"
}

# ---------- main ------------------------------------------------------------
preflight

case "$STAGE" in
    all)      stage_blend; stage_bvh; stage_avatar; summary ;;
    blend)    stage_blend; summary ;;
    bvh)      stage_bvh; summary ;;
    avatar)   stage_avatar; summary ;;
    summary)  summary ;;
    *)        err "unknown stage: $STAGE (use: all | blend | bvh | avatar | summary)" ;;
esac
