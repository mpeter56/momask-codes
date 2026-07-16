#!/usr/bin/env bash
# ============================================================================
# run_chorus.sh — Spectrum Chorus media art piece, end-to-end.
#
# Pipeline:
#   1. Ensure blended npys exist for all 10 dance values
#      (calls run_blend_and_render.py with the full spectrum)
#   2. Render the chorus MP4 (calls render_chorus.py)
#
# Usage:
#   bash run_chorus.sh                    # full 1080p render (~10-20 min)
#   bash run_chorus.sh --preview          # 720p, faster iteration (~4-8 min)
#   bash run_chorus.sh --width 3840 --height 2160  # 4K
# ============================================================================
set -euo pipefail

SCRIPTS="$HOME/Downloads/momask-codes-main/identity_preservation/scripts"
BLEND_DIR="$HOME/Downloads/momask-codes-main/identity_preservation/outputs/pipeline_output"
OUT_DIR="$HOME/Downloads/momask-codes-main/identity_preservation/outputs/media_art"

DANCE_VALUES="0.09,0.23,0.28,0.40,0.42,0.56,0.70,0.74,0.97,1.00"
WIDTH=1920
HEIGHT=1080
FPS=20

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preview) WIDTH=1280; HEIGHT=720; shift ;;
        --width)   WIDTH="$2"; shift 2 ;;
        --height)  HEIGHT="$2"; shift 2 ;;
        --fps)     FPS="$2"; shift 2 ;;
        *) echo "unknown flag: $1"; exit 1 ;;
    esac
done

log() { printf "\033[36m[%s]\033[0m %s\n" "$(date +%H:%M:%S)" "$*"; }
ok()  { printf "\033[32m[%s] OK\033[0m %s\n" "$(date +%H:%M:%S)" "$*"; }

# Activate momask env if we're not already in it
if [[ "${CONDA_DEFAULT_ENV:-}" != "momask" ]]; then
    log "activating conda env 'momask'"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate momask
fi

mkdir -p "$OUT_DIR"

# ---------- stage 1: blend all 10 values ------------------------------------
log "stage 1/2  blending all 10 dance values through MiniGainNet"
python "$SCRIPTS/run_blend_and_render.py" \
    --dance-values "$DANCE_VALUES" \
    --output-dir "$BLEND_DIR" \
    --fps "$FPS"
ok "blends ready"

# ---------- stage 2: render the chorus --------------------------------------
OUT_MP4="$OUT_DIR/spectrum_chorus_${WIDTH}x${HEIGHT}.mp4"
log "stage 2/2  rendering chorus  ->  $OUT_MP4"
python "$SCRIPTS/render_chorus.py" \
    --blend-dir "$BLEND_DIR" \
    --out "$OUT_MP4" \
    --width "$WIDTH" --height "$HEIGHT" --fps "$FPS"

ok "SPECTRUM CHORUS complete"
echo ""
echo "==== OUTPUT ===="
ls -la "$OUT_MP4"
