#!/usr/bin/env bash
# ============================================================================
# run_spectrum_tour.sh — end-to-end Spectrum Tour render.
#
# Steps:
#   1. Ensure blended npys exist for all 10 dance values (calls
#      run_blend_and_render.py — needs 'momask' conda env with torch)
#   2. Render the tour piece (calls render_spectrum_tour.py)
#   3. Optionally prepend the original input.mp4 as an opening shot (ffmpeg)
#
# Usage:
#   bash run_spectrum_tour.sh                    # 1080p, no input.mp4 prepended
#   bash run_spectrum_tour.sh --preview          # 720p (faster iteration)
#   bash run_spectrum_tour.sh --with-input       # prepend the source video
#   bash run_spectrum_tour.sh --preview --with-input
# ============================================================================
set -euo pipefail

SCRIPTS="$HOME/Downloads/momask-codes-main/identity_preservation/scripts"
BLEND_DIR="$HOME/Downloads/momask-codes-main/identity_preservation/outputs/pipeline_output"
OUT_DIR="$HOME/Downloads/momask-codes-main/identity_preservation/outputs/media_art"
INPUT_MP4="$HOME/Downloads/momask-codes-main/sessions/20260623_150730/input.mp4"

DANCE_VALUES="0.09,0.23,0.28,0.40,0.42,0.56,0.70,0.74,0.97,1.00"
WIDTH=1920
HEIGHT=1080
FPS=20
WITH_INPUT=0
ZOOM=2.0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preview)       WIDTH=1280; HEIGHT=720; shift ;;
        --width)         WIDTH="$2"; shift 2 ;;
        --height)        HEIGHT="$2"; shift 2 ;;
        --fps)           FPS="$2"; shift 2 ;;
        --with-input)    WITH_INPUT=1; shift ;;
        --dance-values)  DANCE_VALUES="$2"; shift 2 ;;
        --zoom)          ZOOM="$2"; shift 2 ;;
        *) echo "unknown flag: $1"; exit 1 ;;
    esac
done

log() { printf "\033[36m[%s]\033[0m %s\n" "$(date +%H:%M:%S)" "$*"; }
ok()  { printf "\033[32m[%s] OK\033[0m %s\n" "$(date +%H:%M:%S)" "$*"; }

# Activate momask env if not already active
if [[ "${CONDA_DEFAULT_ENV:-}" != "momask" ]]; then
    log "activating conda env 'momask'"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate momask
fi

mkdir -p "$OUT_DIR"

# ---------- stage 1: ensure blends exist for all 10 values ------------------
log "stage 1/3  blending all 10 dance values through MiniGainNet"
python "$SCRIPTS/run_blend_and_render.py" \
    --dance-values "$DANCE_VALUES" \
    --output-dir "$BLEND_DIR" \
    --fps "$FPS"
ok "blends ready"

# ---------- stage 2: render the tour ---------------------------------------
TOUR_MP4="$OUT_DIR/spectrum_tour_${WIDTH}x${HEIGHT}.mp4"
log "stage 2/3  rendering tour  ->  $TOUR_MP4"
python "$SCRIPTS/render_spectrum_tour.py" \
    --blend-dir "$BLEND_DIR" \
    --out "$TOUR_MP4" \
    --dance-values "$DANCE_VALUES" \
    --width "$WIDTH" --height "$HEIGHT" --fps "$FPS" \
    --zoom "$ZOOM"
ok "tour rendered"

# ---------- stage 3: (optional) prepend input.mp4 --------------------------
FINAL_MP4="$TOUR_MP4"
if [[ "$WITH_INPUT" -eq 1 ]]; then
    if [[ ! -f "$INPUT_MP4" ]]; then
        log "WARN: input.mp4 not found at $INPUT_MP4 — skipping prepend"
    else
        log "stage 3/3  prepending source input.mp4"
        FINAL_MP4="$OUT_DIR/spectrum_tour_with_input_${WIDTH}x${HEIGHT}.mp4"

        # Re-encode input.mp4 to match tour resolution/fps
        INPUT_NORMALIZED="$OUT_DIR/_input_normalized.mp4"
        ffmpeg -y -loglevel error \
            -i "$INPUT_MP4" \
            -vf "scale=${WIDTH}:${HEIGHT}:force_original_aspect_ratio=decrease,pad=${WIDTH}:${HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black,fps=${FPS}" \
            -c:v libx264 -pix_fmt yuv420p -crf 18 \
            "$INPUT_NORMALIZED"

        # Concat via ffmpeg concat demuxer
        CONCAT_LIST="$OUT_DIR/_concat.txt"
        printf "file '%s'\nfile '%s'\n" "$INPUT_NORMALIZED" "$TOUR_MP4" > "$CONCAT_LIST"
        ffmpeg -y -loglevel error \
            -f concat -safe 0 -i "$CONCAT_LIST" \
            -c copy \
            "$FINAL_MP4"

        rm -f "$INPUT_NORMALIZED" "$CONCAT_LIST"
        ok "concatenated with input.mp4"
    fi
fi

ok "SPECTRUM TOUR complete"
echo ""
echo "==== OUTPUT ===="
ls -la "$FINAL_MP4"
