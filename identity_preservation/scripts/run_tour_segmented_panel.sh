#!/usr/bin/env bash
# ============================================================================
# run_tour_segmented_panel.sh — Split spectrum_tour_1280x720.mp4 into 10
# per-intensity segments, put the raw webcam alongside, serve as a panel.
#
# The tour layout (from render_spectrum_tour.py) is:
#     10 clips × 196 frames + 9 crossfades × 5 frames  = 2005 frames @ 20 fps
# Each intensity block starts at frame  i * 201  and lasts 196 frames (9.80 s).
#
# Usage:
#   bash run_tour_segmented_panel.sh                                          # auto
#   bash run_tour_segmented_panel.sh path/to/spectrum_tour.mp4 path/to/webcam.mp4
# ============================================================================
set -euo pipefail

SCRIPTS="$HOME/Downloads/momask-codes-main/identity_preservation/scripts"
MEDIA_DIR="$HOME/Downloads/momask-codes-main/identity_preservation/outputs/media_art"
OUT_DIR="$HOME/Downloads/momask-codes-main/identity_preservation/outputs/tour_segmented_panel"
PORT=8000

# ---------- pick tour + webcam ---------------------------------------------
if [[ $# -ge 1 ]]; then
    TOUR="$1"
else
    TOUR=""
    for candidate in \
        "$MEDIA_DIR/spectrum_tour_1280x720.mp4" \
        "$MEDIA_DIR/spectrum_tour_1920x1080.mp4"; do
        [[ -f "$candidate" ]] && { TOUR="$candidate"; break; }
    done
    [[ -z "$TOUR" ]] && { echo "No spectrum_tour_*.mp4 found in $MEDIA_DIR"; exit 1; }
fi
echo "[panel] tour: $TOUR"

if [[ $# -ge 2 ]]; then
    WEBCAM="$2"
else
    WEBCAM=$(ls -t "$HOME/Downloads/momask-codes-main/input_videos/"webcam*.mp4 2>/dev/null | head -1 || true)
    [[ -z "$WEBCAM" ]] && { echo "No webcam*.mp4 found. Pass an explicit path as 2nd arg."; exit 1; }
fi
echo "[panel] webcam: $WEBCAM"

mkdir -p "$OUT_DIR"

# ---------- split the tour --------------------------------------------------
# Each intensity block: start at t = i * (201/20) = i * 10.05s, duration 9.80s.
DANCE_VALUES=(0.09 0.11 0.28 0.39 0.40 0.51 0.63 0.74 0.80 1.00)

for i in "${!DANCE_VALUES[@]}"; do
    d="${DANCE_VALUES[$i]}"
    START=$(python3 -c "print(round(${i} * 201 / 20.0, 3))")
    DUR=$(python3 -c "print(round(196 / 20.0, 3))")
    OUT="$OUT_DIR/segment_d${d}.mp4"
    echo "  segment d=${d}   start=${START}s   duration=${DUR}s   -> $(basename "$OUT")"
    ffmpeg -y -loglevel error \
        -ss "$START" -t "$DUR" -i "$TOUR" \
        -c:v libx264 -pix_fmt yuv420p -crf 20 \
        -movflags +faststart \
        "$OUT"
done

# ---------- link webcam + html ----------------------------------------------
ln -sf "$WEBCAM" "$OUT_DIR/webcam.mp4"
cp -f "$SCRIPTS/tour_segmented_panel.html" "$OUT_DIR/index.html"

# ---------- serve -----------------------------------------------------------
echo ""
echo "[panel] starting server at http://localhost:$PORT"
echo "        Ctrl+C to stop."
cd "$OUT_DIR"
open "http://localhost:$PORT" || true
python -m http.server "$PORT"
