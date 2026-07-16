#!/usr/bin/env bash
# ============================================================================
# run_spectrum_panel.sh — Panel dashboard for the 10 identity_spectrum videos
# + the raw webcam. Symlinks everything into outputs/spectrum_panel/ and
# starts a local HTTP server.
#
# Usage:
#   bash run_spectrum_panel.sh                                    # auto-detect newest webcam
#   bash run_spectrum_panel.sh path/to/some_webcam.mp4
# ============================================================================
set -euo pipefail

SCRIPTS="$HOME/Downloads/momask-codes-main/identity_preservation/scripts"
BLEND_DIR="$HOME/Downloads/momask-codes-main/identity_preservation/outputs/pipeline_output"
OUT_DIR="$HOME/Downloads/momask-codes-main/identity_preservation/outputs/spectrum_panel"
PORT=8000

# ---------- source webcam ---------------------------------------------------
if [[ $# -ge 1 ]]; then
    WEBCAM="$1"
else
    WEBCAM=$(ls -t "$HOME/Downloads/momask-codes-main/input_videos/"webcam*.mp4 2>/dev/null | head -1 || true)
    [[ -z "$WEBCAM" ]] && { echo "No webcam*.mp4 found. Pass an explicit path."; exit 1; }
fi
echo "[panel] source webcam: $WEBCAM"

mkdir -p "$OUT_DIR"

# ---------- link the webcam + 10 spectrum mp4s ------------------------------
ln -sf "$WEBCAM" "$OUT_DIR/webcam.mp4"

MISSING=0
for d in 0.09 0.11 0.28 0.39 0.40 0.51 0.63 0.74 0.80 1.00; do
    SRC="$BLEND_DIR/identity_spectrum_d${d}.mp4"
    if [[ -f "$SRC" ]]; then
        ln -sf "$SRC" "$OUT_DIR/identity_spectrum_d${d}.mp4"
        echo "  d=$d  linked"
    else
        echo "  d=$d  MISSING — the panel will show a broken video for this one"
        echo "         (run webcam_spectrum.py to generate it)"
        MISSING=$((MISSING + 1))
    fi
done

# ---------- copy html -------------------------------------------------------
cp -f "$SCRIPTS/spectrum_panel.html" "$OUT_DIR/index.html"

[[ $MISSING -gt 0 ]] && echo "[panel] warning: $MISSING spectrum videos missing"

# ---------- serve -----------------------------------------------------------
echo ""
echo "[panel] starting server at http://localhost:$PORT"
echo "        Ctrl+C to stop."
cd "$OUT_DIR"
open "http://localhost:$PORT" || true
python -m http.server "$PORT"
