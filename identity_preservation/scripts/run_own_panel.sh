#!/usr/bin/env bash
# ============================================================================
# run_own_panel.sh — Build your own Unnoticed-Dance panel from a webcam clip.
#
# Steps:
#   1. Extract MediaPipe pose landmarks from the source video → pose_data.json
#   2. Copy the video + HTML page into outputs/own_panel/
#   3. Start a local server + open the browser
#
# Usage:
#   bash run_own_panel.sh                          # auto-detect newest webcam mp4
#   bash run_own_panel.sh path/to/some_video.mp4   # explicit
# ============================================================================
set -euo pipefail

SCRIPTS="$HOME/Downloads/momask-codes-main/identity_preservation/scripts"
OUT_DIR="$HOME/Downloads/momask-codes-main/identity_preservation/outputs/own_panel"
PORT=8000

# ---------- pick the source video --------------------------------------------
if [[ $# -ge 1 ]]; then
    VIDEO="$1"
else
    VIDEO=$(ls -t "$HOME/Downloads/momask-codes-main/input_videos/"webcam*.mp4 2>/dev/null | head -1 || true)
    [[ -z "$VIDEO" ]] && { echo "No webcam*.mp4 in ~/Downloads/momask-codes-main/input_videos/. Pass an explicit path."; exit 1; }
fi
echo "[panel] using video: $VIDEO"

# ---------- conda env --------------------------------------------------------
if [[ "${CONDA_DEFAULT_ENV:-}" != "momask" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate momask
fi

# ---------- MediaPipe check --------------------------------------------------
if ! python -c "import mediapipe" 2>/dev/null; then
    echo "[panel] installing mediapipe (one-time)…"
    pip install mediapipe
fi

mkdir -p "$OUT_DIR"

# ---------- 1. extract pose json --------------------------------------------
python "$SCRIPTS/extract_pose_json.py" \
    --video "$VIDEO" \
    --out   "$OUT_DIR/pose_data.json"

# ---------- 2. stage the html + video ---------------------------------------
cp -f "$SCRIPTS/own_panel.html" "$OUT_DIR/index.html"
cp -f "$VIDEO"                  "$OUT_DIR/webcam.mp4"

# ---------- 3. serve ---------------------------------------------------------
echo ""
echo "[panel] starting local server at http://localhost:$PORT"
echo "        Ctrl+C to stop."
cd "$OUT_DIR"
open "http://localhost:$PORT" || true
python -m http.server "$PORT"
