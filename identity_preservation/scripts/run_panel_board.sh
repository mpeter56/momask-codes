#!/usr/bin/env bash
# ============================================================================
# run_panel_board.sh — generate the Unnoticed-Dance-style spectrum dashboard,
# start a local HTTP server, and open it in the default browser.
#
# Usage:
#   bash run_panel_board.sh                                              # auto-detect newest webcam
#   bash run_panel_board.sh --webcam ~/Downloads/momask-codes-main/input_videos/webcam_20260708_202958.mp4
# ============================================================================
set -euo pipefail

SCRIPTS="$HOME/Downloads/momask-codes-main/identity_preservation/scripts"
OUT_DIR="$HOME/Downloads/momask-codes-main/identity_preservation/outputs/panel_board"
PORT=8000
WEBCAM_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --webcam) WEBCAM_ARG="--webcam $2"; shift 2 ;;
        --port)   PORT="$2"; shift 2 ;;
        *) echo "unknown flag: $1"; exit 1 ;;
    esac
done

if [[ "${CONDA_DEFAULT_ENV:-}" != "momask" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate momask
fi

python "$SCRIPTS/generate_panel_board.py" $WEBCAM_ARG

echo ""
echo "==== starting local server on http://localhost:$PORT ===="
echo "Ctrl+C to stop."
cd "$OUT_DIR"
open "http://localhost:$PORT"
python -m http.server "$PORT"
