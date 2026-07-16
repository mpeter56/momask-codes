#!/usr/bin/env bash
# Batch-render all 4 dance-value BVHs onto character.fbx via headless Blender.
#
# Usage:
#   bash "/Users/neek526/Downloads/momask-codes-main/identity_preservation/scripts/render_all_avatars.sh"
#
# One-time introspection (dump character bones to figure out naming):
#   bash "/Users/neek526/Downloads/momask-codes-main/identity_preservation/scripts/render_all_avatars.sh" introspect

set -euo pipefail

BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"
SCRIPT="/Users/neek526/Downloads/momask-codes-main/identity_preservation/scripts/blender_render_avatar.py"
FBX="$HOME/Downloads/Y Bot.fbx"   # switch to "X Bot.fbx" if you prefer the robot look
BVH_DIR="/Users/neek526/Downloads/momask-codes-main/identity_preservation/outputs/pipeline_output/bvh"
OUT_DIR="/Users/neek526/Downloads/momask-codes-main/identity_preservation/outputs/pipeline_output/avatar_mp4"

mkdir -p "$OUT_DIR"

if [[ ! -x "$BLENDER" ]]; then
    echo "ERROR: Blender not found at $BLENDER"
    echo "Install from https://www.blender.org/download/ or update the BLENDER var above."
    exit 1
fi

if [[ "${1:-}" == "introspect" ]]; then
    echo "=== Introspecting character.fbx (dumps bone names) ==="
    "$BLENDER" -b --python "$SCRIPT" -- --fbx "$FBX" --introspect 2>&1 | grep -E "\[fbx\]|BONES|  |introspect"
    exit 0
fi

for d in 0.09 0.40 0.74 1.00; do
    BVH="$BVH_DIR/blended_d${d}.bvh"
    OUT="$OUT_DIR/avatar_blended_d${d}.mp4"
    if [[ ! -f "$BVH" ]]; then
        echo "SKIP d=$d — no BVH at $BVH"
        continue
    fi
    echo ""
    echo "=========================================="
    echo "Rendering d=$d  ->  $OUT"
    echo "=========================================="
    "$BLENDER" -b --python "$SCRIPT" -- \
        --fbx "$FBX" \
        --bvh "$BVH" \
        --out "$OUT" \
        --fps 20 --width 720 --height 720
done

echo ""
echo "=== ALL DONE ==="
ls -la "$OUT_DIR"
