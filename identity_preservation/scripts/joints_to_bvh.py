"""
joints_to_bvh.py — Convert blended (T, 22, 3) joint npy files to .bvh via Ted's
Joint2BVHConvertor (from momask-codes-main/visualization/joints2bvh.py).

Output BVH files can be imported into Blender / Maya / Unity / Unreal and
retargeted onto any rigged humanoid (e.g. character.fbx in Downloads).

Usage:
    conda activate momask
    python "/Users/neek526/Downloads/momask-codes-main/identity_preservation/scripts/joints_to_bvh.py"
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HOME = Path.home()
MOMASK_REPO = HOME / "Downloads" / "momask-codes-main"
BLENDED_DIR = HOME / "Downloads" / "momask-codes-main" / "identity_preservation" / "outputs" / "pipeline_output"
OUT_DIR     = BLENDED_DIR / "bvh"

# Joint2BVHConvertor loads './visualization/data/template.bvh' with a relative path,
# so we MUST cwd into the momask repo before instantiating it.
os.chdir(str(MOMASK_REPO))
sys.path.insert(0, str(MOMASK_REPO))

from visualization.joints2bvh import Joint2BVHConvertor  # noqa: E402


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    convertor = Joint2BVHConvertor()

    dance_values = [0.09, 0.40, 0.74, 1.00]
    for d in dance_values:
        npy_path = BLENDED_DIR / f"blended_d{d:.2f}.npy"
        bvh_path = OUT_DIR / f"blended_d{d:.2f}.bvh"

        if not npy_path.exists():
            print(f"  MISSING: {npy_path}")
            continue

        positions = np.load(npy_path).astype(np.float32)
        # standardize (should already be (196, 22, 3) from run_blend_and_render.py)
        if positions.ndim == 4:
            positions = positions[0]
        assert positions.shape[1:] == (22, 3), f"unexpected shape {positions.shape}"

        print(f"[d={d:.2f}] converting {positions.shape} -> {bvh_path.name}")
        # convert() runs IK with 10 iterations + foot-IK to remove sliding
        _anim, _glb = convertor.convert(positions, str(bvh_path), iterations=10, foot_ik=True)
        print(f"           saved: {bvh_path}")

    print("\n" + "=" * 60)
    print("SUMMARY — BVH files ready for Blender import")
    print("=" * 60)
    for d in dance_values:
        bvh = OUT_DIR / f"blended_d{d:.2f}.bvh"
        status = "OK" if bvh.exists() else "MISSING"
        print(f"  d={d:.2f}: {status}  {bvh}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
