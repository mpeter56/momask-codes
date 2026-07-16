"""
simple_blend.py — Dead-simple identity-preserving spectrum blend.

For every dance value d in the 10-step spectrum:
    blended = SRC_WEIGHT * source + (1 - SRC_WEIGHT) * target(d)

Where source is dance_mode_0.09 (the walking baseline) and target is the
MoMask output at intensity d. Because source stays fixed and target changes,
the SAME walking motion is always present, and the dance style progressively
mixes in as d rises.

No neural network. No time-warping. Just a linear blend of joint positions.

Usage:
    conda activate momask
    python simple_blend.py                              # 10 default STEPS
    python simple_blend.py --src-weight 0.7             # more identity, less dance
    python simple_blend.py --dance-values 0.09,0.5,1.00 # a subset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

HOME = Path.home()
MOMASK_REPO   = HOME / "Downloads" / "momask-codes-main"
MINI_SCAFFOLD = HOME / "Downloads" / "momask-codes-main" / "identity_preservation"
sys.path.insert(0, str(MOMASK_REPO))

from utils.plot_script import plot_3d_motion               # noqa: E402
from utils.paramUtil import t2m_kinematic_chain            # noqa: E402


TED_STEPS = [0.09, 0.11, 0.28, 0.39, 0.40, 0.51, 0.63, 0.74, 0.80, 1.00]
NUM_FRAMES = 196


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--editing-dir", default=str(MOMASK_REPO / "editing"))
    ap.add_argument("--output-dir",  default=str(MINI_SCAFFOLD / "outputs" / "pipeline_output"))
    ap.add_argument("--dance-values", default=",".join(f"{d:.2f}" for d in TED_STEPS))
    ap.add_argument("--src-weight", type=float, default=0.6,
                    help="how much of the walking source to keep (0..1). 0.6 = 60%% walk + 40%% dance")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--skip-render", action="store_true")
    return ap.parse_args()


def find_joint_npy(editing_dir: Path, dance: float) -> Optional[Path]:
    tag = f"{dance:.2f}"; tag_pct = str(int(round(dance * 100)))
    for base in (f"dance_mode_{tag}", f"run_d{tag}"):
        for sub in ("joints", "animations"):
            p = editing_dir / base / sub / "0" / "sample0_repeat0_len196.npy"
            if p.exists():
                return p
    for sub in ("joints", "animations"):
        for pattern in (f"*{tag}*/**/{sub}/**/*.npy",
                        f"*d {tag_pct}/{sub}/**/*.npy",
                        f"*d_{tag_pct}/{sub}/**/*.npy"):
            hits = sorted(editing_dir.glob(pattern))
            raw = [h for h in hits if "_ik" not in h.name]
            if raw:   return raw[0]
            if hits:  return hits[0]
    return None


def standardize(arr: np.ndarray, T: int = NUM_FRAMES) -> np.ndarray:
    a = np.asarray(arr).astype(np.float32)
    if a.ndim == 4:
        a = a[0]
    if a.shape[0] > T:
        a = a[:T]
    elif a.shape[0] < T:
        pad = np.repeat(a[-1:], T - a.shape[0], axis=0)
        a = np.concatenate([a, pad], axis=0)
    return a


def main() -> int:
    args = parse_args()
    editing_dir = Path(args.editing_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dance_values = [float(x) for x in args.dance_values.split(",")]
    w_src = float(args.src_weight)
    w_gen = 1.0 - w_src

    source_d = min(dance_values)
    source_path = find_joint_npy(editing_dir, source_d)
    if source_path is None:
        raise SystemExit(f"[fatal] no source joint file for d={source_d:.2f}")
    print(f"[blend] SOURCE (walking baseline, d={source_d:.2f}): {source_path}")
    source = standardize(np.load(source_path))
    print(f"[blend] {w_src*100:.0f}% source + {w_gen*100:.0f}% target — source movement always dominates")

    for d in dance_values:
        target_path = find_joint_npy(editing_dir, d)
        if target_path is None:
            print(f"  d={d:.2f}  MISSING target joint file — skipping")
            continue
        target = standardize(np.load(target_path))

        blended = (w_src * source + w_gen * target).astype(np.float32)

        npy_out = output_dir / f"blended_d{d:.2f}.npy"
        np.save(npy_out, blended)

        if not args.skip_render:
            mp4_out = output_dir / f"identity_spectrum_d{d:.2f}.mp4"
            plot_3d_motion(
                str(mp4_out),
                t2m_kinematic_chain,
                blended,
                title=f"d={d:.2f}  ·  {w_src*100:.0f}% walk + {w_gen*100:.0f}% dance",
                fps=args.fps,
            )
            print(f"  d={d:.2f}  ->  {npy_out.name}   {mp4_out.name}")
        else:
            print(f"  d={d:.2f}  ->  {npy_out.name}")

    print(f"\n[blend] done. {len(dance_values)} spectrum videos in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
