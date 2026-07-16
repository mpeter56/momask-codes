"""
walk_to_dance.py — Walk-to-dance identity-preserved spectrum.

Takes TWO source videos:
  --walk    : the low-intensity anchor (a walking / neutral clip)
  --dance   : the high-intensity anchor (a dancing / expressive clip)

Extracts MediaPipe pose from both, converts to 22-joint SMPL, standardises
each to 196 frames. Then for each spectrum value d in [0.09 … 1.00]:

    blended = (1 - d) * walk_pose + d * dance_pose

At d = 0.09 the output is 91 % walk + 9 % dance — the identity walks with
a subtle sway. At d = 1.00 the output is entirely the dance clip. Because
BOTH clips are recorded motion of the same person, identity is preserved
throughout by construction, and the transition is a genuine walk → dance
morph rather than an amplification.

Usage:
    conda activate momask
    python walk_to_dance.py \\
        --walk  ~/Downloads/walk_clip.mp4 \\
        --dance ~/Downloads/dance_clip.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

HOME = Path.home()
MOMASK_REPO   = HOME / "Downloads" / "momask-codes-main"
SCAFFOLD      = MOMASK_REPO / "identity_preservation"
sys.path.insert(0, str(MOMASK_REPO))
sys.path.insert(0, str(SCAFFOLD / "scripts"))

from utils.plot_script import plot_3d_motion               # noqa: E402
from utils.paramUtil import t2m_kinematic_chain            # noqa: E402
from webcam_spectrum import (                              # noqa: E402
    mediapipe_to_smpl22, standardise_length, normalise_worldish,
    confidence_smooth, extract_frames, add_oscillation, NUM_FRAMES,
)


DEFAULT_DANCE_VALUES = [0.09, 0.11, 0.28, 0.39, 0.40, 0.51, 0.63, 0.74, 0.80, 1.00]


def _normalise_common(js: np.ndarray, target_torso: float = 0.55) -> np.ndarray:
    """Common normalisation for Y-up joint sequences: torso-scale, centre XZ
    at the pelvis mean, drop feet to Y ≈ 0. Applied to BOTH walk and dance
    so their coordinate systems match before blending."""
    j = js.copy()
    torso = float(np.linalg.norm(j[:, 12] - j[:, 0], axis=-1).mean())
    j *= target_torso / max(torso, 1e-6)
    pelvis_mean_xz = j[:, 0, :].mean(axis=0)
    pelvis_mean_xz[1] = 0.0
    j -= pelvis_mean_xz
    j[..., 1] -= float(j[..., 1].min())
    return j.astype(np.float32)


def load_and_prepare(source: Path, label: str, confidence_threshold: float) -> np.ndarray:
    """Accepts either a video (mp4/mov) — runs MediaPipe — or a pre-computed
    joint (.npy) file — loads directly. Both are standardised to (196, 22, 3)
    AND normalised into the same coordinate frame (Y-up, torso-scaled, feet
    at floor, pelvis centred) so they can be linearly blended."""
    suffix = source.suffix.lower()
    if suffix in (".npy",):
        print(f"\n[{label}] loading pre-computed joints from {source.name}")
        joints = np.load(source).astype(np.float32)
        if joints.ndim == 4:
            joints = joints[0]
        if joints.shape[1:] != (22, 3):
            raise SystemExit(f"[fatal] {source} has shape {joints.shape}; expected (T, 22, 3)")
        joints = standardise_length(joints, NUM_FRAMES)
        # MoMask joints are already Y-up — skip the flip, apply torso-scale + centring
        joints = _normalise_common(joints)
    else:
        print(f"\n[{label}] extracting MediaPipe pose from {source.name}")
        frames = extract_frames(source)
        joints = np.stack([mediapipe_to_smpl22(f) for f in frames])
        joints = confidence_smooth(joints, frames, threshold=confidence_threshold)
        joints = standardise_length(joints, NUM_FRAMES)
        # MediaPipe is Y-down; normalise_worldish flips Y then does torso scaling
        joints = normalise_worldish(joints)
    print(f"[{label}] shape={joints.shape}   height={float(joints[..., 1].max()):.2f}   "
          f"range=[{joints.min():.2f}, {joints.max():.2f}]")
    return joints


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk",  required=True, help="path to the low-intensity anchor mp4")
    ap.add_argument("--dance", required=True, help="path to the high-intensity anchor mp4")
    ap.add_argument("--output-dir",
                    default=str(SCAFFOLD / "outputs" / "pipeline_output"))
    ap.add_argument("--dance-values",
                    default=",".join(f"{d:.2f}" for d in DEFAULT_DANCE_VALUES))
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--confidence-threshold", type=float, default=0.5)
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--no-hand-motion", action="store_true",
                    help="skip the arm/head sine-wave oscillation. By default it's ON so "
                         "hands + head visibly move at high d even if neither source has "
                         "articulate hand motion.")
    ap.add_argument("--hand-motion-scale", type=float, default=1.5,
                    help="how strongly the arm/head oscillation grows across the spectrum. "
                         "1.0 = subtle, 1.5 = default, 3.0 = dramatic")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dance_values = [float(x) for x in args.dance_values.split(",")]

    walk_p  = Path(args.walk).expanduser()
    dance_p = Path(args.dance).expanduser()

    walk_joints  = load_and_prepare(walk_p,  "walk",  args.confidence_threshold)
    dance_joints = load_and_prepare(dance_p, "dance", args.confidence_threshold)

    # Save the raw anchors for reference
    np.save(out_dir / "walk_source.npy",  walk_joints)
    np.save(out_dir / "dance_source.npy", dance_joints)

    print(f"\n[blend] interpolating walk → dance across {len(dance_values)} spectrum values"
          + ("" if args.no_hand_motion else f"  (+ hand oscillation × {args.hand_motion_scale})"))
    for d in dance_values:
        blended = ((1.0 - d) * walk_joints + d * dance_joints).astype(np.float32)
        # Overlay a hand + head sine wave — grows with d — so wrists, hand
        # tips, and head visibly move even when neither source has expressive
        # hand articulation. Only touches joints 15/18/19/20/21 and 9/12.
        if not args.no_hand_motion:
            blended = add_oscillation(blended, d * args.hand_motion_scale, fps=args.fps)
        npy_out = out_dir / f"blended_d{d:.2f}.npy"
        np.save(npy_out, blended)

        if not args.skip_render:
            mp4_out = out_dir / f"identity_spectrum_d{d:.2f}.mp4"
            plot_3d_motion(
                str(mp4_out),
                t2m_kinematic_chain,
                blended,
                title=f"d = {d:.2f}  ·  {int((1-d)*100)}% walk + {int(d*100)}% dance",
                fps=args.fps,
            )
            print(f"  d={d:.2f}  {int((1-d)*100)}%walk+{int(d*100)}%dance  ->  {npy_out.name}  {mp4_out.name}")
        else:
            print(f"  d={d:.2f}  {int((1-d)*100)}%walk+{int(d*100)}%dance  ->  {npy_out.name}")

    print(f"\n[done] outputs in {out_dir}")
    print( "       next: render the polished tour on top with\n"
           "       python .../render_spectrum_tour.py --dance-values "
           + ",".join(f"{d:.2f}" for d in dance_values) + " --out ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
