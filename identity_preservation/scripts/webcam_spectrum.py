"""
webcam_spectrum.py — Identity spectrum from your webcam, no MoMask.

Flow:
  1. Run MediaPipe Pose on webcam mp4                              → 33 landmarks / frame
  2. Convert to 22-joint SMPL layout (approximate mapping)          → (T, 22, 3)
  3. Standardise to 196 frames (uniform subsample or tail-pad)
  4. Normalise to world-ish coords (Y-up, ~1.7 unit tall, feet ≈ 0)
  5. For each spectrum intensity d:
         amplified = mean_pose + (source - mean_pose) * (1 + d)
  6. Save as blended_d{d:.2f}.npy + render identity_spectrum_d{d:.2f}.mp4

Identity is preserved by construction: every clip is the SAME person doing the
SAME recorded motion. The spectrum is the amplitude of that motion. At d=0.09
the motion is 1.09× the recording (barely detectable). At d=1.00 it's 2× the
recording — same body doing the same actions, just twice as big.

Usage:
    conda activate momask
    python webcam_spectrum.py --webcam ~/Downloads/momask-codes-main/input_videos/webcam_20260708_202958.mp4

Then run the spectrum tour on the outputs:
    bash ~/Downloads/3rd\\ Jul/identity_preservation_mini/scripts/run_spectrum_tour.sh --preview
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HOME = Path.home()
MOMASK_REPO   = HOME / "Downloads" / "momask-codes-main"
MINI_SCAFFOLD = HOME / "Downloads" / "momask-codes-main" / "identity_preservation"
sys.path.insert(0, str(MOMASK_REPO))

from utils.plot_script import plot_3d_motion               # noqa: E402
from utils.paramUtil import t2m_kinematic_chain            # noqa: E402


DEFAULT_DANCE_VALUES = [0.09, 0.11, 0.28, 0.39, 0.40, 0.51, 0.63, 0.74, 0.80, 1.00]
NUM_FRAMES = 196


# ------------------------------------------------------------------
# MediaPipe (33) → SMPL-H 22-joint layout
# ------------------------------------------------------------------
# MediaPipe indexing (excerpt):
#   0=nose, 11=Lshoulder, 12=Rshoulder, 13=Lelbow, 14=Relbow,
#   15=Lwrist,  16=Rwrist, 23=Lhip, 24=Rhip, 25=Lknee, 26=Rknee,
#   27=Lankle, 28=Rankle, 31=Lfoot_index, 32=Rfoot_index
def mediapipe_to_smpl22(mp_landmarks) -> np.ndarray:
    def g(i):
        lm = mp_landmarks[i]
        return np.array([lm["x"], lm["y"], lm["z"]], dtype=np.float32)

    Lhip, Rhip = g(23), g(24)
    Lsho, Rsho = g(11), g(12)
    pelvis = (Lhip + Rhip) * 0.5
    neck   = (Lsho + Rsho) * 0.5
    chest  = pelvis + (neck - pelvis) * 0.8

    j = np.zeros((22, 3), dtype=np.float32)
    j[0]  = pelvis
    j[1]  = Lhip;    j[2]  = Rhip
    j[3]  = pelvis + (neck - pelvis) * 0.33   # spine1
    j[4]  = g(25);   j[5]  = g(26)             # knees
    j[6]  = pelvis + (neck - pelvis) * 0.66   # spine2
    j[7]  = g(27);   j[8]  = g(28)             # ankles
    j[9]  = chest                              # spine3 / chest
    j[10] = g(31);   j[11] = g(32)             # toes
    j[12] = neck
    j[13] = Lsho;    j[14] = Rsho
    j[15] = g(0)                                # head (nose)
    j[16] = g(13);   j[17] = g(14)             # elbows
    j[18] = g(15);   j[19] = g(16)             # wrists
    # small extension from wrist for hand tips
    j[20] = g(15) + (g(15) - g(13)) * 0.15
    j[21] = g(16) + (g(16) - g(14)) * 0.15
    return j


def normalise_worldish(js: np.ndarray, target_torso: float = 0.55) -> np.ndarray:
    """MediaPipe image coords → world-ish coords, torso-based scaling.

    Scaling uses shoulder-to-hip distance (torso) rather than total Y range,
    so seated / partial-body poses stay in proportion instead of getting
    stretched vertically.
    """
    j = js.copy()
    j[..., 1] *= -1                                        # image y is downward

    # torso length = pelvis (0) to neck (12), averaged over frames
    torso = float(np.linalg.norm(j[:, 12] - j[:, 0], axis=-1).mean())
    j *= target_torso / max(torso, 1e-6)                   # ~0.55 unit torso ≈ 1.7 m human

    pelvis_mean_xz = j[:, 0, :].mean(axis=0)
    pelvis_mean_xz[1] = 0.0                                # keep Y so feet stay at floor
    j -= pelvis_mean_xz
    j[..., 1] -= float(j[..., 1].min())                    # feet ≈ 0
    return j.astype(np.float32)


def confidence_smooth(joints_seq: np.ndarray, mp_frames: list, threshold: float = 0.5) -> np.ndarray:
    """Replace low-visibility joint positions with linear interpolation
    between the nearest confident neighbours. Handles occluded legs while
    seated, dropped detections, etc.

    Mapping from SMPL-22 joint index → MediaPipe visibility source:
    same MP index we sampled from (see mediapipe_to_smpl22).
    """
    # Map SMPL joint idx → MP idx used for its position
    SMPL_TO_MP = {
        0: [23, 24], 1: [23], 2: [24], 4: [25], 5: [26], 7: [27], 8: [28],
        10: [31], 11: [32], 12: [11, 12], 13: [11], 14: [12], 15: [0],
        16: [13], 17: [14], 18: [15], 19: [16], 20: [15], 21: [16],
        # 3, 6, 9 are interpolated from pelvis + neck — use both visibilities
        3: [11, 12, 23, 24], 6: [11, 12, 23, 24], 9: [11, 12, 23, 24],
    }
    T, J, _ = joints_seq.shape
    smoothed = joints_seq.copy()
    fixed = 0
    for smpl_j, mp_ids in SMPL_TO_MP.items():
        vis = np.array([
            min(mp_frames[t][mp_i]["visibility"] for mp_i in mp_ids)
            for t in range(T)
        ])
        good = vis >= threshold
        if good.all() or not good.any():
            continue
        good_idx = np.where(good)[0]
        for t in np.where(~good)[0]:
            left  = good_idx[good_idx < t]
            right = good_idx[good_idx > t]
            if len(left) and len(right):
                l, r = left[-1], right[0]
                alpha = (t - l) / (r - l)
                smoothed[t, smpl_j] = (1 - alpha) * joints_seq[l, smpl_j] + alpha * joints_seq[r, smpl_j]
            elif len(left):
                smoothed[t, smpl_j] = joints_seq[left[-1], smpl_j]
            elif len(right):
                smoothed[t, smpl_j] = joints_seq[right[0], smpl_j]
            fixed += 1
    print(f"[webcam] confidence smoothing: {fixed} low-visibility joint samples interpolated")
    return smoothed


def add_oscillation(js: np.ndarray, strength: float, fps: int = 20) -> np.ndarray:
    """Overlay a gentle sinusoidal wave on arms + head so a still subject
    still produces visible dance at high strength. No-op at strength=0."""
    T = js.shape[0]
    t = np.arange(T)
    ms = t * (1000.0 / fps)
    out = js.copy()
    a = strength * 0.05                                     # ~5 cm max wave at strength=1

    head_bob = np.sin(ms / 500.0) * a
    out[:, 15, 1] += head_bob                               # head Y

    arm_sway_l = np.sin(ms / 380.0) * a * 1.5
    arm_sway_r = np.cos(ms / 420.0) * a * 1.5
    out[:, 18, 0] += arm_sway_l                             # L wrist X
    out[:, 19, 0] -= arm_sway_r                             # R wrist X
    out[:, 18, 1] += np.cos(ms / 600.0) * a
    out[:, 19, 1] += np.sin(ms / 640.0) * a

    torso_lean = np.sin(ms / 900.0) * a * 0.6
    out[:, 9, 0]  += torso_lean                             # chest lean
    out[:, 12, 0] += torso_lean                             # neck lean

    return out.astype(np.float32)


# ------------------------------------------------------------------
# MediaPipe pose extraction
# ------------------------------------------------------------------
def extract_frames(webcam_path: Path):
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(str(webcam_path))
    if not cap.isOpened():
        raise SystemExit(f"[fatal] cv2 could not open {webcam_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[webcam] video: {webcam_path.name}   fps={fps:.2f}   frames={total}")

    pose = mp.solutions.pose.Pose(
        model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    frames = []
    detected = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = pose.process(rgb)
        if res.pose_landmarks:
            frames.append(
                [{"x": float(lm.x), "y": float(lm.y),
                  "z": float(lm.z), "visibility": float(lm.visibility)}
                 for lm in res.pose_landmarks.landmark]
            )
            detected += 1
        else:
            frames.append(frames[-1] if frames else
                          [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.0}] * 33)
    cap.release()
    pose.close()
    print(f"[webcam] detected pose in {detected}/{len(frames)} frames")
    return frames


def standardise_length(js: np.ndarray, T: int = NUM_FRAMES) -> np.ndarray:
    if js.shape[0] > T:
        idx = np.linspace(0, js.shape[0] - 1, T).astype(int)
        return js[idx]
    if js.shape[0] < T:
        pad = np.repeat(js[-1:], T - js.shape[0], axis=0)
        return np.concatenate([js, pad], axis=0)
    return js


def amplify_motion(js: np.ndarray, strength: float, amp_boost: float = 3.0) -> np.ndarray:
    """(1 + strength × amp_boost) × deviation from mean pose.

    amp_boost multiplies the amplification, so subtle motion still produces
    visible dance at high strength. amp_boost=1.0 = old behaviour (d=1 → 2×).
    amp_boost=3.0 (default) means d=1.00 amplifies motion by 4×. amp_boost=5.0
    means 6×. Push higher for very still subjects.
    """
    mean_pose = js.mean(axis=0, keepdims=True)               # (1, 22, 3)
    return (mean_pose + (js - mean_pose) * (1.0 + strength * amp_boost)).astype(np.float32)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--webcam", required=True)
    ap.add_argument("--output-dir",
                    default=str(MINI_SCAFFOLD / "outputs" / "pipeline_output"))
    ap.add_argument("--dance-values",
                    default=",".join(f"{d:.2f}" for d in DEFAULT_DANCE_VALUES))
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--skip-render", action="store_true",
                    help="write .npy files only, skip matplotlib mp4 render")
    ap.add_argument("--amp-boost", type=float, default=3.0,
                    help="how much to multiply amplification. 1.0=subtle, 3.0=default, 6.0=very still subject")
    ap.add_argument("--oscillation", action="store_true",
                    help="overlay a gentle sine wave on arms+head so still subjects still visibly dance")
    ap.add_argument("--confidence-threshold", type=float, default=0.5,
                    help="MediaPipe visibility below this triggers interpolation from neighbours")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dance_values = [float(x) for x in args.dance_values.split(",")]

    # 1. extract 33-landmark pose per frame
    frames = extract_frames(Path(args.webcam).expanduser())

    # 2. convert each frame to (22, 3)
    joints = np.stack([mediapipe_to_smpl22(f) for f in frames])

    # 2.5 interpolate low-confidence joints from their neighbours
    joints = confidence_smooth(joints, frames, threshold=args.confidence_threshold)

    # 3. standardise length + normalise coords
    joints = standardise_length(joints, NUM_FRAMES)
    joints = normalise_worldish(joints)
    print(f"[webcam] source shape: {joints.shape}   "
          f"height={float(joints[..., 1].max()):.2f}   "
          f"pelvis_mean={joints[:, 0].mean(axis=0)}")

    # 4. save the un-amplified source too (for the panel board / reference)
    np.save(out_dir / "webcam_source.npy", joints)

    # 5. amplify + optionally add oscillation + save + optionally render
    print(f"[webcam] amp_boost={args.amp_boost}   oscillation={args.oscillation}")
    for d in dance_values:
        amplified = amplify_motion(joints, d, amp_boost=args.amp_boost)
        if args.oscillation:
            amplified = add_oscillation(amplified, d, fps=args.fps)
        factor = 1.0 + d * args.amp_boost
        npy_out = out_dir / f"blended_d{d:.2f}.npy"
        np.save(npy_out, amplified)

        if not args.skip_render:
            mp4_out = out_dir / f"identity_spectrum_d{d:.2f}.mp4"
            plot_3d_motion(
                str(mp4_out),
                t2m_kinematic_chain,
                amplified,
                title=f"d = {d:.2f}   ·   motion × {factor:.2f}",
                fps=args.fps,
            )
            print(f"  d={d:.2f}   ×{factor:.2f}   ->   {npy_out.name}   {mp4_out.name}")
        else:
            print(f"  d={d:.2f}   ×{factor:.2f}   ->   {npy_out.name}")

    print(f"\n[webcam] done. blended npys + mp4s at {out_dir}")
    print( "         next: run the spectrum tour on top:\n"
           "         bash ~/Downloads/3rd\\ Jul/identity_preservation_mini/scripts/run_spectrum_tour.sh --preview")
    return 0


if __name__ == "__main__":
    sys.exit(main())
