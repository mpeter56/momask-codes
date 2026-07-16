"""
similarity_align.py — Frame-by-frame similarity alignment between two motion
sequences.

Given:
  --original   an mp4 (webcam) OR a pre-extracted joints .npy — the identity anchor
  --generated  an mp4 OR a joints .npy — usually the MoMask-conditioned
               output at some dance intensity (e.g. d=0.39)

Steps:
  1. Convert both sequences to a common 22-joint SMPL pose format.
  2. Compute pose features (centred at pelvis, normalised by torso length,
     so global translation / body size do not affect similarity).
  3. Build a pairwise L2 distance matrix D[i, j] = || feat_orig[i] - feat_gen[j] ||.
  4. Run Dynamic Time Warping to find the optimal monotonic alignment path
     that minimises total distance. Similar poses across the two sequences
     end up paired at the same time index in the output.
  5. Emit:
       - alignment.json          matched pairs + per-pair similarity
       - distance_matrix.png     heatmap + DTW path overlay
       - side_by_side.mp4        two-panel skeleton video playing matched
                                 frames simultaneously
       - report.txt              summary stats

Usage:
    conda activate momask
    python similarity_align.py \\
        --original  ~/Desktop/walk_riku_cam1.mp4 \\
        --generated ~/Downloads/momask-codes-main/editing/run_d0.40/joints/0/sample0_repeat0_len196.npy \\
        --output-dir ~/Downloads/momask-codes-main/identity_preservation/outputs/alignment
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

HOME = Path.home()
MOMASK_REPO   = HOME / "Downloads" / "momask-codes-main"
SCAFFOLD      = MOMASK_REPO / "identity_preservation"
sys.path.insert(0, str(MOMASK_REPO))
sys.path.insert(0, str(SCAFFOLD / "scripts"))

from utils.paramUtil import t2m_kinematic_chain                # noqa: E402
from webcam_spectrum import (                                   # noqa: E402
    mediapipe_to_smpl22, extract_frames, standardise_length,
    confidence_smooth, normalise_worldish, NUM_FRAMES,
)


# ==============================================================================
# I/O — load original or generated from mp4 or npy
# ==============================================================================
def _common_normalise(js: np.ndarray, target_torso: float = 0.55) -> np.ndarray:
    """Shared normaliser used for both sources so their coordinate systems
    match before we compute distances."""
    j = js.copy()
    torso = float(np.linalg.norm(j[:, 12] - j[:, 0], axis=-1).mean())
    j *= target_torso / max(torso, 1e-6)
    pelvis_xz = j[:, 0, :].mean(axis=0); pelvis_xz[1] = 0.0
    j -= pelvis_xz
    j[..., 1] -= float(j[..., 1].min())
    return j.astype(np.float32)


def load_source(path: Path, label: str) -> np.ndarray:
    """Return standardised, normalised (T, 22, 3) joint array from mp4 or npy."""
    if path.suffix.lower() == ".npy":
        print(f"[{label}] loading joints from {path.name}")
        j = np.load(path).astype(np.float32)
        if j.ndim == 4:
            j = j[0]
        if j.shape[1:] != (22, 3):
            raise SystemExit(f"[fatal] {path} has shape {j.shape} — expected (T, 22, 3)")
        j = standardise_length(j, NUM_FRAMES)
        return _common_normalise(j)
    print(f"[{label}] extracting MediaPipe pose from {path.name}")
    frames = extract_frames(path)
    j = np.stack([mediapipe_to_smpl22(f) for f in frames])
    j = confidence_smooth(j, frames, threshold=0.5)
    j = standardise_length(j, NUM_FRAMES)
    return normalise_worldish(j)


# ==============================================================================
# Similarity + DTW alignment
# ==============================================================================
def pose_features(js: np.ndarray) -> np.ndarray:
    """Flatten each frame's pose to a 1D descriptor, centred at pelvis and
    normalised by torso so translation + body scale don't distort similarity."""
    centred = js - js[:, 0:1, :]                                        # (T, 22, 3)
    torso = np.linalg.norm(js[:, 12] - js[:, 0], axis=-1)               # (T,)
    scale = np.clip(torso, 1e-6, None)[:, None, None]
    normed = centred / scale
    return normed.reshape(normed.shape[0], -1).astype(np.float32)       # (T, 66)


def distance_matrix(feat_a: np.ndarray, feat_b: np.ndarray) -> np.ndarray:
    """Pairwise L2 between every pose in a and every pose in b. Vectorised."""
    a = feat_a[:, None, :]
    b = feat_b[None, :, :]
    return np.linalg.norm(a - b, axis=-1).astype(np.float32)            # (Ta, Tb)


def dtw(cost: np.ndarray):
    """Return (path, total_cost). path is a list of (i, j) index pairs."""
    Ta, Tb = cost.shape
    INF = np.float32(np.inf)
    dp = np.full((Ta + 1, Tb + 1), INF, dtype=np.float32)
    parent = np.zeros((Ta + 1, Tb + 1), dtype=np.int8)                  # 0=diag, 1=up, 2=left
    dp[0, 0] = 0.0
    for i in range(1, Ta + 1):
        for j in range(1, Tb + 1):
            diag = dp[i - 1, j - 1]
            up   = dp[i - 1, j]
            left = dp[i,     j - 1]
            best = min(diag, up, left)
            dp[i, j] = cost[i - 1, j - 1] + best
            parent[i, j] = 0 if best == diag else (1 if best == up else 2)
    # backtrack
    path = []
    i, j = Ta, Tb
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = parent[i, j]
        if k == 0:   i, j = i - 1, j - 1
        elif k == 1: i -= 1
        else:        j -= 1
    return path[::-1], float(dp[Ta, Tb])


# ==============================================================================
# Visualisations
# ==============================================================================
def draw_skeleton(ax, joints, color, lw=2.4):
    ax.cla()
    ax.set_facecolor("#0f0f18")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((0.06, 0.06, 0.10, 1.0)); pane.set_edgecolor((0, 0, 0, 0))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_ticklabels([]); axis.set_ticks([]); axis.line.set_visible(False)
    j = joints[:, [0, 2, 1]]                                             # Y-up → Z-up for matplotlib
    for chain in t2m_kinematic_chain:
        ax.plot(j[chain, 0], j[chain, 1], j[chain, 2], color=color, linewidth=lw)
    ax.scatter(j[:, 0], j[:, 1], j[:, 2], s=12, color=color)


def render_heatmap(cost: np.ndarray, path, out_png: Path):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    ax.imshow(cost, cmap="magma", aspect="auto", origin="lower")
    xs = [p[1] for p in path]; ys = [p[0] for p in path]
    ax.plot(xs, ys, color="#e8b880", linewidth=2, label="DTW path")
    ax.set_xlabel("generated frame")
    ax.set_ylabel("original frame")
    ax.set_title("pose-distance matrix + optimal alignment")
    ax.legend(loc="lower right", facecolor="black", labelcolor="white")
    fig.tight_layout()
    fig.savefig(out_png, facecolor="#0a0a12")
    plt.close(fig)


def render_side_by_side(orig: np.ndarray, gen: np.ndarray, path, out_mp4: Path, fps=20,
                        width=1600, height=720, dpi=120):
    """2-panel video: left = original skeleton at path[t][0], right = generated at path[t][1]."""
    fig_w, fig_h = width / dpi, height / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="#0a0a12")
    ax_l = fig.add_subplot(121, projection="3d", facecolor="#0f0f18")
    ax_r = fig.add_subplot(122, projection="3d", facecolor="#0f0f18")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.94, wspace=0.05)

    mins = np.minimum(orig.reshape(-1, 3).min(0), gen.reshape(-1, 3).min(0))
    maxs = np.maximum(orig.reshape(-1, 3).max(0), gen.reshape(-1, 3).max(0))
    center = (mins + maxs) / 2
    radius = float(np.max(maxs - mins)) * 0.55

    def frame_setup(ax, title, color):
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[2] - radius, center[2] + radius)   # (Y-up→Z-up swap)
        ax.set_zlim(center[1] - radius * 0.05, center[1] + radius * 1.2)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=15, azim=-75)
        ax.set_title(title, color=color, fontsize=13, weight="light", family="serif")

    def draw(t):
        oi, gi = path[t]
        draw_skeleton(ax_l, orig[oi], "#5cd6ff")
        draw_skeleton(ax_r, gen[gi],  "#e8b880")
        frame_setup(ax_l, f"ORIGINAL   frame {oi:3d}", "#e8e6ff")
        frame_setup(ax_r, f"GENERATED  frame {gi:3d}", "#e8b880")
        fig.suptitle(f"aligned pair {t+1}/{len(path)}",
                     color="#8b8aa8", fontsize=12, style="italic", family="serif", y=0.98)

    ani = FuncAnimation(fig, draw, frames=len(path), interval=1000 / fps)
    writer = FFMpegWriter(fps=fps, bitrate=5000, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p", "-crf", "20"])
    ani.save(str(out_mp4), writer=writer, dpi=dpi, savefig_kwargs={"facecolor": "#0a0a12"})
    plt.close(fig)


# ==============================================================================
# Main
# ==============================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--original",  required=True, help="mp4 or joints .npy — identity anchor")
    ap.add_argument("--generated", required=True, help="mp4 or joints .npy — MoMask output")
    ap.add_argument("--output-dir",
                    default=str(SCAFFOLD / "outputs" / "alignment"))
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--width",  type=int, default=1600)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    orig = load_source(Path(args.original).expanduser(),  "original")
    gen  = load_source(Path(args.generated).expanduser(), "generated")
    print(f"[shapes] original={orig.shape}   generated={gen.shape}")

    feat_o = pose_features(orig)
    feat_g = pose_features(gen)
    print(f"[features] original={feat_o.shape}   generated={feat_g.shape}")

    print("[distance] building pairwise pose-distance matrix ...")
    cost = distance_matrix(feat_o, feat_g)
    print(f"[distance] shape={cost.shape}   min={cost.min():.3f}   max={cost.max():.3f}   mean={cost.mean():.3f}")

    print("[dtw] running dynamic time warping ...")
    path, total = dtw(cost)
    per_pair = total / len(path)
    print(f"[dtw] pairs={len(path)}   total cost={total:.2f}   per-pair avg={per_pair:.3f}")

    # ---- write alignment.json ------------------------------------------------
    alignment = {
        "num_pairs": len(path),
        "total_cost": total,
        "avg_per_pair": per_pair,
        "shape_original": list(orig.shape),
        "shape_generated": list(gen.shape),
        "pairs": [
            {"t": t, "original_frame": int(i), "generated_frame": int(j),
             "distance": float(cost[i, j])}
            for t, (i, j) in enumerate(path)
        ],
    }
    (out_dir / "alignment.json").write_text(json.dumps(alignment, indent=2))
    print(f"[out] wrote {out_dir / 'alignment.json'}")

    # ---- heatmap -------------------------------------------------------------
    heat_path = out_dir / "distance_matrix.png"
    render_heatmap(cost, path, heat_path)
    print(f"[out] wrote {heat_path}")

    # ---- side-by-side aligned video -----------------------------------------
    sbs_path = out_dir / "side_by_side.mp4"
    print(f"[out] rendering side-by-side comparison → {sbs_path.name}  ({len(path)} frames)")
    render_side_by_side(orig, gen, path, sbs_path, fps=args.fps,
                        width=args.width, height=args.height)
    print(f"[out] wrote {sbs_path}")

    # ---- text report ---------------------------------------------------------
    report = f"""similarity_align.py — report
========================================
original  : {args.original}
generated : {args.generated}

sequence lengths      : original={orig.shape[0]}  generated={gen.shape[0]}
pose-distance         : min={cost.min():.3f}  max={cost.max():.3f}  mean={cost.mean():.3f}
DTW alignment         : {len(path)} pairs
total alignment cost  : {total:.2f}
average per-pair cost : {per_pair:.3f}

lower per-pair cost ⇒ higher pose similarity between the two sequences.
identity retention ~= 1 - per_pair_cost / mean_cost.
"""
    (out_dir / "report.txt").write_text(report)
    print(f"[out] wrote {out_dir / 'report.txt'}")
    print(f"\n[done] all outputs at {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
