"""
Sample generation from a trained MiniGainNet checkpoint.

Loads a checkpoint, picks representative source/target pairs from MoMask's
editing/ runs, produces blended outputs, and renders them as stick-figure
matplotlib animations (MP4).

Also produces a comparison grid image (single frame from source, from
generated, from blended) for Slide 20.

Usage:
    conda activate momask
    cd ~/identity_preservation_mini
    python scripts/sample_mini.py --checkpoint outputs/checkpoints/mini_gain_net_final.pt

Output:
    outputs/samples/sample_XX_source_to_target.mp4  (rendered animations)
    outputs/samples/comparison_grid.png             (for Slide 20)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

sys.path.insert(0, str(Path(__file__).parent.parent))
from config_mini import cfg
from models.mini_gain_net import MiniGainNet
from data.mini_pair_dataset import discover_editing_clips


# HumanML3D 22-joint skeleton connectivity (SMPL-H body)
SMPL_H_BONES = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (4, 7), (7, 10),
    (2, 5), (5, 8), (8, 11),
    (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 13), (13, 16), (16, 18), (18, 20),
    (9, 14), (14, 17), (17, 19), (19, 21),
]


def render_stick_figure_mp4(
    motion: np.ndarray,     # (T, 22, 3)
    out_path: Path,
    title: str = "",
    fps: int = 20,
) -> None:
    """Render 22-joint stick figure as MP4."""
    T = motion.shape[0]

    # Center the figure
    center = motion[:, 0:1, :].mean(axis=0, keepdims=True)   # (1, 1, 3)
    m = motion - center

    # Find plot bounds
    lo, hi = m.min(), m.max()
    r = (hi - lo) * 0.6

    fig = plt.figure(figsize=(4, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(lo, hi)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=15, azim=-70)

    lines = [ax.plot([], [], [], "k-", linewidth=1.5)[0] for _ in SMPL_H_BONES]
    joints_scatter = ax.scatter([], [], [], c="teal", s=8)

    def init():
        for ln in lines:
            ln.set_data([], [])
            ln.set_3d_properties([])
        return lines + [joints_scatter]

    def update(frame_idx):
        pose = m[frame_idx]  # (22, 3)
        for i, (a, b) in enumerate(SMPL_H_BONES):
            lines[i].set_data([pose[a, 0], pose[b, 0]], [pose[a, 1], pose[b, 1]])
            lines[i].set_3d_properties([pose[a, 2], pose[b, 2]])
        joints_scatter._offsets3d = (pose[:, 0], pose[:, 1], pose[:, 2])
        return lines + [joints_scatter]

    ani = animation.FuncAnimation(fig, update, frames=T, init_func=init, interval=1000 // fps, blit=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    ani.save(str(out_path), writer=writer)
    plt.close(fig)


def render_comparison_grid(
    source: np.ndarray,    # (T, 22, 3)
    generated: np.ndarray,
    blended: np.ndarray,
    alpha_series: np.ndarray,  # (T,)
    out_path: Path,
    src_name: str,
    tgt_name: str,
) -> None:
    """
    Static PNG: 3 rows × 4 columns of key frames from source / generated
    / blended, plus α timeline underneath. Suitable for Slide 20.
    """
    T = source.shape[0]
    key_frames = np.linspace(0, T - 1, 4).astype(int)

    fig = plt.figure(figsize=(14, 10))
    labels = ["source (low spectrum)", "generated (high spectrum)", "blended (GainNet)"]
    data_stack = [source, generated, blended]

    for row, (label, data) in enumerate(zip(labels, data_stack)):
        for col, frame in enumerate(key_frames):
            ax = fig.add_subplot(4, 4, row * 4 + col + 1, projection="3d")
            pose = data[frame] - data[frame, 0:1]
            for a, b in SMPL_H_BONES:
                ax.plot([pose[a, 0], pose[b, 0]],
                        [pose[a, 1], pose[b, 1]],
                        [pose[a, 2], pose[b, 2]],
                        "k-", linewidth=1)
            ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c="teal", s=6)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.view_init(elev=15, azim=-70)
            if col == 0:
                ax.set_ylabel(label, fontsize=10)
            if row == 0:
                ax.set_title(f"frame {frame}", fontsize=9)

    # α timeline on the bottom row spanning all 4 cols
    ax_alpha = fig.add_subplot(4, 1, 4)
    ax_alpha.plot(np.arange(T), alpha_series, color="teal", linewidth=2)
    ax_alpha.set_xlabel("frame")
    ax_alpha.set_ylabel("learned α")
    ax_alpha.set_ylim(0, 1)
    ax_alpha.set_title(f"identity gain over time  ({src_name} → {tgt_name})", fontsize=10)
    ax_alpha.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


def load_clip(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    T = cfg.NUM_FRAMES
    if arr.shape[0] > T:
        arr = arr[:T]
    elif arr.shape[0] < T:
        pad = np.repeat(arr[-1:], T - arr.shape[0], axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    return arr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default=str(cfg.CHECKPOINTS_ROOT / "mini_gain_net_final.pt"))
    parser.add_argument("--skip_videos", action="store_true",
                        help="Skip MP4 rendering (slow), only produce comparison grid")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Load model
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = MiniGainNet(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"[sample] loaded checkpoint from epoch {ckpt['epoch']}")

    # Pick pairs to sample: sort clips by spectrum, pick low vs high pairs
    clips = discover_editing_clips(cfg.EDITING_DIR)
    if len(clips) < 2:
        raise RuntimeError(f"Need at least 2 clips, found {len(clips)}")

    # Sample pairs: (lowest, highest) plus two mid-band pairs
    n = len(clips)
    pair_indices = [
        (0, n - 1),                       # extremes
        (0, n // 2),                      # low vs mid
        (n // 2, n - 1),                  # mid vs high
    ]
    if n >= 4:
        pair_indices.append((n // 4, 3 * n // 4))  # inner band

    for i, (src_idx, tgt_idx) in enumerate(pair_indices):
        src_clip = clips[src_idx]
        tgt_clip = clips[tgt_idx]

        src = load_clip(src_clip["path"])
        tgt = load_clip(tgt_clip["path"])

        src_tensor = torch.from_numpy(src).unsqueeze(0).to(device)
        tgt_tensor = torch.from_numpy(tgt).unsqueeze(0).to(device)

        with torch.no_grad():
            alpha, blended = model(src_tensor, tgt_tensor)

        alpha_np = alpha.squeeze().cpu().numpy()      # (T,)
        blended_np = blended.squeeze(0).cpu().numpy() # (T, 22, 3)

        tag = f"{i:02d}_d{src_clip['spectrum']:.2f}_to_d{tgt_clip['spectrum']:.2f}"
        print(f"[sample] pair {i}: {src_clip['name']} (d={src_clip['spectrum']:.2f}) "
              f"→ {tgt_clip['name']} (d={tgt_clip['spectrum']:.2f})  "
              f"α_mean={alpha_np.mean():.3f}")

        # Comparison grid PNG
        grid_path = cfg.SAMPLES_ROOT / f"comparison_{tag}.png"
        render_comparison_grid(
            src, tgt, blended_np, alpha_np,
            grid_path,
            src_clip["name"], tgt_clip["name"],
        )
        print(f"        ✓ {grid_path.name}")

        # MP4s (slow — skip on demand)
        if not args.skip_videos:
            for label, data in [("source", src), ("generated", tgt), ("blended", blended_np)]:
                mp4_path = cfg.SAMPLES_ROOT / f"{tag}_{label}.mp4"
                render_stick_figure_mp4(data, mp4_path, title=f"{label}: {tag}")
                print(f"        ✓ {mp4_path.name}")

    print()
    print(f"[sample] done. outputs in {cfg.SAMPLES_ROOT}")


if __name__ == "__main__":
    main()
