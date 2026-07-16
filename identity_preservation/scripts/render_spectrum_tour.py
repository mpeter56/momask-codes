"""
render_spectrum_tour.py — SPECTRUM TOUR

A single 3D skeleton figure moves through 10 dance intensities sequentially,
from identity-preserved walking (d=0.09) to full dance (d=1.00). On-screen UI
shows exactly how far along the spectrum the current clip sits.

Structure:
  - 10 blended motions, played back-to-back (196 frames each = ~10 s at 20 fps)
  - 5-frame linear-interp crossfade between clips → smooth continuous motion
  - Skeleton color morphs cool blue → warm orange as intensity rises
  - Slow camera orbit ~45° over the whole ~100 s piece
  - Overlays:
      * top-left  : "CLIP N/10 • <descriptor>"
      * top-right : current dance value (huge)
      * bottom    : horizontal spectrum bar with moving marker
                    "identity preserved ←── ── ──→ intensively dancing"

Usage:
    conda activate momask
    python render_spectrum_tour.py \\
      --out outputs/media_art/spectrum_tour.mp4 \\
      --width 1920 --height 1080 --fps 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# HumanML3D 22-joint SMPL body chain
KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],           # right leg
    [0, 1, 4, 7, 10],           # left leg
    [0, 3, 6, 9, 12, 15],       # spine → head
    [9, 14, 17, 19, 21],        # right arm
    [9, 13, 16, 18, 20],        # left arm
]

DEFAULT_DANCE_VALUES = [0.09, 0.23, 0.28, 0.40, 0.42, 0.56, 0.70, 0.74, 0.97, 1.00]

CLIP_LEN         = 196          # frames per source clip
CROSSFADE_FRAMES = 5            # smooth transition between adjacent clips


def descriptor_for(d: float) -> str:
    """Descriptor string generated from the dance-intensity value itself,
    so any set of --dance-values works without a hand-written label list."""
    if d < 0.15:  return "identity preserved"
    if d < 0.30:  return "gentle deviation"
    if d < 0.45:  return "mild transformation"
    if d < 0.60:  return "mid transformation"
    if d < 0.75:  return "moderate dance"
    if d < 0.90:  return "strong dance"
    return "intensively dancing"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend-dir", default=str(Path.home() / "Downloads" / "momask-codes-main" / "identity_preservation" / "outputs" /
                                                "pipeline_output"))
    ap.add_argument("--out", default=str(Path.home() / "Downloads" / "momask-codes-main" / "identity_preservation" / "outputs" /
                                          "media_art" / "spectrum_tour.mp4"))
    ap.add_argument("--dance-values",
                    default=",".join(f"{d:.2f}" for d in DEFAULT_DANCE_VALUES),
                    help="comma-separated dance intensities (must match blended npys on disk)")
    ap.add_argument("--width",  type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps",    type=int, default=20)
    ap.add_argument("--dpi",    type=int, default=120)
    ap.add_argument("--zoom",   type=float, default=2.0,
                    help="camera zoom (larger = character fills more of the frame). "
                         "1.0 = old default (character is a speck). 2.0 = default now.")
    return ap.parse_args()


def load_all_motions(blend_dir: Path, dance_values):
    """Return list of (196, 22, 3) float32 arrays, one per DANCE_VALUE.

    HumanML3D joint data is Y-up (Y is vertical). Matplotlib's 3D projection
    defaults to Z-up. We swap Y and Z here so the character stands upright
    without any camera-angle gymnastics downstream.
    """
    motions = []
    for d in dance_values:
        p = blend_dir / f"blended_d{d:.2f}.npy"
        if not p.exists():
            raise SystemExit(f"[fatal] missing blend: {p}\n"
                             f"        run chorus_prep.sh (or run_chorus.sh) first.")
        arr = np.load(p).astype(np.float32)
        if arr.ndim == 4:
            arr = arr[0]
        assert arr.shape[1:] == (22, 3), f"bad shape {arr.shape} in {p}"
        T = CLIP_LEN
        if arr.shape[0] > T:
            arr = arr[:T]
        elif arr.shape[0] < T:
            pad = np.repeat(arr[-1:], T - arr.shape[0], axis=0)
            arr = np.concatenate([arr, pad], axis=0)
        # Y-up -> Z-up: swap columns 1 and 2
        arr = arr[:, :, [0, 2, 1]]
        motions.append(arr)
    return motions


def build_timeline(motions):
    """
    Concatenate all clips into one long (N_TOTAL, 22, 3) array with linear-interp
    crossfades between adjacent clips. Also returns per-frame (clip_idx, in_clip_t)
    metadata so overlays can track state.
    """
    parts_motion = []
    parts_meta   = []
    num_clips = len(motions)

    for i, motion in enumerate(motions):
        parts_motion.append(motion.copy())
        # metadata: for each frame, which clip it belongs to and its within-clip position 0..1
        for t in range(CLIP_LEN):
            parts_meta.append((i, t / (CLIP_LEN - 1)))

        # crossfade into next clip
        if i < num_clips - 1:
            nxt = motions[i + 1]
            fade = np.zeros((CROSSFADE_FRAMES, 22, 3), dtype=np.float32)
            for k in range(CROSSFADE_FRAMES):
                w = (k + 1) / (CROSSFADE_FRAMES + 1)
                fade[k] = (1 - w) * motion[-1] + w * nxt[0]
                parts_meta.append((i, 1.0 + (k + 1) / CROSSFADE_FRAMES))  # >1 = fade zone
            parts_motion.append(fade)

    full_motion = np.concatenate(parts_motion, axis=0)   # (N_TOTAL, 22, 3)
    return full_motion, parts_meta


def color_for_intensity(d_value: float):
    """Turbo colormap in (0.15, 0.90) range, indexed by dance intensity 0.09→1.00."""
    norm = (d_value - 0.09) / (1.00 - 0.09)
    return plt.get_cmap("turbo")(0.15 + 0.75 * norm)


def build_scene(fig, ax_3d, radius, center):
    ax_3d.set_facecolor("#0f0f18")
    ax_3d.grid(False)
    for pane in (ax_3d.xaxis.pane, ax_3d.yaxis.pane, ax_3d.zaxis.pane):
        pane.set_facecolor((0.06, 0.06, 0.10, 1.0))
        pane.set_edgecolor((0, 0, 0, 0))
    for axis in (ax_3d.xaxis, ax_3d.yaxis, ax_3d.zaxis):
        axis.set_ticklabels([]); axis.set_ticks([]); axis.line.set_visible(False)


def draw_spectrum_bar(ax_overlay, current_d: float, spectrum_min=0.09, spectrum_max=1.00):
    ax_overlay.clear()
    ax_overlay.set_xlim(0, 1); ax_overlay.set_ylim(0, 1)
    ax_overlay.axis("off")

    # bar background
    bar_y = 0.5; bar_h = 0.18
    bar_left, bar_right = 0.12, 0.88
    # gradient bar drawn as many small colored rectangles
    N = 200
    for i in range(N):
        x0 = bar_left + (bar_right - bar_left) * i / N
        w  = (bar_right - bar_left) / N
        d  = spectrum_min + (spectrum_max - spectrum_min) * i / N
        ax_overlay.add_patch(mpatches.Rectangle(
            (x0, bar_y - bar_h / 2), w, bar_h,
            facecolor=color_for_intensity(d), edgecolor="none", alpha=0.85,
        ))
    # bar frame
    ax_overlay.add_patch(mpatches.Rectangle(
        (bar_left, bar_y - bar_h / 2),
        bar_right - bar_left, bar_h,
        facecolor="none", edgecolor="#e8e6ff", linewidth=0.8, alpha=0.6,
    ))

    # current position marker
    frac = (current_d - spectrum_min) / (spectrum_max - spectrum_min)
    marker_x = bar_left + frac * (bar_right - bar_left)
    ax_overlay.plot(
        [marker_x, marker_x],
        [bar_y - bar_h * 0.9, bar_y + bar_h * 0.9],
        color="#ffffff", linewidth=2.4, alpha=1.0,
    )
    ax_overlay.scatter([marker_x], [bar_y + bar_h * 1.05], s=60,
                       facecolor="#ffffff", edgecolor="none", zorder=5)

    # labels below the bar
    ax_overlay.text(bar_left,  bar_y - bar_h * 1.4,
                    "identity preserved", color="#8b8aa8",
                    fontsize=11, ha="left",  va="top", family="serif", style="italic")
    ax_overlay.text(bar_right, bar_y - bar_h * 1.4,
                    "intensively dancing", color="#e8b880",
                    fontsize=11, ha="right", va="top", family="serif", style="italic")
    ax_overlay.text((bar_left + bar_right) / 2, bar_y + bar_h * 1.6,
                    f"{current_d:.2f}",
                    color="#ffffff", fontsize=13, ha="center", va="bottom",
                    family="serif", weight="light", alpha=0.7)


def render_character(ax_3d, joints_frame: np.ndarray, color, linewidth=2.4):
    for chain in KINEMATIC_CHAIN:
        ax_3d.plot(
            joints_frame[chain, 0], joints_frame[chain, 1], joints_frame[chain, 2],
            color=color, linewidth=linewidth, solid_capstyle="round", alpha=0.95,
        )
    ax_3d.scatter(
        joints_frame[:, 0], joints_frame[:, 1], joints_frame[:, 2],
        s=14, color=color, alpha=1.0, edgecolors="none",
    )


def main() -> int:
    args = parse_args()
    blend_dir = Path(args.blend_dir)
    out_path  = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dance_values = [float(x) for x in args.dance_values.split(",")]
    NUM_CLIPS = len(dance_values)

    print(f"[tour] loading {NUM_CLIPS} motions from {blend_dir}")
    print(f"[tour] dance values: {dance_values}")
    motions = load_all_motions(blend_dir, dance_values)

    print(f"[tour] building timeline with {CROSSFADE_FRAMES}-frame crossfades")
    full_motion, meta = build_timeline(motions)
    N_TOTAL = full_motion.shape[0]
    print(f"[tour] total frames: {N_TOTAL}  (~{N_TOTAL / args.fps:.1f} s)")

    # View bounds from all frames
    mins = full_motion.reshape(-1, 3).min(axis=0)
    maxs = full_motion.reshape(-1, 3).max(axis=0)
    center = (mins + maxs) / 2
    # Use character height (Z after Y↔Z swap) as the reference, not the max
    # extent across all axes — otherwise wide lateral motion blows up the
    # cube and the character shrinks to a speck.
    char_height = float(maxs[2] - mins[2])
    radius = char_height * 0.7 / max(0.1, args.zoom)

    # Figure with 3D canvas + bottom overlay strip
    fig_w, fig_h = args.width / args.dpi, args.height / args.dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=args.dpi, facecolor="#0a0a12")
    ax_3d = fig.add_axes([0, 0.14, 1, 0.86], projection="3d")
    ax_overlay = fig.add_axes([0, 0.00, 1, 0.14])
    build_scene(fig, ax_3d, radius, center)

    def draw(frame: int):
        ax_3d.cla()
        build_scene(fig, ax_3d, radius, center)

        # Camera slow orbit ~45° over the whole piece
        t_all = frame / max(1, N_TOTAL - 1)
        azim = -75 + 45 * t_all
        elev = 12 + 4 * np.sin(t_all * np.pi)
        ax_3d.view_init(elev=elev, azim=azim)
        ax_3d.set_xlim(center[0] - radius, center[0] + radius)
        ax_3d.set_ylim(center[1] - radius, center[1] + radius)
        # Z range: fit the character with a bit of headroom + floor,
        # scaled by the same zoom factor
        z_half = char_height * 0.65 / max(0.1, args.zoom)
        ax_3d.set_zlim(center[2] - z_half, center[2] + z_half)
        ax_3d.set_box_aspect((1, 1, 1))

        # Determine current dance intensity for coloring + overlay
        clip_idx, in_clip = meta[frame]
        if in_clip > 1.0:  # crossfade zone
            next_idx = min(clip_idx + 1, NUM_CLIPS - 1)
            w = in_clip - 1.0
            current_d = (1 - w) * dance_values[clip_idx] + w * dance_values[next_idx]
            desc_a = descriptor_for(dance_values[clip_idx])
            desc_b = descriptor_for(dance_values[next_idx])
            descriptor = desc_a + " → " + desc_b if desc_a != desc_b else desc_a
            title_idx = clip_idx + 1  # still label as current clip
        else:
            current_d = dance_values[clip_idx]
            descriptor = descriptor_for(current_d)
            title_idx = clip_idx + 1

        color = color_for_intensity(current_d)
        render_character(ax_3d, full_motion[frame], color, linewidth=2.6)

        # Top-left overlay: clip counter + descriptor
        # (spaces between letters give the airy typographic feel — matplotlib
        # doesn't support real letter-spacing via kwarg)
        clip_title = f"C L I P   {title_idx}/{NUM_CLIPS}"

        # spectrum bar min/max come from the actual dance values in this run
        d_min, d_max = min(dance_values), max(dance_values)
        ax_3d.text2D(
            0.02, 0.95, clip_title,
            transform=ax_3d.transAxes,
            color="#e8e6ff", alpha=0.85, fontsize=13, weight="light",
            family="serif",
        )
        ax_3d.text2D(
            0.02, 0.92, descriptor,
            transform=ax_3d.transAxes,
            color="#8b8aa8", alpha=0.75, fontsize=11, style="italic", family="serif",
        )

        # Top-right overlay: current d value (huge)
        ax_3d.text2D(
            0.98, 0.94, f"d = {current_d:.2f}",
            transform=ax_3d.transAxes,
            color="#ffffff", alpha=0.9, fontsize=32, weight="light",
            family="serif", ha="right",
        )

        # Bottom overlay: spectrum bar
        draw_spectrum_bar(ax_overlay, current_d, spectrum_min=d_min, spectrum_max=d_max)

    print(f"[tour] rendering {N_TOTAL} frames at {args.width}x{args.height} @ {args.fps} fps")
    ani = FuncAnimation(fig, draw, frames=N_TOTAL, interval=1000 / args.fps, blit=False)
    writer = FFMpegWriter(fps=args.fps, bitrate=6000, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
    ani.save(str(out_path), writer=writer, dpi=args.dpi,
             savefig_kwargs={"facecolor": "#0a0a12"})
    plt.close(fig)
    print(f"[done] saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
