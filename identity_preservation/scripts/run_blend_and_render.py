"""
run_blend_and_render.py — Plan C step 4+5 combined.

For each dance value in DANCE_VALUES:
  1. Load Ted's raw generated joints (T, 22, 3) from editing/<run>/joints/0/*.npy
  2. Use d=0.09 as source proxy (self-blend for d=0.09 itself)
  3. Blend via MiniGainNet: (alpha, blended) = model(source, generated)
  4. Save blended (T, 22, 3) npy
  5. Render blended via the plot_3d_motion function (blue skeleton, gray gridded
     background — identical style to Ted's dance_mode_X.XX_repeat0.mp4)

Prereqs (run BEFORE this script):
  conda activate momask
  cd ~/Downloads/momask-codes-main
  # First run the MoMask pipeline once to produce the joint npys:
  cp sessions/20260623_150730/momask_input.npz .
  cp sessions/20260623_150730/input.mp4 input_videos/
  python scripts/video_to_spectrum.py --video input.mp4 --skip_mediapipe

Then:
  cd ~/Downloads/momask-codes-main   # or anywhere; paths are absolute
  python "/Users/neek526/Downloads/momask-codes-main/identity_preservation/scripts/run_blend_and_render.py"

Constraints (from the handoff):
  - Read-only on MiniGainNet checkpoint (loads, does not modify)
  - No retraining
  - CPU is fine (batch=1, ~1s/render); MPS not needed but auto-used if available
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ------------------------------------------------------------------
# Paths (all absolute — this script must be runnable from anywhere)
# ------------------------------------------------------------------
HOME = Path.home()
MINI_SCAFFOLD = HOME / "Downloads" / "momask-codes-main" / "identity_preservation"
MOMASK_REPO   = HOME / "Downloads" / "momask-codes-main"
CHECKPOINT    = HOME / "identity_preservation_mini" / "outputs" / "checkpoints" / "mini_gain_net_final.pt"
EDITING_DIR   = MOMASK_REPO / "editing"
OUT_DIR       = MINI_SCAFFOLD / "outputs" / "pipeline_output"

# Where the 4 target dance values live in Ted's STEPS
DANCE_VALUES = [0.09, 0.40, 0.74, 1.00]
NUM_FRAMES   = 196
NUM_JOINTS   = 22
FPS          = 20

# Import from both project roots.
# ORDER MATTERS: MINI_SCAFFOLD must sit at sys.path[0] so `models` resolves
# to identity_preservation_mini/models/, not momask-codes-main/models/
# (the latter has an __init__.py and would shadow us).
sys.path.insert(0, str(MOMASK_REPO))
sys.path.insert(0, str(MINI_SCAFFOLD))


# ------------------------------------------------------------------
# Locate the joint npy for a given dance value inside editing/
# ------------------------------------------------------------------
def find_joint_npy(editing_dir: Path, dance: float) -> Optional[Path]:
    """
    Search several editing/ subdir naming conventions for the joint npy file
    matching a given dance value. Handles all of:
      - editing/dance_mode_0.23/joints/0/sample0_repeat0_len196.npy   (fresh pipeline)
      - editing/run_d0.23/joints/0/sample0_repeat0_len196.npy         (Plan C direct)
      - editing/run_1_d 23/joints/0/sample0_repeat0_len196.npy        (numbered runs)
      - editing/run_1_d_23/joints/0/sample0_repeat0_len196.npy        (underscore variant)
    Prefers raw (non-*_ik.npy) versions.
    """
    tag = f"{dance:.2f}"                     # "0.23"
    tag_pct = str(int(round(dance * 100)))   # "23"

    # 1. Canonical / Plan-C paths — exact filename
    canonical = [
        editing_dir / f"dance_mode_{tag}" / "joints" / "0" / "sample0_repeat0_len196.npy",
        editing_dir / f"dance_mode_{tag}" / "animations" / "0" / "sample0_repeat0_len196.npy",
        editing_dir / f"run_d{tag}"       / "joints" / "0" / "sample0_repeat0_len196.npy",
        editing_dir / f"run_d{tag}"       / "animations" / "0" / "sample0_repeat0_len196.npy",
    ]
    for p in canonical:
        if p.exists():
            return p

    # 2. Glob patterns for various naming conventions.
    #    * matches any characters within a path segment (not slashes).
    for sub in ("joints", "animations"):
        patterns = [
            f"*{tag}*/**/{sub}/**/*.npy",              # any dir containing "0.23"
            f"*d {tag_pct}/{sub}/**/*.npy",            # "run_1_d 23" — trailing "d 23"
            f"*d_{tag_pct}/{sub}/**/*.npy",            # "run_1_d_23"
            f"*d {tag_pct}/**/{sub}/**/*.npy",         # extra depth safeguard
        ]
        for pattern in patterns:
            hits = sorted(editing_dir.glob(pattern))
            raw = [h for h in hits if "_ik" not in h.name]
            if raw:
                return raw[0]
            if hits:
                return hits[0]

    return None


def standardize(arr: np.ndarray, T: int = NUM_FRAMES, J: int = NUM_JOINTS) -> np.ndarray:
    """Return (T, J, 3) float32. Drops leading batch dim if present, pads/truncates."""
    a = np.asarray(arr).astype(np.float32)
    if a.ndim == 4:  # (B, T, J, 3)
        a = a[0]
    if a.ndim != 3 or a.shape[1] != J or a.shape[2] != 3:
        raise ValueError(f"unexpected joint array shape {a.shape} (expected (T,{J},3))")
    if a.shape[0] > T:
        a = a[:T]
    elif a.shape[0] < T:
        pad = np.repeat(a[-1:], T - a.shape[0], axis=0)
        a = np.concatenate([a, pad], axis=0)
    return a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(CHECKPOINT))
    ap.add_argument("--editing-dir", default=str(EDITING_DIR))
    ap.add_argument("--output-dir", default=str(OUT_DIR))
    ap.add_argument("--dance-values", default=",".join(f"{d:.2f}" for d in DANCE_VALUES),
                    help="comma-separated floats")
    ap.add_argument("--fps", type=int, default=FPS)
    args = ap.parse_args()

    dance_values = [float(x) for x in args.dance_values.split(",")]
    editing_dir  = Path(args.editing_dir)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Late imports (need sys.path already set)
    # ------------------------------------------------------------------
    from config_mini import cfg                          # noqa: E402
    from models.mini_gain_net import MiniGainNet         # noqa: E402
    from utils.plot_script import plot_3d_motion         # noqa: E402
    from utils.paramUtil import t2m_kinematic_chain      # noqa: E402

    # ------------------------------------------------------------------
    # Load MiniGainNet (CPU is fine for 4 forward passes on batch=1)
    # ------------------------------------------------------------------
    ckpt_path = Path(args.checkpoint)
    print(f"[load] checkpoint: {ckpt_path}")
    if not ckpt_path.exists():
        print(f"  MISSING checkpoint — aborting")
        return 1
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    model = MiniGainNet(cfg)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"  loaded state_dict ({sum(p.numel() for p in model.parameters()):,} params)")

    # ------------------------------------------------------------------
    # Locate source-proxy joints (dance=0.09) once
    # ------------------------------------------------------------------
    source_path = find_joint_npy(editing_dir, 0.09)
    if source_path is None:
        print(f"[fatal] cannot find dance=0.09 joint npy under {editing_dir}")
        print(f"        run video_to_spectrum.py first, then rerun this script")
        return 2
    print(f"[source d=0.09] {source_path}")
    source = standardize(np.load(source_path))

    # ------------------------------------------------------------------
    # Process each dance value
    # ------------------------------------------------------------------
    results = []
    for d in dance_values:
        print(f"\n--- d={d:.2f} ---")
        gen_path = find_joint_npy(editing_dir, d)
        if gen_path is None:
            print(f"  MISSING joint npy for dance={d:.2f} — skipping")
            results.append((d, None, None, None))
            continue
        print(f"  joints: {gen_path}")
        generated = standardize(np.load(gen_path))

        # Self-blend for d=0.09 (source == generated → blended == generated)
        src = source if abs(d - 0.09) > 1e-6 else generated

        src_t = torch.from_numpy(src).unsqueeze(0)        # (1, T, J, 3)
        gen_t = torch.from_numpy(generated).unsqueeze(0)
        with torch.no_grad():
            alpha, blended = model(src_t, gen_t)
        blended_np = blended.squeeze(0).numpy().astype(np.float32)
        alpha_mean = float(alpha.mean().item())

        # Save blended joints
        npy_out = output_dir / f"blended_d{d:.2f}.npy"
        np.save(npy_out, blended_np)

        # Render via Ted's own renderer (blue skeleton, gray gridded bg)
        mp4_out = output_dir / f"blended_d{d:.2f}.mp4"
        plot_3d_motion(
            str(mp4_out),
            t2m_kinematic_chain,
            blended_np,
            title=f"identity-preserved d={d:.2f}",
            fps=args.fps,
        )
        print(f"  alpha_mean = {alpha_mean:.4f}")
        print(f"  saved npy: {npy_out}")
        print(f"  saved mp4: {mp4_out}")
        results.append((d, alpha_mean, npy_out, mp4_out))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for d, am, _npy, mp4 in results:
        if am is None:
            print(f"  d={d:.2f}: SKIPPED (no joint file)")
        else:
            print(f"  d={d:.2f}: alpha_mean={am:.4f}  ->  {mp4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
