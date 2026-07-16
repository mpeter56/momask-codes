"""
run_rules_blend.py — Apply all 10 source-only perturbation rules to the
source pose sequence at every "spectrum" strength (dance value).

The source is the lowest-intensity MoMask output (dance_mode_0.09) which is
the closest thing on disk to your raw walking motion. Every output is a
perturbation of the SAME source at a given strength — identity preserved
by construction.

Writes 100 .npy + .mp4 files plus rules_metadata.json.

Usage:
    conda activate momask
    python run_rules_blend.py
    python run_rules_blend.py --dance-values 0.09,0.40,1.00 --rules wave,rhythm_lock
    python run_rules_blend.py --skip-render        # .npy only, no matplotlib render
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

HOME = Path.home()
MOMASK_REPO   = HOME / "Downloads" / "momask-codes-main"
MINI_SCAFFOLD = HOME / "Downloads" / "momask-codes-main" / "identity_preservation"
sys.path.insert(0, str(MINI_SCAFFOLD))
sys.path.insert(0, str(MOMASK_REPO))

from identity_rules import RULES, apply_rule, perturbation_amount        # noqa: E402
from utils.plot_script import plot_3d_motion                             # noqa: E402
from utils.paramUtil import t2m_kinematic_chain                          # noqa: E402


DEFAULT_DANCE_VALUES = [0.09, 0.11, 0.28, 0.39, 0.40, 0.51, 0.63, 0.74, 0.80, 1.00]
NUM_FRAMES = 196


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--editing-dir", default=str(MOMASK_REPO / "editing"))
    ap.add_argument("--output-dir",  default=str(MINI_SCAFFOLD / "outputs" / "pipeline_output"))
    ap.add_argument("--dance-values", default=",".join(f"{d:.2f}" for d in DEFAULT_DANCE_VALUES))
    ap.add_argument("--rules", default="", help="comma-separated rule keys; empty = all 10")
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
    rule_keys    = [k.strip() for k in args.rules.split(",") if k.strip()] or [r.key for r in RULES]
    rules_to_run = [r for r in RULES if r.key in rule_keys]

    # ---------- load THE source ---------------------------------------------
    # We use the lowest-intensity MoMask output as our source — it's the
    # closest thing on disk to the raw walking motion. Every rule then
    # perturbs THIS one source at various strengths.
    source_d = min(dance_values)
    source_path = find_joint_npy(editing_dir, source_d)
    if source_path is None:
        raise SystemExit(f"[fatal] no source joint file for d={source_d:.2f}")
    print(f"[rules] SOURCE (d={source_d:.2f}): {source_path}")
    source = standardize(np.load(source_path))
    print(f"[rules] source shape: {source.shape}   pose scale: {np.linalg.norm(source, axis=-1).mean():.3f}")

    # ---------- apply each rule at each strength ----------------------------
    metadata = {
        "rules": [{"key": r.key, "title": r.title, "tagline": r.tagline,
                   "preserves": r.preserves, "borrows": r.borrows} for r in rules_to_run],
        "dance_values": dance_values,
        "source_d": source_d,
        "philosophy": "source-only perturbation (Unnoticed Dance principle)",
        "entries": [],
    }

    for r in rules_to_run:
        print(f"\n=== rule: {r.title} ({r.key}) ===")
        for d in dance_values:
            strength = float(d)
            blended = apply_rule(r.key, source, strength=strength, fps=args.fps)
            pert = perturbation_amount(source, blended)

            npy_out = output_dir / f"blended_{r.key}_d{d:.2f}.npy"
            np.save(npy_out, blended)
            print(f"  d={d:.2f}  strength={strength:.2f}  perturbation={pert:.4f}  ->  {npy_out.name}")

            metadata["entries"].append({
                "rule": r.key, "d": d, "strength": strength,
                "npy":  npy_out.name,
                "mp4":  f"blended_{r.key}_d{d:.2f}.mp4",
                "perturbation": round(pert, 4),
            })

            if args.skip_render:
                continue
            mp4_out = output_dir / f"blended_{r.key}_d{d:.2f}.mp4"
            plot_3d_motion(
                str(mp4_out),
                t2m_kinematic_chain,
                blended,
                title=f"{r.title}  ·  strength={strength:.2f}",
                fps=args.fps,
            )

    meta_path = output_dir / "rules_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"\n[rules] wrote {meta_path}")
    print(f"[rules] {len(metadata['entries'])} (rule, strength) pairs generated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
