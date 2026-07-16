"""
Mini pair dataset for overnight training.

Loads Ted's editing/ workspace outputs (raw 22-joint xyz npy files) and
constructs (source, target) pairs from them for identity-preservation
training.

Pairing strategy:
    Each editing/ run represents a spectrum-varied version of some
    underlying motion. We treat lower-spectrum runs as "source" (more
    identity-like) and higher-spectrum runs as "target" (more transformed).
    A learned GainNet must then predict how to blend the two.

Data augmentation is aggressive because the real clip count is tiny.
Each real pair is expanded N_AUGMENT_PER_CLIP times with time-warp and
joint-noise perturbations.

Expected directory layout under EDITING_DIR:
    editing/
        run_d0.09/joints/0/sample0_repeat0_len196.npy
        run_d0.40/joints/0/sample0_repeat0_len196.npy
        run_d0.74/joints/0/sample0_repeat0_len196.npy
        run_d1.00/joints/0/sample0_repeat0_len196.npy
        run_1_d 23/joints/0/sample0_repeat0_len196.npy
        ... etc
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _extract_spectrum_value(run_name: str) -> float | None:
    """
    Parse spectrum value from folder name.

    Recognized patterns:
        run_d0.09      → 0.09
        run_d0.40      → 0.40
        run_1_d 23     → 0.23
        run_2_d 28     → 0.28
        run_5_d 70     → 0.70
    """
    # Try decimal first: d0.NN
    m = re.search(r"d(\d+\.\d+)", run_name)
    if m:
        return float(m.group(1))
    # Then integer percent: d NN or d_NN
    m = re.search(r"d[_ ]?(\d+)", run_name)
    if m:
        val = int(m.group(1))
        return val / 100.0 if val > 1 else float(val)
    return None


def discover_editing_clips(editing_dir: Path) -> list[dict]:
    """
    Walk editing_dir and return a list of {name, spectrum, path} dicts,
    sorted by spectrum value.
    """
    clips = []
    for run_dir in editing_dir.iterdir():
        if not run_dir.is_dir():
            continue
        # Look for the standard joint file
        candidates = list(run_dir.glob("joints/0/sample0_repeat0_len*.npy"))
        # Exclude IK-cleaned variants — we want the raw pipeline output
        candidates = [p for p in candidates if "_ik" not in p.name]
        if not candidates:
            continue
        spec_val = _extract_spectrum_value(run_dir.name)
        if spec_val is None:
            continue
        clips.append({
            "name": run_dir.name,
            "spectrum": spec_val,
            "path": candidates[0],
        })
    clips.sort(key=lambda c: c["spectrum"])
    return clips


def _time_warp(motion: np.ndarray, factor: float) -> np.ndarray:
    """
    Resample motion in time by factor, then pad/truncate to original length.
    motion: (T, J, 3), factor > 0
    """
    T, J, D = motion.shape
    new_T = max(2, int(round(T * factor)))
    src_idx = np.linspace(0, T - 1, new_T)
    orig_idx = np.arange(T)

    warped = np.empty((new_T, J, D), dtype=motion.dtype)
    for j in range(J):
        for d in range(D):
            warped[:, j, d] = np.interp(src_idx, orig_idx, motion[:, j, d])

    if new_T >= T:
        return warped[:T]
    else:
        pad = np.repeat(warped[-1:], T - new_T, axis=0)
        return np.concatenate([warped, pad], axis=0)


class MiniPairDataset(Dataset):
    """
    Constructs (source, target) pairs from Ted's editing/ runs.

    Strategy:
        1. Discover all editing/ clips with known spectrum values
        2. Sort by spectrum ascending
        3. For every pair (lower_spec, higher_spec), yield (lower as source,
           higher as target)
        4. Multiply by N_AUGMENT_PER_CLIP with random augmentations

    Also yields "identity pairs" — source and target from the SAME clip
    (i.e. the GainNet should learn α ≈ 1 in that case).
    """

    def __init__(self, cfg, split: str = "train") -> None:
        self.cfg = cfg
        self.split = split

        self.clips = discover_editing_clips(cfg.EDITING_DIR)
        if len(self.clips) < 2:
            raise RuntimeError(
                f"Found only {len(self.clips)} usable clips in {cfg.EDITING_DIR}. "
                f"Need at least 2 to form pairs."
            )

        # Load all clips into memory (they're small)
        self.clip_data = {}
        for clip in self.clips:
            arr = np.load(clip["path"]).astype(np.float32)   # (T, J, 3)
            # Ensure standard length
            T = cfg.NUM_FRAMES
            if arr.shape[0] > T:
                arr = arr[:T]
            elif arr.shape[0] < T:
                pad = np.repeat(arr[-1:], T - arr.shape[0], axis=0)
                arr = np.concatenate([arr, pad], axis=0)
            self.clip_data[clip["name"]] = arr

        # Build the list of source→target pair definitions
        self.pair_defs = self._build_pair_defs()

        print(f"[MiniPairDataset:{split}] {len(self.clips)} clips, "
              f"{len(self.pair_defs)} augmented pairs")

    def _build_pair_defs(self) -> list[tuple[str, str, float]]:
        """
        Returns list of (source_name, target_name, spectrum_delta) triplets.
        """
        pairs = []
        n = len(self.clips)
        # All ordered pairs where source_spec < target_spec
        for i in range(n):
            for j in range(n):
                if i == j:
                    # Self-pair — identity should be preserved (α ≈ 1)
                    pairs.append((
                        self.clips[i]["name"],
                        self.clips[j]["name"],
                        0.0,
                    ))
                elif self.clips[i]["spectrum"] < self.clips[j]["spectrum"]:
                    delta = self.clips[j]["spectrum"] - self.clips[i]["spectrum"]
                    pairs.append((
                        self.clips[i]["name"],
                        self.clips[j]["name"],
                        delta,
                    ))
        # Expand by augmentation factor
        return pairs * self.cfg.N_AUGMENT_PER_CLIP

    def __len__(self) -> int:
        return len(self.pair_defs)

    def _augment(self, motion: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply random time-warp + joint noise."""
        # Time warp
        lo, hi = self.cfg.TIME_WARP_RANGE
        factor = float(rng.uniform(lo, hi))
        m = _time_warp(motion, factor)
        # Joint noise
        noise = rng.standard_normal(m.shape).astype(np.float32) * self.cfg.JOINT_NOISE_STD
        m = m + noise
        return m

    def __getitem__(self, idx: int) -> dict:
        src_name, tgt_name, delta = self.pair_defs[idx]
        rng = np.random.default_rng(idx)

        src = self.clip_data[src_name].copy()
        tgt = self.clip_data[tgt_name].copy()

        src = self._augment(src, rng)
        tgt = self._augment(tgt, rng)

        return {
            "source": torch.from_numpy(src).float(),          # (T, J, 3)
            "target": torch.from_numpy(tgt).float(),          # (T, J, 3)
            "spectrum_delta": torch.tensor(delta, dtype=torch.float32),
            "src_name": src_name,
            "tgt_name": tgt_name,
        }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config_mini import cfg

    ds = MiniPairDataset(cfg)
    print(f"Dataset length: {len(ds)}")
    item = ds[0]
    print(f"Sample:")
    print(f"  source:         {tuple(item['source'].shape)}")
    print(f"  target:         {tuple(item['target'].shape)}")
    print(f"  spectrum_delta: {item['spectrum_delta'].item():.3f}")
    print(f"  {item['src_name']} → {item['tgt_name']}")
