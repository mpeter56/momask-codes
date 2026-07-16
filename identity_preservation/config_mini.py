"""
Mini config for overnight local training.

This is the SIMPLIFIED variant that operates directly in joint space
(196, 22, 3) — NOT the full RVQ-codebook GainNet. See REPORT.md and the
docstring in models/mini_gain_net.py for why this is defensible as
"proof of architecture" for the defense.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MiniConfig:
    # ------------------------------------------------------------------
    # Paths — Ted's editing/ workspace holds our training data
    # ------------------------------------------------------------------
    PROJECT_ROOT: Path = Path.home() / "identity_preservation_mini"
    MOMASK_REPO: Path = Path.home() / "Downloads" / "momask-codes-main"
    EDITING_DIR: Path = Path.home() / "Downloads" / "momask-codes-main" / "editing"
    EXAMPLE_DATA_DIR: Path = Path.home() / "Downloads" / "momask-codes-main" / "example_data"

    OUTPUTS_ROOT: Path = PROJECT_ROOT / "outputs"
    CHECKPOINTS_ROOT: Path = OUTPUTS_ROOT / "checkpoints"
    SAMPLES_ROOT: Path = OUTPUTS_ROOT / "samples"
    LOGS_ROOT: Path = OUTPUTS_ROOT / "logs"

    # ------------------------------------------------------------------
    # Motion representation
    # ------------------------------------------------------------------
    NUM_FRAMES: int = 196     # HumanML3D standard clip length
    NUM_JOINTS: int = 22      # SMPL-H body joints
    JOINT_DIM: int = 3        # xyz

    # ------------------------------------------------------------------
    # Mini GainNet architecture — small enough for MPS overnight
    # ------------------------------------------------------------------
    GAIN_LATENT_DIM: int = 128
    GAIN_N_HEADS: int = 4
    GAIN_N_LAYERS: int = 2
    GAIN_DROPOUT: float = 0.1

    # ------------------------------------------------------------------
    # Training — deliberately small for overnight MPS
    # ------------------------------------------------------------------
    BATCH_SIZE: int = 2       # tiny — we only have ~10 real clips
    LR: float = 2e-4
    NUM_EPOCHS: int = 200     # small dataset → many epochs, still overnight-friendly
    WARMUP_STEPS: int = 100
    GRAD_CLIP: float = 1.0

    # Data augmentation — needed because we have so few real clips
    N_AUGMENT_PER_CLIP: int = 20   # 20 augmented pairs per real clip
    TIME_WARP_RANGE: tuple = (0.9, 1.1)
    JOINT_NOISE_STD: float = 0.005

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    SAMPLE_EVERY_N_EPOCHS: int = 20
    N_SAMPLES: int = 4

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    DEVICE: str = "mps"       # MacBook. Change to "cuda" on RunPod, "cpu" for CI.
    SEED: int = 42


cfg = MiniConfig()


if __name__ == "__main__":
    print("=" * 60)
    print("Mini Config")
    print("=" * 60)
    for k, v in cfg.__dict__.items():
        print(f"  {k:25s} = {v}")
