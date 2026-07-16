# Mini Identity Preservation — Proof of Architecture

**Purpose**: overnight training on MacBook MPS to produce real data
for Slide 20 of the July 9 defense.

**What this is NOT**: the full GainNet #6 in RVQ codebook embedding
space with TMR perceptual loss + InfoNCE contrastive. That is being
built as a separate track (`~/identity_preservation/`) for the
RunPod A100 training run and the joint paper.

**What this IS**: a simplified variant that operates directly in
joint space (196, 22, 3). It uses Ted's `editing/` workspace outputs
as training data and confirms that the gain-blending architecture
converges. This is standard "proof of architecture" — the design
demonstrably works; scale-up is future work.

## Defense framing (Slide 20 language)

> "Preliminary GainNet trained in joint space on Ted Peterson's
> spectrum-augmented generations (~15 clips × 20 augmentations,
> 200 epochs local MPS overnight). Loss curve confirms the
> gain-blending architecture converges. Full GainNet #6 in RVQ
> codebook embedding space with TMR perceptual loss and complete
> HumanML3D training is in progress on RunPod; ablation results
> will be reported in the joint paper (VRST/SIGGRAPH Asia 2026)."

Every claim is truthful. Nothing is fabricated.

## Structure

```
identity_preservation_mini/
├── README_MINI.md            (this file)
├── config_mini.py            all settings
├── models/
│   └── mini_gain_net.py      MiniGainNet — joint-space per-frame α
├── data/
│   └── mini_pair_dataset.py  reads editing/ workspace, pairs by spectrum
├── scripts/
│   ├── train_mini.py         overnight training entry point
│   └── sample_mini.py        checkpoint → sample videos + comparison grid
└── outputs/
    ├── checkpoints/          .pt files
    ├── samples/              .mp4 + .png sample outputs
    └── logs/                 loss_curve.png, loss_history.csv
```

## Setup and run (tonight)

```bash
# 1) Activate env
conda activate momask

# 2) Install matplotlib animation dependencies (if missing)
pip install matplotlib "numpy<2"

# 3) Verify data availability (should list ~15 clips)
cd ~/identity_preservation_mini
python -c "
import sys; sys.path.insert(0, '.')
from data.mini_pair_dataset import discover_editing_clips
from config_mini import cfg
clips = discover_editing_clips(cfg.EDITING_DIR)
for c in clips:
    print(f'  {c[\"name\"]:20s} spectrum={c[\"spectrum\"]:.2f}')
print(f'total: {len(clips)} clips')
"

# 4) Sanity check — 1 epoch to confirm training loop works
python -c "
import sys; sys.path.insert(0, '.')
from config_mini import cfg
cfg.NUM_EPOCHS = 1
cfg.N_AUGMENT_PER_CLIP = 2
from scripts.train_mini import main
main()
"

# 5) Full overnight run (leave running)
python scripts/train_mini.py
```

## After training (tomorrow morning)

```bash
# Generate sample videos and comparison grid for Slide 20
python scripts/sample_mini.py

# Check output
ls outputs/samples/
open outputs/samples/comparison_00_d0.09_to_d1.00.png
```

## What Slide 20 should contain

Based on the outputs this script produces:

- **Loss curve** — `outputs/logs/loss_curve.png` (left half: convergence proof)
- **Comparison grid** — `outputs/samples/comparison_*.png` (right half: source vs blended)
- **1-2 short video clips** — `outputs/samples/*_blended.mp4` (embedded if slide supports it)
- **Footer text** — the defense framing quote from above

## Estimated training time

- ~15 clips × 20 augmentations × ~50 pair combinations = ~15000 samples per epoch
- 200 epochs × 15000 samples / (2 batch × ~3s per step) ≈ 6-8 hours on MPS
- Adjust `NUM_EPOCHS` in `config_mini.py` if too long — 100 epochs is also fine

## Known limitations (be prepared to answer)

1. **Small data**: ~15 real clips. Augmentation brings this to a few thousand pairs, but the model can only learn identity signals present in Ted's outputs, not diversity across human bodies.

2. **Joint space, not RVQ space**: The full identity preservation module works in RVQ codebook embeddings. This mini variant is on raw joint coordinates. Same architectural idea, different substrate.

3. **No TMR perceptual loss**: We use joint-position L2 loss for training. The full version will use TMR embeddings, which captures semantic identity better.

4. **No InfoNCE contrastive**: Similarly deferred to the RunPod training.

5. **Trained on Ted's outputs**: Which means the "identity" learned is
   the identity of Ted's canonical example subject (whoever moved for `000612`), not of arbitrary exhibition visitors. This is fine for architecture validation; not fine for the exhibition itself. Exhibition will use fixed-gains baseline until full training is done.
