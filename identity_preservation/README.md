# identity_preservation

Motion identity preservation on top of MoMask + MediaPipe. Extracts pose from a
webcam recording, blends it against MoMask-generated dance motion across a
spectrum of intensities, renders side-by-side comparisons, and quantifies
identity retention via DTW alignment.

## What's in the box

- **`scripts/webcam_spectrum.py`** — MediaPipe pose extraction from a webcam
  video, converted to SMPL-22 joint format, amplified per spectrum intensity.
  Identity preserved by construction (each output is the same recorded pose
  scaled around its mean).
- **`scripts/walk_to_dance.py`** — Linear blend between two motion sources
  (a "walk" and a "dance") across the spectrum. Accepts either video or
  pre-computed joint `.npy` files.
- **`scripts/similarity_align.py`** — Frame-by-frame DTW alignment between
  two motion sequences. Outputs alignment JSON, distance-matrix heatmap, and
  a synchronised side-by-side comparison video.
- **`scripts/render_spectrum_tour.py`** — Polished 100-second tour piece
  cycling through 10 spectrum values with spectrum-bar UI overlay.
- **`scripts/run_spectrum_tour.sh`** — end-to-end orchestrator for the tour.
- **`scripts/extract_pose_json.py`** — MediaPipe pose → JSON for browser panels.
- **`scripts/*_panel.html`** + **`scripts/run_*_panel.sh`** — three browser
  dashboards for interactive review (own_panel, spectrum_panel,
  tour_segmented_panel).
- **`models/mini_gain_net.py`** + **`train_mini.py`** — trained per-frame
  identity gain (learned α blend). Not required for the deterministic pipelines
  above.

## Quick start

```bash
conda activate momask
pip install -r identity_preservation/requirements.txt

# 1. Extract + amplify a webcam clip across the spectrum
python identity_preservation/scripts/webcam_spectrum.py \
    --webcam path/to/your_clip.mp4 --amp-boost 2.0

# 2. Render the polished 100-second tour
python identity_preservation/scripts/render_spectrum_tour.py \
    --dance-values 0.09,0.11,0.28,0.39,0.40,0.51,0.63,0.74,0.80,1.00 \
    --width 1280 --height 720 \
    --out identity_preservation/outputs/media_art/tour.mp4

# 3. Quantify identity retention against a MoMask dance target
python identity_preservation/scripts/similarity_align.py \
    --original  path/to/your_clip.mp4 \
    --generated identity_preservation/outputs/pipeline_output/blended_d0.40.npy
```

## Concept

Identity in motion is captured as the temporal signature of a specific body:
which joints move, when, how far, in what rhythm. The pipeline treats every
generated dance clip as a perturbation of a reference motion (the source
webcam recording), never as a de novo generation. This guarantees identity
is preserved at every point on the spectrum — the output is always a
transformation of *your* motion, not a substitution.

`similarity_align.py` makes this quantitative: DTW pairs matching poses
between original and generated, and reports per-pair L2 distance. Lower
per-pair distance = higher identity retention.

## Data flow

```
    webcam.mp4
        │
        ▼  MediaPipe Pose (33 landmarks)
    ┌───────────────────────┐
    │  mediapipe_to_smpl22  │  →  (T, 22, 3)  joint positions
    └───────────────────────┘
        │
        ▼  normalise_worldish (torso-scale, Y-up, feet at floor)
    ┌───────────────────────┐
    │  amplify_motion (×d)  │  →  10 spectrum variants
    │  OR                   │
    │  walk_to_dance blend  │
    └───────────────────────┘
        │
        ├─→  render_spectrum_tour.py  →  100-second tour MP4
        └─→  similarity_align.py       →  DTW alignment + side-by-side MP4
```

## Notes

- MediaPipe pose detection requires **mediapipe ≤ 0.10.14** (later releases
  moved to a different API). `requirements.txt` pins this.
- All outputs go to `identity_preservation/outputs/` and are `.gitignore`d.
- Some earlier experimental paths (Blender avatar retargeting via Rokoko,
  MiniGainNet-based blending, deterministic 10-rule module) live alongside
  the current core scripts for reference — see `scripts/` for the full list.
