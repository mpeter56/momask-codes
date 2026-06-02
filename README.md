# MoMask Semantic Spectrum

**Conditioning Motion Generation on Continuous Kinematic Dimensions**

*Maura Peterson · Xiaoyue Lian*

Built on top of [MoMask (Guo et al., CVPR 2024)](https://arxiv.org/abs/2312.00063).

---

## What This Is

This project extends MoMask with a **semantic spectrum conditioning system** that lets you control motion style using continuous numeric labels rather than free-form text alone.

Instead of writing "a person dances a little bit," you write:

```
[dance:0.40][walk:0.91] A person dances with a walking base.
```

The model was fine-tuned on the HumanML3D dataset augmented with 29,799 synthetic spectrum captions, enabling it to respond to these structured tags and produce a smooth walk-to-dance intensity progression.

The primary demo is a **video-to-motion pipeline**: drop in an MP4 of a person walking, and the system generates 10 versions of that motion ranging from pure walking to full dance, compiled into a progression reel and a side-by-side quad grid.

---

## Quick Start

### 1. Install dependencies

```bash
conda env create -f environment.yml
conda activate momask
pip install git+https://github.com/openai/CLIP.git
pip install mediapipe opencv-python scipy
```

### 2. Download pretrained models

```bash
bash prepare/download_models.sh
```

The fine-tuned MaskTransformer and ResidualTransformer checkpoints should be placed at:

```
checkpoints/t2m/MaskTransformer/model/latest.tar
checkpoints/t2m/ResTransformer/model/latest.tar
```

### 3. Run the full pipeline

Drop your video into `input_videos/` and run:

```bash
python scripts/video_to_spectrum.py
```

Or specify the file by name:

```bash
python scripts/video_to_spectrum.py --video my-walk.mp4
```

On first run, the MediaPipe pose model (~25 MB) is downloaded automatically to `video_bridge/`.

**Output** is saved to `outputs/<video_name>/`:

```
outputs/my-walk/
    dance_mode_0.09_repeat0.mp4    # step 1 of 10 (low dance)
    dance_mode_0.11_repeat0.mp4
    dance_mode_0.28_repeat0.mp4
    dance_mode_0.39_repeat0.mp4
    dance_mode_0.40_repeat0.mp4
    dance_mode_0.51_repeat0.mp4
    dance_mode_0.63_repeat0.mp4
    dance_mode_0.74_repeat0.mp4
    dance_mode_0.80_repeat0.mp4
    dance_mode_1.00_repeat0.mp4    # step 10 of 10 (full dance)
    progression.mp4                # all 10 in sequence
    quad.mp4                       # 2x5 grid of all 10 simultaneously
```

### Skip stages you have already run

```bash
# Already have momask_input.npz — skip MediaPipe
python scripts/video_to_spectrum.py --video my-walk.mp4 --skip_mediapipe

# Already ran inference — just rebuild the videos
python scripts/video_to_spectrum.py --video my-walk.mp4 --skip_mediapipe --skip_inference

# Save a MediaPipe skeleton overlay for sanity-checking the pose extraction
python scripts/video_to_spectrum.py --video my-walk.mp4 --visualize
```

---

## Intensity Steps

The pipeline uses ten intensity values derived from the mode of the spectrum score distribution across the HumanML3D corpus. Walk is held constant at 0.91 throughout.

| Step | Dance | Prompt tail |
|------|-------|-------------|
| 1 | 0.09 | A person walks forward. |
| 2 | 0.11 | A person walks forward. |
| 3 | 0.28 | A person dances with a walking base. |
| 4 | 0.39 | A person dances with a walking base. |
| 5 | 0.40 | A person dances with a walking base. |
| 6 | 0.51 | A person dances with a walking base. |
| 7 | 0.63 | A person dances with a walking base. |
| 8 | 0.74 | A person dances with a walking base. |
| 9 | 0.80 | A person dances expressively. |
| 10 | 1.00 | A person dances expressively. |

---

## Running Individual Inference Commands

To run a single intensity step manually:

```bash
python edit_t2m.py \
  --gpu_id 0 \
  --ext dance_mode_0.40 \
  --name MaskTransformer \
  --dataset_name t2m \
  --res_name ResTransformer \
  --text_prompt "[dance:0.40][walk:0.91] A person dances with a walking base." \
  --source_motion momask_input.npz \
  --mask_edit_section 0.0,1.0 \
  --use_res_model \
  --repeat_times 3
```

Output is saved to `editing/dance_mode_0.40/animations/0/`.

---

## Blind A/B Comparison

To compare our fine-tuned model against the pretrained baseline across all prompt styles (tags+text, text-only, tags-only, spectrum steps):

```bash
cd .claude/worktrees/fervent-satoshi-4e02b9
python scripts/run_and_compare.py --skip_inference
```

> **Note on comparison scripts:** `run_and_compare.py` runs full inference for both models and then assembles the blind review videos. If you have already run inference and only need to rebuild the videos, pass `--skip_inference`. The separate `blind_comparison.py` (legacy) only assembles videos and assumes all `editing/` output folders already exist.

This produces `blind_review/A.mp4`, `blind_review/B.mp4`, and `blind_review/key.txt` (open the key only after reviewing both videos).

---

## Spectrum Caption Format

The eight kinematic dimensions are scored independently on [0, 1] and formatted as tag prefixes:

```
[dance:0.82][walk:0.18] A person walks forward with rhythmic upper body movement.
```

Dimensions: `dance`, `walk`, `run`, `jump`, `stand`, `gesture`, `spin`, `crouch`

Tags are alphabetically ordered. Only dimensions above 0.05 appear. At most four tags are included per caption. The natural language tail is generated deterministically from the top two scoring dimensions.

---

## Training

The MaskTransformer and ResidualTransformer were fine-tuned on HumanML3D augmented with synthetic spectrum captions (50/50 mixing ratio). The RVQVAE tokenizer is the pretrained MoMask checkpoint and was not modified.

### Re-run spectrum augmentation

```bash
python scripts/run_spectrum_pipeline.py \
  --motion_dir dataset/HumanML3D/new_joints \
  --text_dir   dataset/HumanML3D/texts \
  --out_dir    semantic_spectrum_data \
  --fit_calibration \
  --n_synthetic 3 \
  --overwrite
```

### Fine-tune the transformers

```bash
# Stage 2: MaskTransformer
python train_t2m_transformer.py \
  --dataset_name t2m \
  --name MaskTransformer \
  --vq_name RVQVAE \
  --gpu_id 0 \
  --is_continue \
  --max_epoch 2000

# Stage 3: ResidualTransformer
python train_res_transformer.py \
  --dataset_name t2m \
  --name ResTransformer \
  --vq_name RVQVAE \
  --gpu_id 0 \
  --is_continue \
  --share_weight \
  --max_epoch 2000
```

---

## Project Structure

```
scripts/
    video_to_spectrum.py      # Main pipeline: video → motion → videos
    run_and_compare.py        # Blind A/B comparison builder
    run_spectrum_pipeline.py  # Spectrum augmentation CLI
    visualize_spectrum.py     # Score distribution plots

video_bridge/
    video_to_humanml3d.py     # MediaPipe → HumanML3D feature vector

semantic_spectrum/
    analyzer.py               # Scores motions on 8 kinematic dimensions
    synthesizer.py            # Generates spectrum caption strings
    augment.py                # Writes _spec_*.txt files alongside originals
    dimensions.py             # Raw kinematic feature extractors

input_videos/                 # Drop your MP4 here
outputs/                      # Generated videos appear here
editing/                      # MoMask edit_t2m.py output folders
```

---

## Acknowledgements

Built on [MoMask](https://github.com/EricGuo5513/momask-codes) (Guo et al., CVPR 2024).
Additional dependencies: [HumanML3D](https://github.com/EricGuo5513/HumanML3D), [CLIP](https://github.com/openai/CLIP), [MediaPipe](https://github.com/google-ai-edge/mediapipe), [T2M-GPT](https://github.com/Mael-zys/T2M-GPT), [MDM](https://github.com/GuyTevet/motion-diffusion-model).

## License

MIT — see [LICENSE](LICENSE).
Original MoMask code: MIT License, Copyright (c) Eric Guo.
