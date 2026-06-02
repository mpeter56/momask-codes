"""
Combine editing outputs into two review videos:

  1. quad_comparison.mp4
     2x2 grid showing all 4 panels side by side for each percentile step.
     Panels: [tags+text | text only | tags only | source]
     Each percentile plays in sync across all four panels, then steps to the next.

  2. progression.mp4
     Single panel, sequential — plays through each percentile of one chosen
     group from lowest to highest, one after another.
     Default group: dance_mode (tags + text). Change with --progression_group.

Usage:
    python scripts/combine_videos.py
    python scripts/combine_videos.py --editing_dir editing --out_dir editing_combined
"""

import argparse
import subprocess
import tempfile
import os
import re
from pathlib import Path

PERCENTILES = ['0.09', '0.11', '0.28', '0.39', '0.40', '0.51', '0.63', '0.74', '0.80', '1.00']

GROUPS = {
    'tags_text':  'dance_mode_{p}',
    'text_only':  'no-tag_{p}',
    'tags_only':  'no-txt_dance_mode_{p}',
}

LABELS = {
    'tags_text':             'Tuned: Tags + Text',
    'text_only':             'Tuned: Text Only',
    'tags_only':             'Tuned: Tags Only',
    'spectrum_edit':         'Tuned: Spectrum Edit',
    'pre_tags_text':         'Pretrained: Tags + Text',
    'pre_text_only':         'Pretrained: Text Only',
    'pre_tags_only':         'Pretrained: Tags Only',
    'pre_spectrum_edit':     'Pretrained: Spectrum Edit',
}

# Map each mode percentile to the nearest spectrum_edit_d* folder
SPECTRUM_EDIT_MAP = {
    '0.09': ('spectrum_edit_d10',          'pretrained_spectrum_d10'),
    '0.11': ('spectrum_edit_d10',          'pretrained_spectrum_d10'),
    '0.28': ('spectrum_edit_d30',          'pretrained_spectrum_d30'),
    '0.39': ('spectrum_edit_d40',          'pretrained_spectrum_d40'),
    '0.40': ('spectrum_edit_d40',          'pretrained_spectrum_d40'),
    '0.51': ('spectrum_edit_d50',          'pretrained_spectrum_d50'),
    '0.63': ('spectrum_edit_d60',          'pretrained_spectrum_d60'),
    '0.74': ('spectrum_edit_d70',          'pretrained_spectrum_d70'),
    '0.80': ('spectrum_edit_d80',          'pretrained_spectrum_d80'),
    '1.00': ('spectrum_edit_d100',         'pretrained_spectrum_d100'),
}

PRETRAINED_GROUPS = {
    'pre_tags_text':  'pretrained_tags_text_{p}',
    'pre_text_only':  'pretrained_text_only_{p}',
    'pre_tags_only':  'pretrained_tags_only_{p}',
}


def find_first_repeat(folder: Path) -> Path | None:
    """Return the first non-source, non-ik mp4 in the animations subdir."""
    anim_dir = folder / 'animations' / '0'
    if not anim_dir.exists():
        return None
    candidates = sorted(
        p for p in anim_dir.glob('*.mp4')
        if '_source_' not in p.name and '_ik.' not in p.name
    )
    return candidates[0] if candidates else None


def find_source(folder: Path) -> Path | None:
    """Return the source mp4."""
    anim_dir = folder / 'animations' / '0'
    if not anim_dir.exists():
        return None
    candidates = sorted(p for p in anim_dir.glob('*_source_*.mp4'))
    return candidates[0] if candidates else None


def ffmpeg(*args, check=True):
    cmd = ['ffmpeg', '-y', '-loglevel', 'error'] + list(args)
    subprocess.run(cmd, check=check)


def add_label(src: Path, label: str, pct: str, tmp_dir: str) -> Path:
    """Burn label and percentile into video."""
    safe_name = re.sub(r'[^a-zA-Z0-9]', '_', label)
    out = Path(tmp_dir) / f'{src.stem}_{safe_name}_{pct}_labeled.mp4'
    # Escape characters that break ffmpeg drawtext filter parsing
    safe_label = label.replace(':', '\\:').replace("'", "\\'")
    text = f'{safe_label} dance={pct}'
    ffmpeg(
        '-i', str(src),
        '-vf', (
            f"drawtext=text='{text}':fontsize=18:fontcolor=white:"
            f"x=10:y=10:box=1:boxcolor=black@0.5:boxborderw=4"
        ),
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
        '-an', str(out)
    )
    return out


def scale_video(src: Path, width: int, height: int, tmp_dir: str, suffix: str = '') -> Path:
    out = Path(tmp_dir) / f'{src.stem}_scaled{suffix}.mp4'
    ffmpeg(
        '-i', str(src),
        '-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,'
               f'pad={width}:{height}:(ow-iw)/2:(oh-ih)/2',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
        '-an', str(out)
    )
    return out


def stack_quad(tl: Path, tr: Path, bl: Path, br: Path, tmp_dir: str, name: str) -> Path:
    """2x2 grid using ffmpeg hstack + vstack."""
    top = Path(tmp_dir) / f'{name}_top.mp4'
    bot = Path(tmp_dir) / f'{name}_bot.mp4'
    out = Path(tmp_dir) / f'{name}_quad.mp4'

    ffmpeg('-i', str(tl), '-i', str(tr),
           '-filter_complex', '[0:v][1:v]hstack=inputs=2[v]',
           '-map', '[v]', '-c:v', 'libx264', '-preset', 'fast', '-crf', '22', str(top))

    ffmpeg('-i', str(bl), '-i', str(br),
           '-filter_complex', '[0:v][1:v]hstack=inputs=2[v]',
           '-map', '[v]', '-c:v', 'libx264', '-preset', 'fast', '-crf', '22', str(bot))

    ffmpeg('-i', str(top), '-i', str(bot),
           '-filter_complex', '[0:v][1:v]vstack=inputs=2[v]',
           '-map', '[v]', '-c:v', 'libx264', '-preset', 'fast', '-crf', '22', str(out))

    return out


def concat_videos(clips: list[Path], out: Path, tmp_dir: str):
    """Concatenate a list of mp4s into one."""
    list_file = Path(tmp_dir) / 'concat_list.txt'
    list_file.write_text('\n'.join(f"file '{str(p)}'" for p in clips))
    ffmpeg('-f', 'concat', '-safe', '0', '-i', str(list_file),
           '-c:v', 'libx264', '-preset', 'fast', '-crf', '22', str(out))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_dir', default='editing')
    parser.add_argument('--out_dir', default='editing_combined')
    parser.add_argument('--panel_w', type=int, default=480, help='Width of each panel')
    parser.add_argument('--panel_h', type=int, default=360, help='Height of each panel')
    parser.add_argument('--progression_group', default='tags_text',
                        choices=list(GROUPS.keys()),
                        help='Which group to use for the progression video')
    args = parser.parse_args()

    editing_dir = Path(args.editing_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        quad_clips = []       # tuned vs pretrained quad (tags+text top, spectrum_edit bottom)
        progression_clips = []  # tuned tags+text progression

        for pct in PERCENTILES:
            print(f'\nProcessing percentile {pct}...')

            # --- Tuned model videos ---
            tuned_videos = {}
            for k in GROUPS:
                v = find_first_repeat(editing_dir / GROUPS[k].format(p=pct))
                if v is None:
                    print(f'  WARNING: missing tuned {k} at {pct}')
                tuned_videos[k] = v

            se_tuned_folder, se_pre_folder = SPECTRUM_EDIT_MAP[pct]
            tuned_videos['spectrum_edit'] = find_first_repeat(editing_dir / se_tuned_folder)

            # --- Pretrained model videos ---
            pre_videos = {}
            for k in PRETRAINED_GROUPS:
                v = find_first_repeat(editing_dir / PRETRAINED_GROUPS[k].format(p=pct))
                if v is None:
                    print(f'  WARNING: missing pretrained {k} at {pct}')
                pre_videos[k] = v
            pre_videos['pre_spectrum_edit'] = find_first_repeat(editing_dir / se_pre_folder)

            # Quad: TL=tuned tags+text, TR=pretrained tags+text
            #       BL=tuned spectrum_edit, BR=pretrained spectrum_edit
            quad_map = {
                'tl': (tuned_videos.get('tags_text'),     'tags_text'),
                'tr': (pre_videos.get('pre_tags_text'),   'pre_tags_text'),
                'bl': (tuned_videos.get('spectrum_edit'), 'spectrum_edit'),
                'br': (pre_videos.get('pre_spectrum_edit'), 'pre_spectrum_edit'),
            }

            if any(v is None for v, _ in quad_map.values()):
                print(f'  Skipping quad for {pct} — missing videos')
            else:
                panels = {}
                for pos, (vid, key) in quad_map.items():
                    labeled = add_label(vid, LABELS[key], pct, tmp_dir)
                    panels[pos] = scale_video(labeled, args.panel_w, args.panel_h,
                                              tmp_dir, suffix=f'_{pos}_{pct}')
                    print(f'  {pos} ({key}): {vid.name}')

                quad = stack_quad(panels['tl'], panels['tr'],
                                  panels['bl'], panels['br'],
                                  tmp_dir, name=f'pct_{pct}')
                quad_clips.append(quad)

            # Progression: tuned tags+text only
            prog_v = tuned_videos.get('tags_text')
            if prog_v:
                prog_labeled = add_label(prog_v, LABELS['tags_text'], pct, tmp_dir)
                prog_scaled = scale_video(prog_labeled, args.panel_w * 2, args.panel_h * 2,
                                          tmp_dir, suffix=f'_prog_{pct}')
                progression_clips.append(prog_scaled)

        # Concatenate quad clips
        if quad_clips:
            out_quad = out_dir / 'quad_comparison.mp4'
            print(f'\nBuilding quad comparison: {out_quad}')
            concat_videos(quad_clips, out_quad, tmp_dir)
            print(f'Saved {out_quad}')

        # Concatenate progression clips
        if progression_clips:
            out_prog = out_dir / 'progression.mp4'
            print(f'Building progression: {out_prog}')
            concat_videos(progression_clips, out_prog, tmp_dir)
            print(f'Saved {out_prog}')

    print('\nDone.')


if __name__ == '__main__':
    main()
