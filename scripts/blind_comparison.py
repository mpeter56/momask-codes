"""
Produce two blind quad videos for model comparison.

Video layout (same for both):
    TL: Tags + Text  |  TR: Text Only
    BL: Tags Only    |  BR: Spectrum Edit

One video uses our fine-tuned model outputs.
One video uses the pretrained model outputs.
They are randomly named A and B. A key file records which is which.

Usage:
    python scripts/blind_comparison.py --editing_dir editing --out_dir blind_review
"""

import argparse
import random
import subprocess
import tempfile
import re
from pathlib import Path

PERCENTILES = ['0.09', '0.11', '0.28', '0.39', '0.40', '0.51', '0.63', '0.74', '0.80', '1.00']

TUNED = {
    'tl': 'dance_mode_{p}',
    'tr': 'no-tag_{p}',
    'bl': 'no-txt_dance_mode_{p}',
}
TUNED_SPECTRUM_MAP = {
    '0.09': 'spectrum_edit_d10',  '0.11': 'spectrum_edit_d10',
    '0.28': 'spectrum_edit_d30',  '0.39': 'spectrum_edit_d40',
    '0.40': 'spectrum_edit_d40',  '0.51': 'spectrum_edit_d50',
    '0.63': 'spectrum_edit_d60',  '0.74': 'spectrum_edit_d70',
    '0.80': 'spectrum_edit_d80',  '1.00': 'spectrum_edit_d100',
}

PRETRAINED = {
    'tl': 'pretrained_tags_text_{p}',
    'tr': 'pretrained_text_only_{p}',
    'bl': 'pretrained_tags_only_{p}',
}
PRETRAINED_SPECTRUM_MAP = {
    '0.09': 'pretrained_spectrum_d10',  '0.11': 'pretrained_spectrum_d10',
    '0.28': 'pretrained_spectrum_d30',  '0.39': 'pretrained_spectrum_d40',
    '0.40': 'pretrained_spectrum_d40',  '0.51': 'pretrained_spectrum_d50',
    '0.63': 'pretrained_spectrum_d60',  '0.74': 'pretrained_spectrum_d70',
    '0.80': 'pretrained_spectrum_d80',  '1.00': 'pretrained_spectrum_d100',
}

PANEL_LABELS = {
    'tl': 'Tags+Text',
    'tr': 'Text Only',
    'bl': 'Tags Only',
    'br': 'Spectrum Edit',
}


def ffmpeg(*args, check=True):
    cmd = ['ffmpeg', '-y', '-loglevel', 'error'] + list(args)
    subprocess.run(cmd, check=check)


def find_first_repeat(folder: Path):
    anim = folder / 'animations' / '0'
    if not anim.exists():
        return None
    candidates = sorted(p for p in anim.glob('*.mp4')
                        if '_source_' not in p.name and '_ik.' not in p.name)
    return candidates[0] if candidates else None


def add_label(src: Path, label: str, pct: str, tmp_dir: str) -> Path:
    safe_name = re.sub(r'[^a-zA-Z0-9]', '_', f'{label}_{pct}')
    out = Path(tmp_dir) / f'{safe_name}_labeled.mp4'
    safe_text = label.replace(':', '\\:').replace("'", "\\'")
    ffmpeg(
        '-i', str(src),
        '-vf', f"drawtext=text='{safe_text} d={pct}':fontsize=16:fontcolor=white:"
               f"x=10:y=10:box=1:boxcolor=black@0.6:boxborderw=4",
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '22', '-an', str(out)
    )
    return out


def scale_video(src: Path, w: int, h: int, suffix: str, tmp_dir: str) -> Path:
    out = Path(tmp_dir) / f'{suffix}_scaled.mp4'
    ffmpeg(
        '-i', str(src),
        '-vf', f'scale={w}:{h}:force_original_aspect_ratio=decrease,'
               f'pad={w}:{h}:(ow-iw)/2:(oh-ih)/2',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '22', '-an', str(out)
    )
    return out


def make_quad(tl, tr, bl, br, name, tmp_dir) -> Path:
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


def concat_clips(clips, out: Path, tmp_dir: str):
    list_file = Path(tmp_dir) / 'concat.txt'
    list_file.write_text('\n'.join(f"file '{str(p)}'" for p in clips))
    ffmpeg('-f', 'concat', '-safe', '0', '-i', str(list_file),
           '-c:v', 'libx264', '-preset', 'fast', '-crf', '22', str(out))


def build_model_video(editing_dir, group_map, spectrum_map, model_tag, pw, ph, tmp_dir):
    clips = []
    for pct in PERCENTILES:
        print(f'  [{model_tag}] percentile {pct}')
        videos = {}
        for pos in ['tl', 'tr', 'bl']:
            v = find_first_repeat(editing_dir / group_map[pos].format(p=pct))
            if v is None:
                print(f'    WARNING: missing {pos} for {pct}')
            videos[pos] = v
        videos['br'] = find_first_repeat(editing_dir / spectrum_map[pct])

        if any(v is None for v in videos.values()):
            print(f'    Skipping {pct} — missing video(s)')
            continue

        panels = {}
        for pos, vid in videos.items():
            labeled = add_label(vid, PANEL_LABELS[pos], pct, tmp_dir)
            panels[pos] = scale_video(labeled, pw, ph,
                                      f'{model_tag}_{pos}_{pct}', tmp_dir)

        quad = make_quad(panels['tl'], panels['tr'],
                         panels['bl'], panels['br'],
                         f'{model_tag}_pct_{pct}', tmp_dir)
        clips.append(quad)

    return clips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_dir', default='editing')
    parser.add_argument('--out_dir', default='blind_review')
    parser.add_argument('--panel_w', type=int, default=480)
    parser.add_argument('--panel_h', type=int, default=360)
    args = parser.parse_args()

    editing_dir = Path(args.editing_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        print('Building fine-tuned model video...')
        tuned_clips = build_model_video(
            editing_dir, TUNED, TUNED_SPECTRUM_MAP,
            'tuned', args.panel_w, args.panel_h, tmp_dir
        )

        print('\nBuilding pretrained model video...')
        pre_clips = build_model_video(
            editing_dir, PRETRAINED, PRETRAINED_SPECTRUM_MAP,
            'pre', args.panel_w, args.panel_h, tmp_dir
        )

        # Randomly assign A / B
        assignment = random.choice([
            {'A': 'fine-tuned', 'B': 'pretrained', 'A_clips': tuned_clips, 'B_clips': pre_clips},
            {'A': 'pretrained', 'B': 'fine-tuned', 'A_clips': pre_clips,   'B_clips': tuned_clips},
        ])

        print('\nConcatenating video A...')
        concat_clips(assignment['A_clips'], out_dir / 'A.mp4', tmp_dir)

        print('Concatenating video B...')
        concat_clips(assignment['B_clips'], out_dir / 'B.mp4', tmp_dir)

    # Write key file
    key_path = out_dir / 'key.txt'
    key_path.write_text(
        f"Blind Comparison Key\n"
        f"====================\n"
        f"A = {assignment['A']} model\n"
        f"B = {assignment['B']} model\n"
    )

    print(f'\nDone.')
    print(f'  editing_combined/A.mp4 = {assignment["A"]} model')
    print(f'  editing_combined/B.mp4 = {assignment["B"]} model')
    print(f'  Key saved to {key_path}')


if __name__ == '__main__':
    main()
