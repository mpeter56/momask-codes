"""
Visualize spectrum caption statistics from generated _spec_*.txt files.

Produces:
  1. Score distribution histogram for each kinematic dimension
  2. Bar chart of dimension occurrence counts
  3. Co-occurrence matrix heatmap
  4. Mode score per 10% bin for all 8 dimensions

Usage:
    python scripts/visualize_spectrum.py
    python scripts/visualize_spectrum.py --text_dir dataset/HumanML3D/texts --out_dir spectrum_viz
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DIMENSIONS = ['dance', 'jump', 'kick', 'run', 'spin', 'stand', 'walk', 'wave']
TAG_RE = re.compile(r'\[([a-z]+):([0-9.]+)\]')


def parse_spec_files(text_dir: Path):
    """
    Read all *_spec_*.txt files and return:
        scores      : {dim: [score, score, ...]}
        cooccurrence: {(dim_a, dim_b): count}  (dim_a <= dim_b)
    """
    scores = defaultdict(list)
    cooccurrence = defaultdict(int)

    spec_files = sorted(text_dir.rglob('*_spec_*.txt'))
    print(f'Found {len(spec_files)} spectrum caption files in {text_dir}')

    for path in spec_files:
        try:
            line = path.read_text(encoding='utf-8').strip().split('#')[0]
        except Exception:
            continue

        found = TAG_RE.findall(line)
        dims_in_file = []
        for dim, val in found:
            if dim in DIMENSIONS:
                scores[dim].append(float(val))
                dims_in_file.append(dim)

        # Co-occurrences (unordered pairs)
        dims_in_file = sorted(set(dims_in_file))
        for i, a in enumerate(dims_in_file):
            for b in dims_in_file[i:]:
                cooccurrence[(a, b)] += 1

    return scores, cooccurrence


def plot_score_distributions(scores, out_dir: Path):
    """One histogram per dimension, arranged in a 2x4 grid."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    fig.suptitle('Spectrum Score Distributions per Dimension', fontsize=15, fontweight='bold')

    for ax, dim in zip(axes.flat, DIMENSIONS):
        vals = scores.get(dim, [])
        if vals:
            ax.hist(vals, bins=40, color='steelblue', edgecolor='white', linewidth=0.4)
            ax.axvline(np.mean(vals), color='tomato', linewidth=1.5, linestyle='--', label=f'mean {np.mean(vals):.2f}')
            ax.legend(fontsize=8)
        ax.set_title(dim, fontweight='bold')
        ax.set_xlabel('Score')
        ax.set_ylabel('Count')
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    plt.tight_layout()
    out = out_dir / 'score_distributions.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved {out}')


def plot_occurrence_bar(scores, out_dir: Path):
    """Bar chart: how many captions each dimension appears in."""
    counts = [len(scores.get(d, [])) for d in DIMENSIONS]
    colors = plt.cm.tab10.colors[:len(DIMENSIONS)]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(DIMENSIONS, counts, color=colors, edgecolor='white', linewidth=0.6)
    ax.bar_label(bars, fmt='%d', padding=4, fontsize=9)
    ax.set_title('Dimension Occurrence Counts', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Number of captions')
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = out_dir / 'occurrence_bar.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved {out}')


def plot_cooccurrence_matrix(cooccurrence, out_dir: Path):
    """Heatmap of dimension co-occurrences."""
    n = len(DIMENSIONS)
    matrix = np.zeros((n, n), dtype=int)
    idx = {d: i for i, d in enumerate(DIMENSIONS)}

    for (a, b), count in cooccurrence.items():
        i, j = idx[a], idx[b]
        matrix[i, j] += count
        if i != j:
            matrix[j, i] += count

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, ax=ax, label='Co-occurrence count')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(DIMENSIONS, rotation=40, ha='right')
    ax.set_yticklabels(DIMENSIONS)
    ax.set_title('Dimension Co-occurrence Matrix', fontsize=14, fontweight='bold')

    # Annotate cells
    thresh = matrix.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{matrix[i, j]:,}',
                    ha='center', va='center', fontsize=8,
                    color='white' if matrix[i, j] > thresh else 'black')

    plt.tight_layout()
    out = out_dir / 'cooccurrence_matrix.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved {out}')


def plot_mode_per_bin(scores, out_dir: Path):
    """
    One bar chart per dimension showing the count of scores in each 10% bin,
    with the mode value annotated on each bar.
    Saved as mode_per_bin_<dim>.png for each dimension.
    """
    bins = np.arange(0, 1.1, 0.1)
    bin_labels = [f'{int(b*100)}-{int((b+0.1)*100)}%' for b in bins[:-1]]
    colors = plt.cm.tab10.colors

    for di, dim in enumerate(DIMENSIONS):
        vals = np.array(scores.get(dim, []))
        if len(vals) == 0:
            continue

        counts = []
        modes = []
        for bi in range(len(bins) - 1):
            lo, hi = bins[bi], bins[bi + 1]
            mask = (vals >= lo) & (vals <= hi if hi >= 1.0 else vals < hi)
            bucket = vals[mask]
            counts.append(len(bucket))
            if len(bucket) == 0:
                modes.append(None)
            else:
                rounded = np.round(bucket, 2)
                unique, cnts = np.unique(rounded, return_counts=True)
                modes.append(unique[np.argmax(cnts)])

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(bin_labels))
        bars = ax.bar(x, counts, color=colors[di % len(colors)],
                      edgecolor='white', linewidth=0.5)

        # Annotate each bar with the mode value and count
        for bar, count, mode in zip(bars, counts, modes):
            if count > 0 and mode is not None:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(counts) * 0.01,
                        f'mode\n{mode:.2f}',
                        ha='center', va='bottom', fontsize=8, color='#333333')

        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=30, ha='right')
        ax.set_xlabel('Score bin')
        ax.set_ylabel('Number of captions')
        ax.set_title(f'{dim.capitalize()} — Count per 10% Bin with Mode', fontsize=13, fontweight='bold')
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)

        plt.tight_layout()
        out = out_dir / f'mode_per_bin_{dim}.png'
        plt.savefig(out, dpi=150)
        plt.close()
        print(f'Saved {out}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_dir', default='dataset/HumanML3D/texts',
                        help='Directory containing *_spec_*.txt files')
    parser.add_argument('--out_dir', default='spectrum_viz',
                        help='Output directory for PNG files')
    args = parser.parse_args()

    text_dir = Path(args.text_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores, cooccurrence = parse_spec_files(text_dir)

    if not any(scores.values()):
        print('No spectrum captions found. Check --text_dir path.')
        return

    total = sum(len(v) for v in scores.values())
    print(f'\nTotal dimension tags parsed: {total:,}')
    for dim in DIMENSIONS:
        vals = scores.get(dim, [])
        if vals:
            print(f'  {dim:10s}: {len(vals):6,} captions  mean={np.mean(vals):.3f}  std={np.std(vals):.3f}')

    plot_score_distributions(scores, out_dir)
    plot_occurrence_bar(scores, out_dir)
    plot_cooccurrence_matrix(cooccurrence, out_dir)
    plot_mode_per_bin(scores, out_dir)

    print(f'\nAll plots saved to {out_dir}/')


if __name__ == '__main__':
    main()
