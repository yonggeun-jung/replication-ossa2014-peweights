import os

import pandas as pd

from _figstyle import C_BLUE, C_RED, SIZE_PANEL, new_axes, save

IN_PATH = os.path.join('05_output', 'appendix', 'flatness_walk.csv')
FIG_DIR = '06_figures'

ESTIMATES = ['Heuristic', 'Baseline', 'Robust']

DIRECTIONS = [
    ('flat', 'flattest direction', C_BLUE, 'o'),
    ('control', 'steepest direction', C_RED, 's'),
]


def plot_estimate(fw, estimate, ylim):
    fig, ax = new_axes(SIZE_PANEL)
    d = fw[fw['Estimate'] == estimate]

    ax.axhline(0.0, color='#808080', linewidth=0.8, zorder=1)
    for key, label, colour, marker in DIRECTIONS:
        dd = d[d['Direction'] == key].sort_values('Step')
        ax.plot(dd['Step'], dd['RSS_change'], color=colour, marker=marker,
                label=label, zorder=2)

    ax.set_xlabel(r'distance moved $\|\Delta\lambda\|$')
    ax.set_ylabel('change in RSS')
    ax.set_ylim(ylim)

    if estimate == ESTIMATES[0]:
        ax.legend(frameon=False, loc='upper left', handlelength=1.5,
                  borderpad=0.1, labelspacing=0.25, handletextpad=0.4)

    save(fig, os.path.join(FIG_DIR, 'walk_%s.pdf' % estimate.lower()))


def main():
    fw = pd.read_csv(IN_PATH)
    lo, hi = fw['RSS_change'].min(), fw['RSS_change'].max()
    pad = 0.05 * (hi - lo)
    ylim = (lo - pad, hi + pad)
    for estimate in ESTIMATES:
        plot_estimate(fw, estimate, ylim)


if __name__ == '__main__':
    main()
