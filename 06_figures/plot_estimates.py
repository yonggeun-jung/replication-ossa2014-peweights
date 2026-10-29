import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _figstyle import (COUNTRIES, C_COUNTRY, C_LIGHT, C_LINE, C_MID,
                       M_COUNTRY, PUB_DPI, SIZE_LEAD_SQUARE, SIZE_LEAD_WIDE,
                       SIZE_OTHERS_SQUARE, SIZE_OTHERS_WIDE, SIZE_SQUARE,
                       new_axes, save, style_axes)

OUT = '05_output'
FIG_DIR = '06_figures'

LEAD = 'Japan'
OTHERS = [c for c in COUNTRIES if c != LEAD]


def ceil1(v):
    return float(np.ceil(v))


def ceil10(v):
    return float(max(10.0, np.ceil(v / 10.0) * 10.0))


def _read(name):
    return pd.read_csv(os.path.join(OUT, name), header=None)


def cell_scatter(ax, x, y, rounder, xlabel, ylabel):
    hi = rounder(max(x.max(), y.max()))
    ax.plot([0, hi], [0, hi], linestyle='--', linewidth=1.0, color=C_MID,
            zorder=1)
    ax.scatter(x, y, s=14, color=C_LINE, linewidths=0.0, zorder=2)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xticks([0, hi])
    ax.set_yticks([0, hi])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def cell_diff(ax, diff, rounder, xlabel, ylabel):
    hi = rounder(np.abs(diff).max())
    ax.axhline(0, color=C_MID, linewidth=0.8, zorder=1)
    ax.scatter(np.arange(1, diff.size + 1), diff, s=14,
               color=C_LINE, linewidths=0.0, zorder=2)
    ax.set_xlim(1, diff.size)
    ax.set_xticks([1, diff.size])
    ax.set_ylim(-hi, hi)
    ax.set_yticks([-hi, 0, hi])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def lead_figure(draw_cell, size, path):
    fig, ax = new_axes(size)
    draw_cell(ax, COUNTRIES.index(LEAD))
    save(fig, os.path.join(FIG_DIR, path))


def others_figure(draw_cell, size, path):
    fig, axes = plt.subplots(2, 3, figsize=size, dpi=PUB_DPI)
    fig.patch.set_facecolor('white')
    for ax, name in zip(axes.ravel(), OTHERS):
        style_axes(ax)
        draw_cell(ax, COUNTRIES.index(name))
        ax.set_title(name)
    fig.tight_layout()
    save(fig, os.path.join(FIG_DIR, path))


def by_country(a, b, scale, rounder, xlabel, ylabel, difflabel, stem):

    def scatter_cell(ax, c):
        cell_scatter(ax, a.iloc[c, :].values * scale,
                     b.iloc[c, :].values * scale, rounder, xlabel, ylabel)

    def diff_cell(ax, c):
        cell_diff(ax, (a.iloc[c, :].values - b.iloc[c, :].values) * scale,
                  rounder, 'Industry index', difflabel)

    lead_figure(scatter_cell, SIZE_LEAD_SQUARE, '%s_scatter_japan.pdf' % stem)
    others_figure(scatter_cell, SIZE_OTHERS_SQUARE,
                  '%s_scatter_others.pdf' % stem)
    lead_figure(diff_cell, SIZE_LEAD_WIDE, '%s_diff_japan.pdf' % stem)
    others_figure(diff_cell, SIZE_OTHERS_WIDE, '%s_diff_others.pdf' % stem)


def pooled_scatter(a, b, hi, xlabel, ylabel, ticks, path):
    n_country, n_ind = a.shape
    x = a.values.flatten()
    y = b.values.flatten()
    idx = np.repeat(np.arange(n_country), n_ind)

    fig, ax = new_axes(SIZE_SQUARE)
    ax.plot([0, hi], [0, hi], linestyle='--', linewidth=1.0, color=C_MID,
            zorder=1)
    for c, name in enumerate(COUNTRIES):
        m = idx == c
        ax.scatter(x[m], y[m], s=22, alpha=0.9, color=C_COUNTRY[name],
                   marker=M_COUNTRY[name], linewidths=0.0, label=name,
                   zorder=2)

    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, loc='lower right', handletextpad=0.3,
              labelspacing=0.35)
    save(fig, os.path.join(FIG_DIR, path))


def main():
    lam_o, lam_l = _read('lambda_ossa.csv'), _read('lambda_lm.csv')
    tar_o = _read('optimal_tariff_ossa.csv')
    tar_l = _read('optimal_tariff_lm.csv')

    pooled_scatter(
        lam_o, lam_l, 5.0,
        r'Political economy weights ($\lambda$), Ossa (2014)',
        r'Political economy weights ($\lambda$), LM',
        [0, 5], 'lam_comparison.pdf')

    pooled_scatter(
        tar_o * 100, tar_l * 100, 750.0,
        r'Optimal tariffs ($\tau^{\mathrm{opt}}$) in percent, Ossa (2014)',
        r'Optimal tariffs ($\tau^{\mathrm{opt}}$) in percent, LM',
        [0, 750], 'tariff_comparison.pdf')

    by_country(lam_o, lam_l, 1.0, ceil1,
               'Political weight (Ossa, 2014)', 'Political weight (LM)',
               'Political weight difference', 'lam')

    by_country(tar_o, tar_l, 100.0, ceil10,
               'Optimal tariff (Ossa, 2014)', 'Optimal tariff (LM)',
               'Tariff difference', 'tariff')


if __name__ == '__main__':
    main()
