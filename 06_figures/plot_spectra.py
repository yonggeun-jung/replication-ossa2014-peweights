import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _figstyle import (COUNTRIES, METHODS, PUB_DPI, SIZE_LEAD_WIDE,
                       SIZE_OTHERS_WIDE, new_axes, save, style_axes)

IN_PATH = os.path.join('05_output', 'appendix', 'identification_spectra.csv')
FIG_DIR = '06_figures'

LEAD = 'Japan'
OTHERS = [c for c in COUNTRIES if c != LEAD]

FLOOR = 1e-16


def draw(ax, sp, name, legend=False):
    for key, label, colour, marker in METHODS:
        d = sp[(sp['Name'] == name) & (sp['Method'] == key)].sort_values('k')
        ax.plot(d['k'], np.maximum(d['sigma_k'], FLOOR),
                color=colour, marker=marker, label=label)

    ax.set_yscale('log')
    ax.set_ylim(3e-17, 1e4)
    ax.set_yticks([1e-15, 1e-10, 1e-5, 1e0])
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$\sigma_k$')

    if legend:
        ax.legend(frameon=False, loc='lower left', handlelength=1.5,
                  borderpad=0.1, labelspacing=0.25, handletextpad=0.4)


def plot_lead(sp):
    fig, ax = new_axes(SIZE_LEAD_WIDE)
    draw(ax, sp, LEAD, legend=True)
    save(fig, os.path.join(FIG_DIR, 'spectra_japan.pdf'))


def plot_others(sp):
    fig, axes = plt.subplots(2, 3, figsize=SIZE_OTHERS_WIDE, dpi=PUB_DPI)
    fig.patch.set_facecolor('white')
    for ax, name in zip(axes.ravel(), OTHERS):
        style_axes(ax)
        draw(ax, sp, name)
        ax.set_title(name)
    fig.tight_layout()
    save(fig, os.path.join(FIG_DIR, 'spectra_others.pdf'))


def main():
    sp = pd.read_csv(IN_PATH)
    plot_lead(sp)
    plot_others(sp)


if __name__ == '__main__':
    main()
