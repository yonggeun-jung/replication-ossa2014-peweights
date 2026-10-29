import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from _figstyle import METHODS, TEXTWIDTH, W_HALF, new_axes, save

IN_PATH = os.path.join('05_output', 'sim', 'sim_trial_all.csv')
TABLE_PATH = os.path.join('05_output', 'sim', 'table_d1.csv')
FIG_DIR = '06_figures'

SIZE = (W_HALF, W_HALF * 0.85)
SIZE_D1 = (0.6 * TEXTWIDTH, 0.6 * TEXTWIDTH * 0.7)

ALL = ['Heuristic', 'Baseline', 'Robust']
SPLIT = 3.5

STYLE = {label.split()[0]: (label, colour, marker)
         for _, label, colour, marker in METHODS}


def scatter(ax, d, x, methods, legend_loc='upper left'):
    for m in methods:
        label, colour, marker = STYLE[m]
        y = d[d.Method == m]
        ax.scatter(y[x], y.Norm_e, s=20, color=colour, marker=marker,
                   alpha=0.85, linewidths=0.0, label=label, zorder=2)
    ax.set_ylabel(r'$\|\hat{\lambda}-\lambda^{\mathrm{true}}\|_2$')
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc=legend_loc, handletextpad=0.2,
              labelspacing=0.3, borderpad=0.1)


def figure_1(d):
    for sc, loc in ((1, 'upper right'), (2, 'upper left')):
        fig, ax = new_axes(SIZE)
        scatter(ax, d[d.Scenario == sc], 'Tariff_Var', ALL, loc)
        ax.set_xlabel('Variance of simulated optimal tariffs')
        fig.subplots_adjust(left=0.22, right=0.97, bottom=0.19, top=0.97)
        save(fig, os.path.join(FIG_DIR, 'scenario%d.pdf' % sc), tight=False)


def figure_d1(d):
    s2 = d[d.Scenario == 2]
    rng = np.random.default_rng(0)
    s2 = s2.assign(NullJit=s2.NullDim_loose + rng.uniform(-0.25, 0.25, len(s2)))
    fig, ax = new_axes(SIZE_D1)
    ax.axvline(SPLIT, color='#808080', linewidth=1.0, linestyle='--', zorder=1)
    scatter(ax, s2, 'NullJit', ALL)
    ax.set_xlabel('Null dimension at the true weights')
    save(fig, os.path.join(FIG_DIR, 'mc_nulldim.pdf'))


def table_d1(d):
    s1, s2 = d[d.Scenario == 1], d[d.Scenario == 2]
    panels = [('A', s1), ('B', s2),
              ('C', s2[s2.NullDim_loose <= SPLIT]),
              ('D', s2[s2.NullDim_loose > SPLIT])]
    rows = []
    for panel, x in panels:
        n = x.Trial.nunique()
        err = x.pivot_table(index='Trial', columns='Method', values='Norm_e')
        for m in ALL:
            y = x[x.Method == m]
            wins = np.nan if m == 'Heuristic' else int((err[m] < err.Heuristic).sum())
            p = (np.nan if m == 'Heuristic' else
                 binomtest(int(wins), n, 0.5, alternative='greater').pvalue)
            rows.append(dict(Panel=panel, Method=m, Trials=n,
                             Mean=y.Norm_e.mean(), Median=y.Norm_e.median(),
                             Max=y.Norm_e.max(), Wins=wins, SignTest_p=p,
                             Err_identified=y.Norm_e_id_loose.mean(),
                             Err_null=y.Norm_e_null_loose.mean()))
    t = pd.DataFrame(rows)
    t.to_csv(TABLE_PATH, index=False)
    print('  Saved: %s' % TABLE_PATH)
    return t


def main():
    d = pd.read_csv(IN_PATH)
    figure_1(d)
    figure_d1(d)
    table_d1(d)


if __name__ == '__main__':
    main()
