import shutil

import matplotlib.pyplot as plt

TEXTWIDTH = 6.34

W_FULL = TEXTWIDTH
W_HALF = 0.49 * TEXTWIDTH
W_THIRD = 0.32 * TEXTWIDTH
W_LEAD = 0.50 * TEXTWIDTH

SIZE_SQUARE = (W_HALF, W_HALF)
SIZE_PANEL = (W_THIRD, W_THIRD * 0.90)

SIZE_LEAD_SQUARE = (W_LEAD, W_LEAD)
SIZE_LEAD_WIDE = (W_LEAD, W_LEAD * 0.78)
SIZE_OTHERS_SQUARE = (W_FULL, W_FULL * 0.62)
SIZE_OTHERS_WIDE = (W_FULL, W_FULL * 0.52)

PUB_DPI = 300

C_TEXT = '#222222'
C_LINE = '#141414'
C_MID = '#5c5c5c'
C_LIGHT = '#bdbdbd'

C_BLACK = '#000000'
C_RED = '#D00000'
C_BLUE = '#0033CC'

C_HEUR = C_BLUE
C_BASE = C_RED
C_ROB = C_BLACK

METHODS = [
    ('Ossa', 'Heuristic', C_HEUR, 'o'),
    ('LM_fast', 'Baseline LM', C_BASE, 's'),
    ('LM_robust', 'Robust LM', C_ROB, '^'),
]

COUNTRIES = ['Brazil', 'China', 'EU', 'India', 'Japan', 'ROW', 'US']

C_COUNTRY = {c: C_BLUE for c in ('Brazil', 'China', 'EU', 'India', 'ROW', 'US')}
C_COUNTRY['Japan'] = C_RED
M_COUNTRY = {
    'Brazil': 'o', 'China': '^', 'EU': 's', 'India': 'P',
    'Japan': 'D', 'ROW': '<', 'US': 'v',
}

USETEX = shutil.which('latex') is not None

plt.rcParams.update({
    'text.usetex': USETEX,
    'text.latex.preamble': r'\usepackage{newpxtext,newpxmath}',
    'font.family': 'serif',
    'font.serif': ['TeX Gyre Pagella', 'Palatino Linotype', 'Palatino',
                   'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.labelcolor': C_TEXT,
    'axes.edgecolor': C_TEXT,
    'text.color': C_TEXT,
    'lines.linewidth': 1.5,
    'lines.markersize': 3.2,
})


def style_axes(ax):
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.0)
        ax.spines[side].set_color(C_TEXT)
    ax.tick_params(colors=C_TEXT, length=3, width=1.0, direction='out')


def new_axes(figsize=SIZE_PANEL):
    fig, ax = plt.subplots(figsize=figsize, dpi=PUB_DPI)
    fig.patch.set_facecolor('white')
    style_axes(ax)
    return fig, ax


def save(fig, path, tight=True):
    fig.savefig(path, facecolor='white', edgecolor='none',
                bbox_inches='tight' if tight else None, pad_inches=0.02)
    plt.close(fig)
    print('  Saved: %s' % path)
