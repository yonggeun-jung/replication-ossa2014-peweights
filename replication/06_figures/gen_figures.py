# %%
# Political economy weights comparison by methods

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

lam_ossa = pd.read_csv('replication/05_output/lambda_ossa.csv', header=None)
lam_lm = pd.read_csv('replication/05_output/lambda_lm.csv', header=None)

n_country, n_ind = lam_ossa.shape

countries = ["Brazil", "China", "EU", "India", "Japan", "ROW", "US"]

# Flatten values
l_ossa_vals = lam_ossa.values.flatten()
l_lm_vals = lam_lm.values.flatten()

country_idx = np.repeat(np.arange(n_country), n_ind)

markers = ['o', '^', 's', 'P', 'D', '<', 'v']
colors = [
    "#0095ff",  # Brazil  
    "#3300ff",  # China   
    "#00aaff",  # EU      
    "#5dade2",  # India   
    "#FF0000",  # Japan
    "#00eeff",  # ROW     
    "#0d00ff"   # US    
]

marker_sizes = [30, 30, 30, 30, 30, 30, 30] 

fig, ax = plt.subplots(figsize=(5, 5))

# Scatter
for c in range(n_country):
    mask = (country_idx == c)
    x = l_ossa_vals[mask]
    y = l_lm_vals[mask]
    ax.scatter(
        x, y,
        alpha=0.8,
        color=colors[c],
        marker=markers[c],
        s=marker_sizes[c],
        label=countries[c],
        edgecolors='white',
        linewidths=0.0,
        zorder = 2
    )

xx = np.linspace(0, 5, 200)
ax.plot(xx, xx, linestyle='--', linewidth=2, color='black', alpha=1.0, zorder = 1)

ax.set_xlim(0, 5)
ax.set_ylim(0, 5)

# ticks only at 0 and 10
ax.set_xticks([0, 5])
ax.set_yticks([0, 5])
ax.set_xticklabels(["", "5"])
ax.set_yticklabels(["0", "5"])

ax.set_xlabel("Political economy weights ($\\lambda$) by Ossa (2014)")
ax.set_ylabel("Political economy weights ($\\lambda$) by LM")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(3)
ax.spines['bottom'].set_linewidth(3)

ax.tick_params(width=0)

ax.legend(frameon=False, loc='best')
plt.tight_layout()
plt.savefig("replication/06_figures/lam_comparison_raw.pdf", bbox_inches='tight')

# %%
# Optimal tariffs comparison by methods

t_ossa = pd.read_csv('replication/05_output/optimal_tariff_ossa.csv', header=None)
t_lm   = pd.read_csv('replication/05_output/optimal_tariff_lm.csv',   header=None)

n_country, n_ind = t_ossa.shape

countries = ["Brazil", "China", "EU", "India", "Japan", "ROW", "US"]

# percent scale
t_ossa_vals = (t_ossa.values * 100).flatten()
t_lm_vals   = (t_lm.values   * 100).flatten()

country_idx = np.repeat(np.arange(n_country), n_ind)

markers = ['o', '^', 's', 'P', 'D', '<', 'v']
colors = [
    "#0095ff",  # Brazil  
    "#3300ff",  # China   
    "#00aaff",  # EU      
    "#5dade2",  # India   
    "#FF0000",  # Japan
    "#00eeff",  # ROW     
    "#0d00ff"   # US    
]

marker_sizes = [30, 30, 30, 30, 30, 30, 30] 

fig, ax = plt.subplots(figsize=(5, 5))

# scatter
for c in range(n_country):
    mask = (country_idx == c)
    ax.scatter(
        t_ossa_vals[mask], t_lm_vals[mask],
        alpha=0.8,
        color=colors[c],
        marker=markers[c],
        s=marker_sizes[c],
        label=countries[c],
        edgecolors='white',
        linewidths=0.0, 
        zorder = 2
    )

xx = np.linspace(0, 750, 300)
ax.plot(xx, xx, '--', color='black', linewidth=2, alpha=1.0, zorder = 1)

# axis range
ax.set_xlim(0, 750)
ax.set_ylim(0, 750)

# ticks only at 0 and 750
ax.set_xticks([0, 750])
ax.set_yticks([0, 750])
ax.set_xticklabels(["", "750"])
ax.set_yticklabels(["0", "750"])

ax.set_xlabel("Optimal tariffs ($\\tau^{\\text{opt}}$) in \\% by Ossa (2014)")
ax.set_ylabel("Optimal tariffs ($\\tau^{\\text{opt}}$) in \\% by LM")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(3)
ax.spines['bottom'].set_linewidth(3)
ax.tick_params(width=0)

ax.legend(frameon=False, loc='best')
plt.tight_layout()
plt.savefig("replication/06_figures/tariff_comparison_raw.pdf", bbox_inches='tight')

# %%
# Political economy weights comparison by methods (country)

lam_ossa = pd.read_csv('replication/05_output/lambda_ossa.csv', header=None)
lam_lm   = pd.read_csv('replication/05_output/lambda_lm.csv',   header=None)

n_country, n_ind = lam_ossa.shape

countries = ["Brazil", "China", "EU", "India", "Japan", "ROW", "US"]

lam_ossa_vals = lam_ossa.values.flatten()
lam_lm_vals   = lam_lm.values.flatten()

country_idx = np.repeat(np.arange(n_country), n_ind)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.flatten()

for c in range(n_country):
    ax = axes[c]

    x = lam_ossa.iloc[c, :].values
    y = lam_lm.iloc[c, :].values

    panel_max = max(x.max(), y.max())
    panel_max = np.ceil(panel_max)

    ax.scatter(x, y, color='black', s=20, alpha=0.7)

    xx = np.linspace(0, panel_max, 100)
    ax.plot(xx, xx, '--', color='red', linewidth=1)

    ax.set_title(countries[c])

    ax.set_xlim(0, panel_max)
    ax.set_ylim(0, panel_max)

    ax.set_xticks([0, panel_max])
    ax.set_yticks([0, panel_max])

    if c % 4 == 0:
        ax.set_ylabel("Political weight (LM)")
    if c >= 4:
        ax.set_xlabel("Political weight (Ossa, 2014)")

for k in range(n_country, len(axes)):
    axes[k].set_visible(False)

plt.tight_layout()
plt.savefig("replication/06_figures/lam_comparison_by_country_raw.pdf", bbox_inches='tight')

# %%
# Political economy weights difference by methods (country)

lam_ossa = pd.read_csv('replication/05_output/lambda_ossa.csv', header=None)
lam_lm   = pd.read_csv('replication/05_output/lambda_lm.csv',   header=None)

n_country, n_ind = lam_ossa.shape

countries = ["Brazil", "China", "EU", "India", "Japan", "ROW", "US"]

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.flatten()

for c in range(n_country):
    ax = axes[c]

    ossa_vals = lam_ossa.iloc[c, :].values
    lm_vals   = lam_lm.iloc[c, :].values

    diff = ossa_vals - lm_vals

    # baseline
    ax.axhline(0, color='gray', linewidth=1)

    ax.scatter(
        np.arange(1, n_ind+1),
        diff,
        color='blue',
        s=20,
        alpha=0.7
    )

    ax.set_title(countries[c])

    ax.set_xlim(1, n_ind)
    ax.set_xticks([1, n_ind])

    diff_max = np.max(np.abs(diff))
    y_lim = np.ceil(diff_max)       

    ax.set_ylim(-y_lim, y_lim)
    ax.set_yticks([-y_lim, 0, y_lim])

    if c % 4 == 0:
        ax.set_ylabel("Political weight difference")
    if c >= 4:
        ax.set_xlabel("Industry index")

for k in range(n_country, len(axes)):
    axes[k].set_visible(False)

plt.tight_layout()
plt.savefig("replication/06_figures/lam_difference_by_country_raw.pdf", bbox_inches='tight')

# %%
# Optimal tariffs comparison by methods (country)

t_ossa = pd.read_csv('replication/05_output/optimal_tariff_ossa.csv', header=None)
t_lm   = pd.read_csv('replication/05_output/optimal_tariff_lm.csv',   header=None)

n_country, n_ind = t_ossa.shape

countries = ["Brazil", "China", "EU", "India", "Japan", "ROW", "US"]

# percent scale
t_ossa_vals = (t_ossa.values * 100).flatten()
t_lm_vals   = (t_lm.values   * 100).flatten()

country_idx = np.repeat(np.arange(n_country), n_ind)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.flatten()

for c in range(n_country):
    ax = axes[c]

    x = t_ossa.iloc[c, :].values * 100
    y = t_lm.iloc[c, :].values * 100

    # each panel's max value
    panel_max = max(x.max(), y.max())
    panel_max = np.ceil(panel_max / 10) * 10   # round up to nearest 10

    ax.scatter(x, y, color='black', s=20, alpha=0.7)

    xx = np.linspace(0, panel_max, 100)
    ax.plot(xx, xx, '--', color='red', linewidth=1)

    ax.set_title(countries[c])

    ax.set_xlim(0, panel_max)
    ax.set_ylim(0, panel_max)

    ax.set_xticks([0, panel_max])
    ax.set_yticks([0, panel_max])

    if c % 4 == 0:
        ax.set_ylabel("Optimal tariff (LM)")
    if c >= 4:
        ax.set_xlabel("Optimal tariff (Ossa, 2014)")

for k in range(n_country, len(axes)):
    axes[k].set_visible(False)

ax.tick_params(width=0)

plt.tight_layout()
plt.savefig("replication/06_figures/tariff_comparison_by_country_raw.pdf", bbox_inches='tight')

# %%
# Optimal tariffs difference by methods (country)

t_ossa = pd.read_csv('replication/05_output/optimal_tariff_ossa.csv', header=None)
t_lm   = pd.read_csv('replication/05_output/optimal_tariff_lm.csv',   header=None)

n_country, n_ind = t_ossa.shape

countries = ["Brazil", "China", "EU", "India", "Japan", "ROW", "US"]

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.flatten()

for c in range(n_country):
    ax = axes[c]

    # tariffs in %
    ossa_vals = t_ossa.iloc[c, :].values * 100
    lm_vals   = t_lm.iloc[c, :].values * 100

    diff = ossa_vals - lm_vals

    ax.axhline(0, color='gray', linewidth=1)

    ax.scatter(
        np.arange(1, n_ind+1),
        diff,
        color='blue',
        s=20,
        alpha=0.7
    )

    ax.set_title(countries[c])

    ax.set_xlim(1, n_ind)
    ax.set_xticks([1, n_ind])

    diff_max = np.max(np.abs(diff))
    y_lim = max(10, np.ceil(diff_max / 10) * 10)  # round up to nearest 10

    ax.set_ylim(-y_lim, y_lim)
    ax.set_yticks([-y_lim, -y_lim/2, 0, y_lim/2, y_lim])

    if c % 4 == 0:
        ax.set_ylabel("Tariff difference")
    if c >= 4:
        ax.set_xlabel("Industry index")

for k in range(n_country, len(axes)):
    axes[k].set_visible(False)

plt.tight_layout()
plt.savefig("replication/06_figures/tariff_difference_by_country_raw.pdf", dpi=500, bbox_inches='tight')


# %%
# Optimal tariffs and political economy weights (Original codes)

tariff = pd.read_csv('replication/05_output/optimal_tariff_ossa.csv', header=None)
lam = pd.read_csv('replication/05_output/lambda_ossa.csv', header=None)

n_country, n_ind = tariff.shape

# Country names
countries = ["Brazil", "China", "EU", "India", "Japan", "ROW", "US"]

# Multiply tariff by 100 for %
t_vals = (tariff.values * 100).flatten()
l_vals = lam.values.flatten()

country_idx = np.repeat(np.arange(n_country), n_ind)

# markers and colors
markers = ['o', '^', 's', 'P', 'D', '<', 'v']
colors = [
    "#0095ff",  # Brazil  
    "#3300ff",  # China   
    "#00aaff",  # EU      
    "#5dade2",  # India   
    "#FF0000",  # Japan
    "#00eeff",  # ROW     
    "#0d00ff"   # US    
]
marker_sizes = [30, 30, 30, 30, 30, 30, 30] 

fig, ax = plt.subplots(figsize=(5, 5))

# Scatter + fitted line per country
for c in range(n_country):
    mask = (country_idx == c)
    x = t_vals[mask]
    y = l_vals[mask]
    ax.scatter(x, y, alpha=0.8, color=colors[c], marker=markers[c], 
               s=marker_sizes[c], label=countries[c], edgecolors='white', linewidths=0.0)
    
ax.set_xlim(0, 750)
ax.set_ylim(0, 2)

# ticks only at 0 and 10
ax.set_xticks([0, 200, 750])
ax.set_yticks([0, 2])
ax.set_xticklabels(["", "200", "750"])
ax.set_yticklabels(["0", "2"])


ax.set_xlabel("Optimal tariffs ($\\tau^{\\text{opt}}$) in % by industry")
ax.set_ylabel("Political economy weights ($\\lambda$) by industry")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(3)
ax.spines['bottom'].set_linewidth(3)

ax.tick_params(width=0)

ax.legend(frameon=False, loc='best')
plt.tight_layout()
plt.savefig("replication/06_figures/ossa_lam_tariff_raw.pdf", bbox_inches='tight')

# %%
## Optimal tariffs and political economy weights (LM)
tariff = pd.read_csv('replication/05_output/optimal_tariff_lm.csv', header=None)
lam = pd.read_csv('replication/05_output/lambda_lm.csv', header=None)

n_country, n_ind = tariff.shape

# Country names
countries = ["Brazil", "China", "EU", "India", "Japan", "ROW", "US"]

# Multiply tariff by 100 for %
t_vals = (tariff.values * 100).flatten()
l_vals = lam.values.flatten()

country_idx = np.repeat(np.arange(n_country), n_ind)

# markers and colors
markers = ['o', '^', 's', 'P', 'D', '<', 'v']
colors = [
    "#0095ff",  # Brazil  
    "#3300ff",  # China   
    "#00aaff",  # EU      
    "#5dade2",  # India   
    "#FF0000",  # Japan
    "#00eeff",  # ROW     
    "#0d00ff"   # US    
]
marker_sizes = [30, 30, 30, 30, 30, 30, 30] 

fig, ax = plt.subplots(figsize=(5, 5))

# Scatter + fitted line per country
for c in range(n_country):
    mask = (country_idx == c)
    x = t_vals[mask]
    y = l_vals[mask]
    ax.scatter(x, y, alpha=0.8, color=colors[c], marker=markers[c], 
               s=marker_sizes[c], label=countries[c], edgecolors='white', linewidths=0.0)
    
ax.set_xlim(0, 250)
ax.set_ylim(0, 5)

ax.set_xticks([0, 100, 250])
ax.set_yticks([0, 5])
ax.set_xticklabels(["", "100", "250"])
ax.set_yticklabels(["0", "5"])


ax.set_xlabel("Optimal tariffs ($\\tau^{\\text{opt}}$) in % by industry")
ax.set_ylabel("Political economy weights ($\\lambda$) by industry")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(3)
ax.spines['bottom'].set_linewidth(3)

ax.tick_params(width=0)

ax.legend(frameon=False, loc='best')
plt.tight_layout()
plt.savefig("replication/06_figures/lm_lam_tariff_raw.pdf", bbox_inches='tight')

# %%
df = pd.read_csv('replication/04_simulations/scenario1_realistic.csv')

fig, ax = plt.subplots(figsize=(5,5))

# Color settings
color_ossa = "#0d00ff"
color_lm   = "#FF0000"

# Scatter for Ossa
ax.scatter(
    df["Tariff_Var"], df["Norm_Ossa"], 
    label="Ossa", alpha=1, s=40, 
    color=color_ossa
)

# Scatter for LM
ax.scatter(
    df["Tariff_Var"], df["Norm_LM"], 
    label="LM", alpha=1, s=40, 
    color=color_lm
)

# Labels
ax.set_xlabel("Tariff variance")
ax.set_ylabel("Recovery error (L2 norm)")

# Spine and tick styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(3)
ax.spines['bottom'].set_linewidth(3)

ax.tick_params(width=0)

# Legend without frame
ax.legend(frameon=False, loc='best')

plt.tight_layout()
plt.savefig("replication/06_figures/scenario1_raw.pdf", bbox_inches='tight')

# %%
df = pd.read_csv('replication/04_simulations/scenario2_unrealistic.csv')

fig, ax = plt.subplots(figsize=(5,5))

# Color settings
color_ossa = "#0d00ff"
color_lm   = "#FF0000"

# Scatter for Ossa
ax.scatter(
    df["Tariff_Var"], df["Norm_Ossa"], 
    label="Ossa", alpha=1, s=40, 
    color=color_ossa
)

# Scatter for LM
ax.scatter(
    df["Tariff_Var"], df["Norm_LM"], 
    label="LM", alpha=1, s=40, 
    color=color_lm
)

# Labels
ax.set_xlabel("Tariff variance")
ax.set_ylabel("Recovery error (L2 norm)")

# Spine and tick styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(3)
ax.spines['bottom'].set_linewidth(3)

ax.tick_params(width=0)

# Legend without frame
ax.legend(frameon=False, loc='best')

plt.tight_layout()
plt.savefig("replication/06_figures/scenario2_raw.pdf", bbox_inches='tight')