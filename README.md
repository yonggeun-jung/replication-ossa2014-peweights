### Replicating Political Economy Weights in Ossa (2014): An Optimization-Based Alternative

The goal of this package is to replicate the computation of political economy weights in:

*Ossa, R. (2014) Trade Wars and Trade Talks with Data, American Economic Review, 104(12), 4104–4146.*

Ossa (2014) computes political economy weights using an efficient but heuristic algorithm. This package reproduces those results with a Gauss–Newton optimization method that provides a formal numerical alternative.

Data used here come directly from Ossa’s publicly available replication package.

---
#### Repository Overview

The `replication/` directory contains all scripts, functions, and data required to reproduce the results in both the main text and the appendix. The structure is organized as follows:

```
replication/
│
├── A_main_run.m
├── B_main_corr_fig.ipynb
├── C_appendix_noUB/
├── D_appendix_corr_fig.ipynb
│
├── 01_data/
├── 02_ossa_original/
├── 03_gn/
├── 04_output/
├── 05_appendix_noUB/
└── 06_figures/
```
---
#### Top-Level Scripts

##### A_main_run.m
Main driver for the replication. It executes:

1. Ossa (2014)’s original heuristic algorithm (`02_ossa_original/`)
2. Gauss–Newton optimization routine (`03_gn/`)

It outputs:
- political economy weights  
- implied optimal tariffs  
- residual diagnostics (RSS, gradients)  

All results are saved in `04_output/`.

##### B_main_corr_fig.ipynb
Generates:
- correlation tables  
- figures (main text + Appendix A)

Outputs saved to `06_figures/`.

##### C_appendix_noUB/
Contains the no-upper-bound version of the replication used for Appendix B.

##### D_appendix_corr_fig.ipynb
Produces appendix correlation tables and figures for the no-UB experiment.

---

#### Subdirectories

##### 01_data/
Raw data from Ossa (2014)’s replication package (unmodified).
##### 02_ossa_original/
Original functions as provided in Ossa’s replication package including:
- `mylambdaj.m`  
- equilibrium and tariff computation routines  
##### 03_gn/
Gauss–Newton implementation:
- `estimate_lambda_gn.m`  
- `evaluate_comparison.m`  
##### 04_output/
All generated output files:
- weights  
- tariffs  
##### 05_appendix_noUB/
reproduce Appendix B robustness results.
- `estimate_lambda_gn_noUB.m`  
- `mylambdaj_noUB.m`  
##### 06_figures/
All figures generated from `B_main_corr_fig.ipynb` and `D_appendix_corr_fig.ipynb`.  

---

#### Contact
**Yonggeun Jung**  
Department of Economics, University of Florida  
Email: yonggeun.jung@ufl.edu
