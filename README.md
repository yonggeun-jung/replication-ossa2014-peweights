### Replicating Political Economy Weights in Ossa (2014): An Optimization-Based Alternative

The goal of this package is to replicate the computation of political economy weights in:

*Ossa, R. (2014) Trade Wars and Trade Talks with Data, American Economic Review, 104(12), 4104–4146.*

Ossa (2014) computes political economy weights using an efficient but heuristic algorithm. This package reproduces those results with a Levenberg-Marquardt optimization method that provides a formal numerical alternative.

Data used here come directly from Ossa’s publicly available replication package.

---
#### Repository Overview

The `replication/` directory contains all scripts, functions, and data required to reproduce the results. The structure is organized as follows:

```
replication/
│
├── main_run.m
│
├── 01_data/
├── 02_ossa_original/
├── 03_lm/
├── 04_simulations/
├── 05_output/
└── 06_figures/
```
---
#### Top-Level Script

##### main_run.m
Main driver for the replication. It executes:

1. Ossa (2014)’s original heuristic algorithm (`02_ossa_original/`)
2. Levenberg-Marquardt optimization routine (`03_lm/`)

It outputs:
- political economy weights  
- implied optimal tariffs  
- residual diagnostics (RSS, gradients)  

All results are saved in `05_output/`.

---

#### Subdirectories

##### 01_data/
Raw data from Ossa (2014)’s replication package (unmodified).
##### 02_ossa_original/
Original functions as provided in Ossa’s replication package including:
- `mylambdaj.m`  
- equilibrium and tariff computation routines  
##### 03_lm/
Levenberg-Marquardt implementation:
- `estimate_lambda_lm.m`  
- `evaluate_comparison.m`  
##### 04_simulations/
Simulation Exercises implementation:
- `run_simulations.m`  
##### 05_output/
All generated output files:
- political economy weights by method  
- optimal tariffs by method
- comparison table 
##### 06_figures/
All figures generated from `gen_figures.py` using output files including simulations.  

---

#### Contact
**Yonggeun Jung**  
Department of Economics, University of Florida  
Email: yonggeun.jung@ufl.edu
