### Replication package for "Replicating Political Economy Weights in Ossa (2014): An Optimization-Based Alternative"

---
#### Run

From the repository root:

```
matlab -batch "master_run"
python 06_figures/gen_all_figures.py
```

`master_run.m` runs three stages in order:

| Stage | Script | Writes to | Paper |
|---|---|---|---|
| 1 | `main_run.m` | `05_output/` | Table 1, Tables A1–A4 |
| 2 | `robust_run.m` | `05_output/appendix/` | Tables B1–B4 and C1, Figures C1–C2 |
| 3 | `simulation_run.m` | `05_output/sim/` | Figure 1, Table D1, Figure D1 |

Stage 2 reads the output of stage 1. Set `RUN_SIMULATION = false` in `master_run.m` to skip stage 3. The switches `RUN_ROBUST`, `RUN_IDENT` and `RUN_SOLVER` at the top of `robust_run.m` turn its parts on and off. `gen_all_figures.py` reads `05_output/` and writes the figures to `06_figures/`.

On a cluster, submit the stages from `slurm/` instead (see `slurm/README.md`).

---
#### Files

| File | Paper |
|---|---|
| `comparison_table.csv` | Table 1 |
| `lambda_ossa.csv`, `lambda_lm.csv` | Tables A1, A2 |
| `optimal_tariff_ossa.csv`, `optimal_tariff_lm.csv` | Tables A3, A4 |
| `appendix/robust_summary.csv` | Tables 1, B1 |
| `appendix/robust_diag_japan.csv` | Table B2 |
| `appendix/lambda_lm_robust.csv`, `optimal_tariff_lm_robust.csv`, `robust_japan_unconstrained.csv` | Table B3 |
| `appendix/solver_comparison.csv` | Table B4 |
| `appendix/identification_table.csv` | Table C1 |
| `appendix/identification_spectra.csv` | Figure C1 |
| `appendix/flatness_walk.csv` | Figure C2 |
| `appendix/sector_sweep.csv` | not reported |
| `sim/sim_trial_all.csv` | Figure 1, Table D1, Figure D1 |
| `sim/table_d1.csv` | Table D1, written by `06_figures/plot_simulation.py` |

Paths are relative to `05_output/`.

---
#### Reference results

`output_reference/` holds the results reported in the paper, with the logs of stages 2 and 3 in `output_reference/logs/`, and `figures_reference/` holds the figures.

Stage 3 took 541 CPU-days on the University of Florida HiPerGator cluster, with a Scenario 2 trial taking up to 25 days on one CPU. To check the simulation numbers without rerunning it, copy `output_reference/sim/` to `05_output/sim/` and run `python 06_figures/plot_simulation.py`.

Results can differ in the last digits across machines and MATLAB versions.

---
#### Requirements

- MATLAB with the Optimization Toolbox and the Statistics and Machine Learning Toolbox. The Parallel Computing Toolbox is optional. Setting `PEW_MAXWORKERS=0` runs without a parallel pool.
- Python 3 with `numpy`, `pandas`, `matplotlib`, and `scipy`. With `latex` on the path the figures use the paper's fonts.
- The reference results were produced on an eight-core desktop with MATLAB R2025b (stage 1) and R2024b (stage 2), and on HiPerGator with R2024b (stage 3).

---
#### Notes

- The data in `01_data/` and the code in `02_ossa_original/` come unmodified from the replication package of *Ossa, R. (2014) Trade Wars and Trade Talks with Data, American Economic Review, 104(12), 4104–4146.*
- In `appendix/solver_comparison.csv`, `InnerSolves` omits the Jacobian solves that run on parallel workers, so it undercounts for LM and TRF. The solve counts behind Table B4 come from the iteration counts and, for TRF, from the `Func-count` column in `output_reference/logs/02_robust_run.log`.
- Table 1 re-solves the inner problem at the reported weights from a common warm start, the mean of the observed tariffs. Ossa's code warm-starts Japan at the observed tariffs, so the RSS computed directly from `optimal_tariff_ossa.csv` is 0.356 for Japan, against the 0.343 in Table 1.

---
#### Contact
**Yonggeun Jung**, Department of Economics, University of Florida, yonggeun.jung@ufl.edu
