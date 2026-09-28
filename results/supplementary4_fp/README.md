# Supplementary Figure 4: saved joint first passage

The finite-volume forward calculation uses
480 cells, 1/60-year timesteps and 96 positive-Gaussian threshold quadrature
nodes, with a 160-year horizon and 2,001 sick-life-fraction bins.

| Scenario | Median fraction of life sick |
| --- | ---: |
| Baseline | 0.1035 |
| Death threshold alone increased 20% | 0.1635 |
| Disease and death thresholds increased 20% | 0.0725 |

The figure rounds these percentages to one decimal place. They are numerical
model outputs, not measured human morbidity estimates or confidence intervals.
The scenario definitions, frozen Swedish Q reference, external mortality of
zero, solver checksum and settings are in `manifest.json`.
The solver checksum identifies the calculation-time file. Its module docstring
was subsequently updated to point to the completed grid checks; executable
code is unchanged (verified by AST comparison excluding that docstring).

- `states.csv`: healthy-alive, sick-alive and dead probability by age/scenario.
- `sick_fraction.csv`: the population distribution of individual sickspan
  divided by end age, retaining the joint onset/death calculation.
- `summary.csv`: medians, surviving mass at the horizon, and worst marginal
  identity error across quadrature nodes.
- `validation/`: corresponding compact tables from the coarser grids and
  machine-readable comparisons. Individual node arrays are not required to
  reproduce the figure and are not shipped.

The 160-cell/0.05-year/32-node and 320-cell/0.025-year/64-node comparisons changed
the median by at most 0.25 percentage points and any state probability by
0.00627. Refining from 320/0.025/64 to 480/(1/60)/96 changed every median by
0.05 percentage points and any state probability by at most 0.001725.
These checks refine space, time and quadrature jointly, not independently.
They establish stability at the reported precision, not a formal error bound.
The finest-grid surviving mass is below 8e-110 and marginal identity error
below 5e-16. The ratio-bin rounding bound is 0.00025 per individual fraction.

`node_baseline_source_sha256s` records source-file checksums. Aggregation
verifies identical baseline parameter values, solver hashes and numerical
settings. Covariance is not used in this forward calculation.

Reproduce the PNG with:

```bash
python3 analysis/figures/supplementary/make_supp_figure4_healthspan_morbidity.py
```

See [methods and optional node commands](../../docs/figure_methods/healthspan_fp.md).
Calculations use single-thread workers, with up to six concurrent workers
for the refined grid. Rendering reads the saved tables.
