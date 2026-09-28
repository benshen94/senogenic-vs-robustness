# Supplementary Figure 1: numerical comparisons

These saved calculations provide controlled comparisons at the fitted
Sweden baseline means, not new fits or bootstrap estimates.

`si_hazards.csv` contains homogeneous, threshold-CV, eta-CV and beta-CV
annual mortality for three numerical configurations. Production uses 320
cells, 0.025-year steps and 64 mixture nodes; the quadrature check uses 160
nodes; the refined grid uses 480 cells, 1/60-year steps and 160 nodes.
All calculations extend to age 260, with external mortality fixed to zero.
Across saved annual midpoints through age 120, the maximum absolute relative
production-to-refined mortality differences are 1.143% (homogeneous), 0.778%
(threshold mixture), 0.163% (eta mixture), and 0.060% (beta mixture).
These checks concern the plotted finite-age mortality, not arbitrary late-age
asymptotes or convergence of every derived quantity.
Threshold CV is the archived baseline CV; eta and beta comparisons impose
20% CV individually with the other parameters homogeneous.

`si_death_bin_means.csv` contains refined-grid parameter means conditional
on death in ten-year lifespan intervals, not means among survivors.
Intervals with probability below 25 per million are omitted.
`si_checks.json` records the baseline and numerical slopes.
`si_approximation_check.csv` compares numerical mortality with the absorbing
endpoint escape approximation and the retired approximation. The latter is
retained as a diagnostic, not as the figure's analytic constraint.

Render without simulation:

```sh
python3 analysis/figures/supplementary/render_gompertz_constraints.py
```

Optional numerical recomputation, explicitly separate from figure rendering:

```sh
python3 analysis/model_fits/supplementary/check_si.py --recompute
```

This writes to `tmp/si_checks`, not the archived results. The empirical
Sweden slopes are read from
`results/tables/supplementary_figure1/sweden2019_decade_slopes.csv`.
The renderer generates the figure from the saved source tables.
