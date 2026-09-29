# Supplementary Figure 1: tolerance estimates and numerical comparisons

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

## Panel b: local heterogeneity tolerance

`heterogeneity_tolerance.csv` and `.json` record the analytic curves and their
reference settings. These use the illustrative SI values T = 90 years,
b = 0.1/year, m(T) = 0.15/year and tau = 100 years. The x coordinate r is
a stipulated budget for the selection contribution to the slope, not the
observed variation of slopes between age windows in panel a.

The hazard-CV budget is sqrt(r*b/m). Divide it by the absolute local
log-hazard sensitivities b*T, 2-b*tau and b*(T-tau) to obtain allowed CVs
for eta, beta and Xc, respectively. The beta curve includes the beta-squared
prefactor in the simplified hazard. At r = 0.2, the estimates are 4.057%,
4.564% and 36.515%. The leading-barrier beta estimate, which neglects the
prefactor, is 3.651%; it is discussed in the SI but is not the plotted curve.

All spreads are among survivors at age 90 and assume one varying parameter
at a time. They are local, first-order approximations, not fitted limits on
initial heterogeneity or confidence intervals. Large threshold-CV values
are especially approximate. No epsilon curve is shown because the current
S4 derivation gives explicit CV estimates for eta, beta and Xc only.

Recompute the table with:

```sh
python3 analysis/model_fits/supplementary/heterogeneity_tolerance.py
```

Panels a and c-f use the same empirical and saved FP data as before.

## Panels c–f: descriptive fits and reference trends

Panel c fits a + b/t to death-bin means with midpoints 80–160 years;
panel d fits a + b*t over 90–160 years. These are unweighted descriptive
least-squares fits, not refits of the SR model. Compared with the legacy
beta window of 90–110 years, the wider window avoids fitting just two bins.
The annotations report the current fitted coefficients and the fixed companion
parameter. Panels e/f show constant and t-squared reference shapes over
90–120 years, with amplitude chosen by least squares in log mortality.
Their exponents are imposed, not fitted, and are not claimed as asymptotic
laws of the current FP mixtures. The numerical curves remain unchanged.
