# Supplementary SR calculations

## Gompertz constraints (Supplementary Fig. 1)

The renderer uses saved empirical slopes, analytic tolerance estimates and checked finite-volume results:

```bash
python3 analysis/figures/supplementary/render_gompertz_constraints.py
```

It compares homogeneous mortality, fitted threshold heterogeneity, and separate
20% positive-Gaussian eta and beta heterogeneity at the current Swedish
Q-reference means. These are controlled comparisons, not new fits.
External mortality is zero. See the [saved-data settings](../../../results/supplementary1_fp/README.md)
for production, quadrature and refined-grid specifications.

The plotted parameter mean is conditioned on **death in a 10-year lifespan
bin**, not survival to that age. For component j, its bin weight is its birth
weight multiplied by S_j(start)-S_j(end). The original display's minimum of
25 observations per million is retained as a minimum bin probability 0.000025.

Mortality is model boundary deaths divided by person-years within annual bins,
shown at bin midpoints. No trajectory smoothing is required. Panel b shows the local tolerance calculation in SI section S4: parameter CV
among age-90 survivors versus the allowed fractional selection-induced slope
reduction. It varies one parameter at a time and retains the beta-squared
prefactor correction. These are approximation-dependent tolerance estimates,
not fitted initial-population limits or confidence intervals. The homogeneous
and threshold-mixture curves remain saved numerical diagnostics, but are not
plotted in panel b. Universal asymptotic fits remain omitted.

The renderer generates the figure from the saved source tables.
It is included in normal saved-output reproduction.

Optional numerical recomputation requires an explicit command:

```sh
python3 analysis/model_fits/supplementary/check_si.py --recompute
```

This writes to `tmp/si_checks`, without overwriting the archive. Rendering never
starts the numerical calculations or falls back to trajectory tables.

## Morbidity (Supplementary Fig. 4)

The optional `healthspan.py` worker uses the tagged joint FP solver in
`src/senogenic_vs_robustness/sr_joint_passage.py`. It preserves the distribution
of disease first passage and death time, including individual sickspan/lifespan.
The saved-table renderer uses `results/supplementary4_fp/`. Mathematical tests,
synthetic renderer tests and a three-grid check pass. The adopted 480-cell grid
changes medians by 0.05 percentage points relative to 320 cells. See
[methods and optional commands](../../../docs/figure_methods/healthspan_fp.md).

Subtracting marginal survival curves cannot recover the individual ratio
distribution. The joint calculation is quadratic in time steps and is run
separately from rendering.

Regenerate the analytic source table (no simulation or optimization):

```sh
python3 analysis/model_fits/supplementary/heterogeneity_tolerance.py
```
