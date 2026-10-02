# Historical Sweden and Denmark

Render the current figures from archived tables, without fits or derivatives:

```bash
python3 analysis/figures/figure4/render_current_history.py
python3 analysis/figures/figure5/render_lifespan.py
```

Outputs are `Figures/Figure4/Fig4.png`,
`Figures/ExtendedDataFigure4/ExtDataFig4.png`, and
`Figures/Figure5/Fig5.png`. Covariance matrices, 61 band records and source
tables are included in `results/historical/`.
Panels a/b retain the age-20 response-plane normalization; panel c retains the
separate period-specific MGG external-mortality estimates and intervals.

## Current uncertainty

The Swedish 2019 reference estimates four parameters. Each historical fit
estimates only Xc and mex, holding other parameters at that reference.
Q is an estimating criterion, not a likelihood and not a basis for AIC.
For each year, fitted expected deaths use the original model's exact
deaths/person-time age-specific rates at ages 20--109.

For positive-exposure age bins, the Pearson dispersion is

$$\phi_y = \frac{1}{n_y-k_y}\sum_a\frac{(D_{ya}-\mu_{ya})^2}{\mu_{ya}}.$$

Here n counts included bins, k=4 for the Swedish reference, and k=2 for each
historical Swedish or Danish fit. Each independent year's estimating-equation
variance block is multiplied by max(1, phi_y) **before** joint inversion:

$$C = A^{-1}\operatorname{blockdiag}\{\max(1,\phi_y)B_y\}A^{-T}.$$

A is the expected derivative of the joint estimating equations. Its inherited
reference blocks propagate shared-reference uncertainty, ratio numerator/
denominator dependence, and cross-year correlations. Covariance off-diagonal
entries are not independently rescaled. `dispersion.json` and
`denmark_dispersion.json` record the per-year adjustment.

All current intervals use normal critical value 1.95996398454 (approximately
1.959964). Threshold-ratio intervals are on the log scale.
The Swedish reference/reference ratio is identically one and has zero width.

Figure 5 propagates the adjusted joint covariance through trends and the
original SR solver by numerical delta propagation. Linear and exponential
Xc trends use 1980--2019 and are anchored at 2019. Future mex stays at its fitted
2019 value, but its estimation uncertainty remains in the covariance. Means
and contour ages are conditional on reaching age 20. The saved values are in
`lifespan_bands.csv`; rendering does not reevaluate the solver.

Denmark uses its own year-specific dispersion and the **adjusted** Swedish
reference covariance, including the Swedish denominator. Danish mex estimates
equal zero in 1960, 1965, 2015 and 2019.

The intervals are approximate pointwise quasi-Poisson sensitivity intervals,
conditional on the estimated dispersion and specified trend form.

The original 100 Poisson recovery datasets and their coverage diagnostics are
retained in `fit_records/`. They do **not** calibrate the current bands. The
superseded split-sample cutoff is retained only at
`diagnostics/poisson_recovery/historical_calibration.json`, outside active
rendering paths.

The original deaths/exposure tables are bundled in `data/hmd/`.
`python3 scripts/prepare_historical_hmd.py` rebuilds the 44,955-row historical
input from checksum-verified sources. See `data/hmd/README.md` for provenance.
Optional portable recomputation and saved-only checks are documented in
`analysis/model_fits/historical/README.md`.
