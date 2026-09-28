# Senogenic heterogeneity: current FP sensitivity profile

```bash
python3 analysis/figures/extended_data/render_senogenic_profile.py
```

The saved-data renderer reproduces Extended Data Fig. 1 pixel for pixel in the
tested environment. It reads nine profiles and selected refined candidates from
`results/senogenic_heterogeneity/`, with Sweden 2019 deaths and exposures for
ages 20--109. No solver is run when plotting.

The imposed timescale spread uses $v\sim N(0,\sigma^2)$,
$\eta_i=\eta\exp(-v/2)$ and $\beta_i=\beta\exp(v/2)$, where
$\sigma^2=\log(1+\mathrm{CV}_{\tau}^2)$. Independent threshold heterogeneity
is also included. Survival-selected mixtures use the finite-volume boundary
death flux divided by model person-time. Parameters are refitted under common
bounds at each imposed spread.

Panel a displays the increase in the relative-error score from the zero-spread
refit. Panel b shows mortality curves after compensation by the fitted
parameters. These are point estimates, without bootstrap confidence intervals.

Small spread has little effect; larger spread worsens the fit. Selected values
are the lowest scores found, not certified global minima or a confidence bound
on biological heterogeneity. Some candidates reach bounds. Broader checks and
production-grid reevaluation informed selection; the 7.5% coarse candidate
was rejected after reevaluation.

The optional initial profile fitter is
`analysis/model_fits/senogenic_heterogeneity/fit.py`. Setting `LSB_JOBINDEX`
from 1 to 9 selects the imposed spread. New fits go to `tmp/senogenic_refit/`.
The initial fitter, zero-spread refinement, wider-bound stress checks, 4% wider
refinement, candidate ranking and numerical checks are all included. Their
execution order and original cluster allocation are documented in
`analysis/model_fits/senogenic_heterogeneity/README.md`. Every numerical worker
requires `--recompute`. Reproducing the selected candidates requires the full
fitting and numerical-check sequence.
