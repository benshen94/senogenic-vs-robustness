# Sweden 2019 Steepness-Longevity Sensitivity Run

This folder stores checkpointed simulation results for the Fig 1D redo.

The central baseline is the Sweden 2019 period tail-focused SR fit:

\[
\eta=0.586837,\quad
\beta=57.8717,\quad
\kappa=0.5,\quad
\epsilon=49.7187,\quad
X_c=21.7406,\quad
\sigma_{X_c}/X_c=0.141421.
\]

The baseline and parameter-factor curves use \(h_{ext}=0\). The extrinsic
mortality curve is a separate absolute sweep over \(h_{ext}\in[10^{-4},10^{-2}]\).

Baseline sensitivity is one-at-a-time over:

\[
\eta,\beta,\epsilon,X_c,\sigma_{X_c}/X_c
\]

with factors \(0.8\) and \(1.2\). Every finished simulation appends rows to
`metrics_long.csv`, one row per starting age \(t_0\).

Current configured simulation count: 726.

Current Fig. 1D plotting outputs:

- `fig1d_new_steepness_longevity_plot_data.csv`: normalized run-level source data.
- `fig1d_new_steepness_longevity_point_intervals.csv`: factor-wise mean positions and empirical 2.5th to 97.5th percentile sensitivity intervals in \(x\) and \(y\).
- `fig1d_new_steepness_longevity_shaded_envelopes.csv`: polygon coordinates for the displayed shaded ribbons, derived from the point-wise intervals. The ribbon polygons include endpoint caps for display so the terminal markers remain inside the shaded sensitivity region.

The displayed intervals exclude runs whose median lifespan is within 5 years of the simulation endpoint, to avoid finite-\(t_{\max}\)-sensitive steepness artifacts.

The current revision figure is written to:

`Figures/Figure1/fig1d_new_steepness_longevity.png`

and the matching vector file is:

`Figures/Figure1/fig1d_new_steepness_longevity.pdf`
