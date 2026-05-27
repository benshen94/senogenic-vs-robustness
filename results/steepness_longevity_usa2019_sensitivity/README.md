# USA 2019 Steepness-Longevity Sensitivity Run

This folder stores checkpointed simulation results for the Fig. 3 companion steepness-longevity panel.

The central baseline is the USA 2019 period hybrid SR fit:

\[
\eta=0.586837,\quad
\beta=57.8717,\quad
\kappa=0.5,\quad
\epsilon=49.7187,\quad
X_c=20.8549,\quad
\sigma_{X_c}/X_c=0.191853.
\]

This baseline uses the shared Sweden-tail90 fit for \(\eta\), \(\beta\), and \(\epsilon\), with USA-refit \(X_c\) and \(X_c\) heterogeneity from `hybrid2019_swe_tail90_usa_refit`.

The baseline and parameter-factor curves use \(h_{ext}=0\). The extrinsic
mortality curve is a separate absolute sweep over \(h_{ext}\in[10^{-4},10^{-2}]\).

Baseline sensitivity is one-at-a-time over:

\[
\eta,\beta,\epsilon,X_c,\sigma_{X_c}/X_c
\]

with factors \(0.8\) and \(1.2\). Every finished simulation appends rows to
`metrics_long.csv`, one row per starting age \(t_0\).

Current configured simulation count: 605.

The saved metrics file currently contains 616 completed run IDs because the
first checkpointed pass also completed 11 central \(\kappa\) runs. Those rows are
kept for traceability but are not used by the Fig. 3 plotting script.

Fig. 3 plotting outputs:

- `Figures/Figure3/fig3_usa_steepness_longevity.png`
- `Figures/Figure3/fig3_usa_steepness_longevity.pdf`
- `fig3_usa_steepness_longevity_plot_data.csv`
- `fig3_usa_steepness_longevity_point_intervals.csv`
- `fig3_usa_steepness_longevity_shaded_envelopes.csv`

The Fig. 3 panel uses the same plotting convention as the revised Fig. 1D panel:
each baseline-sensitivity scenario is normalized to its own baseline, the curve
shows the mean normalized position, and the shaded ribbons visualize the
empirical 2.5th to 97.5th percentile model-sensitivity envelope in the
steepness-longevity plane.

For plotting, interval summaries exclude runs with median lifespan within 8
years of \(t_{\max}\). This removes finite-horizon-sensitive steepness artifacts
from the shaded ribbons while leaving the raw simulation rows saved in
`metrics_long.csv`.
