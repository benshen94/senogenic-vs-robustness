# Methods Log

## 2026-05-19: Sweden 2019 Steepness-Longevity Sensitivity Plane

Goal: remake the steepness-longevity response plane using the Sweden 2019 period SR fit as the baseline, and use shaded regions to show model sensitivity rather than statistical confidence intervals.

Baseline model:

\[
\eta=0.5868368258,\quad
\beta=57.8717377207,\quad
\kappa=0.5,\quad
\epsilon=49.7186593046,\quad
X_c=21.7405634007,\quad
\sigma_{X_c}/X_c=0.1414213562.
\]

The baseline simulations and all parameter-factor curves were run with:

\[
h_{ext}=0.
\]

The extrinsic mortality curve was treated separately as an absolute sweep:

\[
h_{ext}\in[10^{-4},10^{-2}]
\]

using 10 log-spaced values.

Simulation grid:

- Central baseline plus one-at-a-time baseline-sensitivity scenarios.
- Baseline sensitivity parameters:
  \[
  \eta,\ \beta,\ \epsilon,\ X_c,\ \sigma_{X_c}/X_c.
  \]
- Each baseline-sensitivity parameter was multiplied by:
  \[
  0.8,\ 1.2.
  \]
- For each baseline scenario, the focal model parameters were varied over factors:
  \[
  0.5,0.6,\ldots,1.5.
  \]
- The focal parameters were:
  \[
  \eta,\ \beta,\ \kappa,\ \epsilon,\ X_c.
  \]
- Metrics were saved for:
  \[
  t_0=0,15,20,30,40,50.
  \]

The completed grid contains 726 simulations. Long-form metrics are saved in:

`saved_results/steepness_longevity_sweden2019_sensitivity/metrics_long.csv`

For the new plotted Fig2 steepness-longevity panel, the plotted \(x\)-coordinate is:

\[
x=\frac{\text{median lifespan in run}}{\text{median lifespan of that scenario baseline}},
\]

and the plotted \(y\)-coordinate is:

\[
y=\frac{\text{IQR steepness in run}}{\text{IQR steepness of that scenario baseline}}.
\]

This means each sensitivity scenario is normalized to its own baseline before contributing to the sensitivity intervals. The intervals therefore represent how the response curve changes when the baseline fit is perturbed, not how the absolute baseline location shifts.

Sensitivity intervals in the current Fig. 1D draft:

- For each focal parameter and factor value, all baseline-sensitivity reruns were collected as two-dimensional points in the normalized steepness-longevity plane.
- The plotted curve connects the mean point:
  \[
  (\bar{x},\bar{y}).
  \]
- The saved point-wise intervals are empirical 2.5th to 97.5th percentile sensitivity intervals separately in \(x\) and \(y\):
  \[
  x_{2.5\%},x_{97.5\%},\quad y_{2.5\%},y_{97.5\%}.
  \]
- The displayed shaded ribbon is a visualization derived from those point-wise intervals. For each factor value, the \(x/y\) sensitivity box is projected onto the local perpendicular direction of the mean curve, and these local widths are connected into one filled ribbon.
- The ribbon is extrapolated by one local ribbon width beyond the first and last plotted mean points. This endpoint cap is a display choice so terminal markers do not visually sit outside the shaded sensitivity region; it does not add new simulations or change the saved point-wise intervals.
- Points with median lifespan within 5 years of the simulation endpoint were excluded from interval summaries, because these runs can produce finite-\(t_{\max}\)-sensitive IQR steepness artifacts.

These intervals are model-sensitivity intervals across deterministic baseline perturbations. They are not statistical confidence intervals from repeated random sampling.

The plotted parameter classes are:

- Senogenic parameters:
  \[
  \eta,\ \beta.
  \]
- Robustness parameters:
  \[
  X_c,\ \epsilon.
  \]

\(\kappa\) was simulated and remains in the saved source data, but it is not drawn in the new grouped parameter-class figure because it is not part of the current senogenic-versus-robustness visual classification.

For visual stability in the current manuscript-style panel, the visible focal-parameter factor range is:

\[
0.6\leq f\leq 1.4,
\]

except that the displayed \(\eta\) curve starts at \(f=0.7\). Some more extreme high-factor/low-factor combinations produced finite-\(t_{\max}\)-sensitive steepness estimates that were not useful for the displayed response plane, while the full numerical results remain saved.

The displayed axis window follows the current Fig. 1D geometry: a square normalized response plane with matching x/y limits and ticks from 0.4 to 1.6.

The current revision output is saved under:

`Figures_new/Fig1new/fig1d_new_steepness_longevity.png`

The corresponding vector output is:

`Figures_new/Fig1new/fig1d_new_steepness_longevity.pdf`

Color mapping for this figure family:

- Production \(\eta\): teal, `#0B7F8C`.
- Removal \(\beta\): dark blue, `#173A6A`.
- Threshold \(X_c\): orange, `#D77A16`.
- Noise \(\epsilon\): amber, `#E5A100`.
- Extrinsic mortality: red, `#C51F2F`.

Extrinsic mortality is not categorized as a robustness parameter. It is shown separately as a mortality-forcing curve.

Legend convention for parameter-effect plots:

- Senogenic and robustness parameters are shown under bold section headers in one in-panel legend.
- Extrinsic mortality is placed below those classes and separated by a thin solid gray rule with approximately 50% opacity.
- Extrinsic mortality is not labeled as a parameter class in the legend.
- The legend uses a white background without a visible border so it reads as an annotation layer rather than a boxed inset.

## 2026-05-20: USA 2019 Fig. 3 Steepness-Longevity Sensitivity Plane

Goal: make a Fig. 3 companion steepness-longevity response plane using the USA 2019 period fit, with the same parameter classes, color mapping, normalization, and model-sensitivity display used for the revised Fig. 1D panel.

Baseline model:

\[
\eta=0.5868368258,\quad
\beta=57.8717377207,\quad
\kappa=0.5,\quad
\epsilon=49.7186593046,\quad
X_c=20.8549424042,\quad
\sigma_{X_c}/X_c=0.1918528239.
\]

This baseline comes from `hybrid2019_swe_tail90_usa_refit`: the shared Sweden-tail90 period fit supplies \(\eta\), \(\beta\), and \(\epsilon\), while USA 2019 refits \(X_c\) and the fractional \(X_c\) heterogeneity.

The baseline simulations and all focal parameter-factor curves were run with:

\[
h_{ext}=0.
\]

The extrinsic mortality curve was treated separately as an absolute sweep:

\[
h_{ext}\in[10^{-4},10^{-2}]
\]

using 10 log-spaced values.

Simulation grid:

- Central baseline plus one-at-a-time baseline-sensitivity scenarios.
- Baseline sensitivity parameters:
  \[
  \eta,\ \beta,\ \epsilon,\ X_c,\ \sigma_{X_c}/X_c.
  \]
- Each baseline-sensitivity parameter was multiplied by:
  \[
  0.8,\ 1.2.
  \]
- The focal plotted parameters were:
  \[
  \eta,\ \beta,\ \epsilon,\ X_c.
  \]
- For each baseline scenario, focal parameters were varied over factors:
  \[
  0.5,0.6,\ldots,1.5.
  \]
- Metrics were saved for:
  \[
  t_0=0,15,20,30,40,50.
  \]

The configured Fig. 3 simulation grid contains 605 simulations. The saved `metrics_long.csv` currently contains 616 completed run IDs and 3696 metric rows because an initial checkpoint also completed 11 central \(\kappa\) runs. Those extra \(\kappa\) rows remain saved for traceability but are not used by the Fig. 3 plotting script.

The plotted \(x\)-coordinate is:

\[
x=\frac{\text{median lifespan in run}}{\text{median lifespan of that scenario baseline}},
\]

and the plotted \(y\)-coordinate is:

\[
y=\frac{\text{IQR steepness in run}}{\text{IQR steepness of that scenario baseline}}.
\]

As in the revised Fig. 1D workflow, each baseline-sensitivity scenario is normalized to its own baseline before contributing to the plotted mean curve or sensitivity interval. The shaded ribbons therefore show model sensitivity to deterministic baseline-parameter perturbations, not statistical confidence intervals.

For the USA 2019 panel, runs with median lifespan within 8 years of the simulation endpoint were excluded from interval summaries. This stricter finite-horizon filter was needed because one high-\(\beta\), high-baseline-\(\beta\) sensitivity run at \(f_\beta=1.3\) had \(\tilde{t}\approx134.5\) with \(t_{\max}=140\), producing an inflated IQR steepness point and a visible artificial spike in the shaded \(\beta\) ribbon. The underlying simulation row remains in `metrics_long.csv`, but it is not used for the plotted interval envelope.

The displayed shaded ribbons are derived from the empirical point-wise 2.5th to 97.5th percentile variation in \(x\) and \(y\) across the perturbed baselines. For display, each local \(x/y\) interval is converted into a perpendicular ribbon width around the mean curve, with endpoint caps so the first and last plotted markers remain inside the shaded region.

The USA Fig. 3 outputs are saved under:

`Figures_new/Fig3_new/fig3_usa_steepness_longevity.png`

and:

`Figures_new/Fig3_new/fig3_usa_steepness_longevity.pdf`

The corresponding source tables are:

- `saved_results/steepness_longevity_usa2019_sensitivity/metrics_long.csv`
- `saved_results/steepness_longevity_usa2019_sensitivity/fig3_usa_steepness_longevity_plot_data.csv`
- `saved_results/steepness_longevity_usa2019_sensitivity/fig3_usa_steepness_longevity_point_intervals.csv`
- `saved_results/steepness_longevity_usa2019_sensitivity/fig3_usa_steepness_longevity_shaded_envelopes.csv`

## 2026-05-20: Fig. 2 New Sweden-2019 Baseline Survival and Maximum-Lifespan Panels

Goal: remake the Fig. 2 survival-tail and maximum-lifespan parameter panels using one explicit Sweden 2019 baseline fit, with shaded regions that reflect local fit uncertainty in the baseline parameters.

Baseline model:

\[
\eta=0.5868368258,\quad
\beta=57.8717377207,\quad
\kappa=0.5,\quad
\epsilon=49.7186593046,\quad
X_c=21.7405634007.
\]

The baseline is loaded from:

`saved_results/fit_archive/records/joint2019_tail90_sweden_emphasis.json`

The fit-local curvature-based 95% intervals are loaded from:

`saved_results/fit_archive/ci/joint2019_tail90_sweden_emphasis_ci.csv`

These intervals are small local fit intervals, not broad biological perturbations. For the Sweden fit used here, the approximate 95% intervals are:

\[
\eta\in[0.582,0.592],\quad
\beta\in[57.36,58.38],\quad
\epsilon\in[49.23,50.21],\quad
X_c\in[21.41,22.07].
\]

All SR simulations in these panels used:

\[
h_{ext}=0,\quad \kappa=0.5.
\]

The plotted parameter classes and colors follow the current Fig. 1D/Fig. 2 convention:

- Senogenic parameters:
  \[
  \eta,\ \beta.
  \]
- Robustness parameters:
  \[
  X_c,\ \epsilon.
  \]
- Production \(\eta\): teal, `#0B7F8C`.
- Removal \(\beta\): dark blue, `#173A6A`.
- Threshold \(X_c\): orange, `#D77A16`.
- Noise \(\epsilon\): amber, `#E5A100`.

### Fig. 2A New: Conditional Survival From Age 90

Source script:

`src/figures/Fig2_new/make_fig2a_new.py`

Output files:

- `Figures_new/Fig2_new/fig2a_new.png`
- `Figures_new/Fig2_new/fig2a_new.pdf`

Central SR curves:

- Four simulations were run from the Sweden 2019 baseline.
- In each simulation, one parameter was heterogeneous across simulated individuals:
  \[
  \eta,\ \beta,\ X_c,\ \epsilon.
  \]
- The heterogeneous parameter was sampled from a positive Gaussian distribution with:
  \[
  \mathrm{CV}=5\%.
  \]
- All other parameters were fixed at the Sweden 2019 baseline.
- Each central curve used:
  \[
  N=10^6,\quad t_{\max}=150,\quad dt=0.025.
  \]
- Survival was plotted conditional on survival to age 90:
  \[
  S(a\mid 90)=\frac{S(a)}{S(90)}.
  \]

Empirical comparison curves:

- Sweden 2019 and USA 2019 HMD period life-table survival were loaded as both-sex period survival curves.
- Both HMD curves were normalized to survival at age 90 using the same conditional-survival definition:
  \[
  S(a\mid 90)=\frac{S(a)}{S(90)}.
  \]
- The HMD curves are shown as dashed grey lines so they remain visually distinct from the solid SR parameter simulations.

Fit-CI shaded bands:

- For each plotted SR curve, CI bands were generated from one-at-a-time baseline-parameter endpoint simulations.
- For each baseline parameter:
  \[
  \eta,\ \beta,\ \epsilon,\ X_c,
  \]
  the baseline was replaced by either its lower or upper 95% fit interval endpoint while the other baseline parameters remained fixed.
- For every endpoint baseline, the same 5% heterogeneity simulation was rerun for each focal heterogeneous parameter.
- Each CI-endpoint survival simulation used:
  \[
  N=5\times10^5,\quad t_{\max}=150,\quad dt=0.025.
  \]
- At each age and for each focal curve, the shaded band is the pointwise min/max envelope across:
  \[
  \text{central baseline},\quad \text{lower endpoint runs},\quad \text{upper endpoint runs}.
  \]

This shaded region is therefore a fit-local baseline-uncertainty envelope. It is not an HMD sampling confidence interval, and it is not a bootstrap over period life tables.

Source tables:

- `saved_results/csv/fig2a_new_conditional_survival.csv`
- `saved_results/csv/fig2a_new_fit_ci_envelopes.csv`

### Fig. 2B New: Maximum Lifespan Under Parameter Heterogeneity

Source script:

`src/figures/Fig2_new/make_fig2bc_new.py`

Output files:

- `Figures_new/Fig2_new/fig2b_new.png`
- `Figures_new/Fig2_new/fig2b_new.pdf`

Definition of maximum lifespan:

\[
L_{\max}=\{t:S(t)=10^{-4}\}.
\]

Operationally, \(L_{\max}\) was estimated from simulated death times as the age at which the unconditional survival fraction reaches \(10^{-4}\).

Central curves:

- The focal parameter was sampled from a positive Gaussian distribution around its Sweden 2019 baseline value.
- The tested heterogeneity levels were:
  \[
  \mathrm{CV}=0\%,5\%,10\%,15\%,20\%.
  \]
- Parameters were varied one at a time:
  \[
  \eta,\ \beta,\ X_c,\ \epsilon.
  \]
- All non-focal parameters stayed fixed at the Sweden 2019 baseline.
- Current central Fig. 2B rows used:
  \[
  N=2\times10^5,\quad t_{\max}=420,\quad dt=0.1.
  \]

Fit-CI shaded bands:

- For each focal parameter curve, the CI band was built by moving that same focal baseline parameter to its lower and upper 95% fit interval endpoints.
- The heterogeneity CV sweep was then rerun from those lower and upper endpoint baselines.
- Current CI-endpoint Fig. 2B rows used:
  \[
  N=2\times10^5,\quad t_{\max}=420,\quad dt=0.1.
  \]
- At each heterogeneity level, the shaded band is the pointwise min/max envelope across:
  \[
  \text{central baseline},\quad \text{focal lower endpoint},\quad \text{focal upper endpoint}.
  \]
- The Fig. 2B central and CI-endpoint simulations use common random numbers for each focal parameter and heterogeneity level. The sampled heterogeneous parameter distribution also uses the same random draw stream for the central, lower-endpoint, and upper-endpoint baselines. This makes the envelope reflect the fit-endpoint parameter shift rather than independent Monte Carlo tail noise.
- At \(0\%\) heterogeneity, all focal-parameter curves share the same central stochastic stream, so the baseline point is visually aligned.

Important interpretation note: the very large rise in \(L_{\max}\) for heterogeneity in \(\eta\), and to a lesser extent \(\beta\), is the model result being visualized. The width of the teal band at high CV reflects the sensitivity of this absolute extreme-tail quantity to small local shifts in the fitted \(\eta\), amplified by the heterogeneous tail.

Source tables:

- `saved_results/csv/fig2b_new_max_lifespan_heterogeneity.csv`
- `saved_results/csv/fig2b_new_fit_ci_envelopes.csv`

### Fig. 2C New: Maximum Lifespan Under Mean-Parameter Shifts

Source script:

`src/figures/Fig2_new/make_fig2bc_new.py`

Output files:

- `Figures_new/Fig2_new/fig2c_new.png`
- `Figures_new/Fig2_new/fig2c_new.pdf`

Definition of maximum lifespan:

\[
L_{\max}=\{t:S(t)=10^{-4}\}.
\]

Central curves:

- Each focal parameter was multiplied one at a time by:
  \[
  0.85,\ 0.90,\ 0.95,\ 1.00,\ 1.05,\ 1.10,\ 1.15.
  \]
- The focal parameters were:
  \[
  \eta,\ \beta,\ X_c,\ \epsilon.
  \]
- All other parameters stayed fixed at the Sweden 2019 baseline.
- Current Fig. 2C rows used:
  \[
  N=2\times10^5,\quad t_{\max}=420,\quad dt=0.1.
  \]

Fit-CI shaded bands:

- For each focal parameter, the central factor sweep was rerun after replacing the focal baseline parameter by its lower or upper 95% fit interval endpoint.
- At each factor value, the shaded band is the pointwise min/max envelope across:
  \[
  \text{central baseline},\quad \text{focal lower endpoint},\quad \text{focal upper endpoint}.
  \]
- The Fig. 2C central and CI-endpoint simulations use common random numbers for each focal parameter and factor value. This reduces visual tail noise and makes the shaded interval reflect the parameter endpoint change rather than independent Monte Carlo scatter.
- At factor \(1.0\), all focal-parameter curves share the same central stochastic stream, so the baseline point is visually aligned.
- No smoothing or recentering is applied to the Fig. 2C curves or shaded bands.

Source tables:

- `saved_results/csv/fig2c_new_max_lifespan_factor.csv`
- `saved_results/csv/fig2c_new_fit_ci_envelopes.csv`

### Visual QA Notes

The current Fig. 2C PNG was inspected directly after rerendering. The shaded regions look appropriate: they are narrow, smooth, and visually centered around the solid curves, which is what we expect from fit-local 95% parameter intervals and common-random-number simulation.

The current Fig. 2A PNG was also inspected directly. The shaded regions are readable through the main survival range; near the bottom of the log-scale tail, some stair-step behavior remains because the plot is close to the finite-simulation survival floor.

The current Fig. 2B PNG was rerun at \(N=2\times10^5\), matching Fig. 2C, and inspected directly. The shaded regions now look appropriate: the senogenic-parameter bands are smooth and visible, while the robustness-parameter bands remain narrow because the fit-local endpoint intervals have little effect on the absolute maximum-lifespan estimate for those curves.

## 2026-05-20: Fig. 6 Progeria Survival and Sweden-Baseline Parameter Fits

Goal: remake the progeria figure using the Sweden 2019 period SR fit as the baseline, show uncertainty around the HGPS survival curve, and test whether one- or two-parameter shifts can reproduce the HGPS survival shape.

Source script:

`src/figures/Fig6_progeria/make_fig6_progeria.py`

Output files:

- `Figures_new/Fig6_progeria/fig6a_progeria_hgps_survival.png`
- `Figures_new/Fig6_progeria/fig6b_single_parameter_fits.png`
- `Figures_new/Fig6_progeria/fig6c_two_parameter_fits.png`
- `Figures_new/Fig6_progeria/fig6_progeria_composite.png`

Vector versions of the same panels are saved as PDFs in the same folder.

### HGPS Survival Curve and Bootstrap Band

The central HGPS survival curve was loaded from:

`saved_results/progeria_data.pkl`

Specifically, the script uses the smooth survival curve:

\[
S_{\mathrm{HGPS}}(t).
\]

The HGPS uncertainty band is a percentile bootstrap over lifespans implied by this smooth survival function. For each bootstrap replicate:

1. Draw \(n=202\) independent survival quantiles:
   \[
   u_i\sim \mathrm{Uniform}(0,1),\quad i=1,\ldots,202.
   \]
2. Convert each quantile to a lifespan by inverse interpolation through the smooth survival curve:
   \[
   T_i=S_{\mathrm{HGPS}}^{-1}(u_i).
   \]
3. Recompute the empirical survival curve on the HGPS age grid:
   \[
   \hat{S}^{(b)}(t)=\frac{1}{202}\sum_{i=1}^{202}\mathbf{1}\{T_i>t\}.
   \]

The displayed shaded HGPS band is the pointwise 2.5th to 97.5th percentile interval across bootstrap replicates:

\[
\left[
Q_{0.025}\left(\hat{S}^{(b)}(t)\right),
Q_{0.975}\left(\hat{S}^{(b)}(t)\right)
\right].
\]

The current rendered figure used:

\[
B=1000
\]

bootstrap replicates. This band is a sampling interval for a cohort of size \(202\) drawn from the fitted HGPS survival distribution. It is not a confidence interval from a new parametric survival fit.

Panel A also includes Sweden and USA 2019 period life-table survival curves from HMD. Each curve is plotted against age normalized by its own median lifespan:

\[
x=\frac{t}{t_{50}},
\quad
S(t_{50})=0.5.
\]

This normalization matches the old Fig. 6A visual comparison: the HGPS curve is shallow relative to period control populations even after aligning median lifespan.

### Sweden 2019 Baseline

All SR model fits in panels B and C start from the Sweden 2019 period tail-focused baseline:

\[
\eta=0.5868368258,\quad
\beta=57.8717377207,\quad
\kappa=0.5,\quad
\epsilon=49.7186593046,\quad
X_c=21.7405634007,\quad
\sigma_{X_c}/X_c=0.1414213562.
\]

The baseline is loaded from:

`saved_results/fit_archive/records/joint2019_tail90_sweden_emphasis.json`

The fitted local 95% endpoint intervals are loaded from:

`saved_results/fit_archive/ci/joint2019_tail90_sweden_emphasis_ci.csv`

The \(X_c\) heterogeneity term is held fixed at the Sweden 2019 value during HGPS fitting. For simulations, \(X_c\) is sampled from a positive Gaussian distribution with mean \(X_c\) and fractional standard deviation:

\[
\frac{\sigma_{X_c}}{X_c}=0.1414213562.
\]

All other fitted parameters are scalar population-level values.

### Single-Parameter Fits

Panel B asks whether a one-parameter shift from the Sweden 2019 baseline can match the HGPS survival shape. The fitted one-parameter cases are:

\[
\eta,\quad \beta,\quad X_c,\quad \epsilon.
\]

For each case, only the focal parameter is changed. All other parameters remain fixed at the Sweden 2019 baseline. The fitted parameter is represented as a log-two factor around the baseline:

\[
\theta=\theta_{\mathrm{SWE2019}}\,2^z.
\]

The optimization minimizes a weighted survival RMSE on the HGPS age grid:

\[
\sqrt{
\frac{\sum_t w(t)\left[S_{\mathrm{model}}(t)-S_{\mathrm{HGPS}}(t)\right]^2}
{\sum_t w(t)}
},
\]

with more weight in the lower-survival portion of the curve:

\[
w(t)=0.4+0.6\left(1-S_{\mathrm{HGPS}}(t)\right).
\]

Survival values are clipped below a small floor before scoring:

\[
S(t)\geq 0.015.
\]

This avoids the optimizer being dominated by finite-simulation behavior at the extreme bottom of the survival curve.

The current single-parameter fits are:

- \(\eta\times 7.03\), cost \(0.10909\).
- \(\beta\times 0.229\), cost \(0.08404\).
- \(X_c\times 0.336\), cost \(0.18846\).
- \(\epsilon\times 4.36\), cost \(0.18348\).

These one-parameter fits are shown in panel B with the HGPS bootstrap band. The legend labels report the factor shift relative to the Sweden 2019 baseline.

### Two-Parameter Fits

Panel C asks whether two-parameter shifts can recover the HGPS survival shape, and separates pairs that do or do not include a senogenic parameter.

The robustness-only pair is:

\[
X_c+\epsilon.
\]

The senogenic-containing pairs are:

\[
\eta+X_c,\quad
\eta+\epsilon,\quad
\beta+X_c,\quad
\beta+\epsilon,\quad
\eta+\beta.
\]

Senogenic parameters are:

\[
\eta,\quad \beta.
\]

Robustness parameters are:

\[
X_c,\quad \epsilon.
\]

The optimizer uses a multi-start grid over log-two parameter factors, followed by local coordinate pattern search. The search is deterministic for a given parameter set because each candidate in the objective uses a fixed simulation seed for that fit family. This reduces stochastic jumps in the objective while preserving simulation-based survival curves.

The current two-parameter fits are:

- \(X_c\times 0.439+\epsilon\times 1.41\), cost \(0.18148\).
- \(\eta\times 3.08+X_c\times 0.420\), cost \(0.05571\).
- \(\eta\times 4.80+\epsilon\times 2.38\), cost \(0.02060\).
- \(\beta\times 0.149+X_c\times 1.41\), cost \(0.05851\).
- \(\beta\times 0.104+\epsilon\times 0.310\), cost \(0.02689\).
- \(\eta\times 1.96+\beta\times 0.339\), cost \(0.01866\).

Thus, the robustness-only \(X_c+\epsilon\) fit remains poor, while several fits containing \(\eta\) or \(\beta\) reproduce the HGPS survival shape much better.

### Model Shaded Bands

The model shaded bands around the fitted SR curves are endpoint envelopes based on the Sweden 2019 fit-local 95% intervals. For each fitted parameter, the fitted factor is applied to the lower and upper Sweden 2019 CI endpoints for that same parameter. For a one-parameter fit this gives two endpoint curves. For a two-parameter fit, all lower/upper endpoint combinations are simulated.

At each age, the displayed model band is:

\[
\left[
\min_j S_j(t),\quad
\max_j S_j(t)
\right],
\]

where \(j\) indexes the central fitted curve and the endpoint-combination curves.

These model bands are therefore fit-local Sweden-baseline endpoint envelopes. They are not bootstrap confidence intervals for the HGPS data, and they are not a full optimizer-derived uncertainty interval for the HGPS refit.

Current rendered curves used:

\[
N=5000,\quad t_{\max}=35,\quad dt=0.025.
\]

The current fitting pass used:

\[
N=800
\]

agents per objective evaluation.

Source tables:

- `saved_results/csv/fig6_progeria_hgps_bootstrap_envelope.csv`
- `saved_results/csv/fig6_progeria_period_survival.csv`
- `saved_results/csv/fig6_progeria_fit_results.csv`
- `saved_results/csv/fig6_progeria_survival_curves.csv`
- `saved_results/csv/fig6_progeria_model_ci_envelopes.csv`

Simulation and fitting cache files:

- `saved_results/cache/simulations/Fig6_progeria/fit_results.json`
- `saved_results/cache/simulations/Fig6_progeria/survival_curves.csv`
- `saved_results/cache/simulations/Fig6_progeria/model_ci_envelopes.csv`
- `saved_results/cache/simulations/Fig6_progeria/metadata.json`

## 2026-05-23: Current New-Figure Methods Audit

This entry records the figure-generation methods currently implemented in the new manuscript and supplement scripts that were not already covered above. It is based on the code in `Figures_new/` and `src/figures/` as of this audit. Presentation-only Illustrator/assembly scripts are listed where they affect figure composition, but the methods below emphasize data generation, fitting, projection, and uncertainty calculations.

General export rule used by the current figure scripts:

- Matplotlib vector exports set `pdf.fonttype: 42`, `ps.fonttype: 42`, and usually `svg.fonttype: "none"` so PDF/SVG text remains editable whenever possible.
- New figure outputs are written under `Figures_new/`.
- Generated source tables and simulation caches are written under `saved_results/csv/`, `saved_results/fig*_*/`, and `saved_results/cache/simulations/`.

## Fig. 1 New: Conceptual Panels and Sweden 2019 Response Plane

Composite source script:

`Figures_new/Fig1new/make_fig1_six_panel_composite.py`

Composite outputs:

- `Figures_new/Fig1new/Fig1_new.png`
- `Figures_new/Fig1new/Fig1_new.pdf`

The six-panel composite places separately generated PDF panels onto one fixed page using `pypdf`, then renders a PNG preview with `fitz`. The composite script is layout-only; the scientific content is generated by the panel scripts below.

### Fig. 1A-B: Stochastic Threshold and Parameter-Class Schematics

Source script:

`Figures_new/Fig1new/make_fig1_panels_ab.py`

Outputs:

- `Figures_new/Fig1new/fig1_panel_a_stochastic_threshold.svg`
- `Figures_new/Fig1new/fig1_panel_a_stochastic_threshold.pdf`
- `Figures_new/Fig1new/fig1_panel_a_stochastic_threshold.png`
- `Figures_new/Fig1new/fig1_panel_b_parameter_classes.svg`
- `Figures_new/Fig1new/fig1_panel_b_parameter_classes.pdf`
- `Figures_new/Fig1new/fig1_panel_b_parameter_classes.png`

These are editable vector schematics. They are not simulation-derived analyses. Panel A illustrates stochastic damage/threshold trajectories and threshold crossing as a conceptual model diagram. Panel B illustrates the parameter-class mapping used throughout the figure family:

- Senogenic parameters: production \(\eta\), removal \(\beta\).
- Robustness parameters: threshold \(X_c\), noise \(\epsilon\).
- Extrinsic mortality is kept visually separate from both classes.

The script draws SVG primitives directly and embeds `Figures_new/Fig1new/clock.png` where needed.

### Fig. 1C: Mortality-Signature Schematic

Source script:

`Figures_new/Fig1new/make_fig1_panel_c_mortality_signatures.py`

Outputs:

- `Figures_new/Fig1new/fig1_panel_c_mortality_signatures.svg`
- `Figures_new/Fig1new/fig1_panel_c_mortality_signatures.pdf`
- `Figures_new/Fig1new/fig1_panel_c_mortality_signatures.png`

This is a schematic, not a fitted-data panel. It contrasts:

- Robustness-like mortality curves that share a common late-life compensation point.
- Senogenic-like curves made of parallel shifts and slope changes that do not require one common compensation line.

The panel uses prescribed line slopes, intercepts, and colors to explain the conceptual distinction between robustness-like and senogenic-like mortality signatures.

### Fig. 1D: SR Model and Parameter-Class Mapping

Source script:

`src/figures/Fig1_new/make_fig1_newc.py`

Outputs:

- `Figures_new/Fig1new/fig1_panel_d_sr_model.png`
- `Figures_new/Fig1new/fig1_panel_d_sr_model.pdf`
- `Figures_new/Fig1new/fig1_panel_d_sr_model.svg`

This is a model-equation schematic. It visually maps the SR model parameters onto the senogenic and robustness classes used in the response-plane figures. It is not a new numerical fit or simulation.

### Fig. 1E: Survival Scaling Versus Steepening Schematic

Source script:

`Figures_new/Fig1new/make_fig1_panel_e_survival_scaling.py`

Outputs:

- `Figures_new/Fig1new/fig1_panel_e_survival_scaling.png`
- `Figures_new/Fig1new/fig1_panel_e_survival_scaling.pdf`
- `Figures_new/Fig1new/fig1_panel_e_survival_scaling.svg`

This panel uses illustrative Weibull-like survival curves:

\[
S(t)=\exp\left[-\log(2)\left(\frac{t}{t_{50}}\right)^k\right].
\]

The scaling row compares curves with the same shape parameter:

\[
k=6.2,\quad t_{50}=58,\ 82.
\]

After normalizing age by median lifespan, these curves overlap. The steepening row compares:

\[
(t_{50},k)=(58,6.2),\quad (74,10.5).
\]

After age normalization, the changed curve remains steeper, illustrating the distinction between simple lifespan scaling and survival-shape steepening.

### Fig. 1F: Sweden 2019 Steepness-Longevity Response Plane

The methodology for this panel is the Sweden 2019 sensitivity-plane entry above. Current outputs are:

- `Figures_new/Fig1new/fig1_panel_f_steepness_longevity.png`
- `Figures_new/Fig1new/fig1_panel_f_steepness_longevity.pdf`
- `Figures_new/Fig1new/fig1d_new_steepness_longevity.png`
- `Figures_new/Fig1new/fig1d_new_steepness_longevity.pdf`

The panel uses the Sweden 2019 SR baseline, one-at-a-time focal parameter factors, scenario-normalized median lifespan and IQR steepness, and model-sensitivity ribbons derived from deterministic baseline perturbations. The ribbons are not statistical confidence intervals.

## Fig. 2D-E New: Sibling Mortality Under Parameter Heterogeneity

Source script:

`src/figures/Fig2_new/make_fig2de_new.py`

Outputs:

- `Figures_new/Fig2_new/fig2de_new.png`
- `Figures_new/Fig2_new/fig2de_new.pdf`

Source tables and caches:

- `saved_results/csv/fig2de_new_mortality_curves.csv`
- `saved_results/csv/fig2de_new_fit_ci_envelopes.csv`
- `saved_results/csv/fig2d_raw_digitized_points.csv`
- `saved_results/cache/simulations/Fig2_new/fig2de_new_plot_records.pkl`
- `saved_results/cache/simulations/Fig2_new/fig2de_new_metadata.json`

Baseline:

The Sweden 2019 SR baseline and fit-local CI endpoints are loaded from:

- `saved_results/fit_archive/records/joint2019_tail90_sweden_emphasis.json`
- `saved_results/fit_archive/ci/joint2019_tail90_sweden_emphasis_ci.csv`

The baseline uses:

\[
h_{ext}=0,\quad \kappa=0.5,\quad t_{\max}=150,\quad dt=0.025.
\]

Simulation design:

- The simulated family structure is DZ sibling pairs using `family="DZ"`.
- One focal parameter is heterogeneous at a time:
  \[
  X_c,\ \epsilon,\ \eta,\ \beta.
  \]
- Focal heterogeneity is sampled from a positive Gaussian distribution.
- Relative standard deviations are:
  \[
  \sigma_{X_c}/X_c=0.20,\quad
  \sigma_{\epsilon}/\epsilon=0.30,\quad
  \sigma_{\eta}/\eta=0.15,\quad
  \sigma_{\beta}/\beta=0.10.
  \]
- Central simulations use:
  \[
  N=2,000,000.
  \]
- Fit-CI endpoint simulations use:
  \[
  N=1,000,000.
  \]

Sibling-group definitions:

- Bad-survivor probands are individuals with death times at or below the bottom 10th percentile of simulated death ages.
- Good-survivor probands are individuals surviving to the age where cohort survival reaches 1%.
- For each selected proband, the paired DZ sibling death time is extracted.
- Full-cohort mortality is plotted alongside sibling-selected mortality.

Mortality estimation:

- Sibling mortality curves are estimated with the Nelson-Aalen estimator from `lifelines`.
- Smoothed hazards use `NelsonAalenFitter.smoothed_hazard_(bandwidth=3)`.
- The visible age window is 50 to 115 years.

Fit-CI shaded bands:

- For each focal parameter panel, the focal baseline parameter is replaced by the lower or upper 95% fit endpoint, and the full DZ simulation is rerun.
- At each age, the band is the pointwise min/max envelope across the central and endpoint curves.
- These bands are fit-local endpoint envelopes, not empirical sibling-data confidence intervals.

Empirical comparison:

The plotted brother/sister mortality points are digitized from the source sibling-mortality panel and saved to `saved_results/csv/fig2d_raw_digitized_points.csv` for auditability. The script stores the series name, sibling sex, proband group, age, and log10 mortality for every plotted point.

## Fig. 3 New: NHANES Exposure Projection Onto the USA 2019 \(X_c\) Response Curve

Source script:

`src/figures/steepness_longevity/make_fig3_exposure_projection.py`

Outputs:

- `Figures_new/Fig3_new/fig3_exposure_projection_panel_a.png`
- `Figures_new/Fig3_new/fig3_exposure_projection_panel_b.png`
- `Figures_new/Fig3_new/fig3_exposure_projection_panel_b.pdf`
- `Figures_new/Fig3_new/fig3_exposure_projection_panel_c.png`
- `Figures_new/Fig3_new/fig3_exposure_projection.png`
- `Figures_new/Fig3_new/fig3_exposure_projection.pdf`

Source tables:

- `saved_results/fig3_exposure_projection/exposure_xc_projection.csv`
- `saved_results/fig3_exposure_projection/fig3_panel_b_xc_factor_curves_fit_ci.csv`
- `saved_results/fig3_exposure_projection/fig3_panel_b_xc_factor_curves_fit_ci_raw.csv`
- `saved_results/fig3_exposure_projection/projected_xc_age_gain_curves.csv`
- `saved_results/fig3_exposure_projection/fig3_panel_b_projection_uncertainty_ribbons.csv`

### Fig. 3A: Exposure Points on the Response Plane

The USA 2019 response-plane underlay is loaded from the Fig. 3 USA sensitivity workflow described above. The exposure points are loaded from:

`saved_results/exposure_groups_results.pkl`

The baseline normalization uses the `without_extrinsic` NHANES baseline. For each exposure group:

\[
x=\frac{\text{group median lifespan}}{\text{baseline median lifespan}},
\quad
y=\frac{\text{group IQR steepness}}{\text{baseline IQR steepness}}.
\]

The plotted exposure categories are:

- Diet.
- Number of friends.
- Income.
- Alcohol.
- Physical activity.
- Sleep duration.
- Sleep frailty.
- Church attendance.
- Education level.

Each exposure point is projected to the closest point on the USA 2019 \(X_c\) mean response curve. The \(X_c\) curve is interpolated with a monotone PCHIP interpolator over the saved Fig. 3 \(X_c\) factor rows.

Projection uncertainty:

- Exposure \(x/y\) errors are read from the exposure-group results when present.
- The script draws 3000 Monte Carlo samples around each exposure point using independent normal errors in \(x\) and \(y\).
- Each sampled point is projected to the closest point on the interpolated \(X_c\) curve.
- The saved interval is the 2.5th to 97.5th percentile range of projected \(X_c\) factors.

This is an approximate projection interval, not a causal estimate and not a full NHANES survival-model confidence interval.

### Fig. 3B: Model-Predicted Extra Years From \(X_c\) Changes

Panel B simulates age-specific median remaining-lifespan gains from \(X_c\) factor changes around the USA 2019 baseline.

Baseline:

- The USA 2019 SR baseline is loaded from `saved_results/steepness_longevity_usa2019_sensitivity/manifest.json`.
- Fit-local USA endpoints are loaded from `saved_results/fit_archive/ci/hybrid2019_swe_tail90_usa_refit_ci.csv`.
- Only USA-specific \(X_c\) and fractional \(X_c\) heterogeneity endpoints are used for the fit-CI scenarios.

Simulation settings:

\[
N=300,000,\quad t_{\max}=140,\quad dt=0.025.
\]

The factor grid is:

\[
X_c\ \text{factor}=0.80,0.85,\ldots,1.20.
\]

For each factor and baseline scenario:

- \(X_c\) is multiplied by the factor.
- \(X_c\) is sampled as a positive Gaussian with the baseline fractional \(X_c\) standard deviation.
- Other parameters are fixed to the USA 2019 baseline for that scenario.
- Median remaining lifespan is computed at ages 60, 65, ..., 100 among simulated individuals alive at each age.

The gain curve is:

\[
\Delta R(a,f)=R(a,f)-R(a,1),
\]

where \(R(a,f)\) is median remaining lifespan at age \(a\) under factor \(f\), and the baseline \(R(a,1)\) is matched within each endpoint scenario.

Uncertainty shown in the panel:

- Fit uncertainty is propagated by rerunning the factor grid under central, lower-endpoint, and upper-endpoint USA \(X_c\)/heterogeneity scenarios.
- Projection uncertainty is propagated by taking local projected-factor intervals from the nearest five exposure groups, then interpolating the simulated gain curves at those lower and upper factor bounds.
- The visible per-line ribbons in panel B use the saved projection-uncertainty ribbons.

The uncertainty method recorded in the table is `fit_ci_plus_projection_boundaries_v1`.

### Fig. 3C: Healthy-Lifestyle Data Comparison

Panel C redraws old healthy-lifestyle gain curves as a comparison panel. The curves are hard-coded from the old Fig. 3C data values at ages:

\[
40,50,60,70,80,90,100.
\]

The curves are smoothed for display with PCHIP interpolation from age 60 to 100. This panel is a visual comparison to published/legacy lifestyle-gain curves, not a new SR simulation.

## Fig. 4 New: Historical Period Projections, Extrinsic Mortality, and Robustness

Primary source scripts:

- `src/figures/Fig4_new/make_fig4_ab_sweden_period_projection.py`
- `src/figures/Fig4_new/make_fig4_sr_contour_projection.py`

Assembly scripts:

- `src/figures/Fig4_new/assemble_figure4_illustrator.jsx`
- `src/figures/Fig4_new/assemble_figure4_illustrator_editable.jsx`

Outputs:

- `Figures_new/Fig4_new/fig4_ab_sweden_period_projection.png`
- `Figures_new/Fig4_new/fig4_ab_sweden_period_projection.pdf`
- `Figures_new/Fig4_new/fig4_ab_denmark_period_projection.png`
- `Figures_new/Fig4_new/fig4_ab_denmark_period_projection.pdf`
- `Figures_new/Fig4_new/Fig4C.png`
- `Figures_new/Fig4_new/Fig4C.pdf`
- `Figures_new/Fig4_new/Fig4D.png`
- `Figures_new/Fig4_new/Fig4D.pdf`
- `Figures_new/Fig4_new/Fig4D_extrap.png`
- `Figures_new/Fig4_new/Fig4D_extrap.pdf`
- `Figures_new/Fig4_new/sweden_sr_contour_projection_1900_2100_n1m.png`
- `Figures_new/Fig4_new/sweden_sr_contour_projection_1900_2100_n1m.pdf`
- `Figures_new/Fig4_new/Figure4_new.png`
- `Figures_new/Fig4_new/Figure4_new.pdf`

### Fig. 4A-B: Sweden and Denmark Period Coordinates on the SR Response Plane

Country period data:

- Sweden HMD period both-sex data, years 1800-2020 where available.
- Denmark HMD period both-sex data, years 1835-2020 where available.

The period calculations use conditional lifespan after:

\[
t_0=20.
\]

For each country-year with extrinsic mortality included, the script calculates HMD lifespan quantile ages:

\[
S(t\mid t_0)=0.75,\ 0.5,\ 0.25.
\]

It then computes:

\[
\text{median lifespan}=t_{50},
\quad
\mathrm{IQR}=t_{25}-t_{75},
\quad
\text{IQR steepness}=\frac{t_{50}}{\mathrm{IQR}}.
\]

For the extrinsic-removed condition, a Gamma-Gompertz-Makeham model is fitted to HMD period mortality:

\[
\text{fit ages}=20\ \text{to}\ 100.
\]

The fitted Makeham term \(m\) is then set to zero, and the model survival curve is integrated over:

\[
t_{\max}=180,\quad dt=0.05.
\]

The same conditional quantiles and IQR steepness are computed from the no-Makeham survival curve.

Normalization:

- The plotted Fig. 4A-B coordinates are normalized to the central Sweden 2019 SR response-plane baseline at \(t_0=20\), loaded from the Sweden 2019 sensitivity metrics.
- The saved tables also include normalization to the 2019 country reference year and the old 2020 reference year for traceability.

Source tables:

- `saved_results/fig4_new/sweden_period_steepness_longevity_projection.csv`
- `saved_results/fig4_new/denmark_period_steepness_longevity_projection.csv`
- `saved_results/fig4_new/country_period_steepness_longevity_projection.csv`
- `saved_results/fig4_new/README.md`

### Fig. 4C: Fitted Extrinsic Mortality Over Time

Fig. 4C uses Sweden period Gamma-Gompertz-Makeham fits. For each year:

- Fit the Gamma-Gompertz-Makeham model on ages 20-100.
- Extract the fitted Makeham term \(m\).
- Request covariance-derived standard errors from the fitting routine.
- Convert the standard error to a positive 95% interval using a lognormal delta-method:

\[
m_{\mathrm{low}}=m\exp(-1.95996\,\mathrm{SE}(m)/m),
\quad
m_{\mathrm{high}}=m\exp(1.95996\,\mathrm{SE}(m)/m).
\]

Nonfinite standard errors are linearly interpolated by year before the interval calculation.

Source tables:

- `saved_results/fig4_new/sweden_period_makeham_m_fit_ci.csv`
- `saved_results/fig4_new/fig4c_extrinsic_mortality_projection.csv`

### Fig. 4D: Historical Robustness Projection

Each Sweden extrinsic-removed point from Fig. 4A-B is projected onto the Sweden 2019 \(X_c\) response curve. Projection method:

- Load finite Sweden 2019 model rows for focal parameter \(X_c\).
- Exclude rows whose median lifespan is within 5 years of the simulation endpoint.
- Build a mean \(X_c\) response curve by averaging \(x/y\) coordinates at each focal value across sensitivity scenarios.
- For each historical point, find the closest position along the piecewise-linear \(X_c\) curve.
- Estimate \(X_c\) factor by linear interpolation along the curve.

Uncertainty:

- The historical point is also projected onto each scenario-specific \(X_c\) curve with enough values.
- The saved interval is the 2.5th to 97.5th percentile of those scenario-specific projected factors.
- The 1918 robustness estimate and interval are replaced by the mean of the adjacent 1917 and 1919 values because the source period point is treated as a known pandemic-era artifact for this projection.

Source table:

`saved_results/fig4_new/fig4d_robustness_projection.csv`

### Fig. 4D Extrapolation

The extrapolation panel uses observed Sweden robustness estimates through 2020 and fits trends over:

\[
1980\leq \text{year}\leq 2020.
\]

Two projections are saved from 2020 to 2100:

- Linear extrapolation: fit a line to \(X_c\) factor versus year.
- Exponential extrapolation: fit a line to \(\log(X_c\ \text{factor})\) versus year.

Source table:

`saved_results/fig4_new/fig4d_robustness_extrapolation.csv`

### Fig. 4 SR Contour Forecast

Source script:

`src/figures/Fig4_new/make_fig4_sr_contour_projection.py`

This panel combines the Fig. 4C extrinsic-mortality series and Fig. 4D robustness series into SR simulations of lifespan contours.

Inputs:

- `saved_results/fig4_new/fig4c_extrinsic_mortality_projection.csv`
- `saved_results/fig4_new/fig4d_robustness_projection.csv`
- `saved_results/fig4_new/sweden_period_both_conditional_age20_contours_1900_2020.csv`

Simulation settings:

\[
N=1,000,000,\quad t_0=20,\quad \text{save time}=t_{\max}.
\]

The baseline \(\eta,\beta,\kappa,\epsilon,X_c,\sigma_{X_c}/X_c,t_{\max},dt\) are imported from the Sweden 2019 sensitivity run. For each target year:

- \(X_c\) is multiplied by the year-specific projected \(X_c\) factor.
- Individual \(X_c\) values are sampled from a positive Gaussian distribution with the Sweden baseline fractional \(X_c\) heterogeneity.
- Extrinsic mortality \(h_{ext}\) is set to the year-specific fitted or projected value.

Historical years are simulated every 10 years from 1900 to 2020. Forecast years are simulated every 10 years from 2020 to 2100 under linear and exponential trend scenarios.

For each target year and scenario, three variants are simulated:

- Central: central \(X_c\) and central \(h_{ext}\).
- Low: low \(X_c\) envelope with high \(h_{ext}\).
- High: high \(X_c\) envelope with low \(h_{ext}\).

The saved lifespan contours are ages at survival levels conditional on survival to age 20:

\[
S(t\mid 20)=0.5,\ 0.1,\ 0.01,\ 10^{-4}.
\]

Source tables:

- `saved_results/fig4_new/sweden_sr_contour_projection_inputs_n1m.csv`
- `saved_results/cache/simulations/Fig4_new/sweden_sr_contour_projection_full_n1m.csv`
- `saved_results/fig4_new/sweden_sr_contour_projection_summary_n1m.csv`
- `saved_results/fig4_new/sweden_fig4d_conservative_xc_envelope_1900_2020.csv`

## Fig. S1 New: Parameter Distributions Among Lifespan Strata

Source script:

`src/figures/FigS1/make_parameter_distribution_supplement.py`

Outputs:

- `Figures_new/FigS1/figs1_parameter_distributions_pretty.png`
- `Figures_new/FigS1/figs1_parameter_distributions_pretty.pdf`

Source cache:

- `saved_results/cache/simulations/FigS1/parameter_distribution_supplement.npz`
- `saved_results/cache/simulations/FigS1/parameter_distribution_supplement_metadata.json`

This figure rebuilds the legacy notebook analysis from:

`src/notebooks/param_distributions_investigation.ipynb`

Baseline:

- Start from `utils.load_baseline_human_params_dict()`.
- Multiply:
  \[
  X_c\leftarrow1.08X_c,\quad
  \eta\leftarrow1.26\eta,\quad
  \beta\leftarrow1.17\beta.
  \]

Simulation design:

- Parameters tested one at a time:
  \[
  \eta,\ \beta,\ X_c,\ \epsilon.
  \]
- The focal parameter is drawn from a positive Gaussian distribution with:
  \[
  \mathrm{CV}=20\%.
  \]
- Other parameters remain fixed at the legacy baseline.
- Current default simulation size:
  \[
  N=1,000,000,\quad t_{\max}=300,\quad dt=0.025.
  \]

Lifespan strata:

- Death times are binned into 10-year intervals:
  \[
  40-50,\ 50-60,\ldots,\ 150-160.
  \]
- For each interval with at least 25 deaths, the focal parameter distribution is plotted as a Gaussian KDE.
- KDE input is subsampled to at most 60,000 values per interval.
- Distributions are normalized by the simulated mean focal-parameter value.

Middle-column mean curves:

- For each lifespan interval, the script plots the mean focal parameter value against interval midpoint.
- Display fits are:
  - \(\eta\): inverse form \(a+b/t\), fitted over ages 80-160.
  - \(\beta\): linear fit over ages 90-110.
  - \(X_c\): exponential fit \(ae^{bt}\), fitted over ages 40-120.
  - \(\epsilon\): linear fit over ages 40-120.

Right-column mortality curves:

- Age-specific mortality is estimated from simulated death times using 1-year bins from age 20 to 122.
- Hazards are deaths divided by number alive at the start of each bin.
- Hazards are clipped below \(10^{-8}\) and smoothed with a Gaussian kernel with \(\sigma=2.2\) bins.

## Supplement: Artificial Survival-Time and Disease-Threshold Panels

Primary scripts:

- `src/exploration/artificial_survival_time/make_artificial_survival_time_exploration.py`
- `src/exploration/artificial_survival_time/make_threshold_schematic.py`
- `src/figures/Supp_Figgs/make_supp_artificial_survival_composite.py`

Outputs:

- `Figures_new/Supp_Figgs/supp_artificial_survival_time_xdisease_075.png`
- `Figures_new/Supp_Figgs/supp_artificial_survival_composite.png`
- `Figures_new/Supp_Figgs/supp_artificial_survival_composite.pdf`

Source tables and caches:

- `saved_results/cache/simulations/artificial_survival_time/matched_sweden2019_event_times.npz`
- `saved_results/cache/simulations/artificial_survival_time/matched_sweden2019_metadata.json`
- `saved_results/csv/artificial_survival_time_summary.csv`
- `saved_results/csv/artificial_survival_time_state_composition.csv`

Baseline:

The SR baseline is the Sweden 2019 fit:

\[
\eta=0.5868368258,\quad
\beta=57.8717377207,\quad
\kappa=0.5,\quad
\epsilon=49.7186593,\quad
X_c=21.7405634,\quad
\sigma_{X_c}/X_c=0.1414213562.
\]

Disease threshold:

\[
X_D=0.75X_c.
\]

Scenarios:

- Baseline: \(X_c\) and \(X_D\) unchanged.
- Death-threshold-only increase: \(X_c\) multiplied by 1.2 while \(X_D\) remains unchanged.
- Proportional increase: both \(X_c\) and \(X_D\) multiplied by 1.2.

Simulation settings:

\[
N=80,000,\quad t_{\max}=160,\quad dt=0.05,\quad h_{ext}=0.
\]

The same sampled baseline \(X_c\) array is reused across scenarios before applying scenario factors. \(X_c\) is sampled from a positive Gaussian with the Sweden 2019 fractional heterogeneity. The disease threshold is computed per individual from the baseline \(X_c\) values and the scenario-specific \(X_D\) factor.

Events extracted:

- Death time.
- Disease-threshold crossing time.
- Healthspan: time before disease-threshold crossing, censored by death or \(t_{\max}\).
- Sickspan: time between disease-threshold crossing and death or \(t_{\max}\).
- Became-sick indicator.
- Alive-at-\(t_{\max}\) indicator.

Summary outputs include mean and median lifespan, healthspan, sickspan, sickspan among sick individuals, sick-life fraction, deaths observed, and censoring counts.

State-composition panels:

- Age grid:
  \[
  0,2,\ldots,140.
  \]
- At each age, the cohort is classified as healthy alive, sick alive, or dead.

The composite figure combines the schematic threshold-row, state-composition row, and median sickspan percentage of lifespan.

## Supplement: Model-Comparison Figure

Source script:

`src/figures/Supp_Figgs/make_supp_model_comparison.py`

Outputs:

- `Figures_new/Supp_Figgs/supp_model_comparison.png`
- `Figures_new/Supp_Figgs/supp_model_comparison.pdf`

Source tables:

- `saved_results/csv/supp_model_comparison_max_lifespan.csv`
- `saved_results/csv/supp_model_comparison_shape_response.csv`

Input caches:

- `saved_results/gamma_factor_sweep.pkl`
- `saved_results/fedichev_model_steepness_longevity_data.pkl`
- `src/notebooks/fedichev_gompertz_models.ipynb`

### Gompertz-Makeham Extreme-Lifespan Panel

The baseline Gompertz-Makeham model is fitted to Denmark 1880 male cohort HMD mortality:

\[
\text{fit ages}=20\ \text{to}\ 105.
\]

After fitting, the script sets:

\[
c=100,\quad m=0.
\]

For maximum-lifespan heterogeneity, the script samples death times analytically from the \(m=0\) Gompertz limit:

\[
T=\frac{\log\left(1+\frac{b[-\log(U)]}{a}\right)}{b},
\quad U\sim \mathrm{Uniform}(0,1).
\]

Simulation settings:

\[
N=300,000.
\]

The plotted extreme-lifespan metric is the age at which simulated survival reaches:

\[
S(t)=0.0001.
\]

With \(N=300{,}000\), this corresponds to the 30th-oldest simulated lifespan.

Heterogeneity CV values:

\[
0,2.5,5,\ldots,20\%.
\]

Gompertz heterogeneity cases:

- Intercept \(a\), sampled linearly from a positive normal distribution.
- Intercept \(a\), sampled as \(a^z\) with \(z\sim N(1,\mathrm{CV})\).
- Slope \(b\), sampled from a positive normal distribution.
- Coupled \(a+b\): sample \(b\), then set \(a\) by the relative \(b\) exponent.

Curves are smoothed for display with `gaussian_filter1d(..., sigma=0.85)`. The Gompertz-Makeham and Fedichev-Gruber extreme-lifespan panels are displayed with an upper y-axis limit of 150 years.

Display colors use the manuscript parameter-class palette: uncoupled Gompertz-Makeham \(a\) and \(b\) curves are blue/teal variants, Makeham \(m\) is red, and the coupled \(a+b\) curve is orange.

### Fedichev-Gruber Extreme-Lifespan Panel

Fedichev-Gruber heterogeneity data are regenerated directly in the figure script with the legacy notebook model equations:

\[
N=300,000,\quad dt=0.05,\quad t_{\max}=1000.
\]

The plotted metric is again the simulated \(S(t)=0.0001\) lifespan for CV values from 0 to 20%. The plotted parameters are:

\[
\beta',\ \epsilon_0,\ \gamma,\ \beta,\ g,\ D_0.
\]

Fedichev-Gruber senogenic parameters are drawn with distinct blue/teal colors from the manuscript senogenic palette, and the \(D_0\) noise curve is drawn in the robustness/noise amber color.

### Shape-Response Panels

Gompertz shape-response data are loaded from `saved_results/gamma_factor_sweep.pkl`. The baseline is `payload["results"]["baseline"][0]`, and all points are normalized as:

\[
x=\frac{t_{50}}{t_{50,\mathrm{baseline}}},
\quad
y=\frac{\text{IQR steepness}}{\text{IQR steepness}_{\mathrm{baseline}}}.
\]

Fedichev-Gruber shape-response data are loaded from `saved_results/fedichev_model_steepness_longevity_data.pkl`. Factor values are:

\[
0.6,0.7,\ldots,1.4.
\]

No fit-CI or model-sensitivity envelope is drawn in this supplement because comparable endpoint uncertainty data are not available for the legacy model calculations.

## Supplement: NHANES Exposure-Group Survival Curves

Source script:

`src/figures/Supp_Figgs/make_supp_fig4_nhanes_exposure_groups.py`

Output:

`Figures_new/Supp_Figgs/supp_fig4_nhanes_exposure_groups.png`

Data source:

NHANES source tables under:

`saved_data/nhanes/`

The script uses `ageing_packages.hetero_analysis.nhanes_analysis` to load the NHANES core table and each exposure topic, applying the same topic grouping strategies as the original notebook/helper code.

Kaplan-Meier fitting:

- Uses `lifelines.KaplanMeierFitter`.
- Uses delayed entry with `entry=entry_age`.
- Durations are `exit_age`.
- Events are `event`.
- Timeline:
  \[
  0\ \text{to}\ 120\ \text{years},\quad 721\ \text{points}.
  \]

The all-participant NHANES curve is fit after dropping rows missing `entry_age`, `exit_age`, or `event`. Each exposure group is fit after dropping rows missing the group label and survival columns.

Panels:

- Diet quality.
- Income-poverty ratio.
- Number of friends.
- Sleep duration.
- Physical activity.
- Alcohol consumption.
- Sleep frailty index.
- Church attendance.
- Education level.

The shaded bands are the Kaplan-Meier confidence intervals from `lifelines`. They are survival-curve confidence intervals for the grouped NHANES data, not SR-model endpoint envelopes.
