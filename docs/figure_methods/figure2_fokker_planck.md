# Figure 2: finite-volume SR simulation

The quantitative workflow uses the production SR forward solver. Saved source
tables are bundled, so normal rendering performs no simulations or fitting.
The complete figure is `Figures/Figure2/Fig2.png`. Run from the repository root:

```bash
python3 analysis/figures/figure2/plot_fig2_fp.py --data-dir results/tables/fig2_fokker_planck --output-dir Figures/Figure2
```

The source tables contain 120 finite-volume calculations. The baseline and
covariance are recorded in `results/fits/records/sweden_2019_fig2_fp_baseline.json`.
Figure 2 uses approximate log-delta bands without the per-year dispersion
adjustment used in Figures 4d and 5. Its covariance is therefore distinct from
`results/historical/joint_covariance.json`. The manifest records source hashes
for the calculation-time scripts.

Full optional recomputation is explicit and writes new tables under `tmp/`:

```bash
python3 analysis/figures/figure2/run_fig2_fp.py --recompute --workers 1
```

Full recomputation evaluates 120 numerical tasks and is substantially more
expensive than rendering the saved tables.

## Baseline and design

The frozen baseline in results/fits/records/sweden_2019_fig2_fp_baseline.json is
the Sweden 2019 equal-age fit used for Figures 4–5:

| Parameter | Value | Role in these simulations |
| --- | ---: | --- |
| η | 0.59 | Fixed mean |
| β | 57.9 | Fixed mean |
| κ | 0.5 | Fixed |
| Xc | 16.7708899457 | Fitted mean |
| ε | 30.1778456492 | Fitted mean |
| CV(Xc) | 0.2317422282 | Replaced by the focal CV below |
| mex | 0.0002152846 yr⁻¹ | Set to zero in these controlled experiments |

The baseline fit used the relative-error estimating criterion
Q = sum(D/mu + log(mu)); this is not a sampling likelihood. The full covariance
source and its hash are preserved alongside the frozen parameter record.

For heterogeneity curves, one parameter at a time (Xc, ε, η or β) follows a
positive-truncated Gaussian distribution. CV is SD/mean before truncation. The
focal spread replaces the fitted Xc spread; all other parameter CVs and mex are
zero. Mean-shift curves are homogeneous and scale one mean by factors from 0.85
to 1.15. Extreme lifespan is the age at which unconditional survival reaches
10⁻⁴. Conditional survival curves start at age 90.

Panel b uses 20% CV for Xc and epsilon and 5% CV for eta and beta.
The black curve is Sweden 2019 period survival, conditioned at age 90.
Panel a shows the heterogeneity sweep. Homogeneous mean shifts are shown
separately in [Supplementary Figure 1](parameter_shifts.md).
Panel d uses CVs of 20%, 30%, 15% and 10% for
Xc, epsilon, eta and beta. Filled/open markers denote siblings of short-/long-lived
probands, matching the empirical marker convention in panel c. The full-cohort
curve is retained in the source table but is not displayed.

Sibling pairs have Gaussian parameter correlation 0.5 before positivity
repair. A nonpositive parameter is independently redrawn from its positive
marginal. Conditional on the parameter
values, damage trajectories evolve independently. Deterministic joint
quadrature weights a sibling by the proband's probability of belonging to the
longest-lived 1% or shortest-lived 10%. Annual mortality is boundary-flux deaths
divided by trapezoidal person-time.

## Solver and uncertainty

The solver in src/senogenic_vs_robustness/sr_finite_volume.py is conservative
and nonuniform: it uses exponentially fitted face flux, implicit Euler time
stepping, a reflecting boundary at zero damage, and absorption at the actual
threshold Xc. Initial probability mass is placed in the first cell. Population
survival, deaths and exposures are mixed over quadrature nodes before
conditional survival or mortality is calculated, so survivor selection updates
the parameter distribution.

The default calculation uses 320 cells, dt=0.025 years, 64 quadrature nodes and
a 420-year horizon for the extreme tail. Gaussian integration is truncated at
±9 SD. The runner raises an error if survival has not reached 10⁻⁴ by the
horizon. Task predictions are cached under tmp/; source tables and a manifest
with relative paths and SHA-256 hashes are written to
`tmp/fig2_fokker_planck_recomputed/` by default. The archived source tables
remain unchanged unless an explicit alternate output directory is supplied.

Shaded model bands are approximate pointwise 95% log-delta intervals from the
joint covariance of log Xc and log ε in the fitted baseline. The covariance is
retained and predictions are differentiated centrally in log-parameter space.
The imposed heterogeneity and other parameters are held fixed. These are not
bootstrap or simultaneous intervals. The empirical sibling-panel band is the
existing OLS mean-fit confidence interval and does not include digitization
uncertainty.

An independent refinement to 480 cells, dt=1/60 year and 96 nodes shifted
tested age 90–110 survival probabilities by less than 0.0002 and annual
mortality rates by less than 0.7%. The broad η-heterogeneity case shifted the
10⁻⁴ tail age by 0.88 years (0.28%) near 311 years. The numerical refinement
record is `results/tables/fig2_fokker_planck_grid_check.json`.
