# Additional calculations

Run from the repository root with the repository requirements installed.
Each runner requires `--run` and a new output directory. Imports do not launch
calculations. The default baseline is
`results/historical/fit_records/point/baseline.json`. Historical calculations
stage copies of the bundled solver and inputs without modifying saved fits.

## Lognormal heterogeneity

```sh
python3 analysis/model_fits/additional/lognormal.py --run --parameter Xc --width 0.2 --convention cv --output tmp/lognormal_xc
```

A focal parameter is assigned at birth as
`p_i = p_mean * exp(sigma*z - sigma^2/2)`, preserving its arithmetic mean.
Choose `--convention cv` for `sigma = sqrt(log(1+CV^2))`, or
`--convention log-sigma` for direct log-space sigma. The choice is required.
Gauss-Hermite quadrature averages component survival, not component hazards.
Independent constant mex contributes `exp(-mex*age)` to survival.

Other SR parameters are homogeneous: the baseline Gaussian Xc CV is **replaced**,
not retained, even when another focal parameter is selected. This is a
single-parameter distribution calculation, not a multi-parameter heterogeneity
fit. `survival.csv` contains unconditional log survival; `contract.json` records
the parameters, width convention and numerical settings. Check node, cell and
time-step convergence before interpreting extreme tails. No similarity to a
Gaussian model or uncertainty interval is assumed.

### Normal-versus-lognormal comparison

```sh
python3 analysis/model_fits/additional/compare_lognormal.py --run --nodes 64 --output tmp/distribution_comparison
```

This follows the Figure 2 controlled experiment: the Figure 2 saved baseline,
mex=0, one focal heterogeneous parameter, and all other parameters homogeneous.
The cases are eta/beta CV=0.05, Xc/epsilon CV=0.20, plus the plotted Xc CV=0.15
and epsilon CV=0.25. The positive-normal distribution uses the specified mean
and CV before truncation at zero; the lognormal matches the specified arithmetic
mean and CV exactly. The small normal truncation-induced moment shift is not
rescaled. Normal integration uses Gauss-Legendre quadrature; lognormal integration
uses Gauss-Hermite quadrature.

The output includes median, quartile and top-0.01% attained ages, restricted mean,
maximum absolute unconditional survival difference, and survival at 100, 110 and
120 conditional on age 90. It records both models and signed differences rather
than applying an arbitrary similarity threshold. No fitted-parameter uncertainty
is propagated. Repeating with `--nodes 128` checks quadrature sensitivity.

The saved `results/additional/lognormal_comparison_64/` records the six-case calculation
at 320 cells, dt=0.025 and horizon 420 years. Signed lognormal-minus-normal
top-0.01% age differences are -1.0282 years (eta CV=0.05), +0.8093 (beta 0.05),
+0.6333 (Xc 0.20), -0.0298 (epsilon 0.20), +0.3504 (Xc 0.15), and -0.0236
(epsilon 0.25). Absolute median differences are at most 0.259 years. The largest
absolute difference between unconditional survival probabilities is 0.01223.
Thus the tested cases retain the broad senogenic-versus-robustness contrast,
but their extreme-tail ages are not identical. These representative cases do
not establish distributional equivalence across all CVs, baseline shifts,
sibling models or fitted populations.
The 128-node repeat is saved in `results/additional/lognormal_comparison_128/`; age metrics
change by less than 9e-10 years, with cells and time step unchanged. This checks
quadrature sensitivity only. Small absolute survival differences do not imply
small relative differences in the far tail: for Xc CV=0.20, survival to age 120
conditional on reaching 90 changes from 1.11e-11 to 1.14e-10. General claims of
identical or interchangeable survival distributions are therefore not warranted.

## Age-dependent extrinsic mortality

Supply a CSV with exactly `age,hazard` columns: age in years and instantaneous
extrinsic hazard in year^-1. It must contain at least two finite rows, strictly
increasing ages starting at zero, and nonnegative hazards. It must cover the
full requested horizon. There is no default hazard shape.

```sh
python3 analysis/model_fits/additional/age_dependent_mex.py --run --hazard-csv path/to/hazard.csv --match cumulative-hazard --match-start 20 --match-end 100 --entry-age 20 --horizon 140 --output tmp/age_dependent_mex
```

The CSV defines a piecewise-linear hazard between supplied knots. The script
integrates each segment exactly to obtain `H(t)`; it does not extrapolate.
The constant comparator matches cumulative hazard over the **explicit** interval
`[a,b]`: `m_constant = (H(b)-H(a))/(b-a)`. Thus extrinsic survival over that
interval matches, but median, mean and survival at intermediate ages need not.
The example interval above is an invocation example, not a fitted or universal
matching convention. Both its endpoints and the conditioning age are required.

The intrinsic model is shared between comparisons and retains baseline
positive-truncated Gaussian threshold heterogeneity. It uses the bundled
population solver with Gauss-Legendre quadrature (default 64 nodes, 320 cells,
dt=0.025 years). Baseline mex is **replaced**, not added:
`log S_age(t) = log S_intrinsic(t) - H(t)` and
`log S_constant(t) = log S_intrinsic(t) - m_constant*t`.

`comparison.csv` records the hazard, cumulative hazard and three unconditional
log-survival curves. `contract.json` includes the matched rate, interval, input
hash, settings, and conditional median/quartile ages and interquartile width.
Means are explicitly restricted to the requested horizon. Quantiles not reached
by that horizon are null. Conditioning divides each curve by its own survival
at the entry age. The input CSV is copied alongside the outputs. This compares
the supplied hazard only; it neither fits a hazard nor asserts general similarity.

## Historical epsilon instead of Xc

```sh
python3 analysis/model_fits/additional/historical_epsilon.py --run --year 1900 --output tmp/epsilon_1900
```

This fits epsilon and profiles constant mex, keeping eta, beta, Xc, Gaussian Xc
CV and kappa fixed at the saved Sweden baseline. It uses checksum-verified
bundled HMD deaths/exposures at ages 20-109 and the existing historical pipeline.
Annual model rates are expected deaths divided by expected person-time, with
direct SR boundary deaths and competing extrinsic deaths.

The objective is
`Q = sum(d/mu + log(mu/max(d,1)) - d/max(d,1))` over positive-exposure ages.
It is not ordinary Poisson likelihood; AIC is not defined from this score.
The existing three starts, bounds, convergence status, bound hits, start
disagreement and age diagnostics are retained in `epsilon_result.json`.
These are conditional point estimates, with no bootstrap or mechanism identification.
`--maxiter` bounds outer iterations per start, not total solver calls.
`--evaluate-baseline` evaluates one curve without fitting and records `fitted: false`.

## Xc required for linear mean lifespan

```sh
python3 analysis/model_fits/additional/required_xc.py --run --years 2100 --output tmp/required_xc
```

This inverts the SR mean for the saved linear mean-lifespan comparator in
`results/historical/naive_hmd_mean_projection.json.gz`: the annual 1980-2019
observed HMD age-20 conditional mean OLS slope, anchored at observed 2019.
The target is not reanchored at the model baseline mean, so an inverse factor
at 2019 need not be exactly one. All parameters except Xc, including mex and
threshold CV, remain fixed at the saved baseline, as in the existing projections.

The historical solver uses 320 cells, dt=0.025 years and 31 Gauss-Hermite threshold
nodes, discarding nonpositive thresholds and renormalizing weights. Its adaptive
horizons are 180, 270, 405 and 610 years. Each mean must satisfy the solver's
tail criterion and an unintegrated mean-tail bound below 1e-4 years.
Brent root finding uses an explicit Xc factor bracket, default 0.5-4, and fails
if a target is unbracketed or the mean is unresolved. Output files retain
settings, source hashes, evaluation diagnostics, targets and achieved means.

The saved calculation in `results/additional/required_xc_2100/` reaches a target mean of
96.9295682 years with Xc=30.3584071, a factor of 1.8101846 relative to the fixed
baseline. This endpoint calculation has no parameter-uncertainty interval and
does not establish super-exponential growth: a growth-shape claim requires a
trajectory, not one endpoint. No Xc growth law is imposed.

The saved `results/additional/required_xc_trajectory/` extends this calculation to 2019,
2040, 2060, 2080 and 2100. Its Xc factors are 0.999687, 1.128328, 1.287605,
1.503453 and 1.810185. Interval-average log growth rises from 0.005764 to
0.006602, 0.007749 and 0.009283 per year. This supports accelerating proportional
growth over those sampled intervals under the stated fixed-parameter model.
It does not provide uncertainty estimates or prove positive continuous-time
log curvature at every intermediate year.

```sh
python3 analysis/model_fits/additional/required_xc.py --run --years 2019 2040 2060 2080 2100 --reuse-from results/additional/required_xc_2100 --output tmp/required_xc_trajectory
```

Reuse requires matching baseline, numerical settings, solver/pipeline hashes and
target input hash. The output records which years came from matching saved results.

## Tests

```sh
python3 -m unittest discover -s analysis/model_fits/additional -p 'test_*.py'
```

Tests cover lognormal moments and zero-width solver parity, exact hazard
integration, matching and CSV validation, constant-hazard end-to-end parity,
inverse root finding and opt-in guards. Synthetic hazards in tests are numerical
fixtures, not empirical inputs or analysis defaults.
