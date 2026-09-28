# Numerical validation

## Fokker-Planck and Monte Carlo survival

The finite-volume Fokker-Planck (FP) solver was compared with SR Monte Carlo
simulations of 20,000 individuals at five parameter sets. The SR drift is
$\eta t-\beta x/(\kappa+x)$, with diffusion coefficient $\epsilon$, reflection
at zero damage and absorption at $X_c$. All comparisons use $\kappa=0.5$.
Thresholds follow a positive-truncated Gaussian restricted to +/-8 standard
deviations. The same threshold distribution and constant external mortality
apply to both methods. Initial damage is $10^{-10}$ in Monte Carlo; FP places
initial mass in the first spatial cell.

Monte Carlo simulations used `SR_sim`, including its Brownian-bridge crossing
correction, with seed 2026092201. FP grids used 160 cells, timestep 0.05 years
and 96 threshold nodes, then 320 cells, timestep 0.025 years and 192 nodes.
The table compares the finer FP solution with each case's finest Monte Carlo
timestep at integer ages 0--110 years.

| Parameter set | MC timestep (years) | Maximum survival difference (percentage points) |
| --- | ---: | ---: |
| Original paper Sweden | 0.0015625 | 0.789 |
| Independently fitted Sweden 2019 | 0.0015625 | 0.568 |
| NHANES | 0.0015625 | 0.768 |
| Sweden 1950, Xc + mex | 0.000390625 | 0.707 |
| Sweden 1950, Xc + eta + mex | 0.0015625 | 0.724 |

The individual 95% Dvoretzky-Kiefer-Wolfowitz sampling half-width is 0.960
percentage points. It bounds Monte Carlo sampling error conditional on the
simulation scheme, not discretization bias or simultaneous error across all
five parameter sets. At timestep 0.025 years, the Sweden 2019 discrepancy was
13.703 percentage points. The Sweden 1950 Xc + mex discrepancy was 1.044
points at timestep 0.0015625 before decreasing to 0.707 with further refinement.
These comparisons therefore support agreement at the tested resolutions,
not interchangeability of numerical schemes at arbitrary timesteps.

`results/validation/records/` contains nine simulation records, including
coarser steps, with parameter values, simulated death times, survival curves,
seeds and original-source hashes. Descriptive source labels identify the
calculation inputs; original-source hashes describe the calculation-time
files, not a mapping to current repository files. `results/validation/replay_current.json` reports exact
agreement between recalculated and saved FP survival for the original paper
Sweden case. The replay reconstructs Monte Carlo survival from saved death
times; it does not generate new stochastic trajectories. Curve agreement does
not establish the accuracy of small likelihood or AIC differences.

## Threshold and parameter quadrature

The SI quadrature comparison changes Gauss-Legendre order from 160 to 256
while fixing 480 spatial cells, timestep 1/60 years and horizon 260 years.
It uses the SI baseline in `results/supplementary1_fp/si_checks.json`, with
external mortality set to zero. Heterogeneity is applied separately to Xc
(CV=0.2317422282), eta (CV=0.2), and beta (CV=0.2). Each distribution is a
positive-truncated Gaussian restricted to +/-9 standard deviations; CV denotes
the standard deviation divided by the mean before truncation.

Population survival mixes component survival probabilities. Annual mortality
is mixed boundary-flux deaths divided by mixed trapezoidal person-time, not
the birth-weighted average of component hazards. Comparisons cover survival
through age 260 and annual mortality intervals beginning at ages 20--254.
This isolates quadrature error; spatial and temporal discretization errors
require separate refinement checks.

The complete comparison is saved in
`results/validation/quadrature_full.json`:

| Heterogeneous parameter | Maximum absolute survival change | Maximum relative annual mortality change |
| --- | ---: | ---: |
| Xc | 4.22e-15 | 6.17e-12 |
| eta | 9.77e-11 | 4.60e-6 |
| beta | 3.22e-15 | 9.21e-14 |

Thus increasing quadrature order from 160 to 256 changed survival by less
than 1e-10 and annual mortality by less than 4.7e-6 relative (0.00047%) in
these three scenarios. Differences near machine precision should be interpreted
as numerical agreement, not a precision guarantee for the physical model.

## Reproduction

Run from the repository root:

```bash
python3 -m unittest discover -s tests -p test_solver_validation.py
python3 analysis/validation/validate.py audit --output tmp/validation_audit.json
python3 analysis/validation/validate.py replay --output tmp/validation_replay.json
python3 analysis/validation/validate.py quadrature --output tmp/quadrature_full.json
```

The audit checks array hashes, reconstructs survival from death times and
recalculates discrepancies and sampling bounds. Result files include Python,
NumPy and SciPy versions and code hashes. Existing output files are not
overwritten. Adding `--smoke` to the quadrature command selects 40 cells,
timestep 0.1 years and horizon 100 years for a shorter implementation check.
