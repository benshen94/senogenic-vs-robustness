# Supplementary Figure 5: joint first passage

The joint FP calculation, saved-table renderer, mathematical tests and
three-grid numerical check are included. The adopted 480-cell results are in
`results/supplementary4_fp/`.

## Preserve the experiment

The three scenarios remain baseline, death threshold multiplied by 1.2 alone,
and disease/death thresholds both multiplied by 1.2. Each individual's baseline
disease threshold is 0.75 of their baseline death threshold. The same Gaussian
threshold factor multiplies both: they are not sampled independently.

Use the current Sweden Q-reference intrinsic parameters and threshold CV from
`results/historical/joint_covariance.json`. As in the original controlled SR
experiment, external mortality is zero. Follow-up ends at 160 years. Disease
means the first threshold crossing, not current damage above threshold:
subsequent downward fluctuations do not restore the healthy label.

The statistic is the population median of each individual's
`(end_age - onset_age) / end_age`, with zero for those never sick. Death age
defines end age for deaths, and the horizon defines it for survivors. This
matches the original finite-horizon convention. Report the residual alive
mass explicitly; the statistic is not a median conditional on observed death.
It is neither a ratio of medians nor identifiable from two marginal survival
curves alone.

## Numerical construction

`src/senogenic_vs_robustness/sr_joint_passage.py` uses the same additive-noise
exponentially fitted face rates, reflecting origin, absorbing death boundary,
and backward-Euler stepping as the production SR finite-volume solver.
It inserts a cell face at the disease threshold. Initially all probability is
healthy in the first cell. Upward probability flux across the disease face
enters an irreversibly tagged sick population on the full death domain.
Sick probability can move below the disease threshold without losing its tag.

Separate columns retain onset timesteps. Their absorbing-boundary death flux
gives the joint onset/death probability and hence the individual ratio
distribution. The linear-system factorization is shared across columns.
Adding healthy and all sick columns gives the ordinary full-domain FP state;
the implementation independently checks that marginal identity on the same
mesh and checks total healthy/sick/dead probability at every age.

The onset transition is a finite-volume lattice approximation: probability
enters the adjacent cell rather than a continuous point exactly at the disease
boundary. Its spatial error must be checked by mesh refinement. Onset and death
ages are assigned to timestep endpoints; their temporal error must be checked
by decreasing the timestep. The ratio histogram defaults to 2,001 points on
[0,1], so rounding contributes at most 0.00025 to each ratio. Median uncertainty
from discretization is not a biological confidence interval.

Small tests compare tridiagonal stepping with a dense linear solve, check
tagged/ordinary marginal equality and probability conservation, and confirm
refinement toward the analytic survival curve for pure diffusion with a
reflecting origin and absorbing boundary. These do not establish convergence
for the fitted manuscript parameters by themselves. The saved fitted-parameter
check jointly refines cells/timestep/quadrature from 160/0.05/32 through
320/0.025/64 to 480/(1/60)/96. The last refinement changes medians by 0.05
percentage points and state probabilities by at most 0.001725. The finest-grid
medians are 0.1035, 0.1635 and 0.0725 for baseline, death-threshold-only and
proportional shifts. Surviving mass at 160 is below 8e-110. This is numerical
stability evidence, not a formal error bound or biological uncertainty.

```bash
python3 -m unittest discover -s tests -p test_joint_passage.py -v
```

## Optional computation

The tagged calculation is quadratic in timestep count, unlike a marginal
survival solve. Do not run it as a routine figure-rendering dependency.
`analysis/model_fits/supplementary/healthspan.py` splits the three scenarios
and threshold quadrature nodes into independent tasks. The adopted grid uses
96 nodes per scenario (288 tasks); defaults provide the intermediate 64-node
grid (192 tasks). Each uses one process; scheduler walltime must be determined
by a measured pilot. No cluster
jobs for this new implementation have been submitted or claimed as provenance.

Example of a single explicitly requested task:

```bash
python3 analysis/model_fits/supplementary/healthspan.py node \
  --scenario baseline --node 0 --nodes 96 --cells 480 \
  --dt 0.016666666666666666 --output tmp/supplementary4_refined --recompute
```

Repeat for zero-based node indices 0--95 and each of `baseline`, `xc_only`,
`proportional`. The worker refuses to replace an existing node record. All
outputs default to `tmp/supplementary4_fp/`, separate from archived results.
Then aggregate without any new solver calls:

```bash
python3 analysis/model_fits/supplementary/healthspan.py aggregate \
  --nodes 96 --cells 480 --dt 0.016666666666666666 \
  --output tmp/supplementary4_refined
```

This writes probability-weighted state curves, ratio distributions, medians,
residual alive mass and a manifest. Nodes must have identical baseline,
solver-source checksum and grid configuration. Use separate output directories
for cell, timestep and quadrature refinements; pass the same settings to
aggregation. The saved comparison script
`analysis/model_fits/supplementary/compare_healthspan_grids.py` compares any two
compatible aggregated grids without solver calls. Numerical checks were run
locally, not on WEXAC; see the saved-results README for provenance.

## Render saved joint results

```bash
python3 analysis/figures/supplementary/render_healthspan_morbidity.py
```

The renderer preserves the three-row schematic/state/distribution layout and colors.
It reads tables only, checks probability mass and recomputes each median from
the joint ratio distribution before plotting. Missing files fail explicitly;
the renderer never launches simulations or falls back to the trajectory cache.
Its three integration tests use clearly synthetic fixtures in temporary
directories, not manuscript curves. They check PNG output, missing-input
failure and rejection of a summary median inconsistent with the distribution.

```bash
python3 -m unittest discover -s tests -p test_healthspan_renderer.py -v
```

Panel c displays the median, interquartile range and 5th–95th percentile
range from the joint-FP sick-life-fraction probability distribution. These
are between-individual model distribution intervals, not sampling or parameter
confidence intervals. All plotted endpoints change by at most 0.05 percentage
points between the saved intermediate and finest grids.
