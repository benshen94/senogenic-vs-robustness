# NHANES likelihood and paired bootstrap

## Rebuild summaries from saved fits

From the repository root:

```bash
python3 analysis/model_fits/nhanes/aggregate_precision_bootstrap.py
```

This audits 1,150 saved replicate/group records and recalculates the tables in
`results/nhanes/`. It does not optimize parameters or simulate a population.
There are 50 paired group resamples, 23 groups and eight models (9,200 fits).
The separate 100 baseline refits are included for Figure 3 uncertainty.

## Scientific specification

The cleaned cohort contains 55,800 records and 7,260 deaths. Fits use individual
death/censoring ages with delayed entry. The full-cohort reference fixes eta,
beta and kappa and estimates Xc, epsilon, threshold CV and nonnegative external
mortality. Each group retains that reference's CV and inactive parameters.

The eight comparisons are external mortality alone; Xc or epsilon plus external
mortality; and five two-intrinsic-parameter alternatives plus external mortality.
AIC counts external mortality even when its estimate is zero. The point-fit
likelihood is unweighted. Paired Rao-Wu PSU resampling within wave/stratum cells
measures stability; it does not make the point estimates nationally representative.

Point-fit JSON files also retain the diagnostic `Xc+epsilon` model. It is not
one of the eight models compared in the paired bootstrap. The archive verifier
checks nine models per point record and eight per bootstrap record explicitly.

The random seed is 20260924. Replicate seeds use `SeedSequence([SEED, rep, 1])`.
The fit grid is 320 cells, 0.025-year steps and 31 threshold nodes, checked at
480 cells, 1/60-year steps and 41 nodes. The archived 640-cell sensitivity
summaries are also retained. Preserve these distinctions when interpreting
groups near the AIC threshold.

## Optional expensive reruns

`group_fit.py` is the initial group worker. `refine_high_grid.py` performs
320-cell refinement.
Its original SHA-256 is
`20f6c854b0e2cf1144017b85d4d1824a2d4c3c8c5291dea011ea956f3c1e2afc`.
The optimization and result-writing functions are unchanged; the command line
now requires an explicit group/index. The 480-cell evaluation is a numerical
check, not a separate optimization. Paired bootstrap refinement is in
`bootstrap_groups.py`.

The initial group worker requires an explicit group or a valid one-based task
index. Running it with no arguments fails without fitting; index zero cannot
silently select the last group.

The source fitting and resampling scripts are included here. Stage a separate
rerun directory using the cleaned input matching the recorded checksum:

```bash
python3 scripts/prepare_nhanes_cohort.py
python3 analysis/model_fits/nhanes/prepare_rerun.py --cohort tmp/nhanes_cohort.csv
```

The preparation script reconstructs the exact archived cohort from the bundled
public-use inputs; see [data preparation](../../../data/nhanes/README.md).
It runs no fitting. The staging command checks
the original input hash and creates `tmp/nhanes_refit/`. It never replaces the
archived fits. The source job used Anaconda3/2024.06-1 and set OMP, OpenBLAS,
MKL and Numba thread counts to one.

By default staging copies saved high-grid point fits for bootstrap starts.
To rebuild point fits instead, create a separate directory:

```bash
python3 analysis/model_fits/nhanes/prepare_rerun.py \
  --cohort tmp/nhanes_cohort.csv --fresh-point-fits --output tmp/nhanes_point_refit
```

From that directory run `python3 group_fit.py --index N`, then
`python3 refine_high_grid.py --index N`, for one-based indices 1--23. These are
expensive fits, not verification commands. Initial fits write `results/`;
refinement reads those and writes `results_high/`. Existing final files are
not overwritten. For LSF arrays the same index can be supplied through
`LSB_JOBINDEX`. Do not copy archived `results_high` into this fresh-point
directory, since that intentionally causes the refinement worker to skip.

### Rebuild the full-cohort reference

The commands above intentionally use the archived reference in
`inputs/best_constrained_fit.json`. To recompute that reference as well, run
the following **inside a new staged fresh-point directory**, before any group
or bootstrap jobs:

```bash
python3 refit_fixed_eta_beta.py
```

This is an expensive multistart optimization, not a smoke test. It writes
`coarse_multistart.json`, `fit_grid_multistart.json`,
`check_grid_multistart.json` and `best_constrained_fit.json` at the staging
directory's top level. The final reference is selected after optimization on
the 480-cell grid; this differs from the group worker's 480-cell evaluation
without reoptimization. Inspect convergence and boundary estimates before
adopting a recomputed reference.
The bundled reference also records a subsequent direct-boundary-flux check
and a high-grid refinement. The multistart script alone does not reproduce
that record's exact JSON schema or establish byte-identical optimizer output;
compare parameter values, likelihood and flux diagnostics separately.

Group workers read `inputs/best_constrained_fit.json`, not that new top-level
file. To deliberately use the recomputed reference in the fresh run:

```bash
cp best_constrained_fit.json inputs/best_constrained_fit.json
```

Then run the initial and refined group workers and the bootstrap dependency
sequence below. Do not combine new-reference results with archived-reference
fits or bootstrap records. Keep the public saved results unchanged; numerical
optimizer differences should be reported rather than silently substituted.

In that staged directory, the computational dependency order is:

1. `bootstrap_baseline.py --rep N`: coarse reference fit for replicates 1--100.
2. `bootstrap_baseline_precision.py --rep N`: refine the matching coarse fit.
3. Run `bootstrap_groups.py` with the corresponding LSF task indices 1--1150,
   setting `NHANES_BOOTSTRAP_BASELINE_FOLDER=baseline_precision` and
   `NHANES_BOOTSTRAP_GROUP_FOLDER=groups_precision`. Retain the exact task-index
   mapping implemented by the script. This covers the first 50 paired draws.

The cluster manifest in `results/nhanes/precision_bootstrap_manifest.json`
records actual job IDs, input hashes, canceled arrays, restart history, and the
documented change to the high-grid multistart policy. The recorded remote run
was `sr_nhanes_paper_anchor_groups_20260924`; jobs 569432 and 600697 include the
baseline and replacement group arrays. Those IDs are historical provenance,
not commands to resubmit. Scheduler resource settings beyond those recorded
in the archive must be chosen for the target cluster.

The source scripts retain their original analysis-local input/output layout;
run expensive fits from the staged directory, not from this source directory.

## Targeted grid sensitivity

The primary archive includes an audited repair of replicate 18, `income__1`,
model `Xc+beta`. Its original fit hit the beta lower bound after one high-grid
start. `bootstrap/repairs/` retains the original record and multistart repair
report; `bootstrap/groups_precision/` contains the accepted repaired record.
The original and repaired records are provenance, not two bootstrap draws.
The optional `repair_precision_group_fit.py --recompute` operates only on
that specific pre-repair case in a staged rerun directory. Do not run it
against the accepted archive or assume every future boundary fit needs this
same repair. The archived report's source hash refers to the original worker,
before the explicit-command guard was added for this release.

The archive includes all 60 saved 640-cell reevaluations and their task list in
`results/nhanes/bootstrap/grid640_validation/`. Rebuild their summary without
solver calls from the repository root:

```bash
python3 analysis/model_fits/nhanes/audit_grid640.py
```

This verifies source parameters against the paired bootstrap records. Among
the 60 cases whose AIC classification differed between 320 and 480 cells,
35 agree with the 320-cell classification and 25 with the 480-cell one.
This targeted check does not replace or refit the primary analysis.

For optional new numerical evaluations, the staged `validate_grid640.py`
supports `--build-manifest`, then explicit `--index N` (1--60 for the archived
run). It evaluates saved intrinsic parameters at 640 cells, 1/120-year steps
and 51 threshold nodes, profiling external mortality again. It is not a full
intrinsic-parameter optimization. The historical manifest builder deliberately
requires 60 discordant cases; a new fit archive may differ and needs a reviewed
task-selection change, not forced substitution of the original task list.
The audit accepts `--results-dir` for a separate archive containing the paired
records and `bootstrap_precision_group_raw.csv` summary.
