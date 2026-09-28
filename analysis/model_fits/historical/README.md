# Optional historical fitting, dispersion, and recovery

Normal figure reproduction reads saved tables. Nothing here is invoked by the
figure renderer. Heavy computation belongs on a suitably configured cluster.

The numerical model and optimizer are taken from the original historical
pipeline; `fitting.py`, `inference.py`, and the workers implement the Q
estimating criterion and shared-baseline sandwich covariance. The older
inverse-observed-death weighted objective is not an alternative default here.
Remaining weighted-deviance fields in fit records are diagnostic quantities.

## Inspect saved recovery results without refitting

```bash
python3 analysis/model_fits/historical/prepare_rerun.py --from-saved
cd tmp/historical_rerun
python3 summarize_recovery.py
```

Preparation copies code and records into an isolated directory. The summary
reads all 100 stored recovery datasets and writes aggregate tables there; no
solver calls or optimization are needed. Existing output directories are
rejected so a prior run cannot be overwritten inadvertently.

`results/historical/fit_records/` includes the baseline multistart audit,
45 historical point fits, 61 propagated-band calculations, 100 recovery
datasets (each with its baseline and 12 historical fits), contour checks,
numerical refinement checks, and the original job manifest. Its recorded
source hashes describe the original cluster code, not these path-adapted files.
The original cluster path in the manifest is provenance, not a required path.

## Explicit full recomputation

The intervals use per-year Pearson max(1, phi) scaling. Figure rendering reads
the saved tables; the commands below recompute the analysis.

Prepare a separate empty directory without `--from-saved`. Run each stage only
after the preceding stage finishes successfully. Commands below are worker
examples, not jobs automatically submitted by the repository.

```bash
python3 analysis/model_fits/historical/prepare_rerun.py --output tmp/history_fresh
cd tmp/history_fresh
python3 fitting.py --years 1900 1980 2019
LSB_JOBINDEX=1 python3 point_year.py
```

The point worker uses 1-based indices 1--45: 1800--2015 every five years,
then 2019. Run all indices, each once, with one CPU per task. `fitting.py`
first refits the baseline from the archived starting reference and audits three
representative years. The full-year worker uses the new fitted baseline.

After all point fits complete:

```bash
python3 bands.py --prepare
python3 bands.py --index 0
python3 joint_run.py --rep 1 --years 1800 1850 1900 1950 1980 1985 1990 1995 2000 2005 2010 2015
```

Run band indices 0--60. Run recovery replicates 1--100 with the **explicit year
list above** to match the archived production run; the original worker's
shorter default year list does not describe production. Each recovery draw
uses `SeedSequence([2026092701, replicate, year])`, refits its own baseline,
and computes its own joint sandwich covariance. After all bands and recovery
fits finish, run `LSB_JOBINDEX=1 python3 calibrate_contours.py` for indices
1--100, then `python3 summarize_recovery.py`.
Despite its historical filename, `calibrate_contours.py` checks the original
Poisson recovery procedure only; its output is not a calibration of the
current dispersion-adjusted bands. Recovery is optional and is not needed
to reproduce the current intervals.

The original WEXAC manifest records one CPU per task: baseline 30 minutes/
3000 MB, recovery 60 minutes per task, bands 10 minutes per task, and contour
checks 10 minutes/1000 MB per task. Production concurrency was 100 recovery,
16 bands, and 40 contour checks. These are historical allocations, not local
runtime guarantees or a recommendation to submit that concurrency elsewhere.
Set numerical thread counts to one as the worker already does. Adapt scheduler
queues and resource requests to the destination cluster.

## Interpretation

Q is an estimating criterion, not a likelihood; it must not provide AIC.
Recovery draws check the original Poisson procedure, not the current adjusted
bands, empirical population uncertainty, or future trend validity. The old
split-sample correction is diagnostic only, stored in
`results/historical/diagnostics/poisson_recovery/historical_calibration.json`.
It is not read by any active renderer or production covariance worker.
Dispersion includes residual shape mismatch, does not correct bias, and its
own estimation uncertainty is omitted. Coverage under misspecification is
not established. See `docs/figure_methods/historical_fits.md`.
## Denmark

The saved-data staging also includes all 38 Danish fits. From that directory,
run `python3 denmark_aggregate.py` to rebuild the joint covariance and panel
c/d source tables without refitting. For optional new fits, first complete the
Swedish baseline stage, then run `python3 denmark_fit.py 0` for each zero-based
index 0--37 (1835--2015 every five years, then 2019). The equivalent cluster
worker accepts 1-based `LSB_JOBINDEX` when no positional index is given.
Complete Swedish `bands.py --prepare` before Danish aggregation, which needs
the Swedish baseline covariance.

The Danish analysis carries Swedish epsilon/CV uncertainty and its covariance
with the Swedish denominator into all year estimates. No Danish recovery
calibration was performed. Saved fit records document the mex=0
boundary years and representative-year multistart checks.

## Recompute the current intervals without refitting

Stage with `--from-saved` into a new directory and change to that directory.
`bands.py --prepare` now computes per-year Pearson dispersion with k=4 for
the reference and k=2 for historical fits, scaling each independent variance
block before joint inversion. It writes adjusted covariance and dispersion.
It evaluates numerical derivatives and is **not** a saved-only operation.
Then run `bands.py --index N` for every N from 0 through 60. These workers
propagate covariance through the original SR solver and also compute expensive
derivatives. Limit concurrency to at most four single-thread workers.
No fitting or new simulations are required for these stages.

After all 61 targets finish:

```bash
python3 denmark_aggregate.py
python3 export_tables.py --input . --output tables
```

Danish aggregation uses saved Jacobians and expected counts, with its own
max(1, phi) adjustment and the updated Swedish covariance. It performs no
solver calls. It writes `denmark/joint_covariance.json`,
`denmark/dispersion.json`, and `denmark/figures/Fig4d_values.csv`.
Promote those to the corresponding `results/historical/denmark_*` files only
after checking unchanged centers. Promote the Swedish covariance, dispersion,
all `bands/*.json` into `fit_records/bands/`, and the two exported tables.
The MGG panel-c table is calculated separately.

To regenerate Swedish tables from the shipped archive only (no derivatives):

```bash
python3 analysis/model_fits/historical/export_tables.py --input results/historical --output tmp/historical_tables
python3 analysis/model_fits/historical/test_dispersion.py
```

Table row order may differ; compare by series/scenario, year, and metric label.
All current ratio and lifespan intervals use normal critical value 1.959964.
No `exec` or runtime source-string rewriting is used in the ported workers.
