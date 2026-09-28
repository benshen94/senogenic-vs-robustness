# Optional HGPS refitting

Normal figure reproduction reads the archived fits; it never runs this script.
To explicitly rerun one model from the repository root:

```bash
python3 analysis/model_fits/hgps/refit.py 1
```

Indices 1--10 follow `MODELS` in `src/senogenic_vs_robustness/hgps_likelihood.py`.
New output is written under ignored `tmp/progeria_refit/`; the archived fits
are preserved. Starting values are the saved earlier fits converted to the Q
reference, with the same additional multistart points as the original job.

The original WEXAC run was array job 852212, queue `medium`, ten concurrent
single-CPU tasks, 1500 MB per task, and a 30-minute wall limit. The recorded
manifest is `results/progeria/cluster_manifest.json`. The following is a
portable submission example using those resources, not the original shell file:

```bash
mkdir -p tmp/progeria_refit/logs
bsub -q medium -J 'hgps[1-10]%10' -n 1 -M 1500 -R 'rusage[mem=1500]' \
  -W 00:30 -oo 'tmp/progeria_refit/logs/%J_%I.out' \
  -eo 'tmp/progeria_refit/logs/%J_%I.err' \
  'env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 analysis/model_fits/hgps/refit.py'
```

Use a Python environment with the repository requirements installed before
submitting. No bootstrap was used for HGPS, and baseline uncertainty is not
propagated. NHANES resampling and historical recovery experiments are separate
analyses and must not be attributed to this fit.
