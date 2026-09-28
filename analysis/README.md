# Analysis scripts

This folder separates saved-result figure rendering from optional numerical
recomputation. Run the commands below from the repository root after following
the [setup instructions](../README.md#setup).

- `figures/`: main and supplementary figure-generation scripts.
- `model_fits/`: optional FP fitting, bootstrap and numerical-check workers,
  plus separate Fedichev-Gruber calculations. Each workflow has its own guide.
- NHANES data preparation lives in repository-root `scripts/prepare_nhanes_cohort.py` and
  `scripts/prepare_nhanes_survival.py`.
- `quality_checks/`: the retained threshold-schematic drawing helper used by
  Supplementary Fig. 4.

Most users should start from the repository root with:

```bash
python3 scripts/verify_repo.py
python3 scripts/reproduce_figures.py --set current
```

These commands use saved results and do not fit models, simulate trajectories,
bootstrap datasets or submit cluster jobs. `current` is the only supported
figure set. See the [output index](../results/index/outputs.csv) for exact
renderer, input and output paths, and the [manuscript map](../README.md#manuscript-map)
for figure numbering.

Most renderers write publication PNGs under `Figures/`. Fig. 1 exports its
quantitative panel under `tmp/figure1/`, separately from the schematic composite.
Figure 2 and all four supplementary figures are rendered as complete composites.

The current Supplementary Fig. 2 renderer is
`analysis/figures/supplementary/render_nhanes_likelihood.py`, using saved
cleaned-cohort survival curves. The `figures/steepness_longevity/` modules
provide shared plotting styles.

Historical SR fitting is documented in `model_fits/historical/`. Follow the
workflow-specific guides linked from the top-level README for recomputation
commands and numerical checks.
