# NHANES Extended Data Figs. 2 and 3, Table 1 and survival summaries


## Extended Data Fig. 3 and Table 1

```bash
python3 analysis/figures/extended_data/render_nhanes_aic.py
```

This reads the 23 high-resolution point-fit JSON files in
`results/nhanes/results_high/` and 1,150 group/replicate summaries in
`bootstrap_precision_group_raw.csv`. It regenerates the figure PNG and
`results/nhanes/extended_data_table1.csv` plus its readable JSON table.
It performs no fitting or bootstrapping.

For each group, compare the lower AIC of Xc+mex and epsilon+mex against the
lowest AIC among the five tested two-intrinsic-parameter alternatives (each
also includes mex). Competitive means the resulting AIC difference is at most
2. This comparison is reselected independently in each bootstrap replicate.
The full model set also includes mex-only; the saved audit checks agreement
with the corresponding all-eight-model classification. AIC counts fitted mex
even when its estimate is zero. These are likelihood-based AIC values, unlike
the historical Sweden Q objective.

Panel a counts competitive groups in each of 50 paired PSU resamples. Panel b
shows each group's frequency across those repeats, not a confidence interval
for the fitted parameter or a probability of biological mechanism identity.
The observed dataset has 22/23 competitive groups; the bootstrap count averages
18.84, with empirical 2.5th and 97.5th percentiles 15.225 and 22. Table 1 sorts
groups by unrounded Xc factor. Its Xc and epsilon factors come from separate
single-intrinsic-parameter fits; they are not a joint Xc+epsilon fit.

The paired baseline/group bootstrap, optimization, numerical checks and optional
cluster rerun instructions are in `analysis/model_fits/nhanes/README.md`.

## Extended Data Fig. 2: exposure-group survival

```bash
python3 analysis/figures/extended_data/render_nhanes_survival.py
```

The renderer uses `results/nhanes/survival_curves.csv`: the full cleaned cohort
and 23 overlapping groups, sampled every 0.25 years from age 20 to 110.
Columns are group ID, age in years, survival, participant count and death count.
The cohort contains 55,800 participants and 7,260 deaths. Curves use unweighted,
left-truncated/right-censored Kaplan-Meier estimation with each participant's
entry age, exit age and death indicator; no extrinsic mortality is removed.
Overlapping groups must not be summed as independent populations.

The renderer writes `Figures/ExtendedDataFigure2/ExtDataFig2.png`.

### Manuscript caption

**Extended Data Fig. 2 | Left-truncated, right-censored Kaplan–Meier survival curves for the cleaned NHANES likelihood cohort (55,800 participants; 7,260 deaths) and the same 23 overlapping exposure groups summarized in Supplementary Table 1.** Estimates use entry at the later of age 20 or enrollment and right censoring at the end of follow-up. Ages before a group's earliest observed entry are left blank; survival from age 20 is not identifiable for groups entering later. Curves are unweighted, unadjusted, and not corrected for extrinsic mortality.

### Regenerating the survival estimates

The source calculation can be regenerated with:

```bash
python3 scripts/prepare_nhanes_cohort.py
python3 scripts/prepare_nhanes_survival.py --cohort tmp/nhanes_cohort.csv
```

The preparation script verifies the exact cleaned-cohort hash, uses lifelines,
and writes summaries under `tmp/` by default. No participant records are needed
for routine rendering. `prepare_nhanes_cohort.py` reconstructs the cleaned
likelihood cohort from the bundled public-use inputs; the survival preparation
script consumes the resulting CSV.

Group curves begin at their observed entry ages. The no-friends group begins
at age 40; preceding KM values are stored as NaN and left blank in the figure.
