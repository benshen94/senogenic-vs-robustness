# Figure 3 methods

## Figure scope

Figure 3 compares NHANES exposure groups with SR-model response directions in normalized median-lifespan and steepness space. The figure supports a class-level interpretation: exposure-group survival signatures align with robustness-like, senogenic-like, or extrinsic-mortality-like model directions. It should not be read as unique identification of latent SR parameters in individuals.

## Panel inventory

| Panel | Output | Script | Main inputs | Notes |
| --- | --- | --- | --- | --- |
| A | `Figures/Figure3/fig3_exposure_projection_panel_a.png` | `analysis/figures/steepness_longevity/make_fig3_exposure_projection.py` | `results/exposure_groups_results.pkl`, `results/steepness_longevity_usa2019_sensitivity/fig3_usa_steepness_longevity_point_intervals.csv` | NHANES exposure points over the USA 2019 SR response plane. |
| B | `Figures/Figure3/fig3_exposure_projection_panel_b.pdf`, `Figures/Figure3/fig3_exposure_projection_panel_b.png` | `analysis/figures/steepness_longevity/make_fig3_exposure_projection.py` | `results/figure3_exposure_projection/fig3_panel_b_xc_factor_curves_fit_ci.csv`, `results/figure3_exposure_projection/projected_xc_age_gain_curves.csv` | Remaining-lifespan gain curves for Xc-equivalent robustness factors. |
| C | `Figures/Figure3/fig3_exposure_projection_panel_c.png` | `analysis/figures/steepness_longevity/make_fig3_exposure_projection.py` | HMD USA period life table and SR fit records | USA period survival context panel. |
| Composite | `Figures/Figure3/fig3_exposure_projection.pdf`, `Figures/Figure3/fig3_exposure_projection.png` | `analysis/figures/steepness_longevity/make_fig3_exposure_projection.py` | Panels A-C | Current manuscript Fig. 3 composite. |
| Extended Data Table 1 | `results/tables/extended_data_table1_fig3_projection.csv` | `analysis/figures/steepness_longevity/make_fig3_coordinate_projection_uncertainty.py` | NHANES public input tables, SR response curves, bootstrap outputs | Coordinate-wise projection audit for Fig. 3a. |

## Input data provenance

- NHANES linked mortality inputs are bundled under `data/nhanes/`. The figure scripts use cleaned participant tables and exposure-group survival summaries derived from these public files.
- HMD USA period life-table files are bundled under `data/hmd/` and are used for the period-survival context.
- The USA 2019 SR baseline is loaded from `results/steepness_longevity_usa2019_sensitivity/manifest.json`; fit-local confidence intervals are loaded from `results/fits/ci/hybrid2019_swe_tail90_usa_refit_ci.csv`.
- SR response-plane source tables are cached under `results/steepness_longevity_usa2019_sensitivity/`.

## Model or statistic

Panel A places each exposure group at:

\[
x=M_i/M_0,\quad y=S_i/S_0,
\]

where \(M_i\) is median lifespan, \(S_i\) is the IQR steepness metric, and \(M_0,S_0\) are the zero-Makeham NHANES baseline values.

Extended Data Table 1 uses log-normalized coordinates:

\[
z_i=(\log(M_i/M_0), \log(S_i/S_0)).
\]

For each SR coordinate curve \(C_p(q)\), including extrinsic mortality \(m_{ex}\), the distance is:

\[
D_{i,p}=\min_q \|z_i-C_p(q)\|_2.
\]

The central coordinate assignment is the nearest curve by this Euclidean log-space distance. Coordinate-level assignments are then grouped into senogenic, robustness, and extrinsic classes for manuscript reporting.

## Uncertainty and bands

- Fig. 3a point error bars come from bootstrap standard errors in `results/exposure_groups_results.pkl`.
- Fig. 3a shaded SR regions are deterministic baseline-sensitivity envelopes from one-at-a-time perturbations of the USA 2019 baseline.
- Extended Data Table 1 propagates uncertainty with 300 participant-level bootstrap resamples using seed `20260528`.
- Each bootstrap replicate resamples the full NHANES baseline and each exposure group, recomputes left-truncated Kaplan-Meier survival, refits one-year central death rates to a Makeham-Gamma-Gompertz model, sets the Makeham term to zero, and recomputes normalized median lifespan and steepness.
- Bootstrap assignment fractions are recomputed across the Fig. 3a baseline-sensitivity curve sets. The table reports the central assignment fraction and the min-max range across curve sets.
- One bootstrap row is excluded and logged: `income / Q4 (Highest)` in replicate 146 produced non-finite normalized steepness.
- The `Xc factor` range in Extended Data Table 1 is loaded from `results/figure3_exposure_projection/exposure_xc_equivalent_projection_full_uncertainty.csv`, the cached manuscript source coupling the Fig. 3a exposure projection with the Fig. 3b Xc-equivalent mapping. Passing `--recompute-xc-factor-ranges` regenerates a direct nearest-Xc-curve fallback rather than using that cached manuscript source.

## Reproduction command

```bash
python3 scripts/reproduce_figures.py --set main
```

To rerun only the coordinate-wise projection table:

```bash
python3 analysis/figures/steepness_longevity/make_fig3_coordinate_projection_uncertainty.py --skip-point-mc
```

## Expected outputs

- `Figures/Figure3/fig3_exposure_projection.pdf`
- `Figures/Figure3/fig3_exposure_projection.png`
- `results/figure3_exposure_projection/exposure_coordinate_projection_paper_summary.csv`
- `results/figure3_exposure_projection/exposure_coordinate_projection_full_bootstrap_assignments.csv`
- `results/figure3_exposure_projection/exposure_coordinate_projection_full_bootstrap_summary.csv`
- `results/figure3_exposure_projection/exposure_coordinate_projection_model_sensitivity_summary.csv`
- `results/figure3_exposure_projection/exposure_coordinate_projection_full_bootstrap_failures.csv`
- `results/figure3_exposure_projection/exposure_coordinate_projection_methods_log.md`
- `results/figure3_exposure_projection/extended_data_table_projection_with_ranges.csv`
- `results/tables/extended_data_table1_fig3_projection.csv`

## Validation checks

- `python3 scripts/verify_repo.py` checks that Extended Data Table 1 has 23 exposure rows and that the single excluded bootstrap row is the known Q4-income row.
- The 300-bootstrap public run was compared against the private manuscript workspace for coordinate distances, full-bootstrap assignments, summary tables, model-sensitivity summary, paper summary, Xc-equivalent factor table, and the Extended Data Table 1 CSV.
- The coordinate-distance and assignment outputs matched the private analysis exactly after regeneration.

## Known caveats

- The point-MC diagnostic file is optional and is not written by the default repo reproduction command; use the coordinate script without `--skip-point-mc` if that diagnostic is needed.
- The analysis supports class-level survival-signature alignment. Avoid wording that implies unique causal identification of \(\eta\), \(\beta\), \(X_c\), \(\epsilon\), or \(m_{ex}\) from an exposure group alone.
