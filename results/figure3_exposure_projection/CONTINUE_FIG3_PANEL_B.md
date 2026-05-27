# Fig. 3 Panel B Update Log

Updated on 2026-05-20 after completing the paused panel B simulation and replacing only panel B inside the manually edited Fig. 3 PDF. Updated again to draw per-factor exposure-projection uncertainty with distinct Coolwarm colors.

## Goal

Redo only Fig. 3 panel B so shaded bands are the USAFIT baseline-fit confidence envelope, not finite simulation uncertainty for the median.

## Final state

- Raw checkpoint file:
  `results/fig3_exposure_projection/fig3_panel_b_xc_factor_curves_fit_ci_raw.csv`
- Completed checkpoint rows: 405 rows.
- Completed factor/scenario runs: 45 of 45.
- Final summarized cache:
  `results/fig3_exposure_projection/fig3_panel_b_xc_factor_curves_fit_ci.csv`
- Panel B visual method:
  central model lines are fixed 0.80 to 1.20 Xc factors. The current figure draws per-line exposure-projection uncertainty ribbons in the same Coolwarm color as each Xc-factor curve. Ribbon boundaries are not stroked because the borders made the shaded regions read like extra curves.
- Existing manually edited composite PDF was updated in place:
  `Figures/Figure3/fig3_exposure_projection.pdf`
- A backup of the pre-replacement PDF was saved at:
  `results/fig3_exposure_projection/fig3_exposure_projection_before_panel_b_update.pdf`

## Code changes already made

- `analysis/figures/steepness_longevity/make_fig3_exposure_projection.py`
  now uses `n = 300000` by default for panel B, has `--panel-b-only`, and checkpoints raw per-run medians.
- Panel B factors are 0.80 to 1.20 in 0.05 jumps.
- Baseline scenarios are central USAFIT plus one-at-a-time endpoints for `Xc` and `xc_std_frac`, loaded from:
  `results/fits/ci/hybrid2019_swe_tail90_usa_refit_ci.csv`
- Gains are computed relative to each scenario's own factor 1.00 baseline.
- The plotted central curve is the central scenario.
- The saved factor-curve CSV still keeps the fit-endpoint columns for traceability. The displayed ribbons come from `fig3_panel_b_projection_uncertainty_ribbons.csv`, computed using the full cached Xc-response grid from 0.60 to 1.40 so edge curves can still have upper/lower projection uncertainty.
- `../../ageing_packages/SR_models/simulation.py` was patched so the fast kernel reads missing `Xdisease` safely with `getattr(..., None)`.

## Regeneration note

Do not use `--force-sim` unless the whole `n = 300000` sweep should be rerun from scratch. The raw checkpoint and final summarized cache are already complete.
