# Figure 1 quantitative panel

`Figures/Figure1/Fig1.png` contains the schematic composite and Q-reference
response plane in panel g. The numerical plotting command reproduces panel g:

```bash
python3 analysis/figures/figure1_schematic/render_fp_panel.py
```

The schematic artwork is provided as a complete composite. The command above
regenerates the numerical panel separately.

This writes `tmp/figure1/panel_g.png` from `results/figure1/metrics_long.csv`.
The quantitative panel uses the original plotting functions and current data.
Its canvas matches the archived panel dimensions after making export padding
explicit. The saved source and regenerated plot show the same numerical curves;
legend placement and typography differ slightly, so pixel identity is not
claimed. The original composite is retained unchanged. Source parameters and numerical settings are recorded in
`results/figure1/manifest.json`.

The reference uses the Swedish Q fit, with external mortality set to zero for
intrinsic response curves and fitted positive-Gaussian threshold heterogeneity
retained. Curves are calculated by deterministic finite-volume first passage,
with 320 cells, 0.025-year steps, 61 threshold nodes and a 240-year horizon.

Shading summarizes one-at-a-time plus/minus 20% perturbations of five nuisance
parameters around the reference. It is a sensitivity envelope, not a confidence
interval. Within-scenario normalization and aggregation follow the source plot.
The optional full response-calculation script is
`analysis/model_fits/response_curves/generate.py`. It uses the bundled production
solver and frozen baseline, writes to `tmp/response_curves_recomputed/`, and
requires `--recompute`. Worker count defaults to one and is bounded at eight.
Normal rendering does not call it. The source design contains 11 sensitivity
scenarios and 495 response records, with duplicate intrinsic mixtures cached.
