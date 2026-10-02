# Supplementary Figure 1: mean-parameter shifts

The figure shows the age reached by the top 0.01% of survivors when one SR
parameter mean is changed at a time. The numerical source is
`results/tables/fig2_fokker_planck/tails_factor.csv`; no fits or simulations run
during rendering.

```bash
python3 analysis/figures/supplementary/render_parameter_shifts.py
```

The output is `Figures/Supplementary/SuppFig1.png`. The figure belongs at the
end of SI section S3, after the SR mortality and late-life scaling derivations,
before the population-heterogeneity analysis in S4.

## Calculation

The frozen Sweden 2019 reference, numerical solver and uncertainty calculation
are shared with [Figure 2](figure2_fokker_planck.md). Each curve scales one
of eta, beta, Xc or epsilon by factors 0.85--1.15, holding the other means
fixed. All parameter CVs and extrinsic mortality are zero. This is a homogeneous
mean-shift experiment, not an experiment that changes individual heterogeneity.

For each factor, the age is interpolated at unconditional survival 0.0001.
The vertical dotted line is the reference factor 1. Shading shows approximate
pointwise 95% log-delta intervals propagated from the joint baseline covariance
of log Xc and log epsilon. These are neither population spread nor simultaneous
confidence intervals.

The producer is `analysis/figures/figure2/run_fig2_fp.py`; its optional full-run
command, 320-cell grid, 0.025-year time step, integration nodes and refinement
checks are documented in the Figure 2 methods. Saved mean-shift values and
their confidence bounds are unchanged by separating this figure from Figure 2.
