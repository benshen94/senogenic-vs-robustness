# Figure 3: NHANES likelihood reference

```bash
python3 analysis/figures/figure3/render_nhanes.py
```

This renders the composite from `results/nhanes/figure3/` without fitting or
simulation.

The cleaned cohort has 55,800 participants and 7,260 deaths. The full-cohort
likelihood reference fixes eta=0.59, beta=57.9 and kappa=0.5, with
Xc=18.45815654294416, epsilon=42.372450276429184, threshold CV=0.2878703142772308,
and external mortality at zero. Response curves retain threshold heterogeneity.

Panel a shows FP response curves and observed delayed-entry Kaplan-Meier group
coordinates. Colored envelopes reflect one-at-a-time baseline perturbations
of plus/minus 20%; they are sensitivity envelopes, not confidence intervals.
Observed error bars are one standard error from 100 paired survey-PSU resamples.
The highest-income steepness ratio exceeds the plot range and is marked above
scale. Exposure coordinates are not projected onto an Xc curve in this analysis.

Panel b shows hypothetical Xc changes with pointwise 95% percentile ribbons
from 100 paired full-cohort reference refits. Both raw draws and the ribbons
are included. These model-conditional intervals hold eta, beta and kappa fixed.
Panel c retains the published Sakaniwa comparison.

The associated exposure-model AIC comparison and its 50 paired group resamples
are documented in `analysis/model_fits/nhanes/README.md`. The group-fit bootstrap
and the 100-draw reference bootstrap have distinct purposes and sample counts.
