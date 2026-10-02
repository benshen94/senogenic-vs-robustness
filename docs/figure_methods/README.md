# Figure Methods

Start with the [current manuscript inventory](current_figure_inventory.md).
The paper includes six main figures, four Extended Data figures,
five supplementary figures and Extended Data Table 1.

## Current Methods

| Figure or analysis | Methods |
| --- | --- |
| Figure 1: FP response plane | [Baseline, response curves and schematic boundary](figure1_fp.md) |
| Figure 2: heterogeneity and lifespan tails | [FP populations, conditioning and siblings](figure2_fokker_planck.md) |
| Figure 3: NHANES | [Likelihood, reference fits and lifespan gains](figure3_likelihood.md) |
| Figures 4–5 and Extended Data 3 | [Sweden/Denmark direct fits, covariance and forecasts](historical_fits.md) |
| Figure 6: HGPS | [Records, likelihood and model comparisons](figure6_progeria.md) |
| Extended Data 1 | [Senogenic timescale heterogeneity profile](senogenic_heterogeneity_fp.md) |
| Extended Data 2, Table 1 and Supplementary 3 | [Paired AIC stability and cleaned-cohort survival](nhanes_aic_and_survival.md) |
| Extended Data 4 and Supplementary 4 | [Fedichev-Gruber calculations](fedichev_gruber.md) |
| Supplementary 2 | [FP Gompertz-constraint generator and conditioning](../../analysis/model_fits/supplementary/README.md) |
| Supplementary 1 | [Mean-parameter shifts and upper-tail survival](parameter_shifts.md) |
| Supplementary 5 | [Joint onset/death FP method and validation status](healthspan_fp.md) |

Supplementary 2 and 5 use checked numerical results included in the repository.
The normal saved-result runner covers all five supplementary figures without
starting new calculations.

## Interpretation

Sweden/Denmark use the age-balanced Q estimating criterion. Do not interpret
Q differences as likelihood ratios or AIC. NHANES and HGPS use individual
survival likelihoods with different observation schemes and reference fits.
Historical Poisson recovery simulations are retained as diagnostics; they do
not calibrate the current dispersion-adjusted bands or define percentile bands.

Figure 2 is regenerated as a complete composite from saved FP tables. Figure 1's
quantitative panel is reproduced separately from its schematic composite.
The FG model is distinct from SR and retains its own simulation procedure.

## Maintaining These Notes

Ground descriptions in the generating code, saved records and current
manuscript. Document parameter restrictions, objectives, conditioning,
normalization, uncertainty and provenance explicitly. Distinguish actual
recorded cluster jobs from example commands. Ordinary reproduction reads saved
outputs; expensive computation must be an explicit separate command.

Publication figures are provided as composite PNGs.
