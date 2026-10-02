# Manuscript Figure Inventory

The paper includes six main figures, four Extended Data figures, five
supplementary figures and Extended Data Table 1. Paths are relative to the
repository root.

## Main Figures

| Figure | Current analysis and saved sources | Rendering entry point |
| --- | --- | --- |
| 1 | SR schematics and FP response plane; `results/figure1/` | `analysis/figures/figure1_schematic/render_fp_panel.py` exports numeric panel g to `tmp/figure1/`. The manuscript composite `Figures/Figure1/Fig1.png` is preserved. |
| 2 | Heterogeneity, tails, age-90 conditioning and siblings; `results/tables/fig2_fokker_planck/` | `analysis/figures/figure2/plot_fig2_fp.py`; reproduces the complete `Figures/Figure2/Fig2.png` FP composite. |
| 3 | NHANES likelihood-derived signatures and lifespan gains; `results/nhanes/figure3/` and saved paired reference fits | `analysis/figures/figure3/render_nhanes.py`; `Figures/Figure3/Fig3.png`. |
| 4 | Historical Sweden coordinates, external mortality and direct Xc+mex fits; `results/historical/` | `analysis/figures/figure4/render_current_history.py`; `Figures/Figure4/Fig4.png`. |
| 5 | Separate historical/forecast lifespan contours conditioned on age 20; saved covariance-derived bands in `results/historical/` | `analysis/figures/figure5/render_lifespan.py`; `Figures/Figure5/Fig5.png`. |
| 6 | HGPS survival likelihood and ten-model AIC comparison; `data/hgps/`, `results/progeria/` | `analysis/figures/figure6_progeria/plot_fig6.py`; `Figures/Figure6/Fig6.png`. |

Historical Sweden/Denmark use Q estimating equations, not a likelihood that
can be compared by AIC. HGPS and NHANES use survival likelihoods. Their
references, cohort constructions and uncertainty calculations are distinct.
Figures 4d and 5 and Extended Data 3 use per-year dispersion-adjusted,
approximate pointwise intervals; see the historical methods for their definition.

## Extended Data

| Figure | Current analysis and saved sources | Rendering entry point |
| --- | --- | --- |
| 1 | Senogenic-timescale heterogeneity profile, relative Q and mortality; `results/senogenic_heterogeneity/` | `analysis/figures/extended_data/render_senogenic_profile.py` |
| 2 | NHANES paired-bootstrap AIC competitiveness; `results/nhanes/bootstrap/` and summary tables | `analysis/figures/extended_data/render_nhanes_aic.py` |
| 3 | Denmark historical analysis; `results/historical/` | Included in `analysis/figures/figure4/render_current_history.py` |
| 4 | Fedichev-Gruber extreme lifespan and response plane only; `results/fedichev_gruber/` | `analysis/figures/extended_data/render_fedichev_constraints.py` |

Outputs are under the corresponding `Figures/ExtendedDataFigureN/` folders.
The FG model uses its own simulation procedure.

Extended Data Table 1 is `results/nhanes/extended_data_table1.csv`,
regenerated with Extended Data Figure 2. It reports groups, fitted parameter
factors and AIC/bootstrap support.

## Supplementary Figures

| Figure | Status and sources | Entry point |
| --- | --- | --- |
| 1 | Homogeneous mean-parameter shifts; `results/tables/fig2_fokker_planck/tails_factor.csv`. | `analysis/figures/supplementary/render_parameter_shifts.py`; output `Figures/Supplementary/SuppFig1.png`. |
| 2 | Empirical mortality slopes, local survivor-conditioned heterogeneity tolerance curves, parameter means by death-lifespan bins and heterogeneous mortality. Saved production, quadrature and refined grids in `results/supplementary1_fp`. | Optional checks: `analysis/model_fits/supplementary/check_si.py --recompute`; renderer: `analysis/figures/supplementary/render_gompertz_constraints.py`. |
| 3 | Current cleaned-cohort delayed-entry KM curves; `results/nhanes/survival_curves.csv`. Early unsupported ages remain missing. | `analysis/figures/supplementary/render_nhanes_likelihood.py`; output `Figures/Supplementary/SuppFig3.png`. |
| 4 | Saved FG survival/mortality; `results/tables/supp_figure3_fedichev_minimal_model_source.csv` and its recorded cache. | `analysis/figures/supplementary/render_fedichev_minimal_model.py`; missing cache does not silently trigger simulation. |
| 5 | Tagged joint onset/death FP calculation; `results/supplementary4_fp/` includes the adopted refined grid and coarser-grid checks. | Workers: `analysis/model_fits/supplementary/healthspan.py`; renderer: `analysis/figures/supplementary/render_healthspan_morbidity.py`. |

## Verification

`python3 scripts/reproduce_figures.py --set current` runs current saved-result
renderers, including the complete FP Figure 2 and all five supplementary figures.
The manual Figure 1 composite remains separate from its numerical preview.

`python3 scripts/verify_repo.py` checks saved archive consistency, paired
bootstrap membership and AIC accounting. It does not refit parameters or
certify solver convergence.

The per-figure methods in this folder describe data, objectives, normalization,
uncertainty and full recomputation commands.
