# Current Figure Inventory

This file is the public repo checklist for the current revised manuscript. It should be updated whenever the manuscript figure list changes.

## Main figures

| Figure | Scientific role | Public output | Main scripts | Source data and caches |
| --- | --- | --- | --- | --- |
| Fig. 1 | Defines stochastic threshold crossing, senogenic versus robustness parameter classes, and the SR steepness-longevity response. | `Figures/Figure1/Fig1_alt.png` | `analysis/figures/figure1_schematic/make_fig1_alt.py`; `analysis/figures/steepness_longevity/make_fig1d_new_steepness_longevity.py` | `results/steepness_longevity_sweden2019_sensitivity/metrics_long.csv`; `results/steepness_longevity_sweden2019_sensitivity/fig1d_new_steepness_longevity_*.csv`; SR fit record `results/fits/records/joint2019_tail90_sweden_emphasis.json`. |
| Fig. 2 | Tests tail survival, maximum-lifespan shifts, and sibling-mortality convergence under robustness versus senogenic heterogeneity. | `Figures/Figure2/Figure2.png` | `analysis/figures/figure2/make_fig2a_new.py`; `make_fig2bc_new.py`; `make_fig2de_new.py`; `assemble_figure2.py` | Sweden HMD files; `results/fits/records/joint2019_tail90_sweden_emphasis.json`; `results/fits/ci/joint2019_tail90_sweden_emphasis_ci.csv`; `results/cache/simulations/figure2/`; `results/tables/fig2*.csv`. |
| Fig. 3 | Projects NHANES exposure-group survival signatures onto SR response classes and maps robustness-equivalent lifespan gains. | `Figures/Figure3/fig3_exposure_projection.png` | `analysis/figures/steepness_longevity/make_fig3_usa_steepness_longevity.py`; `make_fig3_exposure_projection.py`; `make_fig3_coordinate_projection_uncertainty.py` | `results/exposure_groups_results.pkl`; `results/steepness_longevity_usa2019_sensitivity/`; `results/figure3_exposure_projection/`; `results/tables/extended_data_table1_fig3_projection.csv`. |
| Fig. 4 | Compares historical Swedish mortality change with SR robustness trajectories and extrapolations. | `Figures/Figure4/Figure4.png` | `analysis/figures/figure4/make_fig4_ab_sweden_period_projection.py`; `make_fig4_sr_contour_projection.py`; `make_fig4_age0_mean_lifespan_projection.py`; `assemble_figure4.py` | Sweden HMD files; `results/figure4/sweden_*`; `results/steepness_longevity_sweden2019_sensitivity/`. |
| Fig. 5 | Uses HGPS/progeria survival as a strong perturbation test for robustness versus senogenic parameter shifts. | `Figures/Figure5_progeria/fig6_progeria_composite.png` | `analysis/figures/figure5_progeria/make_fig6_progeria.py` | `results/progeria_data.pkl`; `results/progeria_fitting_results.pkl`; `results/cache/simulations/figure5_progeria/`; `results/tables/fig6_progeria_*.csv`. |

## Extended Data figures

| Figure | Scientific role | Public output | Main scripts | Source data and caches |
| --- | --- | --- | --- | --- |
| Extended Data Fig. 1 | Shows that imposed \(\tau_{\rm sen}=\beta/\eta\) heterogeneity worsens Sweden 2019 mortality fits even after wide refitting. | `Figures/ExtendedDataFigure1/extended_data_figure1_tau_spread_constraint.png` | `analysis/figures/extended_data/make_extended_data_figure1_tau_spread_constraint.py`; fit generators in `analysis/model_fits/tau_wide_refit_profile/` | `results/tau_wide_refit_profile/`; `results/tables/extended_data_figure1_tau_spread_constraint_source_data.csv`. |
| Extended Data Fig. 2 | Repeats the historical robustness/extrinsic-mortality analysis for Denmark. | `Figures/ExtendedDataFigure2/extended_data_figure2_denmark_robustness.png` | `analysis/figures/extended_data/make_extended_data_figure2_denmark_robustness.py` | Denmark HMD files; `results/figure4/denmark_period_steepness_longevity_projection.csv`; `results/figure4/denmark_fig4c_*`; `results/figure4/denmark_fig4d_*`. |
| Extended Data Fig. 3 | Compares Gompertz-Makeham and Fedichev-Gruber model behavior against the SR response classes. | `Figures/ExtendedDataFigure3/extended_data_figure3_model_comparison.png` | `analysis/figures/extended_data/make_extended_data_figure3_model_comparison.py` | `results/gamma_factor_sweep.pkl`; `results/fedichev_model_steepness_longevity_data.pkl`; `results/tables/supp_model_comparison_*.csv`. |

## Supplementary figures

| Figure | Scientific role | Public output | Main scripts | Source data and caches |
| --- | --- | --- | --- | --- |
| Supplementary Fig. 1 | Shows Gompertz mortality constraints on broad senogenic heterogeneity. | `Figures/Supplementary/supp_figure1_gompertz_constraint.png` | `analysis/figures/supplementary/make_supp_figure1_gompertz_constraint.py` | `results/tables/supplementary_figure1/sweden2019_decade_slopes.csv`; `allowed_parameter_cv_vs_slope_distortion.csv`; `survivor_parameter_means.csv`; `senogenic_heterogeneity_hazards.csv`. |
| Supplementary Fig. 2 | Shows raw NHANES Kaplan-Meier survival curves by exposure group. | `Figures/Supplementary/supp_figure2_nhanes_exposure_groups.png` | `analysis/figures/supplementary/make_supp_figure2_nhanes_exposure_groups.py` | Public NHANES files in `data/nhanes/`; Kaplan-Meier curves computed directly from bundled linked-mortality tables. |
| Supplementary Fig. 3 | Displays the Fedichev-Gruber minimal model survival and mortality behavior used as a model-comparison reference. | `Figures/Supplementary/supp_figure3_fedichev_minimal_model.png` | `analysis/figures/supplementary/make_supp_figure3_fedichev_minimal_model.py` | `results/tables/supp_figure3_fedichev_minimal_model_source.csv`; regenerated with `--force-sim` if needed. |
| Supplementary Fig. 4 | Shows the SR disease-threshold healthspan/morbidity analysis. | `Figures/Supplementary/supp_figure4_healthspan_morbidity.png` | `analysis/figures/supplementary/make_supp_figure4_healthspan_morbidity.py`; source exploration in `analysis/quality_checks/artificial_survival_time/` | `results/cache/simulations/artificial_survival_time/matched_sweden2019_event_times.npz`; `results/tables/artificial_survival_time_summary.csv`; `results/tables/artificial_survival_time_state_composition.csv`. |

## Tables

| Table | Public output | Script | Notes |
| --- | --- | --- | --- |
| Extended Data Table 1 | `results/tables/extended_data_table1_fig3_projection.csv` | `analysis/figures/steepness_longevity/make_fig3_coordinate_projection_uncertainty.py --skip-point-mc` | NHANES exposure-group assignment fractions and Euclidean log-space distances projected onto SR response classes. |
| Supplementary Table 1 | `results/tables/nhanes_sample_stats_summary.csv` | `analysis/nhanes/print_nhanes_sample_stats.py` | NHANES sample summary for the exposure groups used in Fig. 3 and Supplementary Fig. 2. |

## Update workflow

When the manuscript changes, update this file first from the manuscript figure list, then check that `README.md`, `scripts/reproduce_figures.py`, `scripts/verify_repo.py`, and `results/index/outputs.csv` refer to the same figure set. The tracked `Figures/` directory should contain only manuscript-level composite PNG previews, not draft panels or local PDF/AI/SVG outputs.
