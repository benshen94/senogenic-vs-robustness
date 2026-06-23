# Senogenic versus robustness in human lifespan

This repository contains the analysis code, fitted model records, input data, cached simulations, and figure-generation scripts for the manuscript:

**Distinct mechanisms govern life expectancy versus extreme longevity in humans**

The central question is why life expectancy has risen strongly while the upper tail of human lifespan has moved only modestly. The analyses use stochastic threshold-crossing models of aging, primarily the Saturating-Removal (SR) model, to separate two parameter classes:

- **Senogenic parameters**: model parameters that alter how the stability landscape deteriorates with age, mainly \(\eta\) and \(\beta\).
- **Robustness parameters**: model parameters that alter threshold-crossing probability within a given landscape, mainly \(X_c\) and \(\epsilon\).

## What is included

The repository is organized as a small research compendium:

- `analysis/`: runnable manuscript analysis scripts, including main figures, supplementary figures, model fits, NHANES summaries, and checks.
- `src/senogenic_vs_robustness/`: project-specific helpers for paths, plotting, and model calibration.
- `src/ageing_packages/`: vendored SR simulation, HMD loading, fitting, and NHANES helper code used by the analyses.
- `data/`: bundled inputs needed to rerun the paper scripts, including HMD and NHANES files.
- `results/`: cached simulations, source tables, fitted parameter records, confidence intervals, and output indices.
- `results/tables/`: manuscript-facing source tables, including Extended Data Table 1 for the Fig. 3 NHANES projection audit and the Extended Data Fig. 1 tau-spread source data.
- `results/fits/`: archived SR baseline fits, fit confidence intervals, source summaries, and previews.
- `results/tau_wide_refit_profile/`: saved tau-senogenic heterogeneity refits and high-\(n\) mortality curves used for Extended Data Fig. 1.
- `Figures/`: tracked PNG previews of current figure outputs. PDF, AI, SVG, and notebook artifacts are intentionally not tracked; the scripts regenerate figure PDFs locally when needed.
- `docs/methods_log.md`: code-grounded methods notes for the figure and model workflows.
- `docs/repo_update_workflow.md`: checklist for updating this public repo from ongoing private analysis work.
- `docs/figure_methods/`: home for detailed per-figure methods dossiers.

## Setup

Use Python 3.11 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The scripts use repo-relative paths by default. To override the bundled HMD location, set:

```bash
export SENOGENIC_HMD_DATA_DIR=/path/to/HMD/files
```

## Quick verification

Run the lightweight self-contained check:

```bash
python3 scripts/verify_repo.py
```

This checks imports, bundled data files, HMD loading for Sweden/USA/Denmark, NHANES core linkage tables, fit records, and one tiny SR simulation.

## Reproducing figures

To rerun the main non-schematic figure scripts from cached source tables and simulations:

```bash
python3 scripts/reproduce_figures.py --set main
```

For a faster smoke test:

```bash
python3 scripts/reproduce_figures.py --set smoke
```

To rerun the Extended Data figures:

```bash
python3 scripts/reproduce_figures.py --set extended
```

To include the main, Extended Data, and Supplementary figure scripts:

```bash
python3 scripts/reproduce_figures.py --set all
```

Outputs are written under `Figures/` and source tables under `results/`, matching the manuscript workflow. The repository tracks PNG previews and the source data/caches needed to regenerate the figures. Generated PDFs are ignored by git so a clone stays lighter; rerun the figure scripts to create local PDF versions for submission or editing.

## Current manuscript figure map

These are the figure outputs intentionally tracked in `Figures/`. Intermediate panel PNGs/PDFs are regenerated locally but ignored by git.

| Manuscript item | Tracked preview | Reproduction script |
| --- | --- | --- |
| Fig. 1 | `Figures/Figure1/Fig1_alt.png` | `analysis/figures/figure1_schematic/make_fig1_alt.py`; quantitative panel from `analysis/figures/steepness_longevity/make_fig1d_new_steepness_longevity.py` |
| Fig. 2 | `Figures/Figure2/Figure2.png` | `analysis/figures/figure2/make_fig2a_new.py`, `make_fig2bc_new.py`, `make_fig2de_new.py`, `assemble_figure2.py` |
| Fig. 3 | `Figures/Figure3/fig3_exposure_projection.png` | `analysis/figures/steepness_longevity/make_fig3_exposure_projection.py`; Extended Data Table 1 from `make_fig3_coordinate_projection_uncertainty.py` |
| Fig. 4 | `Figures/Figure4/Figure4.png` | `analysis/figures/figure4/make_fig4_ab_sweden_period_projection.py`, `make_fig4_sr_contour_projection.py`, `make_fig4_age0_mean_lifespan_projection.py`, `assemble_figure4.py` |
| Fig. 5 | `Figures/Figure5_progeria/fig6_progeria_composite.png` | `analysis/figures/figure5_progeria/make_fig6_progeria.py` |
| Extended Data Fig. 1 | `Figures/ExtendedDataFigure1/extended_data_figure1_tau_spread_constraint.png` | `analysis/figures/extended_data/make_extended_data_figure1_tau_spread_constraint.py` |
| Extended Data Fig. 2 | `Figures/ExtendedDataFigure2/extended_data_figure2_denmark_robustness.png` | `analysis/figures/extended_data/make_extended_data_figure2_denmark_robustness.py` |
| Extended Data Fig. 3 | `Figures/ExtendedDataFigure3/extended_data_figure3_model_comparison.png` | `analysis/figures/extended_data/make_extended_data_figure3_model_comparison.py` |
| Supplementary Fig. 1 | `Figures/Supplementary/supp_figure1_gompertz_constraint.png` | `analysis/figures/supplementary/make_supp_figure1_gompertz_constraint.py` |
| Supplementary Fig. 2 | `Figures/Supplementary/supp_figure2_nhanes_exposure_groups.png` | `analysis/figures/supplementary/make_supp_figure2_nhanes_exposure_groups.py` |
| Supplementary Fig. 3 | `Figures/Supplementary/supp_figure3_fedichev_minimal_model.png` | `analysis/figures/supplementary/make_supp_figure3_fedichev_minimal_model.py` |
| Supplementary Fig. 4 | `Figures/Supplementary/supp_figure4_healthspan_morbidity.png` | `analysis/figures/supplementary/make_supp_figure4_healthspan_morbidity.py` |

Figure-specific methods notes live in `docs/figure_methods/current_figure_inventory.md`, with deeper dossiers for Fig. 3 and Extended Data Fig. 1.

## Data notes

HMD period and cohort files needed by the figure scripts are included so the HMD-based analyses can run without an external machine path. The original source is the Human Mortality Database, cited in the manuscript.

NHANES files are public source files and linked mortality tables used for the exposure-group Kaplan-Meier analyses. Manuscript-level summaries are saved under `results/tables/`; `extended_data_table1_fig3_projection.csv` is the current Extended Data Table 1 source for the coordinate-wise Fig. 3 projection analysis.

Extended Data Fig. 1 uses saved Sweden 2019 SR refits with imposed \(\tau_{\rm sen}=\beta/\eta\) heterogeneity. The final best-per-CV candidate table, all refined candidates, high-\(n\) mortality curves, and targeted 10% stress-refit outputs are saved under `results/tau_wide_refit_profile/`.

HGPS/progeria inputs and fit outputs are saved under `results/progeria*`, `results/cache/simulations/figure5_progeria/`, and `results/tables/fig6_progeria_*`.

No license file has been added yet. Until a license is chosen, the default GitHub terms apply.
