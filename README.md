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
- `results/fits/`: archived SR baseline fits, fit confidence intervals, source summaries, and previews.
- `Figures/`: current generated PDF/PNG figure outputs.
- `docs/methods_log.md`: code-grounded methods notes for the figure and model workflows.

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

To include the supplementary figure scripts as well:

```bash
python3 scripts/reproduce_figures.py --set all
```

Outputs are written under `Figures/` and source tables under `results/`, matching the manuscript workflow. Some scripts can recompute missing simulation caches, but the repository includes the saved caches used for the current figures.

## Important figure/data scripts

- Fig. 1 steepness-longevity response plane: `analysis/figures/steepness_longevity/make_fig1d_new_steepness_longevity.py`
- Fig. 2 tail and sibling panels: `analysis/figures/figure2/`
- Fig. 3 NHANES/exposure projections: `analysis/figures/steepness_longevity/make_fig3_usa_steepness_longevity.py` and `make_fig3_exposure_projection.py`
- Fig. 4 historical HMD/extrapolation analyses: `analysis/figures/figure4/`
- Fig. 5 progeria analyses: `analysis/figures/figure5_progeria/make_fig6_progeria.py`
- Supplementary parameter heterogeneity: `analysis/figures/supplementary_parameter_distributions/make_parameter_distribution_supplement.py`
- Supplementary model comparison: `analysis/figures/supplementary/make_supp_model_comparison.py` uses saved source CSVs by default; pass `--force-sim` to rerun the expensive extreme-lifespan simulations.
- Supplementary NHANES survival curves: `analysis/figures/supplementary/make_supp_fig4_nhanes_exposure_groups.py`

## Data notes

HMD period and cohort files needed by the figure scripts are included so the HMD-based analyses can run without an external machine path. The original source is the Human Mortality Database, cited in the manuscript.

NHANES files are public source files and linked mortality tables used for the exposure-group Kaplan-Meier analyses. Manuscript-level summaries are also saved under `results/tables/`.

HGPS/progeria inputs and fit outputs are saved under `results/progeria*`, `results/cache/simulations/figure5_progeria/`, and `results/tables/fig6_progeria_*`.

No license file has been added yet. Until a license is chosen, the default GitHub terms apply.
