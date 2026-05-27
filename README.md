# Senogenic versus robustness in human lifespan

This repository contains the analysis code, fitted model records, input data, cached simulations, and figure-generation scripts for the manuscript:

**Distinct mechanisms govern life expectancy versus extreme longevity in humans**

The central question is why life expectancy has risen strongly while the upper tail of human lifespan has moved only modestly. The analyses use stochastic threshold-crossing models of aging, primarily the Saturating-Removal (SR) model, to separate two parameter classes:

- **Senogenic parameters**: model parameters that alter how the stability landscape deteriorates with age, mainly \(\eta\) and \(\beta\).
- **Robustness parameters**: model parameters that alter threshold-crossing probability within a given landscape, mainly \(X_c\) and \(\epsilon\).

## What is included

- `ageing_packages/`: vendored SR simulation, HMD loading, fitting, and NHANES helper code used by the manuscript analyses.
- `src/shared/thresholds/`: project-specific path helpers, plotting utilities, and model-calibration helpers.
- `src/figures/`: scripts that regenerate paper figure panels and supporting figures.
- `src/exploration/`: fitting and analysis scripts used to build archived results.
- `saved_results/fit_archive/`: SR baseline fits, confidence intervals, source summaries, and fit previews.
- `saved_results/`: cached simulation outputs, source CSVs, fit tables, and indexed analysis artifacts.
- `saved_data/hmd/`: HMD period life-table files used for Sweden, USA, and Denmark analyses.
- `saved_data/nhanes/`: NHANES input tables and public XPT files used to construct exposure-group survival curves.
- `Figures_new/`: current generated PDF/PNG figure outputs.
- `docs/methods_log.md`: code-grounded methods notes for the current figure and model workflows.

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

This checks imports, bundled data files, HMD loading for Sweden/USA/Denmark, NHANES core linkage tables, fit archive records, and one tiny SR simulation.

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

Outputs are written under `Figures_new/` and source tables under `saved_results/`, matching the manuscript workflow. Some scripts can recompute missing simulation caches, but the repository includes the saved caches used for the current figures.

## Important figure/data scripts

- Fig. 1 steepness-longevity response plane: `src/figures/steepness_longevity/make_fig1d_new_steepness_longevity.py`
- Fig. 2 tail and sibling panels: `src/figures/Fig2_new/`
- Fig. 3 NHANES/exposure projections: `src/figures/steepness_longevity/make_fig3_usa_steepness_longevity.py` and `make_fig3_exposure_projection.py`
- Fig. 4 historical HMD/extrapolation analyses: `src/figures/Fig4_new/`
- Fig. 5 progeria analyses: `src/figures/Fig6_progeria/make_fig6_progeria.py`
- Supplementary parameter heterogeneity: `src/figures/FigS1/make_parameter_distribution_supplement.py`
- Supplementary model comparison: `src/figures/Supp_Figgs/make_supp_model_comparison.py` uses saved source CSVs by default; pass `--force-sim` to rerun the expensive extreme-lifespan simulations.
- Supplementary NHANES survival curves: `src/figures/Supp_Figgs/make_supp_fig4_nhanes_exposure_groups.py`

## Data notes

HMD period and cohort files needed by the figure scripts are included so the HMD-based analyses can run without an external machine path. The original source is the Human Mortality Database, cited in the manuscript.

NHANES files are public source files and linked mortality tables used for the exposure-group Kaplan-Meier analyses. Manuscript-level summaries are also saved under `saved_results/csv/`.

HGPS/progeria inputs and fit outputs are saved under `saved_results/progeria*`, `saved_results/cache/simulations/Fig6_progeria/`, and `saved_results/csv/fig6_progeria_*`.

No license file has been added yet. Until a license is chosen, the default GitHub terms apply.
