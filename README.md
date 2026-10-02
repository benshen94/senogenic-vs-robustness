# Senogenic versus robustness in human lifespan

Code, data summaries, fitted models and figure workflows for:

**Extreme human longevity and lifespan variation through the lens of stochastic
threshold-crossing models of aging**

Ben Shenhar, Shachaf Frenkel, Tomer Levy and Uri Alon.

Archived code and data, version 0.1.0:
[doi:10.5281/zenodo.23015915](https://doi.org/10.5281/zenodo.23015915).
Citation metadata are provided in [CITATION.cff](CITATION.cff).

This compendium separates senogenic changes in aging dynamics from robustness
changes in threshold crossing. The analyses use a finite-volume
Fokker-Planck solver for the Saturating-Removal (SR) model.

The repository contains the analyses for six main figures, five Extended Data
figures, four supplementary figures, Extended Data Table 1, Table M1 and
Supplementary Table 1. Figures can be
reproduced from the included source tables and fitted models.

## Setup

Use Python 3.11 (the tested version), from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Python 3.11 on macOS arm64 is the tested environment. The checks below exercise
the saved records, joint first-passage solver, renderer input validation and
historical dispersion calculation without expensive fitting:

```bash
python3 scripts/verify_repo.py
python3 -m unittest discover -s tests -v
python3 analysis/model_fits/historical/test_dispersion.py
```

## Reproduce Saved Results

```bash
python3 scripts/reproduce_figures.py --set current
```

This command renders figures from the included source tables and fitted models.
Composite PNGs are written under `Figures/`. Figure 1 includes manually drawn
schematics; its numerical panel is rendered separately to `tmp/figure1/`.

Supplementary Fig. 4 uses joint first-passage results with three-grid numerical checks.
The [output index](results/index/outputs.csv) records every current artifact,
its source script and inputs.

## Manuscript Map

| Item | Analysis | Entry point |
| --- | --- | --- |
| Fig. 1 | Parameter classes; SR response plane | `analysis/figures/figure1_schematic/render_fp_panel.py`; original composite retained; regenerated panel has minor typographic differences |
| Fig. 2 | Heterogeneity, extreme survivors and siblings | `analysis/figures/figure2/plot_fig2_fp.py`; saved FP tables reproduce [the complete figure](Figures/Figure2/Fig2.png) |
| Fig. 3 | NHANES survival signatures and remaining lifespan | `analysis/figures/figure3/render_nhanes.py` |
| Fig. 4 | Swedish historical mortality and fitted robustness | `analysis/figures/figure4/render_current_history.py` |
| Fig. 5 | Historical and extrapolated lifespan contours | `analysis/figures/figure5/render_lifespan.py` |
| Fig. 6 | HGPS likelihood and AIC comparison | `analysis/figures/figure6_progeria/plot_fig6.py` |
| Extended Data Fig. 1 | Imposed senogenic heterogeneity | `analysis/figures/extended_data/render_senogenic_profile.py` |
| Extended Data Fig. 2 | Cleaned-cohort NHANES KM curves | `analysis/figures/extended_data/render_nhanes_survival.py` |
| Extended Data Fig. 3 | NHANES bootstrap AIC stability | `analysis/figures/extended_data/render_nhanes_aic.py` |
| Extended Data Fig. 4 | Danish historical mortality and fitted robustness | Included in `analysis/figures/figure4/render_current_history.py` |
| Extended Data Fig. 5 | Fedichev-Gruber parameter constraints | `analysis/figures/extended_data/render_fedichev_constraints.py` |
| Supplementary Fig. 1 | Upper-tail response to mean-parameter shifts | `analysis/figures/supplementary/render_parameter_shifts.py`; saved homogeneous FP sweep |
| Supplementary Fig. 2 | Gompertz slopes, heterogeneity tolerance and selection | Saved analytic tolerance table and FP grid checks; `analysis/figures/supplementary/render_gompertz_constraints.py` |
| Supplementary Fig. 3 | Fedichev-Gruber survival and mortality | `analysis/figures/supplementary/render_fedichev_minimal_model.py` |
| Supplementary Fig. 4 | Disease onset and morbidity | `analysis/figures/supplementary/render_healthspan_morbidity.py` |
| Extended Data Table 1 | Exposure-group fits, counts and AIC support | Regenerated with Extended Data Fig. 3; `results/nhanes/extended_data_table1.csv` |
| Table M1 | Sweden and NHANES SR parameter estimates and intervals | `analysis/tables/export_table_m1.py`; [table export documentation](analysis/tables/README.md) |
| Supplementary Table 1 | Absolute NHANES Kaplan-Meier summaries and bootstrap precision | `analysis/tables/export_supplementary_table1.py`; [table export documentation](analysis/tables/README.md) |

## Organization

- `analysis/figures/`: figure renderers.
- `analysis/model_fits/`: explicit optional fitting and bootstrap workflows.
- `scripts/`: saved-result verification, figure orchestration and input preparation.
- `src/senogenic_vs_robustness/`: shared solver and project helpers.
- `src/ageing_packages/`: vendored data, statistical and plotting helpers.
- `data/`: source inputs with provenance.
- `results/`: saved fits, survival curves, uncertainty records and figure source tables.
- `results/index/outputs.csv`: current manuscript artifact inventory.
- `analysis/validation/`: validation code; saved validation records are under
  `results/validation/`.
- `Figures/`: publication figure PNGs.
- `docs/figure_methods/`: analysis-specific methods.
- `tmp/`: ignored staging and optional rerun outputs.

## Data and Saved Fits

**HMD:** Sweden and Denmark deaths/person-time exposure tables are bundled.
`python3 scripts/prepare_historical_hmd.py` regenerates the exact 44,955-row
historical input. Life tables are separate inputs, not exposure substitutes.
See [HMD provenance and licensing](data/hmd/README.md).

**NHANES:** the current likelihood cohort contains 55,800 participants and
7,260 deaths. `results/nhanes/` contains the 23 group point fits, paired
bootstrap records, AIC summaries and full-cohort/group survival curves.
Routine plotting requires no participant-level input. Some groups have no
observation support at age 20; their early KM values remain undefined.
`python3 scripts/prepare_nhanes_cohort.py` rebuilds the exact fitting input from
the bundled public-use sources and verifies its checksum, without fitting.
See [input preparation](data/nhanes/README.md).
See [NHANES methods](docs/figure_methods/figure3_likelihood.md) and
[AIC and survival figures](docs/figure_methods/nhanes_aic_and_survival.md).

**HGPS:** [204 published death/censoring records](data/hgps/README.md), including
102 deaths and treatment-start censoring, feed the individual survival
likelihood. They are not a binned exposure table. `results/progeria/` contains
the ten current Q-reference-anchored model fits and numerical checks.
[Figure 6 methods](docs/figure_methods/figure6_progeria.md).

**Historical fits:** `results/historical/` contains Sweden/Denmark fitted
trajectories, covariance matrices and lifespan bands. `results/historical/fit_records/` retains
the actual point fits and 100 Swedish recovery datasets. Figures 4d, 5 and
Extended Data 4 use per-year dispersion-adjusted intervals;
Poisson recovery diagnostics are provided separately from the interval calculation.
[Historical methods](docs/figure_methods/historical_fits.md).

**Senogenic heterogeneity:** `results/senogenic_heterogeneity/` contains the
current FP profiles, selected candidates, inputs and mortality curves.
[Profile methods](docs/figure_methods/senogenic_heterogeneity_fp.md).

## Optional Expensive Computation

Full recomputation is separate from figure rendering. No heavy jobs are
automatically launched when saved results exist or an input is missing.

- [Historical fitting, covariance and recovery](analysis/model_fits/historical/README.md)
- [NHANES likelihood and paired bootstrap](analysis/model_fits/nhanes/README.md)
- [Additional optional model comparisons](analysis/model_fits/additional/README.md)
- [HGPS model fitting](analysis/model_fits/hgps/README.md)
- [Senogenic heterogeneity profiles and candidate checks](analysis/model_fits/senogenic_heterogeneity/README.md)
- [Figure 2 FP calculations](docs/figure_methods/figure2_fokker_planck.md)
- [Fedichev-Gruber figures and optional calculations](docs/figure_methods/fedichev_gruber.md)
- [Supplementary 2 FP checks](analysis/model_fits/supplementary/README.md)
- [Supplementary 4 joint first passage](docs/figure_methods/healthspan_fp.md)
- [Publication table exports](analysis/tables/README.md)

Additional optional model-fit outputs are written under `results/additional/`
when those scripts are run. Validation scripts remain under
`analysis/validation/`, while their saved records are published under
`results/validation/`.

Sweden/Denmark use the age-balanced Q estimating criterion, **not a sampling
likelihood**. NHANES and HGPS use survival likelihoods for AIC. Historical
recovery simulations are retained diagnostics, not draws defining percentile
bands or calibration of the current dispersion-adjusted intervals. Baselines
and uncertainty types are analysis-specific.

`python3 scripts/verify_repo.py` checks current saved-record integrity without
simulation: NHANES group/reference bootstrap pairing, model counts and AIC,
HGPS records and AIC, historical dispersion and the reference/profile grid,
Supplementary 2 and 4 numerical archives, and the complete figure inventory.
Numerical checks are documented in the figure methods.
Original code is available under the [MIT License](LICENSE). Third-party code
and datasets retain their respective licenses and data-use conditions.
