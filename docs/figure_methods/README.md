# Figure methods dossier

This folder is for detailed, code-grounded methods notes for each manuscript figure and supplementary figure. The goal is to make the public repository useful to a reviewer who wants to understand exactly how every plotted quantity was produced, not only rerun the scripts.

## File naming

Use one Markdown file per figure:

```text
docs/figure_methods/figure_1.md
docs/figure_methods/figure_2.md
docs/figure_methods/figure_3.md
docs/figure_methods/figure_4.md
docs/figure_methods/figure_5_progeria.md
docs/figure_methods/supplementary_figure_1.md
docs/figure_methods/supplementary_model_comparison.md
docs/figure_methods/supplementary_nhanes_exposure_groups.md
```

## Required sections for each figure

Each figure methods file should include:

1. **Figure scope.** What biological or modeling claim the figure supports.
2. **Panel inventory.** One row per panel with output file, generating script, source data, and cached inputs.
3. **Input data provenance.** Raw/public data sources, transformed summaries, excluded or nonredistributable data, and where each lives in `data/` or `results/`.
4. **Model or statistic.** Exact model, fitted parameters, constraints, objective functions, normalization, and any fixed constants.
5. **Uncertainty and bands.** What each shaded band, interval, ribbon, or bootstrap region means.
6. **Reproduction command.** Exact command from repo root.
7. **Expected outputs.** PDF/PNG/table files produced by the command.
8. **Validation checks.** What was compared against HMD/NHANES/progeria/source tables, plus any smoke-test or cached-run check.
9. **Known caveats.** Expensive simulations, cached artifacts, numerical tolerances, visual-only schematics, or manuscript-facing simplifications.

## Template

```markdown
# Figure X methods

## Figure scope

Short claim-level summary.

## Panel inventory

| Panel | Output | Script | Main inputs | Notes |
| --- | --- | --- | --- | --- |
| A | `Figures/...` | `analysis/...` | `data/...`, `results/...` | ... |

## Input data provenance

- ...

## Model or statistic

- ...

## Uncertainty and bands

- ...

## Reproduction command

```bash
python3 scripts/reproduce_figures.py --set all
```

## Expected outputs

- ...

## Validation checks

- ...

## Known caveats

- ...
```

## Writing rule

Do not guess from memory. For every figure methods file, verify wording against the current scripts, fit records, cached tables, and `docs/methods_log.md`.
