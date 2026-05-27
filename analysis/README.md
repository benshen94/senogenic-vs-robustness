# Analysis scripts

This folder contains runnable scripts used to reproduce the manuscript analyses.

- `figures/`: main and supplementary figure-generation scripts.
- `model_fits/`: scripts that generated or audit the archived SR/HMD/progeria fits.
- `nhanes/`: NHANES exposure-group data preparation and sample summaries.
- `quality_checks/`: focused checks and diagnostic plots that support specific figure choices.

Most users should start from the repository root with:

```bash
python3 scripts/verify_repo.py
python3 scripts/reproduce_figures.py --set main
```
