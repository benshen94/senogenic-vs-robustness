# Fedichev-Gruber figures

Extended Data Fig. 4 and Supplementary Fig. 3 use the Fedichev-Gruber model,
which has its own drift, moving instability threshold, noise and death rule.
All three panels/calculations use 300,000 individuals per simulated curve,
including the reference used to normalize the response plane.

## Saved-output rendering

```bash
python3 analysis/figures/extended_data/render_fedichev_constraints.py
python3 analysis/figures/supplementary/make_supp_figure3_fedichev_minimal_model.py
```

Both read CSVs and save PNGs only. Neither starts a simulation when a cache is
missing. Extended Data Fig. 4 contains two FG panels. The 54 extreme
lifespan rows and 54 response-plane rows match the research source tables.
The regenerated layout is not pixel-identical to the manually cropped and
relettered manuscript image; the data and displayed axis ranges are unchanged.

## Extended Data Fig. 4

`results/fedichev_gruber/extreme_lifespan.csv` contains parameter CVs 0--20% in
2.5-point steps. One parameter varies at a time as a positive Gaussian.
Each explicit trajectory calculation uses 300,000 individuals,
dt=0.05 years and a maximum horizon of 1,000 years. The statistic is the age
when survivors drop below the 0.01% rank. Display curves use Gaussian smoothing
with sigma 0.85 across the CV grid; both raw and smoothed columns are retained.
The plot deliberately clips some curves at age 150, as in the manuscript.

`shape_response.csv` varies six parameters by factors 0.6--1.4. Median lifespan
and median/IQR steepness are normalized to the simulated baseline. Each curve,
including the baseline, uses 300,000 individuals, dt=0.05, a 200-year horizon,
the original linspace time grid and random seed 20260604 for the full sweep.

The shape baseline uses beta-prime=0.013333333333333334, chosen to place the
deterministic instability at 120 years. The extreme-lifespan calculation uses
0.013333. Other values are epsilon0=4, gamma=1, beta=0.015, g=0.8 and D0=1.1.
`shape_reference.json` records the baseline metrics, simulation settings,
random seed and source-table checksum.

The standalone simulation script writes into a temporary
directory and does not overwrite paper results:

```bash
python3 analysis/model_fits/fedichev_gruber/recompute.py --recompute --panel extreme --output tmp/fg_extreme
python3 analysis/model_fits/fedichev_gruber/recompute.py --recompute --panel shape --output tmp/fg_shape
```

The shape calculation records its random seed. Use the same seed and environment
to reproduce the saved realization.

## Supplementary Fig. 3

The saved table is `results/tables/supp_figure3_fedichev_minimal_model_source.csv`.
It uses 300,000 individuals, dt=0.05, horizon125, seed20260604 and the same
baseline with beta-prime=0.013333. Death occurs on crossing the unstable root
or when the stability discriminant becomes nonpositive. Survival is evaluated
annually. Mortality uses annual deaths divided by the number at risk at the
interval start, followed by Gaussian smoothing (sigma1.1). Its shading is an
approximate Monte Carlo count-error band, not uncertainty in biological model
parameters. Unresolved individuals are assigned the horizon by the original
code; this convention is retained and must not be interpreted as observed death.

Only an explicit `--force-sim` reruns this calculation. This optional simulation
is not invoked by the saved-output runner.

```bash
python3 analysis/figures/supplementary/make_supp_figure3_fedichev_minimal_model.py --force-sim --n-sim 300000
```

The source table records the population size and seed. Rendering rejects a
cache with a different population size rather than silently using it.
