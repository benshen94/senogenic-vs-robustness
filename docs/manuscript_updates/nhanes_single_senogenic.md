# NHANES single-senogenic comparison: manuscript additions

These additions belong in `revised MS v3` of *Maximal human lifespan paper*.
The live manuscript additions and the two new table columns are marked blue.
The fits and full audit are in `results/nhanes/single_senogenic/`.

## Results: immediately after the no-high-school sentence

By comparison, a single senogenic change (η or β) was competitive in only 10 of 23 groups. Compared directly, the best single robustness fit had lower AIC than the best single senogenic fit in 17 groups, by more than two units in 16, whereas no group favored a senogenic change by that margin (Extended Data Table 1).

## Methods: NHANES exposure groups, after the bootstrap sentence

We also fitted single senogenic models (η + mₑₓ or β + mₑₓ) for each group with the same likelihood and reference. These comparisons were not bootstrapped. Evaluations on finer grids gave the same 17/23 ranking (14–15 groups favoring robustness by more than two units), and AIC comparisons are approximate where mₑₓ is estimated at zero.

## Extended Data Table 1 caption

**Extended Data Table 1 | SR model fit comparisons to NHANES exposure groups.** Xc and ε factors are relative to the full cohort, and come from separate single-parameter fits, each with free extrinsic mortality. Groups are ordered from highest to lowest Xc. Best ΔAIC compares the better robustness fit with the best two-intrinsic-parameter fit: negative values favor the robustness fit, and values ≤2 are competitive. Best senogenic ΔAIC uses the same paired reference for the better η or β fit (with free extrinsic mortality); Robustness – senogenic ΔAIC subtracts the best senogenic AIC from the best robustness AIC (negative favors robustness). The last column gives the percentage of 50 bootstrap repeats in which either robustness fit was competitive. Groups overlap; fitting and resampling details are in Methods.

## Column definitions and checks

- `delta_aic_best_senogenic` = min(AICη, AICβ) − min(AIC of the five intrinsic pairs). Ten values are ≤2.
- `delta_aic_robustness_minus_senogenic` = min(AICXc, AICε) − min(AICη, AICβ). Seventeen values are negative, sixteen are below −2, and none exceeds +2.
- All models include fitted extrinsic mortality. Each single model has k=2; each pair has k=3.
- These are primary 320-cell point comparisons. The existing bootstrap column describes robustness sufficiency only.

Rebuild the table without regenerating the figure or fitting any model:

```bash
python3 analysis/figures/extended_data/render_nhanes_aic.py --table-only
```
