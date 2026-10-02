# Single-senogenic NHANES point-fit comparison

All models use the same full-cohort anchor and individual delayed-entry death/censoring likelihood. Each single model fits one intrinsic parameter plus extrinsic mortality (k=2); each pair fits two intrinsic parameters plus extrinsic mortality (k=3). CV and kappa remain fixed. No new bootstrap was run.

Against the best of five intrinsic-parameter pairs, Xc or epsilon is competitive in 22/23 groups; eta or beta in 10/23. Eta alone is competitive in 6/23 and beta alone in 10/23.

Comparing the two single-parameter families directly, robustness is better by more than 2 AIC units in 16 groups, senogenic models in 0, and the families differ by at most 2 in 7. The median signed difference (best senogenic minus best robustness AIC) is 3.70.

Positive differences below favor robustness; negative differences favor a senogenic fit. The five-pair reference is eta+beta, Xc+eta, Xc+beta, epsilon+beta, or epsilon+eta, with mex in every model.

| Group | Xc vs pair | epsilon vs pair | eta vs pair | beta vs pair | Best senogenic minus best robustness |
|---|---:|---:|---:|---:|---:|
| diet__0 (Poor) | -1.98 | -1.95 | -2.00 | -1.99 | -0.02 |
| diet__1 (Good) | +1.90 | -0.57 | +17.65 | +12.82 | +13.39 |
| number_of_friends__0 (0 friends) | -1.73 | -1.34 | -1.97 | -1.99 | -0.26 |
| number_of_friends__1 (1+ friends) | -1.50 | -1.63 | -1.36 | -1.39 | +0.24 |
| income__0 (Q1 (Lowest)) | +5.00 | +1.18 | +18.89 | +15.49 | +14.31 |
| income__1 (Q2) | -1.38 | -1.07 | +2.96 | +0.90 | +2.28 |
| income__2 (Q3) | -1.26 | -1.73 | +2.08 | +0.94 | +2.67 |
| income__3 (Q4 (Highest)) | -0.54 | -0.98 | +22.98 | +12.44 | +13.42 |
| alcohol__0 (>4 drinks/day) | -1.66 | -1.67 | +4.14 | +2.03 | +3.70 |
| alcohol__1 (0-1 drink/day) | +10.67 | +1.08 | +71.83 | +52.75 | +51.67 |
| physical_activity__0 (No Activity) | +0.47 | -1.34 | +8.57 | +6.10 | +7.44 |
| physical_activity__1 (Some Activity) | -0.39 | +2.44 | +2.11 | -0.46 | -0.07 |
| sleep_duration__0 (1-<5 hours) | +0.03 | -1.47 | +6.29 | +4.86 | +6.33 |
| sleep_duration__1 (5-<7 hours) | +0.89 | +1.49 | -0.76 | -0.37 | -1.66 |
| sleep_duration__2 (7-<9 hours) | +1.94 | -0.69 | +21.64 | +14.88 | +15.58 |
| sleep_duration__3 (>=9 hours) | -0.14 | -0.98 | +3.39 | +2.40 | +3.38 |
| sleep_frailty__0 (Q4 (highest)) | +1.37 | -0.22 | +7.58 | +6.35 | +6.57 |
| sleep_frailty__1 (Q1 (lowest)) | -0.59 | -1.17 | +2.23 | +1.54 | +2.70 |
| church_frequency__0 (never) | -0.85 | +1.16 | -1.79 | -1.98 | -1.13 |
| church_frequency__1 (sometimes) | -1.86 | -1.64 | -1.99 | -1.97 | -0.13 |
| church_frequency__2 (weekly) | -0.64 | -1.69 | +4.90 | +3.27 | +4.96 |
| education_level__0 (no highschool) | +10.15 | +4.38 | +23.36 | +20.40 | +16.02 |
| education_level__1 (some college) | +5.71 | +0.09 | +47.17 | +33.50 | +33.41 |

## Numerical sensitivity

| Grid (cells, time step, threshold nodes) | Robustness competitive vs pairs | Senogenic competitive vs pairs |
|---|---:|---:|
| 320, 1/40 y, 31 | 22/23 | 10/23 |
| 480, 1/60 y, 41 | 22/23 | 11/23 |
| 640, 1/120 y, 51 | 22/23 | 11/23 |

All 46 new fits converged and passed checks from two nearby starts; 18 had mex=0 and none hit an intrinsic bound. There were 0 paired-model likelihood nesting flags (details in summary.json). The primary results use the same grid as the manuscript comparison. Finer-grid rows are sensitivity evaluations, not reoptimized fits.

These are conditional model-sufficiency comparisons, not evidence identifying unique biological mechanisms. The groups overlap, and these point estimates do not propagate baseline or survey-sampling uncertainty for the new senogenic models.

AIC retains mex in the parameter count at its zero boundary; regular AIC approximations are therefore imperfect for those fits. All four single models use the same parameter count and boundary convention.
