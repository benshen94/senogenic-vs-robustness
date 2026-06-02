# Tau-senogenic heterogeneity refits

This folder contains the wide compensatory refit scripts behind Extended Data Fig. 1.

The fitted heterogeneity coordinate is

\[
\tau_{\rm sen}=\frac{\beta}{\eta}.
\]

Individual spread is imposed with

\[
\eta_i=\eta_0 e^{-v_i/2},\qquad
\beta_i=\beta_0 e^{v_i/2},\qquad
v_i\sim N(0,\sigma_v^2),
\]

so \(\tau_i\) has log-spread \(\sigma_v\).

## Scripts

- `run_tau_wide_refit_profile.py`: original wide profile scan.
- `run_tau_cv010_stress_refit.py`: targeted stronger forced-10% \(\mathrm{CV}(\tau)\) search.
- `run_tau_stress_grid_refit.py`: final all-CV stress-grid search used for the manuscript figure.

## Saved outputs

The public repo tracks the manuscript-ready saved outputs under:

- `results/tau_wide_refit_profile/`
- `results/cache/simulations/tau_wide_refit_profile/`

The most important files are:

- `results/tau_wide_refit_profile/tau_wide_refit_profile_candidates.csv`: best candidate per forced \(\tau_{\rm sen}\) spread.
- `results/tau_wide_refit_profile/tau_wide_refit_profile_all_candidates.csv`: all refined stress-grid candidates.
- `results/tau_wide_refit_profile/tau_wide_refit_profile_curves_n1000000.csv`: high-\(n\) Sweden mortality and tail curves.
- `results/tau_wide_refit_profile/tau_cv010_stress_refit_candidates.csv`: targeted 10% rescue-search candidates.

## Lightweight check

The plotting/report path can be checked without rerunning the expensive fit:

```bash
python3 analysis/model_fits/tau_wide_refit_profile/run_tau_stress_grid_refit.py --plots-only --no-parallel
```

This creates local diagnostic plots in `analysis/model_fits/tau_wide_refit_profile/plots/`. PDFs are ignored by git.
