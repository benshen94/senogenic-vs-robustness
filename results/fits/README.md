# Thresholds Noise Fit Archive

Created UTC: `2026-05-14T06:35:18Z`

This folder collects the SR fits produced during the USA/Sweden fitting session. Each fit has a short reference name, a JSON record, and a PNG preview when a completed plot exists.

## Layout

- `index.json`: machine-readable table of all archived records.
- `fit_explorer.html`: double-clickable local explorer for browsing fits, PNGs, fitted targets, parameters, metrics, CIs, and serial names.
- `build_fit_explorer.py`: small generator for rebuilding `fit_explorer.html` after new records are added.
- `records/*.json`: one JSON record per fit, including labels, descriptions, constraints, source files, fitted targets, fitted parameters, metrics, and CI rows when available.
- `pngs/*.png`: copied plot previews showing the fitted curves.
- `ci/*.csv`: copied CI tables when available.

## Fits

- `joint2019_initial_shared_eta_beta`: 2019 initial joint USA/SWE. Status: `completed`. PNG: `pngs/joint2019_initial_shared_eta_beta.png`
- `seq2019_usa_first_then_sweden_70_100`: 2019 sequential 70-100. Status: `completed`. PNG: `pngs/seq2019_usa_first_then_sweden_70_100.png`
- `joint2019_shared_eta_beta_epsilon_60_100`: 2019 shared eta/beta/epsilon 60-100. Status: `completed`. PNG: `pngs/joint2019_shared_eta_beta_epsilon_60_100.png`
- `joint2019_shared_eta_beta_epsilon_65_100_n100k`: 2019 shared eta/beta/epsilon 65-100 n100k. Status: `completed`. PNG: `pngs/joint2019_shared_eta_beta_epsilon_65_100_n100k.png`
- `joint2019_tail90_sweden_emphasis`: 2019 tail90 Sweden emphasis. Status: `completed`. PNG: `pngs/joint2019_tail90_sweden_emphasis.png`
- `hybrid2019_swe_tail90_usa_refit`: 2019 hybrid SWE tail90 plus USA refit. Status: `completed`. PNG: `pngs/hybrid2019_swe_tail90_usa_refit.png`
- `swehist_cohort1900_fixed_eta_beta`: SWE cohort 1900 fixed eta/beta. Status: `completed`. PNG: `pngs/swehist_cohort1900_fixed_eta_beta.png`
- `swehist_cohort1920_fixed_eta_beta`: SWE cohort 1920 fixed eta/beta. Status: `completed`. PNG: `pngs/swehist_cohort1920_fixed_eta_beta.png`
- `swehist_period1900_fixed_eta_beta_attempt`: SWE period 1900 fixed eta/beta attempt. Status: `attempted_not_completed`. PNG: `no PNG; run did not complete`
