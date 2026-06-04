# SR Fit Archive

This folder contains the archived SR fits used by the current manuscript figures.

## Layout

- `index.json`: machine-readable list of retained fit records.
- `records/*.json`: fit records and copied source summaries.
- `ci/*.csv`: local curvature confidence-interval tables used for figure envelopes.
- `previews/*.png`: lightweight fit previews for quick visual auditing.

## Retained fits

- `joint2019_shared_eta_beta_epsilon_65_100_n100k`: intermediate shared USA/Sweden 2019 fit used as a reference/start for later refits.
- `joint2019_tail90_sweden_emphasis`: current Sweden-oriented 2019 baseline used by Fig. 1, Fig. 2, Fig. 4, Fig. 5, Extended Data Fig. 1, and Supplementary Fig. 4.
- `hybrid2019_swe_tail90_usa_refit`: current USA 2019 baseline used by Fig. 3 and the NHANES projection workflow.

Older exploratory, sequential, and incomplete historical fits were removed from the public archive to keep the repository aligned to the current manuscript.
