# Figure 6: HGPS likelihood comparison

Saved-output reproduction:

```bash
python3 analysis/figures/figure6_progeria/plot_fig6.py
```

The command reads `data/hgps/records.csv` and `results/progeria/`, and writes
`Figures/Figure6/Fig6.png`. It performs no fitting or simulation.


The frozen Swedish Q reference is included in `results/progeria/baseline.json`.
HGPS fits use the individual survival likelihood for 204 reconstructed records,
including 102 deaths and right censoring. External mortality is zero; threshold
CV, kappa, and inactive parameters are fixed. Four single-parameter and six
two-parameter models vary active means from 0.05 to 20 times the reference.

For each model, $\mathrm{AIC}=2\mathrm{NLL}+2k$, where $k$ counts active HGPS
parameters. The frozen Swedish reference is not refitted to HGPS. The Swedish
Q fitting criterion is not a sampling likelihood and is not used for AIC.

The saved Q-reference fits were calculated by WEXAC job 852212. The best-fit
resolution check was job 852363. All ten selected optimizations converged.
The best model is $X_c+\beta$, with AIC 657.15162; the best single-parameter
model has delta AIC 42.77406. The best pair is interior; the robustness-only
pair reaches the threshold lower bound. These comparisons remain conditional
on the model family, fixed reference, and parameter bounds.

The 480-cell numerical check changed best-model NLL by 0.04443 and survival
by at most 0.000332. Full archived evidence is in `best_resolution.json`.
No HGPS bootstrap or reference-uncertainty propagation was performed.
Observed-survival shading uses Greenwood log-log pointwise intervals.

The optional fit runner and resource-matched cluster submission example are
documented in `analysis/model_fits/hgps/README.md`. They use the bundled
finite-volume solver and original starting fits. They are excluded from normal
figure rendering. Bibliographic details, record extraction, censoring at
treatment initiation, and likelihood assumptions are documented in
`data/hgps/README.md`. Run `scripts/reconstruct_hgps.py` against the published
supplement to verify all bundled records without rerunning any fit.
