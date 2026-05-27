# Fig4 New Period Projections

This folder stores reusable historical period coordinates and fitted Fig4 C/D projections for Sweden and Denmark.

For each available country-year, the saved tables contain the median lifespan and IQR steepness from age 20. Steepness is calculated as:

\[
\mathrm{steepness} = \frac{t_{50}}{t_{25} - t_{75}}
\]

Panel A uses the raw period HMD survival curve. Panel B fits a Gamma-Gompertz-Makeham model to the same period year, saves the fitted Makeham term \(m\), sets \(m = 0\), and recalculates the quantiles from the fitted survival curve.

The plotted new-panel coordinates are `x_relative_to_sr` and `y_relative_to_sr`, normalized to the central Sweden 2019 zero-\(h_{ext}\) SR baseline used in the new Figure 1E steepness-longevity plane.

Fig4C plots the fitted Sweden period Makeham term \(m\) from the Gamma-Gompertz-Makeham fit used to make Panel B. Its 95% interval is a lognormal delta-method interval from the covariance matrix returned by the GGM curve fit.

Fig4D fits each Sweden extrinsic-removed point to the Figure 1E threshold \(X_c\) response curve. The Fig4D central fitted value is the closest point on the mean response curve. The saved Fig4D 95% intervals repeat the same closest-curve fit across the raw scenario-specific curves underlying the shaded Figure 1E ribbons and take the 2.5th and 97.5th percentiles. The 1918 robustness artifact is replaced by the mean of the 1917 and 1919 fitted values.

Fig4D_extrap fits the 1980-2020 Sweden robustness trajectory and extends it to 2100 with two anchored trends:

\[
X_c(t) = X_c(2020) + s(t - 2020)
\]

and

\[
X_c(t) = X_c(2020)\exp\left(k(t - 2020)\right).
\]
