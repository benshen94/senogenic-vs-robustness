# Joint SR Fit Methods: USA and Sweden 2019 Period Curves

## Data

Period life table data for 2019 were loaded from the local Human Mortality
Database files using `HMD(country, "both", "period")`. The fitted hazard target
was the both-sex central death rate, `mx`, for ages 60-100.
Survival curves were read from `lx` and normalized to survival from age 30.

## Model Constraint

The stochastic repair model used fixed \(\kappa = 0.5\). The
parameters \(\eta\), \(\beta\), and \(\epsilon\) were constrained to be
identical for USA and Sweden. Country-specific variation was allowed only for
the threshold mean \(X_c\) and the Gaussian fractional heterogeneity of
\(X_c\). The country-specific external Makeham term, \(h_{ext}\), was fixed
from a Gamma-Gompertz-Makeham fit to each country's HMD hazard curve and was not
optimized as an SR parameter in this fit.

## Objective Function

For each candidate parameter vector, SR hazard and survival curves were
simulated with \(n=50,000\) particles and \(dt=0.025\) years during
optimization. The objective minimized weighted residuals over ages
60-100. Hazard residuals were computed on the log scale:

\[
r_h(a) = \log h_{SR}(a) - \log h_{HMD}(a).
\]

A mild extra weight was applied to ages \(a \ge 85\) within the old-age window.
Survival residuals over the same age window were included as a secondary
regularizer so that the fitted hazards also reproduced cumulative survival:

\[
r_S(a) = S_{SR}(a) - S_{HMD}(a).
\]

The final reported curves were re-simulated with \(n=160,000\) particles.

## Confidence Intervals

Approximate 95% confidence intervals were estimated from the local curvature of
the fitted residual surface in log2-parameter space. Around the final parameter
vector, each fitted parameter was perturbed by \(\Delta = 0.035\) in log2
units, using common random seeds for the plus and minus simulations. A numerical
Jacobian \(J\) of the residual vector was computed by central differences.
The covariance matrix was approximated as:

\[
\widehat{\mathrm{Cov}}(\theta) =
\hat\sigma^2 (J^T J)^+,
\]

where \((J^T J)^+\) is the Moore-Penrose pseudoinverse and
\(\hat\sigma^2\) is the residual variance divided by residual degrees of
freedom. Intervals were computed in log2 space and transformed back to natural
parameter units. These intervals are curvature-based uncertainty intervals for
the fitting criterion; they do not include all possible HMD sampling or model
misspecification uncertainty.

## CI Output

The numeric CI table is saved at `saved_results/joint_shared_eta_beta_epsilon_60_100_fit_2019_ci.csv`.
