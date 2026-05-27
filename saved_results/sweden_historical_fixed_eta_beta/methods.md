# Sweden Historical SR Fits With Fixed Eta and Beta

## Data

The fits used local Human Mortality Database life tables for Sweden, both sexes.
The fitted targets were Sweden cohort 1900, Sweden cohort 1920, and Sweden
period 1900. Hazard targets used central death rates, \(m_x\), over ages
65-100. Survival curves used \(l_x\), normalized to
survival from age 30.

## Model Constraint

The stochastic repair model fixed \(\eta = 0.5868368257640714\),
\(\beta = 57.87173772073557\), and \(\kappa = 0.5\) from the
2019 Sweden-tail/USA-refit model. For each Sweden historical target, the fitted
parameters were \(\epsilon\), \(X_c\), fractional Gaussian heterogeneity in
\(X_c\), and the external mortality term \(h_{ext}\).

## Objective

During optimization, each candidate parameter vector was simulated with
\(n=100,000\) particles and \(dt=0.025\) years. The objective minimized
weighted residuals over ages 65-100. Hazard residuals were
computed on the log scale:

\[
r_h(a) = \log h_{SR}(a) - \log h_{HMD}(a).
\]

Additional hazard weight was applied for ages \(a \ge 80\), with stronger
weight for ages \(a \ge 90\). Survival residuals over the same age window
were included as a secondary regularizer:

\[
r_S(a) = S_{SR}(a) - S_{HMD}(a).
\]

Final reported curves were re-simulated with \(n=250,000\) particles.

## Confidence Intervals

Approximate 95% confidence intervals were estimated from local curvature of the
residual surface in log2-parameter space. Each fitted parameter was perturbed by
\(\Delta = 0.035\) in log2 units with common random seeds, and a numerical
Jacobian was computed by central differences. The covariance approximation was:

\[
\widehat{\mathrm{Cov}}(\theta) =
\hat\sigma^2 (J^T J)^+.
\]

These are fitting-criterion uncertainty intervals and do not include all
possible HMD sampling or model misspecification uncertainty.
