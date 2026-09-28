# SR reference records

The analyses use the following analysis-specific reference fits.

## Current references

- `records/sweden_2019_fig2_fp_baseline.json`: frozen Figure 2 reference and provenance.
- `../historical/joint_covariance.json`: Sweden 2019 age-balanced Q reference,
  including covariance, used for historical fits and controlled SR comparisons.
- `../../analysis/model_fits/nhanes/inputs/best_constrained_fit.json`: cleaned-cohort
  NHANES individual-likelihood reference, including its high-grid diagnostics.
- `../progeria/results.json`: HGPS model-specific likelihood fits anchored to
  the Sweden reference; this is not a newly fitted universal baseline.

See [current figure methods](../../docs/figure_methods/README.md) for which
parameters vary, external mortality conventions and uncertainty definitions.
Q is an estimating criterion, not a likelihood suitable for AIC.
