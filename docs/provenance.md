# Analysis provenance

Saved manifests record input checksums, model settings, random seeds and
numerical resolutions. Source hashes identify the exact artifacts used for a
recorded calculation. They are not checksums of replacement or reorganized
scripts, even when those scripts serve the same role. In particular,
`original_source_sha256`, `original_script_sha256`, `local_source_sha256`,
`fit_script_sha256` and `current_source_sha256` are run-time records, not a
live inventory of this checkout.

## Calculation entry points

| Analysis | Code |
| --- | --- |
| Historical point and joint fitting | `analysis/model_fits/historical/fitting.py`, `point_year.py`, `joint_run.py` |
| Historical uncertainty | `analysis/model_fits/historical/bands.py`, `inference.py` |
| Denmark fits | `analysis/model_fits/historical/denmark_fit.py`, `denmark_aggregate.py` |
| Response plane | `analysis/figures/steepness_longevity/response_plane.py` |
| NHANES fitting and bootstrap | `analysis/model_fits/nhanes/` |
| HGPS fitting | `analysis/model_fits/hgps/refit.py` |
| Numerical validation | `analysis/validation/validate.py` |

The [manuscript index](../results/index/outputs.csv) connects each figure and
table to its renderer and saved inputs. Analysis-specific methods describe
conditioning, parameter conventions and uncertainty calculations.

## Source data

Dataset-specific documentation records source publications, input versions and
reuse conditions. HMD and NHANES preparation scripts verify the bundled inputs
against their expected checksums. The HGPS reconstruction script checks the
published supplement and transcribed records. Digitized sibling curves are
published summary curves, not individual observations.

Historical hashes for source artifacts not bundled here remain identifiers of
those original artifacts; no identity with a current file is asserted.
