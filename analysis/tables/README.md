# Publication table exports

These exporters assemble publication tables from saved fits, covariance
records and bootstrap inputs. They do not refit the SR model.

## Table M1

From the repository root:

```bash
python3 analysis/tables/export_table_m1.py --output results/tables
```

The exporter writes `tableM1_source.csv`, `tableM1_replicates.csv` and
`tableM1_manifest.json`. It exports 14 parameter rows: seven for Sweden 2019
and seven for NHANES. Sweden intervals use the saved joint covariance block,
with its recorded 2019 Pearson variance multiplier removed before normal 95%
intervals are constructed. NHANES intervals use 100 paired Rao-Wu PSU
bootstrap draws and linear 2.5th/97.5th percentiles. Fixed `eta`, `beta` and
`kappa` values have no interval. NHANES `mex` uses the saved 480-cell
check-grid profile value; the 320-cell fit-grid value remains in the replicate
file.

## Supplementary Table 1

Prepare the exact cohort input, then export the table:

```bash
python3 scripts/prepare_nhanes_cohort.py
python3 analysis/tables/export_supplementary_table1.py \
  --cohort tmp/nhanes_cohort.csv --output results/tables
```

The exporter writes the 24-row `supplementaryTable1_source.csv`, the
2,400-row `supplementaryTable1_replicates.csv`, the archive comparison and a
manifest. It uses 100 paired Rao-Wu within-wave/stratum PSU draws. The point
summaries are unweighted delayed-entry Kaplan-Meier quantities: absolute
median age and steepness, plus their ratios to the full cohort. Each standard
error is the sample standard deviation (`ddof=1`) of the valid draws for that
statistic. Unresolved quantiles are omitted separately per statistic and are
never endpoint-clamped; therefore valid bootstrap counts can differ, and no
SR refit is performed.
