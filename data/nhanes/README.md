# NHANES inputs for the likelihood analysis

This directory contains public-use questionnaire and demographic XPT files,
the merged public-use mortality linkage table (`nhanes_mortality_all_years.csv`),
and the age table (`all_cohort_age_data.csv`). These are source inputs, not
newly collected participant records or restricted-access linkage data.
`SEQN` is the public NHANES survey identifier. No names or contact details are
required by this analysis.

## Official sources and reuse

Source: Centers for Disease Control and Prevention, National Center for Health
Statistics, National Health and Nutrition Examination Survey. Questionnaire
and demographic files are available from the
[NHANES data portal](https://wwwn.cdc.gov/nchs/nhanes/Default.aspx).
CDC provides [suggested citations and reuse guidance](https://wwwn.cdc.gov/nchs/nhanes/NhanesCitation.aspx)
for these federal public-use data. The repository does not assign a new license
to the source files or claim ownership of them.

The official [public mortality file directory](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/)
and [2019 public-use linked mortality description](https://www.cdc.gov/nchs/data/datalinkage/public-use-linked-mortality-file-description.pdf)
document the public linkage release. NCHS distinguishes public-use files from
restricted files accessible through its Research Data Center; this repository
does not contain the restricted linkage records.

Public-use mortality linkage applies disclosure-protection perturbation to
selected follow-up times and causes of death. The survival analysis therefore
uses released public-use follow-up, not unmodified confidential event times.
The merged CSVs are analysis inputs derived from public files, not new official
CDC releases. Complete preparation metadata for these merged inputs is
unavailable. The cohort checksum identifies the analyzed snapshot; fresh
CDC downloads may differ.

From the repository root:

```bash
python3 scripts/prepare_nhanes_cohort.py
python3 scripts/prepare_nhanes_survival.py --cohort tmp/nhanes_cohort.csv
```

The first command rebuilds `tmp/nhanes_cohort.csv` with 55,800 participants,
7,260 deaths and 23 overlapping exposure-group indicators. It checks every
group's sample and death counts and the complete CSV SHA-256 against the input
used for the archived fits:

`0fb93c36eb9ee48b1a5d5d1aab02f4e48444a4530b0ae03ba2740619cc9b0201`

The second command rebuilds the 24 delayed-entry Kaplan-Meier summaries.
The committed summaries are in `results/nhanes/survival_curves.csv`, so ordinary
figure rendering does not require rebuilding the participant-level table.
Neither command runs SR fitting, simulation or bootstrap jobs.

## Observation intervals and groups

Only linkage-eligible records are retained. Entry age uses the age table's
`age_in_years`, falling back to `age_at_screening`; exit age adds interview-based
follow-up months divided by 12. `mortstat` supplies the death indicator.
The likelihood conditions survival on entry age and treats surviving records
as right-censored. It is not a binned population-exposure likelihood.

Records at the top-coded age are excluded: entry age must be below 85 before
2007 and below 80 thereafter. Exit must exceed 20, event must be 0 or 1, and
ages must be present. Entry is then clipped to 20; zero-length intervals are
extended by half a month, matching the source analysis. Negative intervals
or duplicate public identifiers fail validation.

The script fixes group order explicitly. Group boundaries and questionnaire
transformations are in `src/ageing_packages/hetero_analysis/nhanes_analysis.py`.
Income cutpoints are constructed before the likelihood age exclusions, as in
the original analysis. Friends counts 7 and 9 remain valid; 7777 and 9999 are
missing codes.
`SDMVSTRA` and `SDMVPSU` are retained for within-wave/stratum PSU bootstrap
resampling. The point likelihood is unweighted; these are not survey-weighted
national estimates. The published cohort uses the retained exposure-group
indicators; no occupation analysis is part of the current results.

Some exposure groups have no entrants near age 20. Their survival summaries
remain missing before the first supported entry age; they must not be described
as observed survival from age 20 over that unsupported interval.

See `analysis/model_fits/nhanes/README.md` for optional expensive refits and
paired bootstrap jobs. The published figure workflow reads saved fits instead.
The table-only bootstrap exports are documented in
`analysis/tables/README.md`; they recompute Kaplan-Meier summaries and saved
parameter intervals without an SR refit.
