# HGPS survival records

`records.csv` contains 204 reconstructed death/censoring records from the
published Gordon supplementary tables, including 102 deaths. `death=1` denotes
death at the recorded age; `death=0` denotes right censoring. IDs and sex are
retained as source-table fields. These are reconstructed published records,
not newly collected clinical data.

The source supplement filename is `gordonlongevitymastersupplrev4clean.pdf`;
its SHA-256 is
`7c4b2a50ca70b663856b29fed53de50b1ea994f850e553024fa1302b7ca2e733`.
The supplement itself is not redistributed here. Source: Gordon et al.,
*Impact of Farnesylation Inhibitors on Survival in Hutchinson-Gilford Progeria
Syndrome*, Circulation 130, 27--34 (2014),
[doi:10.1161/CIRCULATIONAHA.113.008285](https://doi.org/10.1161/CIRCULATIONAHA.113.008285).
The [article and supplementary materials](https://pmc.ncbi.nlm.nih.gov/articles/PMC4082404/)
provide Tables S1 (censored) and S3 (deceased untreated cohort).

## Redistribution

The transcribed records are redistributed with permission obtained by Ben
Shenhar. This permission applies to the bundled transcription; it does not
relicense the source article or supplement. The repository's MIT license covers
original code, not these third-party records.

The 204 records include 161 untreated participants (102 deaths and 59 censored)
and 43 trial participants censored at treatment initiation. Thus only their
pre-treatment follow-up contributes; this is not 204 participants followed
untreated throughout. No mutation or contact fields are included.

To check the reconstruction, install Poppler (`pdftotext`) and run:

```bash
python3 scripts/reconstruct_hgps.py /path/to/gordonlongevitymastersupplrev4clean.pdf
```

The script verifies the PDF hash and checks every extracted row against the
bundled CSV. It uses the original analysis's layout-aware table extraction.
Optional `--output /path/to/new-records.csv` writes a new copy without overwriting
existing files. It does not run fitting.

`risk_audit.json` compares reconstructed risk sets with published values.
At age 10, the reconstruction gives 104 at risk versus 103 in the published
summary; this discrepancy is retained explicitly.

The current HGPS fitting input is individual follow-up with censoring. It is
not a binned deaths/person-time exposure table. Each individual's survival
contribution accounts for time at risk. HMD population exposure tables used
for the Swedish reference are a separate input.

For death indicator $d_i$ and age $t_i$, the fitted log-likelihood contribution
is $d_i\log h(t_i)+\log S(t_i)$. Censored records contribute survival through
their censoring age, not a death. The likelihood treats follow-up as starting
at birth and assumes non-informative censoring; it does not model registry
ascertainment or delayed entry. Published recruitment is retrospective, so
these are assumptions of this reanalysis, not prospective follow-up guarantees.
