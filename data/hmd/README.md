# Human Mortality Database inputs

Source: HMD. Human Mortality Database. Max Planck Institute for Demographic
Research (Germany), University of California, Berkeley (USA), and French
Institute for Demographic Studies (France). Available at
[mortality.org](https://www.mortality.org/).

HMD-produced estimates are distributed under
[CC BY 4.0, with attribution](https://www.mortality.org/Data/UserAgreement).
This does not extend to original national-source files under the HMD site's
separate "Input Data" section. The files here are HMD processed period tables
and life tables, not those national-source input files. They retain original
headers; no data values have been edited. Newer HMD releases may differ.

## Deaths and exposure used for current historical fits

| Files | Source snapshot | Use |
| --- | --- | --- |
| `SWE_Deaths_1x1.txt`, `SWE_Exposures_1x1.txt` | Archived Sweden analysis inputs from 17 September 2026; exact download time not recorded | Swedish reference and historical fits |
| `DNK_Deaths_1x1_20260923.txt`, `DNK_Exposures_1x1_20260923.txt` | Downloads snapshot dated 23 September 2026 | Denmark historical fits |

Deaths are HMD age-year death counts (fractional values are retained).
Exposure is person-years at risk, not population headcount or life-table
survivors. The fitting input uses the Total (both-sex) columns. Sweden
1800--2019 and Denmark 1835--2019 are retained in the combined input; individual
fit scripts further select years and ages 20--109. The 110+ open interval is
identified explicitly and is not silently treated as a closed one-year bin.

```bash
python3 scripts/prepare_historical_hmd.py
```

This checks the four SHA-256 hashes and creates `tmp/historical_inputs/hmd.csv`.
All 44,955 rows match the archived Sweden and Denmark fitting input exactly.
It performs no fitting. The source USA deaths/exposure files in Downloads are
not added here merely because they exist: they are not required by this
historical Sweden/Denmark pipeline.

## Existing life table

The repository retains one HMD period life table:
`mortality.org_File_GetDocument_hmd.v6_SWE_STATS_bltper_1x1.txt`. It is a
Swedish life table and is distinct from the deaths/exposure likelihood inputs.
No HMD cohort life table or Danish/US life-table file is included. The original
header remains the source version record; the exact historical download date
was not recorded in this repository.
