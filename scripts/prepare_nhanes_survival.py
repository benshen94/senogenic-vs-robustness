#!/usr/bin/env python3
"""Generate shareable survival summaries from the exact cleaned likelihood cohort."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

ROOT = Path(__file__).resolve().parents[1]
COHORT_SHA256 = '0fb93c36eb9ee48b1a5d5d1aab02f4e48444a4530b0ae03ba2740619cc9b0201'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'tmp/nhanes_survival_curves.csv')
    args = parser.parse_args()
    if hashlib.sha256(args.cohort.read_bytes()).hexdigest() != COHORT_SHA256:
        raise ValueError('Cohort differs from the archived likelihood input.')
    frame = pd.read_csv(args.cohort)
    groups = json.loads((ROOT/'analysis/model_fits/nhanes/inputs/groups.json').read_text())
    timeline = np.arange(20, 110.01, .25)
    rows = []
    for name in ['all', *groups]:
        subset = frame if name == 'all' else frame[frame['group_'+name] == 1]
        fit = KaplanMeierFitter().fit(subset.exit.to_numpy(float),
            event_observed=subset.event.to_numpy(bool),
            entry=subset.entry.to_numpy(float), timeline=timeline)
        survival = fit.survival_function_.iloc[:, 0].to_numpy(float)
        supported = timeline >= subset.entry.min()
        if (not np.isnan(survival[~supported]).all()
                or not np.isfinite(survival[supported]).all()
                or (np.diff(survival[supported]) > 1e-12).any()):
            raise ValueError(f'Invalid left-truncated KM survival: {name}')
        rows.extend(dict(group=name, age=age, survival=s,
                         n=len(subset), deaths=int(subset.event.sum()))
                    for age, s in zip(timeline, survival))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f'Saved 24 KM curves ({len(frame):,} participants, {int(frame.event.sum()):,} deaths). No SR fits run.')


if __name__ == '__main__':
    main()
