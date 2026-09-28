"""Rebuild active Swedish tables from saved adjusted covariance and bands.

No SR solver calls, fits, or metric derivatives. Input may be the public
results directory or a staged optional rerun directory.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from inference import projected_params, parameter_mapping

Z = 1.95996398454


def export_tables(source, output):
    data = json.loads((source/'joint_covariance.json').read_text())
    if 'Pearson' not in data['diagnostics']['method']:
        raise ValueError('Expected dispersion-adjusted covariance, not Poisson recovery covariance')
    q = np.asarray(data['q']); cov = np.asarray(data['covariance'])
    base = data['baseline']; years = data['years']
    rows = []
    for year in sorted([*years, 2019]):
        row = data['ratios'].get(str(year), dict(estimate=1., log_se=0.))
        est = row['estimate']; se = row['log_se']
        rows.append(dict(year=year, estimate=est, ci_low=est*np.exp(-Z*se),
                         ci_high=est*np.exp(Z*se), series='data'))
    for scenario in ('linear', 'exponential'):
        for year in [2019, *range(2020, 2101, 5)]:
            params = projected_params(q, base, years, year, scenario)
            g = parameter_mapping(q, base, years, year, scenario)[0]; g[0] -= 1
            se = np.sqrt(max(0., g@cov@g)); est = params['Xc']/base['Xc']
            rows.append(dict(year=year, estimate=est, ci_low=est*np.exp(-Z*se),
                             ci_high=est*np.exp(Z*se), series=scenario+'_extrapolation'))
    bands = source/'bands'
    if not bands.exists():
        bands = source/'fit_records/bands'
    paths = sorted(bands.glob('*.json'))
    if len(paths) != len(data['targets']):
        raise ValueError('Incomplete saved band targets')
    metrics = []
    for path in paths:
        row = json.loads(path.read_text())
        if 'Pearson' not in row['method']:
            raise ValueError(f'Expected adjusted band record: {path.name}')
        metrics.extend(dict(scenario=row['scenario'], year=row['year'], **metric)
                       for metric in row['metrics'])
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output/'sweden_threshold_fits.csv', index=False)
    pd.DataFrame(metrics).to_csv(output/'lifespan_bands.csv', index=False)
    print(f'Exported {len(rows)} threshold rows and {len(metrics)} lifespan rows; no solver calls.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export_tables(args.input, args.output)
