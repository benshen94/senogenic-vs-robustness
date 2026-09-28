#!/usr/bin/env python3
"""Export Table M1 from saved parameters/covariance/bootstrap; no fitting."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PARAMETERS = ('eta', 'beta', 'kappa', 'Xc', 'CV', 'epsilon', 'mex')
UNITS = dict(eta='year^-2', beta='year^-1', kappa='dimensionless',
             Xc='dimensionless', CV='dimensionless', epsilon='year^-1', mex='year^-1')


def sweden_intervals(joint, dispersion):
    """Undo the baseline block's recorded Pearson multiplier, not other years'."""
    phi = next(row['variance_multiplier'] for row in dispersion if row['year'] == 2019)
    if not np.isfinite(phi) or phi <= 0:
        raise ValueError('Invalid baseline dispersion multiplier')
    covariance = np.asarray(joint['covariance'])[:4, :4] / phi
    q = np.asarray(joint['q'])[:4]
    expected = [np.log(joint['baseline']['Xc']), np.log(joint['baseline']['epsilon']),
                joint['baseline']['CV'], 1000 * joint['baseline']['mex']]
    np.testing.assert_allclose(q, expected, rtol=0, atol=1e-12)
    z = NormalDist().inv_cdf(.975)
    result = {}
    for i, name in enumerate(('Xc', 'epsilon', 'CV', 'mex')):
        low, high = q[i] + np.array([-1, 1]) * z * np.sqrt(covariance[i, i])
        if name in ('Xc', 'epsilon'):
            low, high = np.exp([low, high])
        elif name == 'mex':
            low, high = low / 1000, high / 1000
        result[name] = (low, high)
    return result, float(phi)


def export(output):
    joint_path = ROOT / 'results/historical/joint_covariance.json'
    dispersion_path = ROOT / 'results/historical/dispersion.json'
    point_path = ROOT / 'analysis/model_fits/nhanes/inputs/best_constrained_fit.json'
    joint = json.loads(joint_path.read_text())
    dispersion = json.loads(dispersion_path.read_text())
    nhanes = json.loads(point_path.read_text())['params']
    intervals, phi = sweden_intervals(joint, dispersion)
    paths = sorted((ROOT / 'results/nhanes/bootstrap/baseline_precision').glob('rep_*.json'))
    records = [json.loads(p.read_text()) for p in paths]
    if len(records) != 100 or sorted(r['replicate'] for r in records) != list(range(1, 101)):
        raise ValueError('Expected exactly 100 distinct reference-bootstrap records')
    raw = []
    for record, path in zip(records, paths):
        if record['seed'] != 20260924 or not record['success']:
            raise ValueError(f'Unexpected bootstrap status/seed: {path.name}')
        for name in PARAMETERS:
            # Table M1 uses the checked-grid profile of mex, not its fit-grid value.
            value = record['check_mex'] if name == 'mex' else record['params'][name]
            raw.append(dict(replicate=record['replicate'], parameter=name, value=value,
                            fit_grid_value=record['params'][name],
                            value_source='check_mex' if name == 'mex' else 'params.' + name,
                            source_path=str(path.relative_to(ROOT))))
    replicates = pd.DataFrame(raw)
    rows = []
    for population, params in [('Sweden 2019', joint['baseline']), ('NHANES', nhanes)]:
        for name in PARAMETERS:
            fixed = name in ('eta', 'beta', 'kappa')
            if fixed:
                low, high, method = np.nan, np.nan, 'fixed; no interval'
            elif population == 'Sweden 2019':
                low, high = intervals[name]
                method = 'unadjusted expected-bread sandwich; normal 95% on estimation scale'
            else:
                values = replicates.loc[replicates.parameter == name, 'value']
                low, high = np.quantile(values, [.025, .975], method='linear')
                method = '100 paired Rao-Wu PSU bootstrap draws; 2.5/97.5 percentiles; linear interpolation'
            scale = 'log' if name in ('Xc', 'epsilon') else ('1000*mex' if name == 'mex' else 'identity')
            rows.append(dict(population=population, parameter=name, unit=UNITS[name],
                             estimate=params[name], ci_low=low, ci_high=high, fixed=fixed,
                             interval_method=method,
                             interval_scale=scale if population == 'Sweden 2019' else 'parameter percentile',
                             pearson_dispersion_adjusted=False,
                             removed_variance_multiplier=phi if population == 'Sweden 2019' and not fixed else np.nan,
                             bootstrap_repeats=100 if population == 'NHANES' and not fixed else 0,
                             bootstrap_value_source=('check_mex (480-cell profile at saved intrinsic parameters)'
                                                     if population == 'NHANES' and name == 'mex'
                                                     else 'params (320-cell fit)' if population == 'NHANES' and not fixed else ''),
                             point_source=str((joint_path if population == 'Sweden 2019' else point_path).relative_to(ROOT))))
    sources = [joint_path, dispersion_path, point_path, Path(__file__).resolve(), *paths]
    manifest = dict(sweden_intervals='baseline block divided by recorded Pearson variance multiplier before constructing intervals',
                    sweden_removed_variance_multiplier=phi, normal_critical_value=NormalDist().inv_cdf(.975),
                    nhanes_mex_convention='check_mex, retaining fit-grid params.mex separately in replicate export',
                    bootstrap_repeats=100, numpy_version=np.__version__, pandas_version=pd.__version__,
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / 'tableM1_source.csv', index=False, float_format='%.17g')
    replicates.to_csv(output / 'tableM1_replicates.csv', index=False, float_format='%.17g')
    (output / 'tableM1_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(f'Exported Table M1 (14 parameter rows, 700 bootstrap values) to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/tables')
    export(parser.parse_args().output)
