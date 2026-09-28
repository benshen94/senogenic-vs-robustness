#!/usr/bin/env python3
"""Audit current saved fit records without fitting or population simulation.

This checks archive integrity, not solver convergence or publication completeness.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
MODELS = {'mex_only': 1, 'Xc': 2, 'epsilon': 2, 'eta+beta': 3,
          'Xc+eta': 3, 'Xc+beta': 3, 'epsilon+beta': 3, 'epsilon+eta': 3}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def check_models(record, label, point=False):
    # Point files also retain the diagnostic two-robustness-parameter fit.
    expected = {**MODELS, 'Xc+epsilon': 3} if point else MODELS
    require(set(record['models']) == set(expected), f'{label}: model inventory')
    for name, fit in record['models'].items():
        require(fit['k'] == expected[name], f'{label}/{name}: parameter count')
        for nll, aic in [('nll', 'AIC'), ('tight_nll', 'tight_AIC')]:
            require(np.isfinite(fit[nll]), f'{label}/{name}: nonfinite {nll}')
            require(np.isclose(fit[aic], 2 * fit[nll] + 2 * fit['k'],
                               rtol=0, atol=1e-8), f'{label}/{name}: {aic}')


def check_nhanes():
    folder = RESULTS / 'nhanes'
    expected = read(ROOT / 'analysis/model_fits/nhanes/inputs/groups.json')
    point = [read(p) for p in sorted((folder / 'results_high').glob('*.json'))]
    require(len(point) == 23 and {p['group'] for p in point} == set(expected),
            'NHANES point-fit inventory')
    for record in point:
        group = record['group']
        require(record['n'] == expected[group]['n']
                and record['deaths'] == expected[group]['deaths'], f'{group}: counts')
        check_models(record, group, point=True)
    baseline_records = [read(p) for p in (folder / 'bootstrap/baseline_precision').glob('*.json')]
    baselines = {p['replicate']: p['params'] for p in baseline_records}
    require(len(baseline_records) == 100 and set(baselines) == set(range(1, 101)),
            'NHANES baseline replicate inventory')
    pairs = set()
    files = list((folder / 'bootstrap/groups_precision').glob('*.json'))
    require(len(files) == 1150, 'NHANES group replicate inventory')
    for path in files:
        record = read(path)
        key = (record['replicate'], record['group'])
        require(key not in pairs, f'Duplicate paired record: {key}')
        pairs.add(key)
        require(record['baseline'] == baselines[record['replicate']],
                f'{key}: group/reference pairing mismatch')
        check_models(record, str(key))
    require(pairs == {(rep, group) for rep in range(1, 51) for group in expected},
            'NHANES missing or unexpected replicate/group pair')
    curves = pd.read_csv(folder / 'survival_curves.csv')
    require(set(curves.group) == {'all', *expected}, 'NHANES curve inventory')
    for name, group in curves.groupby('group', sort=False):
        finite = group.survival.dropna().to_numpy()
        require(len(group) == 361 and len(finite) > 0, f'{name}: curve length')
        require(np.isfinite(finite).all() and ((finite >= 0) & (finite <= 1)).all()
                and (np.diff(finite) <= 1e-12).all(), f'{name}: invalid survival')
        require(not group.survival.loc[group.survival.first_valid_index():].isna().any(),
                f'{name}: interior missing survival')
    print('ok NHANES: 23 point fits, 100 references, 1,150 paired records, 24 curves')


def check_hgps():
    records = pd.read_csv(ROOT / 'data/hgps/records.csv')
    require(len(records) == 204 and int(records.death.sum()) == 102, 'HGPS counts')
    models = read(RESULTS / 'progeria/results.json')['models']
    require(len(models) == 10 and len({m['model'] for m in models}) == 10,
            'HGPS model inventory')
    for model in models:
        require(np.isclose(model['aic'], 2 * model['nll'] + 2 * len(model['active']),
                           rtol=0, atol=1e-8), f"HGPS {model['model']}: AIC")
    require(min(models, key=lambda m: m['aic'])['model'] == 'Xc+beta',
            'HGPS archived optimum changed')
    print('ok HGPS: 204 records, ten likelihood models and AIC accounting')


def check_historical_reference():
    path = RESULTS / 'historical/joint_covariance.json'
    require(hashlib.sha256(path.read_bytes()).hexdigest()
            == '7147df0fef19eccdcc178938305f8015cd4dd96a72b1718da4a04b6dfa2e586c',
            'Historical covariance/reference differs from archived source')
    baseline = read(path)['baseline']
    require(baseline['eta'] == .59 and baseline['beta'] == 57.9,
            'Historical baseline fixed parameters')
    sw = np.asarray(read(path)['covariance'])
    dk = np.asarray(read(RESULTS / 'historical/denmark_joint_covariance.json')['covariance'])
    require(np.array_equal(sw[:4, :4], dk[:4, :4]),
            'Denmark does not use the adjusted Swedish-reference covariance')
    for row in read(RESULTS / 'historical/dispersion.json'):
        require(np.isclose(row['phi'], row['pearson'] / row['df'])
                and row['variance_multiplier'] == max(1., row['phi']),
                'Historical year dispersion differs from Pearson scaling')
    bands = pd.read_csv(RESULTS / 'historical/lifespan_bands.csv')
    require(np.allclose(bands.ci_low, bands.estimate - 1.95996398454 * bands.se)
            and np.allclose(bands.ci_high, bands.estimate + 1.95996398454 * bands.se),
            'Historical lifespan bands do not use the current normal cutoff')
    profile = pd.read_csv(RESULTS / 'senogenic_heterogeneity/profile_summary.csv')
    require(np.allclose(profile.tau_cv, [0, .025, .04, .05, .075, .1, .15, .2, .25]),
            'Senogenic profile grid')
    require(np.isfinite(profile[['score', 'refine_score']].to_numpy()).all(),
            'Senogenic profile nonfinite objectives')
    print('ok historical Q reference checksum and nine-point senogenic profile')


def check_supplementary1():
    folder = RESULTS / 'supplementary1_fp'
    checks = read(folder / 'si_checks.json')
    require(checks['baseline'] == read(RESULTS / 'historical/joint_covariance.json')['baseline'],
            'Supplementary 1 baseline differs from historical reference')
    curves = pd.read_csv(folder / 'si_hazards.csv')
    scenarios = {'homogeneous', 'Xc', 'eta', 'beta'}
    grids = {'production', 'quadrature', 'refined'}
    require(set(zip(curves.grid, curves.scenario))
            == {(grid, scenario) for grid in grids for scenario in scenarios},
            'Supplementary 1 grid/scenario inventory')
    require(not curves.duplicated(['grid', 'scenario', 'age']).any(),
            'Supplementary 1 duplicate ages')
    for key, group in curves.groupby(['grid', 'scenario']):
        require(np.array_equal(group.age, np.arange(20, 255) + .5),
                f'Supplementary 1 age grid: {key}')
        require(np.isfinite(group.mortality).all() and (group.mortality > 0).all(),
                f'Supplementary 1 invalid mortality: {key}')
    plotted = curves[curves.age <= 120].pivot(
        index=['scenario', 'age'], columns='grid', values='mortality')
    discrepancy = (plotted.production / plotted.refined - 1).abs().max()
    require(discrepancy < .012, 'Supplementary 1 archived grid discrepancy exceeds 1.2%')
    means = pd.read_csv(folder / 'si_death_bin_means.csv')
    require(set(means.parameter) == {'eta', 'beta'}, 'Supplementary 1 death-bin parameters')
    require(not means.duplicated(['parameter', 'lifespan_midpoint']).any(),
            'Supplementary 1 duplicate death bins')
    values = means[['mean_parameter', 'death_probability']].to_numpy()
    require(np.isfinite(values).all() and (values > 0).all()
            and (means.death_probability >= 25e-6).all()
            and (means.death_probability <= 1).all(),
            'Supplementary 1 invalid conditional means or bin masses')
    print(f'ok Supplementary 1: 12 saved curves; maximum grid difference {100*discrepancy:.3f}%')


def check_supplementary4():
    folder = RESULTS / 'supplementary4_fp'
    manifest = read(folder / 'manifest.json')
    require(manifest['baseline'] == read(RESULTS / 'historical/joint_covariance.json')['baseline'],
            'Supplementary 4 baseline differs from historical reference')
    require((manifest['cells'], manifest['nodes']) == (480, 96)
            and np.isclose(manifest['dt'], 1 / 60), 'Supplementary 4 adopted grid')
    states = pd.read_csv(folder / 'states.csv')
    fractions = pd.read_csv(folder / 'sick_fraction.csv')
    summary = pd.read_csv(folder / 'summary.csv').set_index('scenario')
    expected = {'baseline': .1035, 'xc_only': .1635, 'proportional': .0725}
    require(set(summary.index) == set(expected), 'Supplementary 4 scenario inventory')
    for scenario, median in expected.items():
        state = states.loc[states.scenario == scenario]
        values = state[['healthy', 'sick', 'dead']].to_numpy()
        require(len(state) == 9601 and np.isfinite(values).all()
                and (values >= -1e-12).all() and np.allclose(values.sum(axis=1), 1, atol=1e-9),
                f'{scenario}: invalid state mass')
        pmf = fractions.loc[fractions.scenario == scenario].sort_values('fraction')
        require(len(pmf) == 2001 and (pmf.probability >= 0).all()
                and np.isclose(pmf.probability.sum(), 1, atol=1e-9),
                f'{scenario}: invalid sick-fraction mass')
        reconstructed = pmf.fraction.iloc[np.searchsorted(pmf.probability.cumsum(), .5)]
        require(np.isclose(reconstructed, median, atol=1e-12)
                and np.isclose(summary.loc[scenario, 'median_sick_life_fraction'], median, atol=1e-12),
                f'{scenario}: sick-fraction median differs')
    report = read(folder / 'validation/production_to_refined.json')
    for row in report['comparison']:
        require(abs(row['median_difference_percentage_points']) <= .050001
                and max(row['max_absolute_state_difference'].values()) <= .001725,
                'Supplementary 4 numerical refinement differs')
    for label, directory in [('reference', folder / 'validation/production'), ('candidate', folder)]:
        for name, digest in report['source_hashes'][label].items():
            require(hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest,
                    f'Supplementary 4 comparison checksum: {label}/{name}')
    print('ok Supplementary 4: joint distributions, state mass, medians and grid comparison')


def check_figure_index():
    index = pd.read_csv(RESULTS / 'index/outputs.csv', keep_default_na=False)
    expected = ({f'figure{i}' for i in range(1, 7)}
                | {f'extended_data_figure{i}' for i in range(1, 5)}
                | {f'supplementary_figure{i}' for i in range(1, 5)}
                | {'extended_data_table1', 'table_m1', 'supplementary_table1'})
    require(len(index) == len(expected) and set(index.task) == expected,
            'Current manuscript output inventory differs')
    for row in index.itertuples():
        require(row.status in {'cached', 'preserved_artwork', 'pending'},
                f'{row.task}: unknown status')
        require((ROOT / row.source_script).is_file(), f'{row.task}: missing renderer')
        for source in row.input_paths.split('; '):
            require((ROOT / source).exists(), f'{row.task}: missing source {source}')
        if row.status != 'pending':
            require(bool(row.path) and (ROOT / row.path).is_file(),
                    f'{row.task}: missing output')
        else:
            require(not row.path, f'{row.task}: pending FP result must not point to a legacy image')
    pending = index.loc[index.status == 'pending', 'task'].tolist()
    if not pending:
        expected_pngs = {str(Path(path)) for path in index.loc[index.artifact_type == 'figure', 'path']}
        actual_pngs = {str(path.relative_to(ROOT)) for path in (ROOT / 'Figures').rglob('*.png')}
        require(actual_pngs == expected_pngs, 'Unindexed or missing manuscript PNG files')
    print('ok manuscript index: 14 figures and three tables; pending:', ', '.join(pending))


def main():
    check_nhanes()
    check_hgps()
    check_historical_reference()
    check_supplementary1()
    check_supplementary4()
    check_figure_index()
    print('Saved-record checks passed; no fits or simulations run.')
    print('These are integrity checks, not independent validation of model assumptions.')


if __name__ == '__main__':
    main()
