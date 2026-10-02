#!/usr/bin/env python3
"""Optional joint FP workers for Supplementary Figure 5; no fitting."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from senogenic_vs_robustness.sr_joint_passage import joint_passage
from senogenic_vs_robustness.sr_population import positive_normal

SCENARIOS = {'baseline': (1., 1.), 'xc_only': (1.2, 1.), 'proportional': (1.2, 1.2)}


def setup(args):
    baseline_path = ROOT / 'results/historical/joint_covariance.json'
    baseline = json.loads(baseline_path.read_text())['baseline']
    factors, weights, _ = positive_normal(baseline['CV'], args.nodes)
    manifest = dict(backend='tagged finite-volume first passage', baseline=baseline,
                    baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                    cells=args.cells, dt=args.dt, nodes=args.nodes, horizon=160.,
                    fraction_bins=args.fraction_bins, disease_fraction=.75,
                    scenarios=SCENARIOS, external_mortality=0.,
                    source_sha256=hashlib.sha256(
                        (ROOT / 'src/senogenic_vs_robustness/sr_joint_passage.py').read_bytes()).hexdigest(),
                    horizon_convention='Alive records end at horizon; never-sick fraction is zero',
                    validation='Calculation settings only; numerical checks are recorded separately')
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return baseline, factors, weights, manifest, fingerprint


def run_node(args):
    if not args.recompute:
        raise SystemExit('Node calculations require --recompute; use saved outputs for plotting.')
    baseline, factors, weights, manifest, fingerprint = setup(args)
    if args.node is None or not 0 <= args.node < len(factors):
        raise ValueError('Node must be a zero-based quadrature index')
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / f'{args.scenario}_{args.node:03d}.npz'
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite {target}')
    death_factor, disease_factor = SCENARIOS[args.scenario]
    threshold = baseline['Xc'] * factors[args.node]
    result = joint_passage(baseline['eta'], baseline['beta'], baseline['epsilon'],
                          threshold * death_factor, .75 * threshold * disease_factor,
                          kappa=baseline['kappa'], n=args.cells, dt=args.dt,
                          horizon=160., fraction_bins=args.fraction_bins)
    np.savez_compressed(target, age=result['age'], states=result['states'],
                        fraction=result['fraction'], fraction_mass=result['fraction_mass'],
                        marginal_error=result['marginal_error'], weight=weights[args.node],
                        fingerprint=fingerprint, manifest=json.dumps(manifest, sort_keys=True))
    print(target)


def aggregate(args):
    _, factors, weights, manifest, _ = setup(args)
    source_versions = set()
    state_rows, fraction_rows, summaries = [], [], []
    for scenario in SCENARIOS:
        state = None
        pmf = None
        max_error = 0.
        for node, weight in enumerate(weights):
            path = args.output / f'{scenario}_{node:03d}.npz'
            with np.load(path, allow_pickle=False) as data:
                node_manifest = json.loads(str(data['manifest']))
                fingerprint = hashlib.sha256(json.dumps(node_manifest, sort_keys=True).encode()).hexdigest()
                if str(data['fingerprint']) != fingerprint or not np.isclose(data['weight'], weight):
                    raise ValueError(f'Configuration mismatch: {path}')
                # Covariance can change without changing this forward calculation's baseline.
                # Preserve its provenance, but require every scientific setting to match.
                source_versions.add(node_manifest['baseline_sha256'])
                provenance = {'baseline_sha256', 'validation'}
                expected = {k: v for k, v in manifest.items() if k not in provenance}
                actual = {k: v for k, v in node_manifest.items() if k not in provenance}
                if json.dumps(expected, sort_keys=True) != json.dumps(actual, sort_keys=True):
                    raise ValueError(f'Scientific configuration mismatch: {path}')
                if state is None:
                    state = np.zeros_like(data['states'])
                    pmf = np.zeros_like(data['fraction_mass'])
                    age = data['age'].copy()
                    fraction = data['fraction'].copy()
                if not np.array_equal(age, data['age']) or not np.array_equal(fraction, data['fraction']):
                    raise ValueError(f'Grid mismatch: {path}')
                state += weight * data['states']
                pmf += weight * data['fraction_mass']
                max_error = max(max_error, float(data['marginal_error']))
        if not np.isclose(pmf.sum(), 1, atol=1e-9, rtol=0):
            raise ValueError('Mixture fraction mass is not one')
        median = float(fraction[np.searchsorted(np.cumsum(pmf), .5)])
        summaries.append(dict(scenario=scenario, median_sick_life_fraction=median,
                              alive_at_horizon=float(state[-1, :2].sum()),
                              max_marginal_error=max_error))
        for t, row in zip(age, state):
            state_rows.append(dict(scenario=scenario, age=t, healthy=row[0], sick=row[1], dead=row[2]))
        for value, mass in zip(fraction, pmf):
            fraction_rows.append(dict(scenario=scenario, fraction=value, probability=mass))
    pd.DataFrame(state_rows).to_csv(args.output / 'states.csv', index=False)
    pd.DataFrame(fraction_rows).to_csv(args.output / 'sick_fraction.csv', index=False)
    pd.DataFrame(summaries).to_csv(args.output / 'summary.csv', index=False)
    manifest['node_baseline_source_sha256s'] = sorted(source_versions)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Aggregated {3 * len(factors)} node records; no solver calls.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['node', 'aggregate'])
    parser.add_argument('--scenario', choices=SCENARIOS, default='baseline')
    parser.add_argument('--node', type=int)
    parser.add_argument('--nodes', type=int, default=64)
    parser.add_argument('--cells', type=int, default=320)
    parser.add_argument('--dt', type=float, default=.025)
    parser.add_argument('--fraction-bins', type=int, default=2001)
    parser.add_argument('--output', type=Path, default=ROOT / 'tmp/supplementary4_fp')
    parser.add_argument('--recompute', action='store_true')
    args = parser.parse_args()
    if args.nodes < 2:
        raise ValueError('At least two threshold quadrature nodes required')
    (run_node if args.stage == 'node' else aggregate)(args)


if __name__ == '__main__':
    main()
