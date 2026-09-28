"""Compare saved joint-FP grids without running simulations or fitting."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compare(reference, candidate):
    first = json.loads((reference / 'manifest.json').read_text())
    second = json.loads((candidate / 'manifest.json').read_text())
    for key in ('baseline', 'horizon', 'disease_fraction', 'scenarios',
                'external_mortality', 'horizon_convention', 'source_sha256'):
        if first[key] != second[key]:
            raise ValueError(f'Incomparable scientific specifications: {key}')
    a = pd.read_csv(reference / 'states.csv')
    b = pd.read_csv(candidate / 'states.csv')
    sa = pd.read_csv(reference / 'summary.csv').set_index('scenario')
    sb = pd.read_csv(candidate / 'summary.csv').set_index('scenario')
    rows = []
    for scenario in first['scenarios']:
        x = a[a.scenario == scenario]
        y = b[b.scenario == scenario]
        if x.empty or y.empty:
            raise ValueError(f'Missing scenario: {scenario}')
        errors = {state: float(np.max(np.abs(
            x[state] - np.interp(x.age, y.age, y[state]))))
            for state in ('healthy', 'sick', 'dead')}
        rows.append(dict(scenario=scenario, max_absolute_state_difference=errors,
                         median_difference_percentage_points=float(100 * (
                             sb.loc[scenario, 'median_sick_life_fraction']
                             - sa.loc[scenario, 'median_sick_life_fraction']))))
    return dict(reference_grid={k: first[k] for k in ('cells', 'dt', 'nodes')},
                candidate_grid={k: second[k] for k in ('cells', 'dt', 'nodes')},
                comparison=rows,
                source_hashes={label: {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                                      for name in ('manifest.json', 'states.csv', 'summary.csv', 'sick_fraction.csv')}
                               for label, folder in [('reference', reference), ('candidate', candidate)]},
                interpretation='Numerical sensitivity, not statistical or biological uncertainty')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['comparison'], indent=2))


if __name__ == '__main__':
    main()
