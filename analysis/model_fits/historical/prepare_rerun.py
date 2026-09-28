#!/usr/bin/env python3
"""Stage an isolated optional historical rerun. Does not fit or submit jobs."""
import argparse
import importlib.util
from pathlib import Path
import shutil

SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'tmp/historical_rerun')
    parser.add_argument('--from-saved', action='store_true',
                        help='Include point fits, bands and recovery records for audits/resuming.')
    args = parser.parse_args()
    dest = args.output.resolve()
    if dest.exists():
        parser.error('Output already exists; choose a new directory.')
    dest.mkdir(parents=True)
    (dest/'inputs').mkdir()
    for path in SOURCE.glob('*.py'):
        if path.name != 'prepare_rerun.py':
            shutil.copy2(path, dest/path.name)
    shutil.copy2(SOURCE/'inputs/baseline.json', dest/'inputs/baseline.json')
    shutil.copy2(SOURCE/'inputs/denmark_panel_c.csv', dest/'inputs/denmark_panel_c.csv')
    shutil.copy2(ROOT/'src/senogenic_vs_robustness/sr_finite_volume.py',
                 dest/'inputs/nonuniform_solver.py')
    spec = importlib.util.spec_from_file_location('hmd_inputs', ROOT/'scripts/prepare_historical_hmd.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.build().to_csv(dest/'inputs/hmd.csv', index=False)
    if args.from_saved:
        records = ROOT/'results/historical/fit_records'
        for name in ('point', 'point_full', 'bands', 'recovery', 'denmark'):
            shutil.copytree(records/name, dest/name)
        for name in ('joint_covariance.json', 'dispersion.json'):
            shutil.copy2(ROOT/'results/historical'/name, dest/name)
        shutil.copy2(ROOT/'results/historical/denmark_joint_covariance.json',
                     dest/'denmark/joint_covariance.json')
    print(f'Staged at {dest}. No fits or cluster jobs were started.')


if __name__ == '__main__':
    main()
