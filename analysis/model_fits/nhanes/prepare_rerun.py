"""Stage optional expensive NHANES refits outside the archived result bundle."""
import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True,
                        help='Cleaned NHANES input CSV matching the archived manifest')
    parser.add_argument('--fresh-point-fits', action='store_true',
                        help='Do not copy saved high-grid fits; run initial and refinement workers first')
    parser.add_argument('--output', type=Path,
                        help='New staging directory; defaults to tmp/nhanes_refit')
    args = parser.parse_args()
    import hashlib
    expected = '0fb93c36eb9ee48b1a5d5d1aab02f4e48444a4530b0ae03ba2740619cc9b0201'
    if hashlib.sha256(args.cohort.read_bytes()).hexdigest() != expected:
        raise ValueError('Cohort does not match the archived fitting input')
    source = Path(__file__).resolve().parent
    root = source.parents[2]
    target = args.output if args.output is not None else root / 'tmp/nhanes_refit'
    target.mkdir(parents=True, exist_ok=False)
    for name in ('bootstrap_baseline.py', 'bootstrap_baseline_precision.py',
                 'bootstrap_groups.py', 'group_fit.py', 'refit_fixed_eta_beta.py',
                 'refine_high_grid.py', 'validate_grid640.py', 'audit_grid640.py',
                 'repair_precision_group_fit.py'):
        shutil.copy2(source / name, target / name)
    shutil.copytree(source / 'inputs', target / 'inputs',
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(args.cohort, target / 'inputs/nhanes.csv')
    shutil.copy2(args.cohort, target / 'nhanes.csv')
    if not args.fresh_point_fits:
        shutil.copytree(root / 'results/nhanes/results_high', target / 'results_high')
    print(target)


if __name__ == '__main__':
    main()
