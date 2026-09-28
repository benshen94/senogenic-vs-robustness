"""Opt-in Sweden epsilon+mex fits with Xc and other baseline parameters fixed."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]

# Run inside the existing isolated rerun layout so its local imports and input
# paths are unchanged. The caller's checkout and archived results stay untouched.
WORKER = """
import json
import sys
import pipeline as p
year, maxiter = map(int, sys.argv[1:3])
d, e = p.dataset(year, 0, 1)
model = p.Model()
if sys.argv[3] == 'evaluate':
    result = dict(params=p.BASE, score=p.score(model, p.BASE, d, e), fitted=False)
else:
    result = p.fit(model, p.BASE, ['epsilon'], d, e, maxiter=maxiter)
    result['fitted'] = True
result.update(year=year, free=['epsilon', 'mex'], fixed=['eta', 'beta', 'Xc', 'CV', 'kappa'],
              objective='Q = sum(d/mu + log(mu/max(d,1)) - d/max(d,1)); e>0',
              uncertainty='None: conditional point fit, not a bootstrap or mechanism identification')
p.save(p.HERE/'epsilon_result.json', result)
print(json.dumps(dict(year=year, score=result['score'], fitted=result['fitted'])))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--maxiter", type=int, default=120)
    parser.add_argument("--evaluate-baseline", action="store_true", help="Evaluate only; do not optimize")
    parser.add_argument("--output", type=Path, required=True, help="New isolated rerun directory")
    args = parser.parse_args()
    if not args.run:
        parser.error("Calculation requires --run")
    if not 1800 <= args.year <= 2019 or args.maxiter < 1:
        parser.error("Require 1800 <= year <= 2019 and positive maxiter")
    dest = args.output.resolve()
    if dest.exists():
        parser.error("Choose a new output directory")
    subprocess.run([sys.executable, str(ROOT / "analysis/model_fits/historical/prepare_rerun.py"),
                    "--output", str(dest)], check=True)
    shutil.copy2(ROOT / "results/historical/fit_records/point/baseline.json",
                 dest / "inputs/baseline.json")
    subprocess.run([sys.executable, "-c", WORKER, str(args.year), str(args.maxiter),
                    "evaluate" if args.evaluate_baseline else "fit"], cwd=dest, check=True)


if __name__ == "__main__":
    main()
