"""Compare normal and mean-preserving lognormal focal heterogeneity for Figure 2."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.integrate import trapezoid

from lognormal import ROOT, calculate
from senogenic_vs_robustness.sr_population import Grid, population

CASES = (("eta", .05), ("beta", .05), ("Xc", .20), ("epsilon", .20),
         ("Xc", .15), ("epsilon", .25))


def metrics(age, log_s):
    result = {}
    for name, fraction in (("q75", .75), ("median", .5), ("q25", .25), ("top_0.01pct", 1e-4)):
        target = np.log(fraction)
        if log_s[-1] > target:
            raise ValueError(f"Unresolved {name} at horizon {age[-1]}")
        result[name] = float(np.interp(target, log_s[::-1], age[::-1]))
    result["iqr"] = result["q25"] - result["q75"]
    result["restricted_mean"] = float(trapezoid(np.exp(log_s), age))
    result["terminal_survival"] = float(np.exp(log_s[-1]))
    for attained_age in (100, 110, 120):
        if age[-1] < attained_age:
            raise ValueError("Horizon must cover conditional survival ages")
        result[f"survival_{attained_age}_given_90"] = float(np.exp(
            np.interp(attained_age, age, log_s) - np.interp(90, age, log_s)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--cells", type=int, default=320)
    parser.add_argument("--dt", type=float, default=.025)
    parser.add_argument("--horizon", type=float, default=420.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.run:
        parser.error("Calculation requires --run")
    if args.output.exists():
        parser.error("Choose a new output directory")
    if (args.cells < 2 or args.nodes < 2 or not np.isfinite([args.dt, args.horizon]).all()
            or args.dt <= 0 or args.horizon < 120
            or not np.isclose(round(args.horizon / args.dt) * args.dt, args.horizon, rtol=0, atol=1e-10)):
        parser.error("Invalid solver grid")
    baseline_path = ROOT / "results/fits/records/sweden_2019_fig2_fp_baseline.json"
    params = dict(json.loads(baseline_path.read_text())["params"], mex=0.)
    grid = Grid(cells=args.cells, nodes=args.nodes, dt=args.dt, horizon=args.horizon)
    args.output.mkdir(parents=True)
    contract = dict(params=params, cases=[dict(parameter=k, cv=cv) for k, cv in CASES],
                    cells=args.cells, nodes=args.nodes, dt=args.dt, horizon=args.horizon,
                    normal="N(mean,(CV*mean)^2) conditional on positivity, quadrature truncated at 9 standard deviations",
                    lognormal="Arithmetic mean=baseline parameter; arithmetic CV=specified CV",
                    matching="Same untruncated normal mean/CV and lognormal arithmetic mean/CV; tiny positive-normal truncation shift is not rescaled",
                    heterogeneity="Only focal parameter heterogeneous; fitted threshold CV replaced; mex=0 as in Figure 2",
                    metrics="Unconditional quantile ages and restricted mean; conditional survival at ages 100,110,120 given age90",
                    inference="Deterministic comparisons, no uncertainty or predeclared similarity threshold",
                    source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                        baseline_path, Path(__file__), Path(__file__).with_name("lognormal.py"),
                        ROOT / "src/senogenic_vs_robustness/sr_finite_volume.py",
                        ROOT / "src/senogenic_vs_robustness/sr_population.py")})
    (args.output / "contract.json").write_text(json.dumps(contract, indent=2, allow_nan=False) + "\n")
    rows = []
    started = time.monotonic()
    for focal, cv in CASES:
        normal = population(params, focal=focal, cv=cv, grid=grid, mex=0.)
        age, log_lognormal = calculate(params, focal, cv, "cv", args.nodes, args.cells, args.dt, args.horizon)
        normal_log = normal.log_survival()
        n_metrics, l_metrics = metrics(age, normal_log), metrics(age, log_lognormal)
        rows.append(dict(parameter=focal, cv=cv, normal=n_metrics, lognormal=l_metrics,
                         lognormal_minus_normal={key: l_metrics[key] - n_metrics[key] for key in n_metrics},
                         max_absolute_survival_difference=float(np.max(np.abs(np.exp(log_lognormal) - np.exp(normal_log))))))
        (args.output / "comparison.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
        print(f"{focal} CV={cv:g}: top-0.01% age difference {l_metrics['top_0.01pct'] - n_metrics['top_0.01pct']:.6g} years", flush=True)
    contract["seconds"] = time.monotonic() - started
    (args.output / "contract.json").write_text(json.dumps(contract, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
