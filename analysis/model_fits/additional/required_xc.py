"""Invert SR mean lifespan for the saved linear HMD target through 2100.

This tests the super-exponential/doubling claim; it does not assume a growth law.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[3]


def invert_mean(mean_at_xc, target, baseline_xc, lower_factor, upper_factor):
    if not np.isfinite([target, baseline_xc, lower_factor, upper_factor]).all():
        raise ValueError("Nonfinite inverse inputs")
    if baseline_xc <= 0 or not 0 < lower_factor < upper_factor:
        raise ValueError("Require positive ordered factor bracket")
    def residual(log_factor):
        value = float(mean_at_xc(baseline_xc * np.exp(log_factor))) - target
        if not np.isfinite(value):
            raise ValueError("Nonfinite mean")
        return value
    root = brentq(residual, np.log(lower_factor), np.log(upper_factor),
                  xtol=1e-7, maxiter=60)
    xc = float(baseline_xc * np.exp(root))
    achieved = float(mean_at_xc(xc))
    if abs(achieved - target) > 1e-4:
        raise ValueError("Mean target not resolved")
    return xc, achieved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--years", type=int, nargs="+", default=[2019, 2050, 2100])
    parser.add_argument("--lower-factor", type=float, default=.5)
    parser.add_argument("--upper-factor", type=float, default=4.)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-from", type=Path, help="Reuse results only if baseline, solver and target contracts match")
    args = parser.parse_args()
    if not args.run:
        parser.error("Calculation requires --run")
    dest = args.output.resolve()
    if dest.exists():
        parser.error("Choose a new output directory")
    if not 0 < args.lower_factor < args.upper_factor:
        parser.error("Require positive ordered factor bracket")
    source = ROOT / "results/historical/naive_hmd_mean_projection.json.gz"
    with gzip.open(source, "rt") as handle:
        targets = {row["year"]: row for row in json.load(handle) if row["replicate"] == 0}
    if any(year not in targets or not 2019 <= year <= 2100 for year in args.years):
        parser.error("Years must be present in the saved 2019-2100 comparator")
    subprocess.run([sys.executable, str(ROOT / "analysis/model_fits/historical/prepare_rerun.py"),
                    "--output", str(dest)], check=True)
    shutil.copy2(ROOT / "results/historical/fit_records/point/baseline.json",
                 dest / "inputs/baseline.json")
    sys.path.insert(0, str(dest))
    import pipeline as p

    baseline = dict(p.BASE)
    projection_baseline = json.loads((ROOT / "results/historical/joint_covariance.json").read_text())["baseline"]
    if baseline != projection_baseline:
        raise ValueError("Saved point and projection baselines differ; resolve provenance first")
    evaluations = []
    started = time.monotonic()
    @lru_cache(maxsize=128)
    def mean_at_xc(xc):
        prediction = p.prediction(dict(baseline, Xc=float(xc)))
        metric = prediction["contours"]["20"]
        bound = metric["mean_tail_upper_bound"]
        if (not prediction["tail_resolved"] or metric["mean_is_restricted"]
                or bound is None or bound > 1e-4):
            raise ValueError("Unresolved unrestricted mean; change bracket or investigate tail")
        evaluations.append(dict(Xc=float(xc), mean=metric["mean_attained_age"],
                                horizon=metric["mean_horizon"], tail_bound=bound))
        return metric["mean_attained_age"]

    rows = []
    contract = dict(baseline=baseline, target_source=str(source.relative_to(ROOT)),
                    fixed=["eta", "beta", "epsilon", "CV", "mex", "kappa"],
                    conditioning_age=20, bracket=[args.lower_factor, args.upper_factor],
                    solver_settings=dict(cells=320, dt=.025, threshold_nodes=31,
                                         threshold_distribution="Gaussian, discard nonpositive Gauss-Hermite nodes and renormalize",
                                         adaptive_horizons=[180, 270, 405, 610], tail_bound_tolerance=1e-4,
                                         log_factor_xtol=1e-7, mean_residual_tolerance=1e-4),
                    solver_sha256=hashlib.sha256((dest / "inputs/nonuniform_solver.py").read_bytes()).hexdigest(),
                    pipeline_sha256=hashlib.sha256((dest / "pipeline.py").read_bytes()).hexdigest(),
                    target_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    method="Inverse mean target, no assumed Xc growth law; no uncertainty",
                    warning="Doubling and super-exponential growth are hypotheses, not imposed results")
    p.save(dest / "inverse_contract.json", contract)
    cached = {}
    if args.reuse_from:
        old_contract = json.loads((args.reuse_from / "inverse_contract.json").read_text())
        for key in ("baseline", "target_sha256", "solver_sha256", "pipeline_sha256",
                    "solver_settings", "conditioning_age", "bracket"):
            if old_contract[key] != contract[key]:
                raise ValueError(f"Cached contract mismatch: {key}")
        cached = {row["year"]: row for row in json.loads((args.reuse_from / "required_xc.json").read_text())}
    reused = []
    for year in sorted(set(args.years)):
        target = targets[year]
        if year in cached:
            row = cached[year]
            if (row["target_contract"] != target or not np.isfinite([row["Xc"], row["achieved_mean"]]).all()
                    or row["Xc"] <= 0 or abs(row["achieved_mean"] - target["mean_attained_age"]) > 1e-4):
                raise ValueError("Cached result does not match target")
            rows.append(row)
            reused.append(year)
            p.save(dest / "required_xc.json", rows)
            continue
        xc, achieved = invert_mean(mean_at_xc, target["mean_attained_age"], baseline["Xc"],
                                   args.lower_factor, args.upper_factor)
        rows.append(dict(year=year, Xc=xc, factor=xc / baseline["Xc"],
                         target_mean=target["mean_attained_age"], achieved_mean=achieved,
                         target_contract=target))
        p.save(dest / "required_xc.json", rows)
    contract.update(seconds=time.monotonic() - started, evaluations=evaluations, reused_years=reused)
    p.save(dest / "inverse_contract.json", contract)
    if len(rows) > 1:
        intervals = [dict(start=a["year"], end=b["year"],
                          log_growth_per_year=float(np.log(b["Xc"] / a["Xc"]) / (b["year"] - a["year"])))
                     for a, b in zip(rows[:-1], rows[1:])]
        p.save(dest / "log_growth.json", dict(intervals=intervals,
               definition="Finite differences of log Xc per calendar year; rising values indicate accelerating proportional growth over sampled intervals",
               limitation="Conditional deterministic trajectory; no uncertainty or continuous-time curvature inference"))
    print(f"Wrote {len(rows)} inverse targets; no growth-law claim or confidence interval fitted.")


if __name__ == "__main__":
    main()
