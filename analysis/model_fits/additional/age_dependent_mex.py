"""Compare supplied age-dependent extrinsic hazard with a matched constant."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.integrate import trapezoid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from senogenic_vs_robustness.sr_population import Grid, population


def read_hazard(path):
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if data.dtype.names != ("age", "hazard"):
        raise ValueError("CSV must have exactly age,hazard columns (years, year^-1)")
    data = np.atleast_1d(data)
    age, hazard = data["age"], data["hazard"]
    if (len(age) < 2 or not np.isfinite(age).all() or not np.isfinite(hazard).all()
            or age[0] != 0 or np.any(np.diff(age) <= 0) or np.any(hazard < 0)):
        raise ValueError("Require >=2 finite rows, increasing ages from zero, nonnegative hazards")
    return age, hazard


def cumulative_hazard(age, hazard, query):
    """Exact integral of piecewise-linear hazard, with no extrapolation."""
    age, hazard, query = map(np.asarray, (age, hazard, query))
    if np.any(~np.isfinite(query)) or np.any(query < age[0]) or np.any(query > age[-1]):
        raise ValueError("Requested ages outside supplied hazard support")
    slope = np.diff(hazard) / np.diff(age)
    prefix = np.r_[0., np.cumsum(np.diff(age) * (hazard[:-1] + hazard[1:]) / 2)]
    index = np.clip(np.searchsorted(age, query, side="right") - 1, 0, len(age) - 2)
    delta = query - age[index]
    return prefix[index] + hazard[index] * delta + .5 * slope[index] * delta**2


def matched_constant(age, hazard, start, end):
    if not np.isfinite([start, end]).all() or not 0 <= start < end:
        raise ValueError("Require finite 0 <= match-start < match-end")
    values = cumulative_hazard(age, hazard, np.array([start, end]))
    return float((values[1] - values[0]) / (end - start))


def curve_metrics(time, log_survival, entry):
    grid = np.r_[entry, time[time > entry]]
    conditional = np.interp(grid, time, log_survival) - np.interp(entry, time, log_survival)
    quantiles = {}
    for fraction in (.75, .5, .25):
        target = np.log(fraction)
        quantiles[str(fraction)] = (float(np.interp(target, conditional[::-1], grid[::-1]))
                                    if conditional[-1] <= target else None)
    q75, q25 = quantiles["0.75"], quantiles["0.25"]
    return dict(quantile_attained_ages=quantiles,
                interquartile_width=None if q75 is None or q25 is None else q25 - q75,
                restricted_mean_attained_age=entry + float(trapezoid(np.exp(conditional), grid)),
                restriction_age=float(time[-1]), terminal_conditional_survival=float(np.exp(conditional[-1])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--hazard-csv", type=Path, required=True)
    parser.add_argument("--match", choices=["cumulative-hazard"], required=True)
    parser.add_argument("--match-start", type=float, required=True)
    parser.add_argument("--match-end", type=float, required=True)
    parser.add_argument("--entry-age", type=float, required=True)
    parser.add_argument("--horizon", type=float, required=True)
    parser.add_argument("--baseline", type=Path, default=ROOT / "results/historical/fit_records/point/baseline.json")
    parser.add_argument("--cells", type=int, default=320)
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--dt", type=float, default=.025)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.run:
        parser.error("Calculation requires --run")
    if args.output.exists():
        parser.error("Choose a new output directory")
    if (not np.isfinite([args.dt, args.horizon, args.entry_age]).all()
            or args.dt <= 0 or not 0 <= args.entry_age < args.horizon
            or args.cells < 2 or args.nodes < 1
            or not np.isclose(round(args.horizon / args.dt) * args.dt, args.horizon)):
        parser.error("Invalid grid, horizon, or entry age")
    age, hazard = read_hazard(args.hazard_csv)
    if args.horizon > age[-1] or args.match_end > args.horizon:
        parser.error("Hazard must cover horizon; match interval must be within horizon")
    constant = matched_constant(age, hazard, args.match_start, args.match_end)
    params = json.loads(args.baseline.read_text())["params"]
    if any(not np.isfinite(params[k]) or params[k] <= 0 for k in ("eta", "beta", "epsilon", "Xc", "kappa")):
        parser.error("SR parameters must be finite and positive")
    if not np.isfinite(params["CV"]) or params["CV"] < 0:
        parser.error("Baseline threshold CV must be finite and nonnegative")
    grid = Grid(cells=args.cells, dt=args.dt, horizon=args.horizon, nodes=args.nodes)
    intrinsic = population(params, focal="Xc", cv=params["CV"], grid=grid, mex=0.)
    time, log_intrinsic = intrinsic.time, intrinsic.log_survival()
    cumulative = cumulative_hazard(age, hazard, time)
    log_age = log_intrinsic - cumulative
    log_constant = log_intrinsic - constant * time
    if not np.isfinite(log_age).all() or np.any(np.diff(log_age) > 1e-10):
        raise ValueError("Invalid survival curve")
    args.output.mkdir(parents=True)
    shutil.copy2(args.hazard_csv, args.output / "input_hazard.csv")
    np.savetxt(args.output / "comparison.csv",
               np.column_stack((time, np.interp(time, age, hazard), cumulative, log_intrinsic, log_age, log_constant)),
               delimiter=",", comments="",
               header="age,extrinsic_hazard,cumulative_hazard,log_survival_intrinsic,log_survival_age_dependent,log_survival_constant")
    contract = dict(params=params, replaced_baseline_mex=params["mex"],
                    hazard_sha256=hashlib.sha256(args.hazard_csv.read_bytes()).hexdigest(),
                    interpolation="Piecewise-linear hazard; exact segment integral; no extrapolation",
                    match=args.match, match_interval=[args.match_start, args.match_end],
                    matched_constant_hazard=constant, entry_age=args.entry_age,
                    cells=args.cells, nodes=args.nodes, dt=args.dt, horizon=args.horizon,
                    threshold_distribution="Positive-truncated Gaussian, CV before truncation; Gauss-Legendre quadrature",
                    model="Independent competing hazard: log S_total = log S_intrinsic - H_ext; baseline mex replaced, not added",
                    age_dependent=curve_metrics(time, log_age, args.entry_age),
                    constant=curve_metrics(time, log_constant, args.entry_age),
                    interpretation="Comparison for this supplied hazard and matching interval only; no fitting or uncertainty")
    (args.output / "contract.json").write_text(json.dumps(contract, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
