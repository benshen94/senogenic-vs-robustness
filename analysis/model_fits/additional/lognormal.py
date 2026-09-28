"""Opt-in, mean-preserving lognormal SR parameter mixture (not a refit)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import logsumexp, roots_hermitenorm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from senogenic_vs_robustness.sr_finite_volume import forward


def lognormal_nodes(width, convention, nodes):
    if not np.isfinite(width) or width < 0 or nodes < 1:
        raise ValueError("Width must be finite/nonnegative and nodes positive")
    if convention not in ("cv", "log-sigma"):
        raise ValueError("Specify cv or log-sigma")
    sigma = np.sqrt(np.log1p(width**2)) if convention == "cv" else width
    if sigma == 0:
        return np.ones(1), np.ones(1)
    z, weights = roots_hermitenorm(nodes)
    factors = np.exp(sigma * z - sigma**2 / 2)
    if not np.isfinite(factors).all() or np.any(factors <= 0):
        raise ValueError("Unresolved lognormal factors")
    return factors, weights / np.sqrt(2 * np.pi)


def calculate(params, focal, width, convention, nodes, cells, dt, horizon):
    if focal not in ("eta", "beta", "epsilon", "Xc"):
        raise ValueError("Unsupported focal parameter")
    if cells < 2 or not np.isfinite([dt, horizon]).all() or dt <= 0 or horizon <= 0:
        raise ValueError("Invalid solver grid")
    steps = round(horizon / dt)
    if steps < 1 or not np.isclose(steps * dt, horizon):
        raise ValueError("Horizon must be an integer multiple of dt")
    if any(not np.isfinite(params[k]) or params[k] <= 0
           for k in ("eta", "beta", "epsilon", "Xc", "kappa")):
        raise ValueError("SR parameters must be finite and positive")
    mex = params["mex"]
    if not np.isfinite(mex) or mex < 0:
        raise ValueError("mex must be finite and nonnegative")
    factors, weights = lognormal_nodes(width, convention, nodes)
    logs = []
    for factor in factors:
        p = dict(params)
        p[focal] *= float(factor)
        logs.append(forward(p["eta"], p["beta"], p["epsilon"], p["Xc"],
                            cells, dt, steps, kappa=p["kappa"], log_output=True))
    time = np.arange(steps + 1) * dt
    survival = logsumexp(np.stack(logs) + np.log(weights)[:, None], axis=0) - mex * time
    if not np.isfinite(survival).all() or np.any(np.diff(survival) > 1e-10):
        raise ValueError("Invalid survival curve")
    return time, survival


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Explicitly enable calculation")
    parser.add_argument("--baseline", type=Path, default=ROOT / "results/historical/fit_records/point/baseline.json")
    parser.add_argument("--parameter", choices=("eta", "beta", "epsilon", "Xc"), required=True)
    parser.add_argument("--width", type=float, required=True)
    parser.add_argument("--convention", choices=("cv", "log-sigma"), required=True)
    parser.add_argument("--nodes", type=int, default=31)
    parser.add_argument("--cells", type=int, default=320)
    parser.add_argument("--dt", type=float, default=.025)
    parser.add_argument("--horizon", type=float, default=140.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.run:
        parser.error("Calculation requires --run")
    if args.output.exists():
        parser.error("Choose a new output directory")
    baseline = json.loads(args.baseline.read_text())
    params = baseline["params"]
    time, log_s = calculate(params, args.parameter, args.width, args.convention,
                            args.nodes, args.cells, args.dt, args.horizon)
    args.output.mkdir(parents=True)
    np.savetxt(args.output / "survival.csv", np.column_stack((time, log_s)),
               delimiter=",", header="age,log_survival", comments="")
    metadata = dict(params=params, parameter=args.parameter, width=args.width,
                    convention=args.convention, nodes=args.nodes, cells=args.cells,
                    dt=args.dt, horizon=args.horizon,
                    ignored_baseline_CV=params.get("CV"),
                    contract="Single focal birth-assigned lognormal mixture; other SR parameters homogeneous. Constant independent mex retained. No fitting or uncertainty.")
    (args.output / "contract.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
