"""Refine each paired NHANES bootstrap anchor on the point-fit solver grid."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from bootstrap_baseline import HERE, POINT, SEED, WeightedLikelihood, resampled_frame


def fit_one(rep: int) -> None:
    out = HERE / "bootstrap" / "baseline_precision" / f"rep_{rep:03d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print("EXISTS", out, flush=True)
        return
    coarse_path = HERE / "bootstrap" / "baseline" / f"rep_{rep:03d}.json"
    coarse = json.loads(coarse_path.read_text())
    if not coarse["success"]:
        raise RuntimeError(f"Coarse bootstrap anchor {rep} failed")
    data = resampled_frame(rep)
    like = WeightedLikelihood(data, 320, 0.025, 31)
    bounds = [(np.log(0.02), np.log(3000.0)),
              (np.log(0.002), np.log(20_000_000.0)), (0.0, 0.5)]
    starts = [coarse["params"], POINT]
    traces = []
    began = time.monotonic()
    for p in starts:
        q0 = np.array([np.log(p["Xc"]), np.log(p["epsilon"]), p["CV"]])

        def objective(q):
            return like.profile_mex(float(np.exp(q[0])), float(np.exp(q[1])),
                                    float(q[2]))[0]

        result = minimize(objective, q0, method="L-BFGS-B", jac="3-point",
                          bounds=bounds, options={"maxiter": 200, "ftol": 1e-9,
                                                   "gtol": 0.002, "maxls": 30,
                                                   "finite_diff_rel_step": 2e-5})
        x, e, c = float(np.exp(result.x[0])), float(np.exp(result.x[1])), float(result.x[2])
        nll, mex = like.profile_mex(x, e, c)
        traces.append({"Xc": x, "epsilon": e, "CV": c, "mex": mex,
                       "nll": nll, "success": bool(result.success),
                       "message": str(result.message), "nfev": int(result.nfev)})
    best = min(traces, key=lambda r: r["nll"])
    checker = WeightedLikelihood(data, 480, 1 / 60, 41)
    check_nll, check_mex = checker.profile_mex(best["Xc"], best["epsilon"], best["CV"])
    payload = {"replicate": rep, "seed": SEED,
               "method": "Rao-Wu n-1 PSU bootstrap within wave and stratum",
               "grid_fit": {"cells": 320, "dt": 0.025, "quadrature": 31},
               "grid_check": {"cells": 480, "dt": 1 / 60, "quadrature": 41},
               "params": {"eta": POINT["eta"], "beta": POINT["beta"],
                          "kappa": POINT["kappa"], "Xc": best["Xc"],
                          "epsilon": best["epsilon"], "CV": best["CV"],
                          "mex": best["mex"]},
               "fit_nll": best["nll"], "check_nll": check_nll,
               "check_mex": check_mex,
               "success": bool(best["success"]), "traces": traces,
               "weighted_n": float(data.bootstrap_weight.sum()),
               "weighted_deaths": float(np.sum(data.bootstrap_weight * data.event)),
               "seconds": time.monotonic() - began}
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False))
    tmp.replace(out)
    print("PRECISION_BASELINE", rep, best["nll"], check_nll,
          payload["success"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep", type=int, default=int(os.environ.get("LSB_JOBINDEX", "1")))
    args = parser.parse_args()
    fit_one(args.rep)
