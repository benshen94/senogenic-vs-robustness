"""Paired PSU-bootstrap NHANES group AIC comparisons, one group per LSF task."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize

import group_fit as fit
from bootstrap_baseline import resampled_frame

HERE = Path(__file__).resolve().parent
BASELINE_FOLDER = os.environ.get("NHANES_BOOTSTRAP_BASELINE_FOLDER", "baseline")
GROUP_FOLDER = os.environ.get("NHANES_BOOTSTRAP_GROUP_FOLDER", "groups")
NAMES = list(fit.GROUPS)
MODELS = ("mex_only", "Xc", "epsilon", "eta+beta", "Xc+eta",
          "Xc+beta", "epsilon+beta", "epsilon+eta")
fit.GRID["tight"] = (480, 1 / 60, 41)


class WeightedGroupLikelihood(fit.Likelihood):
    def __init__(self, frame, level):
        super().__init__(frame, level)
        self.weight = frame.bootstrap_weight.to_numpy(float)
        self.event_weight = self.weight[self.event]
        self.followup = float(np.sum(self.weight * (self.exit - self.entry)))

    def profile_mex(self, p: dict) -> tuple[float, float]:
        log_s, hazard = self.curve(*(float(p[k]) for k in fit.PARAMETERS))
        survival_term = (np.interp(self.exit, self.grid, log_s)
                         - np.interp(self.entry, self.grid, log_s))
        event_hazard = np.interp(self.event_exit, self.midpoints, hazard)

        def derivative(m):
            return self.followup - np.sum(self.event_weight /
                 np.maximum(event_hazard + m, 1e-250))

        if derivative(0) >= 0:
            mex = 0.0
        elif derivative(fit.MEX_MAX) <= 0:
            mex = fit.MEX_MAX
        else:
            mex = float(brentq(derivative, 0, fit.MEX_MAX, xtol=1e-12))
        nll = (-np.sum(self.weight * survival_term) + mex * self.followup
               - np.sum(self.event_weight * np.log(
                   np.maximum(event_hazard + mex, 1e-250))))
        return float(nll), float(mex)


def refine_model(like, model, coarse, finished, point, audit_multistart=False):
    """Optimize on the same 320-cell grid as the point AIC comparison."""
    active = fit.MODELS[model]
    if not active:
        nll, mex = like.profile_mex(fit.BASELINE)
        return {"params": {**fit.BASELINE, "mex": mex}, "nll": nll,
                "k": 1, "success": True, "traces": [],
                "bounds": ["mex"] if mex == 0 else []}
    bounds = [tuple(np.log(fit.BOUNDS[k])) for k in active]
    candidates = [coarse["params"], point[model]["params"], fit.BASELINE]
    for key, value in finished.items():
        if set(fit.MODELS[key]).issubset(active):
            candidates.append(value["params"])
    unique = []
    for p in candidates:
        x = np.clip([np.log(p[k]) for k in active],
                    [b[0] for b in bounds], [b[1] for b in bounds])
        if not any(np.allclose(x, old, atol=1e-10, rtol=0) for old in unique):
            unique.append(x)
    best = [np.inf, None, None]
    traces = []
    began = time.monotonic()

    def objective(x):
        p = dict(fit.BASELINE)
        p.update({k: float(np.exp(v)) for k, v in zip(active, x)})
        nll, mex = like.profile_mex(p)
        if np.isfinite(nll) and nll < best[0]:
            best[:] = [nll, p, mex]
        if time.monotonic() - began > 1800:
            raise TimeoutError(f"High-grid {model} exceeded 1800 s")
        return nll

    for x in unique:
        try:
            r = minimize(objective, x, method="L-BFGS-B", jac="3-point",
                         bounds=bounds, options={"maxiter": 120, "ftol": 1e-9,
                                                  "gtol": 0.002, "maxls": 30,
                                                  "finite_diff_rel_step": 2e-5})
            traces.append({"nll": float(r.fun), "success": bool(r.success),
                           "message": str(r.message), "nfev": int(r.nfev)})
            # Coarse multi-start search precedes this high-grid refinement.
            # Audit a prespecified 5% of tasks with every high-grid start.
            if (not audit_multistart and r.success and
                    abs(float(r.fun) - best[0]) <= .005):
                break
        except TimeoutError as error:
            traces.append({"nll": float(best[0]), "success": False,
                           "message": str(error)})
            break
    if best[1] is None:
        raise RuntimeError(f"No finite 320-cell fit: {model}")
    nll, p, mex = best
    p = {**p, "mex": float(mex)}
    success = any(t["success"] and abs(t["nll"] - nll) < .005 for t in traces)
    hits = [k for k in active if min(
        abs(np.log(p[k]) - np.log(fit.BOUNDS[k][0])),
        abs(np.log(p[k]) - np.log(fit.BOUNDS[k][1]))) < .002]
    if mex < 1e-10 or mex > fit.MEX_MAX - 1e-10:
        hits.append("mex")
    return {"params": p, "nll": float(nll), "k": len(active) + 1,
            "success": bool(success), "traces": traces, "bounds": hits,
            "seconds": time.monotonic() - began,
            "audited_multistart": bool(audit_multistart),
            "planned_starts": len(unique), "run_starts": len(traces)}


def run(rep: int, name: str) -> None:
    baseline_path = HERE / "bootstrap" / BASELINE_FOLDER / f"rep_{rep:03d}.json"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Bootstrap baseline missing: {baseline_path}")
    anchor = json.loads(baseline_path.read_text())
    if not anchor["success"]:
        raise RuntimeError(f"Bootstrap baseline did not converge: replicate {rep}")
    fit.BASELINE = anchor["params"]
    dest = HERE / "bootstrap" / GROUP_FOLDER / f"rep_{rep:03d}_{name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print("EXISTS", dest, flush=True)
        return
    frame = resampled_frame(rep)
    frame = frame[frame["group_" + name] == 1].copy()
    if not np.isfinite(frame.bootstrap_weight.sum()) or frame.bootstrap_weight.sum() <= 0:
        raise RuntimeError("Invalid group bootstrap weights")
    like = WeightedGroupLikelihood(frame, "fit")
    high = WeightedGroupLikelihood(frame, "check")
    tight = WeightedGroupLikelihood(frame, "tight")
    checkpoint = dest.with_suffix(".progress.json")
    rows = json.loads(checkpoint.read_text())["models"] if checkpoint.exists() else {}
    old_path = HERE / "results_high" / f"{name}.json"
    point = json.loads(old_path.read_text())["models"] if old_path.exists() else {}
    started = time.monotonic()
    for model in MODELS:
        if model in rows and rows[model]["success"]:
            continue
        # Point fits are warm starts only. The bootstrap likelihood is always
        # reoptimized, and the resampled full-cohort anchor remains frozen.
        starts = {**point, **rows}
        coarse = fit.fit_model(like, model, starts, seconds=1800)
        audit_multistart = ((rep - 1) * len(NAMES) + NAMES.index(name)) % 20 == 0
        row = refine_model(high, model, coarse, rows, point,
                           audit_multistart=audit_multistart)
        row["coarse_nll"] = coarse["nll"]
        row["AIC"] = 2 * row["nll"] + 2 * row["k"]
        row["tight_nll"], row["tight_mex"] = tight.profile_mex(row["params"])
        row["tight_AIC"] = 2 * row["tight_nll"] + 2 * row["k"]
        rows[model] = row
        fit.save(checkpoint, {"replicate": rep, "group": name,
                              "baseline": anchor["params"], "models": rows})
        print("FIT", rep, name, model, row["nll"], row["success"], flush=True)
    result = {"replicate": rep, "group": name, "baseline": anchor["params"],
              "bootstrap_seed": anchor["seed"], "models": rows,
              "weighted_n": float(frame.bootstrap_weight.sum()),
              "weighted_deaths": float(np.sum(frame.bootstrap_weight * frame.event)),
              "optimized_grid": fit.GRID["check"],
              "validation_grid": fit.GRID["tight"],
              "seconds": time.monotonic() - started}
    fit.save(dest, result)
    checkpoint.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int,
                        default=int(os.environ.get("LSB_JOBINDEX", "1")))
    parser.add_argument("--rep", type=int)
    parser.add_argument("--group", choices=NAMES)
    args = parser.parse_args()
    rep = args.rep or ((args.index - 1) // len(NAMES) + 1)
    name = args.group or NAMES[(args.index - 1) % len(NAMES)]
    run(rep, name)
