"""Group SR likelihood comparisons anchored at the fixed-eta/beta NHANES fit.

One LSF array task fits one of the original 23 overlapping exposure groups.
No bootstrap resampling is performed. Xc heterogeneity CV remains fixed at
the full-cohort fitted value in every group and model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import logsumexp, roots_hermitenorm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "inputs"))
from nonuniform_solver import forward

BASELINE = json.loads((HERE / "inputs" / "best_constrained_fit.json").read_text())["params"]
GROUPS = json.loads((HERE / "inputs" / "groups.json").read_text())
PARAMETERS = ("eta", "beta", "epsilon", "Xc")
MODELS = {
    "mex_only": (),
    "Xc": ("Xc",),
    "epsilon": ("epsilon",),
    "eta+beta": ("eta", "beta"),
    "Xc+eta": ("Xc", "eta"),
    "Xc+beta": ("Xc", "beta"),
    "epsilon+beta": ("epsilon", "beta"),
    "epsilon+eta": ("epsilon", "eta"),
    "Xc+epsilon": ("Xc", "epsilon"),
}
BOUNDS = {"eta": (0.0005, 80000.0), "beta": (0.05, 8000000.0),
          "epsilon": (0.002, 20000000.0), "Xc": (0.02, 3000.0)}
GRID = {"screen": (96, 0.1, 15), "fit": (160, 0.05, 21),
        "check": (320, 0.025, 31)}
MEX_MAX = 0.2


def save(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


class Likelihood:
    def __init__(self, frame: pd.DataFrame, level: str):
        self.n, self.dt, self.quad = GRID[level]
        self.steps = round(110 / self.dt)
        self.grid = np.arange(self.steps + 1) * self.dt
        self.midpoints = self.grid[1:] - self.dt / 2
        self.entry = frame.entry.to_numpy(float)
        self.exit = frame.exit.to_numpy(float)
        self.event = frame.event.to_numpy(bool)
        self.event_exit = self.exit[self.event]
        self.followup = float(np.sum(self.exit - self.entry))
        self.calls = 0
        z, w = roots_hermitenorm(self.quad)
        thresholds = BASELINE["Xc"] * (1 + BASELINE["CV"] * z)
        keep = thresholds > 0
        self.z, self.w = z[keep], (w[keep] / w[keep].sum())

    @lru_cache(maxsize=12)
    def curve(self, eta: float, beta: float, epsilon: float, xc: float):
        thresholds = xc * (1 + BASELINE["CV"] * self.z)
        outputs = [forward(eta, beta, epsilon, float(x), self.n, self.dt,
                           self.steps, kappa=BASELINE["kappa"],
                           log_output=True, return_deaths=True)
                   for x in thresholds]
        component_logs = np.stack([r[0] for r in outputs])
        component_q = np.stack([r[1] for r in outputs])
        log_s = logsumexp(component_logs + np.log(self.w)[:, None], axis=0)
        # The same direct finite-volume death/person-time hazard as the new
        # full NHANES fit. Do not differentiate or log-ratio the survival curve.
        weighted_at_risk = self.w[:, None] * np.exp(component_logs[:, :-1])
        deaths = np.sum(weighted_at_risk * component_q, axis=0)
        exposure = np.sum(weighted_at_risk * self.dt *
                          (1 - 0.5 * component_q), axis=0)
        hazard = np.divide(deaths, exposure, out=np.zeros_like(deaths),
                           where=exposure > 0)
        self.calls += 1
        return log_s, hazard

    def profile_mex(self, p: dict) -> tuple[float, float]:
        log_s, hazard = self.curve(*(float(p[k]) for k in PARAMETERS))
        intrinsic_log_survival = (np.interp(self.exit, self.grid, log_s)
                                  - np.interp(self.entry, self.grid, log_s))
        event_hazard = np.interp(self.event_exit, self.midpoints, hazard)
        def derivative(m):
            return self.followup - np.sum(1 / np.maximum(event_hazard + m, 1e-250))
        if derivative(0) >= 0:
            mex = 0.0
        elif derivative(MEX_MAX) <= 0:
            mex = MEX_MAX
        else:
            mex = float(brentq(derivative, 0, MEX_MAX, xtol=1e-12))
        nll = (-np.sum(intrinsic_log_survival) + mex * self.followup
               - np.sum(np.log(np.maximum(event_hazard + mex, 1e-250))))
        return float(nll), mex


def fit_model(like: Likelihood, name: str, previous: dict,
              seconds: float = 1500) -> dict:
    active = MODELS[name]
    if not active:
        nll, mex = like.profile_mex(BASELINE)
        return {"params": {**BASELINE, "mex": mex}, "nll": nll,
                "k": 1, "success": True, "traces": [], "bounds":
                (["mex"] if mex == 0 else [])}
    bounds = [tuple(np.log(BOUNDS[k])) for k in active]
    seeds = [dict(BASELINE)]
    for key, record in previous.items():
        if record["success"] and set(MODELS[key]).issubset(active):
            seeds.append(record["params"])
    if len(active) == 1:
        for scale in (0.5, 1.5):
            q = dict(BASELINE); q[active[0]] *= scale; seeds.append(q)
    else:
        for scale in (0.4, 2.5):
            q = dict(BASELINE)
            for k in active: q[k] *= scale
            seeds.append(q)
    if name == "eta+beta":
        for eta_factor, beta_factor in ((0.4, 2.5), (2.5, 0.4)):
            q = dict(BASELINE)
            q["eta"] *= eta_factor; q["beta"] *= beta_factor
            seeds.append(q)
    unique = []
    for seed in seeds:
        x = np.clip([np.log(seed[k]) for k in active],
                    [b[0] for b in bounds], [b[1] for b in bounds])
        if not any(np.allclose(x, y, atol=1e-10, rtol=0) for y in unique):
            unique.append(x)
    start = time.monotonic()
    traces = []
    best_nll = np.inf
    best_params = None
    best_mex = None
    def objective(x):
        nonlocal best_nll, best_params, best_mex
        p = dict(BASELINE)
        p.update({k: float(np.exp(v)) for k, v in zip(active, x)})
        nll, mex = like.profile_mex(p)
        if np.isfinite(nll) and nll < best_nll:
            best_nll, best_params, best_mex = nll, p, mex
        if time.monotonic() - start > seconds:
            raise TimeoutError(f"{name} exceeded {seconds}s")
        return nll
    for x0 in unique:
        try:
            r = minimize(objective, x0, method="L-BFGS-B", jac="3-point",
                         bounds=bounds, options={"maxiter": 180, "ftol": 1e-9,
                                                  "gtol": 0.002, "maxls": 30,
                                                  "finite_diff_rel_step": 2e-5})
            traces.append({"nll": float(r.fun), "success": bool(r.success),
                           "message": str(r.message), "iterations": int(r.nit),
                           "nfev": int(r.nfev)})
        except TimeoutError as error:
            traces.append({"nll": float(best_nll), "success": False,
                           "message": str(error)})
            break
    if best_params is None:
        raise RuntimeError(f"No finite {name} likelihood")
    best_params = {**best_params, "mex": float(best_mex)}
    # A converged trace at essentially the same objective establishes the
    # reported best point even when a later start timed out.
    success = any(t["success"] and abs(t["nll"] - best_nll) <= 0.005
                  for t in traces)
    bound_hits = [k for k in active if min(
        abs(np.log(best_params[k]) - np.log(BOUNDS[k][0])),
        abs(np.log(best_params[k]) - np.log(BOUNDS[k][1]))) < 0.002]
    if best_mex < 1e-10 or best_mex > MEX_MAX - 1e-10:
        bound_hits.append("mex")
    return {"params": best_params, "nll": best_nll,
            "k": len(active) + 1, "success": bool(success),
            "traces": traces, "bounds": bound_hits,
            "seconds": time.monotonic() - start, "curve_calls": like.calls,
            "unique_starts": len(unique)}


def group_frame(name: str) -> pd.DataFrame:
    frame = pd.read_csv(HERE / "inputs" / "nhanes.csv")
    return frame[frame["group_" + name] == 1].copy()


def run_group(name: str, smoke: bool = False) -> None:
    if name not in GROUPS:
        raise KeyError(name)
    dest = HERE / "results" / (name + ("_smoke" if smoke else "") + ".json")
    if dest.exists():
        print(f"EXISTS {dest}", flush=True); return
    data = group_frame(name)
    if len(data) != GROUPS[name]["n"] or int(data.event.sum()) != GROUPS[name]["deaths"]:
        raise RuntimeError(f"Group membership mismatch: {name}")
    models = ("mex_only", "Xc", "epsilon") if smoke else tuple(MODELS)
    like = Likelihood(data, "fit")
    checker = Likelihood(data, "check")
    checkpoint = dest.with_suffix(".progress.json")
    rows = json.loads(checkpoint.read_text())["models"] if checkpoint.exists() else {}
    started = time.monotonic()
    for model in models:
        if model in rows and rows[model]["success"]:
            continue
        row = fit_model(like, model, rows, seconds=180 if smoke else 1500)
        row["AIC_fit_grid"] = 2 * row["nll"] + 2 * row["k"]
        row["check_nll"], row["check_mex"] = checker.profile_mex(row["params"])
        row["AIC"] = 2 * row["check_nll"] + 2 * row["k"]
        rows[model] = row
        save(checkpoint, {"group": name, "models": rows})
        print("FIT", name, model, f"NLL={row['nll']:.6f}",
              f"check={row['check_nll']:.6f}", "success=" + str(row["success"]),
              "seconds=" + str(round(row.get("seconds", 0))), flush=True)
    result = {"group": name, "label": GROUPS[name]["label"], "n": len(data),
              "deaths": int(data.event.sum()), "baseline": BASELINE,
              "CV_fixed": BASELINE["CV"], "kappa_fixed": BASELINE["kappa"],
              "likelihood": "Individual delayed-entry death/censoring; direct FP boundary-flux deaths/person-time hazard; profiled group mex",
              "grid": GRID, "models": rows, "seconds": time.monotonic() - started,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save(dest, result)
    checkpoint.unlink(missing_ok=True)
    print("DONE", name, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group")
    parser.add_argument("--index", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    tasks = list(GROUPS)
    index = args.index if args.index is not None else int(os.environ.get("LSB_JOBINDEX", "0"))
    if args.group is None and not 1 <= index <= len(tasks):
        parser.error("Provide --group or an index in 1..23; no fitting is started by default")
    target = args.group if args.group is not None else tasks[index - 1]
    if target not in GROUPS:
        parser.error("Unknown group")
    run_group(target, args.smoke)
