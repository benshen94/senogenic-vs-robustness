"""Refine all group models at the common 320-cell grid, then check at 480 cells."""
from __future__ import annotations
import json
import os
import time

import numpy as np
from scipy.optimize import minimize

from group_fit import HERE, BASELINE, GROUPS, MODELS, BOUNDS, GRID, Likelihood, group_frame, save

GRID["tight"] = (480, 1 / 60, 41)


def refine(like: Likelihood, name: str, first: dict, finished: dict) -> dict:
    active = MODELS[name]
    if not active:
        nll, mex = like.profile_mex(BASELINE)
        return {"params": {**BASELINE, "mex": mex}, "nll": nll, "k": 1,
                "success": True, "traces": [], "bounds": ["mex"] if mex == 0 else []}
    bounds = [tuple(np.log(BOUNDS[k])) for k in active]
    seed = first["params"]
    candidates = [seed]
    for lower_name, lower in finished.items():
        if set(MODELS[lower_name]).issubset(active):
            candidates.append(lower["params"])
    if name == "eta+beta":
        candidates.append(BASELINE)
    unique = []
    for p in candidates:
        x = np.clip([np.log(p[k]) for k in active],
                    [b[0] for b in bounds], [b[1] for b in bounds])
        if not any(np.allclose(x, y, atol=1e-10, rtol=0) for y in unique):
            unique.append(x)
    best = (np.inf, None, None)
    traces = []
    started = time.monotonic()
    def objective(x):
        nonlocal best
        p = dict(BASELINE)
        p.update({k: float(np.exp(v)) for k, v in zip(active, x)})
        nll, mex = like.profile_mex(p)
        if np.isfinite(nll) and nll < best[0]:
            best = (nll, p, mex)
        if time.monotonic() - started > 1200:
            raise TimeoutError("High-grid model exceeded 1200s")
        return nll
    for x in unique:
        try:
            r = minimize(objective, x, method="L-BFGS-B", jac="3-point",
                         bounds=bounds, options={"maxiter": 120, "ftol": 1e-9,
                                                  "gtol": 0.002, "maxls": 30,
                                                  "finite_diff_rel_step": 2e-5})
            traces.append({"nll": float(r.fun), "success": bool(r.success),
                           "message": str(r.message), "nit": int(r.nit),
                           "nfev": int(r.nfev)})
        except TimeoutError as error:
            traces.append({"nll": float(best[0]), "success": False,
                           "message": str(error)})
            break
    if best[1] is None:
        raise RuntimeError(f"No finite high-grid fit for {name}")
    nll, p, mex = best
    p = {**p, "mex": float(mex)}
    success = any(t["success"] and abs(t["nll"] - nll) < 0.005 for t in traces)
    hits = [k for k in active if min(
        abs(np.log(p[k]) - np.log(BOUNDS[k][0])),
        abs(np.log(p[k]) - np.log(BOUNDS[k][1]))) < 0.002]
    if mex < 1e-10 or mex > 0.2 - 1e-10: hits.append("mex")
    return {"params": p, "nll": float(nll), "k": len(active) + 1,
            "success": bool(success), "traces": traces, "bounds": hits,
            "seconds": time.monotonic() - started}


def run(name: str):
    dest = HERE / "results_high" / (name + ".json")
    if dest.exists():
        print("EXISTS", name, flush=True); return
    original = json.loads((HERE / "results" / (name + ".json")).read_text())
    data = group_frame(name)
    if len(data) != original["n"]:
        raise RuntimeError("Group membership changed")
    like = Likelihood(data, "check")
    tight = Likelihood(data, "tight")
    checkpoint = dest.with_suffix(".progress.json")
    rows = json.loads(checkpoint.read_text())["models"] if checkpoint.exists() else {}
    started = time.monotonic()
    for model in MODELS:
        if model in rows and rows[model]["success"]: continue
        row = refine(like, model, original["models"][model], rows)
        row["AIC"] = 2 * row["nll"] + 2 * row["k"]
        row["tight_nll"], row["tight_mex"] = tight.profile_mex(row["params"])
        row["tight_AIC"] = 2 * row["tight_nll"] + 2 * row["k"]
        rows[model] = row
        save(checkpoint, {"group": name, "models": rows})
        print("REFINE", name, model, f"NLL={row['nll']:.6f}",
              f"tight={row['tight_nll']:.6f}", "success=" + str(row["success"]),
              "seconds=" + str(round(row.get("seconds", 0))), flush=True)
    save(dest, {"group": name, "label": GROUPS[name]["label"],
                "n": len(data), "deaths": int(data.event.sum()),
                "baseline": BASELINE, "models": rows, "seconds": time.monotonic()-started,
                "optimized_grid": GRID["check"], "validation_grid": GRID["tight"]})
    checkpoint.unlink(missing_ok=True)
    print("DONE", name, flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group")
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    tasks = list(GROUPS)
    index = args.index if args.index is not None else int(os.environ.get("LSB_JOBINDEX", "0"))
    if args.group is None and not 1 <= index <= len(tasks):
        parser.error("Provide --group or an index in 1..23; no fitting starts by default")
    name = args.group if args.group is not None else tasks[index - 1]
    if name not in GROUPS:
        parser.error("Unknown group")
    run(name)
