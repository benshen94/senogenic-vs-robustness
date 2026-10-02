"""Point-only eta+mex and beta+mex comparisons with archived NHANES fits.

Stage the exact cohort, then run one explicitly selected array task per model
and group. This does not alter the model inventory of the bootstrap analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

HERE = Path(__file__).resolve().parent
COHORT_SHA256 = "0fb93c36eb9ee48b1a5d5d1aab02f4e48444a4530b0ae03ba2740619cc9b0201"
PAIRS = ("eta+beta", "Xc+eta", "Xc+beta", "epsilon+beta", "epsilon+eta")
REFERENCE_MODELS = ("mex_only", "Xc", "epsilon") + PAIRS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage(cohort: Path, output: Path) -> None:
    if sha256(cohort) != COHORT_SHA256:
        raise ValueError("Cohort does not match the archived likelihood input")
    archive = HERE.parents[2] / "results/nhanes/results_high"
    output.mkdir(parents=True, exist_ok=False)
    for name in ("group_fit.py", "single_senogenic.py"):
        shutil.copy2(HERE / name, output / name)
    shutil.copytree(HERE / "inputs", output / "inputs",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(cohort, output / "inputs/nhanes.csv")
    shutil.copytree(archive, output / "results_high")
    groups = json.loads((output / "inputs/groups.json").read_text())
    tasks = [{"group": group, "model": model}
             for group in groups for model in ("eta", "beta")]
    files = [p for p in output.rglob("*") if p.is_file()]
    manifest = {"analysis": "NHANES single-senogenic point fits; no bootstrap",
                "tasks": tasks, "cohort_sha256": COHORT_SHA256,
                "optimized_grid": [320, 0.025, 31],
                "validation_grids": [[480, 1 / 60, 41], [640, 1 / 120, 51]],
                "files": {str(p.relative_to(output)): sha256(p) for p in files}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "logs").mkdir()
    (output / "run_task.sh").write_text(
        '#!/bin/bash\nset -euo pipefail\n'
        'cd "$(dirname "$0")"\nmodule load Anaconda3/2024.06-1\n'
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1\n'
        'exec python3 -u single_senogenic.py --index "${LSB_JOBINDEX:?}"\n')
    (output / "run_task.sh").chmod(0o755)
    print(f"Staged {len(tasks)} point-fit tasks in {output}")


def run(index: int) -> None:
    import group_fit as fit

    manifest = json.loads((HERE / "manifest.json").read_text())
    if not 1 <= index <= len(manifest["tasks"]):
        raise IndexError("Task index outside staged manifest")
    if sha256(HERE / "inputs/nhanes.csv") != COHORT_SHA256:
        raise ValueError("Staged cohort hash changed")
    task = manifest["tasks"][index - 1]
    name, model = task["group"], task["model"]
    dest = HERE / "single_senogenic" / f"{name}_{model}.json"
    if dest.exists() and json.loads(dest.read_text())["fit"]["success"]:
        print(f"EXISTS {dest}", flush=True)
        return
    # Add these names only in this point worker, leaving the legacy nine-model
    # group/paired-bootstrap workflow unchanged in group_fit.py.
    fit.MODELS.update({"eta": ("eta",), "beta": ("beta",)})
    fit.GRID["tight"] = (480, 1 / 60, 41)
    fit.GRID["grid640"] = (640, 1 / 120, 51)
    old_path = HERE / "results_high" / f"{name}.json"
    old = json.loads(old_path.read_text())
    if old["baseline"] != fit.BASELINE:
        raise ValueError("Archived point fit uses a different baseline")
    data = fit.group_frame(name)
    if len(data) != old["n"] or int(data.event.sum()) != old["deaths"]:
        raise ValueError("Group membership mismatch")
    started = time.monotonic()
    like = fit.Likelihood(data, "check")
    row = fit.fit_model(like, model, {}, seconds=900)
    row["AIC"] = 2 * row["nll"] + 2 * row["k"]
    for level in ("tight", "grid640"):
        checker = fit.Likelihood(data, level)
        nll, mex = checker.profile_mex(row["params"])
        row[level + "_nll"] = nll
        row[level + "_mex"] = mex
        row[level + "_AIC"] = 2 * nll + 2 * row["k"]
    reference = {}
    if model == "eta":
        # Only one task per group reevaluates the archived alternatives. The
        # finer grid is a parameter-held-fixed sensitivity, with mex reprofiled.
        fine = fit.Likelihood(data, "grid640")
        for other in REFERENCE_MODELS:
            source = old["models"][other]
            nll, mex = like.profile_mex(source["params"])
            if abs(nll - source["nll"]) > 1e-7:
                raise ValueError(f"Archived likelihood mismatch: {name} {other}")
            fine_nll, fine_mex = fine.profile_mex(source["params"])
            reference[other] = {"nll": nll, "mex": mex,
                                "grid640_nll": fine_nll,
                                "grid640_mex": fine_mex,
                                "grid640_AIC": 2 * fine_nll + 2 * source["k"]}
    result = {"group": name, "label": old["label"], "model": model,
              "n": len(data), "deaths": int(data.event.sum()),
              "baseline": fit.BASELINE, "fit": row, "reference": reference,
              "optimized_grid": fit.GRID["check"],
              "validation_grids": {k: fit.GRID[k] for k in ("tight", "grid640")},
              "seconds": time.monotonic() - started,
              "archived_point_sha256": sha256(old_path),
              "script_sha256": sha256(Path(__file__)),
              "likelihood": "Individual delayed-entry death/censoring; direct FP boundary-flux deaths/person-time hazard; profiled mex",
              "uncertainty": "Point fits only; no new bootstrap."}
    fit.save(dest, result)
    print(f"DONE {index} {name} {model}: NLL={row['nll']:.6f} "
          f"success={row['success']} seconds={result['seconds']:.1f}", flush=True)
    if not row["success"]:
        raise RuntimeError("Saved best encountered fit did not pass convergence check")


def validate_starts(index: int) -> None:
    """Check the reported minimum from two nearby, non-plateau starts."""
    import numpy as np
    from scipy.optimize import minimize
    import group_fit as fit

    manifest = json.loads((HERE / "manifest.json").read_text())
    if not 1 <= index <= len(manifest["tasks"]):
        raise IndexError("Task index outside staged manifest")
    task = manifest["tasks"][index - 1]
    name, model = task["group"], task["model"]
    dest = HERE / "single_senogenic" / f"{name}_{model}.json"
    result = json.loads(dest.read_text())
    if result.get("nearby_start_validation", {}).get("success"):
        print(f"VALIDATED {index} already checked", flush=True)
        return
    if sha256(HERE / "inputs/nhanes.csv") != COHORT_SHA256 or result["baseline"] != fit.BASELINE:
        raise ValueError("Validation inputs changed")
    row = result["fit"]
    like = fit.Likelihood(fit.group_frame(name), "check")
    started = time.monotonic()
    best = (row["nll"], row["params"].copy())
    traces = []

    def objective(x):
        nonlocal best
        params = dict(fit.BASELINE)
        params[model] = float(np.exp(x[0]))
        nll, mex = like.profile_mex(params)
        if nll < best[0]:
            best = (nll, {**params, "mex": mex})
        if time.monotonic() - started > 480:
            raise TimeoutError("Nearby-start check exceeded 480s")
        return nll

    bounds = [tuple(np.log(fit.BOUNDS[model]))]
    for factor in (0.95, 1.05):
        initial = [np.log(row["params"][model] * factor)]
        optimized = minimize(objective, initial, method="L-BFGS-B", jac="3-point",
                             bounds=bounds, options={"maxiter": 120, "ftol": 1e-9,
                                                      "gtol": 0.002, "maxls": 30,
                                                      "finite_diff_rel_step": 2e-5})
        traces.append({"start_factor": factor, "nll": float(optimized.fun),
                       "success": bool(optimized.success), "message": str(optimized.message),
                       "iterations": int(optimized.nit), "nfev": int(optimized.nfev)})
    validation = {"traces": traces, "improvement": row["nll"] - best[0],
                  "seconds": time.monotonic() - started,
                  "success": all(t["success"] and abs(t["nll"] - best[0]) <= 0.005
                                 for t in traces), "script_sha256": sha256(Path(__file__))}
    if best[0] < row["nll"] - 1e-5:
        result["initial_fit"] = dict(row)
        row = {**row, "params": best[1], "nll": best[0], "AIC": 2 * best[0] + 4,
               "traces": row["traces"] + traces, "unique_starts": 5}
        fit.GRID["tight"] = (480, 1 / 60, 41)
        fit.GRID["grid640"] = (640, 1 / 120, 51)
        for level in ("tight", "grid640"):
            nll, mex = fit.Likelihood(fit.group_frame(name), level).profile_mex(best[1])
            row[level + "_nll"], row[level + "_mex"] = nll, mex
            row[level + "_AIC"] = 2 * nll + 4
        row["bounds"] = [model] if min(abs(np.log(best[1][model]) - b) for b in bounds[0]) < 0.002 else []
        if best[1]["mex"] < 1e-10 or best[1]["mex"] > 0.2 - 1e-10:
            row["bounds"].append("mex")
        result["fit"] = row
    result["nearby_start_validation"] = validation
    fit.save(dest, result)
    print(f"VALIDATION {index} success={validation['success']} "
          f"improvement={validation['improvement']:.9f}", flush=True)
    if not validation["success"]:
        raise RuntimeError("Nearby starts did not agree at a converged minimum")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--index", type=int,
                        default=int(os.environ.get("LSB_JOBINDEX", "0")))
    parser.add_argument("--validate-starts", action="store_true")
    args = parser.parse_args()
    if args.stage:
        if args.cohort is None or args.output is None:
            parser.error("Staging requires --cohort and a new --output directory")
        stage(args.cohort, args.output)
    elif args.index and args.validate_starts:
        validate_starts(args.index)
    elif args.index:
        run(args.index)
    else:
        parser.error("Specify --stage or --index; fitting never starts implicitly")
