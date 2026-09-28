"""Evaluate saved NHANES group fits on a finer grid for discordant tasks."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bootstrap_baseline as baseline
import bootstrap_groups as groups
import group_fit as fit

PAIRS = ("eta+beta", "Xc+eta", "Xc+beta", "epsilon+beta", "epsilon+eta")
MODELS = ("mex_only", "Xc", "epsilon") + PAIRS
INPUT = HERE / "bootstrap" / "groups_precision"
OUTPUT = HERE / "bootstrap" / "grid640_validation"
MANIFEST = OUTPUT / "tasks.json"


def build_manifest() -> list[dict]:
    tasks = []
    for path in sorted(INPUT.glob("rep_*.json")):
        if path.name.endswith(".progress.json"):
            continue
        result = json.loads(path.read_text())
        if not 1 <= int(result["replicate"]) <= 50:
            continue
        models = result["models"]
        pair320 = min(models[name]["AIC"] for name in PAIRS)
        pair480 = min(models[name]["tight_AIC"] for name in PAIRS)
        gap320 = min(models[name]["AIC"] for name in ("Xc", "epsilon")) - pair320
        gap480 = min(models[name]["tight_AIC"] for name in ("Xc", "epsilon")) - pair480
        if (gap320 <= 2) != (gap480 <= 2):
            tasks.append({"replicate": result["replicate"], "group": result["group"],
                          "gap320": gap320, "gap480": gap480})
    if len(tasks) != 60:
        raise RuntimeError(f"Expected 60 grid-discordant tasks, found {len(tasks)}")
    tasks.sort(key=lambda row: (row["replicate"], row["group"]))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({"grid": [640, 1 / 120, 51], "tasks": tasks}, indent=2))
    return tasks


def run(index: int) -> None:
    manifest = json.loads(MANIFEST.read_text())
    tasks = manifest["tasks"]
    if index < 1 or index > len(tasks):
        raise IndexError(f"Task index {index} outside 1..{len(tasks)}")
    task = tasks[index - 1]
    rep, name = int(task["replicate"]), str(task["group"])
    dest = OUTPUT / f"rep_{rep:03d}_{name}.json"
    if dest.exists():
        print(f"EXISTS {dest}", flush=True)
        return

    anchor_path = HERE / "bootstrap" / "baseline_precision" / f"rep_{rep:03d}.json"
    anchor = json.loads(anchor_path.read_text())
    if not anchor["success"]:
        raise RuntimeError(f"Baseline resample {rep} did not converge")
    fit.BASELINE = anchor["params"]
    frame = baseline.resampled_frame(rep)
    frame = frame[frame["group_" + name] == 1].copy()
    fit.GRID["grid640"] = (640, 1 / 120, 51)
    like = groups.WeightedGroupLikelihood(frame, "grid640")
    original = json.loads((INPUT / f"rep_{rep:03d}_{name}.json").read_text())
    started = time.monotonic()
    evaluated = {}
    for model in MODELS:
        old = original["models"][model]
        nll, mex = like.profile_mex(old["params"])
        evaluated[model] = {"params_source": old["params"], "nll640": nll,
                            "mex640": mex, "k": old["k"],
                            "AIC640": 2 * nll + 2 * old["k"]}
    payload = {"replicate": rep, "group": name, "baseline": anchor["params"],
               "source_gaps": task, "grid": [640, 1 / 120, 51],
               "models": evaluated, "seconds": time.monotonic() - started,
               "interpretation": "AIC reevaluation at saved 320-grid parameters with mex reprofiled; no intrinsic-parameter refit."}
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False))
    tmp.replace(dest)
    print(f"DONE {index}/{len(tasks)} {rep} {name} {payload['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-manifest", action="store_true")
    parser.add_argument("--index", type=int, default=int(os.environ.get("LSB_JOBINDEX", "0")))
    args = parser.parse_args()
    if args.build_manifest:
        print(json.dumps({"tasks": len(build_manifest()), "manifest": str(MANIFEST)}, indent=2))
    else:
        if args.index < 1:
            parser.error('Specify --index or LSB_JOBINDEX; evaluations never start implicitly.')
        run(args.index)
