"""Multistart repair for the one NHANES bootstrap fit flagged by the audit."""
from __future__ import annotations

import hashlib
import argparse
import json
import sys
import time
from pathlib import Path

import bootstrap_baseline as baseline
import bootstrap_groups as groups
import group_fit as fit

HERE = Path(__file__).resolve().parent
REP = 18
GROUP = "income__1"
MODEL = "Xc+beta"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def main() -> None:
    source = HERE / "bootstrap" / "groups_precision" / f"rep_{REP:03d}_{GROUP}.json"
    baseline_path = HERE / "bootstrap" / "baseline_precision" / f"rep_{REP:03d}.json"
    point_path = HERE / "results_high" / f"{GROUP}.json"
    repair_dir = HERE / "bootstrap" / "repairs"
    backup = repair_dir / f"rep_{REP:03d}_{GROUP}_original.json"
    report_path = repair_dir / f"rep_{REP:03d}_{GROUP}_{MODEL}_repair.json"
    if report_path.exists():
        raise RuntimeError(f"Repair report already exists: {report_path}")

    original = json.loads(source.read_text())
    baseline_fit = json.loads(baseline_path.read_text())
    if not original["models"][MODEL]["bounds"]:
        raise RuntimeError("Expected the original beta-bound fit before repair")
    fit.BASELINE = baseline_fit["params"]
    frame = baseline.resampled_frame(REP)
    frame = frame[frame["group_" + GROUP] == 1].copy()
    coarse_like = groups.WeightedGroupLikelihood(frame, "fit")
    high_like = groups.WeightedGroupLikelihood(frame, "check")
    tight_like = groups.WeightedGroupLikelihood(frame, "tight")
    point = json.loads(point_path.read_text())["models"]

    started = time.monotonic()
    coarse = fit.fit_model(coarse_like, MODEL, {**point, **original["models"]})
    repaired = groups.refine_model(
        high_like, MODEL, coarse, original["models"], point,
        audit_multistart=True,
    )
    repaired["coarse_nll"] = coarse["nll"]
    repaired["AIC"] = 2 * repaired["nll"] + 2 * repaired["k"]
    repaired["tight_nll"], repaired["tight_mex"] = tight_like.profile_mex(
        repaired["params"]
    )
    repaired["tight_AIC"] = 2 * repaired["tight_nll"] + 2 * repaired["k"]
    repaired["repair_diagnostic"] = {
        "reason": "Original fit hit beta lower bound and stopped after one high-grid start.",
        "method": "Refit coarse model, then audit every candidate start on the 320-cell grid.",
        "coarse_params": coarse["params"],
        "coarse_traces": coarse["traces"],
        "elapsed_seconds": time.monotonic() - started,
    }
    if not repaired["success"] or any(b != "mex" for b in repaired["bounds"]):
        raise RuntimeError(f"Repair remains unconverged or intrinsically bounded: {repaired}")
    if len(repaired["traces"]) < 2:
        raise RuntimeError("Repair did not evaluate multiple high-grid starts")

    if not backup.exists():
        write_json(backup, original)
    output = dict(original)
    output["models"] = dict(original["models"])
    output["models"][MODEL] = repaired
    output["repair"] = {
        "reason": "Strict bootstrap audit flagged beta at its lower bound.",
        "replacement": report_path.name,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_json(source, output)
    report = {
        "replicate": REP,
        "group": GROUP,
        "model": MODEL,
        "bootstrap_seed": original["bootstrap_seed"],
        "baseline": baseline_fit["params"],
        "original_model": original["models"][MODEL],
        "repaired_model": repaired,
        "grids": {"optimized": fit.GRID["check"], "validation": fit.GRID["tight"]},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_json(report_path, report)
    print(json.dumps({"source": str(source), "backup": str(backup),
                      "report": str(report_path), "repaired_model": repaired},
                     indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recompute', action='store_true')
    if not parser.parse_args().recompute:
        parser.error('Repair fitting requires explicit --recompute in a staged rerun directory.')
    sys.path.insert(0, str(HERE))
    main()
