"""Audit targeted 640-cell AIC reevaluations of grid-discordant bootstrap fits."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PAIRS = ("eta+beta", "Xc+eta", "Xc+beta", "epsilon+beta", "epsilon+eta")
SINGLES = ("Xc", "epsilon")
MODELS = ("mex_only",) + SINGLES + PAIRS


def classify(models: dict, field: str) -> tuple[float, bool]:
    pair_best = min(models[name][field] for name in PAIRS)
    single_best = min(models[name][field] for name in SINGLES)
    gap = single_best - pair_best
    return float(gap), bool(gap <= 2)


def main() -> None:
    manifest = json.loads((HERE / "bootstrap" / "grid640_validation" / "tasks.json").read_text())
    task_rows = manifest["tasks"]
    task_keys = [(int(x["replicate"]), x["group"]) for x in task_rows]
    if len(task_keys) != 60 or len(set(task_keys)) != 60:
        raise RuntimeError("Expected 60 unique grid-discordant tasks")

    raw_path = HERE / "bootstrap_precision_group_raw.csv"
    raw = pd.read_csv(raw_path)
    if len(raw) != 1150 or raw.duplicated(["replicate", "group"]).any():
        raise RuntimeError("Expected 1,150 unique paired bootstrap group rows")
    raw = raw.set_index(["replicate", "group"])
    out_dir = HERE / "bootstrap" / "grid640_validation"
    records = []
    for rep, group in task_keys:
        path = out_dir / f"rep_{rep:03d}_{group}.json"
        payload = json.loads(path.read_text())
        original_path = HERE / "bootstrap" / "groups_precision" / path.name
        original = json.loads(original_path.read_text())
        evaluated = payload["models"]
        if payload["replicate"] != rep or payload["group"] != group:
            raise RuntimeError(f"Result key mismatch: {path.name}")
        if set(evaluated) != set(MODELS):
            raise RuntimeError(f"Incomplete model set: {path.name}")
        for model in MODELS:
            for k in ("AIC640", "nll640", "mex640"):
                if not np.isfinite(evaluated[model][k]):
                    raise RuntimeError(f"Nonfinite {k}: {path.name} {model}")
            if evaluated[model]["params_source"] != original["models"][model]["params"]:
                raise RuntimeError(f"Parameter source mismatch: {path.name} {model}")
        gap320 = float(raw.loc[(rep, group), "high_gap_best_single"])
        gap480 = float(raw.loc[(rep, group), "tight_gap_best_single"])
        gap640, comp640 = classify(evaluated, "AIC640")
        records.append({"replicate": rep, "group": group,
                        "gap320": gap320, "competitive320": gap320 <= 2,
                        "gap480": gap480, "competitive480": gap480 <= 2,
                        "gap640": gap640, "competitive640": comp640,
                        "matches320": bool((gap320 <= 2) == comp640),
                        "matches480": bool((gap480 <= 2) == comp640)})

    expected = set(task_keys)
    observed = {(r["replicate"], r["group"]) for r in records}
    if observed != expected:
        raise RuntimeError("Grid-640 outputs do not match the manifest")
    result = pd.DataFrame(records)
    result.to_csv(HERE / "bootstrap_precision_grid640_discordant.csv", index=False)

    # Compare all 1,150 tasks: use 640 only for the 60 grid-discordant cases.
    all_rows = []
    replacement = {(r["replicate"], r["group"]): r["competitive640"] for r in records}
    for (rep, group), row in raw.iterrows():
        key = (int(rep), group)
        all_rows.append({"replicate": int(rep), "group": group,
                         "competitive320": bool(row["high_competitive_either"]),
                         "competitive480": bool(row["tight_competitive_either"]),
                         "competitive_grid_consensus": replacement.get(
                             key, bool(row["high_competitive_either"]))})
    all_data = pd.DataFrame(all_rows)
    per_rep = all_data.groupby("replicate").agg(
        grid320=("competitive320", "sum"),
        grid480=("competitive480", "sum"),
        targeted640=("competitive_grid_consensus", "sum"),
    ).reset_index()
    per_rep.to_csv(HERE / "bootstrap_precision_grid640_replicate_summary.csv", index=False)

    def stats(column: str) -> dict:
        values = per_rep[column]
        return {"mean_groups": float(values.mean()),
                "median_groups": float(values.median()),
                "q025_q975_groups": list(map(float, values.quantile([.025, .975]))),
                "fraction_resamples_at_least_18": float((values >= 18).mean())}

    summary = {
        "grid640_tasks": len(records),
        "grid640_files": len(list(out_dir.glob("rep_*.json"))) - 0,
        "grid640_matches_320": int(result.matches320.sum()),
        "grid640_matches_480": int(result.matches480.sum()),
        "classification_changes_vs_320": int((~result.matches320).sum()),
        "classification_changes_vs_480": int((~result.matches480).sum()),
        "reevaluation": "640 cells, 1/120-year steps, 51 quadrature nodes; intrinsic parameters fixed at 320-grid estimates and mex reprofiled",
        "all_task_counts": {"320": stats("grid320"), "480": stats("grid480"),
                            "targeted_640_consensus": stats("targeted640")},
        "targeted_640_is_a_grid_sensitivity_check_not_a_refitted_primary_analysis": True,
    }
    (HERE / "bootstrap_precision_grid640_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path,
                        default=Path(__file__).resolve().parents[3] / 'results/nhanes')
    HERE = parser.parse_args().results_dir
    main()
