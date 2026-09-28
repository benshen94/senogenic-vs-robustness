"""Audit and summarize all 50 paired-PSU group likelihood replicates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "results/nhanes"
REPLICATES = 50
GROUPS = json.loads((Path(__file__).resolve().parent / "inputs/groups.json").read_text())
SINGLES = ("Xc", "epsilon")
PAIRS = ("eta+beta", "Xc+eta", "Xc+beta", "epsilon+beta", "epsilon+eta")
MODELS = ("mex_only",) + SINGLES + PAIRS


def main() -> None:
    paths = sorted((HERE / "bootstrap" / "groups_precision").glob("rep_*.json"))
    paths = [p for p in paths if not p.name.endswith(".progress.json")]
    records = []
    flags = []
    for path in paths:
        result = json.loads(path.read_text())
        rep, group = int(result["replicate"]), result["group"]
        if path.name != f"rep_{rep:03d}_{group}.json":
            flags.append(f"file/key mismatch: {path.name}")
        if rep not in range(1, REPLICATES + 1) or group not in GROUPS:
            flags.append(f"unexpected key: {rep}, {group}")
        if set(result["models"]) != set(MODELS):
            flags.append(f"incomplete model set: {rep}, {group}")
            continue
        models = result["models"]
        for model, item in models.items():
            if not item["success"]:
                flags.append(f"unconverged: {rep}, {group}, {model}")
            if not np.isfinite(item["AIC"]) or not np.isfinite(item["tight_AIC"]):
                flags.append(f"nonfinite AIC: {rep}, {group}, {model}")
            if any(b != "mex" for b in item["bounds"]):
                flags.append(f"intrinsic bound: {rep}, {group}, {model}: {item['bounds']}")
        row = {"replicate": rep, "group": group,
               "label": GROUPS[group]["label"],
               "weighted_n": result["weighted_n"],
               "weighted_deaths": result["weighted_deaths"]}
        for prefix, field in (("high", "AIC"), ("tight", "tight_AIC")):
            pair = min(models[name][field] for name in PAIRS)
            best_requested = min(models[name][field] for name in MODELS)
            row[f"{prefix}_best_pair"] = min(PAIRS, key=lambda name: models[name][field])
            row[f"{prefix}_best_requested"] = min(MODELS, key=lambda name: models[name][field])
            row[f"{prefix}_gap_Xc"] = models["Xc"][field] - pair
            row[f"{prefix}_gap_epsilon"] = models["epsilon"][field] - pair
            row[f"{prefix}_gap_best_single"] = min(row[f"{prefix}_gap_Xc"],
                                                     row[f"{prefix}_gap_epsilon"])
            row[f"{prefix}_gap_best_single_vs_requested"] = min(
                models[name][field] for name in SINGLES) - best_requested
            row[f"{prefix}_competitive_Xc"] = row[f"{prefix}_gap_Xc"] <= 2
            row[f"{prefix}_competitive_epsilon"] = row[f"{prefix}_gap_epsilon"] <= 2
            row[f"{prefix}_competitive_either"] = row[f"{prefix}_gap_best_single"] <= 2
            row[f"{prefix}_competitive_either_vs_requested"] = (
                row[f"{prefix}_gap_best_single_vs_requested"] <= 2)
        records.append(row)
    data = pd.DataFrame(records)
    expected = {(rep, group) for rep in range(1, REPLICATES + 1) for group in GROUPS}
    observed = set(zip(data.replicate, data.group)) if len(data) else set()
    missing = sorted(expected - observed)
    duplicates = data[data.duplicated(["replicate", "group"], keep=False)] if len(data) else data
    if missing:
        flags.append(f"missing {len(missing)} replicate/group files; first: {missing[:10]}")
    if len(duplicates):
        flags.append(f"duplicate replicate/group rows: {len(duplicates)}")
    if len(data):
        data.sort_values(["replicate", "group"]).to_csv(
            HERE / "bootstrap_precision_group_raw.csv", index=False)
    if flags:
        (HERE / "bootstrap_precision_audit.json").write_text(json.dumps(
            {"files": len(paths), "valid_rows": len(data), "missing": len(missing),
             "flags": flags}, indent=2))
        raise RuntimeError(f"Bootstrap audit failed with {len(flags)} flags; see audit JSON")
    (HERE / "bootstrap_precision_audit.json").write_text(json.dumps(
        {"status": "passed", "files": len(paths), "valid_rows": len(data),
         "missing": len(missing), "duplicates": len(duplicates), "flags": []},
        indent=2))

    point = pd.read_csv(HERE / "high_grid_aic_comparisons.csv").set_index("group")
    rows = []
    for group, frame in data.groupby("group", sort=False):
        frame = frame.sort_values("replicate")
        entry = {"group": group, "label": GROUPS[group]["label"],
                 "n": int(point.loc[group, "n"]),
                 "deaths": int(point.loc[group, "deaths"]),
                 "point_gap_Xc": float(point.loc[group, "high_delta_Xc_vs_best_pair"]),
                 "point_gap_epsilon": float(point.loc[group, "high_delta_epsilon_vs_best_pair"]),
                "point_gap_best_single": float(point.loc[group, "high_best_single_gap"])}
        entry["point_gap_best_single_vs_requested"] = float(
            min(point.loc[group, "high_AIC_Xc"],
                point.loc[group, "high_AIC_epsilon"]) -
            min(point.loc[group, f"high_AIC_{name}"] for name in MODELS))
        for grid in ("high", "tight"):
            for name in ("Xc", "epsilon", "either"):
                entry[f"{grid}_p_competitive_{name}"] = float(
                    frame[f"{grid}_competitive_{name}"].mean())
            entry[f"{grid}_p_competitive_either_vs_requested"] = float(
                frame[f"{grid}_competitive_either_vs_requested"].mean())
            for scope in ("either", "either_vs_requested"):
                proportion = entry[f"{grid}_p_competitive_{scope}"]
                entry[f"{grid}_mcse_competitive_{scope}"] = float(
                    np.sqrt(proportion * (1 - proportion) / REPLICATES))
            for name in ("Xc", "epsilon", "best_single"):
                lo, median, hi = frame[f"{grid}_gap_{name}"].quantile([.025, .5, .975])
                entry[f"{grid}_gap_{name}_q025"] = lo
                entry[f"{grid}_gap_{name}_median"] = median
                entry[f"{grid}_gap_{name}_q975"] = hi
            lo, median, hi = frame[f"{grid}_gap_best_single_vs_requested"].quantile(
                [.025, .5, .975])
            entry[f"{grid}_gap_best_single_vs_requested_q025"] = lo
            entry[f"{grid}_gap_best_single_vs_requested_median"] = median
            entry[f"{grid}_gap_best_single_vs_requested_q975"] = hi
        rows.append(entry)
    summary = pd.DataFrame(rows).sort_values("group")
    summary.to_csv(HERE / "bootstrap_precision_group_summary.csv", index=False)
    per_rep = data.groupby("replicate").agg(
        high_competitive=("high_competitive_either", "sum"),
        tight_competitive=("tight_competitive_either", "sum"),
        high_competitive_vs_requested=("high_competitive_either_vs_requested", "sum"),
        tight_competitive_vs_requested=("tight_competitive_either_vs_requested", "sum"),
    ).reset_index()
    per_rep.to_csv(HERE / "bootstrap_precision_replicate_summary.csv", index=False)
    total = {"replicates": REPLICATES, "groups": len(GROUPS),
             "fits": len(data) * len(MODELS), "unconverged": 0,
             "point_competitive": int((summary.point_gap_best_single <= 2).sum()),
             "point_competitive_vs_requested": int(
                 (summary.point_gap_best_single_vs_requested <= 2).sum()),
             "high_competitive_count_mean": float(per_rep.high_competitive.mean()),
             "high_competitive_count_q025_q975": list(map(float,
                 per_rep.high_competitive.quantile([.025, .975]))),
             "fraction_of_resamples_with_at_least_18_competitive": float(
                 (per_rep.high_competitive >= 18).mean()),
             "high_competitive_vs_requested_count_mean": float(
                 per_rep.high_competitive_vs_requested.mean()),
             "high_competitive_vs_requested_count_q025_q975": list(map(float,
                 per_rep.high_competitive_vs_requested.quantile([.025, .975]))),
             "fraction_of_resamples_with_at_least_18_competitive_vs_requested": float(
                 (per_rep.high_competitive_vs_requested >= 18).mean()),
             "grid_classification_disagreements": int(np.sum(
                 data.high_competitive_either != data.tight_competitive_either)),
             "max_abs_gap_change_high_to_tight": float(np.max(np.abs(
                 data.high_gap_best_single - data.tight_gap_best_single))),
             "pointwise_intervals": "percentile summaries of paired survey-PSU resamples",
             "reference": "best of five two-intrinsic-parameter alternatives; mex free in every model"}
    total["maximum_binomial_mcse_for_group_support"] = float(
        np.sqrt(.25 / REPLICATES))
    (HERE / "bootstrap_precision_summary.json").write_text(json.dumps(total, indent=2))
    print(json.dumps(total, indent=2))


if __name__ == "__main__":
    main()
