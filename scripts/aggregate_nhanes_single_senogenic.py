#!/usr/bin/env python3
"""Audit and summarize point-only NHANES single-senogenic model comparisons."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ("eta+beta", "Xc+eta", "Xc+beta", "epsilon+beta", "epsilon+eta")
SINGLES = ("Xc", "epsilon", "eta", "beta")
GRIDS = ("primary320", "validation480", "validation640")


def comparisons(aic: dict[str, float]) -> dict:
    """Signed family comparisons; smaller AIC is better, <=2 is competitive."""
    pair = min(aic[m] for m in PAIRS)
    robustness = min(aic[m] for m in ("Xc", "epsilon"))
    senogenic = min(aic[m] for m in ("eta", "beta"))
    all_models = min(aic.values())
    return {
        "best_pair_model": min(PAIRS, key=aic.get),
        "best_robustness_model": min(("Xc", "epsilon"), key=aic.get),
        "best_senogenic_model": min(("eta", "beta"), key=aic.get),
        "best_overall_model": min(aic, key=aic.get),
        "delta_senogenic_vs_robustness": senogenic - robustness,
        "robustness_vs_pair": robustness - pair,
        "senogenic_vs_pair": senogenic - pair,
        "robustness_vs_all": robustness - all_models,
        "senogenic_vs_all": senogenic - all_models,
        **{f"delta_{m}_vs_pair": aic[m] - pair for m in SINGLES},
        **{f"delta_{m}_vs_robustness": aic[m] - robustness for m in ("eta", "beta")},
    }


def summarize(rows: list[dict]) -> dict:
    return {
        "groups": len(rows),
        **{f"{m}_competitive_vs_pairs": sum(r[f"delta_{m}_vs_pair"] <= 2 for r in rows)
           for m in SINGLES},
        "either_robustness_competitive_vs_pairs": sum(r["robustness_vs_pair"] <= 2 for r in rows),
        "either_senogenic_competitive_vs_pairs": sum(r["senogenic_vs_pair"] <= 2 for r in rows),
        "either_senogenic_competitive_vs_robustness": sum(r["delta_senogenic_vs_robustness"] <= 2 for r in rows),
        "robustness_lower_AIC": sum(r["delta_senogenic_vs_robustness"] > 0 for r in rows),
        "senogenic_lower_AIC": sum(r["delta_senogenic_vs_robustness"] < 0 for r in rows),
        "families_exact_AIC_tie": sum(r["delta_senogenic_vs_robustness"] == 0 for r in rows),
        "robustness_better_by_more_than_2": sum(r["delta_senogenic_vs_robustness"] > 2 for r in rows),
        "senogenic_better_by_more_than_2": sum(r["delta_senogenic_vs_robustness"] < -2 for r in rows),
        "families_within_2": sum(abs(r["delta_senogenic_vs_robustness"]) <= 2 for r in rows),
        "either_robustness_competitive_vs_all": sum(r["robustness_vs_all"] <= 2 for r in rows),
        "either_senogenic_competitive_vs_all": sum(r["senogenic_vs_all"] <= 2 for r in rows),
        "median_senogenic_minus_robustness_AIC": statistics.median(r["delta_senogenic_vs_robustness"] for r in rows),
        "range_senogenic_minus_robustness_AIC": [min(r["delta_senogenic_vs_robustness"] for r in rows),
                                                 max(r["delta_senogenic_vs_robustness"] for r in rows)],
    }


def main(raw: Path, output: Path) -> None:
    groups = json.loads((ROOT / "analysis/model_fits/nhanes/inputs/groups.json").read_text())
    baseline = json.loads((ROOT / "analysis/model_fits/nhanes/inputs/best_constrained_fit.json").read_text())["params"]
    expected = {f"{group}_{model}.json" for group in groups for model in ("eta", "beta")}
    actual = {p.name for p in raw.glob("*.json")}
    if actual != expected:
        raise ValueError(f"Missing/unexpected results: {expected - actual} / {actual - expected}")
    rows = {g: [] for g in GRIDS}
    fits = []
    nesting = []
    archived_hashes = {}
    for group, info in groups.items():
        point_path = ROOT / f"results/nhanes/results_high/{group}.json"
        point_hash = hashlib.sha256(point_path.read_bytes()).hexdigest()
        old = json.loads(point_path.read_text())
        archived_hashes[group] = point_hash
        new = {m: json.loads((raw / f"{group}_{m}.json").read_text()) for m in ("eta", "beta")}
        for m, result in new.items():
            fit = result["fit"]
            assert result["group"] == group and result["model"] == m
            assert result["baseline"] == baseline == old["baseline"]
            assert result["archived_point_sha256"] == point_hash
            assert result["n"] == info["n"] and result["deaths"] == info["deaths"]
            assert fit["k"] == 2 and fit["success"], (group, m, "convergence")
            validation = result.get("nearby_start_validation", {})
            assert validation.get("success"), (group, m, "nearby-start check missing/failed")
            assert set(fit["bounds"]).issubset({"mex"}), (group, m, "intrinsic boundary")
            assert 0 <= fit["params"]["mex"] < 0.2 - 1e-10, (group, m, "upper mex boundary")
            assert any(t["success"] and abs(t["nll"] - fit["nll"]) <= 0.005
                       for t in fit["traces"]), (group, m, "no converged best start")
            for p, value in baseline.items():
                if p not in (m, "mex"):
                    assert fit["params"][p] == value, (group, m, "fixed parameter changed")
            for key in ("nll", "AIC", "tight_nll", "tight_AIC", "grid640_nll", "grid640_AIC"):
                assert math.isfinite(fit[key]), (group, m, key)
            for prefix in ("", "tight_", "grid640_"):
                assert abs(fit[prefix + "AIC"] - 2 * fit[prefix + "nll"] - 4) < 1e-8
            fits.append({"group": group, "model": m, "bounds": fit["bounds"],
                         "mex": fit["params"]["mex"],
                         "nearby_start_validation": validation,
                         "converged_starts": sum(t["success"] for t in fit["traces"]),
                         "starts": fit["unique_starts"], "seconds": result["seconds"]})
            for pair in PAIRS:
                if m in pair.split("+") and old["models"][pair]["nll"] > fit["nll"] + 0.005:
                    nesting.append({"group": group, "single": m, "pair": pair,
                                    "NLL_pair_minus_single": old["models"][pair]["nll"] - fit["nll"]})
        for grid in GRIDS:
            if grid == "primary320":
                aic = {m: old["models"][m]["AIC"] for m in ("mex_only",) + SINGLES[:2] + PAIRS}
                aic.update({m: new[m]["fit"]["AIC"] for m in ("eta", "beta")})
            elif grid == "validation480":
                aic = {m: old["models"][m]["tight_AIC"] for m in ("mex_only",) + SINGLES[:2] + PAIRS}
                aic.update({m: new[m]["fit"]["tight_AIC"] for m in ("eta", "beta")})
            else:
                aic = {m: r["grid640_AIC"] for m, r in new["eta"]["reference"].items()}
                aic.update({m: new[m]["fit"]["grid640_AIC"] for m in ("eta", "beta")})
            row = {"group": group, "label": info["label"], "n": info["n"], "deaths": info["deaths"],
                   **comparisons(aic), **{f"AIC_{m}": v for m, v in aic.items()}}
            row.update({f"NLL_{m}": (value - 2 * (1 if m == "mex_only" else 3 if m in PAIRS else 2)) / 2
                        for m, value in aic.items()})
            for m in SINGLES:
                fit = new[m]["fit"] if m in new else old["models"][m]
                row[f"{m}_factor"] = fit["params"][m] / baseline[m]
                row[f"{m}_mex"] = fit["params"]["mex"] if grid == "primary320" else (
                    fit["tight_mex"] if grid == "validation480" else (
                        fit["grid640_mex"] if m in new else new["eta"]["reference"][m]["grid640_mex"]))
            rows[grid].append(row)
    output.mkdir(parents=True, exist_ok=True)
    for grid, values in rows.items():
        with (output / f"comparisons_{grid}.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
    summary = {grid: summarize(values) for grid, values in rows.items()}
    summary["audit"] = {"complete_results": len(fits), "all_converged": True,
                        "nearby_start_checks_passed": len(fits),
                        "maximum_nearby_start_NLL_improvement": max(f["nearby_start_validation"]["improvement"] for f in fits),
                        "intrinsic_bound_hits": 0,
                        "mex_zero_fits": sum(fit["mex"] == 0 for fit in fits),
                        "paired_nesting_flags": nesting, "fits": fits,
                        "archived_point_hashes": archived_hashes,
                        "baseline": baseline,
                        "grid_interpretation": "320-cell optimized fits; 480/640 evaluations hold intrinsic parameters fixed and reprofile mex.",
                        "uncertainty": "Point fits only; the earlier robustness bootstrap does not provide uncertainty for these new senogenic fits."}
    summary["classification_changes"] = {
        grid: [a["group"] for a, b in zip(rows["primary320"], rows[grid])
               if (a["senogenic_vs_pair"] <= 2) != (b["senogenic_vs_pair"] <= 2)]
        for grid in GRIDS[1:]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    s = summary["primary320"]
    lines = ["# Single-senogenic NHANES point-fit comparison", "",
             "All models use the same full-cohort anchor and individual delayed-entry death/censoring likelihood. "
             "Each single model fits one intrinsic parameter plus extrinsic mortality (k=2); each pair fits two intrinsic parameters plus extrinsic mortality (k=3). CV and kappa remain fixed. No new bootstrap was run.", "",
             f"Against the best of five intrinsic-parameter pairs, Xc or epsilon is competitive in {s['either_robustness_competitive_vs_pairs']}/23 groups; eta or beta in {s['either_senogenic_competitive_vs_pairs']}/23. "
             f"Eta alone is competitive in {s['eta_competitive_vs_pairs']}/23 and beta alone in {s['beta_competitive_vs_pairs']}/23.", "",
             f"Comparing the lower AIC of Xc+mex and epsilon+mex with the lower AIC of eta+mex and beta+mex, robustness has lower AIC in {s['robustness_lower_AIC']}/23 groups and senogenic fits in {s['senogenic_lower_AIC']}/23. "
             f"Robustness is better by more than 2 AIC units in {s['robustness_better_by_more_than_2']} groups, senogenic models in {s['senogenic_better_by_more_than_2']}, and the families differ by at most 2 in {s['families_within_2']}. "
             f"The median signed difference (best senogenic minus best robustness AIC) is {s['median_senogenic_minus_robustness_AIC']:.2f}.", "",
             "Positive differences below favor robustness; negative differences favor a senogenic fit. The five-pair reference is eta+beta, Xc+eta, Xc+beta, epsilon+beta, or epsilon+eta, with mex in every model.", "",
             "| Group | Xc vs pair | epsilon vs pair | eta vs pair | beta vs pair | Best senogenic minus best robustness |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in rows["primary320"]:
        lines.append(f"| {r['group']} ({r['label']}) | " + " | ".join(
            f"{r[f'delta_{m}_vs_pair']:+.2f}" for m in SINGLES) +
            f" | {r['delta_senogenic_vs_robustness']:+.2f} |")
    lines += ["", "## Numerical sensitivity", "",
              "| Grid (cells, time step, threshold nodes) | Robustness competitive vs pairs | Senogenic competitive vs pairs |",
              "|---|---:|---:|"]
    for grid, spec in zip(GRIDS, ("320, 1/40 y, 31", "480, 1/60 y, 41", "640, 1/120 y, 51")):
        lines.append(f"| {spec} | {summary[grid]['either_robustness_competitive_vs_pairs']}/23 | {summary[grid]['either_senogenic_competitive_vs_pairs']}/23 |")
    lines += ["", f"All 46 new fits converged and passed checks from two nearby starts; {summary['audit']['mex_zero_fits']} had mex=0 and none hit an intrinsic bound. "
              f"There were {len(nesting)} paired-model likelihood nesting flags (details in summary.json). "
              "The primary results use the same grid as the manuscript comparison. Finer-grid rows are sensitivity evaluations, not reoptimized fits.", "",
              "These are conditional model-sufficiency comparisons, not evidence identifying unique biological mechanisms. "
              "The groups overlap, and these point estimates do not propagate baseline or survey-sampling uncertainty for the new senogenic models.", ""]
    lines += ["AIC retains mex in the parameter count at its zero boundary; regular AIC approximations are therefore imperfect for those fits. "
              "All four single models use the same parameter count and boundary convention.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps({grid: summary[grid] for grid in GRIDS}, indent=2))
    if nesting:
        raise RuntimeError("Paired fits need review before claims against the five-pair reference")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "results/nhanes/single_senogenic/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "results/nhanes/single_senogenic")
    args = parser.parse_args()
    main(args.raw, args.output)
