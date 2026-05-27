#!/usr/bin/env python3
"""Run Sweden 2019 steepness-longevity sensitivity simulations.

The run is intentionally checkpointed at the single-simulation level. Each
finished simulation appends all requested metrics to ``metrics_long.csv`` so a
long run can be resumed without losing completed work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGING_PYTHON_ROOT = PROJECT_ROOT.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(AGING_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(AGING_PYTHON_ROOT))

from ageing_packages.utils import sr_utils as utils
from src.shared.thresholds.paths import SAVED_RESULTS_DIR


OUTPUT_DIR = SAVED_RESULTS_DIR / "steepness_longevity_sweden2019_sensitivity"
METRICS_PATH = OUTPUT_DIR / "metrics_long.csv"
SCENARIOS_PATH = OUTPUT_DIR / "baseline_scenarios.csv"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
README_PATH = OUTPUT_DIR / "README.md"

RUN_LABEL = "sweden2019_tail90_zero_hext_sensitivity"

BASELINE = {
    "eta": 0.5868368257640714,
    "beta": 57.87173772073557,
    "kappa": 0.5,
    "epsilon": 49.718659304628446,
    "Xc": 21.74056340066893,
    "xc_std_frac": 0.14142135623730953,
}

PARAMS_TO_VARY = ("eta", "beta", "kappa", "epsilon", "Xc")
NUISANCE_PARAMS = ("eta", "beta", "epsilon", "Xc", "xc_std_frac")
NUISANCE_FACTORS = (0.8, 1.2)
FACTOR_VALUES = tuple(np.round(np.arange(0.5, 1.51, 0.1), 10))
H_EXT_VALUES = tuple(np.logspace(-4, -2, 10))
FROM_T_VALUES = (0, 15, 20, 30, 40, 50)

BASE_H_EXT = 0.0
TMAX = 140.0
DT = 0.025
SAVE_TIMES = 140.0
RNG_SEED = 20260519

METRIC_FIELDS = (
    "steepness_iqr_relative",
    "steepness_iqr_absolute",
    "steepness_cv_relative",
    "steepness_cv_absolute",
    "t_median_relative",
    "t_median_absolute",
    "t_max",
)


@dataclass(frozen=True)
class BaselineScenario:
    scenario_id: str
    nuisance_param: str
    nuisance_factor: float
    eta: float
    beta: float
    kappa: float
    epsilon: float
    Xc: float
    xc_std_frac: float


@dataclass(frozen=True)
class SimulationSpec:
    run_id: str
    scenario_id: str
    nuisance_param: str
    nuisance_factor: float
    curve_type: str
    focal_param: str
    focal_value: float
    h_ext: float
    random_seed: int


def stable_seed(*parts: object) -> int:
    """Create a deterministic uint32 seed from readable run metadata."""
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def scenario_id(nuisance_param: str, nuisance_factor: float) -> str:
    if nuisance_param == "central":
        return "central"

    suffix = "low" if nuisance_factor < 1.0 else "high"
    return f"{nuisance_param}_{suffix}"


def build_scenarios() -> list[BaselineScenario]:
    """Build central and one-at-a-time nuisance baseline scenarios."""
    scenarios = [
        BaselineScenario(
            scenario_id="central",
            nuisance_param="central",
            nuisance_factor=1.0,
            **BASELINE,
        )
    ]

    for param in NUISANCE_PARAMS:
        for factor in NUISANCE_FACTORS:
            values = dict(BASELINE)
            values[param] = values[param] * factor
            scenarios.append(
                BaselineScenario(
                    scenario_id=scenario_id(param, factor),
                    nuisance_param=param,
                    nuisance_factor=factor,
                    **values,
                )
            )

    return scenarios


def build_specs(scenarios: Iterable[BaselineScenario]) -> list[SimulationSpec]:
    """Build the full simulation grid for all scenarios and curves."""
    specs = []

    for scenario in scenarios:
        specs.append(
            make_spec(
                scenario=scenario,
                curve_type="baseline",
                focal_param="baseline",
                focal_value=1.0,
                h_ext=BASE_H_EXT,
            )
        )

        for param in PARAMS_TO_VARY:
            for factor in FACTOR_VALUES:
                specs.append(
                    make_spec(
                        scenario=scenario,
                        curve_type="parameter_factor",
                        focal_param=param,
                        focal_value=float(factor),
                        h_ext=BASE_H_EXT,
                    )
                )

        for h_ext in H_EXT_VALUES:
            specs.append(
                make_spec(
                    scenario=scenario,
                    curve_type="h_ext_absolute",
                    focal_param="h_ext",
                    focal_value=float(h_ext),
                    h_ext=float(h_ext),
                )
            )

    return specs


def make_spec(
    scenario: BaselineScenario,
    curve_type: str,
    focal_param: str,
    focal_value: float,
    h_ext: float,
) -> SimulationSpec:
    run_id = (
        f"{scenario.scenario_id}__{curve_type}__"
        f"{focal_param}_{format_value_for_id(focal_value)}"
    )
    return SimulationSpec(
        run_id=run_id,
        scenario_id=scenario.scenario_id,
        nuisance_param=scenario.nuisance_param,
        nuisance_factor=scenario.nuisance_factor,
        curve_type=curve_type,
        focal_param=focal_param,
        focal_value=focal_value,
        h_ext=h_ext,
        random_seed=stable_seed(RUN_LABEL, run_id),
    )


def format_value_for_id(value: float) -> str:
    return f"{value:.8g}".replace(".", "p").replace("-", "m")


def sample_positive_gaussian(mean: float, rel_std: float, n: int, seed: int) -> np.ndarray:
    """Draw positive Gaussian samples by resampling non-positive values."""
    rng = np.random.default_rng(seed)
    std = mean * rel_std
    values = rng.normal(loc=mean, scale=std, size=n)

    while True:
        bad = values <= 0
        if not np.any(bad):
            return values

        values[bad] = rng.normal(loc=mean, scale=std, size=int(bad.sum()))


def build_param_dict(
    scenario: BaselineScenario,
    spec: SimulationSpec,
    n: int,
) -> dict[str, np.ndarray]:
    """Build a simulation parameter dictionary for one spec."""
    params = {
        "eta": scenario.eta,
        "beta": scenario.beta,
        "kappa": scenario.kappa,
        "epsilon": scenario.epsilon,
    }

    for key in list(params):
        if spec.curve_type == "parameter_factor" and spec.focal_param == key:
            params[key] = params[key] * spec.focal_value

    xc_mean = scenario.Xc
    if spec.curve_type == "parameter_factor" and spec.focal_param == "Xc":
        xc_mean = xc_mean * spec.focal_value

    params["Xc"] = sample_positive_gaussian(
        mean=xc_mean,
        rel_std=scenario.xc_std_frac,
        n=n,
        seed=stable_seed(RUN_LABEL, scenario.scenario_id, "xc_vector"),
    )

    for key, value in params.items():
        if key == "Xc":
            continue
        params[key] = np.full(n, value, dtype=float)

    return params


def calculate_metrics(sim) -> dict[int, dict[str, float | None]]:
    """Calculate the saved steepness and lifespan metrics."""
    metrics = {}

    for from_t in FROM_T_VALUES:
        metrics[from_t] = {
            "steepness_iqr_relative": sim.calc_steepness(
                method="IQR",
                from_t=from_t,
                relative=True,
            ),
            "steepness_iqr_absolute": sim.calc_steepness(
                method="IQR",
                from_t=from_t,
                relative=False,
            ),
            "steepness_cv_relative": sim.calc_steepness(
                method="CV",
                from_t=from_t,
                relative=True,
            ),
            "steepness_cv_absolute": sim.calc_steepness(
                method="CV",
                from_t=from_t,
                relative=False,
            ),
            "t_median_relative": sim.find_time_at_survival(
                0.5,
                from_t=from_t,
                relative=True,
            ),
            "t_median_absolute": sim.find_time_at_survival(
                0.5,
                from_t=from_t,
                relative=False,
            ),
            "t_max": sim.find_time_at_survival(
                0.0001,
                from_t=from_t,
                relative=False,
            ),
        }

    return metrics


def completed_run_ids() -> set[str]:
    if not METRICS_PATH.exists():
        return set()

    with METRICS_PATH.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["run_id"] for row in reader if row.get("run_id")}


def write_setup_files(scenarios: list[BaselineScenario], specs: list[SimulationSpec], n: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with SCENARIOS_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(scenarios[0]).keys()))
        writer.writeheader()
        for scenario in scenarios:
            writer.writerow(asdict(scenario))

    manifest = {
        "run_label": RUN_LABEL,
        "baseline": BASELINE,
        "baseline_h_ext": BASE_H_EXT,
        "h_ext_curve_values": list(H_EXT_VALUES),
        "nuisance_params": list(NUISANCE_PARAMS),
        "nuisance_factors": list(NUISANCE_FACTORS),
        "params_to_vary": list(PARAMS_TO_VARY),
        "factor_values": list(FACTOR_VALUES),
        "from_t_values": list(FROM_T_VALUES),
        "n": n,
        "tmax": TMAX,
        "dt": DT,
        "save_times": SAVE_TIMES,
        "simulation_count": len(specs),
        "metrics_path": str(METRICS_PATH),
        "scenarios_path": str(SCENARIOS_PATH),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    README_PATH.write_text(build_readme(manifest))


def build_readme(manifest: dict[str, object]) -> str:
    return f"""# Sweden 2019 Steepness-Longevity Sensitivity Run

This folder stores checkpointed simulation results for the Fig 1D redo.

The central baseline is the Sweden 2019 period tail-focused SR fit:

\\[
\\eta={BASELINE["eta"]:.6g},\\quad
\\beta={BASELINE["beta"]:.6g},\\quad
\\kappa={BASELINE["kappa"]:.6g},\\quad
\\epsilon={BASELINE["epsilon"]:.6g},\\quad
X_c={BASELINE["Xc"]:.6g},\\quad
\\sigma_{{X_c}}/X_c={BASELINE["xc_std_frac"]:.6g}.
\\]

The baseline and parameter-factor curves use \\(h_{{ext}}=0\\). The extrinsic
mortality curve is a separate absolute sweep over \\(h_{{ext}}\\in[10^{{-4}},10^{{-2}}]\\).

Baseline sensitivity is one-at-a-time over:

\\[
\\eta,\\beta,\\epsilon,X_c,\\sigma_{{X_c}}/X_c
\\]

with factors \\(0.8\\) and \\(1.2\\). Every finished simulation appends rows to
`metrics_long.csv`, one row per starting age \\(t_0\\).

Current configured simulation count: {manifest["simulation_count"]}.
"""


def ensure_metrics_header() -> None:
    if METRICS_PATH.exists():
        return

    fieldnames = list(base_output_fields()) + list(METRIC_FIELDS)
    with METRICS_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def base_output_fields() -> tuple[str, ...]:
    return (
        "run_id",
        "scenario_id",
        "nuisance_param",
        "nuisance_factor",
        "curve_type",
        "focal_param",
        "focal_value",
        "h_ext",
        "from_t",
        "n",
        "tmax",
        "dt",
        "save_times",
        "random_seed",
    )


def append_metrics(spec: SimulationSpec, metrics: dict[int, dict[str, float | None]], n: int) -> None:
    fieldnames = list(base_output_fields()) + list(METRIC_FIELDS)

    with METRICS_PATH.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)

        for from_t, values in metrics.items():
            row = {
                "run_id": spec.run_id,
                "scenario_id": spec.scenario_id,
                "nuisance_param": spec.nuisance_param,
                "nuisance_factor": spec.nuisance_factor,
                "curve_type": spec.curve_type,
                "focal_param": spec.focal_param,
                "focal_value": spec.focal_value,
                "h_ext": spec.h_ext,
                "from_t": from_t,
                "n": n,
                "tmax": TMAX,
                "dt": DT,
                "save_times": SAVE_TIMES,
                "random_seed": spec.random_seed,
            }
            row.update(values)
            writer.writerow(row)


def run_spec(spec: SimulationSpec, scenario: BaselineScenario, n: int) -> None:
    params = build_param_dict(scenario=scenario, spec=spec, n=n)
    sim = utils.create_sr_simulation(
        params_dict=params,
        n=n,
        h_ext=spec.h_ext,
        parallel=True,
        tmax=TMAX,
        dt=DT,
        save_times=SAVE_TIMES,
        break_early=True,
        random_seed=spec.random_seed,
    )
    metrics = calculate_metrics(sim)
    append_metrics(spec=spec, metrics=metrics, n=n)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = build_scenarios()
    scenario_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    specs = build_specs(scenarios)

    write_setup_files(scenarios=scenarios, specs=specs, n=args.n)

    if args.dry_run:
        print(f"Would run {len(specs)} simulations.")
        print(f"Outputs: {OUTPUT_DIR}")
        return

    ensure_metrics_header()
    done = completed_run_ids()
    pending = [spec for spec in specs if spec.run_id not in done]

    if args.limit is not None:
        pending = pending[: args.limit]

    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Completed simulations found: {len(done)}")
    print(f"Pending simulations in this invocation: {len(pending)}")

    for index, spec in enumerate(pending, start=1):
        print(
            f"[{index}/{len(pending)}] {spec.run_id} "
            f"(h_ext={spec.h_ext:.3g}, seed={spec.random_seed})",
            flush=True,
        )
        run_spec(spec=spec, scenario=scenario_by_id[spec.scenario_id], n=args.n)

    print("Done.")


if __name__ == "__main__":
    main()
