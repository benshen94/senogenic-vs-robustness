#!/usr/bin/env python3
"""Optional original Fedichev-Gruber calculations, independent of the SR solver."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.figures.extended_data.render_fedichev_constraints import FEDICHEV_PARAMS, FEDICHEV_LABELS

RANDOM_SEED = 20260520
TAIL_SURVIVAL_FRACTION = 1e-4
FEDICHEV_N = 300_000
FEDICHEV_DT = .05
FEDICHEV_TMAX = 1000.
FEDICHEV_STD_VALUES = np.arange(0., .2001, .025)

def positive_normal(rng: np.random.Generator, mean: float, rel_std: float, n: int) -> np.ndarray:
    """Draw positive values from a normal distribution."""
    values = rng.normal(loc=mean, scale=rel_std * mean, size=n)
    while np.any(values <= 0):
        bad = values <= 0
        values[bad] = rng.normal(loc=mean, scale=rel_std * mean, size=int(np.sum(bad)))
    return values

def tail_survivor_rank(n: int) -> int:
    """Return the empirical rank corresponding to S(t)=0.0001."""
    return max(1, int(round(n * TAIL_SURVIVAL_FRACTION)))

def parameter_or_scalar(values: np.ndarray | float, indices: np.ndarray) -> np.ndarray | float:
    """Return per-alive values while preserving scalars for speed."""
    if isinstance(values, np.ndarray):
        return values[indices]
    return values

def subset_or_scalar(values: np.ndarray | float, mask: np.ndarray) -> np.ndarray | float:
    """Return per-stable values while preserving scalars for speed."""
    if isinstance(values, np.ndarray):
        return values[mask]
    return values

def fedichev_parameter_values(
    rng: np.random.Generator,
    param_name: str,
    std: float,
    n: int,
) -> dict[str, np.ndarray | float]:
    """Return baseline Fedichev-Gruber parameters with one heterogeneous parameter."""
    params: dict[str, np.ndarray | float] = {
        "epsilon_0": 4.0,
        "D0": 1.1,
        "beta": 0.015,
        "g": 0.8,
        "gamma": 1.0,
        "beta_prime": 0.013333,
    }
    params[param_name] = positive_normal(rng, float(params[param_name]), std, n)
    return params

def simulate_fedichev_tail_lifespan(
    rng: np.random.Generator,
    param_name: str,
    std: float,
    n: int,
    dt: float,
    tmax: float,
) -> float:
    """Simulate the Fedichev-Gruber model until the S(t)=0.0001 tail is known."""
    params = fedichev_parameter_values(rng, param_name, std, n)
    target_rank = tail_survivor_rank(n)
    alive_indices = np.arange(n)
    z_alive = np.zeros(n)
    sqrt_dt = np.sqrt(dt)
    noise_strength = np.sqrt(2 * params["D0"]) if isinstance(params["D0"], np.ndarray) else np.sqrt(2 * float(params["D0"]))

    for step in range(int(tmax / dt)):
        t = step * dt
        if alive_indices.size == 0:
            return float("nan")

        epsilon_0 = parameter_or_scalar(params["epsilon_0"], alive_indices)
        d0_noise = parameter_or_scalar(noise_strength, alive_indices)
        beta = parameter_or_scalar(params["beta"], alive_indices)
        g = parameter_or_scalar(params["g"], alive_indices)
        gamma = parameter_or_scalar(params["gamma"], alive_indices)
        beta_prime = parameter_or_scalar(params["beta_prime"], alive_indices)

        z_driver = gamma * t
        epsilon_eff = epsilon_0 - beta_prime * z_driver
        discriminant = epsilon_eff**2 - 4 * g * (beta * z_driver)
        stable_mask = np.full(alive_indices.size, bool(discriminant > 0)) if np.ndim(discriminant) == 0 else discriminant > 0
        dead_mask = ~stable_mask

        if np.any(stable_mask):
            stable_positions = np.where(stable_mask)[0]
            eps_stable = subset_or_scalar(epsilon_eff, stable_mask)
            disc_stable = subset_or_scalar(discriminant, stable_mask)
            g_stable = subset_or_scalar(g, stable_mask)
            beta_stable = subset_or_scalar(beta, stable_mask)
            z_driver_stable = subset_or_scalar(z_driver, stable_mask)
            noise_stable = subset_or_scalar(d0_noise, stable_mask)

            z_unstable = (eps_stable + np.sqrt(disc_stable)) / (2 * g_stable)
            z_current = z_alive[stable_positions]
            drift = beta_stable * z_driver_stable - eps_stable * z_current + g_stable * z_current**2
            diffusion = noise_stable * rng.normal(0, sqrt_dt, size=stable_positions.size)
            z_new = z_current + drift * dt + diffusion
            z_alive[stable_positions] = z_new
            dead_mask[stable_positions[z_new > z_unstable]] = True

        alive_indices = alive_indices[~dead_mask]
        z_alive = z_alive[~dead_mask]
        if alive_indices.size < target_rank:
            return float(t)

    return float(tmax)

def stable_seed(*parts: object) -> int:
    """Create a reproducible uint32 seed from readable metadata."""
    text = "|".join(str(part) for part in (RANDOM_SEED, *parts))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)

def make_fedichev_max_lifespan_data() -> pd.DataFrame:
    """Regenerate the Fedichev-Gruber heterogeneity panel data."""
    rows = []
    for param_name in FEDICHEV_PARAMS:
        source_param = "epsilon_0" if param_name == "epsilon_0_init" else param_name
        for std in FEDICHEV_STD_VALUES:
            print(f"Running Fedichev-Gruber extreme simulation for {source_param}, CV={std:.3f}")
            seed = stable_seed("fedichev", source_param, round(float(std), 4))
            rng = np.random.default_rng(seed)
            rows.append(
                {
                    "model": "Fedichev-Gruber",
                    "parameter": param_name,
                    "parameter_label": FEDICHEV_LABELS[param_name],
                    "variation_percent": 100 * float(std),
                    "max_lifespan": simulate_fedichev_tail_lifespan(
                        rng=rng,
                        param_name=source_param,
                        std=float(std),
                        n=FEDICHEV_N,
                        dt=FEDICHEV_DT,
                        tmax=FEDICHEV_TMAX,
                    ),
                }
            )

    return pd.DataFrame(rows)

def add_smoothed_max_lifespan(data: pd.DataFrame) -> pd.DataFrame:
    """Add the smoothed curves used for display."""
    smoothed = []
    for _, group in data.groupby(["model", "parameter"], sort=False):
        group = group.sort_values("variation_percent").copy()
        group["max_lifespan_smoothed"] = gaussian_filter1d(group["max_lifespan"], sigma=0.85)
        smoothed.append(group)
    return pd.concat(smoothed, ignore_index=True)


def run_simulation_beta_prime(n_individuals, dt, beta_prime, epsilon_0_init, gamma, beta, g, D0):
    sim_max_time = 200.0
    time_steps = int(sim_max_time / dt)
    times = np.linspace(0, sim_max_time, time_steps)

    z = np.zeros(n_individuals)
    death_times = np.full(n_individuals, np.nan)
    alive_mask = np.ones(n_individuals, dtype=bool)

    sqrt_dt = np.sqrt(dt)
    sqrt_2D0 = np.sqrt(2 * D0)

    for t in times:
        n_alive = np.sum(alive_mask)
        if n_alive == 0:
            break

        Z = gamma * t
        epsilon_eff = epsilon_0_init - beta_prime * Z
        discriminant = epsilon_eff**2 - 4 * g * (beta * Z)

        if discriminant > 0:
            sqrt_disc = np.sqrt(discriminant)
            z_unstable = (epsilon_eff + sqrt_disc) / (2 * g)

            z_alive = z[alive_mask]
            drift = (beta * Z) - (epsilon_eff * z_alive) + (g * z_alive**2)
            diffusion = sqrt_2D0 * np.random.normal(0, sqrt_dt, size=n_alive)

            z_alive_new = z_alive + drift * dt + diffusion
            z[alive_mask] = z_alive_new

            died_now_local = z_alive_new > z_unstable
            if np.any(died_now_local):
                alive_indices = np.where(alive_mask)[0]
                dying_indices = alive_indices[died_now_local]
                death_times[dying_indices] = t
                alive_mask[dying_indices] = False
        else:
            alive_indices = np.where(alive_mask)[0]
            death_times[alive_indices] = t
            alive_mask[alive_indices] = False

    return death_times

def calculate_metrics(death_times):
    valid_deaths = death_times[~np.isnan(death_times)]
    if len(valid_deaths) < 10:
        return np.nan, np.nan
    median = np.median(valid_deaths)
    q75, q25 = np.percentile(valid_deaths, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return median, np.nan
    steepness = median / iqr
    return median, steepness


def shape_response(seed):
    np.random.seed(seed)
    baseline = dict(beta_prime=(4.-np.sqrt(4*.8*.015*120))/120,
                    epsilon_0_init=4., gamma=1., beta=.015, g=.8, D0=1.1)
    median, steepness = calculate_metrics(run_simulation_beta_prime(FEDICHEV_N, .05, **baseline))
    rows = []
    for name in FEDICHEV_PARAMS:
        for factor in np.arange(.6, 1.40001, .1):
            params = dict(baseline)
            params[name] *= factor
            print(f"Shape response: {name}, factor={factor:.1f}, n={FEDICHEV_N}", flush=True)
            med, steep = calculate_metrics(run_simulation_beta_prime(FEDICHEV_N, .05, **params))
            if not np.isfinite([med, steep]).all():
                raise ValueError('Unresolved shape metric; retain the failure rather than dropping a point.')
            rows.append(dict(model='Fedichev-Gruber', parameter=name,
                             parameter_label=FEDICHEV_LABELS[name], factor=factor,
                             x_norm=med/median, y_norm=steep/steepness))
    return pd.DataFrame(rows), dict(base_params=baseline, base_med=median, base_steep=steepness,
                                   seed=seed,
                                   n=FEDICHEV_N, dt=.05, horizon=200.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recompute', action='store_true')
    parser.add_argument('--panel', choices=['extreme', 'shape'], required=True)
    parser.add_argument('--seed', type=int, default=20260604,
                        help='Random seed for the shape sweep.')
    parser.add_argument('--output', type=Path, default=ROOT/'tmp/fedichev_recomputed')
    args = parser.parse_args()
    if not args.recompute:
        parser.error('Numerical sweeps require --recompute; plotting uses saved CSVs.')
    args.output.mkdir(parents=True, exist_ok=False)
    if args.panel == 'extreme':
        frame = add_smoothed_max_lifespan(make_fedichev_max_lifespan_data())
        name = 'extreme_lifespan.csv'
        metadata = dict(n=FEDICHEV_N, dt=FEDICHEV_DT, horizon=FEDICHEV_TMAX,
                        seed=RANDOM_SEED, seed_rule='stable_seed per parameter and CV',
                        smoothing_sigma=.85)
    else:
        frame, metadata = shape_response(args.seed)
        name = 'shape_response.csv'
    frame.to_csv(args.output/name, index=False)
    (args.output/'manifest.json').write_text(json.dumps(metadata, indent=2))
    print(args.output/name)


if __name__ == '__main__':
    main()
