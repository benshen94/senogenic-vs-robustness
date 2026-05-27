import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGING_PYTHON_ROOT = PROJECT_ROOT.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(AGING_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(AGING_PYTHON_ROOT))

from ageing_packages.SR_models.hazard_only import SRHazardSim
from src.exploration.hmd_fits.joint_usa_sweden_sr_fit_2019 import load_country_data, build_xc_vector, _interpolate_on_times
from src.shared.thresholds.paths import SAVED_RESULTS_DIR

OUTPUT_DIR = SAVED_RESULTS_DIR
PLOT_PATH = OUTPUT_DIR / "sweden_xc_redo.png"

AGE_START = 30
FIT_DT = 0.025
FINAL_N = 90_000

# Params from the summary JSON
params = {
    "shared": {
      "eta": 0.54,
      "beta": 54.75,
      "kappa": 0.5
    },
    "SWE": {
      "epsilon": 51.83,
      "Xc": 21.0, # Baseline
      "xc_std_frac": 0.18025009252216606,
      "h_ext_factor": 1.0
    }
}

def simulate_sweden(xc_val):
    data = load_country_data("SWE")
    
    sr_params = {
        "eta": params["shared"]["eta"],
        "beta": params["shared"]["beta"],
        "kappa": params["shared"]["kappa"],
        "epsilon": params["SWE"]["epsilon"],
        "Xc": build_xc_vector(
            mean_xc=xc_val,
            std_frac=params["SWE"]["xc_std_frac"],
            n=FINAL_N,
            seed=20260612,
        ),
    }
    
    sim = SRHazardSim(
        n=FINAL_N,
        eta=sr_params["eta"],
        beta=sr_params["beta"],
        kappa=sr_params["kappa"],
        epsilon=sr_params["epsilon"],
        Xc=sr_params["Xc"],
        h_ext=data.h_ext * params["SWE"]["h_ext_factor"],
        tmax=112,
        dt=FIT_DT,
        parallel=True,
        break_early=True,
        random_seed=20260612 + 10_000,
        chunk_size=10_000,
    )
    
    fitted_hazard = _interpolate_on_times(sim.tspan_hazard, sim.hazard, data.ages_hazard)
    fitted_survival = _interpolate_on_times(sim.tspan_survival, sim.survival, data.ages_survival)
    fitted_survival = np.clip(fitted_survival, 1e-12, 1.0)
    fitted_survival = fitted_survival / fitted_survival[0]
    
    return {
        "hazard": np.maximum(fitted_hazard, 1e-12),
        "survival": fitted_survival,
        "data": data
    }

def main():
    xc_values = [21.0, 22.0, 23.0]
    results = {}
    for xc in xc_values:
        print(f"Simulating Xc = {xc}...")
        results[xc] = simulate_sweden(xc)
        
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    survival_ax = axes[0]
    hazard_ax = axes[1]
    
    data = results[21.0]["data"]
    
    survival_ax.plot(data.ages_survival, data.survival, "o", ms=3, color="black", label=f"Sweden HMD")
    hazard_ax.plot(data.ages_hazard, data.hazard, "o", ms=3, color="black", label=f"Sweden HMD")
    
    colors = {21.0: "blue", 22.0: "orange", 23.0: "green"}
    
    for xc in xc_values:
        fit = results[xc]
        survival_ax.plot(data.ages_survival, fit["survival"], lw=2.5, color=colors[xc], label=f"SR fit (Xc={xc})")
        hazard_ax.plot(data.ages_hazard, fit["hazard"], lw=2.5, color=colors[xc], label=f"SR fit (Xc={xc})")
        
    survival_ax.set_ylabel(f"Sweden\nSurvival from age {AGE_START}")
    survival_ax.grid(alpha=0.25)
    survival_ax.legend(frameon=False)

    hazard_ax.axvspan(60, 90, color="0.90", zorder=-1)
    hazard_ax.set_yscale("log")
    hazard_ax.set_ylabel("Hazard [1/year]")
    hazard_ax.grid(alpha=0.25)
    hazard_ax.legend(frameon=False)

    axes[0].set_title("Survival")
    axes[1].set_title("Hazard")
    axes[0].set_xlabel("Age [years]")
    axes[1].set_xlabel("Age [years]")

    fig.suptitle("Sweden SR fit with different Xc values")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {PLOT_PATH}")

if __name__ == "__main__":
    main()
