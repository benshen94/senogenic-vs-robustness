"""
Utility functions for the SR model, including loading baseline parameters, 
creating parameter dictionaries and running SR simulations.

This module provides functions for:
- Loading and managing SR model parameters
- Loading named SR fit presets
- Creating parameter distributions for twin studies
- Running SR simulations with various configurations
"""

### Description: Utility functions for the SR model, including loading baseline parameters, creating parameter dictionaries and running SR
import pandas as pd
import numpy as np
from ..SR_models.simulation import SR_sim, SimulationParams
from ..SR_models.go_with_winners import SR_go_ww
from scipy.stats import invgamma # type: ignore

# Constants
eta_karin = 0.00135 * 365
beta_karin = 0.15 * 365
kappa_karin = 0.5
eps_karin = 0.142 * 365
Xc_karin = 17

karin_params = {
    'eta': np.array([eta_karin]),
    'beta': np.array([beta_karin]),
    'kappa': np.array([kappa_karin]),
    'epsilon': np.array([eps_karin]),
    'Xc': np.array([Xc_karin])
}
# param colors for plotting
param_colors = {
    'eta': 'blue',
    'beta': 'green',
    'epsilon': '#CC79A7',
    'Xc': 'purple',
    'kappa' : 'orange'
}
# param descriptions
param_descriptions = {
'eta' : 'Production',
'beta' : 'Removal',
'epsilon' : 'Noise',
'Xc' : 'Threshold',
'kappa' : 'Sensitivity'
}
# param_names for plotting
param_names = {
    'eta' : r'$\eta$',
    'beta' : r'$\beta$',
    'epsilon' : r'$\epsilon$',
    'Xc' : r'$X_c$',
    'kappa' : r'$\kappa$'
}

twin_line_styles = {
    'MZ': '--',
    'DZ' : ':',
    'None' : '-'
}

twin_alphas = {
    'MZ': 1,
    'DZ' : 0.3,
    'None' : 1
}

def load_SR_params(species_name):
    """
    Load SR parameters for a specific species from CSV file.
    
    Args:
        species_name (str): Name of the species to load parameters for

    Returns:
        dict: Dictionary containing SR parameters for the species

    Raises:
        ValueError: If species is not found in the database
    """
    df = pd.read_csv('../../../datasets/SR_params.csv')
    species_data = df[df['species'] == species_name]
    
    if species_data.empty:
        raise ValueError(f"Species '{species_name}' not found in the data.")
    
    species_params = species_data.iloc[0].to_dict()
    relevant_params = {key: value for key, value in species_params.items() 
                       if key not in ['species', 'source', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10']}
    return relevant_params

def load_baseline_human_params_dict(params_dict=karin_params):
    return params_dict.copy()


def list_sr_fit_names():
    """Return the available named SR fit presets."""
    from .sr_fits import list_sr_fit_names as _list_sr_fit_names

    return _list_sr_fit_names()


def load_sr_fit(fit_name):
    """Return a named SR fit preset."""
    from .sr_fits import get_sr_fit

    return get_sr_fit(fit_name)


def build_sr_simulation_inputs_from_fit(
    fit_name,
    n=None,
    family=None,
    param_updates=None,
    h_ext=None,
):
    """
    Build simulation inputs from a named SR fit preset.

    Returns a dictionary with:
    - `fit`: the fit metadata
    - `n`: the resolved simulation size
    - `params_dict`: scalar or heterogeneous SR parameters
    - `h_ext`: the resolved extrinsic hazard
    """
    fit = load_sr_fit(fit_name)

    if n is None:
        n = fit['default_n']
    n = int(n)

    if family is None:
        family = fit.get('default_family', 'None')

    base_params = {
        key: float(fit['params'][key])
        for key in ['eta', 'beta', 'kappa', 'epsilon', 'Xc']
    }

    if param_updates:
        for key, value in param_updates.items():
            if key in base_params:
                base_params[key] = float(value)

    heterogeneity = fit.get('heterogeneity', {})
    hetero_param = heterogeneity.get('param')
    hetero_std = float(heterogeneity.get('std', 0.0))
    hetero_dist = heterogeneity.get('dist_type', 'gaussian')

    if hetero_param and hetero_std > 0:
        params_dict = create_param_distribution_dict(
            params=hetero_param,
            std=hetero_std,
            n=n,
            dist_type=hetero_dist,
            params_dict=base_params,
            family=family,
        )
    else:
        params_dict = base_params

    resolved_h_ext = fit.get('h_ext') if h_ext is None else h_ext

    return {
        'fit': fit,
        'n': n,
        'params_dict': params_dict,
        'h_ext': resolved_h_ext,
    }


def create_sr_simulation_from_fit(
    fit_name,
    n=None,
    family=None,
    param_updates=None,
    h_ext=None,
    **kwargs,
):
    """
    Create an SR simulation directly from a named SR fit preset.

    This is a thin wrapper around `create_sr_simulation`.
    """
    sim_inputs = build_sr_simulation_inputs_from_fit(
        fit_name=fit_name,
        n=n,
        family=family,
        param_updates=param_updates,
        h_ext=h_ext,
    )

    return create_sr_simulation(
        n=sim_inputs['n'],
        params_dict=sim_inputs['params_dict'],
        h_ext=sim_inputs['h_ext'],
        **kwargs,
    )


def _mean_scalar(value):
    """Return a scalar mean value for a scalar or array parameter."""
    if isinstance(value, np.ndarray):
        return float(np.mean(value))
    if isinstance(value, (list, tuple)):
        return float(np.mean(np.asarray(value, dtype=float)))
    return float(value)


def plot_hmd_vs_sr_fit_panel(
    *,
    country,
    gender,
    data_type,
    year,
    params_dict,
    h_ext=None,
    save_path=None,
    age_start=None,
    age_end=None,
    hazard_mode='interval',
    n=5000,
    family='None',
    tmax=140,
    dt=0.25,
    save_times=0.25,
    parallel=False,
    break_early=True,
    survival_log_scale=False,
):
    """
    Save or return a 1x2 panel that compares HMD data to an SR simulation fit.

    Left panel: survival from `age_start`.
    Right panel: hazard on a log scale.
    """
    from pathlib import Path
    import matplotlib.pyplot as plt

    from ..mortality_data_analysis.HMD_lifetables import HMD

    hmd = HMD(country, gender, data_type)

    ages_survival, survival = hmd.get_survival(year)
    ages_hazard, hazard = hmd.get_hazard(year, haz_type='mx')

    if age_start is None:
        age_start = int(np.min(ages_survival))
    if age_end is None:
        age_end = int(np.max(ages_survival))

    survival_mask = (
        (ages_survival >= age_start)
        & (ages_survival <= age_end)
        & np.isfinite(survival)
        & (survival > 0)
    )
    hazard_mask = (
        (ages_hazard >= age_start)
        & (ages_hazard <= age_end)
        & np.isfinite(hazard)
        & (hazard > 0)
    )

    ages_survival = ages_survival[survival_mask].astype(float)
    survival = survival[survival_mask].astype(float)
    ages_hazard = ages_hazard[hazard_mask].astype(float)
    hazard = hazard[hazard_mask].astype(float)

    if len(ages_survival) == 0 or len(ages_hazard) == 0:
        raise ValueError("No HMD data left after age-range filtering.")

    if survival[0] > 0:
        survival = survival / survival[0]

    sim = create_sr_simulation(
        species='human',
        n=n,
        params_dict=params_dict,
        h_ext=h_ext,
        tmax=tmax,
        dt=dt,
        save_times=save_times,
        parallel=parallel,
        break_early=break_early,
    )

    sim_survival_df = sim.survival
    sim_survival_times = np.asarray(sim_survival_df.index.values, dtype=float)
    sim_survival_values = np.asarray(sim_survival_df.iloc[:, 0].values, dtype=float)

    survival_idx = np.searchsorted(sim_survival_times, ages_survival, side='right') - 1
    survival_idx = np.clip(survival_idx, 0, len(sim_survival_times) - 1)
    sim_survival_on_hmd = sim_survival_values[survival_idx]
    if sim_survival_on_hmd[0] > 0:
        sim_survival_on_hmd = sim_survival_on_hmd / sim_survival_on_hmd[0]

    if hazard_mode == 'interval':
        sim_hazard_times = np.asarray(sim.tspan_interval_hazard, dtype=float)
        sim_hazard_values = np.asarray(sim.interval_hazard, dtype=float).reshape(-1)
    else:
        sim_hazard_times = np.asarray(sim.tspan_hazard, dtype=float)
        sim_hazard_values = np.asarray(sim.hazard, dtype=float).reshape(-1)

    hazard_plot_mask = (
        (sim_hazard_times >= age_start)
        & (sim_hazard_times <= age_end)
    )
    sim_hazard_times = sim_hazard_times[hazard_plot_mask]
    sim_hazard_values = sim_hazard_values[hazard_plot_mask]
    sim_hazard_values = np.maximum(sim_hazard_values, 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    axes[0].plot(ages_survival, survival, 'o', markersize=3, label='HMD data')
    axes[0].plot(ages_survival, sim_survival_on_hmd, '-', linewidth=2, label='SR fit')
    axes[0].set_xlabel('Age [years]')
    axes[0].set_ylabel(f'Survival from age {age_start}')
    axes[0].set_title('Survival')
    if survival_log_scale:
        axes[0].set_yscale('log')
    axes[0].set_xlim(age_start, age_end)
    axes[0].grid(alpha=0.3)
    axes[0].legend(frameon=False)

    axes[1].plot(ages_hazard, hazard, 'o', markersize=3, label='HMD data')
    axes[1].plot(sim_hazard_times, sim_hazard_values, '-', linewidth=2, label='SR fit')
    axes[1].set_xlabel('Age [years]')
    axes[1].set_ylabel('Hazard [1/year]')
    axes[1].set_title('Hazard')
    axes[1].set_yscale('log')
    axes[1].set_xlim(age_start, age_end)
    axes[1].grid(alpha=0.3)
    axes[1].legend(frameon=False)

    param_lines = [
        f"eta = {_mean_scalar(params_dict['eta']):.6f}",
        f"beta = {_mean_scalar(params_dict['beta']):.6f}",
        f"kappa = {_mean_scalar(params_dict['kappa']):.6f}",
        f"epsilon = {_mean_scalar(params_dict['epsilon']):.6f}",
        f"Xc_mean = {_mean_scalar(params_dict['Xc']):.6f}",
        f"h_ext = {0.0 if h_ext is None else float(h_ext):.6f}",
        f"n = {int(n):,}",
        f"family = {family}",
        f"hazard_mode = {hazard_mode}",
        f"fit ages = {age_start}-{age_end}",
    ]
    axes[1].text(
        1.02,
        0.98,
        '\n'.join(param_lines),
        transform=axes[1].transAxes,
        va='top',
        fontsize=10,
        bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.9},
    )

    fig.suptitle(f'SR fit vs HMD: {country} {gender} {data_type} {year}', fontsize=14)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')

    return fig, axes


def save_named_sr_fit_panel(
    fit_name,
    save_path=None,
    n=None,
    family=None,
    hazard_mode=None,
    survival_log_scale=False,
    **kwargs,
):
    """
    Save a 1x2 survival/hazard comparison panel for a named SR fit preset.
    """
    from pathlib import Path

    sim_inputs = build_sr_simulation_inputs_from_fit(
        fit_name=fit_name,
        n=n,
        family=family,
    )
    fit = sim_inputs['fit']
    context = fit.get('fit_context', {})

    if hazard_mode is None:
        hazard_mode = context.get('hazard_mode') or 'interval'

    if save_path is None:
        root_dir = Path(__file__).resolve().parents[3]
        save_path = root_dir / 'results' / 'sr_fits' / f"{fit['name']}_panel.png"

    return plot_hmd_vs_sr_fit_panel(
        country=context['country'],
        gender=context['gender'],
        data_type=context['data_type'],
        year=context['year'],
        params_dict=sim_inputs['params_dict'],
        h_ext=sim_inputs['h_ext'],
        save_path=save_path,
        age_start=context.get('age_start'),
        age_end=context.get('age_end'),
        hazard_mode=hazard_mode,
        n=sim_inputs['n'],
        family=fit.get('default_family', 'None') if family is None else family,
        survival_log_scale=survival_log_scale,
        **kwargs,
    )

# create param distribution dictionary for a given parameter, with a given standard deviation, and a given distribution type. specify MZ, DZ twins, or just distribution
def create_param_distribution_dict(params, std, n=40000, dist_type='gaussian', params_dict=None, family='MZ' , corr = None, nu=None):
    """
    Create parameter distributions for twin studies.
    
    Args:
        params (str/list): Parameter(s) to create distributions for ('eta', 'beta', etc.)
        std (float): Standard deviation for the distribution
        n (int): Number of samples (will be divided by 2 for twins)
        dist_type (str): Type of distribution ('gaussian', 'lognormal', 'lognormal_flipped', 't-test')
        params_dict (dict): Base parameter dictionary (defaults to karin_params)
        family (str): Type of twins ('MZ', 'DZ', or 'None')
        corr (float): Correlation coefficient for 'corr' family type
        nu (float): Degrees of freedom for t-test distribution (required if dist_type='t-test')

    Returns:
        dict: Dictionary with distributed parameters
    """
    if params_dict is None:
        params_dict = karin_params

    new_params_dict = params_dict.copy()
    n_pairs = int(n/2)

    def generate_base_distribution(n_samples):
        if dist_type == 'gaussian':
            return np.random.normal(1, std, n_samples)
        elif dist_type == 'lognormal':
            return np.random.lognormal(0, std, n_samples)
        elif dist_type == 'lognormal_flipped':
            return 2 - np.random.lognormal(0, std, n_samples)
        elif dist_type == 't-test':
            if nu is None:
                raise ValueError("nu (degrees of freedom) must be provided for t-test distribution")
            # Scale-mixture of normals: heteroskedastic individuals
            # Each person has a random variance V ~ InvGamma(nu/2, nu/2)
            # Y | V ~ N(mu, sigma^2 * V)
            # Marginally, Y ~ Student-t_nu
            V = invgamma.rvs(a=nu/2, scale=nu/2, size=n_samples)
            # Generate Y | V ~ N(1, std^2 * V) centered at 1
            return np.random.normal(loc=1, scale=std * np.sqrt(V))
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

    def ensure_positive(dist):
        while np.any(dist < 0):
            neg_mask = dist < 0
            dist[neg_mask] = generate_base_distribution(np.sum(neg_mask))
        return dist

    if family == 'MZ':
        # Generate one set of values and repeat for both twins
        dist = ensure_positive(generate_base_distribution(n_pairs))
        dist = np.repeat(dist, 2)
    elif family == 'DZ':
        # Generate correlated values using bivariate normal with rho = 0.5, works for gaussian only
        Z1 = np.random.randn(n_pairs)
        Z2 = np.random.randn(n_pairs)
        dist1 = 1 + std * Z1
        dist2 = 1 + std * (0.5 * Z1 + np.sqrt(1 - 0.5**2) * Z2)
        dist = ensure_positive(np.ravel(np.column_stack((dist1, dist2))))
        
    elif family == 'corr':
        # Generate correlated values using bivariate normal with rho = corr, works for gaussian only
        Z1 = np.random.randn(n_pairs)
        Z2 = np.random.randn(n_pairs)
        dist1 = 1 + std * Z1
        dist2 = 1 + std * (corr * Z1 + np.sqrt(1 - corr**2) * Z2)
        dist = ensure_positive(np.ravel(np.column_stack((dist1, dist2))))
        
    elif family == 'None':
        # Generate independent values for both twins
        dist1 = ensure_positive(generate_base_distribution(n_pairs))
        dist2 = ensure_positive(generate_base_distribution(n_pairs))
        dist = np.ravel(np.column_stack((dist1, dist2)))
    else:
        raise ValueError("family must be one of 'MZ', 'DZ', or 'None'")

    if isinstance(params, str):
        params = [params]
    
    for param in params:
        # Get the base value - needs careful handling for h_ext which might not be in karin_params by default
        if param == 'h_ext':
            # Default h_ext to 0 if not present, assuming no external hazard is the base
            base_value = params_dict.get(param, 0.0)
            # Ensure base_value is scalar for distribution generation
            if hasattr(base_value, '__iter__') and not isinstance(base_value, str):
                 base_value = base_value[0] # Use first element if it's an array
            elif not np.isscalar(base_value):
                 raise ValueError(f"Base value for h_ext must be a scalar or array, got {type(base_value)}")
        else:
             # Original logic for other params
             base_value = params_dict[param][0] if hasattr(params_dict[param], '__iter__') else params_dict[param]

        # Apply distribution: multiply base value by the generated distribution factor
        # Ensure non-negativity for hazard rates
        if param == 'h_ext':
             new_params_dict[param] = ensure_positive(dist * base_value) 
        else:
             new_params_dict[param] = dist * base_value

    # --- Final Expansion and Validation --- 
    target_n = len(dist) # n for the simulation
    # List all parameters expected by SimulationParams + h_ext
    all_expected_params = ['eta', 'beta', 'kappa', 'epsilon', 'Xc', 'h_ext']
    
    for p in all_expected_params:
        if p in new_params_dict:
            current_val = new_params_dict[p]
            param_was_distributed = (p in params) # Check if this param was generated

            if isinstance(current_val, (np.ndarray, list)):
                current_len = len(current_val)
                if current_len == target_n:
                    pass # Length is already correct
                elif current_len == 1 and not param_was_distributed:
                    # Expand length-1 array if it wasn't the distributed param
                    # print(f"Info: Expanding parameter '{p}' from length 1 to {target_n}.") # Optional info
                    new_params_dict[p] = np.full(target_n, current_val[0])
                else:
                    # Length mismatch is an error, either internal (if distributed) or input (if not)
                    error_type = "Internal" if param_was_distributed else "Input"
                    raise ValueError(f"{error_type} error: Parameter '{p}' has unexpected length {current_len}, expected {target_n}")
                    
            elif np.isscalar(current_val):
                # Expand scalar to full array
                new_params_dict[p] = np.full(target_n, current_val)
            else:
                # Handle unexpected types
                raise TypeError(f"Parameter '{p}' has unexpected type: {type(current_val)}")
                
        elif p in params_dict:
            # If parameter wasn't distributed and wasn't even copied initially (e.g., h_ext not in base dict), expand from base.
            base_val = params_dict[p]
            if np.isscalar(base_val):
                 new_params_dict[p] = np.full(target_n, base_val)
            elif isinstance(base_val, (np.ndarray, list)) and len(base_val) == 1:
                 new_params_dict[p] = np.full(target_n, base_val[0])
            else:
                 raise ValueError(f"Base parameter '{p}' (not distributed) cannot be expanded to length {target_n}. Value: {base_val}")
        
        elif p != 'h_ext': # If a core param is missing entirely
             raise ValueError(f"Core parameter '{p}' is missing from parameter dictionaries.")

    return new_params_dict

def create_sr_simulation(species='human', n=1e5, save_times=100, params_dict=None, param_updates=None,
                        drift_expr=None, drift_mode='replace', extra_params=None, h_ext=None,
                        x0=None, use_fast_kernel=True, dt_schedule=None, adaptive_dt=None,
                        Xdisease=None,
                        **kwargs):
    """
    Create and configure an SR simulation.

    Args:
        species (str): Species to simulate ('human' or other species in database)
        n (int): Number of simulations to run
        save_times (int): Interval for saving simulation states
        params_dict (dict): Optional custom parameter dictionary providing base values 
                          for eta, beta, kappa, epsilon, Xc. Can also include 'x0' as an array.
        param_updates (dict): Updates to specific parameters (eta, beta, kappa, epsilon, Xc).
                              Values can be scalar or array of length n.
        drift_expr (str): Custom drift expression
        drift_mode (str): How to apply custom drift ('replace' or 'add')
        extra_params (dict): Additional parameters for custom drift
        h_ext (float/callable/array/None): External hazard rate override. 
               Takes precedence over h_ext possibly present in params_dict or param_updates.
               See SimulationParams docstring for details.
        x0 (float/array/None): Initial condition. Can be scalar (same for all) or 
               array of length n. If None, uses default. Takes precedence over params_dict.
        Xdisease (float/array/None): Optional disease threshold. When supplied,
               the returned simulation includes `sim.disease_times`.
        use_fast_kernel (bool): Use numba-optimized kernel when possible (default True).
               Set to False if you need to debug or if you experience issues.
        dt_schedule (list, optional): Piecewise time-step schedule.
               Each item can be `(t_end, dt)` or `{'t_end': ..., 'dt': ...}`.
        adaptive_dt (list, optional): Alias for `dt_schedule`.
        random_seed (int, optional): Seed for simulation randomness. Passed through `**kwargs`.
        **kwargs: Additional simulation parameters passed directly to SimulationParams 
                  (e.g., tmin, tmax, dt, parallel, break_early, units).

    Returns:
        SR_sim: Configured simulation object
    """
    n = int(n)
    
    # Set defaults based on species
    if species == 'human':
        base_params = load_baseline_human_params_dict()
        defaults = {'tmin': 0, 'tmax': 140, 'x0': 1e-6, 'dt': 0.025, 'save_times': save_times, 
                   'units': 'years', 'parallel': True, 'break_early': True, 
                   'use_fast_kernel': use_fast_kernel}
    else:
        base_params = load_SR_params(species)
        defaults = {'tmin': 0, 'tmax': 1400, 'x0': 1e-6, 'dt': 1, 'save_times': save_times,
                   'units': 'days', 'parallel': True, 'break_early': True,
                   'use_fast_kernel': use_fast_kernel}

    # Build final parameters
    final_params = {**defaults, **kwargs, 'n': n}
    final_params.update(base_params)

    if dt_schedule is not None and adaptive_dt is not None:
        raise ValueError("Use only one of dt_schedule or adaptive_dt.")

    if dt_schedule is not None:
        final_params['dt_schedule'] = dt_schedule
    elif adaptive_dt is not None:
        final_params['dt_schedule'] = adaptive_dt
    
    if params_dict:
        # Handle parameter array expansion and consistency
        sr_params = ['eta', 'beta', 'kappa', 'epsilon', 'Xc', 'Xdisease']
        array_lens = {}
        
        # Check lengths of all SR parameters
        for param in sr_params:
            if param in params_dict:
                val = params_dict[param]
                if isinstance(val, (np.ndarray, list)):
                    array_lens[param] = len(val)
                else:
                    array_lens[param] = 1  # scalar or size-1 array
        
        # Find target length
        lengths = list(array_lens.values())
        if lengths:
            if all(l == 1 for l in lengths):
                target_n = n  # All scalars/size-1, use input n
            elif any(l == n for l in lengths):
                target_n = n  # At least one is size n, use input n
            else:
                # Check for inconsistent lengths (not 1 and not n)
                inconsistent = [param for param, length in array_lens.items() 
                              if length != 1 and length != n]
                if inconsistent:
                    raise ValueError(f"Inconsistent array lengths in params_dict: {inconsistent} have lengths {[array_lens[p] for p in inconsistent]}")
                target_n = n
        
        # Expand parameters to target_n
        expanded_params = {}
        for param in sr_params:
            if param in params_dict:
                val = params_dict[param]
                if isinstance(val, (np.ndarray, list)):
                    if len(val) == 1:
                        expanded_params[param] = np.full(target_n, val[0])
                    elif len(val) == target_n:
                        expanded_params[param] = val
                    else:
                        raise ValueError(f"Parameter {param} has length {len(val)}, expected 1 or {target_n}")
                else:
                    expanded_params[param] = np.full(target_n, val)
        
        final_params.update(expanded_params)
    
    if param_updates:
        final_params.update({k: v for k, v in param_updates.items() 
                           if k in ['eta', 'beta', 'kappa', 'epsilon', 'Xc', 'Xdisease']})
    
    # Handle h_ext precedence
    final_params['h_ext'] = h_ext if h_ext is not None else final_params.get('h_ext')
    
    # Handle x0 precedence - explicit x0 argument takes highest priority
    if x0 is not None:
        final_params['x0'] = x0
    elif params_dict and 'x0' in params_dict:
        # x0 from params_dict
        x0_val = params_dict['x0']
        if isinstance(x0_val, (np.ndarray, list)):
            if len(x0_val) == 1:
                final_params['x0'] = float(x0_val[0])
            elif len(x0_val) == n:
                final_params['x0'] = np.array(x0_val)
            else:
                raise ValueError(f"x0 in params_dict has length {len(x0_val)}, expected 1 or {n}")
        else:
            final_params['x0'] = x0_val

    if Xdisease is not None:
        final_params['Xdisease'] = Xdisease
    
    # Add drift parameters
    if drift_expr:
        final_params.update({'drift_expr': drift_expr, 'drift_mode': drift_mode, 'extra_params': extra_params})

    return SR_sim(SimulationParams(**final_params))

def create_go_ww_simulation(species='human', n=5000, params_dict=None, param_updates=None,
                            drift_expr=None, drift_mode='replace', extra_params=None,
                            calc_pdf=False, bin_nums=100, num_of_pdfs=150,
                            print_out=False, save_paths=False,
                            n_param_bins=50, min_particles_per_bin=10, x0=None,
                            h_ext=None, random_seed=None, **kwargs):
    """
    Create and configure a Go-with-Winners SR simulation.
    
    This is a helper function similar to create_sr_simulation but for the
    go-with-winners algorithm, which is more efficient for estimating
    survival/hazard at late times (rare events).
    
    Supports heterogeneous parameters - when parameters are arrays of length n,
    particles are grouped into bins based on parameter similarity. Dead particles
    are resampled only from survivors in the same bin.
    
    Now uses numba-optimized kernels for faster performance.

    Args:
        species (str): Species to simulate ('human' or other species in database)
        n (int): Number of particles for the simulation
        params_dict (dict): Optional custom parameter dictionary providing base values 
                          for eta, beta, kappa, epsilon, Xc. Can contain arrays of 
                          length n for heterogeneous simulations. Can also include 'x0'.
        param_updates (dict): Updates to specific parameters (eta, beta, kappa, epsilon, Xc).
                              Values can be scalar or array of length n.
        drift_expr (str): Custom drift expression
        drift_mode (str): How to apply custom drift ('replace' or 'add')
        extra_params (dict): Additional parameters for custom drift
        calc_pdf (bool): Whether to calculate PDFs at specified times
        bin_nums (int): Number of bins for PDF histograms
        num_of_pdfs (int): Number of time points for PDF calculation
        print_out (bool): Whether to print progress
        save_paths (bool): Whether to save particle paths
        n_param_bins (int): Number of bins for grouping heterogeneous parameters.
                           More bins = more accurate parameter preservation but 
                           fewer particles per bin. Default 50.
        min_particles_per_bin (int): Minimum particles per bin. If a bin would have
                           fewer, bins are merged. Default 10.
        x0 (float/array/None): Initial condition. Can be scalar (same for all) or 
               array of length n (note: go-with-winners typically uses scalar x0).
               If None, uses default.
        h_ext (float/array/None): Constant external hazard rate override.
        random_seed (int, optional): Seed for go-with-winners randomness.
        **kwargs: Additional simulation parameters (tmin, tmax, dt, units).

    Returns:
        SR_go_ww: Configured go-with-winners simulation object
        
    Example (homogeneous):
        >>> sim = create_go_ww_simulation(species='human', n=10000, tmax=150)
        >>> plt.plot(sim.tspan, sim.survival)
        
    Example (heterogeneous with parameter distribution):
        >>> params = create_param_distribution_dict('Xc', std=0.1, n=10000, family='None')
        >>> sim = create_go_ww_simulation(params_dict=params, n=10000, tmax=150)
    """
    n = int(n)
    
    # Set defaults based on species
    if species == 'human':
        base_params = load_baseline_human_params_dict()
        defaults = {'tmin': 0, 'tmax': 140, 'x0': 1e-6, 'dt': 0.025, 'units': 'years'}
    else:
        base_params = load_SR_params(species)
        defaults = {'tmin': 0, 'tmax': 1400, 'x0': 1e-6, 'dt': 1, 'units': 'days'}

    # Build final parameters
    final_params = {**defaults, **kwargs}
    
    # Get SR parameters from base
    sr_params_names = ['eta', 'beta', 'kappa', 'epsilon', 'Xc']
    for param in sr_params_names:
        if param in base_params:
            val = base_params[param]
            # Convert to scalar if it's a length-1 array
            if isinstance(val, (np.ndarray, list)) and len(val) == 1:
                final_params[param] = float(val[0])
            else:
                final_params[param] = val
    
    # Override with params_dict if provided
    if params_dict:
        for param in sr_params_names:
            if param in params_dict:
                val = params_dict[param]
                # Keep arrays as-is for heterogeneous support
                if isinstance(val, (np.ndarray, list)):
                    if len(val) == 1:
                        final_params[param] = float(val[0])
                    elif len(val) == n:
                        final_params[param] = np.array(val)
                    else:
                        raise ValueError(f"Parameter {param} has length {len(val)}, expected 1 or {n}")
                else:
                    final_params[param] = val
    
    # Apply param_updates
    if param_updates:
        for param, val in param_updates.items():
            if param in sr_params_names:
                final_params[param] = val
    
    # Add drift parameters
    if drift_expr:
        final_params['drift_expr'] = drift_expr
        final_params['drift_mode'] = drift_mode
        final_params['extra_params'] = extra_params
    
    # Handle x0 precedence
    if x0 is not None:
        final_params['x0'] = float(np.atleast_1d(x0)[0]) if np.isscalar(x0) or np.size(x0) == 1 else x0
    elif params_dict and 'x0' in params_dict:
        x0_val = params_dict['x0']
        final_params['x0'] = float(np.atleast_1d(x0_val)[0]) if np.isscalar(x0_val) or np.size(x0_val) == 1 else x0_val

    final_params['h_ext'] = h_ext if h_ext is not None else final_params.get('h_ext', 0.0)
    
    # Add go-with-winners specific parameters
    final_params['calc_pdf'] = calc_pdf
    final_params['bin_nums'] = bin_nums
    final_params['num_of_pdfs'] = num_of_pdfs
    final_params['print_out'] = print_out
    final_params['save_paths'] = save_paths
    
    return SR_go_ww(
        eta=final_params['eta'],
        beta=final_params['beta'],
        kappa=final_params['kappa'],
        epsilon=final_params['epsilon'],
        Xc=final_params['Xc'],
        n=n,
        tmin=final_params['tmin'],
        tmax=final_params['tmax'],
        x0=final_params['x0'],
        dt=final_params['dt'],
        units=final_params['units'],
        calc_pdf=calc_pdf,
        bin_nums=bin_nums,
        num_of_pdfs=num_of_pdfs,
        print_out=print_out,
        save_paths=save_paths,
        drift_expr=final_params.get('drift_expr'),
        drift_mode=final_params.get('drift_mode', 'replace'),
        extra_params=final_params.get('extra_params'),
        n_param_bins=n_param_bins,
        min_particles_per_bin=min_particles_per_bin,
        h_ext=final_params.get('h_ext', 0.0),
        random_seed=random_seed,
    )


# Define the Gompertz hazard function here
def gompertz_hazard(t, m, a, b):
    """Calculates the Gompertz hazard rate."""
    return m + a * np.exp(b * t)
