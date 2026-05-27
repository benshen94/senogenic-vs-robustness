"""
Historical HMD Fitting Script

Fits SR model parameters to historical Human Mortality Database (HMD) survival/hazard curves.
Supports weighted fitting by population size (lx) and provides visualization tools.

Usage:
    python3 historical_hmd_fitting.py
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGING_PYTHON_ROOT = PROJECT_ROOT.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(AGING_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(AGING_PYTHON_ROOT))

from ageing_packages.mortality_data_analysis.HMD_lifetables import HMD
from ageing_packages.utils.sr_fitter import fit_with_restarts, fit_hybrid_two_stage
from ageing_packages.utils.sr_utils import karin_params, create_sr_simulation

# ============================================================================
# CONFIGURATION
# ============================================================================

# HMD Data Parameters
COUNTRY = 'SWE'           # Country code (e.g., 'USA', 'SWE', 'DNK', 'GBR_NP')
GENDER = 'both'           # 'male', 'female', or 'both'
DATA_TYPE = 'cohort'      # 'period' or 'cohort'
YEARS = [1840]

# Initial Parameters for Fitting
INITIAL_PARAMS = {
    'eta': 0.66,           # year⁻²
    'beta': 62.05,         # year⁻¹
    'kappa': 0.5,          # dimensionless
    'epsilon': 51.83,      # day⁻¹
    'Xc': 14.0,            # dimensionless
    'h_ext': 1e-2          # year⁻¹
}

# Parameter Bounds
PARAM_BOUNDS = {
    'Xc': (0.1, 30.0),     # Xc cannot exceed 30
    'eta': (0.01, 10.0),
    'beta': (1.0, 200.0),
    'kappa': (0.01, 2.0),
    'epsilon': (1.0, 200.0),
    'h_ext': (1e-5, 1.0)
}

# Parameter Variation Configuration
# Specify which parameter should have heterogeneity and the standard deviation
# hetero_std is expressed as a fraction (e.g., 0.2 = 20% variation)
PARAM_HETERO = 'Xc'           # Parameter to make heterogeneous ('Xc', 'eta', 'beta', etc., or None)
HETERO_STD = 0.2              # Standard deviation as fraction of mean (0.2 = 20% variation)
HETERO_DIST = 'gaussian'      # Distribution type: 'gaussian' or 'lognormal'

# Fitting Parameters
SINGLE_PARAM_FITS = []
TWO_PARAM_FITS = [
    ['h_ext', 'eta' , 'beta', 'Xc'],
]

# Cost Function Configuration
FIT_TO = 'survival'       # 'survival' or 'hazard'
COST = 'sse'              # 'sse', 'ks', or 'greenwood'
USE_LX_WEIGHTS = False     # Use lx (population size) as weights

# Age range for fitting (optional)
AGE_START = 30             # Start age for survival curve
AGE_END = 90             # End age for survival curve
# Simulation Parameters
N_AGENTS = 10000                    # Number of agents (particles) per simulation
N_RESTARTS = 3                     # Number of warm restarts for single parameter fits
N_STAGE1 = 10000                    # Number of agents for Stage 1 (coordinate descent)
N_STAGE2 = 10000                    # Number of agents for Stage 2 (joint optimization)
MAX_CYCLES_STAGE1 = 1              # Maximum coordinate descent cycles in Stage 1
N_RESTARTS_PER_PARAM = 2           # Restarts per parameter in Stage 1 coordinate descent
N_RESTARTS_STAGE2 = 2              # Restarts for Stage 2 joint optimization

# Paths
RESULTS_PATH = PROJECT_ROOT / "saved_results" / f"historical_hmd_fitting_{COUNTRY}_{GENDER}_{DATA_TYPE}.pkl"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def convert_karin_params_to_scalars():
    """Convert karin_params arrays to scalar dictionary."""
    return {k: float(v[0]) if hasattr(v, '__len__') else float(v) 
            for k, v in karin_params.items()}

def get_lx_weights(hmd, year, age_start=0, age_end=105):
    """Extract normalized lx weights from HMD data."""
    year_data = hmd.data[hmd.data['Year'] == year]
    ages = year_data['Age'].values
    lx = year_data['lx'].values
    
    # Filter by age range
    mask = (ages >= age_start) & (ages <= age_end)
    lx_filtered = lx[mask]
    
    # Normalize (will be re-normalized in SRFitter but this ensures positivity)
    weights = lx_filtered / lx_filtered.sum()
    return weights

def print_section_header(title, char='=', width=80):
    """Print formatted section header."""
    print(f"\n{char * width}")
    print(f"{title.center(width)}")
    print(f"{char * width}\n")

def print_fit_start(param_list, year, method, fit_num, total_fits):
    """Print start of fitting run."""
    param_str = ' + '.join(param_list) if isinstance(param_list, list) else param_list
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'─' * 80}")
    print(f"[{timestamp}] Fit {fit_num}/{total_fits}: Year {year}, {param_str} ({method})")
    print(f"{'─' * 80}")

# ============================================================================
# MAIN FITTING FUNCTION
# ============================================================================

def fit_hmd_years(country, gender, data_type, years, fit_params, 
                  initial_params=None, use_lx_weights=True, fit_to='survival',
                  cost='sse', age_start=0, age_end=105, n_agents=5000, 
                  n_restarts=3, method='fit_with_restarts', verbose=True, 
                  use_sequential_init=True, param_bounds=None, **kwargs):
    """
    Fit SR parameters to HMD data for multiple years.
    
    Parameters
    ----------
    country : str
        Country code (e.g., 'SWE', 'USA')
    gender : str
        'male', 'female', or 'both'
    data_type : str
        'period' or 'cohort'
    years : list
        Years to fit
    fit_params : list or str
        Parameters to fit (e.g., ['eta', 'beta', 'h_ext'])
    initial_params : dict, optional
        Initial parameter values for first year
    use_lx_weights : bool
        Use lx weights for fitting
    fit_to : str
        'survival' or 'hazard'
    cost : str
        'sse', 'ks', or 'greenwood'
    age_start, age_end : int
        Age range for fitting
    n_agents : int
        Number of simulation agents
    n_restarts : int
        Number of restarts for fitting
    method : str
        'fit_with_restarts' or 'fit_hybrid_two_stage'
    use_sequential_init : bool
        If True, use previous year's fit as initial params for next year
    param_bounds : dict, optional
        Dictionary of parameter bounds {param_name: (min, max)}
    **kwargs : additional arguments for fitting functions
    
    Returns
    -------
    list : Results for each year
    """
    # Load HMD data
    hmd = HMD(country, gender, data_type)
    
    if initial_params is None:
        initial_params = INITIAL_PARAMS.copy()
    
    if param_bounds is None:
        param_bounds = PARAM_BOUNDS.copy()
    
    results = []
    current_initial_params = initial_params.copy()
    
    for year_idx, year in enumerate(years):
        if verbose:
            print(f"\n{'='*80}")
            print(f"FITTING YEAR {year} ({year_idx + 1}/{len(years)})")
            if use_sequential_init and year_idx > 0:
                print(f"Using previous year's fit as initial parameters")
            print(f"{'='*80}")
        
        try:
            # Get data based on fit_to mode
            if fit_to == 'survival':
                # Get survival curve and ages
                ages, survival = hmd.get_survival(year)
                mask = (ages >= age_start) & (ages <= age_end)
                ages_fit = ages[mask]
                survival_fit = survival[mask]
                
                # Normalize survival to start at 1 at age_start (survival from age_start onwards)
                if len(survival_fit) > 0 and survival_fit[0] > 0:
                    survival_fit = survival_fit / survival_fit[0]
                
                target_data = survival_fit
            
            elif fit_to == 'hazard':
                # Get hazard data (no normalization needed for hazard)
                ages, hazard = hmd.get_hazard(year, haz_type='mx')
                mask = (ages >= age_start) & (ages <= age_end)
                ages_fit = ages[mask]
                hazard_fit = hazard[mask]
                target_data = hazard_fit
            
            else:
                raise ValueError(f"fit_to must be 'survival' or 'hazard', got '{fit_to}'")
            
            # Get weights if requested
            weights = get_lx_weights(hmd, year, age_start, age_end) if use_lx_weights else None
            
            if verbose:
                print(f"Data: {len(ages_fit)} age points from {ages_fit[0]} to {ages_fit[-1]}")
                if weights is not None:
                    print(f"Using lx weights (range: {weights.min():.6f} to {weights.max():.6f})")
                print(f"Initial parameters for this year:")
                for param in fit_params:
                    print(f"  {param} = {current_initial_params.get(param, 'N/A')}")
            
            # Extract heterogeneity parameters from kwargs or use defaults
            param_hetero = kwargs.get('param_hetero', None)
            hetero_std = kwargs.get('hetero_std', 0.2)
            hetero_dist = kwargs.get('hetero_dist', 'gaussian')
            
            # Fit parameters
            if method == 'fit_with_restarts':
                best_params, cost_history = fit_with_restarts(
                    target_times=ages_fit,
                    target_array=target_data,
                    fit_params=fit_params,
                    initial_params=current_initial_params,
                    fit_to=fit_to,
                    cost=cost,
                    n=n_agents,
                    n_restarts=n_restarts,
                    sample_weights=weights,
                    param_hetero=param_hetero,
                    hetero_std=hetero_std,
                    hetero_dist=hetero_dist,
                    param_bounds=param_bounds,
                    verbose=verbose,
                    **{k: v for k, v in kwargs.items() if k not in ['param_hetero', 'hetero_std', 'hetero_dist']}
                )
                final_cost = cost_history[-1]
                extra_info = {'cost_history': cost_history}
                
            elif method == 'fit_hybrid_two_stage':
                # Extract stage-specific parameters from kwargs
                n_stage1 = kwargs.get('n_stage1', 5000)
                n_stage2 = kwargs.get('n_stage2', 5000)
                max_cycles_stage1 = kwargs.get('max_cycles_stage1', 1)
                n_restarts_per_param = kwargs.get('n_restarts_per_param', 3)
                n_restarts_stage2 = kwargs.get('n_restarts_stage2', 3)
                
                best_params, info = fit_hybrid_two_stage(
                    target_times=ages_fit,
                    target_array=target_data,
                    fit_params=fit_params,
                    initial_params=current_initial_params,
                    fit_to=fit_to,
                    cost=cost,
                    n_stage1=n_stage1,
                    n_stage2=n_stage2,
                    max_cycles_stage1=max_cycles_stage1,
                    n_restarts_per_param=n_restarts_per_param,
                    n_restarts_stage2=n_restarts_stage2,
                    sample_weights=weights,
                    param_hetero=param_hetero,
                    hetero_std=hetero_std,
                    hetero_dist=hetero_dist,
                    param_bounds=param_bounds,
                    verbose=verbose,
                    **{k: v for k, v in kwargs.items() if k not in ['param_hetero', 'hetero_std', 'hetero_dist', 'n_stage1', 'n_stage2', 'max_cycles_stage1', 'n_restarts_per_param', 'n_restarts_stage2']}
                )
                final_cost = info['stage2_cost']
                extra_info = info
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Extract scalar mean values from best_params (arrays can occur with param_hetero)
            best_params_scalar = {}
            for k, v in best_params.items():
                if isinstance(v, np.ndarray):
                    best_params_scalar[k] = float(np.mean(v))
                else:
                    best_params_scalar[k] = float(v)
            
            # Apply parameter bounds to best_params_scalar (safety check)
            for param in fit_params:
                if param in param_bounds:
                    lower, upper = param_bounds[param]
                    if best_params_scalar[param] < lower:
                        if verbose:
                            print(f"  ⚠ Warning: {param} = {best_params_scalar[param]:.6f} below lower bound {lower}, clipping")
                        best_params_scalar[param] = lower
                    elif best_params_scalar[param] > upper:
                        if verbose:
                            print(f"  ⚠ Warning: {param} = {best_params_scalar[param]:.6f} above upper bound {upper}, clipping")
                        best_params_scalar[param] = upper
            
            # Store results
            result = {
                'year': year,
                'fit_params': fit_params,
                'best_params': best_params,
                'final_cost': final_cost,
                'method': method,
                'fit_to': fit_to,
                'ages': ages_fit,
                'weights': weights,
                **extra_info
            }
            
            # Store the appropriate target data based on fit mode
            if fit_to == 'survival':
                result['survival'] = survival_fit
            elif fit_to == 'hazard':
                result['hazard'] = hazard_fit
            
            # Add individual parameter values
            # For fitted parameters, use best_params_scalar; for non-fitted, use INITIAL_PARAMS (not current_initial_params!)
            for p in ['eta', 'beta', 'kappa', 'epsilon', 'Xc', 'h_ext']:
                if p in fit_params:
                    # This parameter was fitted, use the result (scalar mean value)
                    result[f'{p}_value'] = best_params_scalar.get(p, np.nan)
                else:
                    # This parameter was NOT fitted, use the original constant value
                    result[f'{p}_value'] = INITIAL_PARAMS.get(p, np.nan)
            
            results.append(result)
            
            if verbose:
                print(f"\n✓ Year {year} complete: cost = {final_cost:.6f}")
                print(f"  Fitted parameters:")
                for param in fit_params:
                    val = best_params_scalar[param]
                    init_val = current_initial_params.get(param, np.nan)
                    if not np.isnan(init_val) and init_val != 0:
                        change = val - init_val
                        fold_change = val / init_val
                        print(f"    {param}: {init_val:.6f} → {val:.6f} (Δ{change:+.6f}, {fold_change:.3f}×)")
                    else:
                        print(f"    {param} = {val:.6f}")
            
            # Update current_initial_params for next year if sequential initialization is enabled
            if use_sequential_init:
                # Update only the fitted parameters for the next year (using scalar mean values)
                for param in fit_params:
                    current_initial_params[param] = best_params_scalar[param]
                
        except Exception as e:
            print(f"\n✗ ERROR fitting year {year}: {str(e)}")
            import traceback
            traceback.print_exc()
            # Don't update initial params if fit failed
            continue
    
    return results

# ============================================================================
# SIMULATION AND PLOTTING FUNCTIONS
# ============================================================================

def simulate_fitted_params(params_dict, tmax=120, n=5000, save_times=0.1, 
                          param_hetero=None, hetero_std=0.2, hetero_dist='gaussian'):
    """
    Simulate SR with fitted parameters.
    
    Parameters
    ----------
    params_dict : dict
        Dictionary of SR parameters
    tmax : float
        Maximum simulation time
    n : int
        Number of agents
    save_times : float
        Interval for saving states
    param_hetero : str, optional
        Parameter to make heterogeneous (e.g., 'Xc')
    hetero_std : float
        Standard deviation for heterogeneous parameter (as fraction, e.g., 0.2 = 20%)
    hetero_dist : str
        Distribution type: 'gaussian' or 'lognormal'
    
    Returns
    -------
    sim : SR_sim
        Simulation object
    """
    # Create a copy to avoid modifying the input
    params_copy = params_dict.copy()
    
    # Apply heterogeneity if requested
    if param_hetero is not None and param_hetero in params_copy:
        mean_val = params_copy[param_hetero]
        np.random.seed(123)  # Consistent seed for reproducibility
        
        if hetero_dist == 'gaussian':
            # Gaussian distribution around mean
            hetero_values = np.random.normal(mean_val, hetero_std * mean_val, size=n)
            # Ensure positive values
            hetero_values = np.maximum(hetero_values, mean_val * 0.01)
        elif hetero_dist == 'lognormal':
            # Lognormal distribution with mean preserved
            sigma = hetero_std
            mu = np.log(mean_val) - 0.5 * sigma**2
            hetero_values = np.random.lognormal(mu, sigma, size=n)
        else:
            raise ValueError(f"Unknown hetero_dist: {hetero_dist}")
        
        params_copy[param_hetero] = hetero_values
    
    sim = create_sr_simulation(
        species='human',
        n=n,
        save_times=save_times,
        params_dict=params_copy,
        h_ext=params_copy.get('h_ext', 0.0),
        tmax=tmax,
        dt=0.025,
        parallel=False,
        break_early=True
    )
    return sim

def plot_survival_comparison(results, hmd, years_to_plot=None, save_path=None, 
                           param_hetero=None, hetero_std=0.2, hetero_dist='gaussian'):
    """
    Plot survival curves: HMD data vs SR model fits.
    
    Parameters
    ----------
    results : list
        List of result dictionaries from fitting
    hmd : HMD
        HMD data object
    years_to_plot : list, optional
        Years to include in plot
    save_path : str, optional
        Path to save figure
    param_hetero : str, optional
        Parameter to make heterogeneous in simulations
    hetero_std : float
        Standard deviation for heterogeneous parameter
    hetero_dist : str
        Distribution type for heterogeneity
    """
    if years_to_plot is None:
        years_to_plot = [r['year'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for result in results:
        year = result['year']
        if year not in years_to_plot:
            continue
        
        # Plot HMD data
        ax.plot(result['ages'], result['survival'], 'o-', alpha=0.5, 
                label=f'HMD {year}', markersize=3)
        
        # Simulate and plot SR model
        sim = simulate_fitted_params(result['best_params'], 
                                     param_hetero=param_hetero,
                                     hetero_std=hetero_std,
                                     hetero_dist=hetero_dist)
        surv_df = sim.survival
        ax.plot(surv_df.index, surv_df.iloc[:, 0], '--', linewidth=2,
                label=f'SR {year}')
    
    ax.set_xlabel('Age [years]', fontsize=14, fontname='Arial')
    ax.set_ylabel('Survival Probability', fontsize=14, fontname='Arial')
    ax.set_title(f'Survival Curves: HMD vs SR Model', fontsize=16, fontname='Arial')
    ax.legend(fontsize=10, ncol=2)
    ax.grid(alpha=0.3)
    
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Arial')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax

def plot_hazard_comparison(results, hmd, years_to_plot=None, save_path=None,
                          param_hetero=None, hetero_std=0.2, hetero_dist='gaussian'):
    """
    Plot hazard curves: HMD data vs SR model fits.
    
    Parameters
    ----------
    results : list
        List of result dictionaries from fitting
    hmd : HMD
        HMD data object
    years_to_plot : list, optional
        Years to include in plot
    save_path : str, optional
        Path to save figure
    param_hetero : str, optional
        Parameter to make heterogeneous in simulations
    hetero_std : float
        Standard deviation for heterogeneous parameter
    hetero_dist : str
        Distribution type for heterogeneity
    """
    if years_to_plot is None:
        years_to_plot = [r['year'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for result in results:
        year = result['year']
        if year not in years_to_plot:
            continue
        
        # Plot HMD hazard
        ages_hmd, hazard_hmd = hmd.get_hazard(year, haz_type='mx')
        ax.plot(ages_hmd, hazard_hmd, 'o-', alpha=0.5, 
                label=f'HMD {year}', markersize=3)
        
        # Simulate and plot SR hazard
        sim = simulate_fitted_params(result['best_params'],
                                     param_hetero=param_hetero,
                                     hetero_std=hetero_std,
                                     hetero_dist=hetero_dist)
        ax.plot(sim.tspan_hazard, sim.hazard.flatten(), '--', linewidth=2,
                label=f'SR {year}')
    
    ax.set_yscale('log')
    ax.set_xlabel('Age [years]', fontsize=14, fontname='Arial')
    ax.set_ylabel('Hazard Rate [1/year]', fontsize=14, fontname='Arial')
    ax.set_title(f'Hazard Curves: HMD vs SR Model', fontsize=16, fontname='Arial')
    ax.legend(fontsize=10, ncol=2)
    ax.grid(alpha=0.3)
    
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Arial')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax

def plot_param_evolution(results, params_to_plot=None, save_path=None):
    """Plot how fitted parameters evolve over time."""
    if not results:
        print("No results to plot")
        return
    
    years = [r['year'] for r in results]
    
    if params_to_plot is None:
        # Auto-detect which params were fitted
        params_to_plot = results[0]['fit_params']
    
    n_params = len(params_to_plot)
    fig, axes = plt.subplots(n_params, 1, figsize=(10, 3*n_params), sharex=True)
    if n_params == 1:
        axes = [axes]
    
    for ax, param in zip(axes, params_to_plot):
        values = [r[f'{param}_value'] for r in results]
        ax.plot(years, values, 'o-', linewidth=2, markersize=8)
        ax.set_ylabel(param, fontsize=14, fontname='Arial')
        ax.grid(alpha=0.3)
        
        for label in ax.get_yticklabels():
            label.set_fontname('Arial')
    
    axes[-1].set_xlabel('Year', fontsize=14, fontname='Arial')
    for label in axes[-1].get_xticklabels():
        label.set_fontname('Arial')
    
    fig.suptitle('Parameter Evolution Over Time', fontsize=16, fontname='Arial')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, axes

def compare_h_ext_to_makeham(results, hmd, save_path=None):
    """Compare fitted h_ext to HMD Makeham term."""
    years = [r['year'] for r in results]
    h_ext_vals = [r.get('h_ext_value', np.nan) for r in results]
    makeham_vals = [hmd.get_makeham_term(y) for y in years]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(years, h_ext_vals, 'o-', label='Fitted h_ext', linewidth=2, markersize=8)
    ax.plot(years, makeham_vals, 's-', label='HMD Makeham term', linewidth=2, markersize=8)
    ax.set_yscale('log')
    ax.set_xlabel('Year', fontsize=14, fontname='Arial')
    ax.set_ylabel('Extrinsic Mortality [1/year]', fontsize=14, fontname='Arial')
    ax.set_title('Fitted h_ext vs HMD Makeham Term', fontsize=16, fontname='Arial')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Arial')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax

def plot_single_year_fit(results, year, hmd=None, plot_type='survival', 
                        param_hetero=None, hetero_std=0.2, hetero_dist='gaussian',
                        save_path=None, n_sim=10000):
    """
    Plot HMD data vs SR model fit for a single year.
    
    Parameters
    ----------
    results : list or dict
        Either a list of result dictionaries or a single result dict
    year : int
        Year to plot
    hmd : HMD, optional
        HMD data object. If None, will use stored data from results
    plot_type : str
        'survival' or 'hazard'
    param_hetero : str, optional
        Parameter to make heterogeneous in simulations
    hetero_std : float
        Standard deviation for heterogeneous parameter
    hetero_dist : str
        Distribution type for heterogeneity
    save_path : str, optional
        Path to save figure
    n_sim : int
        Number of agents for simulation (default 10000 for better statistics)
    
    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    # Find the result for the requested year
    if isinstance(results, dict):
        result = results
    else:
        result = None
        for r in results:
            if r['year'] == year:
                result = r
                break
        if result is None:
            raise ValueError(f"Year {year} not found in results")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    if plot_type == 'survival':
        # Get age range from HMD data
        age_start = result['ages'][0]
        
        # Plot HMD data (already normalized during fitting)
        ax.plot(result['ages'], result['survival'], 'o-', 
                color='black', alpha=0.7, label=f'HMD {year}', 
                markersize=5, linewidth=2)
        
        # Simulate and plot SR model
        sim = simulate_fitted_params(result['best_params'], n=n_sim,
                                     param_hetero=param_hetero,
                                     hetero_std=hetero_std,
                                     hetero_dist=hetero_dist)
        surv_df = sim.survival
        model_ages = surv_df.index.values
        model_survival = surv_df.iloc[:, 0].values
        
        # Normalize SR model survival to S(age_start) = 1
        # Find survival at age_start
        idx_start = np.searchsorted(model_ages, age_start, side='right') - 1
        idx_start = np.clip(idx_start, 0, len(model_ages) - 1)
        S_at_start = model_survival[idx_start]
        if S_at_start > 0:
            model_survival_normalized = model_survival / S_at_start
        else:
            model_survival_normalized = model_survival
        
        ax.plot(model_ages, model_survival_normalized, '--', 
                color='red', linewidth=3, label=f'SR Model Fit', alpha=0.8)
        
        ax.set_xlabel('Age [years]', fontsize=16, fontname='Arial')
        ax.set_ylabel('Survival Probability', fontsize=16, fontname='Arial')
        ax.set_title(f'Survival Curve: {year}', fontsize=18, fontname='Arial', fontweight='bold')
        
    elif plot_type == 'hazard':
        # Get HMD hazard data
        if hmd is None:
            raise ValueError("hmd object required for hazard plotting")
        ages_hmd, hazard_hmd = hmd.get_hazard(year, haz_type='mx')
        ax.plot(ages_hmd, hazard_hmd, 'o-', 
                color='black', alpha=0.7, label=f'HMD {year}', 
                markersize=5, linewidth=2)
        
        # Simulate and plot SR hazard
        sim = simulate_fitted_params(result['best_params'], n=n_sim,
                                     param_hetero=param_hetero,
                                     hetero_std=hetero_std,
                                     hetero_dist=hetero_dist)
        ax.plot(sim.tspan_hazard, sim.hazard.flatten(), '--', 
                color='red', linewidth=3, label=f'SR Model Fit', alpha=0.8)
        
        ax.set_yscale('log')
        ax.set_xlabel('Age [years]', fontsize=16, fontname='Arial')
        ax.set_ylabel('Hazard Rate [1/year]', fontsize=16, fontname='Arial')
        ax.set_title(f'Hazard Curve: {year}', fontsize=18, fontname='Arial', fontweight='bold')
    
    else:
        raise ValueError(f"plot_type must be 'survival' or 'hazard', got '{plot_type}'")
    
    # Add fitted parameters as text box
    param_text = "Fitted Parameters:\n"
    for param in result['fit_params']:
        val = result['best_params'][param]
        # Handle both scalar and array values
        if isinstance(val, np.ndarray):
            param_text += f"{param} = {float(np.mean(val)):.4f} (mean)\n"
        else:
            param_text += f"{param} = {float(val):.4f}\n"
    param_text += f"\nCost = {result['final_cost']:.4f}"
    
    ax.text(0.02, 0.98, param_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontname='Arial')
    
    ax.legend(fontsize=13, loc='upper right')
    ax.grid(alpha=0.3)
    
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(13)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    print_section_header("HISTORICAL HMD FITTING", char='#')
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  Country: {COUNTRY}, Gender: {GENDER}, Data: {DATA_TYPE}")
    print(f"  Years: {YEARS}")
    print(f"  Age range: {AGE_START}-{AGE_END}")
    print(f"  N_AGENTS = {N_AGENTS}, N_RESTARTS = {N_RESTARTS}")
    print(f"  FIT_TO = '{FIT_TO}', COST = '{COST}'")
    print(f"  USE_LX_WEIGHTS = {USE_LX_WEIGHTS}")
    print(f"\nParameter Heterogeneity:")
    if PARAM_HETERO:
        print(f"  {PARAM_HETERO} with {HETERO_STD*100:.0f}% variation ({HETERO_DIST} distribution)")
    else:
        print(f"  None (homogeneous population)")
    print(f"\nInitial Parameters:")
    for param, value in INITIAL_PARAMS.items():
        print(f"  {param} = {value}")
    
    # Load HMD for later use
    hmd = HMD(COUNTRY, GENDER, DATA_TYPE)
    print(f"\n✓ Loaded HMD data for {COUNTRY}")
    
    initial_params = INITIAL_PARAMS.copy()

    # ========================================================================
    # PERFORM FITTING
    # ========================================================================
    all_results = []
    total_fits = len(SINGLE_PARAM_FITS) + len(TWO_PARAM_FITS)
    fit_counter = 0
    
    print_section_header("SINGLE PARAMETER FITS")
    
    for param in SINGLE_PARAM_FITS:
        fit_counter += 1
        print(f"\n{'='*80}")
        print(f"FIT {fit_counter}/{total_fits}: {param}")
        print(f"{'='*80}")
        
        try:
            results = fit_hmd_years(
                country=COUNTRY,
                gender=GENDER,
                data_type=DATA_TYPE,
                years=YEARS,
                fit_params=[param],
                initial_params=initial_params,
                use_lx_weights=USE_LX_WEIGHTS,
                fit_to=FIT_TO,
                cost=COST,
                age_start=AGE_START,
                age_end=AGE_END,
                n_agents=N_AGENTS,
                n_restarts=N_RESTARTS,
                method='fit_with_restarts',
                verbose=True,
                use_sequential_init=True,
                param_bounds=PARAM_BOUNDS,
                param_hetero=PARAM_HETERO,
                hetero_std=HETERO_STD,
                hetero_dist=HETERO_DIST
            )
            
            # Add param name for easy identification
            for r in results:
                r['param_name'] = param
            all_results.extend(results)
            
        except Exception as e:
            print(f"\n✗ ERROR in {param} fitting: {str(e)}")
            continue
    
    print_section_header("TWO PARAMETER FITS")
    
    for param_pair in TWO_PARAM_FITS:
        fit_counter += 1
        print(f"\n{'='*80}")
        print(f"FIT {fit_counter}/{total_fits}: {' + '.join(param_pair)}")
        print(f"{'='*80}")
        
        try:
            results = fit_hmd_years(
                country=COUNTRY,
                gender=GENDER,
                data_type=DATA_TYPE,
                years=YEARS,
                fit_params=param_pair,
                initial_params=initial_params,
                use_lx_weights=USE_LX_WEIGHTS,
                fit_to=FIT_TO,
                cost=COST,
                age_start=AGE_START,
                age_end=AGE_END,
                n_agents=N_AGENTS,
                n_restarts=N_RESTARTS,
                method='fit_hybrid_two_stage',
                n_stage1=N_STAGE1,
                n_stage2=N_STAGE2,
                max_cycles_stage1=MAX_CYCLES_STAGE1,
                n_restarts_per_param=N_RESTARTS_PER_PARAM,
                n_restarts_stage2=N_RESTARTS_STAGE2,
                verbose=True,
                use_sequential_init=True,
                param_bounds=PARAM_BOUNDS,
                param_hetero=PARAM_HETERO,
                hetero_std=HETERO_STD,
                hetero_dist=HETERO_DIST
            )
            
            # Add param name for easy identification
            for r in results:
                r['param_name'] = ' + '.join(param_pair)
            all_results.extend(results)
            
        except Exception as e:
            print(f"\n✗ ERROR in {' + '.join(param_pair)} fitting: {str(e)}")
            continue
    
    # ========================================================================
    # PROCESS AND DISPLAY RESULTS
    # ========================================================================
    print_section_header("RESULTS SUMMARY")
    
    if not all_results:
        print("ERROR: No successful fits!")
        return
    
    # Create summary DataFrame
    df_data = []
    for r in all_results:
        row = {
            'param_name': r['param_name'],
            'year': r['year'],
            'final_cost': r['final_cost'],
        }
        # Add all parameter values
        for param in ['eta', 'beta', 'Xc', 'epsilon', 'h_ext', 'kappa']:
            row[param] = r[f'{param}_value']
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Print summary by parameter type
    for param_name in df['param_name'].unique():
        subset = df[df['param_name'] == param_name].sort_values('year')
        print(f"\n{param_name}:")
        print(subset.to_string(index=False))
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    print_section_header("SAVING RESULTS")
    
    save_data = {
        'results': all_results,
        'dataframe': df,
        'configuration': {
            'COUNTRY': COUNTRY,
            'GENDER': GENDER,
            'DATA_TYPE': DATA_TYPE,
            'YEARS': YEARS,
            'FIT_TO': FIT_TO,
            'COST': COST,
            'USE_LX_WEIGHTS': USE_LX_WEIGHTS,
            'AGE_START': AGE_START,
            'AGE_END': AGE_END,
            'N_AGENTS': N_AGENTS,
            'N_RESTARTS': N_RESTARTS,
            'PARAM_HETERO': PARAM_HETERO,
            'HETERO_STD': HETERO_STD,
            'HETERO_DIST': HETERO_DIST,
            'initial_params': initial_params
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"✓ Results saved to: {RESULTS_PATH}")
    
    print_section_header("FITTING COMPLETE", char='#')
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total successful fits: {len(all_results)}")
    print()


if __name__ == '__main__':
    main()
