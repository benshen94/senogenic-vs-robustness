"""
Progeria Fitting Results Visualization

Plot fitted SR model survival curves against progeria data to visualize fit quality.

Usage:
    from progeria_plotting import plot_fit
    
    # Plot a specific fit
    plot_fit('beta + epsilon')
    
    # Plot multiple fits
    plot_fit(['beta', 'Xc', 'beta + epsilon'])
    
    # Show all fits ranked by quality
    plot_all_fits(top_n=5)
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from typing import Union, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGING_PYTHON_ROOT = PROJECT_ROOT.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(AGING_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(AGING_PYTHON_ROOT))

from ageing_packages.utils.sr_utils import create_sr_simulation, karin_params
from src.shared.thresholds.paths import SAVED_RESULTS_DIR

# Convert karin params to scalars for fold-change calculation
KARIN_SCALARS = {
    'eta': float(karin_params['eta'][0]),
    'beta': float(karin_params['beta'][0]),
    'kappa': float(karin_params['kappa'][0]),
    'epsilon': float(karin_params['epsilon'][0]),
    'Xc': float(karin_params['Xc'][0])
}

# Unicode symbols for parameters
PARAM_SYMBOLS = {
    'eta': 'η',
    'beta': 'β',
    'kappa': 'κ',
    'epsilon': 'ε',
    'Xc': 'Xc'
}

# Paths (should match progeria_fitting.py)
RESULTS_PATH = SAVED_RESULTS_DIR / "progeria_fitting_results.pkl"

# Plotting settings
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


def load_results(results_path: str = RESULTS_PATH):
    """
    Load progeria fitting results from pickle file.
    
    Parameters
    ----------
    results_path : str
        Path to results pickle file
        
    Returns
    -------
    dict : Dictionary containing results, dataframe, configuration, and progeria data
    """
    with open(results_path, 'rb') as f:
        data = pickle.load(f)
    return data


def find_fit_result(param_name: str, results_data: dict):
    """
    Find fitting result for a specific parameter combination.
    
    Parameters
    ----------
    param_name : str
        Name of parameter combination (e.g., 'beta', 'beta + epsilon', 'Xc + beta')
        
    results_data : dict
        Loaded results data
        
    Returns
    -------
    dict : Result dictionary for the specified fit, or None if not found
    """
    for result in results_data['results']:
        if result['param_name'] == param_name:
            return result
    
    # Try alternative formats (with/without spaces)
    alt_name = param_name.replace(' ', '')
    for result in results_data['results']:
        if result['param_name'].replace(' ', '') == alt_name:
            return result
    
    return None


def simulate_with_params(params: dict, n: int = 10000, tmax: float = 30.0, 
                        save_times: float = 0.1):
    """
    Run SR simulation with given parameters.
    
    Parameters
    ----------
    params : dict
        Parameter dictionary with keys like 'eta', 'beta', etc.
    n : int
        Number of agents
    tmax : float
        Maximum simulation time
    save_times : float
        Save interval
        
    Returns
    -------
    SR_sim : Simulation object with survival curves
    """
    sim = create_sr_simulation(
        species='human',
        n=n,
        save_times=save_times,
        params_dict=params,
        tmax=tmax,
        dt=0.025,
        parallel=True,
        break_early=True
    )
    return sim


def sim_fit(param_name: str,
            n_sim: int = 10000,
            results_path: str = RESULTS_PATH,
            tmax: Optional[float] = None) -> 'SR_sim':
    """
    Simulate SR model with fitted parameters for a given fit.
    
    Parameters
    ----------
    param_name : str
        Parameter combination to simulate (e.g., 'beta + epsilon')
    n_sim : int
        Number of agents for simulation
    results_path : str
        Path to results pickle file
    tmax : float, optional
        Maximum simulation time. If None, uses progeria data max time + 5
        
    Returns
    -------
    SR_sim : Simulation object with survival curves
    
    Examples
    --------
    >>> sim = sim_fit('beta + epsilon', n_sim=10000)
    >>> sim.survival  # Access survival data
    """
    # Load results
    results_data = load_results(results_path)
    
    # Find the fit result
    result = find_fit_result(param_name, results_data)
    
    if result is None:
        available = [r['param_name'] for r in results_data['results']]
        raise ValueError(f"No result found for '{param_name}'. Available fits: {available}")
    
    # Get fitted parameters
    best_params = result['best_params']
    
    # Determine tmax
    if tmax is None:
        progeria_times = results_data['progeria_data']['times']
        tmax = progeria_times[-1] + 5
    
    # Run simulation
    print(f"Running simulation for {param_name} with n={n_sim}...")
    sim = simulate_with_params(best_params, n=n_sim, tmax=tmax)
    
    return sim


def plot_fit(param_name: Union[str, List[str]], 
             n_sim: int = 10000,
             results_path: str = RESULTS_PATH,
             figsize: tuple = (8, 6),
             show_cost: bool = False,
             save_path: Optional[str] = None,
             ax: Optional[plt.Axes] = None,
             colors: Optional[Union[str, List[str]]] = None,
             label_fold_changes: bool = True,
             fold_change_decimals: int = 2,
             **kwargs):
    """
    Plot fitted survival curve against progeria data.
    
    Parameters
    ----------
    param_name : str or list of str
        Parameter combination(s) to plot (e.g., 'beta + epsilon')
        Can be a single name or list of names to plot multiple fits
    n_sim : int
        Number of agents for simulation
    results_path : str
        Path to results pickle file
    figsize : tuple
        Figure size (width, height) - only used if ax is None
    show_cost : bool
        Whether to show cost in legend (default: False)
    save_path : str, optional
        Path to save figure (if None, displays instead)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure and axes.
    colors : str or list of str, optional
        Color(s) for the fitted curves. Can be a single color (applied to all)
        or a list of colors (one per param_name). If None, uses default color cycle.
    label_fold_changes : bool
        If True, labels include fold changes relative to baseline (e.g., 'ε ×2.5')
        If False, labels show only parameter symbols (default: False)
    fold_change_decimals : int
        Number of decimal places for fold-change values in legend (default: 2).
        E.g., 1 for tenths (×2.5), 2 for hundredths (×2.50).
    **kwargs : dict
        Additional keyword arguments passed to ax.plot() for fitted curves
        (e.g., linewidth, alpha, linestyle)
        
    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    
    Examples
    --------
    >>> plot_fit('beta + epsilon')
    >>> plot_fit(['beta', 'beta + epsilon', 'eta + beta'], n_sim=15000)
    >>> fig, ax = plt.subplots(); plot_fit('beta', ax=ax)
    >>> plot_fit(['beta', 'epsilon'], colors=['red', 'blue'], linewidth=3)
    >>> plot_fit('epsilon', label_fold_changes=True)
    >>> plot_fit('beta', fold_change_decimals=1)  # tenths in legend (×2.5)
    """
    # Load results
    results_data = load_results(results_path)
    progeria_times = results_data['progeria_data']['times']
    progeria_survival = results_data['progeria_data']['survival']
    
    # Convert single name to list
    if isinstance(param_name, str):
        param_names = [param_name]
    else:
        param_names = param_name
    
    # Handle colors
    if colors is None:
        # Default color cycle
        color_list = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
    elif isinstance(colors, str):
        # Single color for all curves
        color_list = [colors] * len(param_names)
    else:
        # List of colors provided
        color_list = colors
        # Extend if not enough colors provided
        if len(color_list) < len(param_names):
            default_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
            while len(color_list) < len(param_names):
                color_list.append(default_colors[len(color_list) % len(default_colors)])
    
    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.get_figure()
        created_fig = False
    
    # Plot progeria data
    ax.plot(progeria_times, progeria_survival, '-', 
            color='black', linewidth=4, 
            label='HGPS patients', zorder=10)
    
    # Helper function to convert param names to symbols
    def param_name_to_symbols(name):
        """Convert parameter names like 'beta + epsilon' to 'β + ε'"""
        # Sort by length (longest first) to avoid partial replacements
        # e.g., replace 'beta' before 'eta' to avoid 'beta' -> 'bηa'
        sorted_params = sorted(PARAM_SYMBOLS.items(), key=lambda x: len(x[0]), reverse=True)
        for param, symbol in sorted_params:
            name = name.replace(param, symbol)
        return name
    # Helper function to create label with fold changes
    def create_fold_change_label(pname, best_params, fit_params, decimals: int):
        """Create label with fold changes for fitted parameters"""
        # Parse the parameter name to get individual parameters
        # e.g., 'beta + epsilon' -> ['beta', 'epsilon']
        param_parts = [p.strip() for p in pname.split('+')]
        
        label_parts = []
        for param in param_parts:
            if param in fit_params:
                fold_change = best_params[param] / KARIN_SCALARS[param]
                symbol = PARAM_SYMBOLS.get(param, param)
                label_parts.append(f"{symbol} ×{fold_change:.{decimals}f}")
            else:
                # This shouldn't happen if pname matches fit_params, but handle it
                symbol = PARAM_SYMBOLS.get(param, param)
                label_parts.append(symbol)
        
        return ' + '.join(label_parts) + ' fold change'
    
    # Set default kwargs for fitted curves
    plot_kwargs = {'linewidth': 2, 'alpha': 0.8}
    plot_kwargs.update(kwargs)  # Override with user-provided kwargs
    
    # Plot each fit
    for idx, pname in enumerate(param_names):
        result = find_fit_result(pname, results_data)
        
        if result is None:
            print(f"Warning: No result found for '{pname}'")
            print(f"Available fits: {[r['param_name'] for r in results_data['results']]}")
            continue
        
        # Get fitted parameters
        best_params = result['best_params']
        final_cost = result['final_cost']
        
        # Run simulation using sim_fit
        sim = sim_fit(pname, n_sim=n_sim, results_path=results_path, 
                     tmax=progeria_times[-1] + 5)
        
        # Extract survival
        sim_times = sim.survival.index.values
        sim_survival = sim.survival.iloc[:, 0].values
        
        # Create label
        color = color_list[idx]
        if label_fold_changes:
            label = create_fold_change_label(pname, best_params, result['fit_params'], fold_change_decimals)
        else:
            label = param_name_to_symbols(pname)
        
        if show_cost:
            label += f" (cost={final_cost:.3f})"
        
        ax.plot(sim_times, sim_survival, '-', 
                color=color, label=label, **plot_kwargs)
        
        # Print fitted parameter values
        print(f"\n{pname}:")
        print(f"  Cost: {final_cost:.6f}")
        print(f"  Fitted parameters:")
        for param in result['fit_params']:
            fold_change = best_params[param] / KARIN_SCALARS[param]
            print(f"    {param} = {best_params[param]:.6f} ({fold_change:.2f}× Karin)")
        print(f"  All parameters (including non-fitted):")
        for param in ['eta', 'beta', 'kappa', 'epsilon', 'Xc']:
            fold_change = best_params[param] / KARIN_SCALARS[param]
            fitted_marker = "*" if param in result['fit_params'] else " "
            print(f"   {fitted_marker} {param} = {best_params[param]:.6f} ({fold_change:.2f}× Karin)")
    
    # Formatting
    ax.set_xlabel('Age (years)', fontsize=11)
    ax.set_ylabel('Survival Probability', fontsize=11)
    ax.set_title('Progeria Data vs. Fitted SR Model', fontsize=12, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    
    # Only handle figure-level operations if we created the figure
    if created_fig:
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nFigure saved to: {save_path}")
        else:
            plt.show()
    
    return fig, ax


def plot_all_fits(top_n: Optional[int] = None,
                  n_sim: int = 10000,
                  results_path: str = RESULTS_PATH,
                  figsize: tuple = (12, 8),
                  save_path: Optional[str] = None,
                  ax: Optional[plt.Axes] = None):
    """
    Plot all fits ranked by quality.
    
    Parameters
    ----------
    top_n : int, optional
        Plot only top N best fits (by cost). If None, plots all.
    n_sim : int
        Number of agents for simulation
    results_path : str
        Path to results pickle file
    figsize : tuple
        Figure size (width, height) - only used if ax is None
    save_path : str, optional
        Path to save figure
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure and axes.
        
    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    """
    # Load results
    results_data = load_results(results_path)
    df = results_data['dataframe']
    
    # Select top N
    if top_n is not None:
        df_plot = df.head(top_n)
    else:
        df_plot = df
    
    param_names = df_plot['params_fitted'].tolist()
    
    print(f"\nPlotting {len(param_names)} fits (ranked by cost):")
    for i, (_, row) in enumerate(df_plot.iterrows(), 1):
        print(f"{i}. {row['params_fitted']}: cost={row['final_cost']:.6f}")
    
    # Plot
    fig, ax = plot_fit(param_names, n_sim=n_sim, results_path=results_path,
                      figsize=figsize, show_cost=False, save_path=save_path, ax=ax)
    
    return fig, ax


def print_summary(results_path: str = RESULTS_PATH):
    """
    Print summary of all fitting results.
    
    Parameters
    ----------
    results_path : str
        Path to results pickle file
    """
    results_data = load_results(results_path)
    df = results_data['dataframe']
    config = results_data['configuration']
    
    print("=" * 80)
    print("PROGERIA FITTING RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"\nFitting Configuration:")
    print(f"  N_AGENTS: {config['N_AGENTS']}")
    print(f"  N_RESTARTS (single param): {config['N_RESTARTS']}")
    print(f"  N_STAGE1/N_STAGE2 (two param): {config['N_STAGE1']}/{config['N_STAGE2']}")
    print(f"  Cost function: {config['COST']}")
    
    print(f"\nTotal fits: {len(df)}")
    print(f"Timestamp: {results_data['timestamp']}")
    
    print("\n" + "-" * 120)
    print(f"{'Rank':<6} {'Parameters':<20} {'Cost':<12} {'eta (fold)':<15} {'beta (fold)':<15} {'epsilon (fold)':<18} {'Xc (fold)':<15}")
    print("-" * 120)
    
    for _, row in df.iterrows():
        rank = row['rank']
        params = row['params_fitted']
        cost = row['final_cost']
        eta_fold = row['eta'] / KARIN_SCALARS['eta']
        beta_fold = row['beta'] / KARIN_SCALARS['beta']
        eps_fold = row['epsilon'] / KARIN_SCALARS['epsilon']
        xc_fold = row['Xc'] / KARIN_SCALARS['Xc']
        print(f"{rank:<6} {params:<20} {cost:<12.6f} {eta_fold:>6.2f}×{row['eta']:>6.2f}  {beta_fold:>6.2f}×{row['beta']:>6.2f}  {eps_fold:>6.2f}×{row['epsilon']:>9.2f}  {xc_fold:>6.2f}×{row['Xc']:>6.2f}")
    
    print("-" * 120)
    print(f"\nBest fit: {df.iloc[0]['params_fitted']}")
    print(f"Cost: {df.iloc[0]['final_cost']:.6f}")
    print(f"\nParameter values for best fit (with fold-change vs Karin):")
    for param in ['eta', 'beta', 'kappa', 'epsilon', 'Xc']:
        val = df.iloc[0][param]
        fold = val / KARIN_SCALARS[param]
        print(f"  {param:8s} = {val:8.4f}  ({fold:6.2f}× Karin)")
    print("=" * 80)


def compare_fits(param_names: List[str],
                n_sim: int = 10000,
                results_path: str = RESULTS_PATH,
                save_path: Optional[str] = None):
    """
    Create comparison plot with subplots for multiple fits.
    
    Parameters
    ----------
    param_names : list of str
        List of parameter combinations to compare
    n_sim : int
        Number of agents for simulation
    results_path : str
        Path to results pickle file
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    fig : matplotlib figure object
    """
    n_fits = len(param_names)
    ncols = min(3, n_fits)
    nrows = int(np.ceil(n_fits / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    if n_fits == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Load results
    results_data = load_results(results_path)
    progeria_times = results_data['progeria_data']['times']
    progeria_survival = results_data['progeria_data']['survival']
    
    # Helper function to convert param names to symbols
    def param_name_to_symbols(name):
        """Convert parameter names like 'beta + epsilon' to 'β + ε'"""
        for param, symbol in PARAM_SYMBOLS.items():
            name = name.replace(param, symbol)
        return name
    
    for idx, (ax, pname) in enumerate(zip(axes, param_names)):
        result = find_fit_result(pname, results_data)
        
        if result is None:
            ax.text(0.5, 0.5, f'No data for\n{pname}', 
                   ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # Plot data
        ax.plot(progeria_times, progeria_survival, 'o', 
               color='black', markersize=3, alpha=0.6, label='HGPS patients')
        
        # Simulate and plot
        best_params = result['best_params']
        sim = simulate_with_params(best_params, n=n_sim, tmax=progeria_times[-1] + 5)
        
        sim_times = sim.survival.index.values
        sim_survival = sim.survival.iloc[:, 0].values
        
        # Use Unicode symbols in label
        fit_label = param_name_to_symbols(pname)
        ax.plot(sim_times, sim_survival, '-', color='#e41a1c', 
               linewidth=2, label=fit_label)
        
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Survival Probability')
        ax.set_title(param_name_to_symbols(pname), fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.05)
    
    # Hide unused subplots
    for idx in range(n_fits, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    else:
        plt.show()
    
    return fig


if __name__ == '__main__':
    # Example usage
    print("Progeria Fitting Results Plotter")
    print("=" * 50)
    # Print summary
    print_summary()
    
    # Plot best fit
    print("\n" + "=" * 50)
    print("Plotting best fit...")
    plot_fit('beta + epsilon', n_sim=10000)
