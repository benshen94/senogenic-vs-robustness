"""
Progeria Parameter Fitting Script

Fits SR model parameters to progeria survival data using various parameter combinations
and fitting strategies. Saves results ordered by fit quality.

"""

import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGING_PYTHON_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(AGING_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(AGING_PYTHON_ROOT))

from ageing_packages.utils.sr_fitter import fit_with_restarts, fit_hybrid_two_stage
from ageing_packages.utils.sr_utils import karin_params
from senogenic_vs_robustness.paths import RESULTS_DIR

# Convert karin params to scalars for fold-change calculation
KARIN_SCALARS = {
    'eta': float(karin_params['eta'][0]),
    'beta': float(karin_params['beta'][0]),
    'kappa': float(karin_params['kappa'][0]),
    'epsilon': float(karin_params['epsilon'][0]),
    'Xc': float(karin_params['Xc'][0])
}

# ============================================================================
# CONFIGURATION - Edit these arrays to add/remove fitting cases
# ============================================================================

# -----------------------------------------------------------------------------
# PARAMETERS TO FIT
# -----------------------------------------------------------------------------
# Available SR model parameters: 'eta', 'beta', 'kappa', 'epsilon', 'Xc'
# Parameters not listed here will be held constant at their initial values

# Single parameter fits (using fit_with_restarts)
# Each parameter is optimized independently using warm restarts
SINGLE_PARAM_FITS = ['eta', 'beta', 'Xc', 'epsilon']

# Two parameter fits (using fit_hybrid_two_stage)
# IMPORTANT: Order matters! The first parameter is optimized first in each
# coordinate descent cycle. Different orderings can yield different results.
# Example: ['beta', 'Xc'] ≠ ['Xc', 'beta']
TWO_PARAM_FITS = [
    ['beta', 'Xc'],      # Optimize beta first, then Xc
    ['beta', 'eta'],      # Optimize Xc first, then beta
    ['beta', 'epsilon'],
    ['eta', 'Xc'],
    ['eta', 'epsilon'],
    ['Xc' , 'epsilon']
]

# -----------------------------------------------------------------------------
# SIMULATION PARAMETERS
# -----------------------------------------------------------------------------
# Number of agents (particles) per simulation
# Higher values = more accurate but slower
# Recommended: 3000-10000
N_AGENTS = 5000

# -----------------------------------------------------------------------------
# SINGLE PARAMETER FITTING PARAMETERS (fit_with_restarts)
# -----------------------------------------------------------------------------
# Pipeline for single parameter fits:
#   1. Start from initial parameters (Karin defaults)
#   2. Fit the parameter using L-BFGS-B optimization
#   3. Restart from the result and fit again (N_RESTARTS times)
#   4. Each restart uses a different random seed for noise realization
#
# Number of warm restarts for single parameter optimization
# More restarts = more robust but slower
# Recommended: 3-5
N_RESTARTS = 3

# -----------------------------------------------------------------------------
# TWO PARAMETER FITTING PARAMETERS (fit_hybrid_two_stage)
# -----------------------------------------------------------------------------
# Pipeline for two parameter fits (e.g., ['beta', 'Xc']):
#   STAGE 1: Coordinate Descent with Restarts
#     For each cycle (MAX_CYCLES_STAGE1 times):
#       - Fit first param (beta) using fit_with_restarts (N_RESTARTS_PER_PARAM restarts)
#       - Update beta, hold Xc constant
#       - Fit second param (Xc) using fit_with_restarts (N_RESTARTS_PER_PARAM restarts)
#       - Update Xc, hold beta constant
#     Continue cycling until convergence or max cycles reached
#
#   STAGE 2: Joint Optimization with Restarts
#     Starting from Stage 1 result:
#       - Optimize both parameters simultaneously using fit_with_restarts
#       - Run N_RESTARTS_STAGE2 restarts for final refinement
#
# Number of agents for Stage 1 (coordinate descent)
N_STAGE1 = 5000

# Number of agents for Stage 2 (joint optimization)
N_STAGE2 = 5000

# Maximum number of coordinate descent cycles in Stage 1
# Each cycle goes through all parameters in order
# Recommended: 3-5
MAX_CYCLES_STAGE1 = 1

# Number of restarts per parameter during coordinate descent (Stage 1)
# Each parameter optimization uses this many restarts
# Recommended: 2-5
N_RESTARTS_PER_PARAM = 3

# Number of restarts for joint optimization in Stage 2
# Final refinement with all parameters free
# Recommended: 2-5
N_RESTARTS_STAGE2 = 3

# -----------------------------------------------------------------------------
# COST FUNCTION CONFIGURATION
# -----------------------------------------------------------------------------
# Target to fit: 'survival' (Kaplan-Meier curve) or 'hazard'
FIT_TO = 'survival'

# Cost function: 'sse' (sum of squared errors), 'ks' (Kolmogorov-Smirnov),
#                'greenwood' (variance-weighted on log-log scale)
COST = 'sse'

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------
DATA_PATH = RESULTS_DIR / "progeria_data.pkl"
RESULTS_PATH = RESULTS_DIR / "progeria_fitting_results.pkl"

# -----------------------------------------------------------------------------
# ADVANCED: HETEROGENEOUS INITIAL PARAMETERS
# -----------------------------------------------------------------------------
# You can introduce population heterogeneity by passing array-valued initial
# parameters for parameters you DON'T fit. For example:
#
#   import numpy as np
#   kappa_dist = np.random.lognormal(np.log(0.5), 0.3, size=N_AGENTS)
#   
#   # In fit function calls, add:
#   initial_params = {'kappa': kappa_dist}
#   
#   # Then fit other params (e.g., beta) with heterogeneous kappa population
#   # This allows testing how well a single fitted parameter can match data
#   # arising from a heterogeneous population
# -----------------------------------------------------------------------------

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def convert_karin_params_to_scalars():
    """Convert karin_params arrays to scalar dictionary for initial_params."""
    return {k: float(v[0]) if hasattr(v, '__len__') else float(v) 
            for k, v in karin_params.items()}

def print_section_header(title, char='=', width=80):
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"{title.center(width)}")
    print(f"{char * width}\n")

def print_fit_start(param_list, method, fit_num, total_fits):
    """Print start of a fitting run."""
    param_str = ' + '.join(param_list) if isinstance(param_list, list) else param_list
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'─' * 80}")
    print(f"[{timestamp}] Fit {fit_num}/{total_fits}: {param_str} (method: {method})")
    print(f"{'─' * 80}")

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    print_section_header("PROGERIA PARAMETER FITTING", char='#')
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  N_AGENTS = {N_AGENTS}")
    print(f"  N_RESTARTS = {N_RESTARTS}")
    print(f"  N_STAGE1 = {N_STAGE1}, N_STAGE2 = {N_STAGE2}")
    print(f"  MAX_CYCLES_STAGE1 = {MAX_CYCLES_STAGE1}")
    print(f"  N_RESTARTS_PER_PARAM = {N_RESTARTS_PER_PARAM}")
    print(f"  N_RESTARTS_STAGE2 = {N_RESTARTS_STAGE2}")
    print(f"  FIT_TO = '{FIT_TO}', COST = '{COST}'")
    
    # ========================================================================
    # 1. LOAD PROGERIA DATA
    # ========================================================================
    print_section_header("LOADING PROGERIA DATA")
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    
    with DATA_PATH.open("rb") as f:
        progeria_data = pickle.load(f)
    
    progeria_survival = progeria_data['survival_curve']['s_pred']
    progeria_times = progeria_data['survival_curve']['t_pred']
    
    print(f"✓ Loaded progeria data from: {DATA_PATH}")
    print(f"  Time points: {len(progeria_times)} (range: {progeria_times[0]:.1f} to {progeria_times[-1]:.1f})")
    print(f"  Survival values: {len(progeria_survival)} (range: {progeria_survival.min():.4f} to {progeria_survival.max():.4f})")
    
    # Get initial parameters
    initial_params = convert_karin_params_to_scalars()
    print(f"\nStarting from Karin parameters:")
    for param, value in initial_params.items():
        print(f"  {param} = {value:.6f}")
    
    # ========================================================================
    # 2. PERFORM FITTING
    # ========================================================================
    results = []
    total_fits = len(SINGLE_PARAM_FITS) + len(TWO_PARAM_FITS)
    fit_counter = 0
    
    print_section_header("SINGLE PARAMETER FITS (fit_with_restarts)")
    
    # Single parameter fits
    for param in SINGLE_PARAM_FITS:
        fit_counter += 1
        print_fit_start([param], 'fit_with_restarts', fit_counter, total_fits)
        
        try:
            best_params, cost_history = fit_with_restarts(
                target_times=progeria_times,
                target_array=progeria_survival,
                fit_params=[param],
                initial_params=initial_params,
                fit_to=FIT_TO,
                cost=COST,
                n=N_AGENTS,
                n_restarts=N_RESTARTS,
                verbose=True
            )
            
            final_cost = cost_history[-1]
            
            # Store results
            result_dict = {
                'fit_params': [param],
                'param_name': param,
                'best_params': best_params,
                'final_cost': final_cost,
                'method': 'fit_with_restarts',
                'cost_history': cost_history
            }
            
            # Add individual parameter values for easy access
            for p in ['eta', 'beta', 'kappa', 'epsilon', 'Xc']:
                result_dict[f'{p}_value'] = best_params.get(p, initial_params[p])
            
            results.append(result_dict)
            
            print(f"\n✓ Completed {param}: final cost = {final_cost:.6f}")
            fold_change = best_params[param] / KARIN_SCALARS[param]
            print(f"  Fitted value: {param} = {best_params[param]:.6f} ({fold_change:.2f}× Karin)")
            
        except Exception as e:
            print(f"\n✗ ERROR fitting {param}: {str(e)}")
            continue
    
    print_section_header("TWO PARAMETER FITS (fit_hybrid_two_stage)")
    
    # Two parameter fits
    for param_pair in TWO_PARAM_FITS:
        fit_counter += 1
        print_fit_start(param_pair, 'fit_hybrid_two_stage', fit_counter, total_fits)
        print(f"  → Stage 1 (coordinate descent): {param_pair[0]} then {param_pair[1]}")
        print(f"  → Stage 2 (joint optimization): {param_pair[0]} + {param_pair[1]}")
        
        try:
            best_params, info = fit_hybrid_two_stage(
                target_times=progeria_times,
                target_array=progeria_survival,
                fit_params=param_pair,
                initial_params=initial_params,
                fit_to=FIT_TO,
                cost=COST,
                n_stage1=N_STAGE1,
                n_stage2=N_STAGE2,
                max_cycles_stage1=MAX_CYCLES_STAGE1,
                n_restarts_per_param=N_RESTARTS_PER_PARAM,
                n_restarts_stage2=N_RESTARTS_STAGE2,
                verbose=True
            )
            
            final_cost = info['stage2_cost']
            stage1_cost = info['stage1_cost']
            
            # Store results
            result_dict = {
                'fit_params': param_pair,
                'param_name': ' + '.join(param_pair),
                'best_params': best_params,
                'final_cost': final_cost,
                'method': 'fit_hybrid_two_stage',
                'stage1_cost': stage1_cost,
                'stage2_cost': final_cost,
                'stage1_params': info['stage1_params'],
                'cost_history_stage1': info['cost_history_stage1']
            }
            
            # Add individual parameter values for easy access
            for p in ['eta', 'beta', 'kappa', 'epsilon', 'Xc']:
                result_dict[f'{p}_value'] = best_params.get(p, initial_params[p])
            
            results.append(result_dict)
            
            improvement = stage1_cost - final_cost
            print(f"\n✓ Completed {' + '.join(param_pair)}: final cost = {final_cost:.6f}")
            print(f"  Stage 1 cost: {stage1_cost:.6f}")
            print(f"  Stage 2 improvement: {improvement:.6f}")
            for param in param_pair:
                fold_change = best_params[param] / KARIN_SCALARS[param]
                print(f"  Fitted value: {param} = {best_params[param]:.6f} ({fold_change:.2f}× Karin)")
            
        except Exception as e:
            print(f"\n✗ ERROR fitting {' + '.join(param_pair)}: {str(e)}")
            continue
    
    # ========================================================================
    # 3. PROCESS AND DISPLAY RESULTS
    # ========================================================================
    print_section_header("RESULTS SUMMARY")
    
    if not results:
        print("ERROR: No successful fits!")
        return
    
    # Create DataFrame for easy viewing
    df_data = []
    for r in results:
        row = {
            'params_fitted': r['param_name'],
            'final_cost': r['final_cost'],
            'method': r['method'],
            'eta': r['eta_value'],
            'beta': r['beta_value'],
            'kappa': r['kappa_value'],
            'epsilon': r['epsilon_value'],
            'Xc': r['Xc_value']
        }
        if 'stage1_cost' in r:
            row['stage1_cost'] = r['stage1_cost']
            row['improvement'] = r['stage1_cost'] - r['final_cost']
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df = df.sort_values('final_cost', ascending=True).reset_index(drop=True)
    df.insert(0, 'rank', range(1, len(df) + 1))
    
    # Print formatted table
    print("\n" + "=" * 120)
    print("FITTING RESULTS (ordered by final cost)")
    print("=" * 120)
    
    # Print header
    print(f"\n{'Rank':<6} {'Params Fitted':<20} {'Final Cost':<15} {'eta':<12} {'beta':<12} {'epsilon':<12} {'Xc':<12}")
    print("-" * 120)
    
    # Print each row
    for _, row in df.iterrows():
        print(f"{row['rank']:<6} {row['params_fitted']:<20} {row['final_cost']:<15.6f} "
              f"{row['eta']:<12.6f} {row['beta']:<12.6f} {row['epsilon']:<12.6f} {row['Xc']:<12.2f}")
    
    print("-" * 120)
    print(f"\nBest fit: {df.iloc[0]['params_fitted']} with cost = {df.iloc[0]['final_cost']:.6f}")
    
    # ========================================================================
    # 4. SAVE RESULTS
    # ========================================================================
    print_section_header("SAVING RESULTS")
    
    save_data = {
        'results': results,
        'dataframe': df,
        'configuration': {
            'N_AGENTS': N_AGENTS,
            'N_RESTARTS': N_RESTARTS,
            'N_STAGE1': N_STAGE1,
            'N_STAGE2': N_STAGE2,
            'MAX_CYCLES_STAGE1': MAX_CYCLES_STAGE1,
            'N_RESTARTS_PER_PARAM': N_RESTARTS_PER_PARAM,
            'N_RESTARTS_STAGE2': N_RESTARTS_STAGE2,
            'FIT_TO': FIT_TO,
            'COST': COST,
            'initial_params': initial_params
        },
        'progeria_data': {
            'times': progeria_times,
            'survival': progeria_survival
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Create directory if it doesn't exist
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with RESULTS_PATH.open("wb") as f:
        pickle.dump(save_data, f)
    
    print(f"✓ Results saved to: {RESULTS_PATH}")
    print(f"  Includes: {len(results)} fitting results, sorted DataFrame, configuration")
    
    print_section_header("FITTING COMPLETE", char='#')
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total fits: {len(results)}/{total_fits} successful")
    print()


if __name__ == '__main__':
    main()
