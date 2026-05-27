import pickle
import argparse
import sys
from pathlib import Path
import numpy as np 

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGING_PYTHON_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(AGING_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(AGING_PYTHON_ROOT))

from ageing_packages.hetero_analysis.nhanes_analysis import calculate_survival_stats
from senogenic_vs_robustness.paths import NHANES_DATA_DIR, RESULTS_DIR

nhanes_data_path = str(NHANES_DATA_DIR) + "/"

def safe_nanstd(arr):
    arr = [x for x in arr if x is not None]
    if len(arr) == 0:
        return np.nan
    return np.nanstd(arr)

def fmt_float(val):
    return f"{val:.2f}" if val is not None and np.isfinite(val) else "nan"

def main():
    parser = argparse.ArgumentParser(description='Calculate and pickle NHANES exposure group survival stats.')
    parser.add_argument('--print-bootstrap', action='store_true', help='Print every bootstrap iteration (verbose)')
    args = parser.parse_args()

    print('Calculating survival stats for all exposure groups...')
    results = calculate_survival_stats(nhanes_data_path, print_bootstrap=True)

    output_path = RESULTS_DIR / "exposure_groups_results.pkl"
    with output_path.open("wb") as f:
        pickle.dump(results, f)

    print(f"Exposure group results saved to {output_path}")

if __name__ == '__main__':
    main() 
    
    
    
