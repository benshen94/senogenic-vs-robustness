#!/usr/bin/env python3
"""
Print comprehensive sample statistics for NHANES exposure groups analysis.

This script loads the precomputed exposure_groups_results.pkl file and prints:
- Baseline sample size and deaths
- Sample sizes and deaths for each exposure group
- Death rates and percentages

Author: Generated for NHANES mortality analysis
Date: 2025
"""

import pickle
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.thresholds.paths import SAVED_RESULTS_DIR

# Path to the precomputed results
PICKLE_PATH = SAVED_RESULTS_DIR / "exposure_groups_results.pkl"

def print_statistics():
    """Load and print all sample statistics from exposure groups analysis."""
    
    # Load the precomputed results
    if not PICKLE_PATH.exists():
        print(f"ERROR: File not found: {PICKLE_PATH}")
        print("Please run the analysis to generate exposure_groups_results.pkl first.")
        return
    
    with open(PICKLE_PATH, 'rb') as f:
        interventions_dict = pickle.load(f)
    
    print("="*80)
    print("NHANES EXPOSURE GROUPS SAMPLE STATISTICS")
    print("="*80)
    
    # =========================================================================
    # BASELINE STATISTICS
    # =========================================================================
    print("\n" + "="*80)
    print("BASELINE (All Participants)")
    print("="*80)
    
    if 'baseline' in interventions_dict:
        for baseline_type, stats in interventions_dict['baseline'].items():
            if stats:
                n_total = stats.get('n', 'N/A')
                n_deaths = stats.get('n_d', 'N/A')
                median_age = stats.get('0.50', 'N/A')
                min_age = stats.get('min_age', 'N/A')
                steepness_iqr_abs = stats.get('steepness_iqr_absolute', 'N/A')
                
                print(f"\n{baseline_type.upper().replace('_', ' ')}:")
                print(f"  Total participants: {int(n_total):,}" if isinstance(n_total, (int, float)) else f"  Total participants: {n_total}")
                print(f"  Deaths observed: {int(n_deaths):,}" if isinstance(n_deaths, (int, float)) else f"  Deaths observed: {n_deaths}")
                print(f"  Median lifespan (KM S=0.5): {int(round(median_age))} years" if isinstance(median_age, float) else f"  Median lifespan: {median_age}")
                print(f"  Steepness (IQR): {steepness_iqr_abs:.1f}" if isinstance(steepness_iqr_abs, float) else f"  Steepness (IQR): {steepness_iqr_abs}")
                print(f"  Minimum entry age: {min_age:.1f} years" if isinstance(min_age, float) else f"  Minimum entry age: {min_age}")
    
    # =========================================================================
    # EXPOSURE-SPECIFIC STATISTICS
    # =========================================================================
    print("\n" + "="*80)
    print("EXPOSURE-SPECIFIC GROUPS")
    print("="*80)
    
    topics_to_print = [k for k in interventions_dict.keys() if k != 'baseline' and k != 'work_regularity']
    
    # Create a summary table for all topics
    summary_data = []
    
    for topic in sorted(topics_to_print):
        groups = interventions_dict[topic]
        
        print(f"\n{'-'*80}")
        print(f"{topic.upper().replace('_', ' ')}")
        print(f"{'-'*80}")
        
        # Calculate total for topic
        total_n = sum(g.get('n', 0) for g in groups.values() if g.get('n'))
        total_deaths = sum(g.get('n_d', 0) for g in groups.values() if g.get('n_d'))
        total_death_rate = (total_deaths / total_n * 100) if total_n > 0 else 0
        
        print(f"Total across all groups: {total_n:,} participants, {total_deaths:,} deaths ({total_death_rate:.2f}%)")
        print()
        
        for group_name in sorted(groups.keys()):
            stats = groups[group_name]
            n = stats.get('n', 'N/A')
            n_d = stats.get('n_d', 'N/A')
            pct_of_topic = (n / total_n * 100) if (isinstance(n, int) and total_n > 0) else 'N/A'
            median_age = stats.get('0.50', 'N/A')
            min_age = stats.get('min_age', 'N/A')
            steepness_iqr_abs = stats.get('steepness_iqr_absolute', 'N/A')
            
            print(f"  {group_name}:")
            print(f"    N = {int(n):,}" if isinstance(n, (int, float)) else f"    N = {n}")
            print(f"    Deaths = {int(n_d):,}" if isinstance(n_d, (int, float)) else f"    Deaths = {n_d}")
            print(f"    % of topic total = {pct_of_topic:.1f}%" if isinstance(pct_of_topic, float) else f"    % of topic total = {pct_of_topic}")
            print(f"    Median lifespan (KM S=0.5) = {int(round(median_age))} years" if isinstance(median_age, float) else f"    Median lifespan = {median_age}")
            print(f"    Steepness (IQR) = {steepness_iqr_abs:.1f}" if isinstance(steepness_iqr_abs, float) else f"    Steepness (IQR) = {steepness_iqr_abs}")
            print(f"    Minimum entry age = {min_age:.1f} years" if isinstance(min_age, float) else f"    Minimum entry age = {min_age}")
            print()
            
            # Add to summary table
            summary_data.append({
                'Topic': topic.replace('_', ' ').title(),
                'Group': group_name,
                'N': int(n) if isinstance(n, (int, float)) else 0,
                'Deaths': int(n_d) if isinstance(n_d, (int, float)) else 0,
                'Median Lifespan': int(round(median_age)) if isinstance(median_age, float) else 'N/A',
                'Steepness (IQR)': f"{steepness_iqr_abs:.1f}" if isinstance(steepness_iqr_abs, float) else 'N/A'
            })
    
    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY TABLE (for easy copy-paste to manuscript)")
    print("="*80)
    
    df_summary = pd.DataFrame(summary_data)
    print("\n" + df_summary.to_string(index=False))
    
    # =========================================================================
    # NOTES
    # =========================================================================
    print("\n" + "="*80)
    print("NOTES")
    print("="*80)
    print("• Median Lifespan: Age where Kaplan-Meier survival curve S(t) = 0.5")
    print("  (This is NOT the average age of death, but the 50th percentile from KM curve)")
    print("• Steepness (IQR): Calculated as -t₅₀/(t₇₅-t₂₅) from KM percentiles")
    print("  Using absolute age (starting from age 0)")
    print("  Higher values = steeper mortality curve (more compressed)")
    print("• Gender/sex breakdown is NOT available in current analysis")
    print("• Some topics may have overlapping participants")
    print("• Baseline includes all linkage-eligible participants")
    print("• Work regularity excluded from this analysis")
    print("="*80)
    
    # =========================================================================
    # AUTOMATIC CSV EXPORT
    # =========================================================================
    output_path = SAVED_RESULTS_DIR / "csv" / "nhanes_sample_stats_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(output_path, index=False)
    print(f"\n✓ Summary table automatically exported to:")
    print(f"  {output_path}")


if __name__ == "__main__":
    print_statistics()
