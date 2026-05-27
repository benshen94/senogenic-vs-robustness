import matplotlib.pyplot as plt # type: ignore
import matplotlib.colors as mcolors # type: ignore
import numpy as np # type: ignore
import pickle # type: ignore
import warnings
from ageing_packages.utils import sr_utils as utils
from ageing_packages.mortality_models.gamma_gompertz import GammaGompertz as gg
from src.shared.thresholds.paths import SAVED_RESULTS_DIR

# Suppress MallocStackLogging warnings
warnings.filterwarnings("ignore", message=".*MallocStackLogging.*")
saved_results_path = str(SAVED_RESULTS_DIR) + "/"

def load_results(pkl_filename):
    """Load results from pickle file.
    
    Args:
        pkl_filename (str): Full filename of the pickle file (e.g., 'param_variation_results.pkl')
        
    Returns:
        dict: Loaded results dictionary
        
    Raises:
        FileNotFoundError: If the results file is not found
    """
    pkl_file = saved_results_path + pkl_filename
    try:
        with open(pkl_file, 'rb') as f:
            results = pickle.load(f)
        return results
    except FileNotFoundError:
        print(f"Error: Results file not found at {pkl_file}")
        raise

def plot_steepness_longevity(
    pkl_file=None,
    param_type='variation',
    from_t=0,
    longevity_metric='t_median_absolute',
    steepness_metric='steepness_iqr_absolute',
    ignore_kappa=True,
    ax=None,
    title=None,
    value_type='normalized',
    alpha=1.0,
    marker_size_range=(20, 100),
    linewidth=2,
    line_alpha=None,
    h_ext=False,
    value_range=None,
    steepness_method=None,
    steepness_type=None,
    legend_fontsize=20,  # font size for legend **entries and title**
    zorder=1,
    legend_loc='upper left',    # New: allow legend location, default is upper left
    legend_title='Parameter Change',  # New: shared legend title
):
    """
    Plot steepness vs longevity for parameter variations or heterogeneity.
    Args:
        pkl_file, param_type, from_t, longevity_metric, steepness_metric, ignore_kappa, ax, title,
        value_type, alpha, marker_size_range, linewidth, line_alpha, h_ext, value_range,
        steepness_method/steepness_type, legend_fontsize, zorder, legend_loc, h_ext_legend_loc,
        legend_title, use_same_legends
    """
    # --- Handle DEPRECATED/LEGACY ARGS ---
    if steepness_method is not None or steepness_type is not None:
        print("Warning: steepness_method and steepness_type parameters are deprecated. Use steepness_metric instead.")
        if steepness_method is not None and steepness_type is not None:
            steepness_metric = f'steepness_{steepness_method}_{steepness_type}'

    if longevity_metric in ['median', 'maximum']:
        print("Warning: longevity_metric values 'median'/'maximum' are deprecated. Use full metric names like 't_median_relative'.")
        longevity_metric = 't_median_relative' if longevity_metric == 'median' else 't_max_relative'

    # --- Choose data file if not explicitly passed ---
    if pkl_file is None:
        if param_type == 'variation':
            pkl_file = 'param_variation_results.pkl'
        elif param_type == 'hetero':
            pkl_file = 'param_distribution_results.pkl'
        else:
            raise ValueError("Invalid param_type. Must be 'variation' or 'hetero' if pkl_file is not specified.")

    # --- Load Data ---
    try:
        results = load_results(pkl_file)
    except FileNotFoundError:
        return

    # --- Set up metric keys and try to get baseline values for normalization ---
    steepness_key = steepness_metric
    longevity_key = longevity_metric

    if value_type == 'normalized':
        try:
            baseline_steepness = results['baseline'][from_t][steepness_key]
            baseline_longevity = results['baseline'][from_t][longevity_key]
            if baseline_longevity is None or baseline_steepness is None:
                print(f"Warning: No baseline data for from_t={from_t}")
                return
        except KeyError:
            print(f"Error: Baseline data not found for from_t={from_t}")
            return

    # --- (Re)Initialize axis if needed ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        new_figure = True
    else:
        new_figure = False

    # --- Choose parameters to plot ---
    default_param_order = ['eta', 'beta', 'kappa', 'epsilon', 'Xc']
    params_to_consider = [p for p in default_param_order if not (ignore_kappa and p == 'kappa')]

    # --- Collect all "variation" values present across parameters ---
    all_values = set()
    for p in params_to_consider:
        if p in results:
            all_values.update(results[p].keys())
    variation_values = sorted(all_values)

    # --- Slice out restricted range if user asks for it ---
    if value_range is not None:
        min_val, max_val = value_range
        variation_values = [v for v in variation_values if min_val <= v <= max_val]

    # --- Set alpha for lines (default to marker alpha) ---
    actual_line_alpha = line_alpha if line_alpha is not None else alpha

    # --- Collect lines for the legend here ---
    legend_elements = []

    # --------- MAIN PLOTTING LOOP ------------
    for param in params_to_consider:
        if param not in results:
            continue

        # ---- Collect valid (var_value, longevity, steepness) triplets -------
        data_points = []
        for var_val in variation_values:
            if var_val in results[param] and from_t in results[param][var_val]:
                try:
                    steep = results[param][var_val][from_t][steepness_key]
                    long_val = results[param][var_val][from_t][longevity_key]
                    if steep is not None and long_val is not None:
                        if value_type == 'normalized':
                            steep  /= baseline_steepness
                            long_val /= baseline_longevity
                        data_points.append((var_val, long_val, steep))
                except KeyError:
                    print(f"Error: KeyError for param={param}, var_val={var_val}, from_t={from_t}")

        if not data_points:
            continue

        # ---- Sort by variation value and get color ----
        data_points.sort()
        color = utils.param_colors[param]

        # ---- Map marker size to parameter value ----
        min_size, max_size = marker_size_range
        if variation_values:
            min_var, max_var = min(variation_values), max(variation_values)
            size_norm = plt.Normalize(min_var, max_var)
        else:
            size_norm = lambda x: 0

        # ---- Plot points/fancy sizes/lines between pairs ----
        for i, (var_val, x, y) in enumerate(data_points):
            marker_size = min_size + (max_size - min_size) * size_norm(var_val) if variation_values else min_size
            ax.scatter(
                x, y,
                color=mcolors.to_rgba(color, alpha),
                s=marker_size,
                edgecolors=color,
                linewidth=1,
                zorder=zorder
            )
            if i > 0:
                prev_x = data_points[i-1][1]
                prev_y = data_points[i-1][2]
                ax.plot(
                    [prev_x, x], [prev_y, y],
                    color=color, linewidth=linewidth,
                    alpha=actual_line_alpha, zorder=zorder
                )

        # ---- Add this parameter as a legend entry (line only) ----
        param_label = f"{utils.param_descriptions[param]} {utils.param_names[param]}"
        legend_elements.append(
            plt.Line2D([0, 1], [0, 1], color=color, linewidth=linewidth, label=param_label)
        )

    # --------- PLOT extrinsic mortality (h_ext) if requested -------------
    # The h_ext "legend" is no longer handled separately; if desired, can simply draw its data points & curve.
    if h_ext and param_type == 'variation' and 'h_ext' in results:
        # ---- Gather h_ext variation data points ----
        h_ext_data = [
            (h, results['h_ext'][h][from_t][steepness_key], results['h_ext'][h][from_t][longevity_key])
            for h in results['h_ext']
            if from_t in results['h_ext'][h] and results['h_ext'][h][from_t][steepness_key] is not None
        ]
        # ---- Plot each (sorted by h value) ----
        if h_ext_data:
            h_ext_data.sort()
            h_values = [h for h, _, _ in h_ext_data]
            min_size, max_size = marker_size_range
            if h_values:
                h_norm = plt.Normalize(min(h_values), max(h_values))
            else:
                h_norm = lambda x: 0

            for i, (h_val, steep, long_val) in enumerate(h_ext_data):
                if value_type == 'normalized':
                    x = long_val / baseline_longevity
                    y = steep / baseline_steepness
                else:
                    x = long_val
                    y = steep
                marker_size = min_size + (max_size - min_size) * h_norm(h_val) if h_values else min_size
                # Make sure both facecolor and edgecolor are red (not darkred or black)
                ax.scatter(
                    x, y,
                    color=mcolors.to_rgba('red', alpha),
                    s=marker_size,
                    edgecolors=mcolors.to_rgba('red', alpha),
                    linewidth=1,
                    zorder=zorder
                )
                if i > 0:
                    prev_x, prev_y = h_ext_data[i-1][2], h_ext_data[i-1][1]
                    if value_type == 'normalized':
                        prev_x, prev_y = prev_x / baseline_longevity, prev_y / baseline_steepness
                    ax.plot(
                        [prev_x, x], [prev_y, y],
                        color='red', linewidth=linewidth, alpha=actual_line_alpha, zorder=zorder
                    )
            # Optionally, one could add a legend entry for h_ext directly to legend_elements:
            legend_elements.append(
                plt.Line2D([0, 1], [0, 1], color='red', linewidth=linewidth, label='Extrinsic mortality')
            )

    # --------- LABELING, AXIS AND LEGEND ---------
    # Label axes with pretty names
    longevity_label = 'Median' if 'median' in longevity_metric else 'Maximum'
    steepness_label = 'IQR' if 'iqr' in steepness_metric else 'CV'
    norm_prefix = 'Normalized ' if value_type == 'normalized' else ''
    relative_label = ' (Relative)' if 'relative' in longevity_metric else ' (Absolute)'

    ax.set_xlabel(f"{norm_prefix}{longevity_label} Lifespan{relative_label}", fontsize=14)
    ax.set_ylabel(f"{norm_prefix}{steepness_label} Steepness{relative_label}", fontsize=14)

    # Build or compose title
    param_desc = 'Heterogeneity' if param_type == 'hetero' else 'Changes'
    if title:
        ax.set_title(title, fontsize=16)
    else:
        ax.set_title(
            f"Parameter {param_desc} on {norm_prefix}{steepness_label} Steepness vs Longevity"
            f"\n(from t={from_t}, {steepness_metric}, {longevity_metric})",
            fontsize=16
        )

    # --- Grid, reference lines and formatting ---
    ax.grid(True, alpha=0.3)
    if value_type == 'normalized':
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)

    # --- Set up legend font properties for legend ---
    from matplotlib.font_manager import FontProperties
    entry_fontprops = FontProperties(family='Arial', size=legend_fontsize)
    title_fontprops = FontProperties(family='Arial', weight='bold', size=legend_fontsize + 6)

    legend = ax.legend(
        handles=legend_elements,
        title=legend_title,
        frameon=True,
        loc=legend_loc,
        prop=entry_fontprops,             # legend entry font size!
        title_fontproperties=title_fontprops,   # legend title font!
    )

    # For some Matplotlib versions, manual forcing for upper-left legend
    if legend:
        for text in legend.get_texts():
            text.set_fontsize(legend_fontsize)
            text.set_fontfamily('Arial')
        if legend.get_title():
            legend.get_title().set_fontweight('bold')
            legend.get_title().set_fontfamily('Arial')
            legend.get_title().set_fontsize(legend_fontsize + 6)

    # --- Show if standalone plot requested ---
    if new_figure:
        plt.tight_layout()
        plt.show()

    return legend


def plot_median_max_plane(
    pkl_file=None,
    param_type='variation',
    from_t=0,
    median_metric='t_median_absolute',
    max_metric='t_max',
    ignore_kappa=True,
    ax=None,
    title=None,
    value_type='normalized',
    alpha=1.0,
    marker_size_range=(20, 100),
    linewidth=2,
    line_alpha=None,
    h_ext=False,
    value_range=None,
    legend_fontsize=20,
    zorder=1,
    legend_loc='upper left',
    legend_title='Parameter Change',
):
    """
    Plot median lifespan (x-axis) vs maximum lifespan (y-axis) for parameter variations or heterogeneity.
    Each parameter curve has its own color, matching the steepness-longevity style.

    Args:
        pkl_file: Pickle filename in saved_results folder (e.g. 'param_variation_results.pkl').
        param_type: 'variation' or 'hetero'.
        from_t: Starting age for analysis.
        median_metric: Key for median lifespan ('t_median_absolute' or 't_median_relative').
        max_metric: Key for maximum lifespan ('t_max').
        ignore_kappa: Exclude kappa parameter.
        ax: Matplotlib axis to plot on (optional).
        title: Custom plot title (optional).
        value_type: 'normalized' (divide by baseline) or 'absolute'.
        alpha: Marker alpha.
        marker_size_range: (min_size, max_size) for marker scaling by parameter value.
        linewidth: Line width between points.
        line_alpha: Line alpha (defaults to alpha if None).
        h_ext: Include extrinsic mortality (h_ext) curve in red.
        value_range: (min_val, max_val) to restrict variation values.
        legend_fontsize: Font size for legend entries.
        zorder: Plot z-order.
        legend_loc: Legend location.
        legend_title: Legend title.

    Returns:
        legend: Matplotlib legend object (or None).
    """
    # --- Choose data file if not explicitly passed ---
    if pkl_file is None:
        if param_type == 'variation':
            pkl_file = 'param_variation_results.pkl'
        elif param_type == 'hetero':
            pkl_file = 'param_distribution_results.pkl'
        else:
            raise ValueError("Invalid param_type. Must be 'variation' or 'hetero' if pkl_file is not specified.")

    # --- Load Data ---
    try:
        results = load_results(pkl_file)
    except FileNotFoundError:
        return None

    # --- Set up metric keys and baseline values ---
    if value_type == 'normalized':
        try:
            baseline_median = results['baseline'][from_t][median_metric]
            baseline_max = results['baseline'][from_t][max_metric]
            if baseline_median is None or baseline_max is None:
                print(f"Warning: No baseline data for from_t={from_t}")
                return None
        except KeyError:
            print(f"Error: Baseline data not found for from_t={from_t}")
            return None

    # --- (Re)Initialize axis if needed ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        new_figure = True
    else:
        new_figure = False

    # --- Choose parameters to plot ---
    default_param_order = ['eta', 'beta', 'kappa', 'epsilon', 'Xc']
    params_to_consider = [p for p in default_param_order if not (ignore_kappa and p == 'kappa')]

    # --- Collect all variation values ---
    all_values = set()
    for p in params_to_consider:
        if p in results:
            all_values.update(results[p].keys())
    variation_values = sorted(all_values)

    if value_range is not None:
        min_val, max_val = value_range
        variation_values = [v for v in variation_values if min_val <= v <= max_val]

    actual_line_alpha = line_alpha if line_alpha is not None else alpha
    legend_elements = []

    # --------- MAIN PLOTTING LOOP ------------
    for param in params_to_consider:
        if param not in results:
            continue

        data_points = []
        for var_val in variation_values:
            if var_val in results[param] and from_t in results[param][var_val]:
                try:
                    med_val = results[param][var_val][from_t][median_metric]
                    max_val = results[param][var_val][from_t][max_metric]
                    if med_val is not None and max_val is not None:
                        if value_type == 'normalized':
                            med_val /= baseline_median
                            max_val /= baseline_max
                        data_points.append((var_val, med_val, max_val))
                except KeyError:
                    pass

        if not data_points:
            continue

        data_points.sort()
        color = utils.param_colors[param]

        min_size, max_size = marker_size_range
        if variation_values:
            min_var, max_var = min(variation_values), max(variation_values)
            size_norm = plt.Normalize(min_var, max_var)
        else:
            size_norm = lambda x: 0

        for i, (var_val, x, y) in enumerate(data_points):
            marker_size = min_size + (max_size - min_size) * size_norm(var_val) if variation_values else min_size
            ax.scatter(
                x, y,
                color=mcolors.to_rgba(color, alpha),
                s=marker_size,
                edgecolors=color,
                linewidth=1,
                zorder=zorder
            )
            if i > 0:
                prev_x = data_points[i - 1][1]
                prev_y = data_points[i - 1][2]
                ax.plot(
                    [prev_x, x], [prev_y, y],
                    color=color, linewidth=linewidth,
                    alpha=actual_line_alpha, zorder=zorder
                )

        param_label = f"{utils.param_descriptions[param]} {utils.param_names[param]}"
        legend_elements.append(
            plt.Line2D([0, 1], [0, 1], color=color, linewidth=linewidth, label=param_label)
        )

    # --------- PLOT h_ext if requested ------------
    if h_ext and param_type == 'variation' and 'h_ext' in results:
        h_ext_data = [
            (h, results['h_ext'][h][from_t][median_metric], results['h_ext'][h][from_t][max_metric])
            for h in results['h_ext']
            if from_t in results['h_ext'][h]
            and results['h_ext'][h][from_t][median_metric] is not None
            and results['h_ext'][h][from_t][max_metric] is not None
        ]
        if h_ext_data:
            h_ext_data.sort()
            h_values = [h for h, _, _ in h_ext_data]
            min_size, max_size = marker_size_range
            h_norm = plt.Normalize(min(h_values), max(h_values)) if h_values else (lambda x: 0)

            for i, (h_val, med_val, max_val) in enumerate(h_ext_data):
                if value_type == 'normalized':
                    x = med_val / baseline_median
                    y = max_val / baseline_max
                else:
                    x, y = med_val, max_val
                marker_size = min_size + (max_size - min_size) * h_norm(h_val) if h_values else min_size
                ax.scatter(
                    x, y,
                    color=mcolors.to_rgba('red', alpha),
                    s=marker_size,
                    edgecolors=mcolors.to_rgba('red', alpha),
                    linewidth=1,
                    zorder=zorder
                )
                if i > 0:
                    prev_x, prev_y = h_ext_data[i - 1][1], h_ext_data[i - 1][2]
                    if value_type == 'normalized':
                        prev_x, prev_y = prev_x / baseline_median, prev_y / baseline_max
                    ax.plot(
                        [prev_x, x], [prev_y, y],
                        color='red', linewidth=linewidth, alpha=actual_line_alpha, zorder=zorder
                    )
            legend_elements.append(
                plt.Line2D([0, 1], [0, 1], color='red', linewidth=linewidth, label='Extrinsic mortality')
            )

    # --------- LABELING ---------
    norm_prefix = 'Normalized ' if value_type == 'normalized' else ''
    rel_label = ' (Relative)' if 'relative' in median_metric else ' (Absolute)'
    ax.set_xlabel(f"{norm_prefix}Median Lifespan{rel_label}", fontsize=14)
    ax.set_ylabel(f"{norm_prefix}Maximum Lifespan" + (" [years]" if value_type != 'normalized' else ""), fontsize=14)

    param_desc = 'Heterogeneity' if param_type == 'hetero' else 'Changes'
    if title:
        ax.set_title(title, fontsize=16)
    else:
        ax.set_title(
            f"Median vs Maximum Lifespan Plane – Parameter {param_desc}\n(from t={from_t})",
            fontsize=16
        )

    ax.grid(True, alpha=0.3)
    if value_type == 'normalized':
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)

    from matplotlib.font_manager import FontProperties
    entry_fontprops = FontProperties(family='Arial', size=legend_fontsize)
    title_fontprops = FontProperties(family='Arial', weight='bold', size=legend_fontsize + 6)

    legend = ax.legend(
        handles=legend_elements,
        title=legend_title,
        frameon=True,
        loc=legend_loc,
        prop=entry_fontprops,
        title_fontproperties=title_fontprops,
    )

    if legend:
        for text in legend.get_texts():
            text.set_fontsize(legend_fontsize)
            text.set_fontfamily('Arial')
        if legend.get_title():
            legend.get_title().set_fontweight('bold')
            legend.get_title().set_fontfamily('Arial')
            legend.get_title().set_fontsize(legend_fontsize + 6)

    if new_figure:
        plt.tight_layout()
        plt.show()

    return legend


def plot_max_lifespan(study_type='variation', ignore_kappa=True, ax=None, new_figure=True, from_t=0, pkl_file=None, normalized=False, **kwargs):
    """Plot maximum lifespan for parameter variations or heterogeneity.
    
    Args:
        study_type (str): Type of parameter study ('variation' or 'hetero'). Default is 'variation'.
        ignore_kappa (bool): Whether to exclude kappa parameter.
        ax (matplotlib.axes.Axes): Optional existing axis to plot on.
        new_figure (bool): Whether to create new figure.
        from_t (float): Starting time for analysis.
        pkl_file (str): Optional filename of pkl file in saved_results folder (filename only, not full path).
        normalized (bool): If True and study_type is 'variation', normalize values by factor=1.0 baseline.
        **kwargs: Optional plot styling arguments (linewidth, markersize, marker, etc.)
    """
    # Default plot styling values
    default_linewidth = 5
    default_markersize = 0
    default_marker = 'o'
    
    # Override defaults with any provided kwargs
    linewidth = kwargs.get('linewidth', default_linewidth)
    markersize = kwargs.get('markersize', default_markersize)
    marker = kwargs.get('marker', default_marker)
    
    if ax is None and new_figure:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set plot properties based on study_type
    if study_type == 'variation':
        default_pkl = 'param_variation_results.pkl'
        xlabel = 'Parameter Factor (relative to baseline)'
        title_type = 'Variation'
        def x_transform(x): return x
    elif study_type == 'hetero':
        default_pkl = 'param_distribution_results.pkl'
        xlabel = 'Parameter Variation (%)'
        title_type = 'Heterogeneity'
        def x_transform(x): return 100 * np.array(x)
        if normalized:
            print("Warning: 'normalized' is not supported for 'hetero' study_type and will be ignored.")
            normalized = False
    else:
        raise ValueError("study_type must be 'variation' or 'hetero'")

    # Load results
    if pkl_file is None:
        pkl_file = os.path.join('saved_results', default_pkl)

    try:
        results = load_results(pkl_file)
    except FileNotFoundError:
        return

    # Plot parameters (exclude kappa if requested)
    params = [p for p in ['eta', 'beta', 'kappa', 'epsilon', 'Xc'] if not (ignore_kappa and p == 'kappa')]
    
    # Legend names with only Unicode characters (no LaTeX)
    legend_labels = {
        'eta':       'Production (η)',
        'beta':      'Removal (β)',
        'Xc':        'Threshold (Xc)',
        'epsilon':   'Noise (ε)',
        'kappa':     'κ',
    }

    for param in params:
        if param not in results:
            continue
        data = results[param]
        # Only include factors that have the required from_t
        filtered = [(factor, v[from_t]['t_max']) for factor, v in sorted(data.items()) if from_t in v and 't_max' in v[from_t]]
        if not filtered:
            continue
        x_vals, y_vals = zip(*filtered)
        
        # Normalize if requested (only for variation)
        if normalized and study_type == 'variation':
            # Find baseline value (factor = 1.0)
            baseline_idx = None
            for i, factor in enumerate(x_vals):
                if abs(factor - 1.0) < 1e-6:  # Account for floating point precision
                    baseline_idx = i
                    break
            
            if baseline_idx is not None:
                baseline_value = y_vals[baseline_idx]
                y_vals = np.array(y_vals) / baseline_value
            else:
                print(f"Warning: No baseline (factor=1.0) found for parameter {param}")
        
        # Apply tiny bit of smoothing
        from scipy.ndimage import gaussian_filter1d
        if study_type == 'hetero':
            y_vals = gaussian_filter1d(y_vals, sigma=2.0)

        # legend label
        label = legend_labels.get(param, param)
        ax.plot(x_transform(x_vals), y_vals, color=utils.param_colors[param],
                linewidth=linewidth, marker=marker, markersize=markersize,
                label=label)

    # Formatting
    legend = ax.legend(title='Model Simulations,\nvariations in', loc='lower left', bbox_to_anchor=(-0.01, 0.4), frameon=True, fontsize=12, title_fontsize=13)
    # Make the legend title bold and set font family (like in the notebook):
    if legend:
        legend.get_title().set_fontweight('bold')
        legend.get_title().set_fontfamily('Arial')
        for text in legend.get_texts():
            text.set_fontfamily('Arial')
    
    ax.set_xlabel(xlabel, fontsize=14, fontfamily='Arial')
    ax.set_ylabel('Maximum Lifespan [years]', fontsize=14, fontfamily='Arial')
    ax.set_title(f'Maximum Lifespan vs Parameter {title_type} (from t={from_t})', fontsize=16, fontfamily='Arial')


def plot_historical_changes(
    hmd_data, 
    years, 
    ref_year=1960, 
    from_t=20, 
    method='iqr', 
    ax=None, 
    without_extrinsic=False, 
    cmap='viridis',
    show_colorbar=True    # New flag: Whether to display the colorbar/colorscale for the z axis (year)
):
    """
    Plot historical changes in lifespan and mortality steepness.

    Args:
        hmd_data: HMD data object
        years: Array of years to analyze
        ref_year: Reference year for normalization (default: 1960)
        from_t: Starting age for calculations (default: 20)
        method: Method for steepness calculation ('iqr' or 'cv')
        ax: Matplotlib axis to plot on (optional)
        without_extrinsic: If True, remove extrinsic mortality by fitting GGM and simulating death times
        cmap: Colormap to use for year coloring (default: 'viridis')
        show_colorbar (bool): Whether to include the colorbar on the plot (default: True)
    Returns:
        norm_medians (np.ndarray): Normalized median lifespans
        norm_steepnesses (np.ndarray): Normalized steepness values
    """
    medians = []
    steepnesses = []

    if without_extrinsic:
        # Use GGM fit and simulation to remove extrinsic mortality
        median_lifespans = []
        steepness_values = []
        year_values = []
        for year in years:
            mgg = hmd_data.fit_ggm(year=year)
            gg_temp = gg()
            gg_temp.a = mgg['a']
            gg_temp.b = mgg['b']
            gg_temp.c = mgg['c']
            gg_temp.m = 0
            death_times = gg_temp.sample_death_times(n=100000, min_age=from_t)
            median_lifespan = np.percentile(death_times, 50)
            iqr = np.abs(np.percentile(death_times, 25) - np.percentile(death_times, 75))
            steepness = median_lifespan / iqr
            median_lifespans.append(median_lifespan)
            steepness_values.append(steepness)
            year_values.append(year)
        # Reference values
        mgg_ref = hmd_data.fit_ggm(year=ref_year)
        gg_ref = gg()
        gg_ref.a = mgg_ref['a']
        gg_ref.b = mgg_ref['b']
        gg_ref.c = mgg_ref['c']
        gg_ref.m = 0
        ref_death_times = gg_ref.sample_death_times(n=100000, min_age=from_t)
        ref_median = np.percentile(ref_death_times, 50)
        ref_iqr = np.abs(np.percentile(ref_death_times, 25) - np.percentile(ref_death_times, 75))
        ref_steepness = ref_median / ref_iqr
        norm_medians = np.array(median_lifespans) / ref_median
        norm_steepnesses = np.array(steepness_values) / ref_steepness
        print(norm_medians[0], norm_steepnesses[0])
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        scatter = ax.scatter(
            norm_medians, 
            norm_steepnesses, 
            c=year_values, 
            cmap=cmap, 
            s=100, 
            marker='^', 
            zorder=100
        )
        ax.set_xlabel(f'Normalized Median Lifespan ({ref_year} = 1)')
        ax.set_ylabel(f'Normalized Steepness ({ref_year} = 1)')
        ax.set_title('Historical Changes in Lifespan and Mortality Steepness (No Extrinsic Mortality)\nSweden Period Data')
        ax.axis('square')
        ax.grid(True, alpha=0.3)
        if show_colorbar:
            plt.colorbar(scatter, ax=ax, label='Year')
        if ax is None:
            plt.tight_layout()
            plt.show()
        return norm_medians, norm_steepnesses
    else:
        # Original method (with extrinsic mortality)
        # Calculate reference values
        ref_median = hmd_data.calculate_median_lifespan(ref_year, age_start=from_t)
        ref_steepness = hmd_data.calculate_steepness(ref_year, age_start=from_t, method=method)
        # Loop through years
        for year in years:
            median = hmd_data.calculate_median_lifespan(year, age_start=from_t)
            medians.append(median)
            steepness = hmd_data.calculate_steepness(year, age_start=from_t, method=method)
            steepnesses.append(steepness)
        medians = np.array(medians)
        steepnesses = np.array(steepnesses)
        norm_medians = medians / ref_median
        norm_steepnesses = steepnesses / ref_steepness
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        scatter = ax.scatter(
            norm_medians, 
            norm_steepnesses, 
            c=years, 
            cmap=cmap, 
            s=100, 
            marker='^', 
            zorder=100
        )
        ax.set_xlabel(f'Normalized median lifespan ({ref_year} = 1)')
        ax.set_ylabel(f'Normalized steepness ({ref_year} = 1)')
        ax.set_title('Historical changes in lifespan and mortality steepness\nSweden period data')
        ax.axis('square')
        ax.grid(True, alpha=0.3)
        if show_colorbar:
            cbar = plt.colorbar(scatter, ax=ax, label='Year')
            cbar.ax.tick_params(labelsize=18)  # set fontsize for colorbar ticks
            cbar.set_label('Year', size=18)  # set fontsize for colorbar label
        if ax is None:
            plt.tight_layout()
            plt.show()
        return norm_medians, norm_steepnesses


def run_parameter_study(study_type='variation', baseline_dict=None, factors=None, stds=None, 
                       name='', n=int(3e5), params=None, include_h_ext=True, **kwargs):
    """
    Run parameter variation or heterogeneity study and save results.
    
    Args:
        study_type (str): 'variation' or 'hetero'
        baseline_dict (dict, optional): Baseline parameters. If None, loads default with adjustments.
        factors (array, optional): Factor values for variation study. Default: np.arange(0.3, 1.51, 0.1)
        stds (array, optional): Standard deviation values for hetero study. Default: [0, 0.02, ..., 0.3]
        name (str): Optional name suffix for saved file
        n (int): Number of simulations. Default: 200,000
        params (list, optional): Parameters to study. Default: ['eta', 'beta', 'epsilon', 'Xc']
        include_h_ext (bool): Whether to run the external hazard sweep.
        **kwargs: Additional arguments (tmax, etc.) passed to simulation
        
    Returns:
        dict: Results dictionary with all computed metrics
    """
    
    # Default parameters (excluding kappa)
    if params is None:
        params = ['eta', 'beta', 'epsilon', 'Xc']
    
    # Load baseline parameters with default adjustments
    if baseline_dict is None:
        base_dict = utils.load_baseline_human_params_dict()
        # Apply default adjustments
        base_dict['Xc'] = 1.08 * base_dict['Xc']
        base_dict['eta'] = 1.26 * base_dict['eta']
        base_dict['beta'] = 1.17 * base_dict['beta']
    else:
        base_dict = baseline_dict.copy()
    
    sim_kwargs = kwargs.copy()
    parallel = sim_kwargs.pop('parallel', True)
    base_h_ext = sim_kwargs.pop('h_ext', None)

    # Default values based on study type
    if study_type == 'variation':
        if factors is None:
            factors = np.arange(0.3, 1.71, 0.1)
        values = factors
        sim_kwargs.setdefault('tmax', 200)
        sim_kwargs.setdefault('break_early', False)
    elif study_type == 'hetero':
        if stds is None:
            stds = np.arange(0, 0.21, 0.01)
        values = stds
        sim_kwargs.setdefault('tmax', 300)
        sim_kwargs.setdefault('break_early', True)
    else:
        raise ValueError("study_type must be 'variation' or 'hetero'")
    
    from_t_values = [0, 15, 20, 30 , 40, 50]
    
    # Helper function to calculate metrics
    def calculate_metrics(sim, from_t_values):
        """Calculate steepness and survival metrics for given from_t values."""
        metrics = {}
        for from_t in from_t_values:
            metrics[from_t] = {
                'steepness_iqr_relative': sim.calc_steepness(method='IQR', from_t=from_t, relative=True),
                'steepness_iqr_absolute': sim.calc_steepness(method='IQR', from_t=from_t, relative=False),
                'steepness_cv_relative': sim.calc_steepness(method='CV', from_t=from_t, relative=True),
                'steepness_cv_absolute': sim.calc_steepness(method='CV', from_t=from_t, relative=False),
                't_median_relative': sim.find_time_at_survival(0.5, from_t=from_t, relative=True),
                't_median_absolute': sim.find_time_at_survival(0.5, from_t=from_t, relative=False),
                't_max': sim.find_time_at_survival(0.0001, from_t=from_t, relative=False)
            }
        return metrics
    
    # Helper function for variation study
    def prepare_params_dict_variation(base_dict, n, param=None, factor=1.0):
        """Prepare parameter dictionary with proper array expansion for variation study."""
        dict_copy = base_dict.copy()
        for key in dict_copy.keys():
            if isinstance(dict_copy[key], (np.ndarray, list)) and len(dict_copy[key]) == 1:
                dict_copy[key] = np.repeat(dict_copy[key], n)
            elif np.isscalar(dict_copy[key]):
                dict_copy[key] = np.repeat(np.array([dict_copy[key]]), n)
        
        if param:
            dict_copy[param] = factor * dict_copy[param]
        
        return dict_copy
    
    # Run baseline simulation
    if study_type == 'variation':
        baseline_dict_sim = prepare_params_dict_variation(base_dict, n)
        baseline_sim = utils.create_sr_simulation(
            params_dict=baseline_dict_sim,
            n=n,
            parallel=parallel,
            h_ext=base_h_ext,
            **sim_kwargs,
        )
    else:  # hetero
        baseline_sim = utils.create_sr_simulation(
            params_dict=base_dict,
            n=n,
            parallel=parallel,
            h_ext=base_h_ext,
            **sim_kwargs,
        )
    
    baseline_results = calculate_metrics(baseline_sim, from_t_values)
    
    # Initialize results structure
    results = {'baseline': baseline_results}
    
    # Run parameter study
    for param in params:
        print(f"Processing parameter: {param}")
        results[param] = {}
        
        for value in values:
            if study_type == 'variation':
                param_dict = prepare_params_dict_variation(base_dict, n, param, value)
                sim = utils.create_sr_simulation(
                    params_dict=param_dict,
                    n=n,
                    parallel=parallel,
                    h_ext=base_h_ext,
                    **sim_kwargs,
                )
                print(f"  Factor: {value:.2f} completed")
            else:  # hetero
                param_dict = utils.create_param_distribution_dict(
                    params=param, 
                    std=value, 
                    n=n, 
                    dist_type='gaussian', 
                    params_dict=base_dict, 
                    family='None'
                )
                sim = utils.create_sr_simulation(
                    params_dict=param_dict,
                    n=n,
                    parallel=parallel,
                    h_ext=base_h_ext,
                    **sim_kwargs,
                )
                print(f"  Std: {value:.2f} completed")
            
            results[param][value] = calculate_metrics(sim, from_t_values)
    
    if include_h_ext:
        print("\nProcessing h_ext values...")
        h_ext_values = np.logspace(-4, -2, 10)
        results['h_ext'] = {}
        
        for h_ext in h_ext_values:
            if study_type == 'variation':
                baseline_dict_sim = prepare_params_dict_variation(base_dict, n)
                sim = utils.create_sr_simulation(
                    params_dict=baseline_dict_sim,
                    h_ext=h_ext,
                    n=n,
                    parallel=parallel,
                    **sim_kwargs,
                )
            else:  # hetero
                sim = utils.create_sr_simulation(
                    params_dict=base_dict,
                    h_ext=h_ext,
                    n=n,
                    parallel=parallel,
                    **sim_kwargs,
                )
            results['h_ext'][h_ext] = calculate_metrics(sim, from_t_values)
            print(f"  h_ext: {h_ext:.2e} completed")
    
    # Save results
    if name:
        filename = f'param_{study_type}_results_{name}.pkl'
    else:
        filename = f'param_{study_type}_results.pkl'
    
    filepath = saved_results_path + filename
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nAll simulations completed and saved to: {filepath}")
    return results


def quick_steepness_longevity(pkl_file='param_variation_results.pkl', new_markers=None, 
                               from_t=20, h_ext=True, figsize=(10, 8), 
                               xlim=None, ylim=None, title=None):
    """
    Quick plotting function for steepness-longevity plane with custom data points.
    
    Args:
        pkl_file (str): Pickle file name (default: 'param_variation_results.pkl')
                       Options: 'param_variation_results_usa_2019.pkl', 
                               'param_variation_results_usa_2019_with_hetero.pkl',
                               'param_variation_results_sweden_cohort_1900.pkl', etc.
        new_markers (list of dict, optional): List of marker dictionaries with keys:
            - 'x': x-coordinate (normalized median ratio)
            - 'y': y-coordinate (normalized steepness ratio)
            - 'label': label for the marker
            - 'color': marker color (default: 'red')
            - 'marker': marker style (default: '*')
            - 'size': marker size (default: 300)
            - 'annotate': whether to add annotation (default: True)
            - 'annotation_offset': tuple (dx, dy) for annotation position (default: (0.05, 0.05))
        from_t (int): Starting age for analysis (default: 20)
        h_ext (bool): Include extrinsic mortality line (default: True)
        figsize (tuple): Figure size (default: (10, 8))
        xlim (tuple, optional): X-axis limits (min, max)
        ylim (tuple, optional): Y-axis limits (min, max)
        title (str, optional): Custom title for the plot
    
    Returns:
        fig, ax: Matplotlib figure and axis objects
        
    Example usage:
        # Basic plot with one marker
        markers = [{
            'x': 1.072, 
            'y': 1.195, 
            'label': 'CHD Repair',
            'color': 'red'
        }]
        fig, ax = th.quick_steepness_longevity(
            pkl_file='param_variation_results_usa_2019_with_hetero.pkl',
            new_markers=markers
        )
        
        # Multiple markers with custom styling
        markers = [
            {'x': 1.05, 'y': 1.1, 'label': 'Intervention A', 'color': 'blue', 'marker': 'o'},
            {'x': 1.08, 'y': 1.15, 'label': 'Intervention B', 'color': 'green', 'marker': 's', 
             'annotate': False}
        ]
        fig, ax = th.quick_steepness_longevity(
            pkl_file='param_variation_results_usa_2019.pkl',
            new_markers=markers,
            xlim=(0.95, 1.15),
            ylim=(0.95, 1.25)
        )
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot steepness-longevity background
    plot_steepness_longevity(
        pkl_file=pkl_file,
        param_type='variation',
        from_t=from_t,
        longevity_metric='t_median_absolute',
        steepness_metric='steepness_iqr_absolute',
        ignore_kappa=True,
        ax=ax,
        value_type='normalized',
        h_ext=h_ext,
        linewidth=2,
        legend_fontsize=12
    )
    
    # Add custom markers if provided
    if new_markers:
        for marker_dict in new_markers:
            # Extract marker properties with defaults
            x = marker_dict['x']
            y = marker_dict['y']
            label = marker_dict.get('label', 'Custom Point')
            color = marker_dict.get('color', 'red')
            marker_style = marker_dict.get('marker', '*')
            size = marker_dict.get('size', 300)
            annotate = marker_dict.get('annotate', True)
            annotation_offset = marker_dict.get('annotation_offset', (0.05, 0.05))
            
            # Edge color is darker version
            if color == 'red':
                edgecolor = 'darkred'
            elif color == 'blue':
                edgecolor = 'darkblue'
            elif color == 'green':
                edgecolor = 'darkgreen'
            else:
                edgecolor = color
            
            # Plot the marker
            ax.scatter(x, y, 
                      s=size, 
                      color=color, 
                      marker=marker_style,
                      edgecolors=edgecolor,
                      linewidth=2,
                      zorder=100,
                      label=label)
            
            # Add annotation if requested
            if annotate:
                ax.annotate(label, 
                           xy=(x, y),
                           xytext=(x + annotation_offset[0], y + annotation_offset[1]),
                           fontsize=12, 
                           fontweight='bold',
                           color=edgecolor,
                           fontfamily='Arial',
                           arrowprops=dict(arrowstyle='->', color=edgecolor, lw=2))
    
    # Set custom limits if provided, otherwise use sensible defaults
    if xlim:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(0.85, 1.15)
    
    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(0.75, 1.25)
    
    # Format axes
    ax.set_xlabel('Normalized Median Lifespan', fontsize=14, fontfamily='Arial')
    ax.set_ylabel('Normalized Steepness (IQR)', fontsize=14, fontfamily='Arial')
    
    # Set title
    if title:
        ax.set_title(title, fontsize=16, fontfamily='Arial')
    else:
        # Default title based on pkl file
        if 'usa' in pkl_file.lower():
            cohort_str = 'USA 2019'
            if 'hetero' in pkl_file.lower():
                cohort_str += ' with Xc Heterogeneity'
        elif 'sweden' in pkl_file.lower():
            cohort_str = 'Sweden'
            if '1900' in pkl_file:
                cohort_str += ' Cohort 1900'
        else:
            cohort_str = 'Baseline'
        ax.set_title(f'Steepness-Longevity Plane ({cohort_str})', 
                    fontsize=16, fontfamily='Arial')
    
    ax.grid(True, alpha=0.3)
    
    # Update legend to include new markers
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles, labels, loc='best', fontsize=10, frameon=True)
    
    # Set font family for legend text
    if legend:
        for text in legend.get_texts():
            text.set_fontfamily('Arial')
    
    plt.tight_layout()
    
    return fig, ax


def map_xc_factor_to_years(
    hmd_data,
    years,
    ref_year=2020,
    from_t=20,
    pkl_file=None,
    n_interp=1000,
    n_samples=50_000,
    ax=None,
    plot=True,
):
    """
    For each year in an HMD dataset, find the Xc factor on the model's Xc curve
    (in the normalized steepness-longevity plane) that is closest to that year's
    data point -- with extrinsic mortality removed (GGM fit, m=0).

    Args:
        hmd_data: HMD data object (e.g. sweden_period)
        years: array-like of years to analyse
        ref_year: reference year whose normalised coordinates are (1, 1)
        from_t: starting age for steepness / longevity calculations
        pkl_file: pickle filename with parameter-variation results
                  (default: 'param_variation_results.pkl')
        n_interp: number of points for cubic interpolation of the Xc curve
        n_samples: number of death-time samples per GGM simulation
        ax: matplotlib axis (created if None and plot=True)
        plot: whether to draw Xc-factor vs year

    Returns:
        xc_per_year : dict  {year: xc_factor}
        years_arr   : np.ndarray of sorted years
        factors_arr : np.ndarray of corresponding Xc factors
    """
    from scipy.interpolate import interp1d

    # ---- A. Build fine-grained Xc curve from stored simulation results ----
    if pkl_file is None:
        pkl_file = 'param_variation_results.pkl'
    results = load_results(pkl_file)

    baseline_steepness = results['baseline'][from_t]['steepness_iqr_absolute']
    baseline_longevity = results['baseline'][from_t]['t_median_absolute']

    xc_factors_raw, xc_long_raw, xc_steep_raw = [], [], []
    for factor in sorted(results['Xc'].keys()):
        if from_t not in results['Xc'][factor]:
            continue
        d = results['Xc'][factor][from_t]
        s = d.get('steepness_iqr_absolute')
        l = d.get('t_median_absolute')
        if s is not None and l is not None:
            xc_factors_raw.append(factor)
            xc_long_raw.append(l / baseline_longevity)
            xc_steep_raw.append(s / baseline_steepness)

    xc_factors_raw = np.array(xc_factors_raw)
    xc_long_raw    = np.array(xc_long_raw)
    xc_steep_raw   = np.array(xc_steep_raw)

    # Cubic interpolation to fine resolution
    fine_factors = np.linspace(xc_factors_raw.min(), xc_factors_raw.max(), n_interp)
    interp_long  = interp1d(xc_factors_raw, xc_long_raw,  kind='cubic')(fine_factors)
    interp_steep = interp1d(xc_factors_raw, xc_steep_raw, kind='cubic')(fine_factors)

    # ---- B. Compute historical points (extrinsic mortality removed) ----
    def _ggm_metrics(year):
        """Fit GGM, set m=0, sample death times, return (median, steepness)."""
        mgg = hmd_data.fit_ggm(year=year)
        gg_temp = gg()
        gg_temp.a = mgg['a']
        gg_temp.b = mgg['b']
        gg_temp.c = mgg['c']
        gg_temp.m = 0  # remove extrinsic mortality
        death_times = gg_temp.sample_death_times(n=n_samples, min_age=from_t)
        median = np.percentile(death_times, 50)
        iqr = np.abs(np.percentile(death_times, 25) - np.percentile(death_times, 75))
        steepness = median / iqr
        return median, steepness

    # Reference year values
    ref_median, ref_steepness = _ggm_metrics(ref_year)

    # All years
    year_medians = []
    year_steepnesses = []
    valid_years = []
    for year in years:
        try:
            med, st = _ggm_metrics(year)
            year_medians.append(med / ref_median)
            year_steepnesses.append(st / ref_steepness)
            valid_years.append(year)
        except Exception as e:
            print(f"  Skipping year {year}: {e}")

    year_medians     = np.array(year_medians)
    year_steepnesses = np.array(year_steepnesses)
    valid_years      = np.array(valid_years)

    # ---- C. Map each year to the nearest point on the Xc curve ----
    xc_per_year = {}
    factors_list = []
    for i, year in enumerate(valid_years):
        dists = np.sqrt(
            (interp_long  - year_medians[i])**2 +
            (interp_steep - year_steepnesses[i])**2
        )
        best_factor = fine_factors[np.argmin(dists)]
        xc_per_year[int(year)] = best_factor
        factors_list.append(best_factor)

    factors_arr = np.array(factors_list)

    # ---- D. Optional plot ----
    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(valid_years, factors_arr, '-o', markersize=3, linewidth=1.5)
        ax.axhline(y=1, linestyle='--', color='gray', alpha=0.6)
        ax.set_xlabel('Year', fontsize=14, fontfamily='Arial')
        ax.set_ylabel('Xc Factor', fontsize=14, fontfamily='Arial')
        ax.set_title(
            f'Xc factor vs year (extrinsic mortality removed, from_t={from_t})',
            fontsize=16, fontfamily='Arial'
        )
        ax.tick_params(labelsize=12)

    return xc_per_year, valid_years, factors_arr
