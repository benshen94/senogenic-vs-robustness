import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import plotly.graph_objects as go
from scipy import interpolate, integrate
from scipy.stats import linregress
from scipy.optimize import curve_fit
import seaborn as sns
from matplotlib.colors import ListedColormap
import os

########## LOG S STUFF #############

def calc_late_time_log_survival_fits(self, year, ages_start=90, ages_end=110, age_jumps=5):
    """
    Calculate fits to log survival curves at late times.
    Analyzes mortality acceleration in elderly population.
    """
    ages, survival = self.calculate_survival(year)
    age_ranges = np.arange(ages_start, ages_end+1, age_jumps)
    
    def exp_func(t, t_0, n):
        return (-(t/t_0)**n)
    
    def calculate_fit(start, end):
        mask = (ages >= start) & (ages <= end)
        ages_filtered = ages[mask]
        survival_filtered = survival[mask]
        
        # Normalize survival to start at 1 for each segment
        survival_normalized = survival_filtered / survival_filtered[0]
        
        # Shift ages to start at 0 for each segment
        ages_shifted = ages_filtered - start
        
        # Fit the function to the segment
        popt, pcov = curve_fit(exp_func, ages_shifted, np.log(survival_normalized), p0=[1, 1])
        t_0, n = popt
        
        # Calculate confidence intervals
        perr = np.sqrt(np.diag(pcov))
        t_0_ci = (t_0 - 1.96*perr[0], t_0 + 1.96*perr[0])
        n_ci = (n - 1.96*perr[1], n + 1.96*perr[1])
        
        # Calculate R-squared
        y_fit = exp_func(ages_shifted, *popt)
        ss_res = np.sum((survival_normalized - y_fit) ** 2)
        ss_tot = np.sum((survival_normalized - np.mean(survival_normalized)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {
            'segment': f'{start}-{end}',
            't_0': t_0,
            't_0_ci': t_0_ci,
            'n': n,
            'n_ci': n_ci,
            'r_squared': r_squared
        }

    results = [calculate_fit(start, end) for start, end in zip(age_ranges[:-1], age_ranges[1:])]
    average_ages = (age_ranges[:-1] + age_ranges[1:]) / 2

            # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(ages, survival, 'b.', label='Data')
    ax.set_yscale('log')

    for result in results:
        start, end = map(int, result['segment'].split('-'))
        x_fit = np.linspace(start, end, 100)
        y_fit = np.exp(exp_func(x_fit - start, result['t_0'], result['n']))
        ax.plot(x_fit, y_fit * survival[ages == start][0], 
                label=f"Fit {result['segment']}: n={result['n']:.3f}, t_0={result['t_0']:.3f}")

    ax.set_xlabel('Age')
    ax.set_ylabel('Survival')
    ax.set_title(f'Late Time Log Survival Fits - Year {year}')
    ax.legend()
    plt.tight_layout()
    plt.show()

    return average_ages, results

def calc_log_survival_slopes(self, year, ages_start=90, ages_end=110, age_jumps=5):
    """
    Calculate slopes of log survival curves in different age segments.
    Used to analyze mortality acceleration patterns.
    """
    ages, survival = self.calculate_survival(year)
    age_ranges = np.arange(ages_start, ages_end + 1, age_jumps)
    age_ranges = np.append(age_ranges, 110) if age_ranges[-1] != 110 else age_ranges
    
    def fit_segment(start, end):
        mask = (ages >= start) & (ages <= end) & (survival > 0)
        if np.sum(mask) < 2:  # Need at least 2 points for linear regression
            return None
        result = linregress(ages[mask], np.log10(survival[mask]))
        return {
            'segment': f'{start}-{end}',
            'slope': result.slope,
            'stderr': result.stderr,
            'intercept': result.intercept,
            'r_squared': result.rvalue**2
        }

    results = [fit_segment(start, end) for start, end in zip(age_ranges[:-1], age_ranges[1:])]
    results = [r for r in results if r is not None]  # Filter out None results
    average_ages = [(int(r['segment'].split('-')[0]) + int(r['segment'].split('-')[1])) / 2 for r in results]

    return average_ages, results

def plot_survival_with_log_slopes(self, years, ax=None, ages_start=90, ages_end=110, age_jumps=5):
    """
    Plot survival curves with log-scale slopes for different age segments.
    Visualizes how mortality acceleration changes with age.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    years = np.atleast_1d(years)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(years)))

    for year, color in zip(years, colors):
        ages, survival = self.calculate_survival(year)
        average_ages, results = self.calc_log_survival_slopes(year, ages_start, ages_end, age_jumps)

        valid_mask = survival > 0
        ax.plot(ages[valid_mask], np.log10(survival[valid_mask]), '.', color=color, label=f'Data {year}')
        for i, r in enumerate(results):
            start, end = map(int, r['segment'].split('-'))
            x_fit = np.linspace(start, end, 100)
            ax.plot(x_fit, r['slope'] * x_fit + r['intercept'], 
                    color=plt.cm.Set2(i / len(results)),
                    label=f"Fit {r['segment']} (Year {year})")

    ax.set(xlabel='Age', ylabel='Log(Survival)', 
            title='Late Time Log Survival Slopes')
    ax.legend()
    return ax

def _get_color_gradient(self, years):
    """
    Helper function to create color gradients for plotting.
    Returns list of colors for visualizing temporal trends.
    """
    if len(years) == 1:
        return ['blue']
    cmap = LinearSegmentedColormap.from_list("custom", ["red", "blue"])
    return [cmap(i) for i in np.linspace(0, 1, len(years))]

def plot_log_survival_slopes(self, years, ax=None, ages_start=90, ages_end=110, age_jumps=5, do_fit=True):
    """
    Plot slopes of log survival curves over time.
    Shows how mortality acceleration changes across years.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    years = np.atleast_1d(years)
    colors = self._get_color_gradient(years)

    # If do_fit is True, fit all years. If it's a list, only fit those years.
    years_to_fit = years if do_fit is True else (np.array(do_fit) if isinstance(do_fit, (list, np.ndarray)) else np.array([]))

    fit_params_dict = {}  # Dictionary to store fit parameters

    for year, color in zip(years, colors):
        average_ages, results = self.calc_log_survival_slopes(year, ages_start, ages_end, age_jumps)
        abs_slopes = np.abs([r['slope'] for r in results])
        yerr = np.maximum(np.array([r['stderr'] for r in results]), 1e-10)

        ax.errorbar(average_ages, abs_slopes, yerr=yerr, fmt='o', capsize=5, color=color, label=f'Year {year}')
        
        if year in years_to_fit:
            fit_params = np.polyfit(average_ages, abs_slopes, 1)
            x_fit = np.linspace(min(average_ages), max(average_ages), 100)
            ax.plot(x_fit, np.poly1d(fit_params)(x_fit), '--', color=color)
            
            # Store fit parameters in the dictionary
            fit_params_dict[year] = {
                'slope': fit_params[0],
                'intercept': fit_params[1]
            }

    if len(years_to_fit) > 0:
        # Add textbox with fit details
        fit_text = "\n".join([f"Year {year}: y = {fit_params_dict[year]['slope']:.3e}x + {fit_params_dict[year]['intercept']:.3e}" for year in years_to_fit])
        ax.text(0.95, 0.05, fit_text, transform=ax.transAxes, 
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set(xlabel='Average Age', ylabel='|log(S) Slope| , hazard [1/age]',
        title='Absolute Slopes with Standard Errors' + (' and Linear Trends' if len(years_to_fit) > 0 else ''))
    ax.legend()
    
    return ax, fit_params_dict

def plot_log_survival_slopes_inverse_age(self, years, ax=None, ages_start=90, ages_end=110, age_jumps=5):
    """
    Plot slopes of log survival curves against inverse age.
    Alternative visualization for analyzing mortality acceleration.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    years = np.atleast_1d(years)
    colors = self._get_color_gradient(years)

    for year, color in zip(years, colors):
        average_ages, results = self.calc_log_survival_slopes(year, ages_start, ages_end, age_jumps)
        abs_slopes = np.abs([r['slope'] for r in results])
        yerr = np.maximum(np.array([r['stderr'] for r in results]), 1e-10)
        inverse_ages = 1 / average_ages

        ax.errorbar(inverse_ages, abs_slopes, yerr=yerr, fmt='o', capsize=5, color=color, label=f'Year {year}')
        
        fit = np.polyfit(inverse_ages, abs_slopes, 1)
        x_fit = np.linspace(min(inverse_ages), max(inverse_ages), 100)
        ax.plot(x_fit, np.poly1d(fit)(x_fit), '--', color=color)

    # Add textbox with fit details
    fit_text = "\n".join([f"Year {year}: y = {np.polyfit(1/self.calc_log_survival_slopes(year, ages_start, ages_end, age_jumps)[0], np.abs([r['slope'] for r in self.calc_log_survival_slopes(year, ages_start, ages_end, age_jumps)[1]]), 1)[0]:.3e}x + {np.polyfit(1/self.calc_log_survival_slopes(year, ages_start, ages_end, age_jumps)[0], np.abs([r['slope'] for r in self.calc_log_survival_slopes(year, ages_start, ages_end, age_jumps)[1]]), 1)[1]:.3e}" for year in years])
    ax.text(0.95, 0.05, fit_text, transform=ax.transAxes, 
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set(xlabel='1 / Average Age', ylabel='|Slope|',
            title='Absolute Slopes vs Inverse Age with Standard Errors and Linear Trends')
    ax.legend()
    return ax

def plot_late_time_hazard_increase_slope_vs_years(self, years, ax=None, ages_start=90, ages_end=110, age_jumps=5):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    years = np.atleast_1d(years)
    colors = self._get_color_gradient(years)

    slopes = []
    slope_errors = []
    for year, color in zip(years, colors):
        average_ages, results = self.calc_log_survival_slopes(year, ages_start, ages_end, age_jumps)
        abs_slopes = np.abs([r['slope'] for r in results])
        fit = linregress(average_ages, abs_slopes)
        slopes.append(fit.slope)
        slope_errors.append(fit.stderr)
        
        # Plot each point individually with its color
        ax.errorbar(year, fit.slope, yerr=fit.stderr, fmt='o', capsize=5, color=color)

    ax.set(xlabel='Year', ylabel='hazard increase slope vs year',
            title='Trend of Mortality Acceleration Over Time')

    return ax

def plot_late_time_hazard_increase_slope_vs_intercepts(self, years, ages_start=90, ages_end=110, age_jumps=5, do_fit=False):
    years = np.array(years)
    slopes = []
    intercepts = []
    slope_errors = []
    intercept_errors = []
    
    for year in years:
        average_ages, results = self.calc_log_survival_slopes(year, ages_start, ages_end, age_jumps)
        abs_slopes = np.abs([r['slope'] for r in results])
        fit = linregress(average_ages, abs_slopes)
        slopes.append(fit.slope)
        intercepts.append(fit.intercept)
        slope_errors.append(fit.stderr)
        intercept_errors.append(fit.intercept_stderr)

    slopes = np.array(slopes)
    intercepts = np.array(intercepts)

    # Create color scale
    colors = [f'rgb({int(255*(1-i/(len(years)-1)))},{0},{int(255*(i/(len(years)-1)))})' for i in range(len(years))]

    fig = go.Figure(data=go.Scatter(
        x=intercepts,
        y=slopes,
        mode='markers',
        marker=dict(
            size=10,
            color=years,
            colorscale=colors,
            colorbar=dict(title='Year'),
            showscale=True
        ),
        text=[f'Year: {year}<br>Slope: {slope:.3e} ± {slope_err:.3e}<br>Intercept: {intercept:.3e} ± {intercept_err:.3e}' 
            for year, slope, slope_err, intercept, intercept_err in zip(years, slopes, slope_errors, intercepts, intercept_errors)],
        hoverinfo='text',
        error_x=dict(
            type='data',
            array=intercept_errors,
            visible=True
        ),
        error_y=dict(
            type='data',
            array=slope_errors,
            visible=True
        )
    ))

    if do_fit:
        # Perform linear regression
        fit = np.polyfit(intercepts, slopes, 1)
        fit_fn = np.poly1d(fit)
        
        # Create x values for the fit line
        x_fit = np.linspace(min(intercepts), max(intercepts), 100)
        
        # Add fit line to the figure
        fig.add_trace(go.Scatter(
            x=x_fit,
            y=fit_fn(x_fit),
            mode='lines',
            name='Linear Fit',
            line=dict(color='red', dash='dash')
        ))
        
        # Add fit equation to the layout
        fit_eq = f'y = {fit[0]:.2e}x + {fit[1]:.2e}'
        fig.add_annotation(
            xref='paper', yref='paper',
            x=0.95, y=0.05,
            text=fit_eq,
            showarrow=False,
            bgcolor='white',
            bordercolor='black',
            borderwidth=1
        )

    fig.update_layout(
        xaxis_title='Intercept',
        yaxis_title='Slope',
        hovermode='closest',
        width=800,
        height=600,
    )
    return fig



class Hazard2006Data:
    def __init__(self, gender):
        if gender.lower() not in ['male', 'female']:
            raise ValueError("Gender must be 'male' or 'female'")
        parent_folder = os.path.dirname(os.getcwd())
        self.filename = os.path.join(parent_folder, f'datasets/mortality_datasets/us_{gender.lower()}_2006.json')
        self.gender = gender
        self.hazard_ext, self.time_ext = self.load_hazard_data('ext')
        self.hazard_int, self.time_int = self.load_hazard_data('int')
        self.interpolate_hazards()
        self.hazard_tot = self.hazard_int + self.hazard_ext
        self.tspan = self.time_ext
        self.survival = self.calc_survival(self.hazard_tot)
        self.med_time = interpolate.interp1d(self.survival, self.tspan)(0.5)

    def load_hazard_data(self, dataset_name):
        with open(self.filename, 'r') as file:
            json_data = json.load(file)
        
        dataset = next((item for item in json_data['datasetColl'] if item['name'] == dataset_name), None)
        if not dataset:
            raise ValueError(f"Dataset '{dataset_name}' not found in the file.")

        data_points = dataset['data']
        hazard = []
        time = []        

        for point in data_points:
            values = point.get('value', [])
            if len(values) == 2:
                time.append(values[0])
                hazard.append(values[1])

        if self.gender == 'male':
            hazard = hazard[:-8]
            time = time[:-8]

        return hazard, time
    
    # make sure hazards are same size and at same timesteps
    def interpolate_hazards(self):
        # Determine the shorter time vector
        shorter_time = self.time_int if len(self.time_int) < len(self.time_ext) else self.time_ext

        # Interpolate the other hazard to match the time points of the shorter one
        if len(self.time_int) < len(self.time_ext):
            self.hazard_ext = np.interp(shorter_time, self.time_ext, self.hazard_ext)
            self.time_ext = shorter_time
        elif len(self.time_ext) < len(self.time_int):
            self.hazard_int = np.interp(shorter_time, self.time_int, self.hazard_int)
            self.time_int = shorter_time

    def calc_survival(self, hazard):
        survival = np.exp(-integrate.cumtrapz(hazard, self.tspan))
        return survival
    

    ################################################################################################
################################################################################################
    ################################################################################################
    ################################################################################################
