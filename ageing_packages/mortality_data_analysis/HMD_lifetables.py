import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import plotly.express as px
import plotly.graph_objects as go
from scipy import interpolate, integrate
from scipy.stats import linregress
from scipy.optimize import curve_fit
import seaborn as sns
import os
from pathlib import Path


class HMD:
    """
    A class to handle loading, analyzing, and plotting mortality data from the Human Mortality Database (HMD).
    """

    # ======================================================================================
    # SECTION 1: INITIALIZATION AND DATA LOADING
    # ======================================================================================

    def __init__(self, country, gender, data_type='period'):
        """
        Initializes the HMD class by loading mortality data for a specific country and gender.

        Parameters
        ----------
        country : str
            The country code (e.g., 'USA', 'SWE').
        gender : str
            The gender ('male', 'female', or 'both').
        data_type : str, default 'period'
            The type of lifetables to load ('period' or 'cohort').
        """
        self.country = country
        self.gender = gender
        repo_root = Path(__file__).resolve().parents[2]
        self.data_folder = Path(
            os.environ.get("SENOGENIC_HMD_DATA_DIR", repo_root / "saved_data" / "hmd")
        )
        self.data = self._load_data(data_type, gender)

    def _load_data(self, data_type, gender):
        """
        Loads mortality data from HMD text files.
        """
        gender_initial = {'both': 'b', 'female': 'f', 'male': 'm'}.get(gender.lower())
        if not gender_initial:
            raise ValueError("Invalid gender. Use 'both', 'female', or 'male'.")

        file_patterns = {
            'period': self.data_folder / f"mortality.org_File_GetDocument_hmd.v6_{self.country.upper()}_STATS_{gender_initial}ltper_1x1.txt",
            'cohort': self.data_folder / f"{self.country.lower()}_cohort_data_{gender}.txt"
        }
        file_name = file_patterns.get(data_type)
        if not file_name:
            raise ValueError(f"Invalid data_type: {data_type}. Use 'period' or 'cohort'.")

        try:
            data = pd.read_csv(file_name, sep='\s+')
            data['Age'] = data['Age'].str.rstrip('+').astype(int)
            for col in ['qx', 'mx']:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {file_name}. Please check the path and country code.")

    @property
    def available_years(self):
        """Returns sorted array of all years available in the dataset."""
        return np.sort(self.data['Year'].unique())
    
    def _filter_available_years(self, years):
        """
        Filters input years to only include years for which data is available.
        
        Parameters
        ----------
        years : int, list, or array-like
            The requested year(s).
            
        Returns
        -------
        np.ndarray
            Array of years that are available in the data, sorted ascending.
        """
        years = np.atleast_1d(years)
        available = set(self.available_years)
        filtered = np.array([y for y in years if y in available])
        return np.sort(filtered)
    
    def is_year_available(self, year):
        """Check if data is available for a given year."""
        return year in self.available_years

    # ======================================================================================
    # SECTION 2: CORE DATA CALCULATION
    # ======================================================================================

    def get_hazard(self, year, haz_type='mx', strict=False):
        """
        Retrieves the mortality hazard rate for a specific year.

        Parameters
        ----------
        year : int
            The year to retrieve data for.
        haz_type : str, default 'mx'
            The type of hazard rate to return ('mx' or 'qx').
        strict : bool, default False
            If True, raises ValueError for unavailable years.
            If False, returns (None, None) for unavailable years.

        Returns
        -------
        tuple
            (ages, hazard_rates) or (None, None) if year not available and strict=False
        """
        if year not in self.data['Year'].values:
            if strict:
                raise ValueError(f"Data for year {year} is not available.")
            return None, None
        year_data = self.data[self.data['Year'] == year]
        return year_data['Age'].values, year_data[haz_type].values

    def get_survival(self, year, strict=False):
        """
        Retrieves the survival probability (lx) for a specific year.

        Parameters
        ----------
        year : int
            The year to retrieve data for.
        strict : bool, default False
            If True, raises ValueError for unavailable years.
            If False, returns (None, None) for unavailable years.

        Returns
        -------
        tuple
            (ages, survival_probabilities) or (None, None) if year not available and strict=False
        """
        if year not in self.data['Year'].values:
            if strict:
                raise ValueError(f"Data for year {year} is not available.")
            return None, None
        year_data = self.data[self.data['Year'] == year]
        return year_data['Age'].values, year_data['lx'].values / 100000

    def get_lifespan_distribution(self, year, age_start=0, age_stop=None):
        """
        Calculates the lifespan distribution (probability of death at each age).
        
        Parameters
        ----------
        year : int or list
            The year(s) to compute the distribution for.
        age_start : int, default 0
            The starting age for the distribution.
        age_stop : int, optional
            The ending age for the distribution.
        
        Returns
        -------
        tuple
            (ages, distribution) where distribution is a 1D array for a single year
            or a 2D array (n_ages x n_years) for multiple years.
        """
        def compute_dist(y):
            ages, survival = self.get_survival(y)
            if ages is None:
                return None, None
            mask = (ages >= age_start) & (ages <= (age_stop if age_stop is not None else ages.max()))
            ages, survival = ages[mask], survival[mask]
            
            if len(ages) == 0:
                return None, None
                
            survival /= survival[0]  # Normalize to 1 at age_start
            
            dist = np.diff(survival)
            dist = -np.append(dist, survival[-1]) # p(death at age) = S(t) - S(t+1)
            dist[dist < 0] = 0 # Ensure non-negative probabilities
            dist /= dist.sum() # Normalize to sum to 1

            return ages, dist

        if np.isscalar(year):
            result = compute_dist(year)
            if result[0] is None:
                return None, None
            return result
        else:
            years_filtered = self._filter_available_years(year)
            if len(years_filtered) == 0:
                return None, None
            results = [compute_dist(y) for y in years_filtered]
            valid_results = [(ages, dist) for ages, dist in results if ages is not None]
            if len(valid_results) == 0:
                return None, None
            ages = valid_results[0][0]
            all_dists = [dist for _, dist in valid_results]
            return ages, np.column_stack(all_dists)

    def get_initial_mortality(self, year, haz_type='mx'):
        """
        Retrieves the initial mortality rate (at age 0) for one or more years.
        
        Parameters
        ----------
        year : int or list
            The year(s) to retrieve data for.
        haz_type : str, default 'mx'
            The type of hazard rate to return ('mx' or 'qx').
        
        Returns
        -------
        float or dict
            If a single year is provided, returns the initial mortality rate.
            If multiple years are provided, returns a dict {year: initial_mortality}.
        """
        def get_single_year(y):
            ages, hazards = self.get_hazard(y, haz_type)
            if ages is None:
                return np.nan
            age_zero_idx = np.where(ages == 0)[0]
            if len(age_zero_idx) == 0:
                return np.nan
            return hazards[age_zero_idx[0]]
        
        if np.isscalar(year):
            return get_single_year(year)
        else:
            years_filtered = self._filter_available_years(year)
            return {y: get_single_year(y) for y in years_filtered}

    def get_minimum_hazard(self, year, haz_type='mx', age_range=None):
        """
        Finds the minimum hazard rate and the age at which it occurs for one or more years.
        
        Parameters
        ----------
        year : int or list
            The year(s) to analyze.
        haz_type : str, default 'mx'
            The type of hazard rate to use ('mx' or 'qx').
        age_range : tuple, optional
            (min_age, max_age) to restrict the search range. If None, searches all ages.
        
        Returns
        -------
        tuple or dict
            If a single year is provided, returns (age_at_min, min_hazard).
            If multiple years are provided, returns a dict {year: (age_at_min, min_hazard)}.
        """
        def get_single_year(y):
            ages, hazards = self.get_hazard(y, haz_type)
            
            if ages is None:
                return (np.nan, np.nan)
            
            if age_range is not None:
                mask = (ages >= age_range[0]) & (ages <= age_range[1])
                ages = ages[mask]
                hazards = hazards[mask]
            
            if len(hazards) == 0:
                return (np.nan, np.nan)
            
            # Filter out invalid values
            valid_mask = ~np.isnan(hazards) & (hazards > 0)
            if not np.any(valid_mask):
                return (np.nan, np.nan)
            
            ages_valid = ages[valid_mask]
            hazards_valid = hazards[valid_mask]
            
            min_idx = np.argmin(hazards_valid)
            return (ages_valid[min_idx], hazards_valid[min_idx])
        
        if np.isscalar(year):
            return get_single_year(year)
        else:
            years_filtered = self._filter_available_years(year)
            return {y: get_single_year(y) for y in years_filtered}

    # ======================================================================================
    # SECTION 3: DERIVED METRICS & ANALYTICS
    # ======================================================================================
    
    def _find_age_at_survival(self, ages, survival, prob):
        """Helper to find the age at which survival probability equals `prob`."""
        # Ensure survival is monotonically decreasing
        survival = pd.Series(survival).cummin().values
        # Interpolate, swapping x and y
        interp_func = interpolate.interp1d(survival, ages, bounds_error=False, fill_value=np.nan)
        return float(interp_func(prob))

    def calculate_lifespan_quantiles(self, year, quantiles=[0.75, 0.5, 0.25], age_start=0):
        """
        Calculates ages at specified survival quantiles.

        Parameters
        ----------
        year : int
            The year to analyze.
        quantiles : list, default [0.75, 0.5, 0.25]
            The survival probabilities for which to find the corresponding ages.
        age_start : int, default 0
            The age to condition the survival on.

        Returns
        -------
        dict
            A dictionary with quantiles as keys and corresponding ages as values.
        """
        ages, survival = self.get_survival(year)
        
        if ages is None:
            return {q: np.nan for q in quantiles}
        
        # Condition the survival curve on surviving to age_start
        start_idx = np.abs(ages - age_start).argmin()
        ages_cond = ages[start_idx:] - age_start
        survival_cond = survival[start_idx:] / survival[start_idx]
        
        results = {q: self._find_age_at_survival(ages_cond, survival_cond, q) for q in quantiles}
        
        # Add the start age back to get absolute age
        for q in results:
            if not np.isnan(results[q]):
                results[q] += age_start
                
        return results

    def calculate_median_lifespan(self, year, age_start=0):
        """
        Calculates the median lifespan (age at 50% survival), optionally conditioned on a starting age.
        """
        return self.calculate_lifespan_quantiles(year, [0.5], age_start)[0.5]
    
    def calculate_steepness(self, year, age_start=0, method='iqr', n_samples=10000):
        """
        Calculates the steepness of the survival curve.
        
        Steepness can be calculated in two ways:
        1. 'IQR' (default): median_lifespan / IQR, where IQR is the interquartile range 
           of lifespans (age at 25% survival - age at 75% survival).
        2. 'CV': The inverse of the coefficient of variation (mean/std) of lifespans 
           sampled from the mortality distribution.

        Parameters
        ----------
        year : int or list
            The year(s) to calculate steepness for.
        age_start : int, default 0
            The age to condition the survival on.
        method : str, default 'IQR'
            The method to use for calculation ('IQR' or 'CV').
        n_samples : int, default 10000
            Number of samples to use if method is 'CV'.
            
        Returns
        -------
        float or dict
            A single steepness value if one year is passed, or a dictionary of {year: steepness}.
        """
        def single_year_steepness(y):
            if method == 'iqr':
                quantile_ages = self.calculate_lifespan_quantiles(y, [0.75, 0.5, 0.25], age_start)
                t25, t50, t75 = quantile_ages[0.25], quantile_ages[0.5], quantile_ages[0.75]
            
                if any(pd.isna([t25, t50, t75])):
                    return np.nan
            
                iqr = t25 - t75
                return t50 / iqr if iqr > 0 else np.nan
            elif method == 'cv':
                lifespans = self.sample_lifespans(y, n_samples=n_samples, age_start=age_start)
                if len(lifespans) > 0:
                    mean_lifespan = np.mean(lifespans)
                    std_lifespan = np.std(lifespans)
                    if mean_lifespan > 0 and std_lifespan > 0:
                        cv = std_lifespan / mean_lifespan
                        return 1 / cv
                return np.nan
            else:
                raise ValueError("Invalid method for steepness calculation. Use 'IQR' or 'CV'.")

        if np.isscalar(year):
            return single_year_steepness(year)
        else:
            years_filtered = self._filter_available_years(year)
            return {y: single_year_steepness(y) for y in years_filtered}

    def fit_gompertz(self, year, age_start=50, age_end=80):
        """
        Fits an exponential function A * exp(B * t) to the hazard rate (Gompertz law).
        
        Returns
        -------
        tuple
            (A, B) coefficients of the fit.
        """
        ages, hazards = self.get_hazard(year)
        if ages is None:
            return np.nan, np.nan
        mask = (ages >= age_start) & (ages <= age_end) & (hazards > 0)
        
        if mask.sum() < 2:
            return np.nan, np.nan
            
        log_hazards = np.log(hazards[mask])
        B, log_A = np.polyfit(ages[mask], log_hazards, 1)
        return np.exp(log_A), B

    def _ggm_hazard_model(self, t, a, b, c, m):
        """Gamma-Gompertz-Makeham hazard function."""
        exp_bt = np.exp(b * t)
        exp_c = np.exp(c)
        return m + a * exp_bt * (exp_c / (exp_c + exp_bt - 1))

    def fit_ggm(self, year, age_start=20, age_end=100, p0=[5e-5, 0.1, 9, 0.005], return_cov=False):
        """
        Fits the Gamma-Gompertz-Makeham (GGM) model to the hazard rate.

        Returns
        -------
        dict
            A dictionary containing the fitted parameters 'a', 'b', 'c', 'm'.
        """
        ages, hazards = self.get_hazard(year)
        if ages is None:
            nan_dict = {param: np.nan for param in ['a', 'b', 'c', 'm']}
            return (nan_dict, nan_dict) if return_cov else nan_dict
        mask = (ages >= age_start) & (ages <= age_end) & (hazards > 0)
        
        if mask.sum() < 4: # Need at least 4 points for 4 parameters
            return {param: np.nan for param in ['a', 'b', 'c', 'm']}

        ages_fit, hazards_fit = ages[mask], hazards[mask]
        
        def log_hazard_model(t, a, b, c, m):
            with np.errstate(all='ignore'):
                return np.log(self._ggm_hazard_model(t, a, b, c, m))

        try:
            popt, pcov = curve_fit(log_hazard_model, ages_fit, np.log(hazards_fit), p0=p0, absolute_sigma=False)
            params = dict(zip(['a', 'b', 'c', 'm'], popt))
            if return_cov:
                perr = np.sqrt(np.diag(pcov))
                param_stderr = dict(zip(['a', 'b', 'c', 'm'], perr))
                return params, param_stderr
            else:
                return params
        except RuntimeError:
            return {param: np.nan for param in ['a', 'b', 'c', 'm']}
    
    def get_makeham_term(self, years):
        """
        Calculates the Makeham term (age-independent mortality) by fitting a GGM model.
        
        Returns
        -------
        float or dict
            The Makeham term 'm' for a single year, or a dict {year: m} for multiple years.
        """
        if np.isscalar(years):
            return self.fit_ggm(years)['m']
        else:
            years_filtered = self._filter_available_years(years)
            return {year: self.fit_ggm(year)['m'] for year in years_filtered}

    # ======================================================================================
    # SECTION 4: SAMPLING METHODS
    # ======================================================================================
    
    def sample_lifespans(self, year, n_samples, age_start=0):
        """
        Samples lifespans from the survival distribution for a given year.
        
        Parameters
        ----------
        year : int
            The year for which to sample.
        n_samples : int
            The number of lifespan samples to generate.
        age_start : int, default 0
            The age from which to start sampling (conditional lifespans).
            
        Returns
        -------
        np.ndarray
            An array of sampled ages at death, or empty array if year not available.
        """
        ages, cdf = self.get_lifespan_distribution(year, age_start=age_start)
        if ages is None:
            return np.array([])
        cdf = np.cumsum(cdf)
        
        u_samples = np.random.uniform(0, 1, n_samples)
        
        # Inverse transform sampling using interpolation
        interp_func = interpolate.interp1d(cdf, ages, kind='nearest', bounds_error=False, fill_value=(ages[0], ages[-1]))
        return interp_func(u_samples)
    '''
    def sample_death_times(self, n, year):
        """
        Sample n lifespans from the survival distribution for the specified year.

        Args:
            n (int): Number of samples to generate.
            year (int): Year for which to sample lifespans.

        Returns:
            numpy.ndarray: Array of sampled lifespans (ages at death).
        """
        ages, survival = self.get_survival(year)
        # Compute probability mass function (pmf) for each age
        pmf = -np.diff(survival)
        pmf = np.append(pmf, survival[-1])
        # Normalize to handle any floating-point inaccuracies
        pmf /= pmf.sum()
        # Sample ages according to the pmf
        sampled_ages = np.random.choice(ages, size=n, p=pmf)
        return sampled_ages

    def sample_death_time_from_age_X(self, year, age_X, n_samples=10000):
        import numpy as np

        # 1. Retrieve the survival curve for ages >= age_X
        ages, survival = self.get_survival(year)
        start_idx = np.abs(ages - age_X).argmin()
        ages_slice = ages[start_idx:]
        survival_slice = survival[start_idx:]

        # 2. Construct the CDF from the survival function, F(a) = 1 - S(a)
        cdf_slice = 1 - survival_slice

        # Re-base so that the CDF starts at 0
        cdf_slice -= cdf_slice[0]

        # Normalize the CDF to ensure it ends at 1
        if cdf_slice[-1] <= 0:
            raise ValueError("Invalid survival function: CDF does not progress appropriately.")
        cdf_slice /= cdf_slice[-1]

        # Ensure monotonicity to avoid flat regions from numerical precision issues
        cdf_slice = np.maximum.accumulate(cdf_slice)

        # 3. Inverse transform sampling: sample uniform random values and invert the CDF
        u_samples = np.random.uniform(0, 1, n_samples)
        death_times = np.interp(u_samples, cdf_slice, ages_slice)

        return death_times
    
    def sample_death_time_from_uniform(self, year, age_X, U_array):
        """
        Given an array of U ~ Uniform(0,1), return age-at-death samples
        consistent with the survival distribution starting at age_X.
        Essentially, an approximate 'inverse CDF' step.

        Parameters
        ----------
        year : int
            The year for which to calculate the survival distribution.
        age_X : float
            The 'current' age from which we measure future lifetimes.
        U_array : np.ndarray
            1D array of Uniform(0,1) values, shape (n_samples,).

        Returns
        -------
        death_ages : np.ndarray
            The corresponding age-at-death for each U in U_array.
        """

        # --- 1) Build discrete CDF for ages >= age_X ---
        ages, survival = self.get_survival(year)
        start_idx = np.abs(ages - age_X).argmin()
        ages_slice = ages[start_idx:]
        survival_slice = survival[start_idx:]

        # Build cdf_slice = 1 - survival, re-base & re-scale so it goes from 0 to 1
        cdf_slice = 1 - survival_slice
        cdf_slice -= cdf_slice[0]
        if cdf_slice[-1] > 0:
            cdf_slice /= cdf_slice[-1]

        # ages_slice is sorted ascending, cdf_slice is also sorted ascending
        # We can invert by a search / interpolation.

        # The simplest approach: np.searchsorted.
        # For each U, we find the smallest index where cdf_slice[idx] >= U.
        indices = np.searchsorted(cdf_slice, U_array, side='right')

        # Because searchsorted might give an index = len(ages_slice) if U=1.0,
        # we clamp the indices to the valid range:
        indices = np.clip(indices, 0, len(ages_slice) - 1)

        death_ages = ages_slice[indices]

        return death_ages
    '''
    # ======================================================================================
    # SECTION 5: PLOTTING
    # ======================================================================================
    
    def _get_color_gradient(self, n_colors, cmap="coolwarm"):
        """Helper to create a color gradient for plots."""
        if n_colors == 1:
            return ['blue']
        return plt.get_cmap(cmap)(np.linspace(0, 1, n_colors))

    def plot_survival(self, years, ax=None, scaled=False, custom_colors=None, cmap="coolwarm", custom_labels=None, **kwargs):
        """
        Plots survival curves for one or more years.
        
        Parameters
        ----------
        years : int or list
            The year(s) to plot.
        ax : matplotlib.axes.Axes, optional
            The axes to plot on.
        scaled : bool, default False
            If True, scales the age axis by the median lifespan for each year.
        custom_colors : str or list, optional
            Custom colors to use for plotting. If years is a single int, can be a single color string.
            If years is a list, must be a list of color strings with same length as years.
        cmap : str, default "coolwarm"
            Colormap to use when generating color gradient if custom_colors is None.
        custom_labels : str or list, optional
            Custom labels to use for the legend. If years is a single int, can be a single label string.
            If years is a list, must be a list of label strings with same length as years.
        """
        if ax is None:
            _, ax = plt.subplots()
        
        years = self._filter_available_years(years)
        if len(years) == 0:
            raise ValueError("No data available for any of the requested years.")
        
        if custom_colors is not None:
            if np.isscalar(custom_colors):
                colors = [custom_colors] if len(years) == 1 else [custom_colors] * len(years)
            else:
                if len(custom_colors) != len(years):
                    raise ValueError(f"custom_colors length ({len(custom_colors)}) must match years length ({len(years)})")
                colors = custom_colors
        else:
            colors = self._get_color_gradient(len(years), cmap=cmap)
        
        if custom_labels is not None:
            if np.isscalar(custom_labels):
                labels = [custom_labels] if len(years) == 1 else [custom_labels] * len(years)
            else:
                if len(custom_labels) != len(years):
                    raise ValueError(f"custom_labels length ({len(custom_labels)}) must match years length ({len(years)})")
                labels = custom_labels
        else:
            labels = [str(year) for year in years]
        
        for i, year in enumerate(years):
            ages, survival = self.get_survival(year)
            if ages is None:
                continue
            x_axis = ages
            label = labels[i]
            
            if scaled:
                median_t = self.calculate_median_lifespan(year)
                if not pd.isna(median_t):
                    x_axis = ages / median_t
                else:
                    x_axis = np.full_like(ages, np.nan) # Cannot scale
            
            ax.plot(x_axis, survival, color=colors[i], label=label, **kwargs)

        ax.set_xlabel('Age' if not scaled else 'Age / Median Lifespan')
        ax.set_ylabel('Survival Probability')
        ax.set_title(f'Survival Curve for {self.country.capitalize()}, {self.gender.capitalize()}')
        if len(years) > 1:
            ax.legend(title='Year')
        return ax

    def plot_conditional_survival(self, age_X, years, ax=None, custom_colors=None, cmap="coolwarm", **kwargs):
        """
        Plots survival curves conditioned on surviving to `age_X`.
        """
        if ax is None:
            _, ax = plt.subplots()

        years = self._filter_available_years(years)
        if len(years) == 0:
            raise ValueError("No data available for any of the requested years.")

        if custom_colors is not None:
            if np.isscalar(custom_colors):
                colors = [custom_colors] if len(years) == 1 else [custom_colors] * len(years)
            else:
                if len(custom_colors) != len(years):
                    raise ValueError(f"custom_colors length ({len(custom_colors)}) must match years length ({len(years)})")
                colors = custom_colors
        else:
            # Special handling for 'Greys' colormap: avoid white and darkest black
            if cmap.lower() == "greys":
                import matplotlib as mpl
                cmap_obj = mpl.cm.get_cmap(cmap)
                n = len(years)
                if n == 1:
                    color_vals = [0.5]
                else:
                    # Avoid first 10% and last 10% to skip pure white and pure black
                    color_vals = np.linspace(0.1, 0.9, n)
                colors = [cmap_obj(val) for val in color_vals]
            else:
                colors = self._get_color_gradient(len(years), cmap=cmap)

        for i, year in enumerate(years):
            ages, survival = self.get_survival(year)
            if ages is None:
                continue
            start_idx = np.abs(ages - age_X).argmin()

            ages_cond = ages[start_idx:]
            survival_cond = survival[start_idx:] / survival[start_idx]

            ax.plot(ages_cond, survival_cond, color=colors[i], label=str(year), **kwargs)

        ax.set_xlabel(f'Age (starting from {age_X})')
        ax.set_ylabel('Conditional Survival Probability')
        ax.set_title(f'Conditional Survival from Age {age_X} for {self.country.capitalize()}')
        if len(years) > 1:
            ax.legend(title='Year')
        return ax

    def plot_hazard(self, years, ax=None, haz_type='mx', smooth_window=None, custom_colors=None, cmap="coolwarm", **kwargs):
        """
        Plots hazard rates for one or more years.
        """
        if ax is None:
            _, ax = plt.subplots()

        years = self._filter_available_years(years)
        if len(years) == 0:
            raise ValueError("No data available for any of the requested years.")
        
        if custom_colors is not None:
            if np.isscalar(custom_colors):
                colors = [custom_colors] if len(years) == 1 else [custom_colors] * len(years)
            else:
                if len(custom_colors) != len(years):
                    raise ValueError(f"custom_colors length ({len(custom_colors)}) must match years length ({len(years)})")
                colors = custom_colors
        else:
            # Special handling for 'Greys' colormap: avoid the very lightest and darkest ends
            if cmap.lower() == "greys":
                import matplotlib as mpl
                cmap_obj = mpl.cm.get_cmap(cmap)
                # Avoid the first 10% and last 10% of the colormap
                n = len(years)
                if n == 1:
                    color_vals = [0.5]
                else:
                    color_vals = np.linspace(0.1, 0.9, n)
                colors = [cmap_obj(val) for val in color_vals]
            else:
                colors = self._get_color_gradient(len(years), cmap=cmap)
        
        for i, year in enumerate(years):
            ages, hazard = self.get_hazard(year, haz_type)
            if ages is None:
                continue
            
            if smooth_window:
                hazard = pd.Series(hazard).rolling(window=smooth_window, center=True).mean()

            ax.plot(ages, hazard, color=colors[i], label=str(year), **kwargs)

        ax.set_yscale('log')
        ax.set_xlabel('Age [years]')
        ax.set_ylabel(f'Hazard Rate ({haz_type}) [1/year]')
        ax.set_title(f'Hazard Rate for {self.country.capitalize()}, {self.gender.capitalize()}')
        if len(years) > 1:
            ax.legend(title='Year')
        return ax
    def plot_normalized_hazard(self, years, ax=None, haz_type='mx', age_max=None,
                                  smooth_window=None, custom_colors=None, cmap="coolwarm",
                                  colorbar=True, **kwargs):
        """
        Plots hazard rates normalized by the hazard at age 0, so all curves start at 1.

        Parameters
        ----------
        years : array-like
            Years to plot.
        ax : matplotlib Axes, optional
        haz_type : str, default 'mx'
            Hazard column to use ('mx' or 'qx').
        age_max : int, optional
            If provided, only plot ages up to this value (inclusive).
        smooth_window : int, optional
            Rolling-average window for smoothing.
        custom_colors : optional
            Single color or list of colors.
        cmap : str, default 'coolwarm'
            Colormap name when custom_colors is not provided.
        colorbar : bool, default True
            If True, show a colorbar mapping colors to years instead of a legend.
        **kwargs
            Forwarded to ax.plot().

        Returns
        -------
        matplotlib Axes
        """
        import matplotlib as mpl

        if ax is None:
            _, ax = plt.subplots()

        years = self._filter_available_years(years)
        if len(years) == 0:
            raise ValueError("No data available for any of the requested years.")

        cmap_obj = mpl.cm.get_cmap(cmap)
        norm = mpl.colors.Normalize(vmin=float(years.min()), vmax=float(years.max()))

        if custom_colors is not None:
            if np.isscalar(custom_colors):
                colors = [custom_colors] if len(years) == 1 else [custom_colors] * len(years)
            else:
                if len(custom_colors) != len(years):
                    raise ValueError(f"custom_colors length ({len(custom_colors)}) must match years length ({len(years)})")
                colors = custom_colors
        else:
            colors = [cmap_obj(norm(yr)) for yr in years]

        for i, year in enumerate(years):
            ages, hazard = self.get_hazard(year, haz_type)
            if ages is None:
                continue

            h0 = hazard[0]
            if h0 == 0 or np.isnan(h0):
                continue
            hazard = hazard / h0

            if age_max is not None:
                mask = ages <= age_max
                ages = ages[mask]
                hazard = hazard[mask]

            if smooth_window:
                hazard = pd.Series(hazard).rolling(window=smooth_window, center=True).mean()

            ax.plot(ages, hazard, color=colors[i], label=str(year), **kwargs)

        ax.set_yscale('log')
        ax.set_xlabel('Age [years]')
        ax.set_ylabel(f'Normalized Hazard h(t)/h(0)')
        ax.set_title(f'Normalized Hazard for {self.country.capitalize()}, {self.gender.capitalize()}')

        if colorbar and len(years) > 1 and custom_colors is None:
            sm = mpl.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.02)
            year_min, year_max = int(years.min()), int(years.max())
            first_tick = ((year_min // 50) + (1 if year_min % 50 != 0 else 0)) * 50
            tick_positions = np.arange(first_tick, year_max + 1, 50)
            tick_positions = tick_positions[(tick_positions >= year_min) & (tick_positions <= year_max)]
            cbar.set_ticks(tick_positions)
            cbar.set_label('Year')
        elif len(years) > 1:
            ax.legend(title='Year')

        return ax

    def plot_gompertz_fit(self, year, ax=None, age_start=50, age_end=80, **kwargs):
        """
        Plots the hazard rate for a year and overlays the fitted Gompertz curve.
        """
        if ax is None:
            _, ax = plt.subplots()
        
        # Plot the actual hazard data for the year
        self.plot_hazard(year, ax=ax)
                    
        A, B = self.fit_gompertz(year, age_start, age_end)
        
        if not (pd.isna(A) or pd.isna(B)):
            ages = np.arange(20, 100 + 1)
            hazard_fit = A * np.exp(B * ages)
            label = f'Gompertz Fit: A={A:.2e}, B={B:.3f}'
            ax.plot(ages, hazard_fit, '--', color='blue', label=label, **kwargs)
            
        ax.legend()
        return ax
        
    def plot_ggm_fit(self, year, ax=None, **kwargs):
        """
        Plots the hazard rate for a year and overlays the fitted GGM curve.
        """
        if ax is None:
            ax = plt.gca()
            
        self.plot_hazard(year, ax=ax, **kwargs)
        params = self.fit_ggm(year)
        
        if not any(pd.isna(list(params.values()))):
            ages = np.linspace(0, 110, 200)
            hazard_fit = self._ggm_hazard_model(ages, **params)
            label = f"GGM Fit (m={params['m']:.2e})"
            ax.plot(ages, hazard_fit, '--', color='red', label=label)

        ax.legend()
        return ax

    def plot_median_lifespan_trend(self, fig=None, **kwargs):
        """
        Plots the trend of median lifespan over the available years using Plotly.
        """
        years = self.data['Year'].unique()
        median_lifespans = [self.calculate_median_lifespan(y) for y in years]
        
        df = pd.DataFrame({'Year': years, 'Median Lifespan': median_lifespans}).dropna()

        if fig is None:
            fig = px.line(df, x='Year', y='Median Lifespan', markers=True,
                          title=f'Median Lifespan Over Time for {self.country.capitalize()}, {self.gender.capitalize()}',
                          labels={'Year': 'Year', 'Median Lifespan': 'Median Lifespan [years]'})
            fig.update_layout(width=700, height=500)
        else:
            fig.add_trace(go.Scatter(x=df['Year'], y=df['Median Lifespan'], mode='lines+markers',
                                    name=f'{self.country.capitalize()} {self.gender.capitalize()}', **kwargs))
        return fig
        
    def plot_steepness_vs_longevity(self, years=None, ax=None, age_start=0, **kwargs):
        """
        Plots the relationship between steepness and median lifespan over a range of years.
        
        Parameters
        ----------
        years : list or None
            A list of years to plot. If None, all available years are used.
        ax : matplotlib.axes.Axes, optional
            The axes to plot on.
        age_start : int, default 0
            The age to condition the metrics on.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
            
        if years is None:
            years = sorted(self.data['Year'].unique())
        else:
            years = self._filter_available_years(years)
            if len(years) == 0:
                raise ValueError("No data available for any of the requested years.")
            
        steepness_vals = self.calculate_steepness(years, age_start=age_start)
        longevity_vals = {y: self.calculate_median_lifespan(y, age_start=age_start) for y in years}
        
        df = pd.DataFrame.from_dict(steepness_vals, orient='index', columns=['Steepness'])
        df['Longevity'] = df.index.map(longevity_vals)
        df = df.dropna().sort_index()

        # Create a scatter plot with a color gradient for the year
        sc = ax.scatter(df['Longevity'], df['Steepness'], c=df.index, cmap='viridis', **kwargs)
        
        # Add a colorbar
        cbar = plt.colorbar(sc)
        cbar.set_label('Year')
        
        ax.set_xlabel(f'Median Lifespan (from age {age_start}) [years]')
        ax.set_ylabel(f'Steepness (Median/IQR)')
        ax.set_title(f'Steepness vs. Longevity for {self.country.capitalize()}, {self.gender.capitalize()}')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    # =========================
    # HAZARD AVERAGING & PLOTTING (CONSOLIDATED)
    # =========================
    """
    This section contains functions for analyzing and visualizing hazard rates:
    
    1. summarize_log_hazard(): Computes mean/std of log hazard for a single year and age range
       Example: mean_log_h, std_log_h = hmd.summarize_log_hazard(2010, (30, 50))
    
    2. summarize_log_hazard_over_time(): Extends above across multiple years, with optional grouping
       Example: years, means, stds = hmd.summarize_log_hazard_over_time(1950, 2020, interval=10)
    
    3. plot_hazard_trend(): Visualizes hazard trends over time with error bars and color coding
       Example: ax = hmd.plot_hazard_trend(1950, 2020, interval=5, age_range=(20, 40))
    
    4. log_average_hazard_curve(): Computes geometric mean hazard curve across years for each age
       Example: ages, mean_h, lower_h, upper_h = hmd.log_average_hazard_curve([2010, 2015, 2020])
    
    5. plot_log_average_hazard_curve(): Plots the geometric mean hazard curve with confidence bands
       Example: ax = hmd.plot_log_average_hazard_curve([2010, 2015, 2020], show_individual=True)
    """

    def summarize_log_hazard(self, year, age_range=(20, 40), haz_type='mx'):
        """
        Compute mean and std of log hazard rate for a single year and age range.
        Parameters
        ----------
        year : int
            Year to analyze
        age_range : tuple, default (20, 40)
            (start_age, end_age) for averaging
        haz_type : str, default 'mx'
            Hazard type
        Returns
        -------
        tuple: (mean_log_hazard, std_log_hazard)
        """
        ages, hazards = self.get_hazard(year, haz_type)
        if ages is None:
            return np.nan, np.nan
        mask = (ages >= age_range[0]) & (ages <= age_range[1])
        log_hazards = np.log(hazards[mask])
        return np.mean(log_hazards), np.std(log_hazards)

    def summarize_log_hazard_over_time(self, year_start, year_end, interval=None, age_range=(20, 40), haz_type='mx'):
        """
        Compute mean log hazard rate over a range of years, optionally grouped by interval.
        Parameters
        ----------
        year_start, year_end : int
            Range of years
        interval : int or None
            If set, group years into intervals of this size
        age_range : tuple, default (20, 40)
            (start_age, end_age) for averaging
        haz_type : str, default 'mx'
        Returns
        -------
        years : np.ndarray
        mean_log_hazards : np.ndarray
        std_log_hazards : np.ndarray
        """
        if interval is None:
            years = self._filter_available_years(np.arange(year_start, year_end + 1))
            if len(years) == 0:
                raise ValueError("No data available for any of the requested years.")
            means, stds = zip(*[self.summarize_log_hazard(y, age_range, haz_type) for y in years])
            return years, np.array(means), np.array(stds)
        else:
            intervals = np.arange(year_start, year_end + 1, interval)
            means, stds, years = [], [], []
            for start in intervals:
                end = min(start + interval, year_end + 1)
                y_range = self._filter_available_years(np.arange(start, end))
                if len(y_range) == 0:
                    continue  # Skip intervals with no available data
                m, s = zip(*[self.summarize_log_hazard(y, age_range, haz_type) for y in y_range])
                means.append(np.mean(m))
                stds.append(np.sqrt(np.sum(np.array(s) ** 2)) / len(m))
                years.append(start)
            return np.array(years), np.array(means), np.array(stds)

    def plot_hazard_trend(self, year_start, year_end, interval=5, age_range=(20, 40), ax=None, haz_type='mx', error_bars=True, cmap='RdBu', **kwargs):
        """
        Plot average hazard rate over time intervals with error bars and colormap.
        Parameters
        ----------
        year_start, year_end : int
        interval : int
        age_range : tuple
        ax : matplotlib.axes.Axes, optional
        haz_type : str
        error_bars : bool
        cmap : str
        **kwargs : passed to scatter
        """
        years, means, stds = self.summarize_log_hazard_over_time(year_start, year_end, interval, age_range, haz_type)
        h_avg = np.exp(means)
        yerr_lower = h_avg - np.exp(means - stds)
        yerr_upper = np.exp(means + stds) - h_avg
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Check if color is specified in kwargs - if so, use single color instead of colormap
        if 'color' in kwargs or 'c' in kwargs:
            scatter = ax.scatter(years, h_avg, s=100, **kwargs)
            show_colorbar = False
        else:
            scatter = ax.scatter(years, h_avg, c=years, cmap=cmap, s=100, **kwargs)
            show_colorbar = True
            
        if error_bars:
            ax.errorbar(years, h_avg, yerr=[yerr_lower, yerr_upper], fmt='none', capsize=3, ecolor='gray', alpha=0.7)
        
        if show_colorbar:
            plt.colorbar(scatter, ax=ax, label='Year')
            
        ax.set_yscale('log')
        ax.set_xlabel('Year', fontname='Arial')
        ax.set_ylabel(r'$h_{\mathrm{avg}}$ [1/year]', fontname='Arial')
        ax.set_title(f'Average Hazard Rate Over Time (Ages {age_range[0]}-{age_range[1]}), {interval}-Year Intervals', fontname='Arial')
        ax.grid(True, which="both", ls="-", alpha=0.2)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Arial')
        return ax

    def log_average_hazard_curve(self, years, haz_type='mx', max_age=103):
        """
        Compute geometric mean hazard curve (across years) for each age.
        Parameters
        ----------
        years : list or array
        haz_type : str
        max_age : int
        Returns
        -------
        ages : np.ndarray
        mean_hazard : np.ndarray
        lower_hazard : np.ndarray
        upper_hazard : np.ndarray
        """
        all_log_hazards = []
        ages = None
        years = self._filter_available_years(years)
        if len(years) == 0:
            raise ValueError("No data available for any of the requested years.")
        for year in years:
            year_ages, hazards = self.get_hazard(year, haz_type)
            if year_ages is None:
                continue
            if ages is None:
                ages_mask = year_ages <= max_age
                ages = year_ages[ages_mask]
            indices = np.where(np.isin(year_ages, ages))[0]
            hazards_filtered = hazards[indices]
            with np.errstate(divide='ignore', invalid='ignore'):
                log_hazards = np.log(hazards_filtered)
            all_log_hazards.append(log_hazards)
        all_log_hazards = np.array(all_log_hazards)
        all_log_hazards[np.isneginf(all_log_hazards)] = np.nan
        mean_log_hazard = np.nanmean(all_log_hazards, axis=0)
        std_log_hazard = np.nanstd(all_log_hazards, axis=0)
        mean_hazard = np.exp(mean_log_hazard)
        upper_hazard = np.exp(mean_log_hazard + std_log_hazard)
        lower_hazard = np.exp(mean_log_hazard - std_log_hazard)
        return ages, mean_hazard, lower_hazard, upper_hazard

    def plot_log_average_hazard_curve(self, years, ax=None, haz_type='mx', max_age=103, show_individual=False, cmap='gray', **kwargs):
        """
        Plot geometric mean hazard curve (across years) for each age, with error bands.
        Parameters
        ----------
        years : list or array
        ax : matplotlib.axes.Axes, optional
        haz_type : str
        max_age : int
        show_individual : bool
        cmap : str
        **kwargs : passed to plot
        """
        if ax is None:
            ax = plt.gca()
        years = self._filter_available_years(years)
        if len(years) == 0:
            raise ValueError("No data available for any of the requested years.")
        ages, mean_hazard, lower_hazard, upper_hazard = self.log_average_hazard_curve(years, haz_type, max_age)
        if show_individual:
            for year in np.atleast_1d(years):
                year_ages, hazards = self.get_hazard(year, haz_type)
                if year_ages is None:
                    continue
                mask = year_ages <= max_age
                ax.plot(year_ages[mask], hazards[mask], color='gray', alpha=0.3)
        valid = ~np.isnan(mean_hazard) & ~np.isnan(ages)
        ax.plot(ages[valid], mean_hazard[valid], 'b-', label='Geometric Mean', **kwargs)
        ax.fill_between(ages[valid], lower_hazard[valid], upper_hazard[valid], alpha=0.2, color='blue', label='_nolegend_')
        ax.set_yscale('log')
        ax.set_xlabel('Age [years]', fontsize=20, fontname='Arial')
        ax.set_ylabel('Hazard [1/year]', fontsize=20, fontname='Arial')
        ax.legend()
        return ax

    # =========================
    # GGM FITTING (WITH CONFIDENCE INTERVALS)
    # =========================
    def fit_ggm(self, year, age_start=20, age_end=97, p0=[5e-5, 0.1, 9, 0.005], return_cov=False):
        """
        Fit Gamma-Gompertz-Makeham (GGM) model to hazard rate for a given year.
        Returns fitted parameters and (optionally) their standard errors.
        Parameters
        ----------
        year : int
        age_start, age_end : int
        p0 : list
        return_cov : bool, default False
            If True, also return parameter standard errors (from covariance matrix)
        Returns
        -------
        params : dict
            Fitted parameters 'a', 'b', 'c', 'm'
        param_stderr : dict (if return_cov)
            Standard errors for each parameter
        """
        ages, hazards = self.get_hazard(year)
        if ages is None:
            nan_dict = {k: np.nan for k in ['a', 'b', 'c', 'm']}
            return (nan_dict, nan_dict) if return_cov else nan_dict
        mask = (ages >= age_start) & (ages <= age_end) & (hazards > 0)
        if mask.sum() < 4:
            nan_dict = {k: np.nan for k in ['a', 'b', 'c', 'm']}
            if return_cov:
                return nan_dict, nan_dict
            else:
                return nan_dict
        ages_fit, hazards_fit = ages[mask], hazards[mask]
        def log_hazard_model(t, a, b, c, m):
            with np.errstate(all='ignore'):
                return np.log(self._ggm_hazard_model(t, a, b, c, m))
        try:
            popt, pcov = curve_fit(log_hazard_model, ages_fit, np.log(hazards_fit), p0=p0, absolute_sigma=False)
            params = dict(zip(['a', 'b', 'c', 'm'], popt))
            if return_cov:
                perr = np.sqrt(np.diag(pcov))
                param_stderr = dict(zip(['a', 'b', 'c', 'm'], perr))
                return params, param_stderr
            else:
                return params
        except Exception:
            nan_dict = {k: np.nan for k in ['a', 'b', 'c', 'm']}
            if return_cov:
                return nan_dict, nan_dict
            else:
                return nan_dict

    def plot_makeham_term_trend(self, years, age_start=20, age_end=105, ax=None, color=None, cmap='plasma', **kwargs):
        """
        Plot the Makeham term (extrinsic mortality, 'm' from GGM fit) vs years, with error bars from fit confidence intervals.

        Parameters
        ----------
        years : array-like
            Years to plot.
        age_start, age_end : int, default 20, 97
            Age range for GGM fitting.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        color : str, optional
            Single color for all scatter points. If provided, overrides colormap.
        cmap : str, default 'plasma'
            Colormap for scatter plot (used only if color is None).
        **kwargs :
            Additional arguments passed to scatter.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        years = self._filter_available_years(years)
        if len(years) == 0:
            raise ValueError("No data available for any of the requested years.")
        m_vals = []
        m_errs = []
        valid_years = []
        for y in years:
            params, perr = self.fit_ggm(y, age_start=age_start, age_end=age_end, return_cov=True)
            m = params['m']
            m_err = perr['m']
            if not (np.isnan(m) or np.isnan(m_err)):
                m_vals.append(m)
                m_errs.append(m_err)
                valid_years.append(y)
        m_vals = np.array(m_vals)
        m_errs = np.array(m_errs)
        valid_years = np.array(valid_years)
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        
        # Set default marker if not provided in kwargs
        if 'marker' not in kwargs:
            kwargs['marker'] = '^'
        
        if color is not None:
            # Use single color for all points
            sc = ax.scatter(valid_years, m_vals, color=color, s=40, **kwargs)
        else:
            # Use colormap
            sc = ax.scatter(valid_years, m_vals, c=valid_years, cmap=cmap, s=40, **kwargs)
            cbar = plt.colorbar(sc, ax=ax, label='Year')
            
            # Set colorbar ticks every 50 years within the range of valid_years
            if len(valid_years) > 0:
                year_min = int(np.min(valid_years))
                year_max = int(np.max(valid_years))
                # Find the first tick that's a multiple of 50 and >= year_min
                first_tick = ((year_min // 50) + (1 if year_min % 50 != 0 else 0)) * 50
                # Generate ticks every 50 years from first_tick to year_max
                tick_positions = np.arange(first_tick, year_max + 1, 50)
                # Only keep ticks that are within the actual data range
                tick_positions = tick_positions[(tick_positions >= year_min) & (tick_positions <= year_max)]
                cbar.set_ticks(tick_positions)
            
            cbar.ax.tick_params(labelsize=18)  # Set fontsize for colorbar ticks
            cbar.set_label('Year', size=18)  # Set fontsize for colorbar label
        
        #ax.errorbar(valid_years, m_vals, yerr=m_errs, fmt='none', ecolor='gray', capsize=3, alpha=0.7)
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel('Extrinsic Mortality [1/years]', fontsize=14)
        ax.set_title(f'Extrinsic Mortality over Time', fontsize=15)
        ax.set_yscale('log')
        ax.grid(True, which='both', ls='--', alpha=0.3)
        return ax

    def plot_sm(self, years, ax=None, colorbar=True, cmap='viridis', **kwargs):
        """
        Plots ln(a) vs b from Gompertz fits across multiple years with a linear fit.
        
        Parameters
        ----------
        years : list or array
            Years to analyze.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        colorbar : bool, default True
            If True, scatter points are colored by year and a colorbar is shown.
        cmap : str, default 'viridis'
            Colormap for scatter points if colorbar is True.
        **kwargs :
            Additional arguments passed to scatter.
            
        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        years = self._filter_available_years(years)
        if len(years) == 0:
            raise ValueError("No data available for any of the requested years.")
        
        b_vals = []
        ln_a_vals = []
        scatter_years = []
        
        # Fit Gompertz for each year and extract parameters
        for year in years:
            a, b = self.fit_gompertz(year)  # fit_gompertz returns (A, B)
            if not (np.isnan(a) or np.isnan(b) or a <= 0):
                b_vals.append(b)
                ln_a_vals.append(np.log(a))
                scatter_years.append(year)
        
        b_vals = np.array(b_vals)
        ln_a_vals = np.array(ln_a_vals)
        scatter_years = np.array(scatter_years)
        
        if len(b_vals) == 0:
            raise ValueError("No valid Gompertz fits found for the given years.")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Scatter plot
        if colorbar:
            sc = ax.scatter(b_vals, ln_a_vals, c=scatter_years, cmap=cmap, s=50, alpha=0.7, **kwargs)
            cbar = plt.colorbar(sc, ax=ax, label='Year', pad=0.02)
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label('Year', size=14, fontname='Arial')
        else:
            ax.scatter(b_vals, ln_a_vals, s=50, alpha=0.7, **kwargs)
        
        # Linear fit
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(b_vals, ln_a_vals)
        
        # Plot fit line
        b_fit = np.linspace(b_vals.min(), b_vals.max(), 100)
        ln_a_fit = slope * b_fit + intercept
        ax.plot(b_fit, ln_a_fit, 'r--', linewidth=2, label='Linear Fit')
        
        # Add equation to plot
        equation_text = f'ln(a) = {slope:.2f}·b + {intercept:.2f}\nR² = {r_value**2:.3f}'
        ax.text(0.05, 0.95, equation_text, transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontname='Arial')
        
        ax.set_xlabel(r'Gompertz $\alpha$ [1/year]', fontname='Arial')
        ax.set_ylabel(r'$ln(h_0)$', fontname='Arial')
        ax.set_title(f'SM Plot for {self.country.capitalize()}, {self.gender.capitalize()}', fontname='Arial')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set Arial font for tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Arial')
        
        return ax

    def print_years(self):
        """
        Prints all the years for which data is available in this HMD object.
        """
        years = self.data['Year'].unique()
        print(f"Available years for {self.country} ({self.gender}): {sorted(years)}")
