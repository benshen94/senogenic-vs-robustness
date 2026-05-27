import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
from ageing_packages.mortality_data_analysis import HMD
from ageing_packages.hetero_analysis import twin_analysis as ta
from lifelines import NelsonAalenFitter, KaplanMeierFitter
from scipy.optimize import minimize
from scipy.stats import ks_2samp  # We'll write our own KS, but you can also use ks_2samp
import copy


class SM:
    def __init__(self, params=None):
        # if params is not None then set the parameters
        if params is not None:
            self.set_params(params)
                   
    def set_params(self, params):
        self.V0 = params['V0']
        self.lambda_ = params['lambda_']
        self.B = params['B']
        self.m = params.get('m', 0)  # Default m to 0 if not provided
        
    # fit params according to HMD hazard data
    def fit_params(self, country, year, gender, data_type='cohort', haz_type='mx', filter_from=20, filter_to=110, print_out=True):
        ages_data, hazard_data = self.load_HMD_hazard_data(country=country, year=year, gender=gender, data_type=data_type, haz_type=haz_type)
        
        # Filter data >= filter_from and remove NaN values while keeping indices aligned
        filter_mask = (ages_data >= filter_from) & (ages_data <= filter_to) & (~np.isnan(ages_data)) & (~np.isnan(hazard_data)) & (hazard_data > 0)
        ages_data_filtered = ages_data[filter_mask]
        hazard_data_filtered = hazard_data[filter_mask]
        
        # Fit all parameters together including m
        initial_guess = [0.7, 10, 0.01, 0.0004]  # Added initial guess for m
        popt, pcov = curve_fit(self.log_hazard_function_SM, ages_data_filtered, np.log(hazard_data_filtered), p0=initial_guess)
        
        # Extract the fitted parameters
        lambda_fit, V0_fit, B_fit, m_fit = popt
        
        self.lambda_ = lambda_fit
        self.V0 = V0_fit
        self.B = B_fit
        self.m = m_fit
        self.hmd_data = {'country': country, 'year': year, 'gender': gender, 'data_type': data_type, 'haz_type': haz_type}
        
        if print_out:
            self.print_params()
        
    # if you manually change the params, you can refit them to the HMD data
    def refit_params(self, print_out=True):
        self.fit_params(
            country=self.hmd_data['country'],
            year=self.hmd_data['year'], 
            gender=self.hmd_data['gender'],
            data_type=self.hmd_data['data_type'],
            haz_type=self.hmd_data['haz_type'],
            filter_from=20, 
            filter_to=110
        )
   
    def print_params(self):
        print('V0: ', self.V0)
        print('lambda: ', self.lambda_)
        print('B: ', self.B)
        print('m: ', self.m)
        
    def hazard_function_SM(self, t, lambda_, V0, B, m):
        # Ensure inputs are numpy arrays and properly shaped for broadcasting
        t = np.asarray(t)
        lambda_ = np.asarray(lambda_)
        V0 = np.asarray(V0)
        B = np.asarray(B)
        m = np.asarray(m)
        
        # Compute condition, will broadcast across all dimensions
        condition = t <= 1/B
        
        # Both expressions will broadcast properly since numpy handles broadcasting
        # of arithmetic operations between arrays of compatible shapes
        return np.where(condition,
                       lambda_ * np.exp(V0 * (B*t - 1)) + m,
                       lambda_ + m)
        
    def log_hazard_function_SM(self, t, lambda_, V0, B, m):
        condition = t <= 1/B
        return np.where(condition,
                       np.log(lambda_*np.exp(V0 * (B*t - 1)) + m),
                       np.log(lambda_ + m))
        
            
    def calculate_hazard_SM(self, t):
        ages = np.linspace(10, 120, 1000)
        return ages, self.hazard_function_SM(ages, self.lambda_, self.V0, self.B, self.m)
    
    def sample_death_times(self, n, dt=0.5, params=None):
        """
        Vectorized version to sample n death times from the hazard function
        using numerical integration and faster interpolation.

        Args:
            n (int): Number of samples to draw
            dt (float): Time step for numerical integration (default 0.5)
            params (dict): Optional dictionary of parameters to use instead of self params
            
        Returns:
            numpy array: Array of n death times sampled from the hazard distribution
        """
        # Use provided params if given, otherwise use self params
        if params is None:
            lambda_ = self.lambda_
            V0 = self.V0
            B = self.B
            m = self.m
        else:
            lambda_ = params['lambda_']
            V0 = params['V0']
            B = params['B']
            m = params['m']

        # Create time grid for numerical integration
        t = np.arange(0, 200, dt)

        # Calculate hazard at each time point
        hazard = self.hazard_function_SM(t, lambda_, V0, B, m)

        # Calculate cumulative hazard by numerical integration
        cumulative_hazard = np.cumsum(hazard * dt)

        # Generate n uniform random numbers
        u = np.random.random(n)

        # Convert to the "survival" threshold (−log(u)) that we compare against 
        survival_thresholds = -np.log(u)

        # We need arrays for interpolation:
        #   x-values = [0, ... cumulative_hazard]
        #   y-values = [t[0], ... t]
        # Inserting zeros at the start lets us handle threshold values < cumulative_hazard[0].
        ch_full = np.concatenate(([0.0], cumulative_hazard))
        t_full  = np.concatenate(([t[0]], t))

        # Interpolate survival_thresholds within ch_full to get times
        death_times = np.interp(survival_thresholds, ch_full, t_full, left=t[0], right=t[-1])
        
        return death_times

    def sample_death_times_with_random_param(self, n, param_name='V0', dt=0.5, std=0.15):
        """
        Sample death times with a randomly drawn parameter value for each individual.

        Args:
            n (int): Number of samples to draw
            param_name (str): Which parameter to randomize. One of {'V0','B','lambda_','m'}.
                            Default 'V0'.
            dt (float): Time step for numerical integration. Default 0.5.
            std (float): Standard deviation for parameter sampling. Default 0.15.

        Returns:
            tuple: (death_times, param_values) where:
                  death_times is numpy array of n death times
                  param_values is numpy array of n parameter values used
        """
        if param_name not in ['V0', 'B', 'lambda_', 'm']:
            raise ValueError("param_name must be one of 'V0','B','lambda_','m'.")

        # Get base parameter value
        pop_mean = getattr(self, param_name)
        
        # Draw n parameter values
        param_values = np.random.normal(loc=pop_mean, scale=std * pop_mean, size=n)
        # Ensure all values are positive by resampling negative values
        while np.any(param_values <= 0):
            neg_mask = param_values <= 0
            param_values[neg_mask] = np.random.normal(
                loc=pop_mean, 
                scale=std * pop_mean, 
                size=np.sum(neg_mask)
            )

        death_times = []
        for param_val in param_values:
            # Create parameter dictionary for this individual
            params = {
                'V0': self.V0 if param_name != 'V0' else param_val,
                'lambda_': self.lambda_ if param_name != 'lambda_' else param_val,
                'B': self.B if param_name != 'B' else param_val,
                'm': self.m if param_name != 'm' else param_val
            }
            
            # Sample single death time with these parameters
            death_time = self.sample_death_times(1, dt=dt, params=params)[0]
            death_times.append(death_time)

        return np.array(death_times), param_values

    def create_death_table_for_twins(self, twin_type='MZ', n=1000, param_name='V0', dt=1, std=0.15):
        """
        Create a DataFrame with death times for n pairs of twins (MZ or DZ).
        
        Args:
            twin_type (str): Either 'MZ' or 'DZ'
            n (int): Number of twin pairs to simulate
            param_name (str): Parameter to randomize ('V0', 'b', 'c', or 'm')
            dt (float): Time step for numerical integration
            std (float): Standard deviation for parameter sampling
            
        Returns:
            pd.DataFrame: DataFrame with columns ['death1','death2']
        """
        if param_name not in ['V0', 'B', 'lambda_', 'm']:
            raise ValueError("param_name must be one of 'V0','B','lambda_','m'")

        # Get base parameter value
        pop_mean = getattr(self, param_name)
        death_times_list = []

        # Generate correlated parameter values for all pairs at once
        if twin_type == 'MZ':
            # For MZ twins, generate one set and repeat
            param_values = np.random.normal(loc=pop_mean, scale=std * pop_mean, size=n)
            param_values = np.column_stack((param_values, param_values))
        elif twin_type == 'DZ':
            # For DZ twins, use correlation formula with rho = 0.5
            Z1 = np.random.randn(n)
            Z2 = np.random.randn(n)
            param_values1 = pop_mean + (std * pop_mean) * Z1
            param_values2 = pop_mean + (std * pop_mean) * (0.5 * Z1 + np.sqrt(1 - 0.5**2) * Z2)
            param_values = np.column_stack((param_values1, param_values2))
        else:
            raise ValueError("twin_type must be 'MZ' or 'DZ'")

        # Ensure all values are positive
        param_values = np.abs(param_values)

        for param_val1, param_val2 in param_values:
            # Create parameter dictionaries for both twins
            params1 = {
                'V0': self.V0 if param_name != 'V0' else param_val1,
                'lambda_': self.lambda_ if param_name != 'lambda_' else param_val1,
                'B': self.B if param_name != 'B' else param_val1,
                'm': self.m if param_name != 'm' else param_val1
            }
            
            params2 = {
                'V0': self.V0 if param_name != 'V0' else param_val2,
                'lambda_': self.lambda_ if param_name != 'lambda_' else param_val2,
                'B': self.B if param_name != 'B' else param_val2,
                'm': self.m if param_name != 'm' else param_val2
            }

            # Sample death times for both twins
            death1 = self.sample_death_times(1, dt=dt, params=params1)[0]
            death2 = self.sample_death_times(1, dt=dt, params=params2)[0]
            death_times_list.append([death1, death2])

        return pd.DataFrame(death_times_list, columns=['death1', 'death2'])
    
    def load_HMD_hazard_data(self, country, year, gender, data_type='cohort', haz_type='mx'):
        country_period_data = HMD(country=country, gender=gender, data_type=data_type)
        ages, hazard = country_period_data.calculate_hazard(year=year, haz_type=haz_type)
        return ages, hazard
    
    def load_HMD_survival_data(self, country, year, gender, data_type='cohort', haz_type='mx'):
        country_period_data = HMD(country=country, gender=gender, data_type=data_type)
        ages, survival = country_period_data.calculate_survival(year=year)
        return ages, survival
    
    def get_HMD_death_distribution(self, country, year, gender, data_type='cohort', haz_type='mx'):
        # Get survival data from HMD
        ages, survival = self.load_HMD_survival_data(country, year, gender)
        
        # Calculate death distribution as negative derivative of survival
        # Use numpy gradient to calculate derivative
        death_distribution = -np.gradient(survival, ages)
        
        return ages, death_distribution
    
    def plot_hazard_with_data_comparison(self, country, year, gender, data_type='cohort', haz_type='mx', t=np.linspace(0, 120, 200) , ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        ages, hazard = self.calculate_hazard_SM(t)
        
        ages_data, hazard_data = self.load_HMD_hazard_data(country=country, year=year, gender=gender, data_type=data_type, haz_type=haz_type)
        
        # Plot the results
        ax.scatter(ages_data, hazard_data, label='Data')
        ax.plot(ages, hazard, 'r-', label='Analytical Hazard Function')
        ax.legend()
        ax.set_xlabel('t')
        ax.set_yscale('log')
        ax.set_ylabel('Hazard [1/year]')
        
        return ax
    def plot_hazard_from_death_times(self, death_times, ax=None, **kwargs):
        """
        Plot hazard function using Nelson-Aalen estimator from observed death times.
        
        Parameters:
        -----------
        death_times : array-like
            Array of observed death times
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, current axes will be used
        **kwargs : dict
            Additional keyword arguments to pass to the plot function
        """
        if ax is None:
            ax = plt.gca()
            
        naf = NelsonAalenFitter()
        naf.fit(death_times, event_observed=np.ones(len(death_times)), timeline = np.arange(0, 110, 0.5))
        naf.plot_hazard(ax=ax, bandwidth=3, **kwargs)
        ax.set_yscale('log')
        return ax
    
    def plot_survival_from_death_times(self, death_times, ax=None, **kwargs):
        """
        Plot survival function using Kaplan-Meier estimator from observed death times.
        
        Parameters:
        -----------
        death_times : array-like
            Array of observed death times
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, current axes will be used
        **kwargs : dict
            Additional keyword arguments to pass to the plot function
        """
        if ax is None:
            ax = plt.gca()
            
        kmf = KaplanMeierFitter()
        kmf.fit(death_times)
        kmf.plot_survival_function(ax=ax, **kwargs)
        return ax
    
    

    def fit_params_with_std_with_ks(self, HMD_params, std=0.35, min_age=25, max_age=90, maxiter=250, callback=True, method='Nelder-Mead'):
        """
        Runs optimization to find best (V0, lambda_, B, m) that minimizes KS distance
        to the HMD-based normalized_distribution in ages min_age..max_age.
        
        Parameters:
        -----------
        HMD_params : dict
            Parameters to pass to get_HMD_death_distribution
        std : float, optional
            Standard deviation for random parameter variation, default 0.35
        min_age : int, optional
            Minimum age to consider in distribution, default 25
        max_age : int, optional 
            Maximum age to consider in distribution, default 90
        maxiter : int, optional
            Maximum number of iterations for optimization, default 250
        callback : bool, optional
            If True, print iteration information, default True
        method : str, optional
            The optimization method to use, default is 'Nelder-Mead'.
        """
        
        def compute_ks_statistic(death_times, ages_filtered, normalized_distribution):
            cdf_theoretical = np.cumsum(normalized_distribution)
            cdf_empirical = np.zeros_like(cdf_theoretical)
            counts, _ = np.histogram(death_times, bins=np.arange(ages_filtered[0], ages_filtered[-1] + 2))
            cdf_empirical = np.cumsum(counts) / sum(counts)
            ks_dist = np.max(np.abs(cdf_empirical - cdf_theoretical))
            return ks_dist

        def objective_function(param_array, model, ages_filtered, normalized_distribution):
            model_copy = copy.deepcopy(model)
            params_dict = {
                'V0':      param_array[0],
                'lambda_': param_array[1],
                'B':       param_array[2],
                'm':       param_array[3]
            }
            model_copy.set_params(params_dict)
            n_samples = 10000
            death_times, _ = model_copy.sample_death_times_with_random_param(
                n=n_samples,
                param_name='V0',
                dt=1, 
                std=std
            )
            ks_dist = compute_ks_statistic(death_times, ages_filtered, normalized_distribution)
            return ks_dist

        ages, death_distribution = self.get_HMD_death_distribution(**HMD_params)
        age_filter = (ages >= min_age) & (ages <= max_age)
        ages_filtered = ages[age_filter]
        ddist_filtered = death_distribution[age_filter]
        normalized_distribution = ddist_filtered / np.sum(ddist_filtered)

        baseline_params = [
            self.V0,
            self.lambda_,
            self.B,
            self.m
        ]

        def local_objective(x):
            return objective_function(x, self, ages_filtered, normalized_distribution)

        iteration = [0]  # Use list to allow modification in callback
        result = minimize(local_objective, 
                        x0=baseline_params, 
                        method=method,
                        options={'maxiter': maxiter, 'disp': True},
                        callback=lambda xk: (print(f"Iteration {iteration[0]}") if callback else None, iteration.__setitem__(0, iteration[0] + 1)))

        self.set_params(
            {
                'V0':      result.x[0],
                'lambda_': result.x[1],
                'B':       result.x[2],
                'm':       result.x[3],
            }
        )
        
        if result.success:
            best_params = result.x
            print("Optimization succeeded.")
            print("Best-fit parameters (V0, lambda_, B, m): ", best_params)
            print("Final KS distance: ", result.fun)
        else:
            print("Optimization failed with message:", result.message)

        return result

    def calc_corr(self, n=100000, std=0.3, twin_type='MZ', param_name='V0', filter_age=15, twin_table=None):
        if twin_table is None:
            twin_table = self.create_death_table_for_twins(n=n, std=std, param_name=param_name, twin_type=twin_type)
        twin_table = ta.filter_death_table(twin_table, filter_age)
        corr = twin_table['death1'].corr(twin_table['death2'])
        return corr
