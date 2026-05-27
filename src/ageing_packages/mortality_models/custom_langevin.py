import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from multiprocessing import Pool, cpu_count
import warnings
import sympy

class CustomLangevin:
    """
    A class to simulate a user-defined Langevin SDE for mortality.
    
    The SDE is of the form:
    dy = f(y, t) dt + g(y, t) * dW
    
    The user provides the functional forms for the drift f(y, t) and the noise term g(y, t)
    as strings.
    
    Death is defined as y crossing below a threshold (default 0).
    The simulation is run for N trajectories and death times are recorded.
    Survival and hazard functions are calculated from the death times.
    """

    def __init__(self, n=50000, dt=1e-2, a0=1.0, parallel=True, save_paths=False, save_interval=1.0):
        """
        Initializes the simulator environment. The SDE is not defined here.
        
        Args:
            n (int): Number of simulation trajectories.
            dt (float): Physical time step for the simulation (years).
            a0 (float): Fundamental curvature scale (yr⁻¹). Controls overall time scale.
            parallel (bool): Whether to run the simulation in parallel. Defaults to True.
            save_paths (bool): Whether to save full trajectories. Defaults to False.
            save_interval (float): Time interval for saving paths (years). Defaults to 1.0.
        """
        self.n = n
        self.a0 = a0  # fundamental curvature (yr⁻¹)
        
        # keep user-facing dt in physical units
        self.dt_phys = dt
        # convert to dimension-less τ units for the integrator
        self.dt = self.dt_phys * self.a0
        
        self.parallel = parallel
        self.save_paths = save_paths
        self.save_interval = save_interval

        # SDE parts
        self.drift_str = None
        self.noise_str = None
        self.drift_expr = None
        self.noise_expr = None
        self.drift_func = None
        self.noise_func = None
        self.params = None
        self.param_symbols = None
        self.param_values = None

        # Simulation-specific, will be set in run_simulation
        self.y0 = None
        self.tmax_phys = None
        self.tmax = None
        self.death_threshold = None

        # To be populated by simulation
        self.death_times = None
        self.paths = None
        self.tspan_paths = None
        self.alive_mask = None
        self.kmf = None
        self.naf = None
        self.survival = None
        self.tspan_survival = None
        self.hazard = None
        self.tspan_hazard = None
        self.median_t = None
        self.steepness = None

    def _tau_to_years(self, arr_tau):
        """Convert dimension-less tau times to physical years."""
        return arr_tau / self.a0
        
    def set_sde(self, drift_str, noise_str):
        """
        Sets the SDE drift and noise functions from strings.
        It also identifies the parameters that need to be provided during simulation.

        Args:
            drift_str (str): String expression for drift f(y, t).
            noise_str (str): String expression for noise g(y, t).
        """
        self.drift_str = drift_str
        self.noise_str = noise_str
        
        y_sym, t_sym = sympy.symbols('y t')
        
        try:
            self.drift_expr = sympy.sympify(drift_str)
            self.noise_expr = sympy.sympify(noise_str)
        except sympy.SympifyError as e:
            raise ValueError(f"Could not parse expressions: {e}")

        drift_symbols = self.drift_expr.free_symbols
        noise_symbols = self.noise_expr.free_symbols
        all_symbols = drift_symbols.union(noise_symbols)
        
        param_symbols = sorted(
            [s for s in all_symbols if s not in [y_sym, t_sym]], 
            key=lambda s: str(s)
        )
        self.param_symbols = param_symbols
        
        print(f"SDE set.")
        if self.param_symbols:
            print(f"Identified parameters: {[str(p) for p in self.param_symbols]}")
            print("Provide their values when calling run_simulation().")
        else:
            print("No free parameters identified.")

    def set_params(self, params):
        """
        Set the parameter values for the SDE ahead of time.
        
        Args:
            params (dict): A dictionary mapping parameter names (as strings)
                          to their numerical values.
        """
        if self.drift_str is None or self.noise_str is None:
            raise RuntimeError("SDE is not set. Call set_sde() first.")
        
        # Check if all required parameters are provided
        param_names = [str(p) for p in self.param_symbols]
        if not all(p_name in params for p_name in param_names):
            missing = [p_name for p_name in param_names if p_name not in params]
            raise ValueError(f"Missing required parameters: {missing}")
        
        self.params = params
        self.param_values = [self.params[p_name] for p_name in param_names]
        
        # Lambdify expressions for numerical evaluation
        y_sym, t_sym = sympy.symbols('y t')
        self.drift_func = sympy.lambdify([y_sym, t_sym] + self.param_symbols, self.drift_expr, 'numpy')
        self.noise_func = sympy.lambdify([y_sym, t_sym] + self.param_symbols, self.noise_expr, 'numpy')
        
        print(f"Parameters set: {self.params}")

    def run_simulation(self, y0, tmax, death_threshold=0.0, params=None):
        """
        Runs the simulation with the provided simulation conditions and SDE parameters.

        Args:
            y0 (float): Initial condition for y.
            tmax (float): Maximum simulation time (years).
            death_threshold (float): y value below which death occurs. Defaults to 0.0.
            params (dict, optional): A dictionary mapping parameter names (as strings)
                                   to their numerical values. If None, uses pre-set parameters.
        """
        if self.drift_str is None or self.noise_str is None:
            raise RuntimeError("SDE is not set. Call set_sde() first.")
        
        # Set simulation-specific parameters
        self.y0 = y0
        self.tmax_phys = tmax
        self.death_threshold = death_threshold
        self.tmax = self.tmax_phys * self.a0

        # Handle parameters - use provided params or pre-set ones
        if params is not None:
            # Use parameters provided directly to run_simulation
            self.params = params
            
            # Check if all required parameters are provided
            param_names = [str(p) for p in self.param_symbols]
            if not all(p_name in self.params for p_name in param_names):
                missing = [p_name for p_name in param_names if p_name not in self.params]
                raise ValueError(f"Missing required parameters: {missing}")

            self.param_values = [self.params[p_name] for p_name in param_names]

            # Lambdify expressions for numerical evaluation
            y_sym, t_sym = sympy.symbols('y t')
            self.drift_func = sympy.lambdify([y_sym, t_sym] + self.param_symbols, self.drift_expr, 'numpy')
            self.noise_func = sympy.lambdify([y_sym, t_sym] + self.param_symbols, self.noise_expr, 'numpy')
        else:
            # Use pre-set parameters
            if self.params is None or self.drift_func is None or self.noise_func is None:
                raise RuntimeError("No parameters set. Either call set_params() first or provide params to run_simulation().")
        
        # Run simulation
        if self.parallel:
            death_times_tau, self.paths = self._run_parallel()
        else:
            death_times_tau, self.paths = self._simulate_chunk(self.n)
        
        # Convert death times from tau to physical years
        self.death_times = self._tau_to_years(death_times_tau)
        self.alive_mask = (self.death_times == np.inf)

        if self.save_paths and self.paths is not None:
            num_saved_points = self.paths.shape[1]
            # saved every save_interval years
            self.tspan_paths = np.arange(num_saved_points) * self.save_interval

        self._post_simulation_calculations()
        print("Simulation and analysis complete.")

    def _run_parallel(self):
        """Runs the simulation in parallel across CPU cores."""
        num_cores = cpu_count()
        n_chunks = np.full(num_cores, self.n // num_cores)
        n_chunks[:self.n % num_cores] += 1
        
        n_chunks = [c for c in n_chunks if c > 0]
        
        with Pool(processes=len(n_chunks)) as pool:
            results = pool.map(self._simulate_chunk, n_chunks)
        
        death_times = np.concatenate([r[0] for r in results])
        
        paths = None
        if self.save_paths:
            paths_list = [r[1] for r in results if r[1] is not None]
            if paths_list:
                max_time_steps = max(p.shape[1] for p in paths_list)
                paths = np.zeros((self.n, max_time_steps))
                
                current_idx = 0
                for p_chunk in paths_list:
                    chunk_size, num_steps = p_chunk.shape
                    paths[current_idx:current_idx + chunk_size, :num_steps] = p_chunk
                    current_idx += chunk_size

        return death_times, paths

    def _simulate_chunk(self, n_chunk):
        """Simulates a chunk of trajectories."""
        y = np.full(n_chunk, self.y0, dtype=np.float64)
        death_times = np.full(n_chunk, np.inf, dtype=np.float64)
        active = np.ones(n_chunk, dtype=bool)
        
        t_steps = np.arange(0, self.tmax, self.dt)

        saved_paths_list = []
        save_interval_steps = 0
        if self.save_paths:
            save_interval_steps = max(1, int(round(self.save_interval / self.dt_phys)))
            saved_paths_list.append(y.copy())  # Save initial state

        for i, t_cur in enumerate(t_steps):
            if not np.any(active):
                break

            mask = active
            y_masked = y[mask]
            
            # Use the user-defined functions
            drift = self.drift_func(y_masked, t_cur, *self.param_values) * self.dt
            noise = self.noise_func(y_masked, t_cur, *self.param_values) * np.sqrt(self.dt) * np.random.randn(np.sum(mask))
                
            y[mask] += drift + noise

            if self.save_paths and (i + 1) % save_interval_steps == 0:
                saved_paths_list.append(y.copy())

            newly_dead_mask = (y[mask] <= self.death_threshold)
            if np.any(newly_dead_mask):
                active_indices = np.where(mask)[0]
                dead_indices_in_active = np.where(newly_dead_mask)[0]
                dead_global_indices = active_indices[dead_indices_in_active]
                
                # Check for those not already dead to avoid overwriting
                not_yet_dead_mask_local = (death_times[dead_global_indices] == np.inf)
                indices_to_update = dead_global_indices[not_yet_dead_mask_local]

                if len(indices_to_update) > 0:
                    death_times[indices_to_update] = t_cur + self.dt
                    active[indices_to_update] = False
    
        paths = None
        if self.save_paths and saved_paths_list:
            paths = np.stack(saved_paths_list, axis=1)

        return death_times, paths

    def _post_simulation_calculations(self):
        """Calculates survival, hazard, etc. from death times."""
        if self.death_times is None:
            return
            
        event_observed = ~self.alive_mask
        
        fittable_death_times = np.copy(self.death_times)
        fittable_death_times[self.alive_mask] = self.tmax_phys
        
        self.kmf = KaplanMeierFitter()
        self.kmf.fit(fittable_death_times, event_observed=event_observed)
        self.survival = self.kmf.survival_function_
        self.tspan_survival = self.kmf.timeline
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.median_t = self.kmf.median_survival_time_

        self.naf = NelsonAalenFitter()
        self.naf.fit(fittable_death_times, event_observed=event_observed)
        
        try:
            sm_haz_table = self.naf.smoothed_hazard_(bandwidth=3)
            self.tspan_hazard = sm_haz_table.index.values
            self.hazard = sm_haz_table.values.flatten()
        except Exception:
            self.tspan_hazard = np.array([])
            self.hazard = np.array([])

        self.steepness = self.calc_steepness()

    def _calc_CV(self):
        """
        Calculate the coefficient of variation (CV = std/mean) of y values over time
        for individuals still alive. Only works if save_paths=True.
        
        Returns:
            tuple: (cv_array, time_array) where cv_array contains CV values and 
                   time_array contains corresponding time points, or (None, None) if paths not saved
        """
        if not self.save_paths or self.paths is None:
            print("Paths were not saved. Cannot calculate CV over time.")
            return None, None
            
        if self.death_times is None:
            print("No simulation results available.")
            return None, None
            
        cv_values = []
        time_points = []
        
        # For each time point in the saved paths
        for i in range(self.paths.shape[1]):
            current_time = i * self.save_interval
            time_points.append(current_time)
            
            # Find individuals still alive at this time point
            alive_mask = self.death_times > current_time
            
            if np.sum(alive_mask) > 1:  # Need at least 2 individuals for meaningful CV
                y_values_alive = self.paths[alive_mask, i]
                
                # Remove any NaN or infinite values
                valid_mask = np.isfinite(y_values_alive)
                y_values_clean = y_values_alive[valid_mask]
                
                if len(y_values_clean) > 1:
                    mean_y = np.mean(y_values_clean)
                    std_y = np.std(y_values_clean)
                    
                    if mean_y != 0:
                        cv = std_y / abs(mean_y)
                    else:
                        cv = np.nan
                else:
                    cv = np.nan
            else:
                cv = np.nan
                
            cv_values.append(cv)
        
        return np.array(cv_values), np.array(time_points)

    def plot_CV(self, ax=None, **kwargs):
        """
        Plot the coefficient of variation over time for surviving individuals.
        
        Args:
            ax: matplotlib axes object. If None, creates new figure.
            **kwargs: Additional arguments passed to plot function.
            
        Returns:
            matplotlib axes object
        """
        cv_values, time_points = self._calc_CV()
        
        if cv_values is None:
            return None
            
        if ax is None:
            _, ax = plt.subplots()
            
        # Remove NaN values for plotting
        valid_mask = ~np.isnan(cv_values)
        if np.sum(valid_mask) > 0:
            ax.plot(time_points[valid_mask], cv_values[valid_mask], **kwargs)
            ax.set_xlabel('Time')
            ax.set_ylabel('Coefficient of Variation (CV)')
            ax.set_title('CV of y-values over time (surviving individuals)')
            ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
        else:
            print("No valid CV values to plot.")
            
        return ax

    def find_time_at_survival(self, S, from_t=None):
        """
        Find time at which survival probability equals S.

        Args:
            S (float): Survival probability (0-1)
            from_t (float, optional): Start time for conditional survival

        Returns:
            float: Time at survival probability S, or None if not reached
        """
        if from_t is not None:
            kmf = self._create_survival_from_t(from_t)
            if kmf is None:
                return None
            survival_func = kmf.survival_function_.iloc[:, 0]
            timeline = kmf.timeline
        else:
            survival_func = self.survival.iloc[:, 0]
            timeline = self.survival.index

        if np.any(survival_func.values <= S):
            return np.interp(S, survival_func.values[::-1], timeline[::-1])
        return None

    def _create_survival_from_t(self, t):
        """
        Create Kaplan-Meier survival curve for individuals alive at time t.

        Args:
            t (float): The time from which to calculate survival.

        Returns:
            KaplanMeierFitter or None: Fitted KMF object, or None if no individuals alive at t.
        """
        alive_mask = self.death_times >= t
        if not np.any(alive_mask):
            return None

        death_times_from_t = self.death_times[alive_mask]
        
        censored_death_times_from_t = np.copy(death_times_from_t)
        inf_mask = (death_times_from_t == np.inf)
        censored_death_times_from_t[inf_mask] = self.tmax_phys
        
        event_times = np.maximum(0, censored_death_times_from_t - t)
        event_observed = (~inf_mask).astype(int)

        kmf = KaplanMeierFitter()
        kmf.fit(event_times, event_observed=event_observed)
        return kmf

    def calc_steepness(self, method='IQR', from_t=None):
        """
        Calculate steepness of survival curve.
        
        Steepness is defined as median survival time divided by
        the time difference between 75% and 25% survival.

        Args:
            method (str): Method for calculating steepness ('IQR' or 'CV')
            from_t (float, optional): Start time for conditional survival

        Returns:
            float: Steepness value, or None if insufficient mortality
        """
        if method == 'IQR':
            t_25 = self.find_time_at_survival(0.25, from_t)
            t_50 = self.find_time_at_survival(0.5, from_t)
            t_75 = self.find_time_at_survival(0.75, from_t)
            
            if all(t is not None for t in [t_25, t_50, t_75]) and t_75 != t_25:
                return -t_50 / (t_75 - t_25)
        elif method == 'CV':
            if from_t is None:
                from_t = 0
            filtered_death_times = self.death_times[(self.death_times >= from_t) & (self.death_times != np.inf)]
            if len(filtered_death_times) > 0:
                mean_time = np.mean(filtered_death_times)
                std_time = np.std(filtered_death_times)
                if mean_time > 0:
                    cv = std_time / mean_time
                    if cv > 0:
                        return 1 / cv
        return None

    def plot_some_paths(self, n, ax=None):
        """Plots a random sample of n trajectories."""
        if not self.save_paths or self.paths is None:
            print("Paths were not saved. Rerun simulation with save_paths=True.")
            return

        if ax is None:
            _, ax = plt.subplots()

        num_paths_to_plot = min(n, self.paths.shape[0])
        for _ in range(num_paths_to_plot):
            random_row_index = np.random.choice(self.paths.shape[0])
            ax.plot(self.tspan_paths, self.paths[random_row_index, :])

        ax.axhline(y=self.death_threshold, color='grey', linestyle=':')  # Death threshold
        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('y')
        return ax

    def plot_survival(self, ax=None, **kwargs):
        """Plots the Kaplan-Meier survival curve."""
        if ax is None:
            _, ax = plt.subplots()
        if self.kmf is None:
            return ax
            
        self.kmf.plot_survival_function(ax=ax, **kwargs)
        ax.set_xlabel(f'Time')
        ax.set_ylabel('Survival Probability')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        return ax
        
    def plot_hazard(self, ax=None, **kwargs):
        """Plots the smoothed hazard function."""
        if ax is None:
            _, ax = plt.subplots()
        if self.hazard is None or len(self.hazard) == 0:
            return ax

        ax.plot(self.tspan_hazard, self.hazard, **kwargs)
        ax.set_xlabel(f'Time')
        ax.set_ylabel(f'Hazard Rate')
        ax.set_yscale('log')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        return ax 

if __name__ == '__main__':
    # ===============================================================
    # Example Usage
    # ===============================================================

    # 1. Instantiate the simulator
    # We specify general environment parameters here.
    print("Instantiating the simulator...")
    langevin_sim = CustomLangevin(
        n=10000, 
        dt=1e-3, 
        save_paths=True, 
        save_interval=1.0
    )

    # 2. Define the SDE
    # Let's use a simple Ornstein-Uhlenbeck process: 
    # dy/dt = -a*y + sqrt(2*epsilon) * dW
    print("\nSetting the SDE...")
    drift_eq = "-a * y"
    noise_eq = "sqrt(2 * epsilon)" # Note: for sympy, use sqrt(), not np.sqrt()
    langevin_sim.set_sde(drift_str=drift_eq, noise_str=noise_eq)
    # This will print the identified parameters: ['a', 'epsilon']

    # 3a. Method 1: Set parameters first, then run simulation
    print("\nMethod 1: Setting parameters first...")
    simulation_params = {
        'a': 0.1,
        'epsilon': 0.05
    }
    langevin_sim.set_params(simulation_params)
    
    # Run the simulation (using pre-set parameters)
    print("Running the simulation...")
    langevin_sim.run_simulation(
        y0=1.0,
        tmax=200,
        death_threshold=0.0
    )

    # 4. Analyze and plot the results
    print("\nSimulation finished. Here are some results:")
    print(f"Median survival time: {langevin_sim.median_t:.2f} years")
    print(f"Steepness (IQR): {langevin_sim.steepness:.2f}")

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot some example trajectories
    print("\nPlotting results...")
    langevin_sim.plot_some_paths(n=10, ax=axes[0,0])
    axes[0,0].set_title("Example Trajectories")

    # Plot survival curve
    langevin_sim.plot_survival(ax=axes[0,1])
    axes[0,1].set_title("Survival Curve")

    # Plot hazard rate
    langevin_sim.plot_hazard(ax=axes[1,0])
    axes[1,0].set_title("Hazard Rate")
    
    # Plot coefficient of variation
    langevin_sim.plot_CV(ax=axes[1,1])
    axes[1,1].set_title("Coefficient of Variation")
    
    plt.tight_layout()
    plt.show()

    # ===============================================================
    # Method 2: Pass parameters directly to run_simulation
    # ===============================================================
    
    print("\n\n--- Method 2: Parameters passed directly to run_simulation ---")
    
    # We can reuse the same simulator instance or create a new one.
    langevin_sim_2 = CustomLangevin(n=10000, dt=1e-3, save_paths=False)
    
    # A bistable potential with a time-dependent term
    # dy/dt = (y - y**3) - lamda*t*y + sigma*dW
    print("\nSetting the SDE...")
    drift_eq_2 = "y - y**3 - lamda * t * y"
    noise_eq_2 = "sigma"
    langevin_sim_2.set_sde(drift_str=drift_eq_2, noise_str=noise_eq_2)

    # Method 2: Pass parameters directly to run_simulation
    print("\nRunning the simulation with parameters passed directly...")
    langevin_sim_2.run_simulation(
        y0=5.0,
        tmax=100,
        params={'lamda': 0.01, 'sigma': 0.4}  # Parameters passed here
    )

    print(f"Median survival time: {langevin_sim_2.median_t:.2f} years")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    langevin_sim_2.plot_some_paths(n=10, ax=axes[0])
    langevin_sim_2.plot_survival(ax=axes[1])
    langevin_sim_2.plot_hazard(ax=axes[2])
    plt.tight_layout()
    plt.show()

    # ===============================================================
    # Example with a fractional power in the drift term
    # ===============================================================
    # To use a drift term like y^(2/3), use the Python power syntax `**`.
    
    print("\n\n--- Example with Fractional Power Drift ---")
    
    langevin_sim_3 = CustomLangevin(n=5000, dt=1e-3)
    
    print("\nSetting the SDE...")
    drift_eq_3 = "-k * y**(2/3)"
    noise_eq_3 = "sigma"
    langevin_sim_3.set_sde(drift_str=drift_eq_3, noise_str=noise_eq_3)

    print("\nRunning the simulation...")
    langevin_sim_3.run_simulation(
        y0=10.0,
        tmax=150,
        death_threshold=0.1,
        params={'k': 0.5, 'sigma': 0.2}
    )

    print(f"Median survival time: {langevin_sim_3.median_t:.2f} years")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    langevin_sim_3.plot_some_paths(n=10, ax=axes[0])
    langevin_sim_3.plot_survival(ax=axes[1])
    langevin_sim_3.plot_hazard(ax=axes[2])
    plt.tight_layout()
    plt.show() 
