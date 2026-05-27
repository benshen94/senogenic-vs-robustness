import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from multiprocessing import Pool, cpu_count
import warnings

class DoubleWell:
    """
    A class to simulate a double-well potential model for mortality.
    
    This class can solve two different SDEs:
    1. dy = 2 * (sqrt(1 - lamda*t)*y - y**3) dt + sqrt(delta) * dW
    2. dy = (2y - 2y^3 (1+lambda*t)) dt + sqrt(delta) * dW
    
    Death is defined as y crossing below 0.
    The simulation is run for N trajectories and death times are recorded.
    Survival and hazard functions are calculated from the death times.
    """

    def __init__(self, lamda, delta, well=1, n=50000, y0=1.0, dt=1e-2, tmax=150, a0=1.0, parallel=True, save_paths=False, save_interval=1.0, break_early=True):
        """
        Initializes and runs the simulation.
        
        Args:
            lamda (float): Parameter in the drift term. Controls the time-dependence.
            delta (float): Noise intensity.
            well (int): Which SDE to simulate. 1 for original, 3 for alternative.
            n (int): Number of simulation trajectories.
            y0 (float): Initial condition for y.
            dt (float): Physical time step for the simulation (years).
            tmax (float, optional): Maximum simulation time (years). If None, defaults to 1/lamda.
            a0 (float): Fundamental curvature scale (yr⁻¹). Controls overall time scale.
            parallel (bool): Whether to run the simulation in parallel. Defaults to True.
            save_paths (bool): Whether to save full trajectories. Defaults to False.
            save_interval (float): Time interval for saving paths (years). Defaults to 1.0.
            break_early (bool): If True, stop simulation early when all trajectories have died. Defaults to True.
        """
        self.lamda = lamda
        self.delta = delta
        self.well = well
        self.n = n
        self.y0 = y0
        self.a0 = a0  # fundamental curvature (yr⁻¹)
        
        # keep user-facing dt,tmax in physical units
        self.dt_phys = dt
        if tmax is None:
            if self.lamda > 0:
                self.tmax_phys = 1.0 / self.lamda
            else:
                raise ValueError("tmax must be provided if lamda is zero or negative.")
        else:
            self.tmax_phys = tmax
            
        # convert to dimension-less τ units for the integrator
        self.dt = self.dt_phys * self.a0
        self.tmax = self.tmax_phys * self.a0
        
        self.parallel = parallel
        self.save_paths = save_paths
        self.save_interval = save_interval
        self.break_early = break_early

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
        
        self.run_simulation()
        self._post_simulation_calculations()

    def _tau_to_years(self, arr_tau):
        """Convert dimension-less tau times to physical years."""
        return arr_tau / self.a0

    def run_simulation(self):
        """Runs the simulation, either in serial or parallel."""
        if self.parallel:
            death_times_tau, self.paths = self._run_parallel()
        else:
            death_times_tau, self.paths = self._simulate_chunk(self.n)
        
        # Convert death times from tau to physical years
        self.death_times = self._tau_to_years(death_times_tau)
        self.alive_mask = (self.death_times == np.inf)

        if self.save_paths and self.paths is not None:
            num_saved_points = self.paths.shape[1]
            # saved every save_interval_phys years
            self.tspan_paths = np.arange(num_saved_points) * self.save_interval

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
            if self.break_early and not np.any(active):
                break

            mask = active
            y_masked = y[mask]
            
            noise = np.sqrt(self.delta * self.dt) * np.random.randn(np.sum(mask))
            
            if self.well == 1:
                # Original SDE
                term = 1 - self.lamda * t_cur
                sqrt_term = np.sign(term) * np.sqrt(np.abs(term))
                drift = 2 * (sqrt_term * y_masked - y_masked**3) * self.dt
            else:  # well == 3
                # Alternative SDE
                drift = (2 * y_masked - 2 * y_masked**3 * (1 + self.lamda * t_cur)) * self.dt
                
            y[mask] += drift + noise

            if self.save_paths and (i + 1) % save_interval_steps == 0:
                saved_paths_list.append(y.copy())

            newly_dead_mask = (y[mask] <= 0)
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
        
        # Create a copy of death times for fitting
        fittable_death_times = np.copy(self.death_times)
        
        # For censored individuals (alive at end), use tmax_phys as censoring time
        fittable_death_times[self.alive_mask] = self.tmax_phys
        
        # Fit Kaplan-Meier
        self.kmf = KaplanMeierFitter()
        self.kmf.fit(fittable_death_times, event_observed=event_observed)
        self.survival = self.kmf.survival_function_
        self.tspan_survival = self.kmf.timeline
        
        # Calculate median survival time
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.median_t = self.kmf.median_survival_time_

        # Fit Nelson-Aalen
        self.naf = NelsonAalenFitter()
        self.naf.fit(fittable_death_times, event_observed=event_observed)
        
        # Calculate smoothed hazard
        try:
            sm_haz_table = self.naf.smoothed_hazard_(bandwidth=3)
            self.tspan_hazard = sm_haz_table.index.values
            self.hazard = sm_haz_table.values.flatten()
        except Exception:
            self.tspan_hazard = np.array([])
            self.hazard = np.array([])

        # Calculate steepness
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
        # Identify individuals alive at time t and get their death times
        alive_mask = self.death_times >= t
        if not np.any(alive_mask):
            return None

        death_times_from_t = self.death_times[alive_mask]
        
        # Replace np.inf with tmax_phys for censored individuals
        # For conditional survival from time t, censored times should be (tmax_phys - t)
        censored_death_times_from_t = np.copy(death_times_from_t)
        inf_mask = (death_times_from_t == np.inf)
        censored_death_times_from_t[inf_mask] = self.tmax_phys
        
        event_times = np.maximum(0, censored_death_times_from_t - t)
        event_observed = (~inf_mask).astype(int)

        # Create and fit Kaplan-Meier Fitter
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

        ax.axhline(y=0, color='grey', linestyle=':')  # Death threshold
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