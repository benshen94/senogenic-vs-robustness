import numpy as np
from numba import njit, prange
from numba.typed import List as NumbaList
from sympy import symbols, sympify, lambdify
from multiprocessing import Pool, cpu_count
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from scipy.stats import gaussian_kde, gennorm, norm, gamma, beta as beta_dist
import collections # Import callable


# ============================================================================
# NUMBA-OPTIMIZED SIMULATION KERNELS
# ============================================================================
# These are standalone JIT-compiled functions for maximum performance.
# They handle the core computation without Python overhead.

@njit(cache=True)
def _default_drift_numba(X, tcur, eta, beta, kappa):
    """
    Default SR model drift function (numba optimized).
    
    Args:
        X (np.ndarray): Current state for all particles
        tcur (float): Current time
        eta (np.ndarray): Production rate (per particle)
        beta (np.ndarray): Removal rate (per particle)
        kappa (np.ndarray): Half-saturation constant (per particle)
    
    Returns:
        np.ndarray: Drift term for SR model
    """
    n = X.shape[0]
    drift = np.empty(n)
    for i in range(n):
        drift[i] = eta[i] * tcur - X[i] * (beta[i] / (X[i] + kappa[i]))
    return drift


@njit(cache=True)
def _seed_numba_rng(seed):
    """
    Seed Numba's internal RNG state.

    Numba-managed random draws do not reliably follow Python-level
    ``np.random.seed(...)`` calls, so the simulation must seed both.
    """
    np.random.seed(seed)


@njit(cache=True)
def _skorokhod_step_numba(X, drift, sqrt_2eps, sqrt_dt):
    """
    Perform one Skorokhod step for all particles (numba optimized).
    
    Implements reflecting boundary condition at zero and
    combines drift and diffusion terms efficiently.
    
    Args:
        X (np.ndarray): Current state for all particles
        drift (np.ndarray): Drift values for all particles
        sqrt_2eps (np.ndarray): sqrt(2*epsilon) for each particle (pre-computed)
        sqrt_dt (float): sqrt(dt) (pre-computed)
    
    Returns:
        np.ndarray: Next state for all particles
    """
    n = X.shape[0]
    X_next = np.empty(n)
    
    for i in range(n):
        # Noise term (only if X > 0)
        if X[i] > 0:
            noise_coeff = sqrt_2eps[i]
        else:
            noise_coeff = 0.0
        
        # Generate random numbers
        Y = noise_coeff * sqrt_dt * np.random.randn()
        U = np.random.random()
        
        # Skorokhod reflection
        Y_sq = Y * Y
        log_U_term = -2.0 * sqrt_dt * sqrt_dt * np.log(max(U, 1e-100))
        inside_sqrt = Y_sq + log_U_term
        M = (Y + np.sqrt(max(inside_sqrt, 0.0))) / 2.0
        
        # Combine drift and reflection
        delta_X = sqrt_dt * sqrt_dt * drift[i]  # dt = sqrt_dt^2
        X_next[i] = max(M - Y, X[i] + delta_X - Y)
    
    return X_next


@njit(cache=True)
def _check_crossings_and_deaths_numba(X, X_prev, Xc, death_times, extrinsic_deaths, 
                                       p_death_ext, eps_dt, tcur, not_yet_crossed_indices):
    """
    Check for threshold crossings and extrinsic deaths (numba optimized).
    Uses Brownian bridge probability to detect mid-step crossings.
    
    Args:
        X (np.ndarray): Current state for all particles
        X_prev (np.ndarray): Previous state for all particles
        Xc (np.ndarray): Threshold for each particle
        death_times (np.ndarray): Death times array (modified in place)
        extrinsic_deaths (np.ndarray): Extrinsic death flags (modified in place)
        p_death_ext (np.ndarray): Probability of extrinsic death for each particle
        eps_dt (np.ndarray): epsilon * dt for each particle (for Brownian bridge)
        tcur (float): Current time
        not_yet_crossed_indices (np.ndarray): Indices of particles not yet crossed
    
    Returns:
        int: Number of new deaths
    """
    n_deaths = 0
    for idx in not_yet_crossed_indices:
        crossed = X[idx] > Xc[idx]
        
        # Brownian bridge crossing check: if both endpoints are below Xc,
        # there is still a probability the continuous path crossed Xc
        bridge_crossed = False
        if not crossed and X_prev[idx] < Xc[idx]:
            # P_cross = exp(-2 * (Xc - x_prev) * (Xc - x_cur) / (eps * dt))
            gap_prev = Xc[idx] - X_prev[idx]
            gap_cur = Xc[idx] - X[idx]
            if eps_dt[idx] > 0.0:
                p_cross = np.exp(-(gap_prev * gap_cur) / eps_dt[idx])
                if np.random.random() < p_cross:
                    bridge_crossed = True
        
        extrinsic = False
        # Check for extrinsic death
        if p_death_ext[idx] > 0.0:
            if np.random.random() < p_death_ext[idx]:
                extrinsic = True
        
        if crossed or bridge_crossed or extrinsic:
            death_times[idx] = tcur
            if extrinsic and not crossed and not bridge_crossed:
                extrinsic_deaths[idx] = 1
            n_deaths += 1
    
    return n_deaths


@njit(cache=True, parallel=False)
def _simulate_paths_fast(
    n,
    time_grid,
    save_indices,
    x0_array,
    eta,
    beta,
    kappa,
    sqrt_2eps,
    Xc,
    Xdisease,
    h_ext_values,
    epsilon,
    break_early,
    track_disease,
):
    """
    Fast simulation kernel on an explicit time grid.

    Includes Brownian bridge correction for mid-step threshold crossings.
    """
    n_steps = len(time_grid)
    n_save_points = len(save_indices)

    # Allocate output arrays
    paths = np.zeros((n, n_save_points))
    death_times = np.full(n, np.inf)
    extrinsic_deaths = np.zeros(n, dtype=np.int32)
    disease_times = np.empty(0, dtype=np.float64)
    if track_disease:
        disease_times = np.full(n, np.inf)

    # Initialize state
    X = x0_array.copy()
    X_prev = x0_array.copy()
    paths[:, 0] = X

    actual_save_points = n_save_points
    next_save_ptr = 1

    for i_t in range(1, n_steps):
        tprev = time_grid[i_t - 1]
        tcur = time_grid[i_t]
        dt = tcur - tprev
        sqrt_dt = np.sqrt(dt)

        if break_early:
            all_dead = True
            for i in range(n):
                if death_times[i] == np.inf:
                    all_dead = False
                    break
            if all_dead:
                actual_save_points = next_save_ptr
                break

        for i in range(n):
            X_prev[i] = X[i]

        drift = np.empty(n)
        for i in range(n):
            drift[i] = eta[i] * tcur - X[i] * (beta[i] / (X[i] + kappa[i]))

        X_next = np.empty(n)
        for i in range(n):
            if X[i] > 0:
                noise_coeff = sqrt_2eps[i]
            else:
                noise_coeff = 0.0

            Y = noise_coeff * sqrt_dt * np.random.randn()
            U = np.random.random()

            Y_sq = Y * Y
            log_U_term = -2.0 * dt * np.log(max(U, 1e-100))
            inside_sqrt = Y_sq + log_U_term
            M = (Y + np.sqrt(max(inside_sqrt, 0.0))) / 2.0

            delta_X = dt * drift[i]
            X_next[i] = max(M - Y, X[i] + delta_X - Y)

        X = X_next

        if next_save_ptr < n_save_points and i_t == save_indices[next_save_ptr]:
            for i in range(n):
                paths[i, next_save_ptr] = X[i]
            next_save_ptr += 1

        for i in range(n):
            if track_disease and death_times[i] == np.inf and disease_times[i] == np.inf:
                disease_crossed = X[i] > Xdisease[i]

                disease_bridge_crossed = False
                if not disease_crossed and X_prev[i] < Xdisease[i]:
                    gap_prev = Xdisease[i] - X_prev[i]
                    gap_cur = Xdisease[i] - X[i]

                    if epsilon[i] > 0.0 and dt > 0.0:
                        p_cross = np.exp(-(gap_prev * gap_cur) / (epsilon[i] * dt))
                        if np.random.random() < p_cross:
                            disease_bridge_crossed = True

                if disease_crossed or disease_bridge_crossed:
                    disease_times[i] = tcur

            if death_times[i] == np.inf:
                crossed = X[i] > Xc[i]

                bridge_crossed = False
                if not crossed and X_prev[i] < Xc[i]:
                    gap_prev = Xc[i] - X_prev[i]
                    gap_cur = Xc[i] - X[i]

                    if epsilon[i] > 0.0 and dt > 0.0:
                        p_cross = np.exp(-(gap_prev * gap_cur) / (epsilon[i] * dt))
                        if np.random.random() < p_cross:
                            bridge_crossed = True

                extrinsic = False
                if h_ext_values[i] > 0.0:
                    p_death_ext = 1.0 - np.exp(-h_ext_values[i] * dt)
                    if np.random.random() < p_death_ext:
                        extrinsic = True

                if crossed or bridge_crossed or extrinsic:
                    death_times[i] = tcur
                    if extrinsic and not crossed and not bridge_crossed:
                        extrinsic_deaths[i] = 1
                    elif track_disease and disease_times[i] == np.inf and Xdisease[i] <= Xc[i]:
                        disease_times[i] = tcur

    return death_times, paths, actual_save_points, extrinsic_deaths, disease_times


class SimulationParams:
    """
    A class to manage simulation parameters for the SR model.
    
    This class handles parameter processing, drift function setup, and parameter validation
    for simulating stochastic resilience (SR) paths.

    Attributes:
        raw_* : Original input parameters before processing
        n (int): Number of simulations to run
        tmin (float): Start time of simulation
        tmax (float): End time of simulation
        dt (float): Time step size
        save_times (float): Interval for saving simulation states
        parallel (bool): Whether to run simulation in parallel
        drift_expr (str/expr): Custom drift function expression
        drift_mode (str): How to apply custom drift ('replace' or 'add')
        extra_params (dict): Additional parameters for custom drift
    """
    
    def __init__(self, eta, beta, kappa, epsilon, Xc, Xdisease=None, n=10000, tmin=0, tmax=1000, x0=1e-10,
                 dt=1, save_times=None, dt_schedule=None, adaptive_dt=None, h_ext=None,
                 units='days', parallel=False, break_early=True, drift_expr=None,
                 drift_mode='replace', extra_params=None, start_idx=0, random_seed=None,
                 use_fast_kernel=True):
        """
        Initialize simulation parameters.

        Args:
            eta (float/array): Production rate
            beta (float/array): Removal rate
            kappa (float/array): Half-saturation constant
            epsilon (float/array): Noise intensity
            Xc (float/array): Critical threshold
            Xdisease (float/array/None): Optional disease threshold. When supplied,
                the simulation records each individual's first crossing time.
            n (int): Number of simulations
            tmin (float): Start time
            tmax (float): End time
            x0 (float/array): Initial condition - can be scalar (same for all) or array of length n
            dt (float): Time step
            save_times (float/array): Save interval or explicit save times
            dt_schedule (list, optional): Piecewise time-step schedule.
                Each item can be `(t_end, dt)` or `{'t_end': ..., 'dt': ...}`.
                Times are absolute in the simulation units.
            adaptive_dt (list, optional): Alias for `dt_schedule`.
            h_ext (float/callable/array/list/None): External hazard rate.
                - float: Constant hazard rate for all agents.
                - callable: Function h(t) returning hazard rate at time t for all agents.
                - np.ndarray: Array of length n specifying a constant hazard rate for each agent.
                - list of callables: List of length n with a specific h(t) for each agent.
                - None: No external hazard.
            units (str): Time units ('days' or 'years')
            parallel (bool): Enable parallel processing
            break_early (bool): Stop if all paths cross threshold
            drift_expr (str/expr): Custom drift expression
            drift_mode (str): How to apply custom drift
            extra_params (dict): Additional drift parameters
            start_idx (int): Starting index for parallel chunks
            random_seed (int, optional): Seed for NumPy random draws used by this simulation.
            use_fast_kernel (bool): Use numba-optimized kernel when possible (default True)
        """
        # Store raw parameters
        self.raw_eta = eta
        self.raw_beta = beta
        self.raw_kappa = kappa
        self.raw_epsilon = epsilon
        self.raw_Xc = Xc
        self.raw_Xdisease = Xdisease
        self.raw_x0 = x0  # Store raw x0 for chunking
        self.raw_dt = dt
        self.raw_dt_schedule = dt_schedule
        self.raw_adaptive_dt = adaptive_dt
        
        # Store other parameters
        self.n = n
        self.tmin = tmin
        self.tmax = tmax
        self.dt = float(dt)
        self.save_times = save_times if save_times is not None else self.dt
        self.use_fast_kernel = use_fast_kernel

        if dt_schedule is not None and adaptive_dt is not None:
            raise ValueError("Use only one of dt_schedule or adaptive_dt.")

        if not np.isscalar(dt):
            raise TypeError("dt must be a scalar float. Use dt_schedule/adaptive_dt for variable time steps.")

        self.dt_schedule = self._normalize_dt_schedule(
            dt_schedule if dt_schedule is not None else adaptive_dt
        )
        
        # Process x0 - can be scalar or array
        self.x0 = self._process_x0(x0)
        
        # Validate and store h_ext
        is_list_of_callables = isinstance(h_ext, list) and all(callable(f) for f in h_ext)
        if h_ext is not None and not isinstance(h_ext, (float, np.ndarray)) and not callable(h_ext) and not is_list_of_callables:
            raise TypeError("h_ext must be None, a float, a callable, a NumPy array, or a list of callables.")

        if isinstance(h_ext, np.ndarray):
            if h_ext.ndim != 1 or len(h_ext) != n:
                raise ValueError(f"If h_ext is an array, it must be 1D and have length n={n}, but got shape {h_ext.shape}")
        elif is_list_of_callables:
            if len(h_ext) != n:
                raise ValueError(f"If h_ext is a list of functions, it must have length n={n}, but got length {len(h_ext)}")

        self.h_ext = h_ext # Store h_ext directly

        self.units = units
        self.parallel = parallel
        self.break_early = break_early
        self.start_idx = start_idx
        self.random_seed = None if random_seed is None else int(random_seed)
        
        # Drift related - store only configuration, not the function
        self.drift_expr = drift_expr
        self.drift_mode = drift_mode
        self.extra_params = extra_params or {}
        
        # Process parameters for actual simulation
        self._process_all_parameters()
        self._build_time_grids()
        
        # Setup drift function after processing parameters
        self.drift_func = None  # Will be set up when needed

    @staticmethod
    def _round_time_array(times):
        """Round time arrays to avoid floating-point duplicate boundaries."""
        return np.round(np.asarray(times, dtype=np.float64), 12)

    @classmethod
    def _unique_sorted_times(cls, times):
        """Return sorted unique times after stable rounding."""
        rounded = cls._round_time_array(times)
        return np.unique(rounded)

    @classmethod
    def _build_uniform_time_grid(cls, start, end, dt):
        """Build a grid from start to end and always include the end time."""
        if dt <= 0:
            raise ValueError("dt must be positive.")

        if end <= start:
            return np.array([float(start)], dtype=np.float64)

        step_count = int(np.floor((end - start) / dt + 1e-12))
        times = start + np.arange(step_count + 1, dtype=np.float64) * dt
        times = times[times < end - 1e-12]
        times = np.concatenate((times, np.array([end], dtype=np.float64)))
        return cls._unique_sorted_times(times)

    @staticmethod
    def _parse_dt_schedule_item(item):
        """Parse a single schedule item."""
        if isinstance(item, dict):
            if 'dt' not in item:
                raise ValueError("Each dt schedule dict must include 'dt'.")

            if 't_end' in item:
                return float(item['t_end']), float(item['dt'])

            if 'until' in item:
                return float(item['until']), float(item['dt'])

            if 'tmax' in item:
                return float(item['tmax']), float(item['dt'])

            raise ValueError("Each dt schedule dict must include 't_end', 'until', or 'tmax'.")

        if isinstance(item, (tuple, list, np.ndarray)) and len(item) == 2:
            return float(item[0]), float(item[1])

        raise TypeError("Each dt schedule item must be a `(t_end, dt)` pair or a dict.")

    def _normalize_dt_schedule(self, dt_schedule):
        """Validate and normalize the adaptive time-step schedule."""
        if dt_schedule is None:
            return None

        normalized = []
        last_end = self.tmin

        for item in dt_schedule:
            t_end, dt_value = self._parse_dt_schedule_item(item)

            if dt_value <= 0:
                raise ValueError("Each adaptive dt value must be positive.")

            if t_end <= last_end:
                raise ValueError("Adaptive dt schedule times must be strictly increasing.")

            normalized.append((t_end, dt_value))
            last_end = t_end

        if not normalized:
            raise ValueError("Adaptive dt schedule cannot be empty.")

        return normalized

    def _normalize_save_times(self):
        """Convert save_times into an explicit array of save points."""
        if np.isscalar(self.save_times):
            save_dt = float(self.save_times)

            if save_dt <= 0:
                raise ValueError("save_times must be positive.")

            save_tspan = np.arange(
                self.tmin,
                self.tmax + 0.000001,
                save_dt,
                dtype=np.float64,
            )

            if save_tspan.size == 0 or abs(save_tspan[0] - self.tmin) > 1e-12:
                save_tspan = np.insert(save_tspan, 0, self.tmin)

            return self._unique_sorted_times(save_tspan)

        save_tspan = np.asarray(self.save_times, dtype=np.float64).ravel()

        if save_tspan.size == 0:
            return np.array([self.tmin], dtype=np.float64)

        save_tspan = save_tspan[np.isfinite(save_tspan)]
        save_tspan = save_tspan[(save_tspan >= self.tmin) & (save_tspan <= self.tmax)]
        save_tspan = np.concatenate((np.array([self.tmin], dtype=np.float64), save_tspan))
        return self._unique_sorted_times(save_tspan)

    def _build_simulation_time_grid(self):
        """Build the simulation step grid from dt or the adaptive schedule."""
        if self.dt_schedule is None:
            return self._build_uniform_time_grid(self.tmin, self.tmax, self.dt)

        full_grid = [self.tmin]
        current_time = self.tmin
        last_dt = self.dt

        for t_end, dt_value in self.dt_schedule:
            if current_time >= self.tmax - 1e-12:
                break

            capped_end = min(t_end, self.tmax)

            if capped_end <= current_time + 1e-12:
                continue

            segment_grid = self._build_uniform_time_grid(current_time, capped_end, dt_value)
            full_grid.extend(segment_grid[1:])
            current_time = capped_end
            last_dt = dt_value

        if current_time < self.tmax - 1e-12:
            tail_grid = self._build_uniform_time_grid(current_time, self.tmax, last_dt)
            full_grid.extend(tail_grid[1:])

        return self._unique_sorted_times(full_grid)

    def _build_time_grids(self):
        """Build the canonical step grid and save grid for the simulation."""
        save_tspan = self._normalize_save_times()
        simulation_grid = self._build_simulation_time_grid()
        combined_grid = self._unique_sorted_times(
            np.concatenate((simulation_grid, save_tspan, np.array([self.tmin, self.tmax])))
        )

        self.time_grid = combined_grid
        self.step_dts = self._round_time_array(np.diff(self.time_grid))
        self.save_tspan = save_tspan
        self.save_indices = np.searchsorted(self.time_grid, self.save_tspan)
        self.has_variable_dt = (
            self.step_dts.size > 1
            and not np.allclose(self.step_dts, self.step_dts[0], rtol=1e-12, atol=1e-12)
        )
        self.dt_min = float(np.min(self.step_dts)) if self.step_dts.size > 0 else float(self.dt)
        self.dt_max = float(np.max(self.step_dts)) if self.step_dts.size > 0 else float(self.dt)

    def _process_all_parameters(self):
        """Process all raw parameters into arrays of appropriate length."""
        self.eta = self._process_parameter(self.raw_eta)
        self.beta = self._process_parameter(self.raw_beta)
        self.kappa = self._process_parameter(self.raw_kappa)
        self.epsilon = self._process_parameter(self.raw_epsilon)
        self.Xc = self._process_parameter(self.raw_Xc)
        self.Xdisease = None
        if self.raw_Xdisease is not None:
            self.Xdisease = self._process_parameter(self.raw_Xdisease)
        self._calc_derived_params()
    def _process_parameter(self, param):
        """
        Convert parameter to array of appropriate length.

        Args:
            param: Scalar or array parameter value

        Returns:
            np.ndarray: Parameter array of length self.n

        Raises:
            ValueError: If parameter can't be converted to correct length
        """
        if np.isscalar(param) or np.size(param) <= 1:
            return np.full(self.n, param, dtype=np.float64)
        elif len(param) == self.n:
            return np.asarray(param, dtype=np.float64)
        else:
            raise ValueError(f"{param} must be a scalar or an array of length {self.n} but is sized {len(param)}")

    def _process_x0(self, x0):
        """
        Process initial condition x0 - can be scalar or array of length n.
        
        Args:
            x0: Scalar or array initial condition
            
        Returns:
            np.ndarray: x0 array of length n
        """
        if np.isscalar(x0) or np.size(x0) <= 1:
            return np.full(self.n, float(np.atleast_1d(x0)[0]), dtype=np.float64)
        elif len(x0) == self.n:
            return np.asarray(x0, dtype=np.float64)
        else:
            raise ValueError(f"x0 must be a scalar or an array of length {self.n} but is sized {len(x0)}")

    def _calc_derived_params(self):
        """
        Calculate derived parameters used in analysis.
        
        Computes various combinations of basic parameters that are useful
        for analyzing the system behavior, including characteristic times
        and dimensionless parameters.
        """
        self.p1 = self.beta * self.kappa / self.epsilon
        self.p2 = self.eta * self.kappa / (self.beta**2)
        self.p3 = self.Xc / self.kappa
        self.alpha = self.eta * (self.kappa + self.Xc) / self.epsilon
        self.tau = self.beta / self.eta
        self.t_r = self.kappa / self.beta
        self.t_D = self.kappa**2 / self.epsilon
        self.t_p = np.sqrt(self.kappa / self.eta)

    @staticmethod
    @njit
    def _default_drift(X, tcur, eta, beta, kappa):
        """
        Default SR model drift function.

        Args:
            X (np.ndarray): Current state
            tcur (float): Current time
            eta (np.ndarray): Production rate
            beta (np.ndarray): Removal rate
            kappa (np.ndarray): Half-saturation constant

        Returns:
            np.ndarray: Drift term for SR model
        """
        return eta * tcur - X * (beta / (X + kappa))

    def _setup_drift(self):
        """
        Set up drift function based on configuration.
        
        Creates appropriate drift function based on drift_expr and drift_mode.
        Handles custom expressions, parameter substitution, and compilation.

        Returns:
            callable: Configured drift function
        """
        if self.drift_expr is None:
            return self._default_drift
        
        try:
            # Define symbols with explicit Symbol constructor to avoid conflicts
            X = symbols('X')
            t = symbols('t')
            eta = symbols('eta', real=True)  # real=True tells SymPy these are not functions
            beta = symbols('beta', real=True)
            kappa = symbols('kappa', real=True)
            basic_params = {'X': X, 't': t, 'eta': eta, 'beta': beta, 'kappa': kappa}
            
            
            # Add extra parameters if provided
            extra_syms = {}
            if self.extra_params:
                extra_syms = {k: symbols(k) for k in self.extra_params.keys()}
                basic_params.update(extra_syms)
            
            # Parse the expression
            if isinstance(self.drift_expr, str):
                expr = sympify(self.drift_expr, locals={'beta': beta})
            else:
                expr = self.drift_expr
                
            # If mode is 'add', add to default drift
            if self.drift_mode == 'add':
                default_expr = eta * t - X * (beta / (X + kappa))
                expr = default_expr + expr
            
            # Create a lambda function
            drift_lambda = lambdify(tuple(basic_params.values()), expr)
            
            # Return a wrapper function that matches the expected signature but can use extra params
            def drift_wrapper(X, tcur, eta, beta, kappa, **extra):
                params = {
                    'X': X, 't': tcur,
                    'eta': eta, 
                    'beta': beta, 
                    'kappa': kappa
                }
                params.update(extra)
                return drift_lambda(**params)
                
            return drift_wrapper
            
        except Exception as e:
            print(f"Error setting up custom drift: {e}")
            print("Falling back to default drift")
            return self._default_drift

    def get_drift(self, X, tcur):
        """
        Calculate drift at current state and time.

        Args:
            X (np.ndarray): Current state
            tcur (float): Current time

        Returns:
            np.ndarray: Drift values
        """
        # Setup drift function if not already set up
        if self.drift_func is None:
            self.drift_func = self._setup_drift()
            
        params = {
            'X': X,
            'tcur': tcur,
            'eta': self.eta,
            'beta': self.beta,
            'kappa': self.kappa
        }
        if self.extra_params:
            params.update(self.extra_params)
        return self.drift_func(**params)

    def create_chunk_params(self, start_idx, chunk_size, chunk_idx=0):
        """
        Create parameter object for parallel chunk processing.

        Args:
            start_idx (int): Starting index for this chunk
            chunk_size (int): Number of paths in chunk

        Returns:
            SimulationParams: New parameter object for chunk
        """
        # Slice the raw parameters if they are arrays of length n
        def slice_param(param, n_total):
            if isinstance(param, (np.ndarray, list)):
                if len(param) == n_total: # Only slice if it's an agent-specific array or list
                    return param[start_idx:start_idx + chunk_size]
            # Otherwise (scalar, single callable, or already chunked), return as is
            return param

        # Pre-slice all raw parameters
        raw_eta_chunk = slice_param(self.raw_eta, self.n)
        raw_beta_chunk = slice_param(self.raw_beta, self.n)
        raw_kappa_chunk = slice_param(self.raw_kappa, self.n)
        raw_epsilon_chunk = slice_param(self.raw_epsilon, self.n)
        raw_Xc_chunk = slice_param(self.raw_Xc, self.n)
        raw_Xdisease_chunk = slice_param(self.raw_Xdisease, self.n)
        # Handle x0 - slice if it's an array
        x0_chunk = slice_param(self.raw_x0, self.n)
        # Handle h_ext separately as it can be callable or float
        h_ext_chunk = slice_param(self.h_ext, self.n) # Slice only if it's an agent-specific array


        chunk_params = SimulationParams(
            eta=raw_eta_chunk,
            beta=raw_beta_chunk,
            kappa=raw_kappa_chunk,
            epsilon=raw_epsilon_chunk,
            Xc=raw_Xc_chunk,
            Xdisease=raw_Xdisease_chunk,
            n=chunk_size,
            tmin=self.tmin,
            tmax=self.tmax,
            x0=x0_chunk,
            dt=self.raw_dt,
            save_times=self.save_times,
            dt_schedule=self.dt_schedule,
            h_ext=h_ext_chunk, # Pass the (potentially sliced) h_ext
            break_early=self.break_early,
            drift_expr=self.drift_expr,
            drift_mode=self.drift_mode,
            extra_params=self.extra_params,
            start_idx=start_idx,
            random_seed=self._derive_chunk_seed(chunk_idx),
            use_fast_kernel=self.use_fast_kernel
        )
        return chunk_params

    def _derive_chunk_seed(self, chunk_idx):
        if self.random_seed is None:
            return None

        return int(self.random_seed) + int(chunk_idx)
        
    def print_params(self):
        """
        Print a concise summary of core simulation parameters.
        """
        eta_val = np.mean(self.eta)
        beta_val = np.mean(self.beta)
        kappa_val = np.mean(self.kappa)
        epsilon_val = np.mean(self.epsilon)
        xc_val = np.mean(self.Xc)

        print_str = (
            f"η = {eta_val:.5f} {self.units}⁻², "
            f"β = {beta_val:.2f} {self.units}⁻¹, "
            f"κ = {kappa_val:.1f}, "
            f"ε = {epsilon_val:.3f} day⁻¹, "
            f"Xc = {int(round(xc_val))}"
        )
        print(print_str)

    def print_full_params_summary(self):
        """
        Print a detailed summary of all simulation parameters.
        
        For array parameters, prints mean and standard deviation if the array
        has more than one unique value.
        """
        print("=== Simulation Parameters ===")
        
        # Core SR model parameters
        print("\nCore SR Model Parameters:")
        for param_name, param_array in [
            ("eta", self.eta),
            ("beta", self.beta),
            ("kappa", self.kappa),
            ("epsilon", self.epsilon),
            ("Xc", self.Xc)
        ]:
            if len(np.unique(param_array)) == 1:
                # Single value parameter
                print(f"  {param_name} = {param_array[0]:.6g}")
            else:
                # Array parameter with multiple values
                mean_val = np.mean(param_array)
                std_val = np.std(param_array)
                min_val = np.min(param_array)
                max_val = np.max(param_array)
                print(f"  {param_name} = {mean_val:.6g} ± {std_val:.6g} (mean ± std)")
                print(f"    range: [{min_val:.6g}, {max_val:.6g}]")
        
        # Derived parameters
        print("\nDerived Parameters:")
        for param_name, param_array in [
            ("p1", self.p1),
            ("p2", self.p2),
            ("p3", self.p3),
            ("alpha", self.alpha),
            ("tau", self.tau),
            ("t_r", self.t_r),
            ("t_D", self.t_D),
            ("t_p", self.t_p)
        ]:
            if len(np.unique(param_array)) == 1:
                print(f"  {param_name} = {param_array[0]:.6g}")
            else:
                mean_val = np.mean(param_array)
                std_val = np.std(param_array)
                print(f"  {param_name} = {mean_val:.6g} ± {std_val:.6g}")
        
        # Simulation settings
        print("\nSimulation Settings:")
        print(f"  Number of simulations (n) = {self.n}")
        print(f"  Time range = [{self.tmin}, {self.tmax}] {self.units}")
        if self.dt_schedule is None:
            print(f"  Time step (dt) = {self.dt} {self.units}")
        else:
            print(f"  Adaptive dt: min={self.dt_min}, max={self.dt_max} {self.units}")
            print(f"  Schedule = {self.dt_schedule}")
        print(f"  Save interval = {self.save_times} {self.units}")
        print(f"  Initial condition (x0) = {self.x0}")
        
        # External hazard
        print("\nExternal Hazard:")
        if self.h_ext is None:
            print("  None")
        elif isinstance(self.h_ext, float):
            print(f"  Constant rate: {self.h_ext}")
        elif isinstance(self.h_ext, np.ndarray):
            mean_h = np.mean(self.h_ext)
            std_h = np.std(self.h_ext)
            if std_h < 1e-10:  # Effectively constant
                print(f"  Constant rate: {mean_h:.6g}")
            else:
                print(f"  Variable rate: {mean_h:.6g} ± {std_h:.6g}")
        elif isinstance(self.h_ext, list):
            print("  List of agent-specific hazard functions")
        else:
            print("  Time-dependent function")
        
        # Other settings
        print("\nOther Settings:")
        print(f"  Parallel processing: {self.parallel}")
        print(f"  Random seed: {self.random_seed}")
        print(f"  Break early: {self.break_early}")
        
        # Custom drift
        print("\nDrift Function:")
        if self.drift_expr is None:
            print("  Default SR model drift")
        else:
            print(f"  Custom drift ({self.drift_mode} mode)")
            print(f"  Expression: {self.drift_expr}")
            if self.extra_params:
                print("  Extra parameters:")
                for k, v in self.extra_params.items():
                    print(f"    {k} = {v}")

class SR_sim:
    """
    Saturated Removal (SR) simulation class.
    
    Handles simulation of SR paths, including parallel processing,
    path generation, and statistical analysis of results.
    """
    
    def __init__(self, params):
        """
        Initialize SR simulation.

        Args:
            params (SimulationParams): Simulation parameters
        """
        self.params = params
        self.paths = None
        self.death_times = None
        self.disease_times = None
        self.tspan = None
        self.alive_mask = None
        self.extrinsic_deaths = None  # New attribute
        self.death_counts = None
        self.at_risk = None
        self.interval_hazard = None
        self.tspan_interval_hazard = None
        self.interval_dt = None
        self.run_simulation()
        self._post_simulation_calculations()

    def run_simulation(self):
        if self.params.parallel:
            (
                self.death_times,
                self.paths,
                self.tspan,
                self.alive_mask,
                self.extrinsic_deaths,
                self.disease_times,
            ) = self._create_paths_parallel()
        else:
            (
                self.death_times,
                self.paths,
                self.tspan,
                self.alive_mask,
                self.extrinsic_deaths,
                self.disease_times,
            ) = self._create_paths()

    def _reset_rng(self):
        if self.params.random_seed is None:
            return

        seed = int(self.params.random_seed)
        np.random.seed(seed)
        _seed_numba_rng(seed)

    def _create_paths(self):
        """
        Generate simulation paths for serial processing.

        Returns:
            tuple: (death_times, paths, tspan_save_times, alive_mask, extrinsic_deaths)
        """
        self._reset_rng()
        tspan = self.params.time_grid
        tspan_save_times = self.params.save_tspan
        paths = np.ones((self.params.n, len(tspan_save_times)))
        paths[:, 0] = self.params.x0
        
        return self._simulate_paths(paths, tspan, tspan_save_times)

    def _create_paths_parallel(self):
        """
        Generate simulation paths using parallel processing.

        Returns:
            tuple: (death_times, paths, tspan, alive_mask, extrinsic_deaths)
        """
        num_cores = min(cpu_count(), self.params.n)
        base_chunk_size = self.params.n // num_cores
        remainder = self.params.n % num_cores

        chunk_params = []
        start_idx = 0

        for chunk_idx in range(num_cores):
            chunk_size = base_chunk_size

            if chunk_idx < remainder:
                chunk_size += 1

            if chunk_size <= 0:
                continue

            chunk_params.append(
                self.params.create_chunk_params(
                    start_idx,
                    chunk_size,
                    chunk_idx=chunk_idx,
                )
            )
            start_idx += chunk_size
        
        with Pool(num_cores) as pool:
            results = pool.map(self._run_chunk, chunk_params)
        
        return self._merge_chunks(results)

    @staticmethod
    def _run_chunk(params):
        """
        Run simulation for a chunk of parameters.

        Args:
            params (SimulationParams): Parameters for the chunk

        Returns:
            tuple: (start_idx, death_times, paths, alive_mask, extrinsic_deaths)
        """
        # Ensure drift function is set up in the child process
        params.drift_func = params._setup_drift()
        sim = SR_sim(params)
        return (
            params.start_idx,
            sim.death_times,
            sim.paths,
            sim.alive_mask,
            sim.extrinsic_deaths,
            sim.disease_times,
        )

    def _merge_chunks(self, results):
        """
        Merge results from parallel chunks while preserving vector structure.

        Args:
            results (list): List of results from parallel chunks

        Returns:
            tuple: (death_times, paths, tspan, alive_mask, extrinsic_deaths)
        """
        # Determine maximum time steps across all chunks
        max_time_steps = max(chunk_paths.shape[1] for _, _, chunk_paths, _, _, _ in results)
        
        # Initialize arrays for merged results
        death_times = np.full(self.params.n, np.inf)
        paths = np.zeros((self.params.n, max_time_steps))
        alive_mask = np.zeros(self.params.n, dtype=bool)
        extrinsic_deaths = np.zeros(self.params.n, dtype=int)
        track_disease = any(chunk_disease is not None for _, _, _, _, _, chunk_disease in results)
        disease_times = None
        if track_disease:
            disease_times = np.full(self.params.n, np.inf)
        
        # Place each chunk's results in the correct position
        for start_idx, chunk_death_times, chunk_paths, chunk_alive_mask, chunk_extrinsic, chunk_disease in results:
            end_idx = start_idx + len(chunk_death_times)
            paths[start_idx:end_idx, :chunk_paths.shape[1]] = chunk_paths
            death_times[start_idx:end_idx] = chunk_death_times
            alive_mask[start_idx:end_idx] = chunk_alive_mask
            extrinsic_deaths[start_idx:end_idx] = chunk_extrinsic
            if disease_times is not None and chunk_disease is not None:
                disease_times[start_idx:end_idx] = chunk_disease
        
        # Create corresponding time span
        tspan = self.params.save_tspan[:max_time_steps]
        
        # Ensure death times don't exceed simulation time
        death_times = np.minimum(death_times, self.params.tmax)
        
        return death_times, paths, tspan, alive_mask, extrinsic_deaths, disease_times

    def _skorokhod_step(self, X, tcur, dt=None):
        """
        Perform one step of Skorokhod simulation.

        Implements reflecting boundary condition at zero and
        combines drift and noise terms.

        Args:
            X (np.ndarray): Current state
            tcur (float): Current time

        Returns:
            np.ndarray: Next state
        """
        if dt is None:
            dt = self.params.dt

        noise = np.sqrt(2 * self.params.epsilon) * (X > 0)
        Y = noise * np.sqrt(dt) * np.random.standard_normal(X.shape)
        U = np.random.random(X.shape)
        M = (Y + np.sqrt(Y**2 - 2 * dt * np.log(U))) / 2
        drift = self.params.get_drift(X, tcur)
        delta_X = dt * drift
        return np.maximum(M-Y, X + delta_X - Y)

    @staticmethod
    @njit
    def _find_first_crossings_numba(X, X_prev, Xc, p_death_ext, eps_dt):
        """
        Identifies crossings and potential extrinsic deaths for a subset of agents.
        Includes Brownian bridge correction for mid-step crossings.
        Numba-jitted for performance.

        Args:
            X (np.ndarray): Current states for the subset of agents being checked.
            X_prev (np.ndarray): Previous states for the subset of agents being checked.
            Xc (float/np.ndarray): Threshold values (scalar or subset array).
            p_death_ext (np.ndarray): Probability of extrinsic death for each agent in this subset.
            eps_dt (np.ndarray): epsilon * dt for each agent (for Brownian bridge).

        Returns:
            tuple: (died_mask, extrinsic_mask)
                   Boolean masks relative to the input subset X.
        """
        # Handle empty input case (if no agents were alive)
        if X.size == 0:
            return np.array([False]), np.array([False]) # Return dummy non-empty boolean arrays

        n = X.shape[0]
        crossed = (X > Xc) # Check crossing against threshold
        extrinsic_cause = np.zeros(n, dtype=np.bool_)
        bridge_crossed = np.zeros(n, dtype=np.bool_)
        
        # Brownian bridge crossing check for particles that didn't cross directly
        for i in range(n):
            if not crossed[i] and X_prev[i] < Xc[i]:
                gap_prev = Xc[i] - X_prev[i]
                gap_cur = Xc[i] - X[i]
                if eps_dt[i] > 0.0:
                    p_cross = np.exp(-(gap_prev * gap_cur) / eps_dt[i])
                    if np.random.random() < p_cross:
                        bridge_crossed[i] = True
        
        died_mask = crossed | bridge_crossed  # Intrinsic deaths (direct + bridge)

        # Determine if any extrinsic death check is needed for this batch
        check_extrinsic = False
        if p_death_ext.size > 0 and np.any(p_death_ext > 0.0):
             check_extrinsic = True

        if check_extrinsic:
            random_nums = np.random.random(n)
            external_death_event = (random_nums < p_death_ext)
            died_mask = died_mask | external_death_event
            # Only mark as extrinsic if it wasn't already an intrinsic death
            for i in range(n):
                if external_death_event[i] and not crossed[i] and not bridge_crossed[i]:
                    extrinsic_cause[i] = True

        # Return masks relative to the input subset X
        return died_mask, extrinsic_cause

    def _can_use_fast_kernel(self):
        """
        Check if the fast numba kernel can be used.
        
        The fast kernel can only be used when:
        - No custom drift expression
        - h_ext is None, float, or array (not callable or list of callables)
        - use_fast_kernel flag is True
        """
        if not self.params.use_fast_kernel:
            return False
        if self.params.drift_expr is not None:
            return False
        if callable(self.params.h_ext) and not isinstance(self.params.h_ext, np.ndarray):
            return False
        if isinstance(self.params.h_ext, list):
            return False
        return True

    def _simulate_paths(self, paths, tspan, tspan_save_times):
        """
        Core simulation loop for generating paths.
        
        Uses optimized numba kernel when possible, falls back to Python loop otherwise.

        Args:
            paths (np.ndarray): Array to store paths
            tspan (np.ndarray): Time points
            tspan_save_times (np.ndarray): Times to save results

        Returns:
            tuple: (death_times, paths, tspan_save_times, alive_mask, extrinsic_deaths)
        """
        if self.params.h_ext is None:
            h_ext_values = np.zeros(self.params.n, dtype=np.float64)
        elif isinstance(self.params.h_ext, float):
            h_ext_values = np.full(self.params.n, self.params.h_ext, dtype=np.float64)
        elif isinstance(self.params.h_ext, np.ndarray):
            h_ext_values = np.asarray(self.params.h_ext, dtype=np.float64)
        else:
            h_ext_values = None

        # Try to use fast kernel
        if self._can_use_fast_kernel():
            return self._simulate_paths_fast_wrapper(paths, tspan, tspan_save_times, h_ext_values)

        return self._simulate_paths_python(paths, tspan, tspan_save_times, h_ext_values)

    def _simulate_paths_fast_wrapper(self, paths, tspan, tspan_save_times, h_ext_values):
        """
        Wrapper to call the fast numba kernel.
        """
        n = self.params.n
        
        # Pre-compute sqrt(2*epsilon) for efficiency
        sqrt_2eps = np.sqrt(2.0 * self.params.epsilon)
        
        # Call the fast kernel
        Xdisease = np.empty(0, dtype=np.float64)
        disease_threshold = getattr(self.params, "Xdisease", None)
        track_disease = disease_threshold is not None
        if track_disease:
            Xdisease = disease_threshold

        death_times, paths_out, actual_save_points, extrinsic_deaths, disease_times = _simulate_paths_fast(
            n,
            tspan,
            self.params.save_indices,
            self.params.x0, self.params.eta, self.params.beta, self.params.kappa,
            sqrt_2eps,
            self.params.Xc,
            Xdisease,
            h_ext_values,
            self.params.epsilon,
            self.params.break_early,
            track_disease,
        )
        
        # Truncate if needed
        if actual_save_points < len(tspan_save_times):
            paths_out = paths_out[:, :actual_save_points]
            tspan_save_times = tspan_save_times[:actual_save_points]
        
        final_alive_mask = (death_times == np.inf)
        self.extrinsic_deaths = extrinsic_deaths
        if not track_disease:
            disease_times = None
        
        return death_times, paths_out, tspan_save_times, final_alive_mask, extrinsic_deaths, disease_times

    def _simulate_paths_python(self, paths, tspan, tspan_save_times, h_ext_values):
        """
        Python fallback for simulation (used when custom drift or callable h_ext).
        
        Optimized with pre-computed arrays and reduced allocations.
        """
        death_times = np.full(self.params.n, np.inf)
        self.extrinsic_deaths = np.zeros(self.params.n, dtype=np.int32)
        track_disease = self.params.Xdisease is not None
        disease_times = None
        if track_disease:
            disease_times = np.full(self.params.n, np.inf)
        
        X = self.params.x0.copy()
        X_prev = self.params.x0.copy()  # Track previous state for Brownian bridge
        paths[:, 0] = X
        
        # Pre-compute constants
        sqrt_2eps = np.sqrt(2.0 * self.params.epsilon)
        
        # Ensure drift_func is initialized
        _ = self.params.get_drift(X, tspan[0])
        
        # Track last processed time index for final truncation
        last_i_t = len(tspan) - 1
        next_save_ptr = 1

        for i_t, tcur in enumerate(tspan[1:], 1):
            last_i_t = i_t
            dt = tspan[i_t] - tspan[i_t - 1]
            sqrt_dt = np.sqrt(dt)
            
            # Early break: if all particles have died
            if self.params.break_early and np.all(death_times != np.inf):
                paths = paths[:, :next_save_ptr]
                tspan_save_times = tspan_save_times[:next_save_ptr]
                break

            # Save previous state for Brownian bridge
            X_prev = X.copy()
            
            # Skorokhod step - vectorized
            noise_mask = (X > 0).astype(np.float64)
            noise_term = sqrt_2eps * noise_mask
            Y_sk = noise_term * sqrt_dt * np.random.standard_normal(X.shape)
            U_sk = np.random.random(X.shape)
            Y_sq_sk = Y_sk * Y_sk
            log_U_term_sk = -2.0 * dt * np.log(np.maximum(U_sk, 1e-100))
            inside_sqrt_sk = Y_sq_sk + log_U_term_sk
            M_sk = (Y_sk + np.sqrt(np.maximum(inside_sqrt_sk, 0.0))) / 2.0

            # Drift calculation
            drift_params = {
                 'X': X, 'tcur': tcur,
                 'eta': self.params.eta, 
                 'beta': self.params.beta, 
                 'kappa': self.params.kappa
            }
            if self.params.extra_params:
                 drift_params.update(self.params.extra_params)
            
            drift = self.params.drift_func(**drift_params)
            delta_X = dt * drift
            
            X = np.maximum(M_sk - Y_sk, X + delta_X - Y_sk)

            # Save state if needed
            if next_save_ptr < len(self.params.save_indices) and i_t == self.params.save_indices[next_save_ptr]:
                paths[:, next_save_ptr] = X
                next_save_ptr += 1

            if track_disease:
                disease_candidates = np.where(
                    (death_times == np.inf) & (disease_times == np.inf)
                )[0]
                if disease_candidates.size > 0:
                    disease_mask, _ = self._find_first_crossings_numba(
                        X[disease_candidates],
                        X_prev[disease_candidates],
                        self.params.Xdisease[disease_candidates],
                        np.zeros(disease_candidates.size, dtype=np.float64),
                        self.params.epsilon[disease_candidates] * dt,
                    )
                    disease_times[disease_candidates[disease_mask]] = tcur

            # Check for crossings
            not_yet_crossed_mask = (death_times == np.inf)
            agents_to_check = np.where(not_yet_crossed_mask)[0]

            if agents_to_check.size > 0:
                X_check = X[agents_to_check]
                Xc_check = self.params.Xc[agents_to_check]

                # Get extrinsic death probabilities
                if h_ext_values is not None:
                    p_death_ext_check = 1.0 - np.exp(-h_ext_values[agents_to_check] * dt)
                elif callable(self.params.h_ext):
                    h_val = self.params.h_ext(tcur)
                    p_death_ext_check = np.full(agents_to_check.size, 
                                                 1.0 - np.exp(-h_val * dt), dtype=np.float64)
                elif isinstance(self.params.h_ext, list):
                    h_vals = np.array([self.params.h_ext[i](tcur) for i in agents_to_check])
                    p_death_ext_check = 1.0 - np.exp(-h_vals * dt)
                else:
                    p_death_ext_check = np.zeros(agents_to_check.size, dtype=np.float64)

                X_prev_check = X_prev[agents_to_check]
                eps_dt_check = self.params.epsilon[agents_to_check] * dt
                
                died_mask, extrinsic_mask = self._find_first_crossings_numba(
                    X_check, X_prev_check, Xc_check, p_death_ext_check, eps_dt_check
                )

                if X_check.size > 0:
                    died_indices = agents_to_check[died_mask]
                    death_times[died_indices] = tcur
                    
                    extrinsic_died_indices = agents_to_check[died_mask & extrinsic_mask]
                    self.extrinsic_deaths[extrinsic_died_indices] = 1

                    if track_disease:
                        intrinsic_death_mask = died_mask & ~extrinsic_mask
                        intrinsic_died_indices = agents_to_check[intrinsic_death_mask]
                        missed_disease = intrinsic_died_indices[
                            disease_times[intrinsic_died_indices] == np.inf
                        ]
                        crossed_after_disease = (
                            self.params.Xdisease[missed_disease]
                            <= self.params.Xc[missed_disease]
                        )
                        disease_times[missed_disease[crossed_after_disease]] = tcur
            
        final_alive_mask = (death_times == np.inf)
        
        # Truncate paths if not broken early
        if not (self.params.break_early and np.all(death_times != np.inf)):
            paths = paths[:, :next_save_ptr]
            tspan_save_times = tspan_save_times[:next_save_ptr]

        return death_times, paths, tspan_save_times, final_alive_mask, self.extrinsic_deaths, disease_times

### post-simulation processing ###

    def _post_simulation_calculations(self):
        """
        Calculate statistical measures after simulation.
        
        Computes survival curves, hazard rates, and various
        statistical measures of the simulation results.
        """
        self.survival, self.tspan_survival, self.kmf = self._create_survival()
        self.naf = self._create_naf()
        self.hazard, self.tspan_hazard = self._calc_hazard()
        (
            self.interval_hazard,
            self.tspan_interval_hazard,
            self.death_counts,
            self.at_risk,
            self.interval_dt,
        ) = self._calc_interval_hazard()
        self.median_t = self.kmf.median_survival_time_
        self.mean_X = self._calc_mean_X()
        self.mean_X_analytical = self._calc_mean_X_analytical()
        self.cv_X = self._calc_cv_X()
        self.std_X = self._calc_std_X()
        self.steepness = self.calc_steepness()

    def _create_survival(self):
        """
        Create Kaplan-Meier survival curve from simulation results.

        Returns:
            tuple: (survival_function, timeline, kmf_object)
        """
        # event_observed is True for individuals who died (intrinsic or extrinsic)
        # and False for individuals who survived until tmax (censored).
        event_observed = ~self.alive_mask

        # Create a copy of death_times to modify for censored individuals
        # Individuals who survived until tmax have death_times == np.inf.
        # For Kaplan-Meier fitting, censored times should be the time of censoring (tmax).
        censored_death_times = np.copy(self.death_times)

        # Replace np.inf with tmax for censored individuals (where alive_mask is True)
        # Note: self.alive_mask is True where death_times was np.inf after simulation.
        # This corresponds to event_observed being False.
        censored_death_times[self.alive_mask] = self.params.tmax

        kmf = KaplanMeierFitter()

        # Fit the KMF using the modified death_times (with tmax for censored)
        # and the event_observed mask.
        kmf.fit(censored_death_times, event_observed=event_observed)

        return kmf.survival_function_, kmf.timeline, kmf

    def _create_naf(self, timeline=None):
        """
        Create Nelson-Aalen cumulative hazard estimator.

        Returns:
            NelsonAalenFitter: Fitted object
        """
        event_observed = ~self.alive_mask
        naf = NelsonAalenFitter()
        
        # Create a copy of death_times to modify for censored individuals
        # Individuals who survived until tmax have death_times == np.inf.
        # For Nelson-Aalen fitting, censored times should be the time of censoring (tmax).
        censored_death_times = np.copy(self.death_times)
        
        # Replace np.inf with tmax for censored individuals (where alive_mask is True)
        # Note: self.alive_mask is True where death_times was np.inf after simulation.
        # This corresponds to event_observed being False.
        censored_death_times[self.alive_mask] = self.params.tmax
        
        if timeline is None:
            naf.fit(censored_death_times, event_observed=event_observed)
        else:
            naf.fit(censored_death_times, event_observed=event_observed, timeline=timeline)
        return naf

    def _calc_hazard(self, timeline=None, bandwidth=3, truncate_boundary=True):
        """
        Calculate smoothed hazard function.
        
        The Nelson-Aalen smoothed hazard uses kernel density estimation which
        has boundary effects near tmax (artificial inflation). By default,
        we truncate the hazard to avoid this artifact.

        Args:
            timeline (np.ndarray, optional): Custom timeline for hazard calculation
            bandwidth (float): Kernel bandwidth for smoothing (default: 3)
            truncate_boundary (bool): If True, truncate hazard near tmax to avoid
                                      boundary effects (default: True)

        Returns:
            tuple: (hazard_values, hazard_times)
        """
        if timeline is None:
            sm_haz_table = self.naf.smoothed_hazard_(bandwidth=bandwidth)
            tspan_hazard = sm_haz_table.index.values
            hazard = sm_haz_table.values
        else:
            temp_naf = self._create_naf(timeline)
            sm_haz_table = temp_naf.smoothed_hazard_(bandwidth=bandwidth)
            tspan_hazard = sm_haz_table.index.values
            hazard = sm_haz_table.values
        
        # Truncate to avoid boundary effects near tmax
        # The kernel smoothing has edge effects within ~3*bandwidth of the boundary
        if truncate_boundary:
            # Be aggressive: cut off at tmax - 3*bandwidth to fully avoid edge effects
            cutoff = self.params.tmax - 3 * bandwidth
            
            # Apply truncation
            valid_mask = tspan_hazard <= cutoff
            if np.any(valid_mask):
                tspan_hazard = tspan_hazard[valid_mask]
                hazard = hazard[valid_mask]
        
        return hazard, tspan_hazard

    def _calc_interval_hazard(self):
        """
        Calculate a stepwise hazard directly from death counts on the simulation grid.

        This is the unsmoothed quantity that matches the simulation grid:
        deaths during an interval divided by those at risk at the start of
        that interval. The stored hazard uses the survival-ratio form
        `-log(S_{i+1} / S_i) / interval_dt`.
        """
        timeline = self.params.time_grid
        interval_ends = timeline[1:]
        interval_dt = self.params.step_dts
        death_counts = np.zeros(interval_dt.size, dtype=np.int64)

        finite_death_times = self.death_times[np.isfinite(self.death_times)]

        if finite_death_times.size > 0:
            death_steps = np.searchsorted(interval_ends, finite_death_times, side='left')
            valid_steps = (death_steps >= 0) & (death_steps < death_counts.size)
            np.add.at(death_counts, death_steps[valid_steps], 1)

        if death_counts.size == 0:
            empty_float = np.array([], dtype=np.float64)
            empty_int = np.array([], dtype=np.int64)
            return empty_float, empty_float, empty_int, empty_int, empty_float

        deaths_before_step = np.concatenate((
            np.array([0], dtype=np.int64),
            np.cumsum(death_counts[:-1], dtype=np.int64),
        ))
        at_risk = self.params.n - deaths_before_step
        interval_deaths = death_counts
        valid = at_risk > 0

        if not np.any(valid):
            empty_float = np.array([], dtype=np.float64)
            empty_int = np.array([], dtype=np.int64)
            return empty_float, empty_float, empty_int, empty_int, empty_float

        interval_survival = 1.0 - (interval_deaths[valid] / at_risk[valid])
        interval_survival = np.clip(interval_survival, 0.0, 1.0)

        interval_hazard = np.full(interval_survival.shape, np.inf, dtype=np.float64)
        positive_survival = interval_survival > 0.0
        interval_hazard[positive_survival] = (
            -np.log(interval_survival[positive_survival]) / interval_dt[valid][positive_survival]
        )

        return (
            interval_hazard,
            interval_ends[valid],
            interval_deaths[valid],
            at_risk[valid],
            interval_dt[valid],
        )
    
    def get_hazard_cutoff(self, bandwidth=3):
        """Get the time cutoff used for hazard truncation."""
        return self.params.tmax - 3 * bandwidth

    def get_hazard_full(self, bandwidth=3):
        """
        Get the full (untruncated) hazard including boundary region.
        
        Use this if you need to see the full hazard curve including the
        region near tmax where boundary effects may occur.
        
        Args:
            bandwidth (float): Kernel bandwidth for smoothing
            
        Returns:
            tuple: (hazard_values, hazard_times)
        """
        return self._calc_hazard(bandwidth=bandwidth, truncate_boundary=False)

    def _calc_mean_X(self):
        """
        Calculate mean state over time.

        Returns:
            np.ndarray: Mean state values
        """
        means = np.mean(self.paths, axis=0)
        means[-1] = np.inf
        return means
    
    def _calc_mean_X_analytical(self,epsilon=False):
        """
        Calculate mean state over time analytically from SR model.

        Returns:
            np.ndarray: Mean state values
        """
        if epsilon:
            return (self.params.kappa[0]*self.params.eta[0]*self.tspan+self.params.epsilon[0]) / (self.params.beta[0] - self.params.eta[0]*self.tspan)
        else:
            return self.params.kappa[0]*self.params.eta[0]*self.tspan / (self.params.beta[0] - self.params.eta[0]*self.tspan)

    def _calc_std_X(self):
        """
        Calculate standard deviation of state over time.

        Returns:
            np.ndarray: Standard deviation values
        """
        return np.std(self.paths, axis=0)

    def _calc_cv_X(self):
        """
        Calculate coefficient of variation of state over time.

        Returns:
            np.ndarray: Coefficient of variation values
        """
        return self._calc_std_X() / self._calc_mean_X()

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
        
        # Replace np.inf with tmax for censored individuals
        # For conditional survival from time t, censored times should be (tmax - t)
        censored_death_times_from_t = np.copy(death_times_from_t)
        inf_mask = (death_times_from_t == np.inf)
        censored_death_times_from_t[inf_mask] = self.params.tmax
        
        event_times = np.maximum(0, censored_death_times_from_t - t)
        event_observed = (~inf_mask).astype(int)

        # Create and fit Kaplan-Meier Fitter
        kmf = KaplanMeierFitter()
        kmf.fit(event_times, event_observed=event_observed)
        return kmf

    def find_time_at_survival(self, S, from_t=None, relative = True):
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
            time_at_S = np.interp(S, survival_func.values[::-1], timeline[::-1])
            if not relative and from_t is not None:
                return from_t + time_at_S
            return time_at_S
        return None

    def calc_steepness(self, method = 'IQR',from_t=None, relative = True):
        """
        Calculate steepness of survival curve.
        
        Steepness is defined as median survival time divided by
        the time difference between 75% and 25% survival.

        Args:
            from_t (float, optional): Start time for conditional survival

        Returns:
            float: Steepness value, or None if insufficient mortality
        """
        if from_t is None:
            from_t = 0
        if method == 'IQR':
            t_25 = self.find_time_at_survival(0.25, from_t, relative)
            t_50 = self.find_time_at_survival(0.5, from_t, relative)
            t_75 = self.find_time_at_survival(0.75, from_t, relative)
            
            if all(t is not None for t in [t_25, t_50, t_75]) and t_75 != t_25:
                return -t_50 / (t_75 - t_25)
        elif method == 'CV':
            if from_t is None:
                from_t = 0
            filtered_death_times = self.death_times[(self.death_times >= from_t) & (self.death_times != np.inf)]
            if len(filtered_death_times) > 0:
                if relative:
                    filtered_death_times = filtered_death_times - from_t
                mean_time = np.mean(filtered_death_times)
                std_time = np.std(filtered_death_times)
                if mean_time > 0:
                    cv = std_time / mean_time
                    if cv > 0:
                        return 1 / cv
