# go_with_winners.py

import numpy as np
from numba import njit
from .simulation import SimulationParams


# ============================================================================
# NUMBA-OPTIMIZED KERNELS FOR GO-WITH-WINNERS
# ============================================================================

def _hazard_from_survival(prev_survival, next_survival, dt):
    """Convert one survival step into a discrete hazard."""
    if prev_survival <= 0.0:
        return np.inf

    if next_survival <= 0.0:
        return np.inf

    survival_ratio = next_survival / prev_survival

    if survival_ratio >= 1.0:
        return 0.0

    return -np.log(survival_ratio) / dt


def _initialize_group_weights(n, group_to_particles, n_groups):
    """
    Create normalized within-group weights and one true mass per group.

    Each active group starts with:
    - `group_masses[g] = number of particles in the group`
    - per-particle weights that sum to 1 within that group
    """
    weights = np.zeros(n, dtype=float)
    group_masses = np.zeros(n_groups, dtype=float)

    for group_id, group_indices in group_to_particles.items():
        group_size = len(group_indices)

        if group_size == 0:
            continue

        group_masses[group_id] = float(group_size)
        weights[group_indices] = 1.0 / group_size

    return weights, group_masses


def _split_donor_weights(weights, donor_indices, clone_indices):
    """
    Split donor families exactly, even when the same donor is selected repeatedly.
    """
    if len(clone_indices) == 0:
        return

    unique_donors, inverse, counts = np.unique(
        donor_indices,
        return_inverse=True,
        return_counts=True,
    )
    donor_weights = weights[unique_donors].copy()
    split_weights = donor_weights / (counts + 1.0)

    weights[unique_donors] = split_weights
    weights[clone_indices] = split_weights[inverse]


def _build_true_weights(weights, group_to_particles, group_masses):
    """Convert normalized within-group weights into true population weights."""
    true_weights = np.zeros_like(weights)

    for group_id, group_indices in group_to_particles.items():
        group_mass = group_masses[group_id]

        if group_mass <= 0.0:
            continue

        true_weights[group_indices] = weights[group_indices] * group_mass

    return true_weights


def _weighted_mean_by_group(Xs, weights, group_to_particles, group_masses):
    """Compute the mean state using true group masses."""
    total_mass = np.sum(group_masses)

    if total_mass <= 0.0:
        return np.nan

    weighted_sum = 0.0

    for group_id, group_indices in group_to_particles.items():
        group_mass = group_masses[group_id]

        if group_mass <= 0.0:
            continue

        weighted_sum += group_mass * np.sum(Xs[group_indices] * weights[group_indices])

    return weighted_sum / total_mass


@njit(cache=True)
def _hazard_from_survival_numba(prev_survival, next_survival, dt):
    """Numba version of the discrete hazard helper."""
    if prev_survival <= 0.0:
        return np.inf

    if next_survival <= 0.0:
        return np.inf

    survival_ratio = next_survival / prev_survival

    if survival_ratio >= 1.0:
        return 0.0

    return -np.log(survival_ratio) / dt

@njit(cache=True)
def _goww_skorokhod_step_homogeneous(X, tcur, eta, beta, kappa, sqrt_2eps, sqrt_dt, dt):
    """
    Perform Skorokhod step for homogeneous go-with-winners (all particles same params).
    
    Args:
        X (np.ndarray): Current state for all particles
        tcur (float): Current time
        eta (float): Production rate (scalar)
        beta (float): Removal rate (scalar)
        kappa (float): Half-saturation constant (scalar)
        sqrt_2eps (float): sqrt(2*epsilon)
        sqrt_dt (float): sqrt(dt)
        dt (float): Time step
    
    Returns:
        np.ndarray: Next state for all particles
    """
    n = X.shape[0]
    X_next = np.empty(n)
    
    for i in range(n):
        # Drift
        drift = eta * tcur - X[i] * (beta / (X[i] + kappa))
        
        # Noise term
        if X[i] > 0:
            noise_coeff = sqrt_2eps
        else:
            noise_coeff = 0.0
        
        Y = noise_coeff * sqrt_dt * np.random.randn()
        U = np.random.random()
        
        Y_sq = Y * Y
        log_U_term = -2.0 * dt * np.log(max(U, 1e-100))
        inside_sqrt = Y_sq + log_U_term
        M = (Y + np.sqrt(max(inside_sqrt, 0.0))) / 2.0
        
        delta_X = dt * drift
        X_next[i] = max(M - Y, X[i] + delta_X - Y)
    
    return X_next


@njit(cache=True)
def _goww_skorokhod_step_heterogeneous(X, tcur, eta, beta, kappa, sqrt_2eps, sqrt_dt, dt):
    """
    Perform Skorokhod step for heterogeneous go-with-winners (per-particle params).
    
    Args:
        X (np.ndarray): Current state for all particles
        tcur (float): Current time
        eta (np.ndarray): Production rate per particle
        beta (np.ndarray): Removal rate per particle
        kappa (np.ndarray): Half-saturation constant per particle
        sqrt_2eps (np.ndarray): sqrt(2*epsilon) per particle
        sqrt_dt (float): sqrt(dt)
        dt (float): Time step
    
    Returns:
        np.ndarray: Next state for all particles
    """
    n = X.shape[0]
    X_next = np.empty(n)
    
    for i in range(n):
        # Drift
        drift = eta[i] * tcur - X[i] * (beta[i] / (X[i] + kappa[i]))
        
        # Noise term
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
        
        delta_X = dt * drift
        X_next[i] = max(M - Y, X[i] + delta_X - Y)
    
    return X_next


@njit(cache=True)
def _goww_homogeneous_loop(n, n_steps, x0, eta, beta, kappa, sqrt_2eps, Xc, 
                            sqrt_dt, dt, tmin, print_interval):
    """
    Main simulation loop for homogeneous go-with-winners (fully JIT compiled).
    
    Uses resampling method: when particles die, resample from survivors.
    All particles have equal weight at all times.
    
    Args:
        n (int): Number of particles
        n_steps (int): Total number of time steps
        x0 (float): Initial condition
        eta (float): Production rate
        beta (float): Removal rate
        kappa (float): Half-saturation constant
        sqrt_2eps (float): sqrt(2*epsilon)
        Xc (float): Threshold
        sqrt_dt (float): sqrt(dt)
        dt (float): Time step
        tmin (float): Start time
        print_interval (int): Steps between prints (0 = no printing)
    
    Returns:
        tuple: (survival, hazard, mean_X)
    """
    # Initialize
    survival = np.zeros(n_steps)
    hazard = np.zeros(n_steps)
    mean_X = np.zeros(n_steps)
    
    X = np.full(n, x0)
    S_cumulative = 1.0
    
    survival[0] = 1.0
    mean_X[0] = x0
    
    # Main loop
    for i_t in range(1, n_steps):
        tcur = tmin + i_t * dt
        
        # Evolution - Skorokhod step
        X_next = np.empty(n)
        for i in range(n):
            drift = eta * tcur - X[i] * (beta / (X[i] + kappa))
            
            if X[i] > 0:
                noise_coeff = sqrt_2eps
            else:
                noise_coeff = 0.0
            
            Y = noise_coeff * sqrt_dt * np.random.randn()
            U = np.random.random()
            
            Y_sq = Y * Y
            log_U_term = -2.0 * dt * np.log(max(U, 1e-100))
            inside_sqrt = Y_sq + log_U_term
            M = (Y + np.sqrt(max(inside_sqrt, 0.0))) / 2.0
            
            delta_X = dt * drift
            X_next[i] = max(M - Y, X[i] + delta_X - Y)
        
        X = X_next
        
        # Count survivors
        n_survivors = 0
        survivor_indices = np.empty(n, dtype=np.int64)
        for i in range(n):
            if X[i] < Xc:
                survivor_indices[n_survivors] = i
                n_survivors += 1
        
        if n_survivors == 0:
            survival[i_t] = 0.0
            hazard[i_t] = np.inf
            mean_X[i_t] = np.nan
            break
        
        # Update survival
        prev_survival = S_cumulative
        p_surv_step = n_survivors / n
        S_cumulative *= p_surv_step
        
        # Resample if deaths occurred
        if n_survivors < n:
            for i in range(n):
                donor_idx = survivor_indices[np.random.randint(0, n_survivors)]
                X[i] = X[donor_idx]
        
        # Store results
        survival[i_t] = S_cumulative
        hazard[i_t] = _hazard_from_survival_numba(prev_survival, S_cumulative, dt)
        
        # Mean X
        sum_X = 0.0
        for i in range(n):
            sum_X += X[i]
        mean_X[i_t] = sum_X / n
    
    return survival, hazard, mean_X


class SR_go_ww:
    """
    Go-with-Winners simulation for the SR model.
    
    This class implements the go-with-winners algorithm for estimating
    survival and hazard functions in the SR model. Unlike direct Monte Carlo,
    this method maintains a population of particles and resamples when
    particles die, allowing efficient estimation of rare event probabilities.
    
    Supports both homogeneous parameters (all particles have same parameters)
    and heterogeneous parameters (each particle has its own parameter values).
    
    For heterogeneous parameters, particles are grouped into bins based on their
    parameter values. When a particle dies, it is resampled from surviving 
    particles in the same bin (with similar parameter values), preserving the
    approximate parameter distribution in the population.
    """
    
    def __init__(self, eta, beta, kappa, epsilon, Xc, n=5000, tmin=0, tmax=1000, x0=1e-10,
                 dt=1, units='days', calc_pdf=False, bin_nums=100, num_of_pdfs=150, print_out=False,
                 save_paths=False, drift_expr=None, drift_mode='replace', extra_params=None,
                 n_param_bins=50, min_particles_per_bin=10, h_ext=None, random_seed=None):
        """
        Initialize Go-with-Winners simulation.

        Args:
            eta (float/array): Production rate (scalar or array of length n)
            beta (float/array): Removal rate (scalar or array of length n)
            kappa (float/array): Half-saturation constant (scalar or array of length n)
            epsilon (float/array): Noise intensity (scalar or array of length n)
            Xc (float/array): Critical threshold (scalar or array of length n)
            n (int): Number of particles
            tmin (float): Start time
            tmax (float): End time
            x0 (float): Initial condition
            dt (float): Time step
            units (str): Time units ('days' or 'years')
            calc_pdf (bool): Whether to calculate PDFs at specified times
            bin_nums (int): Number of bins for PDF histograms
            num_of_pdfs (int): Number of time points for PDF calculation
            print_out (bool): Whether to print progress
            save_paths (bool): Whether to save particle paths
            drift_expr (str/expr): Custom drift expression (optional)
            drift_mode (str): How to apply custom drift ('replace' or 'add')
            extra_params (dict): Additional drift parameters (optional)
            n_param_bins (int): Number of bins for grouping heterogeneous parameters.
                               More bins = more accurate parameter preservation but 
                               fewer particles per bin. Default 50.
            min_particles_per_bin (int): Minimum particles per bin. If a bin would have
                               fewer, bins are merged. Default 10.
            h_ext (float/array, optional): Constant external hazard rate. Scalar or length-n array.
            random_seed (int, optional): Seed for NumPy random draws used by the simulation.
        """
        self.n = int(n)
        self.tmin = tmin
        self.tmax = tmax
        self.x0 = x0
        self.dt = dt
        self.units = units
        self.n_param_bins = n_param_bins
        self.min_particles_per_bin = min_particles_per_bin
        self.print_out = print_out  # Set early since _create_param_groups uses it
        self.random_seed = None if random_seed is None else int(random_seed)
        
        # Process parameters - determine if heterogeneous or homogeneous
        self.eta_array = self._process_param(eta)
        self.beta_array = self._process_param(beta)
        self.kappa_array = self._process_param(kappa)
        self.epsilon_array = self._process_param(epsilon)
        self.Xc_array = self._process_param(Xc)
        self.h_ext_array = self._process_param(h_ext if h_ext is not None else 0.0)
        
        # Check if we have heterogeneous parameters
        self.is_heterogeneous = self._check_heterogeneity()
        
        # For homogeneous case, store scalar values for efficiency
        if not self.is_heterogeneous:
            self.eta = float(self.eta_array[0])
            self.beta = float(self.beta_array[0])
            self.kappa = float(self.kappa_array[0])
            self.epsilon = float(self.epsilon_array[0])
            self.Xc = float(self.Xc_array[0])
            self.h_ext = float(self.h_ext_array[0])
        
        # For heterogeneous case, create parameter groups for resampling
        if self.is_heterogeneous:
            self._create_param_groups()
        
        # Create SimulationParams for drift function management
        self.params = SimulationParams(
            eta=eta, beta=beta, kappa=kappa, epsilon=epsilon, Xc=Xc,
            n=self.n, tmin=tmin, tmax=tmax, x0=x0, dt=dt,
            units=units, parallel=False, break_early=False,
            drift_expr=drift_expr, drift_mode=drift_mode, extra_params=extra_params
        )
        
        # Go-with-winners specific settings
        self.calc_pdf = calc_pdf
        self.bin_nums = bin_nums
        self.num_of_pdfs = num_of_pdfs
        self.save_paths = save_paths
        self.drift_expr = drift_expr
        
        # Results storage
        self.survival = None
        self.hazard = None
        self.pdfs = None
        self.tspan_pdfs = None
        self.tspan = None
        self.weights = None
        self.bins = None
        self.mean_X = None
        self.survival_area = None
        self.paths = None
        self.path_weights = None
        self.path_times = None

        # Run the simulation
        self.run_go_with_winners()

    def _process_param(self, param):
        """Convert parameter to array of length n."""
        if np.isscalar(param) or np.size(param) == 1:
            return np.full(self.n, float(np.atleast_1d(param)[0]))
        elif len(param) == self.n:
            return np.array(param, dtype=float)
        else:
            raise ValueError(f"Parameter must be scalar or array of length {self.n}, got length {len(param)}")

    def _check_heterogeneity(self):
        """Check if any parameter has variation across particles."""
        for arr in [self.eta_array, self.beta_array, self.kappa_array, 
                    self.epsilon_array, self.Xc_array]:
            if len(np.unique(arr)) > 1:
                return True
        return False

    def _create_param_groups(self):
        """
        Create parameter groups for heterogeneous resampling using binning.
        
        For continuous parameter distributions, we bin each varying parameter
        into n_param_bins quantile-based bins, then create groups from the
        combination of bins. This ensures roughly equal particles per group.
        
        When a particle dies, it can only be resampled from survivors
        in the same group (with similar parameters).
        """
        # Identify which parameters vary
        varying_params = []
        param_arrays = {
            'eta': self.eta_array,
            'beta': self.beta_array,
            'kappa': self.kappa_array,
            'epsilon': self.epsilon_array,
            'Xc': self.Xc_array
        }
        
        for name, arr in param_arrays.items():
            if len(np.unique(arr)) > 1:
                varying_params.append((name, arr))
        
        if len(varying_params) == 0:
            # No variation - shouldn't happen if is_heterogeneous is True
            self.param_group_indices = np.zeros(self.n, dtype=int)
            self.n_groups = 1
            self.group_to_particles = {0: np.arange(self.n)}
            return
        
        # For each varying parameter, create bin assignments using quantiles
        # This ensures roughly equal particles per bin
        bin_assignments = []
        
        for name, arr in varying_params:
            n_unique = len(np.unique(arr))
            
            if n_unique <= self.n_param_bins:
                # Fewer unique values than bins - use exact values
                _, bin_idx = np.unique(arr, return_inverse=True)
            else:
                # Use quantile-based binning for continuous distributions
                # This ensures roughly equal particles per bin
                percentiles = np.linspace(0, 100, self.n_param_bins + 1)
                bin_edges = np.percentile(arr, percentiles)
                # Make edges unique to handle ties
                bin_edges = np.unique(bin_edges)
                # Assign each particle to a bin
                bin_idx = np.digitize(arr, bin_edges[1:-1])  # n_bins - 1 edges for n_bins bins
            
            bin_assignments.append(bin_idx)
        
        # Combine bin assignments into group indices
        if len(bin_assignments) == 1:
            combined_bins = bin_assignments[0]
        else:
            # Stack and find unique combinations
            stacked = np.column_stack(bin_assignments)
            _, combined_bins = np.unique(stacked, axis=0, return_inverse=True)
        
        # Merge small groups if needed
        self.param_group_indices, self.n_groups = self._merge_small_groups(combined_bins)
        
        # Create mapping from group ID to particle indices
        self.group_to_particles = {}
        for group_id in range(self.n_groups):
            self.group_to_particles[group_id] = np.where(self.param_group_indices == group_id)[0]
        
        # Store group mean parameters for reference
        self._compute_group_mean_params()
        
        if self.print_out:
            group_sizes = [len(self.group_to_particles[g]) for g in range(self.n_groups)]
            print(f"Heterogeneous mode: {self.n_groups} parameter groups")
            print(f"  Group sizes: min={min(group_sizes)}, max={max(group_sizes)}, "
                  f"mean={np.mean(group_sizes):.1f}")
            print(f"  Varying parameters: {[name for name, _ in varying_params]}")

    def _merge_small_groups(self, group_indices):
        """
        Merge groups that have fewer than min_particles_per_bin particles.
        
        Small groups are merged with their nearest neighbor (by group index,
        which roughly corresponds to parameter similarity due to binning).
        """
        unique_groups = np.unique(group_indices)
        group_sizes = {g: np.sum(group_indices == g) for g in unique_groups}
        
        # Check if any groups need merging
        small_groups = [g for g, size in group_sizes.items() if size < self.min_particles_per_bin]
        
        if len(small_groups) == 0:
            # Renumber groups to be consecutive
            _, new_indices = np.unique(group_indices, return_inverse=True)
            return new_indices, len(unique_groups)
        
        # Merge small groups with neighbors
        # Sort groups by their index (which roughly corresponds to parameter value)
        sorted_groups = sorted(unique_groups)
        
        # Create merge mapping
        merge_map = {}
        current_merged_group = 0
        accumulated_size = 0
        groups_in_current = []
        
        for g in sorted_groups:
            groups_in_current.append(g)
            accumulated_size += group_sizes[g]
            
            if accumulated_size >= self.min_particles_per_bin:
                # Assign all accumulated groups to current merged group
                for merged_g in groups_in_current:
                    merge_map[merged_g] = current_merged_group
                current_merged_group += 1
                accumulated_size = 0
                groups_in_current = []
        
        # Handle remaining groups
        if groups_in_current:
            if current_merged_group > 0:
                # Merge with previous group
                for merged_g in groups_in_current:
                    merge_map[merged_g] = current_merged_group - 1
            else:
                # Only one group total
                for merged_g in groups_in_current:
                    merge_map[merged_g] = 0
                current_merged_group = 1
        
        # Apply merge mapping
        new_indices = np.array([merge_map[g] for g in group_indices])
        n_groups = len(set(merge_map.values()))
        
        return new_indices, n_groups

    def _compute_group_mean_params(self):
        """Compute mean parameter values for each group (for reference/debugging)."""
        self.group_mean_params = {}
        for group_id in range(self.n_groups):
            particles = self.group_to_particles[group_id]
            self.group_mean_params[group_id] = {
                'eta': np.mean(self.eta_array[particles]),
                'beta': np.mean(self.beta_array[particles]),
                'kappa': np.mean(self.kappa_array[particles]),
                'epsilon': np.mean(self.epsilon_array[particles]),
                'Xc': np.mean(self.Xc_array[particles]),
                'n_particles': len(particles)
            }

    def noise(self, X, tcur, indices=None):
        """
        Calculate noise coefficient for the SR model.
        
        Args:
            X (np.ndarray): Current state
            tcur (float): Current time (unused, for interface compatibility)
            indices (np.ndarray, optional): Particle indices for heterogeneous case
            
        Returns:
            np.ndarray: Noise coefficient sqrt(2*epsilon) where X > 0
        """
        if self.is_heterogeneous:
            if indices is None:
                eps = self.epsilon_array
            else:
                eps = self.epsilon_array[indices]
            return np.sqrt(2 * eps) * (X > 0)
        else:
            return np.sqrt(2 * self.epsilon) * (X > 0)

    def drift_classic(self, X, tcur, indices=None):
        """
        Calculate drift for the SR model (classic formulation).
        
        Args:
            X (np.ndarray): Current state
            tcur (float): Current time
            indices (np.ndarray, optional): Particle indices for heterogeneous case
            
        Returns:
            np.ndarray: Drift values
        """
        # Use custom drift if specified
        if self.drift_expr is not None:
            return self.params.get_drift(X, tcur)
        
        if self.is_heterogeneous:
            if indices is None:
                eta = self.eta_array
                beta = self.beta_array
                kappa = self.kappa_array
            else:
                eta = self.eta_array[indices]
                beta = self.beta_array[indices]
                kappa = self.kappa_array[indices]
            return eta * tcur - X * (beta / (X + kappa))
        else:
            return self.eta * tcur - X * (self.beta / (X + self.kappa))

    def get_Xc(self, indices=None):
        """Get threshold values for specified particles."""
        if self.is_heterogeneous:
            if indices is None:
                return self.Xc_array
            return self.Xc_array[indices]
        else:
            return self.Xc

    def run_go_with_winners(self):
        """Run the go-with-winners simulation and store results."""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        if self.is_heterogeneous:
            results = self._calc_survival_hazard_heterogeneous()
        else:
            results = self._calc_survival_hazard_homogeneous()
            
        if self.save_paths:
            (self.survival, self.hazard, self.pdfs, self.tspan_pdfs, self.tspan, 
             self.weights, self.bins, self.mean_X, self.paths, self.path_weights, 
             self.path_times) = results
        else:
            (self.survival, self.hazard, self.pdfs, self.tspan_pdfs, self.tspan, 
             self.weights, self.bins, self.mean_X) = results[:8]

    def _calc_survival_hazard_homogeneous(self):
        """
        Calculate survival and hazard for HOMOGENEOUS parameters using Resampling.
        
        This method avoids the "zombie particle" problem by tracking cumulative survival
        and fully resampling from survivors each step. All particles remain "fresh"
        with equal weight.
        
        Uses numba-optimized kernel when PDF/path saving not needed.
        
        Logic:
        1. Evolve all particles (vectorized).
        2. Count survivors, update cumulative survival.
        3. Resample ALL particle slots from survivors.
        4. Calculate hazard and statistics.
        """
        tspan = np.arange(self.tmin, self.tmax + 0.01, self.dt)
        
        # --- PDF SETUP ---
        if self.calc_pdf:
            tspan_pdfs = np.arange(self.tmin, self.tmax + 1, int((self.tmax - self.tmin) / self.num_of_pdfs))
            ind_pdfs = 0
            pdfs = np.zeros([np.size(tspan_pdfs), self.bin_nums])
        else:
            tspan_pdfs = []
            pdfs = []

        # Use fast kernel if no PDF/path saving and no custom drift
        use_fast = (
            not self.calc_pdf
            and not self.save_paths
            and self.drift_expr is None
            and self.h_ext <= 0.0
        )
        
        if use_fast:
            sqrt_2eps = np.sqrt(2.0 * self.epsilon)
            sqrt_dt = np.sqrt(self.dt)
            
            survival, hazard, mean_X = _goww_homogeneous_loop(
                self.n, len(tspan), self.x0, self.eta, self.beta, self.kappa,
                sqrt_2eps, self.Xc, sqrt_dt, self.dt, self.tmin, 0
            )
            
            return survival, hazard, pdfs, tspan_pdfs, tspan, None, None, mean_X

        # --- INITIALIZATION ---
        Xs = self.x0 * np.ones(self.n)
        
        # Pre-compute for efficiency
        sqrt_2eps = np.sqrt(2.0 * self.epsilon)
        sqrt_dt = np.sqrt(self.dt)
        
        # Track cumulative survival (not per-particle weights)
        S_cumulative = 1.0
        
        survival = np.zeros(len(tspan))
        hazard = np.zeros(len(tspan))
        mean_X = np.zeros(len(tspan))
        
        # Initial stats
        survival[0] = 1.0
        mean_X[0] = self.x0

        # Path saving setup
        if self.save_paths:
            save_interval = 1
            num_save_points = len(range(0, len(tspan), save_interval))
            paths = np.zeros((self.n, num_save_points))
            path_weights = np.zeros((self.n, num_save_points))
            path_times = np.zeros(num_save_points)
            save_idx = 0
            
            # Initial Save
            paths[:, save_idx] = Xs
            path_weights[:, save_idx] = np.ones(self.n) / self.n
            path_times[save_idx] = tspan[0]
            save_idx += 1

        bins = None
        
        # --- MAIN LOOP ---
        for i_t in range(1, np.size(tspan)):
            tcur = tspan[i_t]
            
            # 1. EVOLUTION - use numba kernel
            Xs = _goww_skorokhod_step_homogeneous(
                Xs, tcur, self.eta, self.beta, self.kappa, 
                sqrt_2eps, sqrt_dt, self.dt
            )
            
            # 2. COUNT SURVIVORS
            alive_mask = Xs < self.Xc
            if self.h_ext > 0.0:
                p_ext = 1.0 - np.exp(-self.h_ext * self.dt)
                alive_mask &= np.random.random(self.n) >= p_ext
            survivors = np.where(alive_mask)[0]
            n_survivors = len(survivors)
            
            if n_survivors == 0:
                if self.print_out:
                    print(f"All particles died at t={tcur}")
                survival[i_t] = 0
                hazard[i_t] = np.inf
                mean_X[i_t] = np.nan
                break
                            
            # Fraction that survived this step
            prev_survival = S_cumulative
            p_surv_step = n_survivors / self.n
            
            # Update cumulative survival
            S_cumulative *= p_surv_step
            
            # 3. RESAMPLE - refill all slots from survivors
            # This keeps all particles "fresh" - no zombies!
            if n_survivors < self.n:
                new_indices = np.random.choice(survivors, size=self.n, replace=True)
                Xs = Xs[new_indices]
            
            # 4. CALCULATE STATS
            survival[i_t] = S_cumulative
            hazard[i_t] = _hazard_from_survival(prev_survival, S_cumulative, self.dt)
            
            # Mean X (all particles have equal weight now)
            mean_X[i_t] = np.mean(Xs)
            
            # 5. SAVE OUTPUTS
            if self.calc_pdf and tspan[i_t] in tspan_pdfs:
                # All particles have equal weight = S_cumulative / n
                particle_weight = S_cumulative / self.n
                hist, bins = np.histogram(Xs, bins=self.bin_nums, range=(0, self.Xc), 
                                         density=True, weights=np.full(self.n, particle_weight))
                pdfs[ind_pdfs, :] = hist * S_cumulative
                ind_pdfs += 1                        
                if self.print_out:
                    print(f't = {tcur:.1f}, S(t) = {S_cumulative:.4e}, n_survivors = {n_survivors}')
            
            if self.save_paths:
                if i_t % save_interval == 0 and save_idx < num_save_points:
                    paths[:, save_idx] = Xs
                    # All particles have equal weight
                    path_weights[:, save_idx] = np.full(self.n, S_cumulative / self.n)
                    path_times[save_idx] = tspan[i_t]
                    save_idx += 1

        if self.save_paths:
            return survival, hazard, pdfs, tspan_pdfs, tspan, None, bins, mean_X, paths, path_weights, path_times
        else:
            return survival, hazard, pdfs, tspan_pdfs, tspan, None, bins, mean_X

    def _calc_survival_hazard_heterogeneous(self):
        """
        Calculate survival and hazard for heterogeneous parameters.

        Each parameter group tracks:
        - one true survival mass for the group
        - normalized weights for particles inside that group

        This keeps all hazard and survival calculations on a single physical
        scale, even after long runs and repeated resampling.
        """
        tspan = np.arange(self.tmin, self.tmax + 0.01, self.dt)

        if self.calc_pdf:
            tspan_pdfs = np.arange(
                self.tmin,
                self.tmax + 1,
                int((self.tmax - self.tmin) / self.num_of_pdfs),
            )
            ind_pdfs = 0
            pdfs = np.zeros([np.size(tspan_pdfs), self.bin_nums])
        else:
            tspan_pdfs = []
            pdfs = []

        Xs = self.x0 * np.ones(self.n)
        weights, group_masses = _initialize_group_weights(
            self.n,
            self.group_to_particles,
            self.n_groups,
        )

        sqrt_2eps = np.sqrt(2.0 * self.epsilon_array)
        sqrt_dt = np.sqrt(self.dt)

        hazard = np.zeros(len(tspan))
        survival = np.zeros(len(tspan))
        mean_X = np.zeros(len(tspan))

        survival[0] = 1.0
        mean_X[0] = self.x0

        if self.save_paths:
            save_interval = 1
            num_save_points = len(range(0, len(tspan), save_interval))
            paths = np.zeros((self.n, num_save_points))
            path_weights = np.zeros((self.n, num_save_points))
            path_times = np.zeros(num_save_points)
            save_idx = 0

            paths[:, save_idx] = self.x0
            path_weights[:, save_idx] = _build_true_weights(
                weights,
                self.group_to_particles,
                group_masses,
            )
            path_times[save_idx] = tspan[0]
            save_idx += 1

        bins = None

        for i_t in range(1, np.size(tspan)):
            tcur = tspan[i_t]
            prev_survival = survival[i_t - 1]

            Xs = _goww_skorokhod_step_heterogeneous(
                Xs,
                tcur,
                self.eta_array,
                self.beta_array,
                self.kappa_array,
                sqrt_2eps,
                sqrt_dt,
                self.dt,
            )

            who_died_mask = Xs >= self.Xc_array
            active_h_ext = self.h_ext_array > 0.0
            if np.any(active_h_ext):
                p_ext = 1.0 - np.exp(-self.h_ext_array * self.dt)
                who_died_mask |= np.random.random(self.n) < p_ext

            for group_id in range(self.n_groups):
                if group_masses[group_id] <= 0.0:
                    continue

                group_indices = self.group_to_particles[group_id]
                group_died_indices = group_indices[who_died_mask[group_indices]]

                if len(group_died_indices) == 0:
                    continue

                group_surv_indices = group_indices[~who_died_mask[group_indices]]

                if len(group_surv_indices) == 0:
                    group_masses[group_id] = 0.0
                    weights[group_indices] = 0.0
                    continue

                survivor_weight_sum = np.sum(weights[group_surv_indices])

                if survivor_weight_sum <= 0.0:
                    group_masses[group_id] = 0.0
                    weights[group_indices] = 0.0
                    continue

                group_masses[group_id] *= survivor_weight_sum
                weights[group_surv_indices] = (
                    weights[group_surv_indices] / survivor_weight_sum
                )
                weights[group_died_indices] = 0.0

                donors = np.random.choice(
                    group_surv_indices,
                    size=len(group_died_indices),
                    p=weights[group_surv_indices],
                )
                Xs[group_died_indices] = Xs[donors]
                _split_donor_weights(weights, donors, group_died_indices)

            total_survival_mass = np.sum(group_masses)
            survival[i_t] = total_survival_mass / self.n
            hazard[i_t] = _hazard_from_survival(
                prev_survival,
                survival[i_t],
                self.dt,
            )
            mean_X[i_t] = _weighted_mean_by_group(
                Xs,
                weights,
                self.group_to_particles,
                group_masses,
            )

            if survival[i_t] <= 0.0:
                if self.print_out:
                    print(f"Global extinction at t={tcur}")
                break

            if self.calc_pdf and tspan[i_t] in tspan_pdfs:
                true_weights = _build_true_weights(
                    weights,
                    self.group_to_particles,
                    group_masses,
                )
                max_Xc = np.max(self.Xc_array)
                hist_true, bins = np.histogram(
                    Xs,
                    bins=self.bin_nums,
                    range=(0, max_Xc),
                    density=False,
                    weights=true_weights,
                )
                pdfs[ind_pdfs, :] = hist_true / self.n
                ind_pdfs += 1

            if self.save_paths:
                if i_t % save_interval == 0 and save_idx < num_save_points:
                    paths[:, save_idx] = Xs
                    path_weights[:, save_idx] = _build_true_weights(
                        weights,
                        self.group_to_particles,
                        group_masses,
                    )
                    path_times[save_idx] = tspan[i_t]
                    save_idx += 1

        final_true_weights = _build_true_weights(
            weights,
            self.group_to_particles,
            group_masses,
        )

        if self.save_paths:
            return (
                survival,
                hazard,
                pdfs,
                tspan_pdfs,
                tspan,
                final_true_weights,
                bins,
                mean_X,
                paths,
                path_weights,
                path_times,
            )

        return survival, hazard, pdfs, tspan_pdfs, tspan, final_true_weights, bins, mean_X
            
    def pdf_at_t(self, t):
        """
        Get the PDF at a specific time.
        
        Args:
            t (float): Time at which to get PDF
            
        Returns:
            tuple: (pdf_values, closest_time)
        """
        if len(self.tspan_pdfs) == 0:
            raise ValueError("PDFs were not calculated. Set calc_pdf=True when initializing.")
        closest_t = min(self.tspan_pdfs, key=lambda x: abs(x - t))
        ind_closest_t = np.where(self.tspan_pdfs == closest_t)[0][0]
        return self.pdfs[ind_closest_t, :], closest_t

    def survival_from_pdf_area(self, t):
        """Calculate survival probability from PDF area at time t."""
        if self.bins is None:
            raise ValueError("Bins not available. Set calc_pdf=True when initializing.")
        return sum(self.pdf_at_t(t)[0] * np.diff(self.bins))
    
    def find_time_at_survival(self, S):
        """Find time at which survival probability equals S."""
        if np.any(self.survival <= S):
            return np.interp(S, self.survival[::-1], self.tspan[::-1])
        return None
    
    def calc_steepness(self):
        """Calculate steepness of survival curve (median / IQR)."""
        t_25 = self.find_time_at_survival(0.25)
        t_50 = self.find_time_at_survival(0.5)
        t_75 = self.find_time_at_survival(0.75)
        
        if all(t is not None for t in [t_25, t_50, t_75]) and t_75 != t_25:
            return -t_50 / (t_75 - t_25)
        return None
    
    def get_group_survival(self):
        """
        Get survival curves for each parameter group separately.
        
        Only available for heterogeneous simulations.
        
        Returns:
            dict: {group_id: {'survival': array, 'mean_params': dict}}
        """
        if not self.is_heterogeneous:
            raise ValueError("Group survival only available for heterogeneous simulations")
        
        # This would require storing per-group survival during simulation
        # For now, return the group mean parameters
        return self.group_mean_params
