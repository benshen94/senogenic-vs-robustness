import numpy as np
from multiprocessing import Pool, cpu_count, get_context


_CORE_PARAMETER_NAMES = ("x0", "eta", "beta", "kappa", "epsilon", "Xc")
_CHUNK_PARAMETER_NAMES = _CORE_PARAMETER_NAMES + ("h_ext",)


def _is_sampler_spec(value):
    if not isinstance(value, dict):
        return False

    return "dist" in value or "sampler" in value


def _slice_array_if_needed(value, start_idx, chunk_size, total_n, name):
    if value is None:
        return None
    if callable(value):
        return value
    if _is_sampler_spec(value):
        return value
    if np.isscalar(value) or np.size(value) <= 1:
        return float(np.atleast_1d(value)[0])

    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a scalar or a 1D array.")

    if array.size == chunk_size:
        return array.copy()

    if array.size != total_n:
        raise ValueError(f"{name} must have length n={total_n}, but got {array.size}.")

    end_idx = start_idx + chunk_size
    return array[start_idx:end_idx].copy()


def _payload_to_array(value, chunk_size):
    if np.isscalar(value) or np.size(value) <= 1:
        return np.full(chunk_size, float(np.atleast_1d(value)[0]), dtype=np.float64)

    return np.asarray(value, dtype=np.float64)


def _normalize_chunk_value(value, chunk_size, name):
    if np.isscalar(value) or np.size(value) <= 1:
        return float(np.atleast_1d(value)[0])

    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must resolve to a scalar or a 1D array.")
    if array.size != chunk_size:
        raise ValueError(f"{name} must resolve to length {chunk_size}, but got {array.size}.")
    return array


def _normalize_h_ext_chunk_value(value, chunk_size):
    if value is None:
        return None
    if callable(value):
        return value

    return _normalize_chunk_value(value, chunk_size, "h_ext")


def _apply_clip(values, spec):
    clip_min = spec.get("clip_min")
    clip_max = spec.get("clip_max")

    if clip_min is not None:
        values = np.maximum(values, clip_min)
    if clip_max is not None:
        values = np.minimum(values, clip_max)

    return values


def _sample_from_distribution(spec, rng, chunk_size, name):
    dist = spec.get("dist")
    if dist is None:
        raise ValueError(f"{name} sampler spec is missing 'dist'.")

    if dist == "normal":
        mean = spec.get("mean", spec.get("loc"))
        std = spec.get("std", spec.get("scale"))
        if mean is None or std is None:
            raise ValueError(f"{name} normal sampler requires 'mean'/'loc' and 'std'/'scale'.")
        values = rng.normal(loc=mean, scale=std, size=chunk_size)
        return _apply_clip(values, spec)

    if dist == "lognormal":
        mean = spec.get("mean")
        sigma = spec.get("sigma")
        if mean is None or sigma is None:
            raise ValueError(f"{name} lognormal sampler requires 'mean' and 'sigma'.")
        values = rng.lognormal(mean=mean, sigma=sigma, size=chunk_size)
        return _apply_clip(values, spec)

    if dist == "gamma":
        shape = spec.get("shape")
        scale = spec.get("scale")
        if shape is None or scale is None:
            raise ValueError(f"{name} gamma sampler requires 'shape' and 'scale'.")
        values = rng.gamma(shape=shape, scale=scale, size=chunk_size)
        return _apply_clip(values, spec)

    if dist == "uniform":
        low = spec.get("low")
        high = spec.get("high")
        if low is None or high is None:
            raise ValueError(f"{name} uniform sampler requires 'low' and 'high'.")
        values = rng.uniform(low=low, high=high, size=chunk_size)
        return _apply_clip(values, spec)

    if dist == "choice":
        values = spec.get("values")
        if values is None:
            raise ValueError(f"{name} choice sampler requires 'values'.")
        probs = spec.get("probs")
        sampled = rng.choice(values, size=chunk_size, p=probs)
        return np.asarray(sampled, dtype=np.float64)

    raise ValueError(f"Unsupported sampler dist for {name}: {dist}")


def _sample_from_callable(spec, rng, chunk_size, start_idx, name):
    sampler = spec.get("sampler")
    if sampler is None:
        raise ValueError(f"{name} sampler spec is missing 'sampler'.")

    sampled = sampler(rng, chunk_size, start_idx)
    return _normalize_chunk_value(sampled, chunk_size, name)


def _resolve_chunk_value(value, start_idx, chunk_size, total_n, name, rng):
    if _is_sampler_spec(value):
        if "sampler" in value:
            return _sample_from_callable(value, rng, chunk_size, start_idx, name)
        return _sample_from_distribution(value, rng, chunk_size, name)

    return _slice_array_if_needed(value, start_idx, chunk_size, total_n, name)


def _resolve_h_ext_chunk(value, start_idx, chunk_size, total_n, rng):
    if value is None:
        return None

    if isinstance(value, list):
        raise TypeError(
            "SRHazardSim does not support list-valued h_ext. "
            "Use None, a scalar, a 1D array, a sampler spec, or a callable h(t)."
        )

    if callable(value):
        return value

    if _is_sampler_spec(value):
        if "sampler" in value:
            sampled = _sample_from_callable(value, rng, chunk_size, start_idx, "h_ext")
            return _normalize_h_ext_chunk_value(sampled, chunk_size)
        sampled = _sample_from_distribution(value, rng, chunk_size, "h_ext")
        return _normalize_h_ext_chunk_value(sampled, chunk_size)

    sliced = _slice_array_if_needed(value, start_idx, chunk_size, total_n, "h_ext")
    return _normalize_h_ext_chunk_value(sliced, chunk_size)


def _prepare_h_ext_payload(h_ext, dt):
    if h_ext is None:
        return "none", None

    if callable(h_ext):
        return "callable", h_ext

    if np.isscalar(h_ext) or np.size(h_ext) <= 1:
        rate = float(np.atleast_1d(h_ext)[0])
        return "scalar", rate

    rate_array = np.asarray(h_ext, dtype=np.float64)
    if rate_array.ndim != 1:
        raise ValueError("Resolved h_ext must be a scalar, callable, or a 1D array.")

    return "array", rate_array


def _call_joint_sampler(joint_sampler, rng, chunk_size, start_idx):
    if joint_sampler is None:
        return {}

    chunk_values = joint_sampler(rng, chunk_size, start_idx)
    if chunk_values is None:
        return {}
    if not isinstance(chunk_values, dict):
        raise TypeError("joint_sampler must return a dictionary.")

    unknown_keys = set(chunk_values) - set(_CHUNK_PARAMETER_NAMES)
    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        raise ValueError(f"joint_sampler returned unsupported keys: {unknown}")

    normalized = {}
    for key, value in chunk_values.items():
        if key == "h_ext":
            normalized[key] = _normalize_h_ext_chunk_value(value, chunk_size)
            continue

        normalized[key] = _normalize_chunk_value(value, chunk_size, key)

    return normalized


def _should_override_value(value, default_value):
    if isinstance(value, np.ndarray):
        return True
    if isinstance(value, (list, dict)):
        return True
    if callable(value):
        return True

    return value != default_value


def normal_sampler(mean, std, clip_min=None, clip_max=None):
    spec = {"dist": "normal", "mean": mean, "std": std}
    if clip_min is not None:
        spec["clip_min"] = clip_min
    if clip_max is not None:
        spec["clip_max"] = clip_max
    return spec


def lognormal_sampler(mean, sigma, clip_min=None, clip_max=None):
    spec = {"dist": "lognormal", "mean": mean, "sigma": sigma}
    if clip_min is not None:
        spec["clip_min"] = clip_min
    if clip_max is not None:
        spec["clip_max"] = clip_max
    return spec


def gamma_sampler(shape, scale, clip_min=None, clip_max=None):
    spec = {"dist": "gamma", "shape": shape, "scale": scale}
    if clip_min is not None:
        spec["clip_min"] = clip_min
    if clip_max is not None:
        spec["clip_max"] = clip_max
    return spec


def uniform_sampler(low, high, clip_min=None, clip_max=None):
    spec = {"dist": "uniform", "low": low, "high": high}
    if clip_min is not None:
        spec["clip_min"] = clip_min
    if clip_max is not None:
        spec["clip_max"] = clip_max
    return spec


def choice_sampler(values, probs=None):
    spec = {"dist": "choice", "values": values}
    if probs is not None:
        spec["probs"] = probs
    return spec


def callable_sampler(sampler):
    return {"sampler": sampler}


def _simulate_chunk(task):
    chunk_size = task["chunk_size"]
    n_steps = task["n_steps"]
    dt = task["dt"]
    tmin = task["tmin"]
    start_idx = task["start_idx"]
    total_n = task["total_n"]
    break_early = task["break_early"]
    xc_drop_age = task["xc_drop_age"]
    xc_drop_fraction = task["xc_drop_fraction"]
    xc_drop_applied = False
    x_bump_age = task["x_bump_age"]
    x_bump_fraction = task["x_bump_fraction"]
    x_bump_applied = False
    x_add_age = task["x_add_age"]
    x_add_amount = task["x_add_amount"]
    x_add_applied = False
    rng = np.random.default_rng(task["sim_seed"])
    param_rng = np.random.default_rng(task["param_seed"])

    joint_values = _call_joint_sampler(
        task["joint_sampler"],
        param_rng,
        chunk_size,
        start_idx,
    )

    x0_value = joint_values.get("x0")
    if x0_value is None:
        x0_value = _resolve_chunk_value(task["x0"], start_idx, chunk_size, total_n, "x0", param_rng)

    eta_value = joint_values.get("eta")
    if eta_value is None:
        eta_value = _resolve_chunk_value(task["eta"], start_idx, chunk_size, total_n, "eta", param_rng)

    beta_value = joint_values.get("beta")
    if beta_value is None:
        beta_value = _resolve_chunk_value(task["beta"], start_idx, chunk_size, total_n, "beta", param_rng)

    kappa_value = joint_values.get("kappa")
    if kappa_value is None:
        kappa_value = _resolve_chunk_value(task["kappa"], start_idx, chunk_size, total_n, "kappa", param_rng)

    epsilon_value = joint_values.get("epsilon")
    if epsilon_value is None:
        epsilon_value = _resolve_chunk_value(task["epsilon"], start_idx, chunk_size, total_n, "epsilon", param_rng)

    Xc_value = joint_values.get("Xc")
    if Xc_value is None:
        Xc_value = _resolve_chunk_value(task["Xc"], start_idx, chunk_size, total_n, "Xc", param_rng)

    h_ext_value = joint_values.get("h_ext")
    if h_ext_value is None:
        h_ext_value = _resolve_h_ext_chunk(task["h_ext"], start_idx, chunk_size, total_n, param_rng)

    X = _payload_to_array(x0_value, chunk_size)
    eta = _payload_to_array(eta_value, chunk_size)
    beta = _payload_to_array(beta_value, chunk_size)
    kappa = _payload_to_array(kappa_value, chunk_size)
    epsilon = _payload_to_array(epsilon_value, chunk_size)
    sqrt_2epsilon = np.sqrt(2.0 * epsilon)
    Xc = _payload_to_array(Xc_value, chunk_size)

    h_ext_mode, h_ext_payload = _prepare_h_ext_payload(h_ext_value, dt)
    h_ext_array = None
    h_ext_scalar = None
    h_ext_callable = None

    if h_ext_mode == "scalar":
        h_ext_scalar = float(h_ext_payload)
    elif h_ext_mode == "array":
        h_ext_array = np.asarray(h_ext_payload, dtype=np.float64)
    elif h_ext_mode == "callable":
        h_ext_callable = h_ext_payload

    deaths_per_step = np.zeros(n_steps, dtype=np.int64)

    if X.size == 0:
        return deaths_per_step

    sqrt_dt = np.sqrt(dt)

    for step_idx in range(1, n_steps):
        if break_early and X.size == 0:
            break

        tcur = tmin + step_idx * dt
        if (
            not xc_drop_applied
            and xc_drop_fraction > 0.0
            and np.isfinite(xc_drop_age)
            and tcur >= xc_drop_age
        ):
            Xc *= 1.0 - xc_drop_fraction
            xc_drop_applied = True

        if (
            not x_bump_applied
            and x_bump_fraction > 0.0
            and np.isfinite(x_bump_age)
            and tcur >= x_bump_age
        ):
            X *= 1.0 + x_bump_fraction
            x_bump_applied = True

            # Immediate threshold check at the moment of the bump —
            # anyone already above Xc right after the jump dies now,
            # before the next SDE step is taken.
            immediately_crossed = X > Xc
            n_immediately_crossed = int(immediately_crossed.sum())
            if n_immediately_crossed > 0:
                deaths_per_step[step_idx] += n_immediately_crossed
                survivors = ~immediately_crossed
                X = X[survivors]
                eta = eta[survivors]
                beta = beta[survivors]
                kappa = kappa[survivors]
                epsilon = epsilon[survivors]
                sqrt_2epsilon = sqrt_2epsilon[survivors]
                Xc = Xc[survivors]
                if h_ext_array is not None:
                    h_ext_array = h_ext_array[survivors]

        if (
            not x_add_applied
            and x_add_amount > 0.0
            and np.isfinite(x_add_age)
            and tcur >= x_add_age
        ):
            X += x_add_amount
            x_add_applied = True

            # Immediate threshold check after additive shift.
            immediately_crossed = X > Xc
            n_immediately_crossed = int(immediately_crossed.sum())
            if n_immediately_crossed > 0:
                deaths_per_step[step_idx] += n_immediately_crossed
                survivors = ~immediately_crossed
                X = X[survivors]
                eta = eta[survivors]
                beta = beta[survivors]
                kappa = kappa[survivors]
                epsilon = epsilon[survivors]
                sqrt_2epsilon = sqrt_2epsilon[survivors]
                Xc = Xc[survivors]
                if h_ext_array is not None:
                    h_ext_array = h_ext_array[survivors]

        X_prev = X

        drift = eta * tcur - X_prev * (beta / (X_prev + kappa))

        noise = sqrt_2epsilon * (X_prev > 0.0)
        Y = noise * sqrt_dt * rng.standard_normal(X_prev.size)
        U = np.clip(rng.random(X_prev.size), 1e-100, 1.0)

        inside_sqrt = Y * Y - 2.0 * dt * np.log(U)
        M = (Y + np.sqrt(np.maximum(inside_sqrt, 0.0))) / 2.0
        X = np.maximum(M - Y, X_prev + dt * drift - Y)

        crossed = X > Xc
        bridge_crossed = np.zeros(X.size, dtype=bool)

        bridge_candidates = (~crossed) & (X_prev < Xc)
        if np.any(bridge_candidates):
            candidate_idx = np.flatnonzero(bridge_candidates)
            gap_prev = Xc[candidate_idx] - X_prev[candidate_idx]
            gap_cur = Xc[candidate_idx] - X[candidate_idx]
            eps_dt = epsilon[candidate_idx] * dt

            valid_bridge = eps_dt > 0.0
            if np.any(valid_bridge):
                valid_idx = candidate_idx[valid_bridge]
                p_cross = np.exp(-2.0 * gap_prev[valid_bridge] * gap_cur[valid_bridge] / eps_dt[valid_bridge])
                bridge_hits = rng.random(valid_idx.size) < p_cross
                bridge_crossed[valid_idx[bridge_hits]] = True

        died = crossed | bridge_crossed

        if h_ext_scalar is not None and h_ext_scalar > 0.0:
            p_death_ext = 1.0 - np.exp(-h_ext_scalar * dt)
            died |= rng.random(X.size) < p_death_ext
        elif h_ext_array is not None:
            if np.any(h_ext_array > 0.0):
                p_death_ext = 1.0 - np.exp(-h_ext_array * dt)
                died |= rng.random(X.size) < p_death_ext
        elif h_ext_callable is not None:
            rate = float(h_ext_callable(tcur))
            p_death_ext = 1.0 - np.exp(-rate * dt)
            if p_death_ext > 0.0:
                died |= rng.random(X.size) < p_death_ext

        n_died = int(died.sum())
        deaths_per_step[step_idx] = n_died

        if n_died == 0:
            continue

        survivors = ~died
        X = X[survivors]
        eta = eta[survivors]
        beta = beta[survivors]
        kappa = kappa[survivors]
        epsilon = epsilon[survivors]
        sqrt_2epsilon = sqrt_2epsilon[survivors]
        Xc = Xc[survivors]

        if h_ext_array is not None:
            h_ext_array = h_ext_array[survivors]

    return deaths_per_step


class SRHazardSim:
    """
    Run the SR simulation while keeping only hazard and survival outputs.

    This class is designed for very large `n`. It does not store paths and it
    does not keep a full `death_times` vector. Instead, it simulates the cohort
    in chunks, counts deaths at each time step, and builds:

    - `survival`
    - `tspan_survival`
    - `hazard`
    - `tspan_hazard`
    """

    def __init__(
        self,
        eta=None,
        beta=None,
        kappa=None,
        epsilon=None,
        Xc=None,
        n=10000,
        tmin=0,
        tmax=1000,
        x0=1e-10,
        dt=1,
        h_ext=None,
        parallel=False,
        n_workers=None,
        chunk_size=1_000_000,
        break_early=True,
        random_seed=None,
        params=None,
        joint_sampler=None,
        xc_drop_age=None,
        xc_drop_fraction=0.0,
        x_bump_age=None,
        x_bump_fraction=0.0,
        x_add_age=None,
        x_add_amount=0.0,
    ):
        resolved = self._resolve_inputs(
            eta=eta,
            beta=beta,
            kappa=kappa,
            epsilon=epsilon,
            Xc=Xc,
            n=n,
            tmin=tmin,
            tmax=tmax,
            x0=x0,
            dt=dt,
            h_ext=h_ext,
            parallel=parallel,
            n_workers=n_workers,
            chunk_size=chunk_size,
            break_early=break_early,
            random_seed=random_seed,
            params=params,
            joint_sampler=joint_sampler,
            xc_drop_age=xc_drop_age,
            xc_drop_fraction=xc_drop_fraction,
            x_bump_age=x_bump_age,
            x_bump_fraction=x_bump_fraction,
            x_add_age=x_add_age,
            x_add_amount=x_add_amount,
        )

        self.raw_eta = resolved["eta"]
        self.raw_beta = resolved["beta"]
        self.raw_kappa = resolved["kappa"]
        self.raw_epsilon = resolved["epsilon"]
        self.raw_Xc = resolved["Xc"]
        self.raw_x0 = resolved["x0"]
        self.raw_h_ext = resolved["h_ext"]
        self.joint_sampler = resolved["joint_sampler"]

        self.n = int(resolved["n"])
        self.tmin = float(resolved["tmin"])
        self.tmax = float(resolved["tmax"])
        self.dt = float(resolved["dt"])
        self.parallel = bool(resolved["parallel"])
        self.break_early = bool(resolved["break_early"])
        self.random_seed = resolved["random_seed"]
        self.xc_drop_age = resolved["xc_drop_age"]
        self.xc_drop_fraction = float(resolved["xc_drop_fraction"])
        self.x_bump_age = resolved["x_bump_age"]
        self.x_bump_fraction = float(resolved["x_bump_fraction"])
        self.x_add_age = resolved["x_add_age"]
        self.x_add_amount = float(resolved["x_add_amount"])

        self._validate_inputs()

        self.chunk_size = min(int(resolved["chunk_size"]), self.n)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        self.n_workers = int(resolved["n_workers"] or cpu_count())
        if self.n_workers <= 0:
            raise ValueError("n_workers must be positive.")
        self.tspan_survival = np.arange(self.tmin, self.tmax + 1e-9, self.dt, dtype=np.float64)

        self.death_counts = None
        self.at_risk = None
        self.survival = None
        self.tspan_hazard = None
        self.hazard = None

        self.run_simulation()

    @classmethod
    def from_dict(cls, params):
        return cls(params=params)

    @staticmethod
    def _resolve_inputs(
        eta,
        beta,
        kappa,
        epsilon,
        Xc,
        n,
        tmin,
        tmax,
        x0,
        dt,
        h_ext,
        parallel,
        n_workers,
        chunk_size,
        break_early,
        random_seed,
        params,
        joint_sampler,
        xc_drop_age,
        xc_drop_fraction,
        x_bump_age,
        x_bump_fraction,
        x_add_age,
        x_add_amount,
    ):
        if params is None and isinstance(eta, dict):
            params = eta
            eta = None

        if params is None:
            resolved = {}
        else:
            resolved = dict(params)

        required_values = {
            "eta": eta,
            "beta": beta,
            "kappa": kappa,
            "epsilon": epsilon,
            "Xc": Xc,
        }
        for key, value in required_values.items():
            if value is not None:
                resolved[key] = value

        optional_values = {
            "n": n,
            "tmin": tmin,
            "tmax": tmax,
            "x0": x0,
            "dt": dt,
            "h_ext": h_ext,
            "parallel": parallel,
            "n_workers": n_workers,
            "chunk_size": chunk_size,
            "break_early": break_early,
            "random_seed": random_seed,
            "joint_sampler": joint_sampler,
            "xc_drop_age": xc_drop_age,
            "xc_drop_fraction": xc_drop_fraction,
            "x_bump_age": x_bump_age,
            "x_bump_fraction": x_bump_fraction,
            "x_add_age": x_add_age,
            "x_add_amount": x_add_amount,
        }

        optional_defaults = {
            "n": 10000,
            "tmin": 0,
            "tmax": 1000,
            "x0": 1e-10,
            "dt": 1,
            "h_ext": None,
            "parallel": False,
            "n_workers": None,
            "chunk_size": 1_000_000,
            "break_early": True,
            "random_seed": None,
            "joint_sampler": None,
            "xc_drop_age": None,
            "xc_drop_fraction": 0.0,
            "x_bump_age": None,
            "x_bump_fraction": 0.0,
            "x_add_age": None,
            "x_add_amount": 0.0,
        }

        for key, value in optional_values.items():
            if params is None:
                resolved[key] = value
                continue

            default_value = optional_defaults[key]
            if _should_override_value(value, default_value):
                resolved[key] = value

        for key, default_value in optional_defaults.items():
            resolved.setdefault(key, default_value)

        missing = []
        for key in ("eta", "beta", "kappa", "epsilon", "Xc"):
            if key not in resolved or resolved[key] is None:
                missing.append(key)

        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(f"Missing required parameters: {missing_str}")

        return resolved

    def _validate_inputs(self):
        if self.n <= 0:
            raise ValueError("n must be positive.")
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        if self.tmax <= self.tmin:
            raise ValueError("tmax must be larger than tmin.")
        if self.joint_sampler is not None and not callable(self.joint_sampler):
            raise TypeError("joint_sampler must be callable.")
        if callable(self.raw_h_ext) and self.parallel:
            raise ValueError("parallel=True is not supported when h_ext is callable.")
        if self.xc_drop_fraction < 0.0 or self.xc_drop_fraction >= 1.0:
            raise ValueError("xc_drop_fraction must be in [0, 1).")
        if self.x_bump_fraction < 0.0:
            raise ValueError("x_bump_fraction must be non-negative.")
        if self.x_add_amount < 0.0:
            raise ValueError("x_add_amount must be non-negative.")
        n_active = sum([
            self.xc_drop_fraction > 0.0,
            self.x_bump_fraction > 0.0,
            self.x_add_amount > 0.0,
        ])
        if n_active > 1:
            raise ValueError("Use only one acute intervention per SRHazardSim run.")

    def run_simulation(self):
        if self.parallel and self.n_workers > 1:
            death_counts = self._run_parallel()
        else:
            death_counts = self._run_serial()

        self._finalize_outputs(death_counts)

    def _run_serial(self):
        total_deaths = np.zeros(self.tspan_survival.size, dtype=np.int64)

        for task in self._iter_chunk_tasks():
            total_deaths += _simulate_chunk(task)

        return total_deaths

    def _run_parallel(self):
        total_deaths = np.zeros(self.tspan_survival.size, dtype=np.int64)

        pool_factory = Pool
        try:
            pool_factory = get_context("fork").Pool
        except ValueError:
            pass

        with pool_factory(self.n_workers) as pool:
            for chunk_deaths in pool.imap_unordered(_simulate_chunk, self._iter_chunk_tasks()):
                total_deaths += chunk_deaths

        return total_deaths

    def _iter_chunk_tasks(self):
        n_steps = self.tspan_survival.size
        start_idx = 0
        chunk_idx = 0

        while start_idx < self.n:
            current_chunk_size = min(self.chunk_size, self.n - start_idx)
            param_seed = None
            sim_seed = None
            if self.random_seed is not None:
                param_seed = int(self.random_seed) + 2 * chunk_idx
                sim_seed = int(self.random_seed) + 2 * chunk_idx + 1

            yield {
                "start_idx": start_idx,
                "total_n": self.n,
                "chunk_size": current_chunk_size,
                "n_steps": n_steps,
                "tmin": self.tmin,
                "dt": self.dt,
                "break_early": self.break_early,
                "xc_drop_age": np.inf if self.xc_drop_age is None else float(self.xc_drop_age),
                "xc_drop_fraction": self.xc_drop_fraction,
                "x_bump_age": np.inf if self.x_bump_age is None else float(self.x_bump_age),
                "x_bump_fraction": self.x_bump_fraction,
                "x_add_age": np.inf if self.x_add_age is None else float(self.x_add_age),
                "x_add_amount": self.x_add_amount,
                "param_seed": param_seed,
                "sim_seed": sim_seed,
                "joint_sampler": self.joint_sampler,
                "x0": _slice_array_if_needed(self.raw_x0, start_idx, current_chunk_size, self.n, "x0"),
                "eta": _slice_array_if_needed(self.raw_eta, start_idx, current_chunk_size, self.n, "eta"),
                "beta": _slice_array_if_needed(self.raw_beta, start_idx, current_chunk_size, self.n, "beta"),
                "kappa": _slice_array_if_needed(self.raw_kappa, start_idx, current_chunk_size, self.n, "kappa"),
                "epsilon": _slice_array_if_needed(self.raw_epsilon, start_idx, current_chunk_size, self.n, "epsilon"),
                "Xc": _slice_array_if_needed(self.raw_Xc, start_idx, current_chunk_size, self.n, "Xc"),
                "h_ext": _slice_array_if_needed(self.raw_h_ext, start_idx, current_chunk_size, self.n, "h_ext"),
            }

            start_idx += current_chunk_size
            chunk_idx += 1

    def _finalize_outputs(self, death_counts):
        cumulative_deaths = np.cumsum(death_counts, dtype=np.int64)
        cumulative_deaths = np.minimum(cumulative_deaths, self.n)

        if self.break_early and np.any(cumulative_deaths >= self.n):
            last_idx = int(np.flatnonzero(cumulative_deaths >= self.n)[0])
            death_counts = death_counts[: last_idx + 1]
            cumulative_deaths = cumulative_deaths[: last_idx + 1]
            self.tspan_survival = self.tspan_survival[: last_idx + 1]

        self.death_counts = death_counts
        self.survival = 1.0 - cumulative_deaths / float(self.n)

        if death_counts.size <= 1:
            self.at_risk = np.array([], dtype=np.int64)
            self.hazard = np.array([], dtype=np.float64)
            self.tspan_hazard = np.array([], dtype=np.float64)
            return

        deaths_before_step = np.cumsum(death_counts[:-1], dtype=np.int64)
        at_risk = self.n - deaths_before_step
        valid = at_risk > 0

        self.at_risk = at_risk[valid]
        self.tspan_hazard = self.tspan_survival[1:][valid]
        self.hazard = death_counts[1:][valid] / (self.at_risk * self.dt)


SR_hazard_sim = SRHazardSim
