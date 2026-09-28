"""Joint disease/death first passage for the finite-volume SR process.

The disease label is irreversible, but damage may subsequently fall below the
disease threshold. Healthy mass crossing that face enters a tagged population
on the full death-threshold domain. Tags retain the onset timestep, so the
sickspan/lifespan distribution is not inferred from marginal survival curves.

This optional solver is quadratic in the number of timesteps. It is not called
by the saved-result figure runner. See results/supplementary4_fp for grid checks.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _rates(edges, eta, beta, epsilon, kappa, age):
    centers = (edges[:-1] + edges[1:]) / 2
    widths = edges[1:] - edges[:-1]
    n = len(widths)
    up = np.empty(n)
    down = np.zeros(n)
    for j in range(n):
        right = centers[j + 1] if j < n - 1 else edges[-1]
        distance = right - centers[j]
        x = (right + centers[j]) / 2
        drift = eta * age - beta * x / (kappa + x)
        z = min(max(drift * distance / epsilon, -500.), 500.)
        if abs(z) < 1e-7:
            bp = 1 - z / 2 + z * z / 12
            bm = 1 + z / 2 + z * z / 12
        else:
            bp = z / np.expm1(z)
            bm = -z / np.expm1(-z)
        up[j] = epsilon / distance * bm / widths[j]
        if j < n - 1:
            down[j] = epsilon / distance * bp / widths[j + 1]
    return up, down


@njit(cache=True)
def _solve(up, down, dt, rhs, columns):
    """Implicit conservative step, sharing one factorization across tags."""
    n = rhs.shape[0]
    diagonal = np.empty(n)
    upper = np.zeros(n)
    for j in range(n):
        diagonal[j] = 1 + dt * (up[j] + (down[j - 1] if j else 0.))
        if j < n - 1:
            upper[j] = -dt * down[j]
    for j in range(1, n):
        weight = -dt * up[j - 1] / diagonal[j - 1]
        diagonal[j] -= weight * upper[j - 1]
        for c in range(columns):
            rhs[j, c] -= weight * rhs[j - 1, c]
    for c in range(columns):
        rhs[-1, c] /= diagonal[-1]
    for j in range(n - 2, -1, -1):
        for c in range(columns):
            rhs[j, c] = (rhs[j, c] - upper[j] * rhs[j + 1, c]) / diagonal[j]


@njit(cache=True)
def _joint(edges, disease_cell, eta, beta, epsilon, kappa, dt, steps, bins):
    n = len(edges) - 1
    healthy = np.zeros((disease_cell, 1))
    healthy[0, 0] = 1.
    sick = np.zeros((n, steps + 1))
    ordinary = np.zeros((n, 1))
    ordinary[0, 0] = 1.
    states = np.zeros((steps + 1, 3))
    states[0, 0] = 1.
    fraction_mass = np.zeros(bins)
    marginal_error = 0.
    for step in range(1, steps + 1):
        up, down = _rates(edges, eta, beta, epsilon, kappa, step * dt)
        _solve(up[:disease_cell], down[:disease_cell], dt, healthy, 1)
        crossing = dt * up[disease_cell - 1] * healthy[-1, 0]
        sick[disease_cell, step] += crossing
        _solve(up, down, dt, sick, step + 1)
        _solve(up, down, dt, ordinary, 1)
        deaths = 0.
        for onset in range(1, step + 1):
            mass = dt * up[-1] * sick[-1, onset]
            deaths += mass
            fraction = (step - onset) / step
            index = min(bins - 1, int(fraction * (bins - 1) + .5))
            fraction_mass[index] += mass
        h = healthy.sum()
        s = 0.
        for j in range(n):
            tagged = 0.
            for onset in range(1, step + 1):
                tagged += sick[j, onset]
            s += tagged
            if j < disease_cell:
                tagged += healthy[j, 0]
            marginal_error = max(marginal_error, abs(tagged - ordinary[j, 0]))
        states[step, 0] = h
        states[step, 1] = s
        states[step, 2] = states[step - 1, 2] + deaths
    # Match the original finite-horizon convention: alive participants use
    # horizon as end time; never-sick participants have sick fraction zero.
    fraction_mass[0] += healthy.sum()
    for onset in range(1, steps + 1):
        fraction = (steps - onset) / steps
        index = min(bins - 1, int(fraction * (bins - 1) + .5))
        fraction_mass[index] += sick[:, onset].sum()
    return states, fraction_mass, marginal_error


def joint_passage(eta, beta, epsilon, xc, xd, *, kappa=.5, n=320,
                  dt=.025, horizon=160., fraction_bins=2001):
    """Return healthy/sick/dead mass and the individual sick-life fraction PMF.

    Uses the additive-noise SR model without external mortality. A mesh edge is
    inserted exactly at Xd. Onset and death are assigned interval-end ages;
    decrease dt to check that discretization. PMF ratios are rounded to the
    nearest bin (maximum rounding error 1/[2*(fraction_bins-1)]).
    """
    values = np.array([eta, beta, epsilon, xc, xd, kappa, dt, horizon], float)
    if (not np.isfinite(values).all() or (values[2:] <= 0).any()
            or eta < 0 or beta < 0 or not xd < xc):
        raise ValueError('Nonnegative drift parameters, positive scales and 0 < Xd < Xc required')
    if n < 4 or fraction_bins < 3:
        raise ValueError('At least four cells and three fraction bins required')
    steps = int(round(horizon / dt))
    if steps < 1 or not np.isclose(steps * dt, horizon, rtol=0, atol=1e-10):
        raise ValueError('Horizon must be an integer multiple of dt')
    edges = kappa * np.expm1(np.linspace(0, np.log1p(xc / kappa), n + 1))
    edges[-1] = xc
    nearby = np.flatnonzero(np.isclose(edges, xd, rtol=0, atol=1e-12 * xc))
    if len(nearby):
        edges[nearby[0]] = xd
    else:
        edges = np.sort(np.append(edges, xd))
    disease_cell = int(np.searchsorted(edges, xd))
    states, pmf, error = _joint(edges, disease_cell, eta, beta, epsilon,
                                kappa, dt, steps, fraction_bins)
    if (states < -1e-12).any() or not np.allclose(states.sum(axis=1), 1, atol=1e-9, rtol=0):
        raise RuntimeError('Joint passage mass conservation failed')
    if not np.isclose(pmf.sum(), 1, atol=1e-9, rtol=0) or error > 1e-9:
        raise RuntimeError('Joint passage fraction mass or marginal parity failed')
    return {'age': np.arange(steps + 1) * dt, 'states': states,
            'fraction': np.linspace(0, 1, fraction_bins), 'fraction_mass': pmf,
            'marginal_error': error, 'edges': edges}
