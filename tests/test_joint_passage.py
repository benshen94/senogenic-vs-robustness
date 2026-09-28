"""Small solver tests only; no manuscript simulations or fitting."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from senogenic_vs_robustness.sr_joint_passage import joint_passage, _rates, _solve


class JointPassageTests(unittest.TestCase):
    def test_tridiagonal_step_matches_dense_solve(self):
        edges = np.array([0., .1, .4, 1., 2.])
        up, down = _rates(edges, .59, 57.9, 30., .5, 20.)
        dt = .025
        matrix = np.diag(1 + dt * (up + np.r_[0., down[:-1]]))
        matrix += np.diag(-dt * down[:-1], 1) + np.diag(-dt * up[:-1], -1)
        initial = np.arange(12, dtype=float).reshape(4, 3) / 100
        expected = np.linalg.solve(matrix, initial)
        actual = initial.copy()
        _solve(up, down, dt, actual, 3)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)

    def test_tagged_mass_and_finite_horizon_fractions(self):
        result = joint_passage(1, 2, 1, 3, 2, n=16, dt=.25, horizon=4,
                               fraction_bins=101)
        state = result['states']
        np.testing.assert_allclose(state.sum(axis=1), 1, rtol=0, atol=1e-12)
        self.assertTrue((np.diff(state[:, 0]) <= 1e-12).all())
        self.assertTrue((np.diff(state[:, 2]) >= -1e-12).all())
        self.assertAlmostEqual(result['fraction_mass'].sum(), 1, places=12)
        self.assertGreaterEqual(result['fraction_mass'][0], state[-1, 0])
        self.assertLess(result['marginal_error'], 1e-12)
        self.assertIn(2., result['edges'])

    def test_diffusion_survival_converges_to_analytic_solution(self):
        # Reflecting origin, absorbing L, point mass initially at zero.
        odd = 2 * np.arange(200) + 1
        exact = float(np.sum(4 / np.pi * (-1.) ** np.arange(200) / odd
                             * np.exp(-(odd * np.pi / 6) ** 2)))
        errors = []
        for n, dt in [(20, .1), (80, .025)]:
            result = joint_passage(0, 0, 1, 3, 1.5, n=n, dt=dt,
                                   horizon=1, fraction_bins=101)
            errors.append(abs(result['states'][-1, :2].sum() - exact))
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[1], .005)

    def test_invalid_thresholds_and_horizon(self):
        with self.assertRaises(ValueError):
            joint_passage(1, 2, 1, 3, 3)
        with self.assertRaises(ValueError):
            joint_passage(1, 2, 1, 3, 2, dt=.3, horizon=1)


if __name__ == '__main__':
    unittest.main()
