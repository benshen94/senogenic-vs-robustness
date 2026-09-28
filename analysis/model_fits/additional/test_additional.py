"""Bounded contract checks; no production fits."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from lognormal import ROOT, calculate, forward, lognormal_nodes
from required_xc import invert_mean
from age_dependent_mex import cumulative_hazard, matched_constant, read_hazard
from compare_lognormal import metrics


class Contracts(unittest.TestCase):
    def test_distribution_comparison_metrics(self):
        age = np.linspace(0, 420, 1681)
        result = metrics(age, -age / 20)
        self.assertAlmostEqual(result["median"], 20 * np.log(2))
        self.assertAlmostEqual(result["top_0.01pct"], -20 * np.log(1e-4))
        self.assertAlmostEqual(result["survival_110_given_90"], np.exp(-1))
        with self.assertRaises(ValueError):
            metrics(age, -age / 100)

    def test_piecewise_linear_hazard(self):
        age = np.array([0., 1., 3.])
        hazard = 1 + 2 * age
        query = np.array([0., .3, 1., 2., 3.])
        np.testing.assert_allclose(cumulative_hazard(age, hazard, query), query + query**2)
        self.assertAlmostEqual(matched_constant(age, hazard, .5, 2.5), 4.)
        with self.assertRaises(ValueError):
            cumulative_hazard(age, hazard, [3.1])
        with self.assertRaises(ValueError):
            matched_constant(age, hazard, 1., 1.)

    def test_hazard_input_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            for content in ("age,hazard\n0,-1\n1,0\n", "age,hazard\n1,0\n2,0\n",
                            "age,hazard\n0,0\n0,1\n", "age,hazard\n0,0\n1,nan\n",
                            "age,rate\n0,0\n1,1\n"):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    read_hazard(path)

    def test_constant_hazard_cli_parity(self):
        here = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            # Synthetic test fixture only; this is not an analysis default.
            path.write_text("age,hazard\n0,0.001\n2,0.001\n")
            output = Path(directory) / "result"
            subprocess.run([sys.executable, str(here / "age_dependent_mex.py"), "--run",
                            "--hazard-csv", str(path), "--match", "cumulative-hazard",
                            "--match-start", "0.5", "--match-end", "2", "--entry-age", "0.5",
                            "--horizon", "2", "--dt", ".25", "--cells", "20", "--nodes", "3",
                            "--output", str(output)], check=True, capture_output=True, text=True)
            rows = np.genfromtxt(output / "comparison.csv", delimiter=",", names=True)
            np.testing.assert_allclose(rows["log_survival_age_dependent"], rows["log_survival_constant"])
            np.testing.assert_allclose(rows["log_survival_intrinsic"] - rows["log_survival_constant"],
                                       rows["age"] * .001, atol=1e-14)
            contract = json.loads((output / "contract.json").read_text())
            self.assertAlmostEqual(contract["matched_constant_hazard"], .001)
            self.assertEqual(contract["age_dependent"], contract["constant"])

    def test_mean_and_cv(self):
        factors, weights = lognormal_nodes(.3, "cv", 31)
        self.assertAlmostEqual(float(weights @ factors), 1., places=12)
        self.assertAlmostEqual(float(weights @ (factors - 1)**2), .3**2, places=12)
        other, _ = lognormal_nodes(np.sqrt(np.log1p(.3**2)), "log-sigma", 31)
        np.testing.assert_allclose(factors, other)

    def test_zero_width_solver_parity(self):
        p = json.loads((ROOT / "results/historical/fit_records/point/baseline.json").read_text())["params"]
        time, actual = calculate(p, "Xc", 0, "cv", 3, 20, .25, 2.)
        expected = forward(p["eta"], p["beta"], p["epsilon"], p["Xc"],
                           20, .25, 8, kappa=p["kappa"], log_output=True) - p["mex"] * time
        np.testing.assert_allclose(actual, expected, atol=1e-13)

    def test_invalid_width(self):
        for width in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                lognormal_nodes(width, "cv", 3)

    def test_inverse_target(self):
        xc, achieved = invert_mean(lambda x: 80 + x / 2, 100, 20, .5, 4)
        self.assertAlmostEqual(xc, 40, places=5)
        self.assertAlmostEqual(achieved, 100, places=5)
        with self.assertRaises(ValueError):
            invert_mean(lambda x: 80 + x / 2, 200, 20, .5, 4)

    def test_opt_in_guards(self):
        here = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "never_created"
            for name, args in (("lognormal.py", ["--parameter", "Xc", "--width", ".2", "--convention", "cv"]),
                               ("historical_epsilon.py", ["--year", "1900"]),
                               ("required_xc.py", []),
                               ("compare_lognormal.py", []),
                               ("age_dependent_mex.py", ["--hazard-csv", "unused.csv", "--match", "cumulative-hazard",
                                "--match-start", "0", "--match-end", "2", "--entry-age", "0", "--horizon", "2"])):
                result = subprocess.run([sys.executable, str(here / name), *args, "--output", str(dest)],
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("requires --run", result.stderr)
                self.assertFalse(dest.exists())


if __name__ == "__main__":
    unittest.main()
