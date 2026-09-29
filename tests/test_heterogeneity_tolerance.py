"""Check the SI reference estimates and their interpretation as local budgets."""
import unittest
import numpy as np
from analysis.model_fits.supplementary.heterogeneity_tolerance import tolerance_table


class HeterogeneityToleranceTests(unittest.TestCase):
    def test_si_reference_and_prefactor_correction(self):
        row = tolerance_table([.2]).iloc[0]
        self.assertAlmostEqual(row.eta_cv*100, 4.0572041297, places=8)
        self.assertAlmostEqual(row.beta_cv*100, 4.5643546459, places=8)
        self.assertAlmostEqual(row.Xc_cv*100, 36.5148371670, places=8)
        # The corrected beta estimate exceeds the leading-barrier 3.65% value.
        self.assertGreater(row.beta_cv, row.hazard_cv/10)

    def test_budget_scaling_and_singular_reference(self):
        rows = tolerance_table([0, .1, .2, .4])
        cols = ['eta_cv','beta_cv','Xc_cv']
        np.testing.assert_array_equal(rows.loc[0,cols].to_numpy(),np.zeros(3))
        np.testing.assert_allclose(rows.loc[3,cols]/rows.loc[1,cols],2)
        with self.assertRaises(ValueError):
            tolerance_table([.2],timescale=90)


if __name__ == '__main__':
    unittest.main()
