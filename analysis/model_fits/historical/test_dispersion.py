"""Fast regression checks using saved artifacts and a synthetic analytic model.

No fitting or expensive SR derivatives are evaluated.
"""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

import denmark_aggregate
from export_tables import export_tables, Z
from inference import encode, joint_covariance, pearson_dispersion

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT/'results/historical'


class DispersionChecks(unittest.TestCase):
    def test_year_blocks_before_inversion(self):
        class Model:
            design = np.column_stack([np.ones(9), np.linspace(-1, 1, 9),
                                      np.linspace(-1, 1, 9)**2,
                                      np.linspace(-1, 1, 9)**3])

            def rates(self, params):
                return np.exp(self.design@encode(params, ['Xc', 'epsilon', 'CV', 'mex']))

        model = Model()
        base = dict(Xc=1., epsilon=1., CV=.2, mex=.001)
        histories = {1980: dict(base, Xc=.9), 2000: dict(base, Xc=.95)}
        exposures = {y: np.full(9, 100.) for y in [2019, *histories]}
        params = {2019: base, **histories}
        means = {y: exposures[y]*model.rates(p) for y, p in params.items()}
        original, _, _ = joint_covariance(model, base, histories, exposures)
        unit, _, _ = joint_covariance(model, base, histories, exposures, deaths=means)
        np.testing.assert_allclose(unit, original)
        counts = {y: mu+np.sqrt(mu*factor*(9-(4 if y == 2019 else 2))/9)
                  for (y, mu), factor in zip(means.items(), [2., 3., 4.])}
        adjusted, _, diag = joint_covariance(model, base, histories, exposures, deaths=counts)
        np.testing.assert_allclose([s['phi'] for s in diag['dispersion']], [2., 3., 4.])
        # Independently assemble analytic bread and meat, including inheritance.
        a = np.zeros((8, 8)); b = np.zeros((8, 8)); j = model.design
        a[:4, :4] = j.T@j; b[:4, :4] = 2.*(j.T/means[2019])@j
        own = j[:, [0, 3]]; inherit = np.zeros_like(j); inherit[:, [1, 2]] = j[:, [1, 2]]
        for i, (year, factor) in enumerate([(1980, 3.), (2000, 4.)]):
            sl = slice(4+2*i, 6+2*i)
            a[sl, sl] = own.T@own; a[sl, :4] = own.T@inherit
            b[sl, sl] = factor*(own.T/means[year])@own
        inv = np.linalg.inv(a)
        np.testing.assert_allclose(adjusted, inv@b@inv.T, atol=1e-10, rtol=1e-8)
        self.assertGreater(np.linalg.eigvalsh(adjusted-original).min(), -1e-10)
        self.assertEqual(pearson_dispersion(np.ones(5), np.ones(5), 4)['variance_multiplier'], 1.)

    def test_saved_tables_and_intervals(self):
        with tempfile.TemporaryDirectory() as temp:
            export_tables(DATA, Path(temp))
            for name, keys in [('sweden_threshold_fits.csv', ['series', 'year']),
                               ('lifespan_bands.csv', ['scenario', 'year', 'label'])]:
                old = pd.read_csv(DATA/name).sort_values(keys).reset_index(drop=True)
                new = pd.read_csv(Path(temp)/name).sort_values(keys).reset_index(drop=True)
                pd.testing.assert_frame_equal(old, new, check_exact=False, rtol=1e-12, atol=1e-12)
        bands = pd.read_csv(DATA/'lifespan_bands.csv')
        np.testing.assert_allclose(bands.ci_low, bands.estimate-Z*bands.se)
        np.testing.assert_allclose(bands.ci_high, bands.estimate+Z*bands.se)
        sw = json.loads((DATA/'joint_covariance.json').read_text())
        dk = json.loads((DATA/'denmark_joint_covariance.json').read_text())
        np.testing.assert_array_equal(np.asarray(sw['covariance'])[:4, :4],
                                      np.asarray(dk['covariance'])[:4, :4])
        for row in json.loads((DATA/'dispersion.json').read_text()):
            self.assertEqual(row['k'], 4 if row['year'] == 2019 else 2)
            self.assertAlmostEqual(row['phi'], row['pearson']/row['df'])
            self.assertEqual(row['variance_multiplier'], max(1., row['phi']))
        for data in [sw, dk]:
            cov = np.asarray(data['covariance'])
            np.testing.assert_allclose(cov, cov.T, atol=1e-12)
            self.assertGreater(np.linalg.eigvalsh(cov).min(), -1e-10)

    def test_denmark_saved_jacobians(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'point').mkdir(); (root/'inputs').mkdir()
            shutil.copy2(DATA/'fit_records/point/baseline.json', root/'point/baseline.json')
            shutil.copy2(DATA/'joint_covariance.json', root/'joint_covariance.json')
            shutil.copy2(DATA/'denmark_external_mortality.csv', root/'inputs/denmark_panel_c.csv')
            shutil.copytree(DATA/'fit_records/denmark/point', root/'denmark/point')
            old_root, old_here = denmark_aggregate.ROOT, denmark_aggregate.HERE
            try:
                denmark_aggregate.ROOT = root; denmark_aggregate.HERE = root/'denmark'
                denmark_aggregate.aggregate()
            finally:
                denmark_aggregate.ROOT, denmark_aggregate.HERE = old_root, old_here
            new = json.loads((root/'denmark/joint_covariance.json').read_text())
            old = json.loads((DATA/'denmark_joint_covariance.json').read_text())
            np.testing.assert_allclose(new['covariance'], old['covariance'], rtol=1e-12, atol=1e-12)
            self.assertEqual(new['q'], old['q'])
            pd.testing.assert_frame_equal(pd.read_csv(root/'denmark/figures/Fig4d_values.csv'),
                                          pd.read_csv(DATA/'denmark_threshold_fits.csv'),
                                          check_exact=False, rtol=1e-12, atol=1e-12)

    def test_preserved_artifacts(self):
        report = json.loads((DATA/'dispersion_update_verification.json').read_text())
        for name, digest in report['preserved_sha256'].items():
            self.assertEqual(hashlib.sha256((DATA/name).read_bytes()).hexdigest(), digest, name)


if __name__ == '__main__':
    unittest.main()
