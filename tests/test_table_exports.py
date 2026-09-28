"""Table-export regressions without any SR solver or fitting calls."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_exporter(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'analysis/tables' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


KM = load_exporter('export_supplementary_table1')
M1 = load_exporter('export_table_m1')


class TableExportTests(unittest.TestCase):
    def test_bootstrap_matches_original_source_function(self):
        # Compile only the original sampling function, avoiding its SR imports.
        source = ROOT / 'analysis/model_fits/nhanes/bootstrap_baseline.py'
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'resampled_frame')
        module = ast.Module(body=[function], type_ignores=[])
        frame = pd.DataFrame(dict(wave=[2, 1, 1, 1, 1, 2, 2, 2],
                                  SDMVSTRA=[3, 1, 1, 1, 1, 3, 3, 3],
                                  SDMVPSU=[2, 3, 1, 2, 2, 1, 2, 1]))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'inputs').mkdir()
            frame.to_csv(root / 'inputs/nhanes.csv', index=False)
            namespace = dict(pd=pd, np=np, HERE=root, SEED=KM.SEED)
            exec(compile(module, str(source), 'exec'), namespace)
            blocks = KM.design_blocks(frame)
            for rep in (1, 2, 50, 100):
                original = namespace['resampled_frame'](rep).bootstrap_weight.to_numpy()
                np.testing.assert_array_equal(KM.bootstrap_weights(blocks, len(frame), rep), original)

    def test_invalid_design_and_replicate_fail(self):
        frame = pd.DataFrame(dict(wave=[1, 1], SDMVSTRA=[1, 1], SDMVPSU=[1, 1]))
        with self.assertRaisesRegex(ValueError, 'fewer than two'):
            KM.design_blocks(frame)
        with self.assertRaisesRegex(ValueError, 'positive'):
            KM.bootstrap_weights([], 2, 0)

    def test_inclusive_entry_ties_and_event_time_interpolation(self):
        result = KM.km_metrics([0, 0, 1, 0], [1, 2, 3, 4], [1, 1, 1, 0], [1]*4)
        self.assertEqual(result['q75_age'], 1)
        self.assertEqual(result['median_age'], 2)
        self.assertEqual(result['q25_age'], 3)
        self.assertEqual(result['steepness'], 1)

    def test_unresolved_quartile_is_not_clamped(self):
        result = KM.km_metrics([0]*4, [1, 2, 3, 4], [1, 1, 0, 0], [1]*4)
        self.assertEqual(result['median_age'], 2)
        self.assertTrue(np.isnan(result['q25_age']))
        self.assertTrue(np.isnan(result['steepness']))

    def test_zero_weight_event_is_excluded(self):
        expected = KM.km_metrics([0]*4, [1, 2, 3, 4], [1, 1, 1, 0], [1]*4)
        actual = KM.km_metrics([0]*5, [1, 1.5, 2, 3, 4], [1, 1, 1, 1, 0], [1, 0, 1, 1, 1])
        self.assertEqual(actual, expected)

    def test_unadjusted_sweden_intervals_match_manuscript_rounding(self):
        joint = json.loads((ROOT / 'results/historical/joint_covariance.json').read_text())
        dispersion = json.loads((ROOT / 'results/historical/dispersion.json').read_text())
        intervals, phi = M1.sweden_intervals(joint, dispersion)
        self.assertAlmostEqual(phi, 4.483524758861447)
        self.assertEqual(tuple(round(x, 1) for x in intervals['Xc']), (14.8, 19.0))
        self.assertEqual(tuple(round(x, 1) for x in intervals['epsilon']), (24.6, 37.1))
        self.assertEqual(tuple(round(x, 2) for x in intervals['CV']), (.22, .25))
        self.assertEqual(tuple(round(x * 1e4, 1) for x in intervals['mex']), (1.7, 2.6))

    def test_saved_km_exports_reproduce_archive_and_absolute_ses(self):
        folder = ROOT / 'results/tables'
        summary = pd.read_csv(folder / 'supplementaryTable1_source.csv').set_index('group_id')
        raw = pd.read_csv(folder / 'supplementaryTable1_replicates.csv')
        comparison = pd.read_csv(folder / 'supplementaryTable1_archive_comparison.csv')
        self.assertEqual(len(raw), 2400)
        self.assertFalse(raw.duplicated(['group_id', 'replicate']).any())
        self.assertTrue(comparison.matches.all())
        self.assertLess(comparison.difference.abs().max(), 1e-9)
        for name, group in raw.groupby('group_id'):
            for metric in ('median_age', 'steepness', 'x', 'y'):
                self.assertEqual(group[metric].count(), summary.loc[name, metric + '_valid_repeats'])
                self.assertAlmostEqual(group[metric].std(ddof=1), summary.loc[name, metric + '_se'], places=12)
        self.assertEqual(summary.loc['income__3', 'steepness_valid_repeats'], 48)
        self.assertEqual(summary.loc['income__3', 'median_age_valid_repeats'], 98)

    def test_m1_checked_mex_and_method_are_explicit(self):
        folder = ROOT / 'results/tables'
        table = pd.read_csv(folder / 'tableM1_source.csv')
        raw = pd.read_csv(folder / 'tableM1_replicates.csv')
        row = table[(table.population == 'NHANES') & (table.parameter == 'mex')].iloc[0]
        self.assertIn('check_mex', row.bootstrap_value_source)
        values = raw[raw.parameter == 'mex']
        self.assertTrue((values.value_source == 'check_mex').all())
        self.assertFalse(np.allclose(values.value, values.fit_grid_value))
        np.testing.assert_allclose([row.ci_low, row.ci_high], values.value.quantile([.025, .975]), atol=1e-15)
        self.assertFalse(table.pearson_dispersion_adjusted.any())


if __name__ == '__main__':
    unittest.main()
