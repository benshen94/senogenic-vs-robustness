import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('validation',ROOT/'analysis/validation/validate.py')
validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validation)


class ValidationRecords(unittest.TestCase):
    def test_archived_records_recalculate(self):
        records = list((ROOT/'results/validation/records').glob('*.json'))
        self.assertEqual(len(records),9)
        for path in records:
            with self.subTest(record=path.name):
                record = validation.audit_record(path)
                self.assertEqual(record['n'],20000)
                self.assertIn('original_source_label',record)
                self.assertNotIn('source',record)
                with np.load(path.with_suffix('.npz'),allow_pickle=False) as data:
                    self.assertEqual(len(data['death_times']),record['n'])
                    for key in ('mc_survival','fp_survival','fp_coarse_survival'):
                        curve = data[key]
                        self.assertTrue(np.all(np.isfinite(curve)))
                        self.assertTrue(np.all((curve>=0)&(curve<=1+1e-12)))
                        self.assertTrue(np.all(np.diff(curve)<=1e-12))

    def test_corrupted_archive_is_rejected(self):
        source = next((ROOT/'results/validation/records').glob('*.json'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/source.name
            path.write_text(source.read_text())
            path.with_suffix('.npz').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                validation.audit_record(path)

    def test_saved_quadrature_is_explicitly_smoke_only(self):
        record = json.loads((ROOT/'results/validation/quadrature_smoke.json').read_text())
        self.assertTrue(record['result']['smoke'])
        self.assertEqual(record['result']['grid']['nodes'],[160,256])
        self.assertEqual({r['parameter'] for r in record['result']['checks']},
                         {'Xc','eta','beta'})
        for label,digest in record['original_source_code_sha256'].items():
            self.assertNotIn('/',label)
            self.assertRegex(digest,r'^[0-9a-f]{64}$')

    def test_full_quadrature_record(self):
        record = json.loads((ROOT/'results/validation/quadrature_full.json').read_text())
        result = record['result']
        self.assertFalse(result['smoke'])
        self.assertEqual(result['grid'],dict(cells=480,dt=1/60,horizon=260.,nodes=[160,256]))
        self.assertEqual({r['parameter'] for r in result['checks']},{'Xc','eta','beta'})
        self.assertEqual(validation.sha(ROOT/'results/supplementary1_fp/si_checks.json'),
                         result['original_source_baseline_sha256'])
        for label,digest in record['original_source_code_sha256'].items():
            self.assertNotIn('/',label)
            self.assertRegex(digest,r'^[0-9a-f]{64}$')
        for row in result['checks']:
            self.assertTrue(0 <= row['max_survival_difference'] < 1e-10)
            self.assertTrue(0 <= row['max_relative_annual_mortality_difference'] < 4.7e-6)

    def test_fine_steps_and_coarse_failures_preserved(self):
        records = [validation.audit_record(p) for p in
                   (ROOT/'results/validation/records').glob('*.json')]
        for case in {r['case'] for r in records}:
            fine = min((r for r in records if r['case']==case),key=lambda r:r['mc_dt'])
            self.assertLess(fine['maximum_survival_difference'],fine['mc_dkw95_half_width'])
        self.assertTrue(any(r['maximum_survival_difference']>r['mc_dkw95_half_width']
                            for r in records))


if __name__ == '__main__':
    unittest.main()
