"""Saved-node validation without any solver calls."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'healthspan_worker', ROOT / 'analysis/model_fits/supplementary/healthspan.py')
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)


class AggregationTests(unittest.TestCase):
    def fixture(self, folder, change_baseline=False):
        args = argparse.Namespace(nodes=2, cells=10, dt=1., fraction_bins=3,
                                  output=Path(folder))
        _, _, weights, manifest, _ = WORKER.setup(args)
        for scenario in WORKER.SCENARIOS:
            for node, weight in enumerate(weights):
                saved = json.loads(json.dumps(manifest))
                saved['baseline_sha256'] = str(node) * 64
                if change_baseline and node == 1:
                    saved['baseline']['Xc'] *= 1.01
                fingerprint = hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()
                np.savez_compressed(
                    args.output / f'{scenario}_{node:03d}.npz',
                    age=[0., 160.], states=[[1., 0., 0.], [0., 0., 1.]],
                    fraction=[0., .5, 1.], fraction_mass=[0., 1., 0.],
                    marginal_error=0., weight=weight, fingerprint=fingerprint,
                    manifest=json.dumps(saved, sort_keys=True))
        return args

    def test_covariance_only_revision_keeps_node_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.fixture(folder)
            WORKER.aggregate(args)
            result = json.loads((args.output / 'manifest.json').read_text())
            self.assertEqual(result['node_baseline_source_sha256s'], ['0' * 64, '1' * 64])

    def test_changed_scientific_baseline_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.fixture(folder, change_baseline=True)
            with self.assertRaisesRegex(ValueError, 'Scientific configuration mismatch'):
                WORKER.aggregate(args)


if __name__ == '__main__':
    unittest.main()
