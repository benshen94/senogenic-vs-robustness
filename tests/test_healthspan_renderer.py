"""Synthetic table fixtures only; these are not manuscript results."""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('healthspan_renderer', ROOT /
    'analysis/figures/supplementary/render_healthspan_morbidity.py')
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


def fixture(path):
    (path / 'manifest.json').write_text(json.dumps(
        {'backend': 'tagged finite-volume first passage', 'horizon': 160.}))
    rows, summaries, fractions = [], [], []
    for scenario in renderer.SCENARIOS:
        rows.extend(dict(scenario=scenario, age=t, healthy=h, sick=s, dead=d)
                    for t, h, s, d in [(0, 1, 0, 0), (80, .3, .5, .2), (160, 0, 0, 1)])
        summaries.append(dict(scenario=scenario, median_sick_life_fraction=.5))
        fractions.extend(dict(scenario=scenario, fraction=f, probability=p)
                         for f, p in [(0, .1), (.5, .8), (1, .1)])
    pd.DataFrame(rows).to_csv(path / 'states.csv', index=False)
    pd.DataFrame(summaries).to_csv(path / 'summary.csv', index=False)
    pd.DataFrame(fractions).to_csv(path / 'sick_fraction.csv', index=False)


class HealthspanRendererTests(unittest.TestCase):
    def test_missing_sources_fail_without_simulation(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FileNotFoundError, 'No simulation'):
                renderer.load_sources(Path(folder))

    def test_render_valid_tables_to_png(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            fixture(path)
            states, summary = renderer.load_sources(path)
            self.assertEqual(summary.loc["baseline","p05"],0)
            self.assertEqual(summary.loc["baseline","p25"],.5)
            self.assertEqual(summary.loc["baseline","p95"],1)
            renderer.configure_matplotlib()
            output = path / 'synthetic_test.png'
            renderer.make_composite(states, summary, output)
            with Image.open(output) as image:
                self.assertEqual(image.format, 'PNG')
                self.assertGreater(image.width, 1000)
                self.assertGreater(image.height, 1000)
            self.assertEqual(list(path.glob('*.pdf')), [])

    def test_inconsistent_median_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            fixture(path)
            summary = pd.read_csv(path / 'summary.csv')
            summary.loc[0, 'median_sick_life_fraction'] = .25
            summary.to_csv(path / 'summary.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'Median disagrees'):
                renderer.load_sources(path)


if __name__ == '__main__':
    unittest.main()
