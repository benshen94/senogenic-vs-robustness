"""Focused checks for retained vendored consumers and schematic isolation."""

import importlib
import io
import os
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('MPLBACKEND', 'Agg')

from PIL import Image, ImageStat

from analysis.quality_checks.artificial_survival_time import make_threshold_schematic as schematic


class CleanupTests(unittest.TestCase):
    def test_nhanes_topics_match_manuscript_preparation(self):
        from scripts.prepare_nhanes_cohort import TOPICS
        from ageing_packages.hetero_analysis.nhanes_analysis import TOPIC_CONFIGS

        self.assertEqual(set(TOPIC_CONFIGS), set(TOPICS))

    def test_retained_preparation_and_figure_imports(self):
        for name in (
            'scripts.prepare_nhanes_cohort',
            'scripts.prepare_nhanes_survival',
            'scripts.prepare_historical_hmd',
            'analysis.figures.figure2.run_fig2_fp',
            'analysis.figures.supplementary.make_supp_figure4_healthspan_morbidity',
        ):
            with self.subTest(module=name):
                importlib.import_module(name)

    def test_standalone_schematic_does_not_change_output_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / 'outputs.csv'
            original = b'artifact_id,path\nexisting,Figures/existing.png\n'
            index.write_bytes(original)
            output = root / 'schematic.png'
            with patch.multiple(schematic, PROJECT_ROOT=root, OUTPUT_DIR=root,
                                PNG_PATH=output), patch.object(
                                    schematic, 'INDEX_PATH', index, create=True), redirect_stdout(io.StringIO()):
                schematic.main()
            self.assertEqual(index.read_bytes(), original)
            self.assertEqual(set(root.iterdir()), {index, output})
            with Image.open(output) as image:
                self.assertEqual(image.format, 'PNG')
                self.assertGreater(image.width, 1000)
                self.assertGreater(min(ImageStat.Stat(image.convert('RGB')).stddev), 10)


if __name__ == '__main__':
    unittest.main()
