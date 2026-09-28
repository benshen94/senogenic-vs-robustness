"""Keep the FG publication curves on a consistent population size."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysis.model_fits.fedichev_gruber import recompute
from analysis.figures.supplementary import make_supp_figure3_fedichev_minimal_model as supp


class PopulationTests(unittest.TestCase):
    def test_shape_baseline_and_every_curve_use_publication_population(self):
        with patch.object(recompute, "run_simulation_beta_prime", return_value=np.arange(1., 101.)) as simulate:
            with contextlib.redirect_stdout(io.StringIO()):
                frame, metadata = recompute.shape_response(20260604)
        self.assertEqual(len(frame), 54)
        self.assertEqual(simulate.call_count, 55)
        self.assertTrue(all(call.args[0] == 300_000 for call in simulate.call_args_list))
        self.assertEqual(metadata["n"], 300_000)
        self.assertEqual(supp.N_SIM, recompute.FEDICHEV_N)

    def test_stale_population_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            pd.DataFrame({"n_sim": [160_000]}).to_csv(path, index=False)
            with patch.object(supp, "SOURCE_PATH", path):
                with self.assertRaisesRegex(ValueError, "300000"):
                    supp.load_or_build_source(force_sim=False, n=300_000)

    def test_publication_population_cache_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            pd.DataFrame({"n_sim": [300_000]}).to_csv(path, index=False)
            with patch.object(supp, "SOURCE_PATH", path):
                source = supp.load_or_build_source(force_sim=False, n=300_000)
            self.assertEqual(source.n_sim.iloc[0], 300_000)
