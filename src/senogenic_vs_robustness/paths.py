"""Central project paths for the manuscript analysis repository."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGING_PYTHON_ROOT = PROJECT_ROOT
AGING_ROOT = PROJECT_ROOT

FIGURES_DIR = PROJECT_ROOT / "Figures"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FITS_DIR = RESULTS_DIR / "fits"
NHANES_DATA_DIR = DATA_DIR / "nhanes"
HMD_DATA_DIR = DATA_DIR / "hmd"
