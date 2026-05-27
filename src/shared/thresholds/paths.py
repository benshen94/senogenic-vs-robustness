"""Central project paths for the threshold/noise aging project."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGING_PYTHON_ROOT = PROJECT_ROOT
AGING_ROOT = PROJECT_ROOT

FIGURES_DIR = PROJECT_ROOT / "Figures"
FIGURES_NEW_DIR = PROJECT_ROOT / "Figures_new"
SAVED_DATA_DIR = PROJECT_ROOT / "saved_data"
SAVED_RESULTS_DIR = PROJECT_ROOT / "saved_results"
NHANES_DATA_DIR = SAVED_DATA_DIR / "nhanes"
HMD_DATA_DIR = SAVED_DATA_DIR / "hmd"
