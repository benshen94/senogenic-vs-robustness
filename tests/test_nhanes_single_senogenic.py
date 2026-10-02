"""Protect the AIC reference and sign convention in the new point comparison."""
import importlib.util
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "scripts/aggregate_nhanes_single_senogenic.py"
SPEC = importlib.util.spec_from_file_location("nhanes_point_summary", SOURCE)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def test_equal_complexity_single_models_compare_likelihood_directly():
    # Same k=2 for every single: an NLL gap of three means an AIC gap of six.
    nll = {"Xc": 100, "epsilon": 101, "eta": 103, "beta": 104}
    aic = {name: 2 * value + 4 for name, value in nll.items()}
    aic.update({pair: 2 * 99 + 6 for pair in SUMMARY.PAIRS})
    aic["mex_only"] = 2 * 110 + 2
    result = SUMMARY.comparisons(aic)
    assert result["delta_senogenic_vs_robustness"] == 6
    assert result["delta_Xc_vs_pair"] == 0
    assert result["senogenic_vs_pair"] == 6


def test_pair_reference_is_distinct_from_best_of_all_models():
    # A negative gap against pairs is valid. It need not make a model
    # competitive against the lower-complexity mex-only alternative.
    aic = {"mex_only": 100, "Xc": 105, "epsilon": 106, "eta": 101, "beta": 104}
    aic.update({pair: 108 for pair in SUMMARY.PAIRS})
    result = SUMMARY.comparisons(aic)
    assert result["delta_senogenic_vs_robustness"] == -4
    assert result["robustness_vs_pair"] == -3
    assert result["robustness_vs_all"] == 5
    assert result["senogenic_vs_all"] == 1
    summary = SUMMARY.summarize([result])
    assert summary["robustness_lower_AIC"] == 0
    assert summary["senogenic_lower_AIC"] == 1
    assert summary["either_robustness_competitive_vs_pairs"] == 1
    assert summary["either_robustness_competitive_vs_all"] == 0
    assert summary["senogenic_better_by_more_than_2"] == 1
