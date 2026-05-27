"""
Named SR fit presets.

Each fit stores scalar SR parameters plus the heterogeneity metadata needed to
build simulation-ready parameter dictionaries.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any, Dict


ETA_KARIN = 0.00135 * 365
BETA_KARIN = 0.15 * 365
KAPPA_KARIN = 0.5
EPSILON_KARIN = 0.142 * 365
XC_KARIN = 17.0


def _params(
    eta: float,
    beta: float,
    epsilon: float,
    xc: float,
    kappa: float = KAPPA_KARIN,
) -> Dict[str, float]:
    return {
        "eta": float(eta),
        "beta": float(beta),
        "kappa": float(kappa),
        "epsilon": float(epsilon),
        "Xc": float(xc),
    }


def _fit(
    *,
    label: str,
    params: Dict[str, float],
    h_ext: float | None,
    hetero_param: str | None,
    hetero_std: float,
    source: str,
    default_n: int = 100_000,
    data_type: str | None = None,
    gender: str | None = None,
    country: str | None = None,
    year: int | None = None,
    year_start: int | None = None,
    year_end: int | None = None,
    fit_target: str | None = None,
    age_start: int | None = None,
    age_end: int | None = None,
    hazard_mode: str | None = None,
    notes: str | None = None,
) -> Dict[str, Any]:
    return {
        "label": label,
        "params": params,
        "h_ext": None if h_ext is None else float(h_ext),
        "default_n": int(default_n),
        "default_family": "None",
        "heterogeneity": {
            "param": hetero_param,
            "std": float(hetero_std),
            "dist_type": "gaussian",
        },
        "fit_context": {
            "source": source,
            "country": country,
            "gender": gender,
            "data_type": data_type,
            "year": year,
            "year_start": year_start,
            "year_end": year_end,
            "fit_target": fit_target,
            "age_start": age_start,
            "age_end": age_end,
            "hazard_mode": hazard_mode,
            "fixed_params": {
                "kappa": float(params["kappa"]),
            },
            "notes": notes,
        },
    }


SR_FITS: Dict[str, Dict[str, Any]] = {
    "denmark_historical_cohort": _fit(
        label="Denmark Historical Cohort",
        params=_params(
            eta=ETA_KARIN * 1.33,
            beta=BETA_KARIN * 1.16,
            epsilon=EPSILON_KARIN,
            xc=17.0,
        ),
        h_ext=0.003,
        hetero_param="Xc",
        hetero_std=0.21,
        source="notebooks/SR_general/model_param_calibrations.py",
        country="DNK",
        gender="both",
        data_type="cohort",
        year_start=1870,
        year_end=1900,
        notes="Legacy calibration used across the Denmark historical cohort notebooks.",
    ),
    "sweden_period_2009_legacy": _fit(
        label="Sweden 2009 Period",
        params=_params(
            eta=ETA_KARIN * 1.33,
            beta=BETA_KARIN * 1.17,
            epsilon=EPSILON_KARIN,
            xc=18.4,
        ),
        h_ext=10 ** ((-2.4676 - 2.5463) / 2),
        hetero_param="Xc",
        hetero_std=0.21,
        source="notebooks/SR_general/model_param_calibrations.py",
        country="SWE",
        gender="both",
        data_type="period",
        year=2009,
        notes="Legacy Sweden modern-period calibration used by the SR notebooks.",
    ),
    "satsa_legacy": _fit(
        label="SATSA Legacy",
        params=_params(
            eta=ETA_KARIN * 1.33,
            beta=BETA_KARIN * 1.17,
            epsilon=EPSILON_KARIN,
            xc=18.4,
        ),
        h_ext=0.002,
        hetero_param="Xc",
        hetero_std=0.20,
        source="notebooks/SR_general/model_param_calibrations.py",
        country="SWE",
        gender="both",
        data_type="study",
        notes="Legacy SATSA calibration kept here for completeness.",
    ),
    "usa_period_2019_legacy": _fit(
        label="USA 2019 Legacy",
        params=_params(
            eta=ETA_KARIN * (0.96 * 1.33),
            beta=BETA_KARIN * (0.98 * 1.16),
            epsilon=EPSILON_KARIN,
            xc=0.96 * XC_KARIN,
        ),
        h_ext=0.0027,
        hetero_param="Xc",
        hetero_std=0.23,
        source="notebooks/SR_general/model_param_calibrations.py",
        country="USA",
        gender="both",
        data_type="period",
        year=2019,
        notes="Existing factor-based USA fit used before direct hazard fitting.",
    ),
    "usa_period_2019_both_hazard": _fit(
        label="USA 2019 Hazard Fit",
        params=_params(
            eta=0.43858182758384656,
            beta=41.86849757284421,
            epsilon=37.51256801824486,
            xc=23.096526094453885,
        ),
        h_ext=0.0,
        hetero_param="Xc",
        hetero_std=0.20,
        source="Direct SRFitter hazard fit on 2026-03-31",
        country="USA",
        gender="both",
        data_type="period",
        year=2019,
        fit_target="hazard",
        age_start=40,
        age_end=95,
        hazard_mode="interval",
        notes=(
            "Fitted eta, beta, epsilon, and mean Xc with fixed kappa=0.5 and "
            "fixed h_ext=0.0. Xc heterogeneity is Gaussian with std=0.2*Xc_mean. "
            "Used lx weighting with a late-robust hazard objective."
        ),
    ),
    "usa_period_2019_both_hazard_age60_100_xc18": _fit(
        label="USA 2019 Hazard Fit (60-100, Xc 18%)",
        params=_params(
            eta=0.4481249470961866,
            beta=53.012821541576294,
            epsilon=46.89772870973996,
            xc=18.991515527822067,
        ),
        h_ext=0.0,
        hetero_param="Xc",
        hetero_std=0.18,
        source="Direct SRFitter hazard fit on 2026-03-31",
        country="USA",
        gender="both",
        data_type="period",
        year=2019,
        fit_target="hazard",
        age_start=60,
        age_end=100,
        hazard_mode="interval",
        notes=(
            "Fitted eta, beta, epsilon, and mean Xc with fixed kappa=0.5 and "
            "fixed h_ext=0.0. Xc heterogeneity is Gaussian with std=0.18*Xc_mean. "
            "Saved as a separate exploratory narrow-age hazard fit."
        ),
    ),
    "usa_period_2019_both_survival_age60_100_xc18": _fit(
        label="USA 2019 Survival Fit (60-100, Xc 18%)",
        params=_params(
            eta=0.4448309291204187,
            beta=55.318697085061814,
            epsilon=46.48220096937755,
            xc=18.96882723734663,
        ),
        h_ext=0.0,
        hetero_param="Xc",
        hetero_std=0.18,
        source="Direct SRFitter survival fit on 2026-03-31",
        country="USA",
        gender="both",
        data_type="period",
        year=2019,
        fit_target="survival",
        age_start=60,
        age_end=100,
        hazard_mode="interval",
        notes=(
            "Fitted eta, beta, epsilon, and mean Xc with fixed kappa=0.5 and "
            "fixed h_ext=0.0. Xc heterogeneity is Gaussian with std=0.18*Xc_mean. "
            "Saved as a separate exploratory narrow-age survival fit."
        ),
    ),
    "eng_period_2019_both_hazard_age50_100_xc18": _fit(
        label="England 2019 Hazard Fit (50-100, Xc 18%)",
        params=_params(
            eta=0.46449616387004455,
            beta=56.14004212535409,
            epsilon=44.24424027696517,
            xc=19.9362615040202,
        ),
        h_ext=0.0,
        hetero_param="Xc",
        hetero_std=0.18,
        source="Direct SRFitter hazard fit on 2026-03-31",
        country="ENG",
        gender="both",
        data_type="period",
        year=2019,
        fit_target="hazard",
        age_start=50,
        age_end=100,
        hazard_mode="interval",
        notes=(
            "Fitted eta, beta, epsilon, and mean Xc with fixed kappa=0.5 and "
            "fixed h_ext=0.0. Xc heterogeneity is Gaussian with std=0.18*Xc_mean. "
            "Uses the repo's ENG HMD data for the UK request."
        ),
    ),
    "eng_period_2019_both_survival_age50_100_xc18": _fit(
        label="England 2019 Survival Fit (50-100, Xc 18%)",
        params=_params(
            eta=0.4342804334905908,
            beta=61.64632934962144,
            epsilon=45.55463451102979,
            xc=18.909829335697086,
        ),
        h_ext=0.0,
        hetero_param="Xc",
        hetero_std=0.18,
        source="Direct SRFitter survival fit on 2026-03-31",
        country="ENG",
        gender="both",
        data_type="period",
        year=2019,
        fit_target="survival",
        age_start=50,
        age_end=100,
        hazard_mode="interval",
        notes=(
            "Fitted eta, beta, epsilon, and mean Xc with fixed kappa=0.5 and "
            "fixed h_ext=0.0. Xc heterogeneity is Gaussian with std=0.18*Xc_mean. "
            "Uses the repo's ENG HMD data for the UK request."
        ),
    ),
    "usa_period_2019_both_hazard_age30_100_xc18_naf3": _fit(
        label='USA 2019 Hazard Fit (30-100, Xc 18%, NAF3)',
        params=_params(
            eta=0.5145659062911081,
            beta=54.75,
            epsilon=47.528319559298104,
            xc=18.538631,
            kappa=0.5,
        ),
        h_ext=0.0008105166160951386,
        hetero_param='Xc',
        hetero_std=0.18,
        source='Direct SR fitter warm-start hazard fit on 2026-04-01',
        country='USA',
        gender='both',
        data_type='period',
        year=2019,
        fit_target='hazard',
        age_start=30,
        age_end=100,
        hazard_mode='naf_bw3',
        notes='Warm-started from a smoke-run winner, then refined with fixed kappa, fixed h_ext from MGG, dt=0.025, and Gaussian Xc heterogeneity fixed at std_frac=0.18.',
    ),
    "dan_cohort_1920_both_hazard_age30_100_xc18_naf3": _fit(
        label='Denmark 1920 Cohort Hazard Fit (30-100, Xc 18%, NAF3)',
        params=_params(
            eta=0.561138428439806,
            beta=54.75,
            epsilon=43.583661,
            xc=17.752654301266034,
            kappa=0.5,
        ),
        h_ext=0.0011668200704625657,
        hetero_param='Xc',
        hetero_std=0.18,
        source='Direct SR fitter warm-start hazard fit on 2026-04-01',
        country='DAN',
        gender='both',
        data_type='cohort',
        year=1920,
        fit_target='hazard',
        age_start=30,
        age_end=100,
        hazard_mode='naf_bw3',
        notes='Warm-started from a smoke-run winner, then refined with fixed kappa, fixed h_ext from MGG, dt=0.025, and Gaussian Xc heterogeneity fixed at std_frac=0.18.',
    ),
    "eng_period_2010_both_hazard_age30_100_xc18_naf3": _fit(
        label='England 2010 Hazard Fit (30-100, Xc 18%, NAF3)',
        params=_params(
            eta=0.49275,
            beta=50.205971,
            epsilon=36.649344,
            xc=17.0,
            kappa=0.5,
        ),
        h_ext=0.00027591465963389936,
        hetero_param='Xc',
        hetero_std=0.18,
        source='Direct SR fitter warm-start hazard fit on 2026-04-01',
        country='ENG',
        gender='both',
        data_type='period',
        year=2010,
        fit_target='hazard',
        age_start=30,
        age_end=100,
        hazard_mode='naf_bw3',
        notes='Warm-started from a smoke-run winner, then refined with fixed kappa, fixed h_ext from MGG, dt=0.025, and Gaussian Xc heterogeneity fixed at std_frac=0.18.',
    ),
    "swe_period_2019_joint_shared_eta_beta_epsilon_60_100": _fit(
        label="Sweden 2019 Joint Shared Fit (60-100)",
        params=_params(
            eta=0.5707897419031454,
            beta=57.87173772073557,
            epsilon=53.472139183189434,
            xc=21.81604116925095,
            kappa=0.5,
        ),
        h_ext=0.0003767783778403702,
        hetero_param="Xc",
        hetero_std=0.1607701981486303,
        source="analysis/model_fits/hmd/joint_shared_eta_beta_epsilon_60_100.py",
        country="SWE",
        gender="both",
        data_type="period",
        year=2019,
        fit_target="hazard_survival_joint",
        age_start=60,
        age_end=100,
        hazard_mode="log_mx_weighted",
        notes="Joint USA/Sweden 2019 fit with shared eta, beta, epsilon and country-specific Xc and Xc heterogeneity.",
    ),
    "usa_period_2019_joint_shared_eta_beta_epsilon_60_100": _fit(
        label="USA 2019 Joint Shared Fit (60-100)",
        params=_params(
            eta=0.5707897419031454,
            beta=57.87173772073557,
            epsilon=53.472139183189434,
            xc=20.284662907421758,
            kappa=0.5,
        ),
        h_ext=0.0008105166160951386,
        hetero_param="Xc",
        hetero_std=0.18403753012497504,
        source="analysis/model_fits/hmd/joint_shared_eta_beta_epsilon_60_100.py",
        country="USA",
        gender="both",
        data_type="period",
        year=2019,
        fit_target="hazard_survival_joint",
        age_start=60,
        age_end=100,
        hazard_mode="log_mx_weighted",
        notes="Joint USA/Sweden 2019 fit with shared eta, beta, epsilon and country-specific Xc and Xc heterogeneity.",
    ),
    "swe_period_2019_joint_tail90_65_100": _fit(
        label="Sweden 2019 Joint Tail90 Fit (65-100)",
        params=_params(
            eta=0.5868368257640714,
            beta=57.87173772073557,
            epsilon=49.718659304628446,
            xc=21.74056340066893,
            kappa=0.5,
        ),
        h_ext=0.0003767783778403702,
        hetero_param="Xc",
        hetero_std=0.14142135623730953,
        source="analysis/model_fits/hmd/joint_shared_eta_beta_epsilon_65_100_n100k_tail90.py",
        country="SWE",
        gender="both",
        data_type="period",
        year=2019,
        fit_target="hazard_survival_joint",
        age_start=65,
        age_end=100,
        hazard_mode="log_mx_tail90_weighted",
        notes="Joint USA/Sweden 2019 fit with shared eta, beta, epsilon; emphasized Sweden ages 88-96 hazard.",
    ),
    "usa_period_2019_hybrid_swe_tail90_usa_refit_65_100": _fit(
        label="USA 2019 Hybrid Fit: Sweden Tail90 Anchor, USA Refit (65-100)",
        params=_params(
            eta=0.5868368257640714,
            beta=57.87173772073557,
            epsilon=49.718659304628446,
            xc=20.854942404177756,
            kappa=0.5,
        ),
        h_ext=0.0008105166160951386,
        hetero_param="Xc",
        hetero_std=0.1918528238650529,
        source="analysis/model_fits/hmd/joint_shared_eta_beta_epsilon_65_100_n100k_swe_tail90_usa_refit.py",
        country="USA",
        gender="both",
        data_type="period",
        year=2019,
        fit_target="hazard_survival_joint",
        age_start=65,
        age_end=100,
        hazard_mode="log_mx_weighted_usa_refit",
        notes="Hybrid fit using Sweden tail90 eta, beta, epsilon, and Sweden Xc/heterogeneity as anchor; only USA Xc and heterogeneity were refit.",
    ),
    "swe_cohort_1900_fixed_eta_beta_age65_100": _fit(
        label="Sweden 1900 Cohort Fixed Eta/Beta Fit (65-100)",
        params=_params(
            eta=0.5868368257640714,
            beta=57.87173772073557,
            epsilon=42.391863695049786,
            xc=16.822467429382066,
            kappa=0.5,
        ),
        h_ext=0.002,
        hetero_param="Xc",
        hetero_std=0.16245047927124714,
        source="analysis/model_fits/hmd/fit_sweden_historical_fixed_eta_beta.py",
        country="SWE",
        gender="both",
        data_type="cohort",
        year=1900,
        fit_target="hazard_survival_joint",
        age_start=65,
        age_end=100,
        hazard_mode="log_mx_weighted",
        notes="Historical Sweden fit with eta and beta fixed from the 2019 Sweden-tail/USA-refit model; fitted epsilon, Xc, Xc heterogeneity, and h_ext.",
    ),
    "swe_cohort_1920_fixed_eta_beta_age65_100": _fit(
        label="Sweden 1920 Cohort Fixed Eta/Beta Fit (65-100)",
        params=_params(
            eta=0.5868368257640714,
            beta=57.87173772073557,
            epsilon=44.808964698686644,
            xc=18.795526489487422,
            kappa=0.5,
        ),
        h_ext=0.002,
        hetero_param="Xc",
        hetero_std=0.16245047927124714,
        source="analysis/model_fits/hmd/fit_sweden_historical_fixed_eta_beta.py",
        country="SWE",
        gender="both",
        data_type="cohort",
        year=1920,
        fit_target="hazard_survival_joint",
        age_start=65,
        age_end=100,
        hazard_mode="log_mx_weighted",
        notes="Historical Sweden fit with eta and beta fixed from the 2019 Sweden-tail/USA-refit model; fitted epsilon, Xc, Xc heterogeneity, and h_ext.",
    ),
}


SR_FIT_ALIASES = {
    "denmark": "denmark_historical_cohort",
    "sweden": "sweden_period_2009_legacy",
    "satsa": "satsa_legacy",
    "usa": "usa_period_2019_both_hazard",
    "usa_2019": "usa_period_2019_both_hazard",
    "usa_period_2019": "usa_period_2019_both_hazard",
    "uk_period_2019_both_hazard_age50_100_xc18": "eng_period_2019_both_hazard_age50_100_xc18",
    "uk_period_2019_both_survival_age50_100_xc18": "eng_period_2019_both_survival_age50_100_xc18",
}


def list_sr_fit_names() -> list[str]:
    """Return the canonical SR fit names."""
    return sorted(SR_FITS.keys())


def resolve_sr_fit_name(fit_name: str) -> str:
    """Resolve aliases to their canonical SR fit name."""
    if fit_name in SR_FITS:
        return fit_name
    if fit_name in SR_FIT_ALIASES:
        return SR_FIT_ALIASES[fit_name]

    available = ", ".join(list_sr_fit_names())
    raise KeyError(f"Unknown SR fit '{fit_name}'. Available fits: {available}")


def get_sr_fit(fit_name: str) -> Dict[str, Any]:
    """Return a deep copy of a named SR fit."""
    canonical_name = resolve_sr_fit_name(fit_name)
    fit = deepcopy(SR_FITS[canonical_name])
    fit["name"] = canonical_name
    return fit


def build_sr_fit_record(
    *,
    label: str,
    params: Dict[str, float],
    h_ext: float | None,
    hetero_std: float,
    source: str,
    country: str | None = None,
    gender: str | None = None,
    data_type: str | None = None,
    year: int | None = None,
    fit_target: str | None = None,
    age_start: int | None = None,
    age_end: int | None = None,
    hazard_mode: str | None = None,
    notes: str | None = None,
) -> Dict[str, Any]:
    return _fit(
        label=label,
        params=_params(
            eta=float(params["eta"]),
            beta=float(params["beta"]),
            epsilon=float(params["epsilon"]),
            xc=float(params["Xc"]),
            kappa=float(params["kappa"]),
        ),
        h_ext=None if h_ext is None else float(h_ext),
        hetero_param="Xc",
        hetero_std=float(hetero_std),
        source=source,
        country=country,
        gender=gender,
        data_type=data_type,
        year=year,
        fit_target=fit_target,
        age_start=age_start,
        age_end=age_end,
        hazard_mode=hazard_mode,
        notes=notes,
    )


def _format_python_value(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)

    if value is None:
        return "None"

    if isinstance(value, bool):
        return "True" if value else "False"

    return repr(value)


def _format_fit_entry(fit_name: str, fit_record: Dict[str, Any]) -> str:
    params = fit_record["params"]
    heterogeneity = fit_record["heterogeneity"]
    context = fit_record["fit_context"]

    lines = [
        f'    "{fit_name}": _fit(',
        f'        label={_format_python_value(fit_record["label"])},',
        "        params=_params(",
        f'            eta={repr(float(params["eta"]))},',
        f'            beta={repr(float(params["beta"]))},',
        f'            epsilon={repr(float(params["epsilon"]))},',
        f'            xc={repr(float(params["Xc"]))},',
        f'            kappa={repr(float(params["kappa"]))},',
        "        ),",
        f'        h_ext={_format_python_value(fit_record["h_ext"])},',
        f'        hetero_param={_format_python_value(heterogeneity["param"])},',
        f'        hetero_std={repr(float(heterogeneity["std"]))},',
        f'        source={_format_python_value(context["source"])},',
        f'        country={_format_python_value(context["country"])},',
        f'        gender={_format_python_value(context["gender"])},',
        f'        data_type={_format_python_value(context["data_type"])},',
        f'        year={_format_python_value(context["year"])},',
        f'        fit_target={_format_python_value(context["fit_target"])},',
        f'        age_start={_format_python_value(context["age_start"])},',
        f'        age_end={_format_python_value(context["age_end"])},',
        f'        hazard_mode={_format_python_value(context["hazard_mode"])},',
        f'        notes={_format_python_value(context["notes"])},',
        "    ),",
    ]
    return "\n".join(lines)


def save_sr_fit_record(
    fit_name: str,
    fit_record: Dict[str, Any],
    file_path: str | Path | None = None,
) -> Path:
    path = Path(__file__).resolve() if file_path is None else Path(file_path).expanduser().resolve()
    text = path.read_text()
    entry_text = _format_fit_entry(fit_name, fit_record)

    pattern = re.compile(
        rf'(?ms)^    "{re.escape(fit_name)}": _fit\(\n.*?^    \),\n?'
    )
    if pattern.search(text):
        text = pattern.sub(entry_text + "\n", text)
    else:
        marker = "\n}\n\n\nSR_FIT_ALIASES = {"
        if marker not in text:
            marker = "\n}\n\nSR_FIT_ALIASES = {"
        if marker not in text:
            raise ValueError("Could not find the SR_FITS block terminator in sr_fits.py.")

        text = text.replace(marker, "\n" + entry_text + ",\n}" + marker[2:], 1)

    path.write_text(text)
    SR_FITS[fit_name] = deepcopy(fit_record)
    return path
