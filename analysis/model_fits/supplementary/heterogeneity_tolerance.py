"""Local, survivor-conditioned heterogeneity estimates from SI section S4.

These deterministic tolerance calculations are not fitted population bounds.
All reference values are the illustrative values stated in S4, not a new fit.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = dict(age=90.0, gompertz_slope=0.1, mortality=0.15, timescale=100.0)


def tolerance_table(tolerances, *, age=90.0, gompertz_slope=0.1,
                    mortality=0.15, timescale=100.0):
    """Convert a selection-slope budget to one-parameter CVs among survivors.

    The hazard-CV budget is sqrt(r*b/m). Local log-hazard sensitivities to
    fractional parameter changes are b*T, 2-b*tau and b*(T-tau) for eta,
    beta and Xc, respectively. Epsilon uses the opposite-sign Xc exponent
    sensitivity, neglecting the epsilon-dependent prefactor. The beta sensitivity retains its beta**2
    prefactor. A zero sensitivity does not yield a finite first-order bound.
    """
    r = np.asarray(tolerances, dtype=float)
    if np.any(~np.isfinite(r)) or np.any(r < 0):
        raise ValueError('Tolerances must be finite and nonnegative')
    if min(age, gompertz_slope, mortality, timescale) <= 0:
        raise ValueError('Reference values must be positive')
    sensitivities = dict(eta=gompertz_slope*age,
                         beta=abs(2-gompertz_slope*timescale),
                         Xc=gompertz_slope*abs(age-timescale),
                         epsilon=gompertz_slope*abs(age-timescale))
    if min(sensitivities.values()) == 0:
        raise ValueError('A zero local sensitivity gives no finite CV estimate')
    budget = np.sqrt(r*gompertz_slope/mortality)
    return pd.DataFrame(dict(tolerance_fraction=r, hazard_cv=budget,
                             **{f'{key}_cv': budget/value
                                for key, value in sensitivities.items()}))


def main():
    out = ROOT/'results/supplementary1_fp'
    tolerance_table(np.linspace(0, .5, 101), **REFERENCE).to_csv(
        out/'heterogeneity_tolerance.csv', index=False)
    settings = dict(reference=REFERENCE,
        conditioning='Parameter distributions among survivors at age 90, not at birth',
        criterion='Var_t(m)/m_pop <= r*b',
        sensitivities=dict(eta='b*T', beta='abs(2-b*tau)', Xc='b*abs(T-tau)', epsilon='b*abs(T-tau), exponent only'),
        epsilon_prefactor_retained=False,
        beta_prefactor_retained=True,
        interpretation='One-parameter, first-order local approximation; not a statistical confidence bound',
        limitation='Large allowed CVs, particularly threshold CV, are rough local estimates')
    (out/'heterogeneity_tolerance.json').write_text(json.dumps(settings, indent=2)+'\n')


if __name__ == '__main__':
    main()
