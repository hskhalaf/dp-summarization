import numpy as np
from math import isfinite
from scipy.optimize import brentq
from scipy.stats import norm

def create_prompt(review: str) -> str:
    """
    Creates a prompt to extract a semantic representation of a single review.

    :param review: The product review text.
    :type review: str

    :return: The constructed prompt string.
    :rtype: str
    """
    return review

def _delta_gdp(epsilon: float, mu: float) -> float:
    if mu <= 0:
        return 1.0
    term1 = norm.cdf(-epsilon / mu + mu / 2.0)
    term2 = np.exp(epsilon) * norm.cdf(-epsilon / mu - mu / 2.0)
    return term1 - term2


def _solve_mu_from_eps_delta(
    eps_target: float,
    delta_target: float,
    mu_min: float = 1e-6,
    mu_max: float = 1000.0,
) -> float:
    if not (0 < delta_target < 1):
        raise ValueError("delta must be in (0, 1)")
    if eps_target <= 0:
        raise ValueError("epsilon must be positive")

    def f(mu: float) -> float:
        return _delta_gdp(eps_target, mu) - delta_target

    f_min = f(mu_min)
    f_max = f(mu_max)

    # Ensure the bracket actually straddles the root
    if f_min < 0:
        # Decrease mu_min if needed
        while f_min < 0 and mu_min > 1e-12:
            mu_min *= 0.5
            f_min = f(mu_min)
    if f_max > 0:
        # Increase mu_max if needed
        while f_max > 0 and mu_max < 1e6:
            mu_max *= 2.0
            f_max = f(mu_max)

    if f_min * f_max > 0:
        raise RuntimeError(
            f"Could not bracket μ for ε={eps_target}, δ={delta_target}: "
            f"f(mu_min)={f_min}, f(mu_max)={f_max}"
        )

    mu = brentq(f, mu_min, mu_max, xtol=1e-10, rtol=1e-10, maxiter=10_000)
    if not isfinite(mu) or mu <= 0:
        raise RuntimeError(f"Solved μ is invalid: {mu}")
    return mu


def calibrate_gaussian_sigma(epsilon: float, delta: float, sensitivity: float) -> float:
    """
    Calibration for the one-shot Gaussian mechanism using mu-GDP.

    We:
      1) Solve for mu such that the Gaussian mechanism is (epsilon, delta)-DP in the
         mu-GDP sense.
      2) For a single Gaussian query with L2-sensitivity Delta, the GDP parameter
         is mu = Delta / sigma  =>  sigma = Delta / mu.

    :param epsilon: Global epsilon (for this one-shot mechanism).
    :param delta:   Global delta (for this one-shot mechanism).
    :param sensitivity: L2-sensitivity Delta of the query (e.g. 2C/n under
                        add/remove adjacency with clipping norm C).
    :return: sigma, the standard deviation of the Gaussian noise.
    """
    mu = _solve_mu_from_eps_delta(epsilon, delta)
    return sensitivity / mu