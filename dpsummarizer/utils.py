import numpy as np
from math import isfinite
from scipy.optimize import brentq
from scipy.stats import norm
from pathlib import Path
import torch
import logging

ADAPTER_DIR = Path("dpsummarizer/adapter_checkpoints")

def create_prompt(review: str, title: str | None = None, categories: list[str] | None = None) -> str:
    """
    Creates a prompt to extract a semantic representation of a single review.

    :param review: The product review text.
    :type review: str
    
    :param title: Optional product title.
    :type title: str | None
    
    :param categories: Optional list of product categories.
    :type categories: list[str] | None

    :return: The constructed prompt string.
    :rtype: str
    """
    meta_bits = []
    if title:
        meta_bits.append(f"Product title: {title}.")
    if categories:
        cats_str = ", ".join(categories)
        meta_bits.append(f"Product categories: {cats_str}.")
    meta_prefix = " ".join(meta_bits)
    if meta_prefix:
        return f"{meta_prefix} Review: {review}"
    else:
        return f"Review: {review}"

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
        while f_min < 0 and mu_min > 1e-12:
            mu_min *= 0.5
            f_min = f(mu_min)
    if f_max > 0:
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

def get_next_adapter_version(model_name: str) -> str:
    """Get the next available version number for this model."""
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    # Extract model name (e.g., "meta-llama/Llama-3.2-1B-Instruct" -> "Llama-3.2-1B-Instruct")
    base_name = model_name.split("/")[-1].replace(".", "_").replace("-", "_")
    
    # Find highest version number
    existing = list(ADAPTER_DIR.glob(f"{base_name}_v*.pt"))
    if not existing:
        return f"{base_name}_v1"
    
    versions = []
    for f in existing:
        try:
            v_str = f.stem.split("_v")[-1]
            versions.append(int(v_str))
        except (ValueError, IndexError):
            pass
    
    next_v = max(versions) + 1 if versions else 1
    return f"{base_name}_v{next_v}"

def save_adapter(adapter, model_name: str):
    """Save adapter to checkpoint file with auto-versioning."""
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_name = get_next_adapter_version(model_name)
    checkpoint_path = ADAPTER_DIR / f"{checkpoint_name}.pt"
    torch.save({
        'adapter': adapter.state_dict(),
    }, checkpoint_path)
    logging.info(f"Adapter saved to {checkpoint_path}")
    return checkpoint_path

def load_adapter(adapter, name: str):
    """Load adapter from checkpoint file."""
    checkpoint_path = ADAPTER_DIR / f"{name}.pt"
    if not checkpoint_path.exists():
        logging.error(f"Checkpoint not found: {checkpoint_path}")
        return False
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    adapter.load_state_dict(checkpoint['adapter'])
    logging.info(f"Adapter loaded from {checkpoint_path}")
    return True