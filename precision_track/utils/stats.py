import math
from statistics import NormalDist
from typing import Tuple


def wilson_bounds(successes: int, n: int, conf_level: float = 0.95) -> Tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    if successes < 0:
        return -1, -1
    z = NormalDist().inv_cdf(0.5 + conf_level / 2)
    phat = successes / n
    denom = 1 + z**2 / n
    centre = phat + z**2 / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)
    lower = max(0.0, (centre - margin) / denom)
    upper = min(1.0, (centre + margin) / denom)
    return lower, upper


def wilson_lower(successes: int, n: int, conf_level: float = 0.95) -> float:
    """
    Wilson score lower bound for a two-sided confidence interval.
    See Wikipedia 'Binomial proportion confidence interval'. :contentReference[oaicite:1]{index=1}
    """
    if n == 0:
        return 0.0
    if successes < 0:
        return -1, -1
    z = NormalDist().inv_cdf(0.5 + conf_level / 2)
    phat = successes / n
    denom = 1 + (z**2) / n
    centre = phat + (z**2) / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + (z**2) / (4 * n)) / n)
    return max(0.0, (centre - margin) / denom)
