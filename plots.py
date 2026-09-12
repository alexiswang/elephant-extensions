"""Statistics helpers shared by the analysis notebooks' plots (Wilson CIs, two-proportion
z-tests, and p-value formatting for chart annotations)."""

import numpy as np
from scipy import stats


def wilson_ci(count, n, alpha=0.05):
    """Wilson score confidence interval for a proportion."""
    z = stats.norm.ppf(1 - alpha / 2)
    p = count / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0, center - margin), min(1, center + margin)


def two_prop_ztest(count1, n1, count2, n2, alternative='two-sided'):
    """Two-proportion z-test."""
    p1 = count1 / n1
    p2 = count2 / n2
    p_pool = (count1 + count2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z = (p1 - p2) / se
    if alternative == 'two-sided':
        p_val = 2 * stats.norm.sf(abs(z))
    elif alternative == 'larger':
        p_val = stats.norm.sf(z)
    else:
        p_val = stats.norm.cdf(z)
    return z, p_val


def format_p(p):
    """Format a p-value for display in plot annotations."""
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.05:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.3f} (n.s.)"
