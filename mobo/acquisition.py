"""Acquisition functions (minimisation convention)."""
import numpy as np
from scipy.stats import norm


def expected_improvement(mu, sd, best, xi=0.01):
    """EI for MINIMISATION: E[max(best - f - xi, 0)]."""
    sd = np.maximum(sd, 1e-12)
    z = (best - mu - xi) / sd
    return (best - mu - xi) * norm.cdf(z) + sd * norm.pdf(z)


def lower_confidence_bound(mu, sd, beta=2.0):
    """Negative LCB so that larger is better (to be maximised)."""
    return -(mu - beta * sd)


def chebyshev(F_norm, w, rho=0.05):
    """Augmented Chebyshev scalarisation used by ParEGO (minimise)."""
    return np.max(w * F_norm, axis=1) + rho * np.sum(w * F_norm, axis=1)
