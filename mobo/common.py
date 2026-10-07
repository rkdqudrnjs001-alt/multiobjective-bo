"""Shared helpers: test problems, Pareto utilities, hypervolume."""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------ test problems
def branin(x: np.ndarray) -> np.ndarray:
    """Branin on [0,1]^2 (rescaled from [-5,10]x[0,15]); global min ~0.398."""
    x = np.atleast_2d(x)
    x1, x2 = 15 * x[:, 0] - 5, 15 * x[:, 1]
    a, b, c, r, s, t = 1, 5.1 / (4 * np.pi ** 2), 5 / np.pi, 6, 10, 1 / (8 * np.pi)
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


def currin(x: np.ndarray) -> np.ndarray:
    """Currin exponential function on [0,1]^2."""
    x = np.atleast_2d(x)
    x1, x2 = x[:, 0], np.maximum(x[:, 1], 1e-10)
    f1 = 1 - np.exp(-1 / (2 * x2))
    f2 = 2300 * x1 ** 3 + 1900 * x1 ** 2 + 2092 * x1 + 60
    f3 = 100 * x1 ** 3 + 500 * x1 ** 2 + 4 * x1 + 20
    return f1 * f2 / f3


def branin_currin(x: np.ndarray) -> np.ndarray:
    """Two-objective benchmark (both minimised) used in the BoTorch tutorials."""
    return np.column_stack([branin(x), currin(x)])


def zdt1(x: np.ndarray) -> np.ndarray:
    """ZDT1, both objectives minimised, d >= 2, x in [0,1]^d."""
    x = np.atleast_2d(x)
    f1 = x[:, 0]
    g = 1 + 9 * x[:, 1:].mean(axis=1)
    f2 = g * (1 - np.sqrt(f1 / g))
    return np.column_stack([f1, f2])


# ------------------------------------------------------------ Pareto tools
def pareto_mask_min(F: np.ndarray) -> np.ndarray:
    """Non-dominated rows for MINIMISATION."""
    F = np.asarray(F)
    keep = np.ones(len(F), bool)
    for i in range(len(F)):
        if keep[i]:
            dom = np.all(F <= F[i], 1) & np.any(F < F[i], 1)
            if dom.any():
                keep[i] = False
    return keep


def hypervolume_2d_min(F: np.ndarray, ref: np.ndarray) -> float:
    """Exact 2-D hypervolume dominated by F (minimisation) up to ref."""
    P = F[pareto_mask_min(F)]
    P = P[np.all(P < ref, 1)]
    if len(P) == 0:
        return 0.0
    P = P[np.argsort(P[:, 0])]
    hv, prev_f2 = 0.0, ref[1]
    for f1, f2 in P:
        if f2 < prev_f2:
            hv += (ref[0] - f1) * (prev_f2 - f2)
            prev_f2 = f2
    return float(hv)
