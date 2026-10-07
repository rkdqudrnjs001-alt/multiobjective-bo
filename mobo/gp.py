"""Gaussian-process regression written from scratch with NumPy.

Kernel: squared exponential with ARD length scales, signal variance and noise.
Hyper-parameters are fitted by maximising the log marginal likelihood
(L-BFGS-B on log-parameters, a few random restarts).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


class GP:
    def __init__(self, noise_floor: float = 1e-6, restarts: int = 3, seed: int = 0):
        self.noise_floor, self.restarts = noise_floor, restarts
        self.rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- kernel
    @staticmethod
    def kernel(A, B, ls, sf2):
        d = (A[:, None, :] - B[None, :, :]) / ls
        return sf2 * np.exp(-0.5 * np.sum(d ** 2, axis=-1))

    def _unpack(self, theta):
        d = self.X.shape[1]
        ls = np.exp(theta[:d])
        sf2 = np.exp(theta[d])
        sn2 = np.exp(theta[d + 1]) + self.noise_floor
        return ls, sf2, sn2

    def _nll(self, theta):
        ls, sf2, sn2 = self._unpack(theta)
        K = self.kernel(self.X, self.X, ls, sf2) + sn2 * np.eye(len(self.X))
        try:
            c = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10
        alpha = cho_solve(c, self.y)
        return 0.5 * self.y @ alpha + np.sum(np.log(np.diag(c[0]))) + 0.5 * len(self.y) * np.log(2 * np.pi)

    # ---------------------------------------------------------------- fit / predict
    def fit(self, X, y):
        self.X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.ym, self.ys = y.mean(), y.std() + 1e-12
        self.y = (y - self.ym) / self.ys
        d = self.X.shape[1]
        best = None
        for r in range(self.restarts):
            th0 = np.concatenate([np.log(self.rng.uniform(0.1, 1.0, d)), [0.0], [np.log(1e-2)]])
            bounds = [(np.log(1e-2), np.log(10))] * d + [(np.log(1e-2), np.log(10)), (np.log(1e-8), np.log(1))]
            res = minimize(self._nll, th0, method="L-BFGS-B", bounds=bounds)
            if best is None or res.fun < best.fun:
                best = res
        self.theta = best.x
        ls, sf2, sn2 = self._unpack(self.theta)
        K = self.kernel(self.X, self.X, ls, sf2) + sn2 * np.eye(len(self.X))
        self._c = cho_factor(K, lower=True)
        self._alpha = cho_solve(self._c, self.y)
        return self

    def predict(self, Xs):
        ls, sf2, _ = self._unpack(self.theta)
        Ks = self.kernel(np.asarray(Xs, float), self.X, ls, sf2)
        mu = Ks @ self._alpha
        v = cho_solve(self._c, Ks.T)
        var = np.maximum(sf2 - np.sum(Ks * v.T, axis=1), 1e-12)
        return mu * self.ys + self.ym, np.sqrt(var) * self.ys
