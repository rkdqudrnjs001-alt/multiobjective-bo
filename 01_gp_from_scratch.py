"""Step 1 - Gaussian-process regression from scratch.

Fit a GP (ARD squared-exponential kernel, hyper-parameters by marginal
likelihood) to 8 noisy samples of a 1-D function and compare the posterior
with scikit-learn's implementation as a sanity check.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from mobo.gp import GP

rng = np.random.default_rng(1)
f = lambda x: np.sin(3 * x) + 0.3 * x ** 2
X = rng.uniform(-2, 2, (8, 1))
y = f(X[:, 0]) + rng.normal(0, 0.1, 8)
Xs = np.linspace(-2.5, 2.5, 300)[:, None]

gp = GP(seed=0).fit(X, y)
mu, sd = gp.predict(Xs)

sk = GaussianProcessRegressor(ConstantKernel() * RBF() + WhiteKernel(1e-2), normalize_y=True,
                              n_restarts_optimizer=3, random_state=0).fit(X, y)
mu_sk = sk.predict(Xs)
print(f"max |mean(mine) - mean(sklearn)| = {np.max(np.abs(mu - mu_sk)):.3f}")
print("fitted length scale = %.3f" % np.exp(gp.theta[0]))

plt.figure(figsize=(6, 3.5))
plt.plot(Xs, f(Xs), "k--", lw=1, label="true f")
plt.plot(Xs, mu, label="GP mean (from scratch)")
plt.fill_between(Xs[:, 0], mu - 2 * sd, mu + 2 * sd, alpha=0.2, label="±2σ")
plt.plot(Xs, mu_sk, ":", label="scikit-learn mean")
plt.scatter(X, y, c="k", zorder=3, label="data")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/01_gp.png", dpi=130)
