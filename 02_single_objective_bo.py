"""Step 2 - single-objective BO (Expected Improvement) on Branin.

Compares BO with random search over 5 seeds; plots the best value found
vs. number of evaluations (log regret).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import qmc

from mobo.acquisition import expected_improvement
from mobo.common import branin
from mobo.gp import GP

F_STAR = 0.397887
N_INIT, N_ITER, N_CAND = 5, 25, 2000


def bo(seed):
    rng = np.random.default_rng(seed)
    X = qmc.LatinHypercube(d=2, seed=seed).random(N_INIT)
    y = branin(X)
    for _ in range(N_ITER):
        gp = GP(seed=seed).fit(X, y)
        C = rng.random((N_CAND, 2))
        mu, sd = gp.predict(C)
        x = C[np.argmax(expected_improvement(mu, sd, y.min()))]
        X, y = np.vstack([X, x]), np.append(y, branin(x))
    return np.minimum.accumulate(y)


def random_search(seed):
    rng = np.random.default_rng(100 + seed)
    return np.minimum.accumulate(branin(rng.random((N_INIT + N_ITER, 2))))


B = np.array([bo(s) for s in range(5)])
R = np.array([random_search(s) for s in range(5)])
print(f"best after {N_INIT + N_ITER} evals  BO: {B[:, -1].mean():.3f}   random: {R[:, -1].mean():.3f}"
      f"   (global min {F_STAR:.3f})")

plt.figure(figsize=(5.5, 3.5))
for M, lab in [(B, "BO (EI)"), (R, "random")]:
    r = np.log10(M - F_STAR + 1e-9)
    plt.plot(r.mean(0), label=lab)
    plt.fill_between(range(r.shape[1]), r.min(0), r.max(0), alpha=0.2)
plt.xlabel("evaluations")
plt.ylabel("log10 regret")
plt.legend()
plt.tight_layout()
plt.savefig("figures/02_bo_branin.png", dpi=130)
