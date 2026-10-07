"""Step 3 - multi-objective BO with ParEGO (from scratch) on Branin-Currin.

Each iteration draws random weights, scalarises the normalised objectives with
the augmented Chebyshev function, fits a GP to the scalarised values and picks
the next point by EI. Progress is measured by the 2-D hypervolume.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import qmc

from mobo.acquisition import chebyshev, expected_improvement
from mobo.common import branin_currin, hypervolume_2d_min, pareto_mask_min
from mobo.gp import GP

REF = np.array([18.0, 6.0])          # reference point (minimisation), as in BoTorch tutorial
N_INIT, N_ITER, N_CAND = 6, 30, 2000


def parego(seed):
    rng = np.random.default_rng(seed)
    X = qmc.LatinHypercube(d=2, seed=seed).random(N_INIT)
    F = branin_currin(X)
    hv = [hypervolume_2d_min(F, REF)]
    for _ in range(N_ITER):
        Fn = (F - F.min(0)) / (np.ptp(F, 0) + 1e-12)
        w = rng.dirichlet([1, 1])
        s = chebyshev(Fn, w)
        gp = GP(seed=seed).fit(X, s)
        C = rng.random((N_CAND, 2))
        mu, sd = gp.predict(C)
        x = C[np.argmax(expected_improvement(mu, sd, s.min()))]
        X = np.vstack([X, x])
        F = np.vstack([F, branin_currin(x)])
        hv.append(hypervolume_2d_min(F, REF))
    return F, np.array(hv)


def random_search(seed):
    rng = np.random.default_rng(100 + seed)
    F = branin_currin(rng.random((N_INIT + N_ITER, 2)))
    return np.array([hypervolume_2d_min(F[: N_INIT + i], REF) for i in range(N_ITER + 1)])


if __name__ == "__main__":
    runs = [parego(s) for s in range(3)]
    H = np.array([h for _, h in runs])
    R = np.array([random_search(s) for s in range(3)])
    print(f"final hypervolume  ParEGO: {H[:, -1].mean():.1f}   random: {R[:, -1].mean():.1f}")

    F = runs[0][0]
    m = pareto_mask_min(F)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].scatter(F[:, 0], F[:, 1], s=12, alpha=0.5, label="evaluated")
    P = F[m][np.argsort(F[m][:, 0])]
    ax[0].plot(P[:, 0], P[:, 1], "r.-", label="Pareto front")
    ax[0].set_xlabel("Branin (min)")
    ax[0].set_ylabel("Currin (min)")
    ax[0].legend()
    for M, lab in [(H, "ParEGO"), (R, "random")]:
        ax[1].plot(M.mean(0), label=lab)
        ax[1].fill_between(range(M.shape[1]), M.min(0), M.max(0), alpha=0.2)
    ax[1].set_xlabel("iteration")
    ax[1].set_ylabel("hypervolume")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig("figures/03_parego.png", dpi=130)
