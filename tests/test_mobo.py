import numpy as np

from mobo.common import branin, hypervolume_2d_min, pareto_mask_min, zdt1
from mobo.gp import GP


def test_branin_known_minimum():
    # (pi, 2.275) in original domain -> rescaled to [0,1]^2
    x = np.array([[(np.pi + 5) / 15, 2.275 / 15]])
    assert abs(branin(x)[0] - 0.397887) < 1e-4


def test_zdt1_true_front():
    x = np.zeros((5, 4)); x[:, 0] = np.linspace(0, 1, 5)
    F = zdt1(x)
    assert np.allclose(F[:, 1], 1 - np.sqrt(F[:, 0]))


def test_pareto_and_hypervolume():
    F = np.array([[0, 1], [1, 0], [0.5, 0.5], [0.9, 0.9]])
    assert pareto_mask_min(F).tolist() == [True, True, True, False]
    assert np.isclose(hypervolume_2d_min(F, np.array([1, 1])), 0.25)


def test_gp_interpolates_smooth_function():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, (15, 1))
    y = np.sin(2 * X[:, 0])
    gp = GP(seed=0).fit(X, y)
    Xs = np.linspace(-1.5, 1.5, 50)[:, None]
    mu, sd = gp.predict(Xs)
    assert np.max(np.abs(mu - np.sin(2 * Xs[:, 0]))) < 0.1
    assert np.all(sd >= 0)
